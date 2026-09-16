package main

import (
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
	"unicode/utf8"
	"unsafe"

	"golang.org/x/sys/unix"
)

// openTestPTY crée une paire de pseudo-terminaux (master, slave) sous Linux en pur Go.
func openTestPTY(rows, cols int) (master, slave *os.File, err error) {
	master, err = os.OpenFile("/dev/ptmx", os.O_RDWR|syscall.O_NOCTTY, 0)
	if err != nil {
		return nil, nil, fmt.Errorf("open /dev/ptmx: %w", err)
	}

	// Déverrouiller le PTY (unlockpt)
	var unlock int = 0
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, master.Fd(), uintptr(unix.TIOCSPTLCK), uintptr(unsafe.Pointer(&unlock))); errno != 0 {
		_ = master.Close()
		return nil, nil, fmt.Errorf("unlockpt: %w", errno)
	}

	// Récupérer le numéro du PTY slave (ptsname)
	var ptyNum int
	if _, _, errno := syscall.Syscall(syscall.SYS_IOCTL, master.Fd(), uintptr(unix.TIOCGPTN), uintptr(unsafe.Pointer(&ptyNum))); errno != 0 {
		_ = master.Close()
		return nil, nil, fmt.Errorf("ptsname: %w", errno)
	}

	// Configurer la taille de fenêtre (rows, cols) sur le PTY
	ws := &unix.Winsize{
		Row: uint16(rows),
		Col: uint16(cols),
	}
	_ = unix.IoctlSetWinsize(int(master.Fd()), unix.TIOCSWINSZ, ws)

	slavePath := fmt.Sprintf("/dev/pts/%d", ptyNum)
	slave, err = os.OpenFile(slavePath, os.O_RDWR|syscall.O_NOCTTY, 0)
	if err != nil {
		_ = master.Close()
		return nil, nil, fmt.Errorf("open slave %s: %w", slavePath, err)
	}

	return master, slave, nil
}

// VTGridEmulator est un oracle de terminal virtuel en mémoire (émulateur VT / DECSTBM déterministe).
// Il ingère le flux ANSI brut émis par l'application et maintient la grille exacte des cellules à l'écran.
type VTGridEmulator struct {
	Rows         int
	Cols         int
	CursorRow    int // 1-indexé
	CursorCol    int // 1-indexé
	ScrollTop    int // 1-indexé
	ScrollBottom int // 1-indexé
	AltScreen    bool
	InvisCursor  bool

	primaryGrid   [][]rune
	alternateGrid [][]rune
	activeGrid    *[][]rune
}

// NewVTGridEmulator initialise une grille de Rows x Cols cellules vierges (espaces).
func NewVTGridEmulator(rows, cols int) *VTGridEmulator {
	emu := &VTGridEmulator{
		Rows:         rows,
		Cols:         cols,
		CursorRow:    1,
		CursorCol:    1,
		ScrollTop:    1,
		ScrollBottom: rows,
	}
	emu.primaryGrid = makeGrid(rows, cols)
	emu.alternateGrid = makeGrid(rows, cols)
	emu.activeGrid = &emu.primaryGrid
	return emu
}

func makeGrid(rows, cols int) [][]rune {
	g := make([][]rune, rows)
	for r := 0; r < rows; r++ {
		g[r] = make([]rune, cols)
		for c := 0; c < cols; c++ {
			g[r][c] = ' '
		}
	}
	return g
}

// Feed ingère un flux d'octets contenant des séquences ANSI et met à jour la matrice.
func (e *VTGridEmulator) Feed(data []byte) {
	grid := *e.activeGrid
	i := 0
	n := len(data)

	for i < n {
		b := data[i]

		// Caractère de contrôle Escape
		if b == 0x1b && i+1 < n {
			if data[i+1] == '[' {
				// Séquence CSI : \x1b[ ...
				end := i + 2
				for end < n && !((data[end] >= '@' && data[end] <= '~') || (data[end] >= 'A' && data[end] <= 'Z') || (data[end] >= 'a' && data[end] <= 'z')) {
					end++
				}
				if end < n {
					cmd := data[end]
					params := string(data[i+2 : end])
					e.handleCSI(params, cmd)
					i = end + 1
					grid = *e.activeGrid
					continue
				}
			} else if data[i+1] == ']' {
				// Séquence OSC : \x1b] ... \x07 ou \x1b\
				end := i + 2
				for end < n && data[end] != 0x07 && !(data[end] == 0x1b && end+1 < n && data[end+1] == '\\') {
					end++
				}
				if end < n {
					if data[end] == 0x07 {
						i = end + 1
					} else {
						i = end + 2
					}
					continue
				}
			}
		}

		// Caractères de contrôle standard
		switch b {
		case '\r':
			e.CursorCol = 1
			i++
			continue
		case '\n':
			if e.CursorRow == e.ScrollBottom {
				// Défilement de la zone de scroll
				e.scrollUpRegion()
			} else if e.CursorRow < e.Rows {
				e.CursorRow++
			}
			i++
			continue
		case '\t':
			nextTab := ((e.CursorCol-1)/8 + 1) * 8 + 1
			if nextTab > e.Cols {
				nextTab = e.Cols
			}
			e.CursorCol = nextTab
			i++
			continue
		}

		// Caractère UTF-8 imprimable
		r, size := utf8.DecodeRune(data[i:])
		if r == utf8.RuneError && size <= 1 {
			i++
			continue
		}
		i += size

		if r < 32 {
			continue
		}

		rw := runeDisplayWidth(r)
		if rw <= 0 {
			continue
		}

		// Inscription dans la grille
		if e.CursorRow >= 1 && e.CursorRow <= e.Rows && e.CursorCol >= 1 && e.CursorCol <= e.Cols {
			grid[e.CursorRow-1][e.CursorCol-1] = r
			if rw == 2 && e.CursorCol < e.Cols {
				// Caractère large : cellule suivante marquée par un espace vide réservé
				grid[e.CursorRow-1][e.CursorCol] = ' '
			}
		}

		e.CursorCol += rw
		if e.CursorCol > e.Cols {
			// Auto-wrap
			e.CursorCol = 1
			if e.CursorRow == e.ScrollBottom {
				e.scrollUpRegion()
			} else if e.CursorRow < e.Rows {
				e.CursorRow++
			}
		}
	}
}

func (e *VTGridEmulator) handleCSI(params string, cmd byte) {
	grid := *e.activeGrid

	switch cmd {
	case 'H', 'f': // Positionnement curseur : \x1b[row;colH
		row, col := 1, 1
		parts := strings.Split(params, ";")
		if len(parts) >= 1 && parts[0] != "" {
			if r, err := strconv.Atoi(parts[0]); err == nil && r >= 1 {
				row = r
			}
		}
		if len(parts) >= 2 && parts[1] != "" {
			if c, err := strconv.Atoi(parts[1]); err == nil && c >= 1 {
				col = c
			}
		}
		if row > e.Rows {
			row = e.Rows
		}
		if col > e.Cols {
			col = e.Cols
		}
		e.CursorRow = row
		e.CursorCol = col

	case 'J': // Effacement écran : \x1b[2J
		if params == "2" || params == "" {
			for r := 0; r < e.Rows; r++ {
				for c := 0; c < e.Cols; c++ {
					grid[r][c] = ' '
				}
			}
		}

	case 'K': // Effacement de ligne : \x1b[2K
		if e.CursorRow >= 1 && e.CursorRow <= e.Rows {
			r := e.CursorRow - 1
			if params == "2" {
				for c := 0; c < e.Cols; c++ {
					grid[r][c] = ' '
				}
			} else if params == "0" || params == "" { // effacer du curseur à la fin
				for c := e.CursorCol - 1; c < e.Cols; c++ {
					if c >= 0 {
						grid[r][c] = ' '
					}
				}
			}
		}

	case 'r': // DECSTBM : \x1b[top;bottomr
		if params == "" {
			e.ScrollTop = 1
			e.ScrollBottom = e.Rows
		} else {
			parts := strings.Split(params, ";")
			top, bot := 1, e.Rows
			if len(parts) >= 1 && parts[0] != "" {
				if t, err := strconv.Atoi(parts[0]); err == nil {
					top = t
				}
			}
			if len(parts) >= 2 && parts[1] != "" {
				if b, err := strconv.Atoi(parts[1]); err == nil {
					bot = b
				}
			}
			if top >= 1 && bot <= e.Rows && top < bot {
				e.ScrollTop = top
				e.ScrollBottom = bot
			}
		}

	case 'h', 'l': // Modes privés DEC
		isSet := (cmd == 'h')
		if strings.HasPrefix(params, "?") {
			modeStr := strings.TrimPrefix(params, "?")
			switch modeStr {
			case "1049": // Alternate Screen
				if isSet {
					e.AltScreen = true
					e.activeGrid = &e.alternateGrid
					// Clear alternate screen
					for r := 0; r < e.Rows; r++ {
						for c := 0; c < e.Cols; c++ {
							e.alternateGrid[r][c] = ' '
						}
					}
				} else {
					e.AltScreen = false
					e.activeGrid = &e.primaryGrid
				}
			case "25": // Curseur visible
				e.InvisCursor = !isSet
			}
		}
	}
}

func (e *VTGridEmulator) scrollUpRegion() {
	grid := *e.activeGrid
	top := e.ScrollTop - 1
	bot := e.ScrollBottom - 1
	if top < 0 || bot >= e.Rows || top >= bot {
		return
	}
	// Décaler d'une ligne vers le haut
	for r := top; r < bot; r++ {
		copy(grid[r], grid[r+1])
	}
	// Vider la dernière ligne de la zone
	for c := 0; c < e.Cols; c++ {
		grid[bot][c] = ' '
	}
}

// RowString retourne le texte complet d'une ligne (1-indexée).
func (e *VTGridEmulator) RowString(row int) string {
	if row < 1 || row > e.Rows {
		return ""
	}
	grid := *e.activeGrid
	return string(grid[row-1])
}

// Cell retourne la rune présente à (row, col) (1-indexées).
func (e *VTGridEmulator) Cell(row, col int) rune {
	if row < 1 || row > e.Rows || col < 1 || col > e.Cols {
		return 0
	}
	grid := *e.activeGrid
	return grid[row-1][col-1]
}

// TestHarness_GridOracle_Sizes vérifie la géométrie exacte de la ChatBox sur différentes résolutions.
func TestHarness_GridOracle_Sizes(t *testing.T) {
	sizes := []struct {
		cols int
		rows int
	}{
		{80, 24},
		{120, 40},
		{100, 30},
		{160, 50},
	}

	for _, sz := range sizes {
		t.Run(fmt.Sprintf("%dx%d", sz.cols, sz.rows), func(t *testing.T) {
			master, slave, err := openTestPTY(sz.rows, sz.cols)
			if err != nil {
				t.Skipf("PTY non disponible: %v", err)
				return
			}
			defer master.Close()
			defer slave.Close()

			cb := &ChatBox{
				inFd:         int(slave.Fd()),
				outFd:        int(slave.Fd()),
				isTTY:        true,
				width:        sz.cols,
				height:       sz.rows,
				chatTop:      sz.rows - 3,
				scrollBottom: sz.rows - 4,
				sigWinch:     make(chan os.Signal, 1),
			}

			// Capture de la sortie du PTY vers l'émulateur
			rPipe, wPipe, _ := os.Pipe()
			oldStdout := os.Stdout
			os.Stdout = wPipe

			cb.InitScreen(28, 16384)
			cb.DrawChatBox("bonjour c2slm", false)

			_ = wPipe.Close()
			os.Stdout = oldStdout

			output, _ := io.ReadAll(rPipe)
			_ = rPipe.Close()

			emu := NewVTGridEmulator(sz.rows, sz.cols)
			emu.Feed(output)

			chatTop := sz.rows - 3

			// 1. Vérifier la ligne supérieure de la boîte (chatTop)
			row1 := emu.RowString(chatTop)
			if !strings.HasPrefix(row1, "┌──") {
				t.Errorf("Ligne supérieure doit commencer par ┌──, obtenu: %q", row1[:min(10, len(row1))])
			}
			lastRuneRow1 := emu.Cell(chatTop, sz.cols)
			if lastRuneRow1 != '┐' {
				t.Errorf("Le coin supérieur droit ┐ doit être en colonne %d, trouvé %c (ligne: %q)", sz.cols, lastRuneRow1, row1)
			}
			// Vérifier qu'aucun autre coin ┐ n'est présent avant la dernière colonne
			for c := 1; c < sz.cols; c++ {
				if emu.Cell(chatTop, c) == '┐' {
					t.Errorf("Coin ┐ prématuré trouvé à la colonne %d au lieu de %d", c, sz.cols)
				}
			}

			// 2. Vérifier la ligne de saisie (chatTop + 1)
			row2 := emu.RowString(chatTop + 1)
			if !strings.HasPrefix(row2, "│ ❯ ") {
				t.Errorf("Ligne de saisie doit commencer par '│ ❯ ', obtenu: %q", row2[:min(10, len(row2))])
			}
			if !strings.Contains(row2, "bonjour c2slm") {
				t.Errorf("Ligne de saisie doit contenir 'bonjour c2slm', obtenu: %q", row2)
			}
			lastRuneRow2 := emu.Cell(chatTop+1, sz.cols)
			if lastRuneRow2 != '│' {
				t.Errorf("La bordure droite │ doit être en colonne %d, trouvé %c (ligne: %q)", sz.cols, lastRuneRow2, row2)
			}

			// 3. Vérifier la ligne inférieure de statut (chatTop + 2)
			row3 := emu.RowString(chatTop + 2)
			if !strings.HasPrefix(row3, "└─") {
				t.Errorf("Ligne inférieure doit commencer par └─, obtenu: %q", row3[:min(10, len(row3))])
			}
			lastRuneRow3 := emu.Cell(chatTop+2, sz.cols)
			if lastRuneRow3 != '┘' {
				t.Errorf("Le coin inférieur droit ┘ doit être en colonne %d, trouvé %c (ligne: %q)", sz.cols, lastRuneRow3, row3)
			}
			for c := 1; c < sz.cols; c++ {
				if emu.Cell(chatTop+2, c) == '┘' {
					t.Errorf("Coin ┘ prématuré trouvé à la colonne %d au lieu de %d", c, sz.cols)
				}
			}

			// 4. Vérifier l'absence totale de résidus de ChatBox dans la zone supérieure (lignes 8 à chatTop-1)
			// (Les lignes 1 à 7 contiennent la bannière d'accueil)
			for r := 8; r < chatTop; r++ {
				line := emu.RowString(r)
				if strings.Contains(line, "┌──") || strings.Contains(line, "└─ [Échap]") {
					t.Errorf("Pollution de ChatBox trouvée à la ligne %d dans la zone de défilement: %q", r, line)
				}
			}
		})
	}
}

// TestHarness_GridOracle_Resize teste l'effacement de l'ancienne boîte lors d'un redimensionnement.
func TestHarness_GridOracle_Resize(t *testing.T) {
	master, slave, err := openTestPTY(24, 80)
	if err != nil {
		t.Skipf("PTY non disponible: %v", err)
		return
	}
	defer master.Close()
	defer slave.Close()

	cb := &ChatBox{
		inFd:         int(slave.Fd()),
		outFd:        int(slave.Fd()),
		isTTY:        true,
		width:        80,
		height:       24,
		chatTop:      21,
		scrollBottom: 20,
		sigWinch:     make(chan os.Signal, 1),
	}

	rPipe, wPipe, _ := os.Pipe()
	oldStdout := os.Stdout
	os.Stdout = wPipe

	cb.InitScreen(28, 16384)
	cb.DrawChatBox("saisie initiale", false)

	// Simuler un resize vers 40x120
	cb.mu.Lock()
	oldChatTop := cb.chatTop
	oldHeight := cb.height

	cb.width = 120
	cb.height = 40
	cb.chatTop = 37
	cb.scrollBottom = 36

	// Effacement de l'ancienne boîte
	for line := oldChatTop; line <= oldHeight; line++ {
		fmt.Printf("\x1b[%d;1H\x1b[2K", line)
	}
	cb.applyScrollRegionLocked()
	cb.drawChatBoxLocked("saisie après resize", false)
	cb.mu.Unlock()

	_ = wPipe.Close()
	os.Stdout = oldStdout

	output, _ := io.ReadAll(rPipe)
	_ = rPipe.Close()

	emu := NewVTGridEmulator(40, 120)
	emu.Feed(output)

	// Vérifier que la boîte n'existe PLUS du tout sur les lignes 21..24
	for r := 21; r <= 24; r++ {
		line := strings.TrimSpace(emu.RowString(r))
		if strings.Contains(line, "┌──") || strings.Contains(line, "└─") || strings.Contains(line, "saisie initiale") {
			t.Errorf("Boîte fantôme non effacée à la ligne %d: %q", r, line)
		}
	}

	// Vérifier que la nouvelle boîte est bien présente à la ligne 37..39
	row1 := emu.RowString(37)
	if !strings.HasPrefix(row1, "┌──") || emu.Cell(37, 120) != '┐' {
		t.Errorf("Nouvelle boîte absente ou mal alignée à la ligne 37: %q", row1)
	}
	row2 := emu.RowString(38)
	if !strings.Contains(row2, "saisie après resize") || emu.Cell(38, 120) != '│' {
		t.Errorf("Nouvelle saisie absente à la ligne 38: %q", row2)
	}
}

// TestHarness_InterruptionESC vérifie l'arrêt immédiat sur réception d'Échap.
func TestHarness_InterruptionESC(t *testing.T) {
	master, slave, err := openTestPTY(30, 100)
	if err != nil {
		t.Skipf("PTY non disponible: %v", err)
		return
	}
	defer master.Close()
	defer slave.Close()

	cb := &ChatBox{
		inFd:         int(slave.Fd()),
		outFd:        int(slave.Fd()),
		isTTY:        true,
		width:        100,
		height:       30,
		chatTop:      27,
		scrollBottom: 26,
		sigWinch:     make(chan os.Signal, 1),
	}
	if termio, err := unix.IoctlGetTermios(int(slave.Fd()), unix.TCGETS); err == nil {
		cb.origTermio = termio
	}

	interrupted, endFn := cb.BeginInference()
	if interrupted.Load() {
		t.Fatal("interrupted doit être faux initialement")
	}

	// Injecter la touche Échap (0x1b) depuis le master PTY
	_, err = master.Write([]byte{0x1b})
	if err != nil {
		t.Fatalf("écriture sur master PTY: %v", err)
	}

	// Attendre que la goroutine de surveillance capte l'interruption
	time.Sleep(50 * time.Millisecond)

	if !interrupted.Load() {
		t.Error("L'octet 0x1b (ESC) n'a pas déclenché le drapeau interrupted")
	}

	endFn()
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
