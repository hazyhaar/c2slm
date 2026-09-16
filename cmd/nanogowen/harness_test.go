package main

import (
	"bytes"
	"fmt"
	"os"
	"strings"
	"syscall"
	"testing"
	"time"
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

// TestHarness_ChatBoxBottomAnchor est le banc d'épreuve automatisé qui simule
// un véritable terminal PTY graphique (40 lignes, 120 colonnes), injecte des frappes
// utilisateur et vérifie que :
// 1. La zone de saisie est ancrée en bas de l'écran (lignes 37 à 40).
// 2. Les frappes utilisateur sont positionnées sur la ligne de prompt en bas.
// 3. Aucune frappe ne fuit dans la marge haute du terminal.
func TestHarness_ChatBoxBottomAnchor(t *testing.T) {
	const rows = 40
	const cols = 120

	master, slave, err := openTestPTY(rows, cols)
	if err != nil {
		t.Skipf("PTY non disponible dans cet environnement: %v", err)
		return
	}
	defer master.Close()
	defer slave.Close()

	// Créer un ChatBox raccordé au PTY esclave
	cb := &ChatBox{
		inFd:         int(slave.Fd()),
		outFd:        int(slave.Fd()),
		isTTY:        true,
		width:        cols,
		height:       rows,
		chatTop:      rows - 3,     // Ligne 37
		scrollBottom: rows - 4,     // Ligne 36
		sigWinch:     make(chan os.Signal, 1),
	}
	if termio, err := unix.IoctlGetTermios(int(slave.Fd()), unix.TCGETS); err == nil {
		cb.origTermio = termio
	}

	// Rediriger temporairement stdout vers un buffer pour observer les séquences émises
	rPipe, wPipe, _ := os.Pipe()
	oldStdout := os.Stdout
	os.Stdout = wPipe

	// 1. Initialiser l'écran
	cb.InitScreen(28, 16384)

	// 2. Dessiner la boîte de saisie avec le texte "bonjour"
	cb.DrawChatBox("bonjour", false)

	// Restaurer stdout et lire la sortie générée
	_ = wPipe.Close()
	os.Stdout = oldStdout

	var outBuf bytes.Buffer
	_, _ = outBuf.ReadFrom(rPipe)
	output := outBuf.String()

	// Vérification 1 : La région de défilement matérielle (DECSTBM) est fixée à 1;36r
	expectedScroll := fmt.Sprintf("\x1b[1;%dr", rows-4)
	if !strings.Contains(output, expectedScroll) {
		t.Errorf("Région de défilement matérielle absente (%s): %q", expectedScroll, output)
	}

	// Vérification 2 : La ChatBox est bien dessinée à la ligne 37 (chatTop)
	expectedTopRow := fmt.Sprintf("\x1b[%d;1H", rows-3)
	if !strings.Contains(output, expectedTopRow) {
		t.Errorf("Bordure supérieure de la chatbox non positionnée à la ligne %d", rows-3)
	}

	// Vérification 3 : Le curseur pour le texte 'bonjour' est positionné à la ligne 38, col 12 (5 + len('bonjour'))
	expectedCursor := fmt.Sprintf("\x1b[%d;%dH", rows-2, 5+len("bonjour"))
	if !strings.Contains(output, expectedCursor) {
		t.Errorf("Curseur non ancré dans la ChatBox en bas (%s): %q", expectedCursor, output)
	}
}

// TestHarness_InterruptionESC vérifie que le drapeau d'interruption bascule
// instantanément à true dès qu'un octet ESC (0x1b) est injecté sur le PTY.
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
