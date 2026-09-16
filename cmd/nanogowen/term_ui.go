package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
	"golang.org/x/term"
)

// Constantes de styles ANSI
const (
	ansiReset      = "\x1b[0m"
	ansiBold       = "\x1b[1m"
	ansiDim        = "\x1b[2m"
	ansiCyan       = "\x1b[36m"
	ansiBoldCyan   = "\x1b[1;36m"
	ansiMagenta    = "\x1b[35m"
	ansiBoldMag    = "\x1b[1;35m"
	ansiYellow     = "\x1b[33m"
	ansiBoldYellow = "\x1b[1;33m"
	ansiRed        = "\x1b[31m"
	ansiBoldRed    = "\x1b[1;31m"
	ansiGreen      = "\x1b[32m"
	ansiGray       = "\x1b[90m"
)

// ChatBox implémente une interface TUI plein écran avec zone de saisie ancrée
// EN BAS de la fenêtre du terminal (DECSTBM scroll region + alternate screen),
// conforme aux standards des CLI conversationnels modernes (Claude Code, Ollama).
type ChatBox struct {
	inFd         int
	outFd        int
	isTTY        bool
	width        int
	height       int
	scrollBottom int
	chatTop      int
	origTermio   *unix.Termios
	scanner      *bufio.Scanner
	sigWinch     chan os.Signal
}

// NewChatBox initialise l'interface et configure la fenêtre.
func NewChatBox() *ChatBox {
	inFd := int(os.Stdin.Fd())
	outFd := int(os.Stdout.Fd())
	isTTY := term.IsTerminal(inFd)

	cb := &ChatBox{
		inFd:     inFd,
		outFd:    outFd,
		isTTY:    isTTY,
		scanner:  bufio.NewScanner(os.Stdin),
		sigWinch: make(chan os.Signal, 1),
	}

	if isTTY {
		if t, err := unix.IoctlGetTermios(inFd, unix.TCGETS); err == nil {
			cb.origTermio = t
		}
		cb.updateDimensions()
		signal.Notify(cb.sigWinch, syscall.SIGWINCH)
		go cb.handleResize()
	} else {
		cb.width = 80
		cb.height = 24
		cb.chatTop = 20
		cb.scrollBottom = 19
	}
	return cb
}

func (c *ChatBox) updateDimensions() {
	w, h, err := term.GetSize(c.inFd)
	if err != nil || w <= 20 || h <= 10 {
		c.width = 80
		c.height = 24
	} else {
		c.width = w
		c.height = h
	}
	// On réserve 4 lignes en bas pour la ChatBox
	c.chatTop = c.height - 3
	c.scrollBottom = c.chatTop - 1
}

func (c *ChatBox) handleResize() {
	for range c.sigWinch {
		c.updateDimensions()
		c.ApplyScrollRegion()
		c.DrawChatBox("", false)
	}
}

// InitScreen bascule sur l'alternate screen et active la zone de défilement.
func (c *ChatBox) InitScreen(numLayers, maxTokens int) {
	if !c.isTTY {
		return
	}
	// Alternate screen (\x1b[?1049h), clear (\x1b[2J), curseur en haut (\x1b[H)
	fmt.Print("\x1b[?1049h\x1b[2J\x1b[H")
	c.ApplyScrollRegion()

	// Bannière affichée dans la zone de défilement supérieure
	w := c.width
	border := strings.Repeat("─", w-2)
	fmt.Printf("%s┌%s┐%s\r\n", ansiCyan, border, ansiReset)
	title := fmt.Sprintf(" nanoGOqwen (Pur Go • AVX2 SIMD • Contexte %dk • %d couches)", maxTokens/1024, numLayers)
	fmt.Printf("%s│%s%s%-*s%s│%s\r\n", ansiCyan, ansiReset, ansiBold, w-2, title, ansiReset, ansiCyan)
	fmt.Printf("├%s┤\r\n", border)
	fmt.Printf("│ %-54s %*s│\r\n", "Commandes : /help, /clear, /tools, /stats, /exit", w-57, "")
	fmt.Printf("│ %-54s %*s│\r\n", "Raccourci : [Échap] interrompt la génération à tout instant", w-57, "")
	fmt.Printf("└%s┘%s\r\n\r\n", border, ansiReset)

	// Dessine la chatbox ancrée tout en bas
	c.DrawChatBox("", false)
}

// ApplyScrollRegion configure la région matérielle de défilement (lignes 1 à scrollBottom).
func (c *ChatBox) ApplyScrollRegion() {
	if !c.isTTY {
		return
	}
	fmt.Printf("\x1b[1;%dr", c.scrollBottom)
}

// ResetScreen restaure le terminal dans son état antérieur.
func (c *ChatBox) ResetScreen() {
	if !c.isTTY {
		return
	}
	// Réinitialise la scroll region, quitte l'alternate screen, réaffiche le curseur
	fmt.Print("\x1b[r\x1b[?1049l\x1b[?25h")
}

// DrawChatBox dessine la boîte de saisie ancrée en bas de l'écran.
func (c *ChatBox) DrawChatBox(content string, isGenerating bool) {
	if !c.isTTY {
		return
	}
	w := c.width
	if w <= 10 {
		return
	}
	// Sauvegarder la position du curseur
	fmt.Print("\x1b[s")

	// Ligne 1 de la Chatbox (chatTop) : Bordure supérieure
	fmt.Printf("\x1b[%d;1H\x1b[2K", c.chatTop)
	if isGenerating {
		tag := " 🤖 nanoGOqwen (Inférence en cours...) "
		rem := w - 2 - len(tag)
		if rem < 0 {
			rem = 0
		}
		fmt.Printf("%s┌──%s%s%s%s┐%s", ansiYellow, ansiBoldYellow, tag, ansiReset, ansiYellow, strings.Repeat("─", rem))
	} else {
		tag := " 💬 Vous (/help, /clear, /tools, /stats, /exit) "
		rem := w - 2 - len(tag)
		if rem < 0 {
			rem = 0
		}
		fmt.Printf("%s┌──%s%s%s%s┐%s", ansiCyan, ansiBoldCyan, tag, ansiReset, ansiCyan, strings.Repeat("─", rem))
	}

	// Ligne 2 : Zone de saisie active
	fmt.Printf("\x1b[%d;1H\x1b[2K", c.chatTop+1)
	if isGenerating {
		fmt.Printf("%s│%s %s⏳ Génération du modèle en streaming...%s%*s%s│%s",
			ansiYellow, ansiReset, ansiDim, ansiReset, w-42, "", ansiYellow, ansiReset)
	} else {
		fmt.Printf("%s│%s %s❯%s %-*s%s│%s",
			ansiCyan, ansiReset, ansiBoldCyan, ansiReset, w-6, content, ansiCyan, ansiReset)
	}

	// Ligne 3 : Ligne de statut / raccourcis
	fmt.Printf("\x1b[%d;1H\x1b[2K", c.chatTop+2)
	if isGenerating {
		status := " [Échap] Interrompre la génération immédiatement "
		rem := w - 2 - len(status)
		if rem < 0 {
			rem = 0
		}
		fmt.Printf("%s└─%s%s%s%s┘%s", ansiYellow, ansiBoldRed, status, ansiReset, ansiYellow, strings.Repeat("─", rem))
	} else {
		status := " [Échap] Interrompre │ [Entrée] Envoyer "
		rem := w - 2 - len(status)
		if rem < 0 {
			rem = 0
		}
		fmt.Printf("%s└─%s%s%s%s┘%s", ansiCyan, ansiGray, status, ansiReset, ansiCyan, strings.Repeat("─", rem))
	}

	// Restaurer la position du curseur
	fmt.Print("\x1b[u")
}

// ReadPrompt attend la saisie utilisateur directement dans la Chatbox ancrée en bas.
func (c *ChatBox) ReadPrompt() (string, error) {
	if !c.isTTY {
		fmt.Print("❯ ")
		if !c.scanner.Scan() {
			return "", io.EOF
		}
		return strings.TrimSpace(c.scanner.Text()), nil
	}

	oldState, err := term.MakeRaw(c.inFd)
	if err != nil {
		fmt.Print("❯ ")
		if !c.scanner.Scan() {
			return "", io.EOF
		}
		return strings.TrimSpace(c.scanner.Text()), nil
	}
	defer func() {
		_ = term.Restore(c.inFd, oldState)
	}()

	var inputRunes []rune
	promptRow := c.chatTop + 1

	for {
		c.DrawChatBox(string(inputRunes), false)

		// Positionner le curseur exactement après "│ ❯ " (colonne 5 + offset)
		cursorCol := 5 + len(string(inputRunes))
		fmt.Printf("\x1b[%d;%dH\x1b[?25h", promptRow, cursorCol)

		var buf [16]byte
		nr, rErr := unix.Read(c.inFd, buf[:])
		if rErr != nil || nr == 0 {
			return "", io.EOF
		}

		// Touches de contrôle
		switch buf[0] {
		case '\r', '\n': // Validation
			text := strings.TrimSpace(string(inputRunes))
			// Effacer la saisie dans la chatbox
			c.DrawChatBox("", false)
			// Positionner dans la zone de scroll pour afficher le message validé
			fmt.Printf("\x1b[%d;1H\r\n%s❯ %s%s\r\n", c.scrollBottom, ansiBoldCyan, text, ansiReset)
			return text, nil

		case 3: // Ctrl+C
			return "/exit", nil

		case 4: // Ctrl+D
			if len(inputRunes) == 0 {
				return "/exit", nil
			}

		case 127, 8: // Backspace
			if len(inputRunes) > 0 {
				inputRunes = inputRunes[:len(inputRunes)-1]
			}

		case 21: // Ctrl+U : effacer toute la ligne
			inputRunes = nil

		case 27: // Échap ou séquence flèche
			if nr >= 3 && buf[1] == '[' {
				// Séquences flèches ignorées pour le prompt simple
				continue
			}
			// Échap sur ligne vide
			if len(inputRunes) == 0 {
				continue
			}

		default:
			// Insertion de caractères imprimables
			if buf[0] >= 32 {
				s := string(buf[:nr])
				for _, r := range s {
					if r >= 32 {
						inputRunes = append(inputRunes, r)
					}
				}
			}
		}
	}
}

// BeginInference prépare l'inférence :
// 1. Affiche le statut d'inférence dans la ChatBox en bas.
// 2. Positionne le curseur dans la région supérieure de défilement.
// 3. Intercepte [Échap] et [Ctrl+C] en tâche de fond.
func (c *ChatBox) BeginInference() (interrupted *atomic.Bool, endFn func()) {
	interrupted = &atomic.Bool{}

	if !c.isTTY || c.origTermio == nil {
		return interrupted, func() {}
	}

	// Met à jour la chatbox en mode génération
	c.DrawChatBox("", true)

	// Positionne le curseur dans la zone de scroll pour le streaming
	fmt.Printf("\x1b[%d;1H", c.scrollBottom)

	// Mode non-canonique et pas d'écho
	raw := *c.origTermio
	raw.Lflag &^= (unix.ICANON | unix.ECHO)
	raw.Oflag |= (unix.OPOST | unix.ONLCR)
	raw.Cc[unix.VMIN] = 1
	raw.Cc[unix.VTIME] = 0
	_ = unix.IoctlSetTermios(c.inFd, unix.TCSETSW, &raw)

	stopR, stopW, pipeErr := os.Pipe()
	if pipeErr != nil {
		return interrupted, func() {
			_ = unix.IoctlSetTermios(c.inFd, unix.TCSETSW, c.origTermio)
		}
	}

	doneChan := make(chan struct{})
	go func() {
		defer close(doneChan)
		fds := []unix.PollFd{
			{Fd: int32(c.inFd), Events: unix.POLLIN},
			{Fd: int32(stopR.Fd()), Events: unix.POLLIN},
		}

		for {
			n, pollErr := unix.Poll(fds, -1)
			if pollErr != nil {
				if errors.Is(pollErr, syscall.EINTR) {
					continue
				}
				return
			}
			if n <= 0 {
				return
			}

			// Fin demandée
			if fds[1].Revents&(unix.POLLIN|unix.POLLHUP|unix.POLLERR) != 0 {
				return
			}

			// Frappe reçue
			if fds[0].Revents&unix.POLLIN != 0 {
				var b [32]byte
				nr, rErr := unix.Read(c.inFd, b[:])
				if rErr != nil || nr == 0 {
					return
				}
				for i := 0; i < nr; i++ {
					if b[i] == 0x1b || b[i] == 0x03 {
						interrupted.Store(true)
						return
					}
				}
			}
		}
	}()

	endFn = func() {
		_, _ = stopW.Write([]byte{1})
		<-doneChan
		_ = stopW.Close()
		_ = stopR.Close()

		// Restaure le terminal
		_ = unix.IoctlSetTermios(c.inFd, unix.TCSETSW, c.origTermio)
		_, _, _ = syscall.Syscall(syscall.SYS_IOCTL, uintptr(c.inFd), 0x540b, 0) // TCFLSH

		// Remet la chatbox en mode saisie
		c.DrawChatBox("", false)
	}

	return interrupted, endFn
}

// PrintAssistantHeader prépare la réponse dans la zone de défilement haute.
func (c *ChatBox) PrintAssistantHeader() {
	if c.isTTY {
		fmt.Printf("\x1b[%d;1H\r\n%s◇ nanoGOqwen%s\r\n", c.scrollBottom, ansiBoldMag, ansiReset)
	} else {
		fmt.Printf("\n◇ nanoGOqwen\n")
	}
}

// PrintToolCall affiche l'appel d'outil dans la zone haute.
func (c *ChatBox) PrintToolCall(name, args string) {
	if c.isTTY {
		fmt.Printf("\r\n%s┌ ⚙️  [%s]%s %s\r\n", ansiYellow, name, ansiReset, args)
	} else {
		fmt.Printf("\n┌ ⚙️  [%s] %s\n", name, args)
	}
}

// PrintToolResult affiche le résultat de l'outil dans la zone haute.
func (c *ChatBox) PrintToolResult(output string) {
	preview := output
	if len(preview) > 100 {
		preview = preview[:100] + "..."
	}
	if c.isTTY {
		fmt.Printf("%s└ 📥  [%d octets reçus]%s %s\r\n\r\n", ansiYellow, len(output), ansiReset, preview)
	} else {
		fmt.Printf("└ 📥  [%d octets reçus] %s\n\n", len(output), preview)
	}
}

// PrintAssistantFooter affiche les métriques de fin de génération.
func (c *ChatBox) PrintAssistantFooter(interrupted bool, totalTokens int, elapsed time.Duration, seqLen, maxTokens int) {
	if interrupted {
		if c.isTTY {
			fmt.Printf("\r\n%s⏹  [Inférence interrompue par l'utilisateur (ESC)]%s\r\n", ansiBoldRed, ansiReset)
		} else {
			fmt.Printf("\n⏹  [Inférence interrompue par l'utilisateur (ESC)]\n")
		}
	}

	tps := 0.0
	if elapsed.Seconds() > 0 && totalTokens > 0 {
		tps = float64(totalTokens) / elapsed.Seconds()
	}

	w := c.width
	info := fmt.Sprintf(" %d jetons en %v (%.2f tok/s) • cache %d/%d ",
		totalTokens, elapsed.Round(time.Millisecond), tps, seqLen, maxTokens)

	dashLen := (w - len(info)) / 2
	if dashLen < 2 {
		dashLen = 2
	}
	sep := strings.Repeat("─", dashLen)

	if c.isTTY {
		fmt.Printf("\r\n%s%s%s%s%s\r\n", ansiGray, sep, info, sep, ansiReset)
	} else {
		fmt.Printf("\n%s%s%s%s%s\n", ansiGray, sep, info, sep, ansiReset)
	}
}

// PrintHelp affiche l'aide dans la zone de défilement haute.
func (c *ChatBox) PrintHelp() {
	c.ApplyScrollRegion()
	msg := fmt.Sprintf("\r\n%sCommandes disponibles :%s\r\n"+
		"  %s/help%s      : Affiche ce menu d'aide\r\n"+
		"  %s/clear%s     : Réinitialise le KV Cache à zéro (vide le contexte passé)\r\n"+
		"  %s/tools%s     : Liste les outils actifs et leurs descriptions\r\n"+
		"  %s/stats%s     : Affiche l'état d'occupation mémoire du cache de contexte\r\n"+
		"  %s/exit%s      : Quitte l'application\r\n"+
		"  %s[Échap]%s    : Interrompt immédiatement le streaming en cours\r\n\r\n",
		ansiBold, ansiReset,
		ansiBoldCyan, ansiReset,
		ansiBoldCyan, ansiReset,
		ansiBoldCyan, ansiReset,
		ansiBoldCyan, ansiReset,
		ansiBoldCyan, ansiReset,
		ansiBoldRed, ansiReset)

	if c.isTTY {
		fmt.Printf("\x1b[%d;1H%s", c.scrollBottom, msg)
	} else {
		fmt.Print(msg)
	}
}
