package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
	"golang.org/x/term"
)

// ANSI color and styling constants
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

// ChatBox orchestre un REPL moderne de type Claude Code / Ollama :
// 1. Saisie utilisateur propre avec prompt distinctif sans perte du premier caractère.
// 2. Isolation stricte pendant l'inférence (écho désactivé via TCSETSW / TCSADRAIN).
// 3. Interruption instantanée au jeton près via [Échap] ou [Ctrl+C].
// 4. Adaptation dynamique à la largeur du terminal (sans boîtes rigides asymétriques).
type ChatBox struct {
	inFd       int
	outFd      int
	isTTY      bool
	origTermio *unix.Termios
	scanner    *bufio.Scanner
}

// NewChatBox instancie la console interactive.
func NewChatBox() *ChatBox {
	inFd := int(os.Stdin.Fd())
	outFd := int(os.Stdout.Fd())
	isTTY := term.IsTerminal(inFd)

	cb := &ChatBox{
		inFd:    inFd,
		outFd:   outFd,
		isTTY:   isTTY,
		scanner: bufio.NewScanner(os.Stdin),
	}

	if isTTY {
		if t, err := unix.IoctlGetTermios(inFd, unix.TCGETS); err == nil {
			cb.origTermio = t
		}
	}
	return cb
}

func (c *ChatBox) Width() int {
	if !c.isTTY {
		return 80
	}
	w, _, err := term.GetSize(c.inFd)
	if err != nil || w <= 20 {
		return 80
	}
	if w > 120 {
		return 120
	}
	return w
}

// PrintBanner affiche une bannière moderne adaptée à la largeur réelle du terminal.
func (c *ChatBox) PrintBanner(numLayers, maxTokens int) {
	w := c.Width()
	border := strings.Repeat("─", w-2)

	fmt.Printf("\n%s┌%s┐%s\n", ansiCyan, border, ansiReset)
	title := fmt.Sprintf(" nanoGOqwen (Pur Go • AVX2 SIMD • Contexte %dk • %d couches)", maxTokens/1024, numLayers)
	fmt.Printf("%s│%s%s%-*s%s│%s\n", ansiCyan, ansiReset, ansiBold, w-2, title, ansiReset, ansiCyan)
	divider := strings.Repeat("─", w-2)
	fmt.Printf("├%s┤\n", divider)
	fmt.Printf("│ %-54s %*s│\n", "Commandes : /help, /clear, /tools, /exit", w-57, "")
	fmt.Printf("│ %-54s %*s│\n", "Raccourci : [Échap] interrompt le streaming à tout instant", w-57, "")
	fmt.Printf("└%s┘%s\n\n", border, ansiReset)
}

// ReadPrompt lit l'invite utilisateur. En mode canonique, le noyau Linux gère
// nativement l'UTF-8, l'historique et l'édition de ligne sans jamais tronquer
// la marge gauche.
func (c *ChatBox) ReadPrompt() (string, error) {
	fmt.Printf("%s❯%s ", ansiBoldCyan, ansiReset)
	if !c.scanner.Scan() {
		if err := c.scanner.Err(); err != nil {
			return "", err
		}
		return "", io.EOF
	}
	line := strings.TrimSpace(c.scanner.Text())
	return line, nil
}

// BeginInference prépare le terminal pour le streaming du modèle :
// - Bascule en mode non-canonique et coupe l'écho (ECHO=0) via unix.TCSETSW
//   (TCSADRAIN) pour que les frappes utilisateur ne se mélangent jamais au modèle.
// - Lance une goroutine de surveillance écoutant [Échap] (0x1b) et [Ctrl+C] (0x03).
// - Purge tout résidu avant de rendre la main.
func (c *ChatBox) BeginInference() (interrupted *atomic.Bool, endFn func()) {
	interrupted = &atomic.Bool{}

	if !c.isTTY || c.origTermio == nil {
		return interrupted, func() {}
	}

	// Configuration : désactiver ICANON et ECHO, conserver OPOST et ONLCR
	raw := *c.origTermio
	raw.Lflag &^= (unix.ICANON | unix.ECHO)
	raw.Oflag |= (unix.OPOST | unix.ONLCR)
	raw.Cc[unix.VMIN] = 1
	raw.Cc[unix.VTIME] = 0

	// Appliquer avec TCSETSW (attend la fin de transmission des octets en cours)
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

			// Arrêt demandé
			if fds[1].Revents&(unix.POLLIN|unix.POLLHUP|unix.POLLERR) != 0 {
				return
			}

			// Frappe clavier reçue pendant l'inférence
			if fds[0].Revents&unix.POLLIN != 0 {
				var buf [32]byte
				nr, rErr := unix.Read(c.inFd, buf[:])
				if rErr != nil || nr == 0 {
					return
				}
				for i := 0; i < nr; i++ {
					// 0x1b = Échap (ESC), 0x03 = Ctrl+C
					if buf[i] == 0x1b || buf[i] == 0x03 {
						interrupted.Store(true)
						return
					}
					// Tout autre caractère est ignoré et absorbé silencieusement.
				}
			}
		}
	}()

	endFn = func() {
		// Clôture propre de la goroutine de surveillance
		_, _ = stopW.Write([]byte{1})
		<-doneChan
		_ = stopW.Close()
		_ = stopR.Close()

		// Restauration du mode canonique initial avec vidage préalable
		_ = unix.IoctlSetTermios(c.inFd, unix.TCSETSW, c.origTermio)

		// Purge des éventuels caractères frappés pendant l'inférence (TCFLSH / TCIFLUSH)
		_, _, _ = syscall.Syscall(syscall.SYS_IOCTL, uintptr(c.inFd), 0x540b, 0)
	}

	return interrupted, endFn
}

// PrintAssistantHeader amorce la réponse de l'assistant de manière épurée.
func (c *ChatBox) PrintAssistantHeader() {
	fmt.Printf("\n%s◇ nanoGOqwen%s\n", ansiBoldMag, ansiReset)
}

// PrintToolCall affiche un appel d'outil déclenché par le modèle.
func (c *ChatBox) PrintToolCall(name, args string) {
	fmt.Printf("\n%s┌ ⚙️  [%s]%s %s\n", ansiYellow, name, ansiReset, args)
}

// PrintToolResult affiche la confirmation de retour de l'outil.
func (c *ChatBox) PrintToolResult(output string) {
	preview := output
	if len(preview) > 100 {
		preview = preview[:100] + "..."
	}
	fmt.Printf("%s└ 📥  [%d octets reçus]%s %s\n\n", ansiYellow, len(output), ansiReset, preview)
}

// PrintAssistantFooter clôture le tour de dialogue et présente les métriques.
func (c *ChatBox) PrintAssistantFooter(interrupted bool, totalTokens int, elapsed time.Duration, seqLen, maxTokens int) {
	fmt.Println()
	if interrupted {
		fmt.Printf("%s⏹  [Inférence interrompue par l'utilisateur (ESC)]%s\n", ansiBoldRed, ansiReset)
	}

	tps := 0.0
	if elapsed.Seconds() > 0 && totalTokens > 0 {
		tps = float64(totalTokens) / elapsed.Seconds()
	}

	w := c.Width()
	info := fmt.Sprintf(" %d jetons en %v (%.2f tok/s) • cache %d/%d ",
		totalTokens, elapsed.Round(time.Millisecond), tps, seqLen, maxTokens)

	dashLen := (w - len(info)) / 2
	if dashLen < 2 {
		dashLen = 2
	}
	sep := strings.Repeat("─", dashLen)
	fmt.Printf("%s%s%s%s%s\n\n", ansiGray, sep, info, sep, ansiReset)
}

// PrintHelp affiche l'aide sur les commandes intégrées.
func (c *ChatBox) PrintHelp() {
	fmt.Printf("\n%sCommandes disponibles :%s\n", ansiBold, ansiReset)
	fmt.Printf("  %s/help%s      : Affiche ce menu d'aide\n", ansiBoldCyan, ansiReset)
	fmt.Printf("  %s/clear%s     : Réinitialise le KV Cache à zéro (vide le contexte passé)\n", ansiBoldCyan, ansiReset)
	fmt.Printf("  %s/tools%s     : Liste les outils actifs et leurs descriptions\n", ansiBoldCyan, ansiReset)
	fmt.Printf("  %s/stats%s     : Affiche l'état d'occupation mémoire du cache de contexte\n", ansiBoldCyan, ansiReset)
	fmt.Printf("  %s/exit%s      : Quitte l'application\n", ansiBoldCyan, ansiReset)
	fmt.Printf("  %s[Échap]%s    : Interrompt immédiatement le streaming en cours\n\n", ansiBoldRed, ansiReset)
}
