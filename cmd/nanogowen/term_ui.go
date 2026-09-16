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

// ChatBox gère l'affichage en console stylisée ("chatbox"), l'isolation stricte
// des frappes utilisateur pendant l'inférence (désactivation de l'écho tty)
// et l'interception instantanée des touches d'interruption [Échap] et [Ctrl+C].
type ChatBox struct {
	inFd       int
	outFd      int
	isTTY      bool
	origState  *term.State
	origTermio *unix.Termios
}

// NewChatBox initialise la gestion de terminal pour stdin et stdout.
func NewChatBox() *ChatBox {
	inFd := int(os.Stdin.Fd())
	outFd := int(os.Stdout.Fd())
	isTTY := term.IsTerminal(inFd)

	cb := &ChatBox{
		inFd:  inFd,
		outFd: outFd,
		isTTY: isTTY,
	}

	if isTTY {
		if t, err := unix.IoctlGetTermios(inFd, unix.TCGETS); err == nil {
			cb.origTermio = t
		}
	}
	return cb
}

// PrintBanner affiche l'en-tête de bienvenue et les raccourcis disponibles.
func (c *ChatBox) PrintBanner(modelName string, numLayers, maxTokens int) {
	fmt.Println("╭──────────────────────────────────────────────────────────────────────────────╮")
	fmt.Printf("│  🤖 %-72s │\n", "nanoGOqwen (Inféreur 100% Pur Go • CGo=0 • AVX2 SIMD • Contexte "+fmt.Sprintf("%dk", maxTokens/1024)+")")
	fmt.Println("├──────────────────────────────────────────────────────────────────────────────┤")
	fmt.Println("│  ⌨️  [Échap] ou [Ctrl+C] : Interrompre immédiatement la génération en cours  │")
	fmt.Println("│  💬 Tapez votre message et appuyez sur [Entrée]                              │")
	fmt.Println("│  🚪 Entrez 'exit' ou 'quit' pour quitter la session                          │")
	fmt.Println("╰──────────────────────────────────────────────────────────────────────────────╯")
}

// ReadPrompt affiche la boîte de saisie "chatbox" et lit une ligne utilisateur.
// En mode TTY, elle active l'édition interactive (flèches, backspace, etc.)
// sans décalage ni pollution visuelle.
func (c *ChatBox) ReadPrompt() (string, error) {
	fmt.Println("\n╭── 👤 Vous ───────────────────────────────────────────────────────────────────╮")

	if !c.isTTY {
		fmt.Print("│ > ")
		sc := bufio.NewScanner(os.Stdin)
		if !sc.Scan() {
			fmt.Println("╰──────────────────────────────────────────────────────────────────────────────╯")
			return "", io.EOF
		}
		line := strings.TrimSpace(sc.Text())
		fmt.Println("╰──────────────────────────────────────────────────────────────────────────────╯")
		return line, nil
	}

	// Mode TTY interactif avec édition propre via term.Terminal
	oldState, err := term.MakeRaw(c.inFd)
	if err != nil {
		fmt.Print("│ > ")
		sc := bufio.NewScanner(os.Stdin)
		if !sc.Scan() {
			fmt.Println("╰──────────────────────────────────────────────────────────────────────────────╯")
			return "", io.EOF
		}
		line := strings.TrimSpace(sc.Text())
		fmt.Println("╰──────────────────────────────────────────────────────────────────────────────╯")
		return line, nil
	}

	termRW := struct {
		io.Reader
		io.Writer
	}{os.Stdin, os.Stdout}

	t := term.NewTerminal(termRW, "│ > ")
	line, err := t.ReadLine()
	_ = term.Restore(c.inFd, oldState)

	fmt.Println("╰──────────────────────────────────────────────────────────────────────────────╯")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(line), nil
}

// BeginInference active le mode silencieux pendant la génération :
// 1. Coupe l'écho TTY (ECHO=0) : les frappes au clavier ne s'impriment pas
//    à l'écran et ne se mélangent plus au flux de tokens du modèle.
// 2. Passe en mode non-canonique (ICANON=0) avec une goroutine de veille
//    qui intercepte en temps réel [Échap] (\x1b) et [Ctrl+C] (\x03).
// 3. Conserve la traduction \n -> \r\n (OPOST+ONLCR) pour que le streaming
//    s'affiche proprement.
// 4. Retourne un booléen atomique 'interrupted' et une fonction de clôture.
func (c *ChatBox) BeginInference() (interrupted *atomic.Bool, endFn func()) {
	interrupted = &atomic.Bool{}

	if !c.isTTY {
		return interrupted, func() {}
	}

	origTerm, err := unix.IoctlGetTermios(c.inFd, unix.TCGETS)
	if err != nil {
		return interrupted, func() {}
	}

	// Configuration : non-canonique, pas d'écho, mais sortie OPOST+ONLCR active
	raw := *origTerm
	raw.Lflag &^= (unix.ICANON | unix.ECHO)
	raw.Oflag |= (unix.OPOST | unix.ONLCR)
	raw.Cc[unix.VMIN] = 1
	raw.Cc[unix.VTIME] = 0

	_ = unix.IoctlSetTermios(c.inFd, unix.TCSETS, &raw)

	// Pipe de signalisation pour réveiller et clore proprement la goroutine de veille
	stopR, stopW, pipeErr := os.Pipe()
	if pipeErr != nil {
		return interrupted, func() {
			_ = unix.IoctlSetTermios(c.inFd, unix.TCSETS, origTerm)
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

			// Signal de fin d'inférence reçu
			if fds[1].Revents&(unix.POLLIN|unix.POLLHUP|unix.POLLERR) != 0 {
				return
			}

			// Frappe clavier captée pendant l'inférence
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
					// Tout autre caractère est intentionnellement consommé et ignoré :
					// il n'est pas répercuté à l'écran et ne souille pas le buffer suivant.
				}
			}
		}
	}()

	endFn = func() {
		// Réveille et arrête la goroutine de surveillance
		_, _ = stopW.Write([]byte{1})
		<-doneChan
		_ = stopW.Close()
		_ = stopR.Close()

		// Restauration de l'état initial du TTY
		_ = unix.IoctlSetTermios(c.inFd, unix.TCSETS, origTerm)

		// Purge des éventuels octets résiduels dans le tampon d'entrée noyau (TCFLSH / TCIFLUSH)
		_, _, _ = syscall.Syscall(syscall.SYS_IOCTL, uintptr(c.inFd), 0x540b, 0)
	}

	return interrupted, endFn
}

// PrintAssistantHeader ouvre la boîte de réponse de l'assistant.
func (c *ChatBox) PrintAssistantHeader() {
	fmt.Println("\n╭── 🤖 nanoGOqwen ────────────────────────────────────────────────────────────╮")
	fmt.Print("│ ")
}

// PrintAssistantFooter referme la boîte de réponse et affiche les métriques.
func (c *ChatBox) PrintAssistantFooter(interrupted bool, totalTokens int, elapsed time.Duration, seqLen, maxTokens int) {
	if interrupted {
		fmt.Println("\n│")
		fmt.Println("│ ⏹️  [Inférence interrompue par l'utilisateur (ESC)]")
	}
	fmt.Println("\n╰──────────────────────────────────────────────────────────────────────────────╯")

	tps := 0.0
	if elapsed.Seconds() > 0 && totalTokens > 0 {
		tps = float64(totalTokens) / elapsed.Seconds()
	}
	fmt.Printf("[%d jetons en %v (%.2f tok/s) | cache=%d/%d]\n",
		totalTokens, elapsed.Round(time.Millisecond), tps, seqLen, maxTokens)
}
