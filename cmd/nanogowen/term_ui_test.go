package main

import (
	"bytes"
	"io"
	"os"
	"strings"
	"testing"
	"time"
)

func TestChatBox_NonTTY(t *testing.T) {
	cb := NewChatBox()
	if cb.isTTY {
		t.Log("Environnement avec TTY détecté")
	} else {
		t.Log("Environnement non-TTY détecté")
	}

	interrupted, endFn := cb.BeginInference()
	if interrupted == nil {
		t.Fatal("interrupted ne doit pas être nil")
	}
	if interrupted.Load() {
		t.Fatal("interrupted doit être false initialement")
	}
	endFn()
}

func TestChatBox_ScreenAndFooters(t *testing.T) {
	cb := NewChatBox()

	// Rediriger stdout temporairement pour capturer l'affichage
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	oldStdout := os.Stdout
	os.Stdout = w
	defer func() {
		os.Stdout = oldStdout
	}()

	cb.InitScreen(28, 16384)
	cb.DrawChatBox("bonjour", false)
	cb.DrawChatBox("", true)
	cb.PrintAssistantHeader()
	cb.PrintToolCall("fetch_url", `{"url":"https://go.dev"}`)
	cb.PrintToolResult("Documentation Go 1.27")
	cb.PrintAssistantFooter(false, 42, 1500*time.Millisecond, 150, 16384)
	cb.PrintAssistantFooter(true, 10, 500*time.Millisecond, 160, 16384)
	cb.PrintHelp()
	cb.ResetScreen()

	_ = w.Close()
	var buf bytes.Buffer
	_, _ = io.Copy(&buf, r)
	out := buf.String()

	if !strings.Contains(out, "nanoGOqwen") {
		t.Errorf("nanoGOqwen absent: %s", out)
	}
	if !strings.Contains(out, "fetch_url") {
		t.Errorf("fetch_url absent: %s", out)
	}
	if !strings.Contains(out, "tok/s") {
		t.Errorf("Statistiques de débit absentes: %s", out)
	}
	if !strings.Contains(out, "/help") {
		t.Errorf("Aide absente: %s", out)
	}
}
