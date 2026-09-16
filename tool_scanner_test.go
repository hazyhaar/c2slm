package c2slm

import (
	"strings"
	"testing"
)

// drainTool pousse une suite de fragments et retourne le texte visible agrégé,
// le dernier appel d'outil complété et le drapeau de complétion.
func drainTool(s *ToolScanner, pieces ...string) (clean string, done bool, toolJSON string) {
	var b strings.Builder
	for _, p := range pieces {
		d, _, c := s.Push(p)
		b.WriteString(c)
		if d {
			done = true
			toolJSON = s.ToolJSON()
		}
	}
	return b.String(), done, toolJSON
}

func TestToolScannerPlainText(t *testing.T) {
	s := NewToolScanner()
	clean, done, _ := drainTool(s, "Le domaine ", "c2.example.org ", "est suspect.")
	if done {
		t.Fatal("aucun outil ne devait être détecté")
	}
	want := "Le domaine c2.example.org est suspect."
	if clean != want {
		t.Fatalf("clean = %q, attendu %q", clean, want)
	}
	if s.State() != StText {
		t.Fatalf("état final = %d, attendu StText", s.State())
	}
}

func TestToolScannerThinkCapture(t *testing.T) {
	s := NewToolScanner()
	clean, done, _ := drainTool(s,
		"<thi", "nk>", "raison", "nement", " interne", "</thi", "nk>", "Réponse finale.")
	if done {
		t.Fatal("aucun outil ne devait être détecté")
	}
	if clean != "Réponse finale." {
		t.Fatalf("clean = %q, attendu sans le bloc de raisonnement", clean)
	}
	if got := s.ThinkText(); got != "raisonnement interne" {
		t.Fatalf("ThinkText = %q, attendu %q", got, "raisonnement interne")
	}
}

func TestToolScannerToolCall(t *testing.T) {
	s := NewToolScanner()
	payload := `<tool_call>
{"name":"inspect_proc","arguments":{"pid":42}}
</tool_call>`
	clean, done, toolJSON := drainTool(s, payload)
	if !done {
		t.Fatal("la clôture </tool_call> devait signaler la complétion")
	}
	if clean != "" {
		t.Fatalf("clean = %q, attendu vide (charge captée)", clean)
	}
	if !strings.Contains(toolJSON, `"pid":42`) || !strings.Contains(toolJSON, `"inspect_proc"`) {
		t.Fatalf("ToolJSON incomplet: %q", toolJSON)
	}
	if s.State() != StText {
		t.Fatalf("après clôture l'état doit revenir à StText, got %d", s.State())
	}
}

// TestToolScannerByteSplit prouve que la reconnaissance des balises résiste à
// une découpe arbitraire : chaque octet est livré dans un fragment distinct.
func TestToolScannerByteSplit(t *testing.T) {
	s := NewToolScanner()
	stream := "Début <tool_call>\n{\"name\":\"ban_ip\",\"arguments\":{\"ip\":\"10.0.0.1\"}}\n</tool_call>"
	var clean strings.Builder
	done := false
	for i := 0; i < len(stream); i++ {
		d, _, c := s.Push(stream[i : i+1])
		clean.WriteString(c)
		if d {
			done = true
		}
	}
	if !done {
		t.Fatal("l'appel d'outil découpé octet par octet n'a pas été complété")
	}
	if clean.String() != "Début " {
		t.Fatalf("clean = %q, attendu %q", clean.String(), "Début ")
	}
	if !strings.Contains(s.ToolJSON(), `"10.0.0.1"`) {
		t.Fatalf("charge capturée incorrecte: %q", s.ToolJSON())
	}
}

// TestToolScannerLiteralLessThan vérifie qu'un « < » littéral n'avale pas le
// texte suivant et ne perturbe pas la détection ultérieure d'une balise.
func TestToolScannerLiteralLessThan(t *testing.T) {
	s := NewToolScanner()
	clean, done, _ := drainTool(s, "si a < b alors ", "a<b", " et c")
	if done {
		t.Fatal("aucun outil ne devait être détecté")
	}
	if clean != "si a < b alors a<b et c" {
		t.Fatalf("clean = %q", clean)
	}

	s2 := NewToolScanner()
	clean2, done2, tool2 := drainTool(s2,
		"1 < 2 ", "<tool_call>", "{}", "</tool_call>")
	if !done2 || clean2 != "1 < 2 " {
		t.Fatalf("clean=%q done=%v", clean2, done2)
	}
	if strings.TrimSpace(tool2) != "{}" {
		t.Fatalf("toolJSON = %q", tool2)
	}
}

// TestToolScannerSequentialCalls vérifie que deux appels successifs ne
// concatènent pas leurs charges.
func TestToolScannerSequentialCalls(t *testing.T) {
	s := NewToolScanner()
	if _, done, js := drainTool(s, `<tool_call>{"name":"a"}</tool_call>`); !done || strings.TrimSpace(js) != `{"name":"a"}` {
		t.Fatalf("premier appel: done=%v js=%q", done, js)
	}
	clean, _, _ := drainTool(s, "entre deux")
	if clean != "entre deux" {
		t.Fatalf("texte intermédiaire = %q", clean)
	}
	if _, done, js := drainTool(s, `<tool_call>{"name":"b"}</tool_call>`); !done || strings.TrimSpace(js) != `{"name":"b"}` {
		t.Fatalf("second appel: done=%v js=%q", done, js)
	}
}

func TestToolScannerOverflow(t *testing.T) {
	s := NewToolScanner()
	b := strings.Repeat("x", maxThinkBytes+64)
	_, _, _ = drainTool(s, "<thi", "nk>", b)
	if !s.Overflow() {
		t.Fatal("le dépassement du tampon de raisonnement devait être signalé")
	}
}

// TestToolScannerZeroAlloc impose l'invariant ARCHTIME : Push ne réalise aucune
// allocation sur le tas, ni sur le chemin texte, ni sur le chemin d'outil.
func TestToolScannerZeroAlloc(t *testing.T) {
	s := NewToolScanner()
	plain := "Le domaine c2.01636c69656e742e657866696c.tunnel.example.org presente une entropie elevee."
	tool := `<tool_call>
{"name":"inspect_socket","arguments":{"port":53}}
</tool_call>`

	if allocs := testing.AllocsPerRun(5000, func() {
		s.Reset()
		s.Push(plain)
	}); allocs != 0 {
		t.Fatalf("chemin texte: %v allocations, attendu 0", allocs)
	}

	if allocs := testing.AllocsPerRun(5000, func() {
		s.Reset()
		s.Push(tool)
	}); allocs != 0 {
		t.Fatalf("chemin outil: %v allocations, attendu 0", allocs)
	}

	// Chemin raisonnement, avec balise coupée entre deux fragments.
	if allocs := testing.AllocsPerRun(5000, func() {
		s.Reset()
		s.Push("<thi")
		s.Push("nk>analyse</thi")
		s.Push("nk>fin")
	}); allocs != 0 {
		t.Fatalf("chemin raisonnement: %v allocations, attendu 0", allocs)
	}
}
