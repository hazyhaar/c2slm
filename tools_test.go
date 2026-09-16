package c2slm

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"
)

func TestToolRegistryRegisterAndDispatch(t *testing.T) {
	reg := NewToolRegistry()

	if err := reg.Register(ToolDefinition{Name: "  "}, func(json.RawMessage) (string, error) { return "", nil }); err == nil {
		t.Fatal("un nom vide devait être refusé")
	}
	if err := reg.Register(ToolDefinition{Name: "x"}, nil); err == nil {
		t.Fatal("un handler nil devait être refusé")
	}

	echo := func(args json.RawMessage) (string, error) {
		return "echo:" + string(args), nil
	}
	if err := reg.Register(ToolDefinition{
		Name:        "inspect_proc",
		Description: "inspecte un processus",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"pid":{"type":"integer"}},"required":["pid"]}`),
	}, echo); err != nil {
		t.Fatalf("Register: %v", err)
	}
	if reg.Len() != 1 {
		t.Fatalf("Len = %d, attendu 1", reg.Len())
	}

	// Une réinscription ne duplique pas l'ordre d'énumération.
	if err := reg.Register(ToolDefinition{Name: "inspect_proc"}, echo); err != nil {
		t.Fatalf("re-Register: %v", err)
	}
	if reg.Len() != 1 {
		t.Fatalf("Len après réinscription = %d, attendu 1", reg.Len())
	}

	out, err := reg.Dispatch("inspect_proc", json.RawMessage(`{"pid":7}`))
	if err != nil {
		t.Fatalf("Dispatch: %v", err)
	}
	if out != `echo:{"pid":7}` {
		t.Fatalf("Dispatch = %q", out)
	}

	_, err = reg.Dispatch("absent", nil)
	if !errors.Is(err, ErrUnknownTool) {
		t.Fatalf("Dispatch inconnu = %v, attendu ErrUnknownTool", err)
	}

	// Un handler qui échoue remonte l'erreur.
	reg.Register(ToolDefinition{Name: "boom"}, func(json.RawMessage) (string, error) {
		return "", errors.New("échec")
	})
	if _, err := reg.Dispatch("boom", nil); err == nil || err.Error() != "échec" {
		t.Fatalf("erreur handler = %v", err)
	}
}

func TestToolRegistrySystemToolsBlock(t *testing.T) {
	empty := NewToolRegistry()
	if block := empty.SystemToolsBlock(); block != "" {
		t.Fatalf("bloc attendu vide, got %q", block)
	}

	reg := NewToolRegistry()
	reg.Register(ToolDefinition{
		Name:        "inspect_socket",
		Description: "localise l'émetteur d'un socket",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"port":{"type":"integer"}}}`),
	}, func(json.RawMessage) (string, error) { return "", nil })

	block := reg.SystemToolsBlock()
	for _, want := range []string{
		"# Tools",
		"<tools>",
		"</tools>",
		"<tool_call>",
		"</tool_call>",
		"inspect_socket",
		`"type":"function"`,
	} {
		if !strings.Contains(block, want) {
			t.Errorf("bloc outils sans %q:\n%s", want, block)
		}
	}
}

func TestParseToolCall(t *testing.T) {
	call, err := ParseToolCall([]byte(`{"name":"ban_ip","arguments":{"ip":"10.0.0.1","duration_seconds":60}}`))
	if err != nil {
		t.Fatalf("objet: %v", err)
	}
	if call.Name != "ban_ip" || !strings.Contains(string(call.Arguments), `"10.0.0.1"`) {
		t.Fatalf("objet parsé = %+v", call)
	}

	// Arguments encodés en chaîne JSON.
	call, err = ParseToolCall([]byte(`{"name":"inspect_proc","arguments":"{\"pid\":9}"}`))
	if err != nil {
		t.Fatalf("string args: %v", err)
	}
	if strings.TrimSpace(string(call.Arguments)) != `{"pid":9}` {
		t.Fatalf("arguments déballés = %q", call.Arguments)
	}

	// Tableau : premier appel exploitable.
	call, err = ParseToolCall([]byte(`[{"name":"a","arguments":{"x":1}},{"name":"b"}]`))
	if err != nil {
		t.Fatalf("tableau: %v", err)
	}
	if call.Name != "a" {
		t.Fatalf("tableau: %+v", call)
	}

	// Arguments absents : objet vide normalisé.
	call, err = ParseToolCall([]byte(`{"name":"c"}`))
	if err != nil {
		t.Fatalf("sans args: %v", err)
	}
	if strings.TrimSpace(string(call.Arguments)) != "{}" {
		t.Fatalf("arguments par défaut = %q", call.Arguments)
	}

	if _, err := ParseToolCall([]byte(`{"arguments":{}}`)); err == nil {
		t.Fatal("un appel sans nom devait être refusé")
	}
	if _, err := ParseToolCall([]byte(`   `)); err == nil {
		t.Fatal("un appel vide devait être refusé")
	}
}

func TestBuildToolsPrompt(t *testing.T) {
	reg := NewToolRegistry()
	reg.Register(ToolDefinition{Name: "inspect_proc"}, func(json.RawMessage) (string, error) { return "", nil })

	prompt := buildToolsPrompt("Tu es un arbitre DNS.", reg, "Analyse ce tunnel.")
	if !strings.HasPrefix(prompt, "<|im_start|>system\n") {
		t.Fatalf("préfixe système manquant: %q", prompt)
	}
	if !strings.Contains(prompt, "Tu es un arbitre DNS.") {
		t.Error("consigne système absente")
	}
	if !strings.Contains(prompt, "# Tools") || !strings.Contains(prompt, "inspect_proc") {
		t.Error("bloc outils absent")
	}
	if !strings.Contains(prompt, "<|im_start|>user\nAnalyse ce tunnel.<|im_end|>\n<|im_start|>assistant\n") {
		t.Errorf("tour utilisateur mal formé: %q", prompt)
	}

	withoutReg := buildToolsPrompt("sys", nil, "q")
	if strings.Contains(withoutReg, "# Tools") {
		t.Error("bloc outils présent sans registre")
	}
}

func TestToolCallTurn(t *testing.T) {
	turn := toolCallTurn(`{"name":"ban_ip"}`, `{"banned":true}`)
	want := "<tool_call>\n{\"name\":\"ban_ip\"}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n{\"banned\":true}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
	if turn != want {
		t.Fatalf("tour outil = %q, attendu %q", turn, want)
	}
}
