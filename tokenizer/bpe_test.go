package tokenizer_test

import (
	"os"
	"strings"
	"testing"

	"github.com/hazyhaar/c2slm/gguf"
	"github.com/hazyhaar/c2slm/tokenizer"
)

const modelPath = "/data/models/qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"

func TestQwenTokenizer(t *testing.T) {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("model file not found at %s", modelPath)
	}

	gf, err := gguf.Open(modelPath)
	if err != nil {
		t.Fatalf("gguf.Open failed: %v", err)
	}
	defer gf.Close()

	tok, err := tokenizer.NewTokenizerFromGGUF(gf)
	if err != nil {
		t.Fatalf("NewTokenizerFromGGUF failed: %v", err)
	}

	imStart, imEnd, _ := tok.SpecialTokenIDs()
	if imStart != 151644 {
		t.Errorf("expected imStart=151644, got %d", imStart)
	}
	if imEnd != 151645 {
		t.Errorf("expected imEnd=151645, got %d", imEnd)
	}

	prompt := "<|im_start|>system\nYou are an AI DNS security arbitrator.<|im_end|>\n<|im_start|>user\nArbitrate incident.<|im_end|>\n<|im_start|>assistant\n"
	tokens := tok.Encode(prompt)
	if len(tokens) == 0 {
		t.Fatalf("Encode returned empty tokens")
	}

	if tokens[0] != imStart {
		t.Errorf("expected first token to be imStart (151644), got %d", tokens[0])
	}

	decoded := tok.Decode(tokens)
	if decoded != prompt {
		t.Errorf("roundtrip mismatch:\nexpected: %q\ngot:      %q", prompt, decoded)
	}

	t.Logf("encoded %d characters into %d tokens (ratio: %.2f chars/token)",
		len(prompt), len(tokens), float64(len(prompt))/float64(len(tokens)))
}

func TestTokenizerJSONArbitrationTokens(t *testing.T) {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("model file not found at %s", modelPath)
	}

	gf, err := gguf.Open(modelPath)
	if err != nil {
		t.Fatalf("gguf.Open failed: %v", err)
	}
	defer gf.Close()

	tok, err := tokenizer.NewTokenizerFromGGUF(gf)
	if err != nil {
		t.Fatalf("NewTokenizerFromGGUF failed: %v", err)
	}

	// Test encoding and decoding of typical arbitration verdict string
	verdict := `{"verdict": "BENIGN_AV_TELEMETRY", "confidence": 0.95, "reason": "legitimate antivirus cloud lookup"}`
	tokens := tok.Encode(verdict)
	decoded := tok.Decode(tokens)

	if !strings.Contains(decoded, "BENIGN_AV_TELEMETRY") {
		t.Errorf("decoded does not contain verdict: %s", decoded)
	}
	if decoded != verdict {
		t.Errorf("roundtrip mismatch for JSON:\nexpected: %q\ngot:      %q", verdict, decoded)
	}
}
