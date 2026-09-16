package c2slm_test

import (
	"os"
	"testing"
	"time"

	"github.com/hazyhaar/c2slm"
)

const modelPath = "/data/models/qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"

func TestEngineLoadAndGenerate(t *testing.T) {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("model file not found at %s", modelPath)
	}

	start := time.Now()
	engine, err := c2slm.NewEngine(modelPath)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}
	defer engine.Close()
	t.Logf("Engine loaded in %v", time.Since(start))

	prompt := "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\nSay hello in one word.<|im_end|>\n<|im_start|>assistant\n"

	genStart := time.Now()
	out, err := engine.Generate(prompt, 10, []string{"\n", "<|im_end|>"})
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	t.Logf("Generated in %v: %q", time.Since(genStart), out)
}
