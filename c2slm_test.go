package c2slm_test

import (
	"math"
	"os"
	"testing"
	"time"

	"github.com/hazyhaar/c2slm"
	"github.com/hazyhaar/c2slm/engine"
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

// TestSessionDifferentialIngestion proves that ingesting a prompt in two
// chunks produces exactly the same logits as ingesting it in one shot: the
// cached prefix is never recomputed and causality is preserved. It also
// verifies the token fingerprint and the persistence of a streaming session.
func TestSessionDifferentialIngestion(t *testing.T) {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("model file not found at %s", modelPath)
	}

	e, err := c2slm.NewEngine(modelPath)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}
	defer e.Close()

	prompt := "system\nTu es un assistant concis.\nuser\nBonjour\nassistant\n"
	tokens := e.Tokenizer.Encode(prompt)
	if len(tokens) < 4 {
		t.Fatalf("prompt tokenized into %d tokens, expected at least 4", len(tokens))
	}

	// Reference: single-shot ingestion of the whole prompt.
	e.KVCache.Reset()
	normRef := e.IngestFrom(0, tokens)
	refLogits := make([]float32, e.Model.VocabSize)
	engine.ComputeLogits(e.Model, e.Arena, normRef, refLogits, nil)

	// Differential: prefix then suffix, no recomputation of the prefix.
	e.KVCache.Reset()
	split := len(tokens) / 2
	e.IngestFrom(0, tokens[:split])
	normChunk := e.IngestFrom(split, tokens[split:])

	if e.KVCache.SeqLen != len(tokens) {
		t.Fatalf("SeqLen = %d after chunked ingestion, expected %d", e.KVCache.SeqLen, len(tokens))
	}
	for i, tok := range tokens {
		if e.KVCache.TokenIDs[i] != tok {
			t.Fatalf("token fingerprint mismatch at %d: got %d want %d", i, e.KVCache.TokenIDs[i], tok)
		}
	}

	gotLogits := make([]float32, e.Model.VocabSize)
	engine.ComputeLogits(e.Model, e.Arena, normChunk, gotLogits, nil)

	for i := range refLogits {
		if diff := math.Abs(float64(refLogits[i] - gotLogits[i])); diff > 1e-5 {
			t.Fatalf("logit %d diverges after differential ingestion: ref=%f chunk=%f diff=%e",
				i, refLogits[i], gotLogits[i], diff)
		}
	}

	// Streaming session must extend the cache without resetting it.
	before := e.KVCache.SeqLen
	delta := e.Tokenizer.Encode("user\nDis bonjour.\nassistant\n")
	_, err = e.GenerateStreamSession(delta, 4, nil, nil)
	if err != nil {
		t.Fatalf("GenerateStreamSession failed: %v", err)
	}
	if e.KVCache.SeqLen <= before {
		t.Fatalf("SeqLen = %d did not grow beyond %d: cache was reset", e.KVCache.SeqLen, before)
	}
	if e.KVCache.SeqLen > e.KVCache.MaxTokens {
		t.Fatalf("SeqLen = %d exceeds MaxTokens = %d", e.KVCache.SeqLen, e.KVCache.MaxTokens)
	}
	t.Logf("session cache grew from %d to %d of %d tokens",
		before, e.KVCache.SeqLen, e.KVCache.MaxTokens)
}
