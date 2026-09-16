package gguf_test

import (
	"os"
	"testing"

	"github.com/hazyhaar/c2slm/gguf"
)

const modelPath = "/data/models/qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"

func TestOpenQwenGGUF(t *testing.T) {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("model file not found at %s", modelPath)
	}

	gf, err := gguf.Open(modelPath)
	if err != nil {
		t.Fatalf("gguf.Open failed: %v", err)
	}
	defer gf.Close()

	if gf.Header.Magic != gguf.GGUFMagic {
		t.Errorf("expected magic 0x%08x, got 0x%08x", gguf.GGUFMagic, gf.Header.Magic)
	}
	if gf.Header.Version != 3 {
		t.Errorf("expected version 3, got %d", gf.Header.Version)
	}
	if gf.Header.TensorCount != 291 {
		t.Errorf("expected 291 tensors, got %d", gf.Header.TensorCount)
	}

	arch, ok := gf.GetString("general.architecture")
	if !ok || arch != "qwen2" {
		t.Errorf("expected architecture qwen2, got %s (ok=%v)", arch, ok)
	}

	layers, ok := gf.GetUint32("qwen2.block_count")
	if !ok || layers != 24 {
		t.Errorf("expected 24 layers, got %d (ok=%v)", layers, ok)
	}

	embd, ok := gf.GetUint32("qwen2.embedding_length")
	if !ok || embd != 896 {
		t.Errorf("expected 896 embedding length, got %d (ok=%v)", embd, ok)
	}

	heads, ok := gf.GetUint32("qwen2.attention.head_count")
	if !ok || heads != 14 {
		t.Errorf("expected 14 heads, got %d (ok=%v)", heads, ok)
	}

	kvHeads, ok := gf.GetUint32("qwen2.attention.head_count_kv")
	if !ok || kvHeads != 2 {
		t.Errorf("expected 2 kv heads, got %d (ok=%v)", kvHeads, ok)
	}

	// Verify key tensors exist and have valid mapped data
	checkTensors := []string{
		"token_embd.weight",
		"blk.0.attn_q.weight",
		"blk.0.attn_q.bias",
		"blk.0.attn_k.weight",
		"blk.0.attn_k.bias",
		"blk.0.attn_v.weight",
		"blk.0.attn_v.bias",
		"blk.0.attn_output.weight",
		"blk.0.attn_norm.weight",
		"blk.0.ffn_gate.weight",
		"blk.0.ffn_up.weight",
		"blk.0.ffn_down.weight",
		"blk.0.ffn_norm.weight",
		"output_norm.weight",
	}

	for _, name := range checkTensors {
		ti, exists := gf.TensorMap[name]
		if !exists {
			t.Errorf("missing tensor %s", name)
			continue
		}
		if len(ti.Data) == 0 {
			t.Errorf("tensor %s has empty data", name)
		}
		t.Logf("tensor %s: type=%s, dims=%v, bytes=%d", name, ti.Type, ti.Dimensions, len(ti.Data))
	}

	if outTi, exists := gf.TensorMap["output.weight"]; exists {
		t.Logf("tensor output.weight exists: type=%s, dims=%v, bytes=%d", outTi.Type, outTi.Dimensions, len(outTi.Data))
	} else {
		t.Logf("tensor output.weight is TIED with token_embd.weight (absent from file)")
	}

	tokModel, _ := gf.GetString("tokenizer.ggml.model")
	tokens, _ := gf.GetStringSlice("tokenizer.ggml.tokens")
	merges, _ := gf.GetStringSlice("tokenizer.ggml.merges")
	t.Logf("tokenizer: model=%s, tokens_count=%d, merges_count=%d", tokModel, len(tokens), len(merges))
}
