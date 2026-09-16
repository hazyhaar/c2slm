package engine

import (
	"os"
	"testing"
)

const qwen31_7BModelPath = "/data/models/qwen3-1.7b-gguf/Qwen_Qwen3-1.7B-Q4_K_M.gguf"

func TestLoadModelQwen3_1_7BConfig(t *testing.T) {
	if _, err := os.Stat(qwen31_7BModelPath); err != nil {
		t.Skipf("model not found: %v", err)
	}

	m, err := LoadModel(qwen31_7BModelPath)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	if m.NumLayers != 28 || m.NumKVHeads != 8 || m.HeadDim != 128 {
		t.Fatalf("dims = layers %d, kvHeads %d, headDim %d; want 28/8/128",
			m.NumLayers, m.NumKVHeads, m.HeadDim)
	}
	if m.NumHeads != 16 {
		t.Fatalf("NumHeads = %d, want 16", m.NumHeads)
	}
	if m.RoPEBase != defaultRoPEBaseQwen3 {
		t.Fatalf("RoPEBase = %v, want %v", m.RoPEBase, defaultRoPEBaseQwen3)
	}
	if m.RoPETable == nil || m.RoPETable.MaxLen != MaxContextLen {
		t.Fatalf("RoPE table is not sized to %d positions", MaxContextLen)
	}
}
