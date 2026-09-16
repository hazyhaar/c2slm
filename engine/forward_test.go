package engine_test

import (
	"math"
	"os"
	"testing"
	"time"

	"github.com/hazyhaar/c2slm/engine"
)

const modelPath = "/data/models/qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"

func TestForwardZeroAllocation(t *testing.T) {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("model file not found at %s", modelPath)
	}

	m, err := engine.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}
	defer m.Close()

	cache := engine.NewKVCache(m.NumLayers, m.NumKVHeads, m.HeadDim, engine.MaxContextLen)
	arena := engine.NewArena(m)

	defer arena.Close()

	// Warm up
	engine.Forward(m, cache, arena, 151644, 0)

	// Measure heap allocations per forward step
	allocs := testing.AllocsPerRun(10, func() {
		engine.Forward(m, cache, arena, 151644, 1)
	})

	if allocs > 0 {
		t.Errorf("Forward had %f allocations per run, expected 0 B/op", allocs)
	} else {
		t.Logf("Forward validated: EXACTLY 0.0 allocations per run (0 B/op)!")
	}
}

func TestComputeLogitsZeroAllocation(t *testing.T) {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("model file not found at %s", modelPath)
	}

	m, err := engine.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}
	defer m.Close()

	cache := engine.NewKVCache(m.NumLayers, m.NumKVHeads, m.HeadDim, engine.MaxContextLen)
	arena := engine.NewArena(m)
	defer arena.Close()

	normX := engine.Forward(m, cache, arena, 151644, 0)

	// Warm up ComputeLogits
	engine.ComputeLogits(m, arena, normX, arena.Logits, nil)

	// Measure heap allocations per ComputeLogits call
	allocs := testing.AllocsPerRun(10, func() {
		engine.ComputeLogits(m, arena, normX, arena.Logits, nil)
	})

	if allocs > 0 {
		t.Errorf("ComputeLogits had %f allocations per run, expected 0 B/op", allocs)
	} else {
		t.Logf("ComputeLogits validated: EXACTLY 0.0 allocations per run (0 B/op) across %d workers!", arena.Pool.NumWorkers())
	}
}

func BenchmarkForward(b *testing.B) {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		b.Skipf("model file not found at %s", modelPath)
	}

	m, err := engine.LoadModel(modelPath)
	if err != nil {
		b.Fatalf("LoadModel failed: %v", err)
	}
	defer m.Close()

	cache := engine.NewKVCache(m.NumLayers, m.NumKVHeads, m.HeadDim, engine.MaxContextLen)
	arena := engine.NewArena(m)
	defer arena.Close()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.Forward(m, cache, arena, 151644, 1)
	}
}

func BenchmarkComputeLogits(b *testing.B) {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		b.Skipf("model file not found at %s", modelPath)
	}

	m, err := engine.LoadModel(modelPath)
	if err != nil {
		b.Fatalf("LoadModel failed: %v", err)
	}
	defer m.Close()

	cache := engine.NewKVCache(m.NumLayers, m.NumKVHeads, m.HeadDim, engine.MaxContextLen)
	arena := engine.NewArena(m)
	defer arena.Close()

	normX := engine.Forward(m, cache, arena, 151644, 0)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		engine.ComputeLogits(m, arena, normX, arena.Logits, nil)
	}
}

func TestComputeLogitsRestrictedARCHTIME(t *testing.T) {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("model file not found at %s", modelPath)
	}

	m, err := engine.LoadModel(modelPath)
	if err != nil {
		t.Fatalf("LoadModel failed: %v", err)
	}
	defer m.Close()

	cache := engine.NewKVCache(m.NumLayers, m.NumKVHeads, m.HeadDim, engine.MaxContextLen)
	arena := engine.NewArena(m)
	defer arena.Close()

	normX := engine.Forward(m, cache, arena, 151644, 0)

	// Calcule d'abord la projection complète pour référence
	fullLogits := make([]float32, m.VocabSize)
	engine.ComputeLogits(m, arena, normX, fullLogits, nil)

	// Liste de jetons d'arbitrage (true, false, null, accolades, chiffres)
	targetTokens := []int32{151644, 151645, 100, 200, 300, 1000, 2000, 5000, 10000}
	restrictedLogits := make([]float32, m.VocabSize)

	// Vérification de la parité numérique exacte
	engine.ComputeLogits(m, arena, normX, restrictedLogits, targetTokens)

	for _, tok := range targetTokens {
		ref := fullLogits[tok]
		got := restrictedLogits[tok]
		diff := math.Abs(float64(got - ref))
		if diff > 1e-5 {
			t.Fatalf("ARCHTIME projection mismatch for token %d: full=%f restricted=%f diff=%e", tok, ref, got, diff)
		}
	}

	// Mesure du temps d'exécution
	start := time.Now()
	const runs = 100
	for i := 0; i < runs; i++ {
		engine.ComputeLogits(m, arena, normX, restrictedLogits, targetTokens)
	}
	elapsed := time.Since(start)
	avgDuration := elapsed / runs

	t.Logf("ARCHTIME restricted projection (%d tokens): avg latency = %v per call (budget < 0.5ms)", len(targetTokens), avgDuration)
	if avgDuration > 500*time.Microsecond {
		t.Errorf("Latency exceeded 0.5ms: %v", avgDuration)
	}

	// Preuve formelle 0 allocation heap
	allocs := testing.AllocsPerRun(10, func() {
		engine.ComputeLogits(m, arena, normX, restrictedLogits, targetTokens)
	})

	if allocs > 0 {
		t.Errorf("ComputeLogits ARCHTIME had %f allocations per run, expected 0 B/op", allocs)
	} else {
		t.Logf("ARCHTIME validated: EXACTLY 0.0 allocations per run (0 B/op)!")
	}
}
