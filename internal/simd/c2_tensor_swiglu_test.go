package simd

import (
	"math"
	"testing"
)

func TestSwiGLUPrecisionAndAllocs(t *testing.T) {
	n := 6144
	gate := make([]float32, n)
	up := make([]float32, n)
	out := make([]float32, n)
	ref := make([]float32, n)

	// Remplissage avec une plage variée de valeurs
	for i := 0; i < n; i++ {
		gate[i] = float32(i-3072) / 256.0 // [-12, 12]
		up[i] = float32(i%100) / 100.0
		g := float64(gate[i])
		ref[i] = float32(g/(1.0+math.Exp(-g))) * up[i]
	}

	C2_tensor_swiglu(out, gate, up, n)

	var maxDiff float32
	for i := 0; i < n; i++ {
		diff := float32(math.Abs(float64(out[i] - ref[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	t.Logf("SwiGLU max difference vs math.Exp: %e", maxDiff)
	if maxDiff > 1e-4 {
		t.Errorf("max difference too high: %e", maxDiff)
	}

	// Preuve formelle 0 allocation
	allocs := testing.AllocsPerRun(10, func() {
		C2_tensor_swiglu(out, gate, up, n)
	})
	if allocs != 0 {
		t.Errorf("SwiGLU allocated %v objects, want 0", allocs)
	}
}
