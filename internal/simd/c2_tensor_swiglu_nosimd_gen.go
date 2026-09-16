//go:build !goexperiment.simd || !amd64

package simd

import "math"

// C2_tensor_swiglu computes out[i] = SiLU(gate[i]) * up[i] in pure Go scalar.
func C2_tensor_swiglu(out, gate, up []float32, n int) {
	C2_tensor_swiglu_nosimd(out, gate, up, n)
}

// C2_tensor_swiglu_nosimd computes out[i] = SiLU(gate[i]) * up[i] in pure Go scalar.
func C2_tensor_swiglu_nosimd(out, gate, up []float32, n int) {
	for i := 0; i < n; i++ {
		g := float64(gate[i])
		silu := float32(g / (1.0 + math.Exp(-g)))
		out[i] = silu * up[i]
	}
}
