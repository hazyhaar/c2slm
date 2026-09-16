package simd

import (
	"math"
	"testing"
)

func TestTensorGEMV2_Q6_K_Q8_K_CorrectnessAndAllocs(t *testing.T) {
	cols := 512
	numBlocks := cols / 256

	weight0 := make([]byte, numBlocks*210)
	weight1 := make([]byte, numBlocks*210)
	x := make([]float32, cols)
	q8k := make([]byte, numBlocks*292)

	for b := 0; b < numBlocks; b++ {
		blk0 := weight0[b*210 : (b+1)*210]
		blk0[208] = 0x00
		blk0[209] = 0x3C // d = 1.0
		for i := 0; i < 128; i++ {
			blk0[i] = byte((b*37 + i*19 + 17) & 0xFF)
		}
		for i := 0; i < 64; i++ {
			blk0[128+i] = byte((b*13 + i*7 + 3) & 0xFF)
		}
		for i := 0; i < 16; i++ {
			blk0[192+i] = byte((i*3 - 8) & 0xFF)
		}

		blk1 := weight1[b*210 : (b+1)*210]
		blk1[208] = 0x00
		blk1[209] = 0x38 // d = 0.5
		for i := 0; i < 128; i++ {
			blk1[i] = byte((b*41 + i*23 + 19) & 0xFF)
		}
		for i := 0; i < 64; i++ {
			blk1[128+i] = byte((b*17 + i*11 + 5) & 0xFF)
		}
		for i := 0; i < 16; i++ {
			blk1[192+i] = byte((i*2 - 7) & 0xFF)
		}
	}

	for i := 0; i < cols; i++ {
		x[i] = float32((i%17)-8) * 0.125
	}

	C2_quantize_row_q8_k(x, q8k, cols)

	var got0, got1 float32
	C2_tensor_gemv2_q6_k_q8_k(weight0, weight1, q8k, cols, &got0, &got1)

	// Comparaison contre les valeurs de référence de l'oracle GCC -O2
	want0 := float32(-8705.69238281)
	want1 := float32(3759.21655273)

	diff0 := float32(math.Abs(float64(got0 - want0)))
	diff1 := float32(math.Abs(float64(got1 - want1)))

	t.Logf("GEMV2 Q6_K x Q8_K:\n  Row 0: got=%.6f want=%.6f diff=%.2e\n  Row 1: got=%.6f want=%.6f diff=%.2e",
		got0, want0, diff0, got1, want1, diff1)

	if diff0 > 1e-2 {
		t.Errorf("Row 0 diff too large: %e", diff0)
	}
	if diff1 > 1e-2 {
		t.Errorf("Row 1 diff too large: %e", diff1)
	}

	// Preuve formelle 0 allocation
	allocs := testing.AllocsPerRun(10, func() {
		C2_tensor_gemv2_q6_k_q8_k(weight0, weight1, q8k, cols, &got0, &got1)
	})
	if allocs != 0 {
		t.Errorf("AllocsPerRun = %v, want 0", allocs)
	}
}
