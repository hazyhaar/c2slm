package tensor_test

import (
	"math"
	"testing"

	"github.com/hazyhaar/c2slm/tensor"
)

func TestFP16ToF32(t *testing.T) {
	cases := []struct {
		in   uint16
		want float32
	}{
		{0x0000, 0.0},
		{0x3C00, 1.0},
		{0xBC00, -1.0},
		{0x4000, 2.0},
		{0x3555, 0.33325195},
	}

	for _, tc := range cases {
		got := tensor.FP16ToF32(tc.in)
		if math.Abs(float64(got-tc.want)) > 1e-4 {
			t.Errorf("FP16(0x%04x): got %f, want %f", tc.in, got, tc.want)
		}
	}
}

func TestRMSNorm(t *testing.T) {
	x := []float32{1.0, 2.0, 3.0, 4.0}
	weight := []float32{1.0, 1.0, 1.0, 1.0}
	out := make([]float32, 4)

	tensor.RMSNorm(out, x, weight, 1e-6)

	// mean(x^2) = (1 + 4 + 9 + 16)/4 = 30/4 = 7.5
	// rms = sqrt(7.5) = 2.738612787
	expected0 := float32(1.0 / math.Sqrt(7.5))
	if math.Abs(float64(out[0]-expected0)) > 1e-5 {
		t.Errorf("RMSNorm[0]: got %f, want %f", out[0], expected0)
	}
}

func TestRoPENeoX(t *testing.T) {
	headDim := 64
	numHeads := 1
	vec := make([]float32, headDim)
	for i := range vec {
		vec[i] = 1.0
	}

	// At pos=0, angle is 0, cos=1, sin=0, so vector should remain identical
	tensor.RoPENeoX(vec, numHeads, headDim, 0, 1000000.0)

	for i := range vec {
		if math.Abs(float64(vec[i]-1.0)) > 1e-6 {
			t.Errorf("RoPE at pos=0 altered vector at %d: got %f", i, vec[i])
		}
	}
}

func TestSoftmax(t *testing.T) {
	x := []float32{1.0, 2.0, 3.0}
	tensor.Softmax(x)

	var sum float32
	for _, v := range x {
		sum += v
	}
	if math.Abs(float64(sum-1.0)) > 1e-6 {
		t.Errorf("Softmax sum != 1.0: got %f", sum)
	}

	// Should be strictly increasing
	if !(x[0] < x[1] && x[1] < x[2]) {
		t.Errorf("Softmax order incorrect: %v", x)
	}
}

func TestGEMVKernels_ZeroAllocation(t *testing.T) {
	cols := 256
	rows := 4

	// Q5_0
	rowBytesQ5_0 := (cols / tensor.QK5_0) * tensor.BlockSizeQ5_0
	wQ5_0 := make([]byte, rows*rowBytesQ5_0)
	x := make([]float32, cols)
	y := make([]float32, rows)

	allocsQ5_0 := testing.AllocsPerRun(20, func() {
		tensor.GEMVQ5_0(y, wQ5_0, x, rows, cols)
	})
	if allocsQ5_0 != 0 {
		t.Fatalf("GEMVQ5_0 allocated %f objects per run, want 0", allocsQ5_0)
	}

	// Q8_0
	rowBytesQ8_0 := (cols / tensor.QK8_0) * tensor.BlockSizeQ8_0
	wQ8_0 := make([]byte, rows*rowBytesQ8_0)
	allocsQ8_0 := testing.AllocsPerRun(20, func() {
		tensor.GEMVQ8_0(y, wQ8_0, x, rows, cols)
	})
	if allocsQ8_0 != 0 {
		t.Fatalf("GEMVQ8_0 allocated %f objects per run, want 0", allocsQ8_0)
	}

	// Q4_K
	rowBytesQ4_K := (cols / tensor.QK4_K) * tensor.BlockSizeQ4_K
	wQ4_K := make([]byte, rows*rowBytesQ4_K)
	allocsQ4_K := testing.AllocsPerRun(20, func() {
		tensor.GEMVQ4_K(y, wQ4_K, x, rows, cols)
	})
	if allocsQ4_K != 0 {
		t.Fatalf("GEMVQ4_K allocated %f objects per run, want 0", allocsQ4_K)
	}

	// Q6_K
	rowBytesQ6_K := (cols / tensor.QK6_K) * tensor.BlockSizeQ6_K
	wQ6_K := make([]byte, rows*rowBytesQ6_K)
	allocsQ6_K := testing.AllocsPerRun(20, func() {
		tensor.GEMVQ6_K(y, wQ6_K, x, rows, cols)
	})
	if allocsQ6_K != 0 {
		t.Fatalf("GEMVQ6_K allocated %f objects per run, want 0", allocsQ6_K)
	}

	// Q4_K_Q8_K (with GEMV2 dual-row fused)
	rowBytesQ4_K_Q8 := (cols / tensor.QK4_K) * tensor.BlockSizeQ4_K
	wQ4_K_Q8 := make([]byte, rows*rowBytesQ4_K_Q8)
	q8k := make([]byte, (cols/256)*tensor.BlockSizeQ8_K)
	tensor.QuantizeRowQ8_K(x, q8k, cols)
	allocsQ4_K_Q8 := testing.AllocsPerRun(20, func() {
		tensor.GEMVQ4_K_Q8_K(y, wQ4_K_Q8, q8k, rows, cols)
	})
	if allocsQ4_K_Q8 != 0 {
		t.Fatalf("GEMVQ4_K_Q8_K allocated %f objects per run, want 0", allocsQ4_K_Q8)
	}

	t.Logf("All GEMV kernels (Q5_0, Q8_0, Q4_K, Q6_K, Q4_K_Q8_K) validated at EXACTLY 0 alloc/op!")
}

func BenchmarkGEMVQ4_K_Q8_K_Cols896(b *testing.B) {
	const cols = 896
	const rows = 128
	w := make([]byte, rows*(cols/tensor.QK4_K)*tensor.BlockSizeQ4_K)
	x := make([]float32, cols)
	q8k := make([]byte, (cols/256)*tensor.BlockSizeQ8_K)
	y := make([]float32, rows)
	tensor.QuantizeRowQ8_K(x, q8k, cols)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tensor.GEMVQ4_K_Q8_K(y, w, q8k, rows, cols)
	}
}

func BenchmarkGEMVQ4_K_Q8_K_Cols2048(b *testing.B) {
	const cols = 2048
	const rows = 128
	w := make([]byte, rows*(cols/tensor.QK4_K)*tensor.BlockSizeQ4_K)
	x := make([]float32, cols)
	q8k := make([]byte, (cols/256)*tensor.BlockSizeQ8_K)
	y := make([]float32, rows)
	tensor.QuantizeRowQ8_K(x, q8k, cols)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tensor.GEMVQ4_K_Q8_K(y, w, q8k, rows, cols)
	}
}
