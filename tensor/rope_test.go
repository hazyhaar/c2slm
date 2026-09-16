package tensor_test

import (
	"math"
	"testing"

	"github.com/hazyhaar/c2slm/tensor"
)

func TestRoPETable16kSizing(t *testing.T) {
	const maxLen, headDim = 16384, 128
	tbl := tensor.NewRoPETable(maxLen, headDim, 1_000_000.0)
	if tbl.MaxLen != maxLen {
		t.Fatalf("MaxLen = %d, want %d", tbl.MaxLen, maxLen)
	}
	if tbl.HalfDim != headDim/2 {
		t.Fatalf("HalfDim = %d, want %d", tbl.HalfDim, headDim/2)
	}
	if len(tbl.Cos) != maxLen*(headDim/2) || len(tbl.Sin) != maxLen*(headDim/2) {
		t.Fatalf("table length = (%d, %d), want %d", len(tbl.Cos), len(tbl.Sin), maxLen*(headDim/2))
	}
}

func TestRoPETableApplyOutOfRangeGuard(t *testing.T) {
	const maxLen, headDim = 16, 8
	tbl := tensor.NewRoPETable(maxLen, headDim, 1_000_000.0)

	vec := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	orig := make([]float32, len(vec))
	copy(orig, vec)

	// Positions outside [0, MaxLen) must be rejected without mutating vec.
	tbl.Apply(vec, 1, maxLen)
	tbl.Apply(vec, 1, maxLen+1000)
	tbl.Apply(vec, 1, -1)
	for i := range vec {
		if vec[i] != orig[i] {
			t.Fatalf("out-of-range pos mutated vec[%d]: got %f, want %f", i, vec[i], orig[i])
		}
	}

	// A head count that overflows vec must also be rejected.
	tbl.Apply(vec, 2, 1)
	for i := range vec {
		if vec[i] != orig[i] {
			t.Fatalf("oversized head count mutated vec[%d]: got %f, want %f", i, vec[i], orig[i])
		}
	}
}

func TestRoPETableApplyInRangeRotates(t *testing.T) {
	const headDim = 8
	tbl := tensor.NewRoPETable(4, headDim, 1_000_000.0)

	vec := []float32{1, 0, 0, 0, 0, 0, 0, 0}
	tbl.Apply(vec, 1, 1)

	// pos=1 rotates the pair (x0, x1) by a non-zero angle: the second half must
	// become sin(angle) and the first half cos(angle), both strictly in (0,1).
	if !(vec[0] > 0 && vec[0] < 1) {
		t.Fatalf("cosine component = %f, want in (0,1)", vec[0])
	}
	if math.Abs(float64(vec[4])) < 1e-9 {
		t.Fatalf("sine component = %f, want a non-zero rotation", vec[4])
	}
}
