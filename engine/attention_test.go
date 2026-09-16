package engine

import (
	"math"
	"testing"

	"github.com/hazyhaar/c2slm/tensor"
)

func fillSyntheticCache(t *testing.T, numHeads, kvHeads, headDim, pos int) (*KVCache, []float32) {
	t.Helper()
	cache := NewKVCache(1, kvHeads, headDim, pos+1)
	state := uint32(24681357)
	next := func() float32 {
		state = state*1664525 + 1013904223
		return float32((state>>8)&0xFFFFFF)/float32(0x1000000)*2 - 1
	}
	k := make([]float32, kvHeads*headDim)
	v := make([]float32, kvHeads*headDim)
	q := make([]float32, numHeads*headDim)
	for i := range q {
		q[i] = next()
	}
	for p := 0; p <= pos; p++ {
		for i := range k {
			k[i] = next()
			v[i] = next()
		}
		cache.StoreKV(0, p, k, v)
	}
	cache.SeqLen = pos + 1
	return cache, q
}

func referenceAttention(cache *KVCache, pos int, q []float32, numHeads, kvHeads, headDim int, scale float32) []float32 {
	qPerKV := numHeads / kvHeads
	out := make([]float32, numHeads*headDim)
	scores := make([]float32, pos+1)
	for h := 0; h < numHeads; h++ {
		kvHead := h / qPerKV
		qHead := q[h*headDim : (h+1)*headDim]
		for p := 0; p <= pos; p++ {
			kVec := cache.GetKey(0, kvHead, p)
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += qHead[d] * kVec[d]
			}
			scores[p] = dot * scale
		}
		tensor.Softmax(scores)
		outHead := out[h*headDim : (h+1)*headDim]
		for d := 0; d < headDim; d++ {
			var val float32
			for p := 0; p <= pos; p++ {
				vVec := cache.GetValue(0, kvHead, p)
				val += scores[p] * vVec[d]
			}
			outHead[d] = val
		}
	}
	return out
}

func TestParallelAttentionMatchesReference(t *testing.T) {
	const numHeads, kvHeads, headDim = 14, 2, 64
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	for _, pos := range []int{100, 300} {
		cache, q := fillSyntheticCache(t, numHeads, kvHeads, headDim, pos)
		got := make([]float32, numHeads*headDim)
		scratch := make([]float32, MaxContextLen)

		pool := NewWorkerPool()
		DispatchAttention(pool, scratch, cache, 0, pos, q, got, numHeads, kvHeads, headDim, scale)
		pool.Close()

		want := referenceAttention(cache, pos, q, numHeads, kvHeads, headDim, scale)

		maxDiff := 0.0
		for i := range got {
			ref := math.Abs(float64(want[i]))
			if ref < 1 {
				ref = 1
			}
			d := math.Abs(float64(got[i]-want[i])) / ref
			if d > maxDiff {
				maxDiff = d
			}
		}
		if maxDiff > 1e-5 {
			t.Fatalf("pos=%d rel diff %e > 1e-5", pos, maxDiff)
		}
		t.Logf("pos=%d parallel attention rel diff=%e (workers=%d)", pos, maxDiff, pool.NumWorkers())
	}
}

func TestParallelAttentionZeroAllocation(t *testing.T) {
	const numHeads, kvHeads, headDim = 14, 2, 64
	pos := 512
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	cache, q := fillSyntheticCache(t, numHeads, kvHeads, headDim, pos)
	out := make([]float32, numHeads*headDim)
	scratch := make([]float32, MaxContextLen)

	pool := NewWorkerPool()
	defer pool.Close()
	DispatchAttention(pool, scratch, cache, 0, pos, q, out, numHeads, kvHeads, headDim, scale)

	allocs := testing.AllocsPerRun(10, func() {
		DispatchAttention(pool, scratch, cache, 0, pos, q, out, numHeads, kvHeads, headDim, scale)
	})
	if allocs != 0 {
		t.Fatalf("DispatchAttention allocated %v objects, want 0", allocs)
	}
}
