package engine

import "testing"

func TestMaxContextLenIs16k(t *testing.T) {
	if MaxContextLen != 16384 {
		t.Fatalf("MaxContextLen = %d, want 16384", MaxContextLen)
	}
}

func TestKVCacheQwen3Size(t *testing.T) {
	c := NewKVCache(28, 8, 128, MaxContextLen)
	defer c.Close()

	const wantElements = 2 * 28 * 8 * 16384 * 128
	if len(c.Data) != wantElements {
		t.Fatalf("Data elements = %d, want %d", len(c.Data), wantElements)
	}
	if len(c.Data)*4 != 3_758_096_384 {
		t.Fatalf("FP32 footprint = %d bytes, want 3758096384", len(c.Data)*4)
	}
	if len(c.TokenIDs) != MaxContextLen {
		t.Fatalf("TokenIDs = %d, want %d", len(c.TokenIDs), MaxContextLen)
	}
	if c.mapped == nil {
		t.Log("KV cache served from the Go heap fallback (mmap unavailable)")
	} else {
		if len(c.mapped) != wantElements*4 {
			t.Fatalf("mapped bytes = %d, want %d", len(c.mapped), wantElements*4)
		}
		t.Logf("KV cache served from anonymous mmap (%d bytes, MADV_HUGEPAGE)", len(c.mapped))
	}
}

func TestKVCacheStoreGetRoundTrip(t *testing.T) {
	const layers, kvHeads, headDim, maxTokens = 2, 2, 4, 8
	c := NewKVCache(layers, kvHeads, headDim, maxTokens)
	defer c.Close()

	k := make([]float32, kvHeads*headDim)
	v := make([]float32, kvHeads*headDim)
	for i := range k {
		k[i] = float32(i) + 0.5
		v[i] = -2*float32(i) - 1
	}

	const layer, pos = 1, 3
	c.StoreKV(layer, pos, k, v)

	for h := 0; h < kvHeads; h++ {
		gotK := c.GetKey(layer, h, pos)
		gotV := c.GetValue(layer, h, pos)
		for i := 0; i < headDim; i++ {
			if gotK[i] != k[h*headDim+i] {
				t.Fatalf("GetKey[layer=%d h=%d i=%d] = %f, want %f", layer, h, i, gotK[i], k[h*headDim+i])
			}
			if gotV[i] != v[h*headDim+i] {
				t.Fatalf("GetValue[layer=%d h=%d i=%d] = %f, want %f", layer, h, i, gotV[i], v[h*headDim+i])
			}
		}
	}

	// Out-of-range stores must be inert, including a negative position.
	c.StoreKV(layer, -1, k, v)
	c.StoreKV(layer, maxTokens, k, v)
}

func TestKVCacheCloseIdempotentAndReleases(t *testing.T) {
	c := NewKVCache(2, 2, 4, 8)
	if c.mapped == nil {
		t.Skip("no mmap backing on this host; nothing to release")
	}
	if err := c.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if c.mapped != nil || c.Data != nil {
		t.Fatalf("Close did not clear the mapping: mapped=%v data=%v", c.mapped != nil, c.Data != nil)
	}
	if err := c.Close(); err != nil {
		t.Fatalf("second Close must be a no-op, got: %v", err)
	}
}
