package engine

import (
	"syscall"
	"unsafe"
)

// MaxContextLen is the maximum supported sequence length for inference.
// Qwen3-1.7B trains on a 32k window, so 16k remains strictly inside the native
// range and requires no RoPE scaling (YaRN is deliberately not applied).
const MaxContextLen = 16384

// madvHugepage requests 2 MiB transparent huge pages from the kernel.
const madvHugepage = 14

// KVCache holds contiguous Key and Value state for all layers, plus the
// token fingerprint materialized at each position. The fingerprint enables a
// caller to prove that a reused prefix matches the sequence it claims, and to
// resume an interrupted conversation without recomputing the past.
// For Qwen3-1.7B (28 layers, 8 KV heads, headDim 128, 16k tokens) the FP32
// footprint is 2 * 28 * 8 * 16384 * 128 * 4 = 3,758,096,384 bytes (~3.75 GB),
// plus TokenIDs: MaxTokens * 4 bytes. The Key/Value block is backed by an
// anonymous mmap advised MADV_HUGEPAGE so the kernel serves it through 2 MiB
// huge pages, which keeps the TLB pressure bounded during long inference.
type KVCache struct {
	NumLayers  int
	NumKVHeads int
	HeadDim    int
	MaxTokens  int

	// Contiguous buffer: [2][NumLayers][NumKVHeads][MaxTokens][HeadDim]
	Data []float32

	// TokenIDs keeps the token id stored at each materialized position.
	TokenIDs []int32

	// SeqLen is the exact number of tokens already computed across every layer.
	SeqLen int

	// mapped aliases Data when the buffer comes from an anonymous mmap; it is
	// nil for the Go heap fallback.
	mapped []byte
}

// NewKVCache allocates the KV cache. The float32 block is first requested from
// an anonymous mmap advised MADV_HUGEPAGE and locked with Mlock on a
// best-effort basis, so the kernel serves it through 2 MiB transparent huge
// pages; when the mapping fails the allocation falls back to a Go heap slice.
func NewKVCache(numLayers, numKVHeads, headDim, maxTokens int) *KVCache {
	totalElements := 2 * numLayers * numKVHeads * maxTokens * headDim
	data, mapped := allocKVCacheData(totalElements)
	return &KVCache{
		NumLayers:  numLayers,
		NumKVHeads: numKVHeads,
		HeadDim:    headDim,
		MaxTokens:  maxTokens,
		Data:       data,
		TokenIDs:   make([]int32, maxTokens),
		SeqLen:     0,
		mapped:     mapped,
	}
}

// allocKVCacheData returns the KV float32 buffer together with the byte view
// required to release an anonymous mapping with Munmap. A failed mapping yields
// a plain heap slice and a nil byte view.
func allocKVCacheData(totalElements int) ([]float32, []byte) {
	if totalElements <= 0 {
		return nil, nil
	}
	totalBytes := totalElements * 4
	mapped, err := syscall.Mmap(-1, 0, totalBytes,
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		return make([]float32, totalElements), nil
	}
	_ = syscall.Madvise(mapped, madvHugepage)
	_ = syscall.Mlock(mapped)
	return unsafe.Slice((*float32)(unsafe.Pointer(&mapped[0])), totalElements), mapped
}

// Close releases the mmap-backed KV buffer. Calling it more than once is safe.
func (c *KVCache) Close() error {
	if c == nil {
		return nil
	}
	c.Data = nil
	c.SeqLen = 0
	if c.mapped == nil {
		return nil
	}
	mapped := c.mapped
	c.mapped = nil
	return syscall.Munmap(mapped)
}

// Reset resets the active token count to 0 with 0 memory allocation. The
// underlying buffers are kept, so the cached values are merely shadowed and
// will be overwritten by the next ingestion.
func (c *KVCache) Reset() {
	c.SeqLen = 0
}

// Rewind moves the cursor back by n materialized tokens without copying any
// memory. The stale Key/Value entries beyond the new cursor remain in the
// buffer but are unreachable; any later ingestion overwrites them in place.
// The cursor never moves below 0 nor beyond the current sequence length.
func (c *KVCache) Rewind(n int) {
	if n < 0 {
		n = 0
	}
	if n > c.SeqLen {
		n = c.SeqLen
	}
	c.SeqLen -= n
}

// StoreTokenID records the token id materialized at the given position.
func (c *KVCache) StoreTokenID(pos int, tokenID int32) {
	if pos < 0 || pos >= c.MaxTokens {
		return
	}
	c.TokenIDs[pos] = tokenID
}

func (c *KVCache) getOffset(isV bool, layer, kvHead, pos int) int {
	// Strides
	headStride := c.HeadDim
	posStride := c.HeadDim
	headGroupStride := c.MaxTokens * headStride
	layerStride := c.NumKVHeads * headGroupStride
	kvStride := c.NumLayers * layerStride

	base := 0
	if isV {
		base = kvStride
	}
	return base + layer*layerStride + kvHead*headGroupStride + pos*posStride
}

// StoreKV stores computed Key and Value vectors at the given layer and position
func (c *KVCache) StoreKV(layer, pos int, k, v []float32) {
	if pos < 0 || pos >= c.MaxTokens {
		return
	}
	for h := 0; h < c.NumKVHeads; h++ {
		srcK := k[h*c.HeadDim : (h+1)*c.HeadDim]
		srcV := v[h*c.HeadDim : (h+1)*c.HeadDim]

		dstKOffset := c.getOffset(false, layer, h, pos)
		dstVOffset := c.getOffset(true, layer, h, pos)

		copy(c.Data[dstKOffset:dstKOffset+c.HeadDim], srcK)
		copy(c.Data[dstVOffset:dstVOffset+c.HeadDim], srcV)
	}
}

// GetKey returns slice of Key vector for given layer, kvHead, pos
func (c *KVCache) GetKey(layer, kvHead, pos int) []float32 {
	offset := c.getOffset(false, layer, kvHead, pos)
	return c.Data[offset : offset+c.HeadDim]
}

// GetValue returns slice of Value vector for given layer, kvHead, pos
func (c *KVCache) GetValue(layer, kvHead, pos int) []float32 {
	offset := c.getOffset(true, layer, kvHead, pos)
	return c.Data[offset : offset+c.HeadDim]
}
