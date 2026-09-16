package engine

// MaxContextLen is the maximum supported sequence length for inference
const MaxContextLen = 2048

// KVCache holds contiguous Key and Value state for all layers
// Size: 2 * 24 layers * 2 kvHeads * 512 tokens * 64 headDim * 4 bytes = 12,582,912 bytes
type KVCache struct {
	NumLayers  int
	NumKVHeads int
	HeadDim    int
	MaxTokens  int

	// Contiguous buffer: [2][NumLayers][NumKVHeads][MaxTokens][HeadDim]
	Data   []float32
	SeqLen int // Current sequence length
}

// NewKVCache creates a pre-allocated 12.58 MB KV Cache
func NewKVCache(numLayers, numKVHeads, headDim, maxTokens int) *KVCache {
	totalElements := 2 * numLayers * numKVHeads * maxTokens * headDim
	return &KVCache{
		NumLayers:  numLayers,
		NumKVHeads: numKVHeads,
		HeadDim:    headDim,
		MaxTokens:  maxTokens,
		Data:       make([]float32, totalElements),
		SeqLen:     0,
	}
}

// Reset resets the active token count to 0 with 0 memory allocation
func (c *KVCache) Reset() {
	c.SeqLen = 0
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
	if pos >= c.MaxTokens {
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
