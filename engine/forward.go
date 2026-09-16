package engine

import (
	"math"

	"github.com/hazyhaar/c2slm/gguf"
	"github.com/hazyhaar/c2slm/internal/simd"
	"github.com/hazyhaar/c2slm/tensor"
)

// Arena holds pre-allocated activation buffers to guarantee 0 heap allocation during forward pass
type Arena struct {
	X          []float32 // [896]
	NormX      []float32 // [896]
	Q          []float32 // [896]
	K          []float32 // [128]
	V          []float32 // [128]
	AttnScores []float32 // [512]
	AttnOut    []float32 // [896]
	Gate       []float32 // [4864]
	Up         []float32 // [4864]
	Down       []float32 // [896]
	Logits     []float32 // [151936]
	Pool       *WorkerPool
}

// NewArena allocates static activation workspace
func NewArena(m *Model) *Arena {
	return &Arena{
		X:          make([]float32, m.EmbdLength),
		NormX:      make([]float32, m.EmbdLength),
		Q:          make([]float32, m.EmbdLength),
		K:          make([]float32, m.NumKVHeads*m.HeadDim),
		V:          make([]float32, m.NumKVHeads*m.HeadDim),
		AttnScores: make([]float32, MaxContextLen),
		AttnOut:    make([]float32, m.EmbdLength),
		Gate:       make([]float32, m.FFNDim),
		Up:         make([]float32, m.FFNDim),
		Down:       make([]float32, m.EmbdLength),
		Logits:     make([]float32, m.VocabSize),
		Pool:       NewWorkerPool(),
	}
}

// Close gracefully terminates persistent background workers
func (a *Arena) Close() {
	if a != nil && a.Pool != nil {
		a.Pool.Close()
	}
}

// Forward computes one Transformer decoder step for the given token at position pos.
// It writes normalized hidden states into arena.NormX and returns it.
func Forward(m *Model, cache *KVCache, a *Arena, tokenID int32, pos int) []float32 {
	// 1. Embed input token
	m.EmbedToken(a.X, tokenID)

	headDim := m.HeadDim
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	kvHeads := m.NumKVHeads

	// 2. Loop through all 24 Transformer layers
	for lIdx := 0; lIdx < m.NumLayers; lIdx++ {
		layer := &m.Layers[lIdx]

		// Pre-attention RMSNorm
		tensor.RMSNorm(a.NormX, a.X, layer.AttnNorm, m.RMSNormEps)

		// Q, K, V Projections
		DispatchGEMV(a.Pool, a.Q, layer.AttnQ, a.NormX, m.EmbdLength, m.EmbdLength)
		tensor.AddBias(a.Q, layer.AttnQBias)

		DispatchGEMV(a.Pool, a.K, layer.AttnK, a.NormX, kvHeads*headDim, m.EmbdLength)
		tensor.AddBias(a.K, layer.AttnKBias)

		DispatchGEMV(a.Pool, a.V, layer.AttnV, a.NormX, kvHeads*headDim, m.EmbdLength)
		tensor.AddBias(a.V, layer.AttnVBias)

		// Per-head QK-Norm (Qwen3)
		if len(layer.AttnQNorm) > 0 {
			for h := 0; h < m.NumHeads; h++ {
				qHead := a.Q[h*headDim : (h+1)*headDim]
				tensor.RMSNorm(qHead, qHead, layer.AttnQNorm, m.RMSNormEps)
			}
		}
		if len(layer.AttnKNorm) > 0 {
			for h := 0; h < kvHeads; h++ {
				kHead := a.K[h*headDim : (h+1)*headDim]
				tensor.RMSNorm(kHead, kHead, layer.AttnKNorm, m.RMSNormEps)
			}
		}

		// RoPE NeoX with precomputed ARCHTIME cache (zero transcendental calls)
		if m.RoPETable != nil {
			m.RoPETable.Apply(a.Q, m.NumHeads, pos)
			m.RoPETable.Apply(a.K, kvHeads, pos)
		} else {
			tensor.RoPENeoX(a.Q, m.NumHeads, headDim, pos, m.RoPEBase)
			tensor.RoPENeoX(a.K, kvHeads, headDim, pos, m.RoPEBase)
		}

		// Store K, V in KV-Cache
		cache.StoreKV(lIdx, pos, a.K, a.V)

		// Multi-Head Attention with Grouped-Query Attention (14:2)
		DispatchAttention(a.Pool, a.AttnScores, cache, lIdx, pos, a.Q, a.AttnOut, m.NumHeads, kvHeads, headDim, scale)

		// Output projection W_o
		DispatchGEMV(a.Pool, a.Down, layer.AttnOut, a.AttnOut, m.EmbdLength, m.EmbdLength)

		// Residual connection
		for i := 0; i < m.EmbdLength; i++ {
			a.X[i] += a.Down[i]
		}

		// Pre-FFN RMSNorm
		tensor.RMSNorm(a.NormX, a.X, layer.FFNNorm, m.RMSNormEps)

		// FFN SwiGLU: gate = W_gate * normX, up = W_up * normX
		DispatchGEMV(a.Pool, a.Gate, layer.FFNGate, a.NormX, m.FFNDim, m.EmbdLength)
		DispatchGEMV(a.Pool, a.Up, layer.FFNUp, a.NormX, m.FFNDim, m.EmbdLength)

		tensor.SwiGLU(a.Gate, a.Gate, a.Up)

		// Down projection: down = W_down * hidden
		DispatchGEMV(a.Pool, a.Down, layer.FFNDown, a.Gate, m.EmbdLength, m.FFNDim)

		// Residual connection
		for i := 0; i < m.EmbdLength; i++ {
			a.X[i] += a.Down[i]
		}
	}

	// Final RMSNorm
	tensor.RMSNorm(a.NormX, a.X, m.OutputNorm, m.RMSNormEps)

	return a.NormX
}

// attentionHead computes one query head against its Grouped-Query KV head:
// scaled Q@K scores, softmax, then the weighted sum Attn@V. Both products are
// delegated to the AVX2 micro-kernels; scores is caller-owned scratch.
func attentionHead(cache *KVCache, layer, pos int, q, out []float32, h, qPerKV, headDim int, scale float32, scores []float32) {
	kvHead := h / qPerKV
	qHead := q[h*headDim : (h+1)*headDim]

	kOff := cache.getOffset(false, layer, kvHead, 0)
	vOff := cache.getOffset(true, layer, kvHead, 0)

	simd.C2_tensor_attn_qk(qHead, cache.Data[kOff:], pos, headDim, scale, scores, headDim)
	tensor.Softmax(scores[:pos+1])

	outHead := out[h*headDim : (h+1)*headDim]
	simd.C2_tensor_attn_av(scores, cache.Data[vOff:], pos, headDim, outHead, headDim)
}

// DispatchAttention runs attention with the pool when available, falling back
// to a sequential head loop on the caller-owned scratch otherwise.
func DispatchAttention(pool *WorkerPool, scratch []float32, cache *KVCache, layer, pos int, q, out []float32, numHeads, kvHeads, headDim int, scale float32) {
	if pool != nil {
		pool.ParallelAttention(scratch, cache, layer, pos, q, out, numHeads, kvHeads, headDim, scale)
		return
	}
	qPerKV := numHeads / kvHeads
	for h := 0; h < numHeads; h++ {
		attentionHead(cache, layer, pos, q, out, h, qPerKV, headDim, scale, scratch[:pos+1])
	}
}

// ComputeLogits projects final hidden states to vocabulary logits (or restricted tokens)
func ComputeLogits(m *Model, a *Arena, normX []float32, outLogits []float32, targetTokens []int32) {
	cols := m.EmbdLength

	if len(targetTokens) > 0 {
		// ARCHTIME restricted projection: only compute for requested tokens
		for _, tokID := range targetTokens {
			if int(tokID) >= m.VocabSize || tokID < 0 {
				continue
			}
			outLogits[tokID] = computeSingleTokenLogit(m, normX, int(tokID), cols)
		}
		return
	}

	// Full vocabulary projection parallelized over worker pool
	var pool *WorkerPool
	if a != nil {
		pool = a.Pool
	}
	DispatchGEMV(pool, outLogits, m.OutputWeight, normX, m.VocabSize, cols)
}

func computeSingleTokenLogit(m *Model, normX []float32, tokID, cols int) float32 {
	switch m.OutputWeight.Type {
	case gguf.GGMLTypeQ8_0:
		rowBytes := (cols / tensor.QK8_0) * tensor.BlockSizeQ8_0
		rowOffset := tokID * rowBytes
		return tensor.DotQ8_0(m.OutputWeight.Data[rowOffset:rowOffset+rowBytes], normX, cols)
	case gguf.GGMLTypeQ5_0:
		rowBytes := (cols / tensor.QK5_0) * tensor.BlockSizeQ5_0
		rowOffset := tokID * rowBytes
		return tensor.DotQ5_0(m.OutputWeight.Data[rowOffset:rowOffset+rowBytes], normX, cols)
	case gguf.GGMLTypeQ6_K:
		rowBytes := (cols / tensor.QK6_K) * tensor.BlockSizeQ6_K
		rowOffset := tokID * rowBytes
		return tensor.DotQ6_K(m.OutputWeight.Data[rowOffset:rowOffset+rowBytes], normX, cols)
	case gguf.GGMLTypeQ4_K:
		rowBytes := (cols / tensor.QK4_K) * tensor.BlockSizeQ4_K
		rowOffset := tokID * rowBytes
		return tensor.DotQ4_K(m.OutputWeight.Data[rowOffset:rowOffset+rowBytes], normX, cols)
	}
	return 0
}
