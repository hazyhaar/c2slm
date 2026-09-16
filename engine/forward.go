package engine

import (
	"math"

	"github.com/hazyhaar/c2slm/gguf"
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
	qPerKV := m.NumHeads / kvHeads // 14 / 2 = 7

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

		// RoPE NeoX (rotate_half)
		tensor.RoPENeoX(a.Q, m.NumHeads, headDim, pos, m.RoPEBase)
		tensor.RoPENeoX(a.K, kvHeads, headDim, pos, m.RoPEBase)

		// Store K, V in KV-Cache
		cache.StoreKV(lIdx, pos, a.K, a.V)

		// Multi-Head Attention with Grouped-Query Attention (14:2)
		for h := 0; h < m.NumHeads; h++ {
			kvHead := h / qPerKV
			qHead := a.Q[h*headDim : (h+1)*headDim]

			// Compute dot product scores for all past and current tokens
			for p := 0; p <= pos; p++ {
				kVec := cache.GetKey(lIdx, kvHead, p)
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += qHead[d] * kVec[d]
				}
				a.AttnScores[p] = dot * scale
			}

			// Softmax over scores
			tensor.Softmax(a.AttnScores[:pos+1])

			// Weighted sum of values
			outHead := a.AttnOut[h*headDim : (h+1)*headDim]
			for d := 0; d < headDim; d++ {
				var val float32
				for p := 0; p <= pos; p++ {
					vVec := cache.GetValue(lIdx, kvHead, p)
					val += a.AttnScores[p] * vVec[d]
				}
				outHead[d] = val
			}
		}

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
	if m.OutputWeight.Type == gguf.GGMLTypeQ8_0 {
		rowBytes := (cols / tensor.QK8_0) * tensor.BlockSizeQ8_0
		rowOffset := tokID * rowBytes
		return tensor.DotQ8_0(m.OutputWeight.Data[rowOffset:rowOffset+rowBytes], normX, cols)
	} else if m.OutputWeight.Type == gguf.GGMLTypeQ5_0 {
		rowBytes := (cols / tensor.QK5_0) * tensor.BlockSizeQ5_0
		rowOffset := tokID * rowBytes
		return tensor.DotQ5_0(m.OutputWeight.Data[rowOffset:rowOffset+rowBytes], normX, cols)
	}
	return 0
}
