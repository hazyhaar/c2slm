package engine

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/hazyhaar/c2slm/gguf"
	"github.com/hazyhaar/c2slm/tensor"
)

// Layer contains all mapped weights and biases for one Transformer block
type Layer struct {
	Index     int
	AttnNorm  []float32 // [896]
	AttnQ     *gguf.TensorInfo
	AttnQBias []float32
	AttnQNorm []float32 // QK-Norm per-head (Qwen3)
	AttnK     *gguf.TensorInfo
	AttnKBias []float32
	AttnKNorm []float32 // QK-Norm per-head (Qwen3)
	AttnV     *gguf.TensorInfo
	AttnVBias []float32 // [128]
	AttnOut   *gguf.TensorInfo
	FFNNorm   []float32 // [896]
	FFNGate   *gguf.TensorInfo
	FFNUp     *gguf.TensorInfo
	FFNDown   *gguf.TensorInfo
}

// Model represents Qwen2.5-0.5B-Instruct architecture
type Model struct {
	EmbdLength int
	NumLayers  int
	NumHeads   int
	NumKVHeads int
	HeadDim    int
	FFNDim     int
	VocabSize  int
	RoPEBase   float32
	RMSNormEps float32

	TokenEmbd    *gguf.TensorInfo
	OutputNorm   []float32
	OutputWeight *gguf.TensorInfo
	RoPETable    *tensor.RoPETable

	Layers   []Layer
	GGUFFile *gguf.File
}

// LoadModel loads Qwen2.5 weights and tensors from GGUF
func LoadModel(path string) (*Model, error) {
	gf, err := gguf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open gguf: %w", err)
	}

	embdLen, _ := gf.GetUint32("qwen2.embedding_length")
	if embdLen == 0 {
		embdLen, _ = gf.GetUint32("qwen3.embedding_length")
	}
	if embdLen == 0 {
		embdLen = 896
	}
	numLayers, _ := gf.GetUint32("qwen2.block_count")
	if numLayers == 0 {
		numLayers, _ = gf.GetUint32("qwen3.block_count")
	}
	if numLayers == 0 {
		numLayers = 24
	}
	numHeads, _ := gf.GetUint32("qwen2.attention.head_count")
	if numHeads == 0 {
		numHeads, _ = gf.GetUint32("qwen3.attention.head_count")
	}
	if numHeads == 0 {
		numHeads = 14
	}
	numKVHeads, _ := gf.GetUint32("qwen2.attention.head_count_kv")
	if numKVHeads == 0 {
		numKVHeads, _ = gf.GetUint32("qwen3.attention.head_count_kv")
	}
	if numKVHeads == 0 {
		numKVHeads = 2
	}
	ffnDim, _ := gf.GetUint32("qwen2.feed_forward_length")
	if ffnDim == 0 {
		ffnDim, _ = gf.GetUint32("qwen3.feed_forward_length")
	}
	if ffnDim == 0 {
		ffnDim = 4864
	}
	ropeTheta, _ := gf.GetFloat32("qwen2.rope.freq_base")
	if ropeTheta == 0 {
		ropeTheta, _ = gf.GetFloat32("qwen3.rope.freq_base")
	}
	if ropeTheta == 0 {
		ropeTheta = 1000000.0
	}
	eps, _ := gf.GetFloat32("qwen2.attention.layer_norm_rms_epsilon")
	if eps == 0 {
		eps, _ = gf.GetFloat32("qwen3.attention.layer_norm_rms_epsilon")
	}
	if eps == 0 {
		eps = 1e-6
	}

	tokEmbd, ok := gf.TensorMap["token_embd.weight"]
	if !ok {
		return nil, fmt.Errorf("missing token_embd.weight")
	}
	vocabSize := int(tokEmbd.Dimensions[1])

	outNormTi, ok := gf.TensorMap["output_norm.weight"]
	if !ok {
		return nil, fmt.Errorf("missing output_norm.weight")
	}
	outNorm := bytesToF32(outNormTi.Data)

	outWeight, ok := gf.TensorMap["output.weight"]
	if !ok {
		// Tied embeddings fallback
		outWeight = tokEmbd
	}

	m := &Model{
		EmbdLength:   int(embdLen),
		NumLayers:    int(numLayers),
		NumHeads:     int(numHeads),
		NumKVHeads:   int(numKVHeads),
		HeadDim:      int(embdLen) / int(numHeads), // 896 / 14 = 64
		FFNDim:       int(ffnDim),
		VocabSize:    vocabSize,
		RoPEBase:     ropeTheta,
		RMSNormEps:   eps,
		TokenEmbd:    tokEmbd,
		OutputNorm:   outNorm,
		OutputWeight: outWeight,
		RoPETable:    tensor.NewRoPETable(MaxContextLen, int(embdLen)/int(numHeads), ropeTheta),
		Layers:       make([]Layer, numLayers),
		GGUFFile:     gf,
	}

	for i := 0; i < int(numLayers); i++ {
		l := &m.Layers[i]
		l.Index = i

		// Attention Norm
		if ti, ok := gf.TensorMap[fmt.Sprintf("blk.%d.attn_norm.weight", i)]; ok {
			l.AttnNorm = bytesToF32(ti.Data)
		} else {
			return nil, fmt.Errorf("missing blk.%d.attn_norm.weight", i)
		}

		// Q, K, V weights and biases
		if ti, ok := gf.TensorMap[fmt.Sprintf("blk.%d.attn_q.weight", i)]; ok {
			l.AttnQ = ti
		}
		if ti, ok := gf.TensorMap[fmt.Sprintf("blk.%d.attn_q.bias", i)]; ok {
			l.AttnQBias = bytesToF32(ti.Data)
		}
		if ti, ok := gf.TensorMap[fmt.Sprintf("blk.%d.attn_q_norm.weight", i)]; ok {
			l.AttnQNorm = bytesToF32(ti.Data)
		}
		if ti, ok := gf.TensorMap[fmt.Sprintf("blk.%d.attn_k.weight", i)]; ok {
			l.AttnK = ti
		}
		if ti, ok := gf.TensorMap[fmt.Sprintf("blk.%d.attn_k.bias", i)]; ok {
			l.AttnKBias = bytesToF32(ti.Data)
		}
		if ti, ok := gf.TensorMap[fmt.Sprintf("blk.%d.attn_k_norm.weight", i)]; ok {
			l.AttnKNorm = bytesToF32(ti.Data)
		}
		if ti, ok := gf.TensorMap[fmt.Sprintf("blk.%d.attn_v.weight", i)]; ok {
			l.AttnV = ti
		}
		if ti, ok := gf.TensorMap[fmt.Sprintf("blk.%d.attn_v.bias", i)]; ok {
			l.AttnVBias = bytesToF32(ti.Data)
		}

		// Output projection
		if ti, ok := gf.TensorMap[fmt.Sprintf("blk.%d.attn_output.weight", i)]; ok {
			l.AttnOut = ti
		}

		// FFN Norm
		if ti, ok := gf.TensorMap[fmt.Sprintf("blk.%d.ffn_norm.weight", i)]; ok {
			l.FFNNorm = bytesToF32(ti.Data)
		}

		// FFN projections
		if ti, ok := gf.TensorMap[fmt.Sprintf("blk.%d.ffn_gate.weight", i)]; ok {
			l.FFNGate = ti
		}
		if ti, ok := gf.TensorMap[fmt.Sprintf("blk.%d.ffn_up.weight", i)]; ok {
			l.FFNUp = ti
		}
		if ti, ok := gf.TensorMap[fmt.Sprintf("blk.%d.ffn_down.weight", i)]; ok {
			l.FFNDown = ti
		}
	}

	return m, nil
}

// Close releases mmap memory
func (m *Model) Close() error {
	if m.GGUFFile != nil {
		return m.GGUFFile.Close()
	}
	return nil
}

// EmbedToken extracts and dequantizes token embedding into out float32 slice
func (m *Model) EmbedToken(out []float32, tokenID int32) {
	if int(tokenID) >= m.VocabSize || tokenID < 0 {
		tokenID = 0
	}
	switch m.TokenEmbd.Type {
	case gguf.GGMLTypeQ4_K:
		rowBytes := (m.EmbdLength / tensor.QK4_K) * tensor.BlockSizeQ4_K
		rowOffset := int(tokenID) * rowBytes
		tensor.DequantizeRowQ4_K(out, m.TokenEmbd.Data[rowOffset:rowOffset+rowBytes], m.EmbdLength)
	case gguf.GGMLTypeQ5_0:
		rowBytes := (m.EmbdLength / tensor.QK5_0) * tensor.BlockSizeQ5_0
		rowOffset := int(tokenID) * rowBytes
		tensor.DequantizeRowQ5_0(out, m.TokenEmbd.Data[rowOffset:rowOffset+rowBytes], m.EmbdLength)
	case gguf.GGMLTypeQ8_0:
		rowBytes := (m.EmbdLength / tensor.QK8_0) * tensor.BlockSizeQ8_0
		rowOffset := int(tokenID) * rowBytes
		tensor.DequantizeRowQ8_0(out, m.TokenEmbd.Data[rowOffset:rowOffset+rowBytes], m.EmbdLength)
	case gguf.GGMLTypeF32:
		rowBytes := m.EmbdLength * 4
		rowOffset := int(tokenID) * rowBytes
		data := m.TokenEmbd.Data[rowOffset : rowOffset+rowBytes]
		for i := 0; i < m.EmbdLength; i++ {
			bits := binary.LittleEndian.Uint32(data[i*4 : (i+1)*4])
			out[i] = math.Float32frombits(bits)
		}
	}
}

// DispatchGEMV performs GEMV based on tensor quantization type, using pool if non-nil
func DispatchGEMV(pool *WorkerPool, y []float32, ti *gguf.TensorInfo, x []float32, rows, cols int) {
	if pool != nil {
		pool.ParallelGEMV(y, ti.Data, x, rows, cols, ti.Type)
		return
	}
	switch ti.Type {
	case gguf.GGMLTypeQ5_0:
		tensor.GEMVQ5_0(y, ti.Data, x, rows, cols)
	case gguf.GGMLTypeQ8_0:
		tensor.GEMVQ8_0(y, ti.Data, x, rows, cols)
	case gguf.GGMLTypeQ6_K:
		tensor.GEMVQ6_K(y, ti.Data, x, rows, cols)
	case gguf.GGMLTypeQ4_K:
		tensor.GEMVQ4_K(y, ti.Data, x, rows, cols)
	default:
		// Fallback zero
		for i := range y {
			y[i] = 0
		}
	}
}

func bytesToF32(b []byte) []float32 {
	n := len(b) / 4
	res := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint32(b[i*4 : (i+1)*4])
		res[i] = math.Float32frombits(bits)
	}
	return res
}
