package tensor_test

import (
	"math"
	"os"
	"testing"

	"github.com/hazyhaar/c2slm/gguf"
	"github.com/hazyhaar/c2slm/tensor"
)

const modelPath = "/data/models/qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"

func TestRealTensorsGEMV(t *testing.T) {
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("model file not found at %s", modelPath)
	}

	gf, err := gguf.Open(modelPath)
	if err != nil {
		t.Fatalf("gguf.Open failed: %v", err)
	}
	defer gf.Close()

	// 1. Test GEMV Q5_0 on blk.0.attn_q.weight (896 x 896)
	qTi, ok := gf.TensorMap["blk.0.attn_q.weight"]
	if !ok {
		t.Fatalf("missing blk.0.attn_q.weight")
	}
	if qTi.Type != gguf.GGMLTypeQ5_0 {
		t.Fatalf("expected Q5_0, got %s", qTi.Type)
	}

	cols := int(qTi.Dimensions[0]) // 896
	rows := int(qTi.Dimensions[1]) // 896
	x := make([]float32, cols)
	for i := range x {
		x[i] = 1.0 / float32(math.Sqrt(float64(cols)))
	}
	y := make([]float32, rows)

	tensor.GEMVQ5_0(y, qTi.Data, x, rows, cols)

	for i, v := range y {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("y[%d] is invalid: %f", i, v)
		}
	}
	t.Logf("Q5_0 GEMV (896x896) success: y[0]=%f, y[10]=%f, y[100]=%f", y[0], y[10], y[100])

	// 2. Test GEMV Q8_0 on blk.0.attn_v.weight (896 x 128)
	vTi, ok := gf.TensorMap["blk.0.attn_v.weight"]
	if !ok {
		t.Fatalf("missing blk.0.attn_v.weight")
	}
	t.Logf("blk.0.attn_v.weight: type=%s, dims=%v", vTi.Type, vTi.Dimensions)
	vCols := int(vTi.Dimensions[0]) // 896
	vRows := int(vTi.Dimensions[1]) // 128
	yV := make([]float32, vRows)

	if vTi.Type == gguf.GGMLTypeQ8_0 {
		tensor.GEMVQ8_0(yV, vTi.Data, x, vRows, vCols)
		t.Logf("Q8_0 GEMV (896x128) success: yV[0]=%f, yV[10]=%f", yV[0], yV[10])
	} else if vTi.Type == gguf.GGMLTypeQ5_0 {
		tensor.GEMVQ5_0(yV, vTi.Data, x, vRows, vCols)
		t.Logf("Q5_0 GEMV (896x128) success: yV[0]=%f, yV[10]=%f", yV[0], yV[10])
	}

	// 3. Test GEMV Q6_K on blk.0.ffn_down.weight (4864 x 896)
	downTi, ok := gf.TensorMap["blk.0.ffn_down.weight"]
	if !ok {
		t.Fatalf("missing blk.0.ffn_down.weight")
	}
	downCols := int(downTi.Dimensions[0]) // 4864
	downRows := int(downTi.Dimensions[1]) // 896
	xDown := make([]float32, downCols)
	for i := range xDown {
		xDown[i] = 1.0 / float32(math.Sqrt(float64(downCols)))
	}
	yDown := make([]float32, downRows)

	if downTi.Type == gguf.GGMLTypeQ6_K {
		tensor.GEMVQ6_K(yDown, downTi.Data, xDown, downRows, downCols)
		t.Logf("Q6_K GEMV (4864x896) success: yDown[0]=%f, yDown[10]=%f", yDown[0], yDown[10])
	} else if downTi.Type == gguf.GGMLTypeQ4_K {
		tensor.GEMVQ4_K(yDown, downTi.Data, xDown, downRows, downCols)
		t.Logf("Q4_K GEMV (4864x896) success: yDown[0]=%f, yDown[10]=%f", yDown[0], yDown[10])
	}
}
