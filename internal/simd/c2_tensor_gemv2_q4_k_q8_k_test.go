package simd

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestTensorGEMV2_Q4_K_Q8_KVsCOracle(t *testing.T) {
	srcCandidates := []string{
		filepath.Join("..", "..", "..", "c2simd", "sources", "c2archsimd"),
		filepath.Join("sources", "c2archsimd"),
		filepath.Join("/devhoros", "c2simd", "sources", "c2archsimd"),
	}
	var srcDir string
	for _, c := range srcCandidates {
		if _, err := os.Stat(filepath.Join(c, "test_gemv2_q4_k_q8_k_oracle.c")); err == nil {
			srcDir = c
			break
		}
	}
	if srcDir == "" {
		t.Fatalf("CRITICAL GATE FAILURE: Oracle C source test_gemv2_q4_k_q8_k_oracle.c introuvable dans %v", srcCandidates)
	}

	oracleSrc := filepath.Join(srcDir, "test_gemv2_q4_k_q8_k_oracle.c")
	quantSrc := filepath.Join(srcDir, "c2_quantize_q8_k.c")
	dotSrc := filepath.Join(srcDir, "c2_tensor_dot_q4_k_q8_k.c")
	gemv2Src := filepath.Join(srcDir, "c2_tensor_gemv2_q4_k_q8_k.c")
	tmpDir := t.TempDir()
	tmpBin := filepath.Join(tmpDir, "test_gemv2_oracle")
	outBin0 := filepath.Join(tmpDir, "oracle_out0.bin")
	outBin1 := filepath.Join(tmpDir, "oracle_out1.bin")

	compileArgs := []string{"-O2", "-mavx2", "-fsanitize=address,undefined", "-I", srcDir, oracleSrc, quantSrc, dotSrc, gemv2Src, "-lm", "-o", tmpBin}
	if !SimdEnabled {
		compileArgs = append(compileArgs, "-DNO_AVX2")
	}

	compileCmd := exec.Command("gcc", compileArgs...)
	if out, err := compileCmd.CombinedOutput(); err != nil {
		t.Fatalf("gcc compile failed: %v\n%s", err, string(out))
	}

	cOut, err := exec.Command(tmpBin, outBin0, outBin1).CombinedOutput()
	if err != nil {
		t.Fatalf("C oracle execution failed: %v\n%s", err, string(cOut))
	}
	t.Logf("C Oracle output:\n%s", string(cOut))

	rawOracle0, err := os.ReadFile(outBin0)
	if err != nil {
		t.Fatalf("Failed to read oracle binary output %s: %v", outBin0, err)
	}
	var expectedOracle0 float32
	if err := binary.Read(bytes.NewReader(rawOracle0), binary.LittleEndian, &expectedOracle0); err != nil {
		t.Fatalf("Failed to decode float32 from oracle binary 0: %v", err)
	}

	rawOracle1, err := os.ReadFile(outBin1)
	if err != nil {
		t.Fatalf("Failed to read oracle binary output %s: %v", outBin1, err)
	}
	var expectedOracle1 float32
	if err := binary.Read(bytes.NewReader(rawOracle1), binary.LittleEndian, &expectedOracle1); err != nil {
		t.Fatalf("Failed to decode float32 from oracle binary 1: %v", err)
	}

	const cols = 512
	const numBlocks = cols / 256
	w0 := make([]byte, numBlocks*144)
	w1 := make([]byte, numBlocks*144)
	x := make([]float32, cols)
	q8k := make([]byte, numBlocks*292)

	for b := 0; b < numBlocks; b++ {
		blk0 := w0[b*144 : (b+1)*144]
		// d = 1.0 (fp16 0x3C00)
		blk0[0] = 0x00
		blk0[1] = 0x3C
		// dmin = 0.5 (fp16 0x3800)
		blk0[2] = 0x00
		blk0[3] = 0x38
		for i := 0; i < 12; i++ {
			blk0[4+i] = byte((b*37 + i*19 + 17) & 0xFF)
		}
		for i := 0; i < 128; i++ {
			blk0[16+i] = byte((b*13 + i*7 + 3) & 0xFF)
		}

		blk1 := w1[b*144 : (b+1)*144]
		// d = 0.75 (fp16 0x3A00)
		blk1[0] = 0x00
		blk1[1] = 0x3A
		// dmin = 0.25 (fp16 0x3400)
		blk1[2] = 0x00
		blk1[3] = 0x34
		for i := 0; i < 12; i++ {
			blk1[4+i] = byte((b*41 + i*23 + 19) & 0xFF)
		}
		for i := 0; i < 128; i++ {
			blk1[16+i] = byte((b*17 + i*11 + 5) & 0xFF)
		}
	}

	for i := 0; i < cols; i++ {
		x[i] = float32((i%17)-8) * 0.125
	}

	C2_quantize_row_q8_k(x, q8k, cols)

	var got0, got1 float32
	C2_tensor_gemv2_q4_k_q8_k(w0, w1, q8k, cols, &got0, &got1)

	t.Logf("Row 0: Go=%f, Oracle=%f, diff=%e", got0, expectedOracle0, math.Abs(float64(got0-expectedOracle0)))
	t.Logf("Row 1: Go=%f, Oracle=%f, diff=%e", got1, expectedOracle1, math.Abs(float64(got1-expectedOracle1)))

	if math.Float32bits(got0) != math.Float32bits(expectedOracle0) {
		t.Fatalf("BIT-EXACT PARITY FAILURE Row 0: Go=%f (bits 0x%08x), Oracle C=%f (bits 0x%08x)",
			got0, math.Float32bits(got0), expectedOracle0, math.Float32bits(expectedOracle0))
	}

	if math.Float32bits(got1) != math.Float32bits(expectedOracle1) {
		t.Fatalf("BIT-EXACT PARITY FAILURE Row 1: Go=%f (bits 0x%08x), Oracle C=%f (bits 0x%08x)",
			got1, math.Float32bits(got1), expectedOracle1, math.Float32bits(expectedOracle1))
	}

	modeStr := "AVX2 SIMD"
	if !SimdEnabled {
		modeStr = "SCALAR FALLBACK"
	}
	t.Logf("PARITY VALIDATED [%s]: C2_tensor_gemv2_q4_k_q8_k is 100%% BIT-EXACT against GCC -O2 C99 oracle!", modeStr)
}
