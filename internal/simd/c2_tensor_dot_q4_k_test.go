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

func TestTensorDotQ4_KVsCOracle(t *testing.T) {
	srcCandidates := []string{
		filepath.Join("..", "..", "c2simd", "sources", "c2archsimd"),
		filepath.Join("sources", "c2archsimd"),
		filepath.Join("/devhoros", "c2simd", "sources", "c2archsimd"),
	}
	var srcDir string
	for _, c := range srcCandidates {
		if _, err := os.Stat(filepath.Join(c, "test_dot_q4_k_oracle.c")); err == nil {
			srcDir = c
			break
		}
	}
	if srcDir == "" {
		t.Fatalf("CRITICAL GATE FAILURE: Oracle C source introuvable dans %v", srcCandidates)
	}

	oracleSrc := filepath.Join(srcDir, "test_dot_q4_k_oracle.c")
	cSrc := filepath.Join(srcDir, "c2_tensor_dot_q4_k.c")
	tmpDir := t.TempDir()
	tmpBin := filepath.Join(tmpDir, "test_dot_q4_k_oracle")
	outBin := filepath.Join(tmpDir, "oracle_out.bin")

	compileArgs := []string{"-O2", "-mavx2", "-fsanitize=address,undefined", "-I", srcDir, oracleSrc, cSrc, "-lm", "-o", tmpBin}
	if !SimdEnabled {
		compileArgs = append(compileArgs, "-DNO_AVX2")
	}

	compileCmd := exec.Command("gcc", compileArgs...)
	if out, err := compileCmd.CombinedOutput(); err != nil {
		t.Fatalf("gcc compile failed: %v\n%s", err, string(out))
	}

	cOut, err := exec.Command(tmpBin, outBin).CombinedOutput()
	if err != nil {
		t.Fatalf("C oracle execution failed: %v\n%s", err, string(cOut))
	}
	t.Logf("C Oracle output: %s", string(cOut))

	// Lecture binaire exacte du résultat oracle C
	rawOracleBytes, err := os.ReadFile(outBin)
	if err != nil {
		t.Fatalf("Failed to read oracle binary output %s: %v", outBin, err)
	}
	var expectedOracle float32
	if err := binary.Read(bytes.NewReader(rawOracleBytes), binary.LittleEndian, &expectedOracle); err != nil {
		t.Fatalf("Failed to decode float32 from oracle binary: %v", err)
	}

	// Go reference test on identical fixture
	const cols = 512
	const numBlocks = cols / 256
	weight := make([]byte, numBlocks*144)
	x := make([]float32, cols)

	for b := 0; b < numBlocks; b++ {
		blk := weight[b*144 : (b+1)*144]
		// d = 1.0 (fp16 0x3C00)
		blk[0] = 0x00
		blk[1] = 0x3C
		// dmin = 0.5 (fp16 0x3800)
		blk[2] = 0x00
		blk[3] = 0x38
		for i := 0; i < 12; i++ {
			blk[4+i] = byte((b*37 + i*19 + 17) & 0xFF)
		}
		for i := 0; i < 128; i++ {
			blk[16+i] = byte((b*13 + i*7 + 3) & 0xFF)
		}
	}

	for i := 0; i < cols; i++ {
		x[i] = float32((i%17)-8) * 0.125
	}

	resGo := C2_tensor_dot_q4_k(weight, x, cols)

	diff := math.Abs(float64(resGo - expectedOracle))
	if math.Float32bits(resGo) != math.Float32bits(expectedOracle) {
		t.Fatalf("BIT-EXACT PARITY FAILURE: Go=%f (bits 0x%08x), Oracle C (raw bin)=%f (bits 0x%08x), diff=%e",
			resGo, math.Float32bits(resGo), expectedOracle, math.Float32bits(expectedOracle), diff)
	}

	modeStr := "AVX2 SIMD"
	if !SimdEnabled {
		modeStr = "SCALAR FALLBACK"
	}
	t.Logf("PARITY VALIDATED [%s]: C2_tensor_dot_q4_k is 100%% BIT-EXACT against GCC -O2 C99 oracle (Go=%f, Oracle=%f, diff=%e)!", modeStr, resGo, expectedOracle, diff)
}
