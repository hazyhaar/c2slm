package tensor_test

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/hazyhaar/c2slm/tensor"
)

// C source code of GGML dequantize_row_q5_0
const cQ5_0Source = `
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef uint16_t ggml_fp16_t;

static inline float fp16_to_fp32(ggml_fp16_t h) {
    uint32_t sign = ((uint32_t)(h >> 15)) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
        exp++;
        mant &= 0x3FF;
        exp = (exp + (127 - 15)) << 23;
        mant <<= 13;
        uint32_t u = sign | exp | mant;
        return *(float *)&u;
    }
    if (exp == 31) return (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
    exp = (exp + (127 - 15)) << 23;
    mant <<= 13;
    uint32_t u = sign | exp | mant;
    return *(float *)&u;
}

typedef struct {
    ggml_fp16_t d;
    uint8_t qh[4];
    uint8_t qs[16];
} block_q5_0;

void dequantize_q5_0(const void *src, float *dst, int k) {
    const block_q5_0 *b = (const block_q5_0 *)src;
    int nb = k / 32;
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(b[i].d);
        uint32_t qh = *(const uint32_t *)b[i].qh;
        for (int j = 0; j < 16; j++) {
            uint8_t byteVal = b[i].qs[j];
            uint8_t x0 = byteVal & 0x0F;
            uint8_t x1 = byteVal >> 4;

            uint8_t h0 = (qh >> j) & 1;
            uint8_t h1 = (qh >> (j + 16)) & 1;

            int32_t q0 = (int32_t)(x0 | (h0 << 4)) - 16;
            int32_t q1 = (int32_t)(x1 | (h1 << 4)) - 16;

            dst[i*32 + j] = d * (float)q0;
            dst[i*32 + j + 16] = d * (float)q1;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) return 1;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    uint8_t buf[22];
    if (fread(buf, 1, 22, f) != 22) return 3;
    fclose(f);

    float out[32];
    dequantize_q5_0(buf, out, 32);

    FILE *fout = fopen(argv[2], "wb");
    if (!fout) return 4;
    fwrite(out, sizeof(float), 32, fout);
    fclose(fout);
    return 0;
}
`

func TestQ5_0VsCOracle(t *testing.T) {
	tmpDir := t.TempDir()
	cFile := filepath.Join(tmpDir, "q5_0.c")
	binFile := filepath.Join(tmpDir, "q5_0")
	if err := os.WriteFile(cFile, []byte(cQ5_0Source), 0644); err != nil {
		t.Fatalf("write c source: %v", err)
	}

	cmd := exec.Command("gcc", "-O2", "-o", binFile, cFile, "-lm")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("gcc compilation failed: %v, output: %s", err, out)
	}

	// Create a known Q5_0 block of 22 bytes
	rawBlock := make([]byte, 22)
	// d = 1.0 (0x3C00)
	binary.LittleEndian.PutUint16(rawBlock[0:2], 0x3C00)
	// qh = 0xAAAAAAAA (alternating high bits)
	binary.LittleEndian.PutUint32(rawBlock[2:6], 0xAAAAAAAA)
	// qs = alternating 0x37
	for i := 0; i < 16; i++ {
		rawBlock[6+i] = byte(i * 17)
	}

	inFile := filepath.Join(tmpDir, "in.bin")
	outFile := filepath.Join(tmpDir, "out.bin")
	if err := os.WriteFile(inFile, rawBlock, 0644); err != nil {
		t.Fatalf("write in.bin: %v", err)
	}

	cmdRun := exec.Command(binFile, inFile, outFile)
	if out, err := cmdRun.CombinedOutput(); err != nil {
		t.Fatalf("run C oracle failed: %v, output: %s", err, out)
	}

	cOutBytes, err := os.ReadFile(outFile)
	if err != nil {
		t.Fatalf("read out.bin: %v", err)
	}

	var cOut [32]float32
	if err := binary.Read(bytes.NewReader(cOutBytes), binary.LittleEndian, &cOut); err != nil {
		t.Fatalf("binary read cOut: %v", err)
	}

	// Now compute using Go DotQ5_0 with unit basis vectors
	for i := 0; i < 32; i++ {
		unitX := make([]float32, 32)
		unitX[i] = 1.0
		goVal := tensor.DotQ5_0(rawBlock, unitX, 32)
		cVal := cOut[i]

		if math.Abs(float64(goVal-cVal)) > 1e-6 {
			t.Errorf("mismatch at index %d: Go=%f, C=%f", i, goVal, cVal)
		}
	}
}

func TestQ8_0VsCOracle(t *testing.T) {
	// Known Q8_0 block: 2 bytes d (FP16), 32 bytes int8
	rawBlock := make([]byte, 34)
	binary.LittleEndian.PutUint16(rawBlock[0:2], 0x3C00) // d = 1.0
	for i := 0; i < 32; i++ {
		rawBlock[2+i] = byte(int8(i*2 - 31))
	}

	for i := 0; i < 32; i++ {
		unitX := make([]float32, 32)
		unitX[i] = 1.0
		goVal := tensor.DotQ8_0(rawBlock, unitX, 32)
		expectedVal := float32(int8(i*2 - 31))

		if math.Abs(float64(goVal-expectedVal)) > 1e-6 {
			t.Errorf("Q8_0 mismatch at %d: Go=%f, expected=%f", i, goVal, expectedVal)
		}
	}
}
