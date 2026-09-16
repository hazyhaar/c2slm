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

const cQ6_KSource = `
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef uint16_t ggml_fp16_t;

static inline float fp16_to_fp32(ggml_fp16_t h) {
    uint32_t sign = ((uint32_t)(h >> 15)) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) {
            float f = 0.0f;
            return sign ? -f : f;
        }
        while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
        exp++;
        mant &= 0x3FF;
        exp = (exp + (127 - 15)) << 23;
        mant <<= 13;
        u = sign | exp | mant;
    } else if (exp == 31) {
        if (mant == 0) {
            return sign ? -INFINITY : INFINITY;
        }
        return NAN;
    } else {
        exp = (exp + (127 - 15)) << 23;
        mant <<= 13;
        u = sign | exp | mant;
    }
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

typedef struct {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t  scales[16];
    ggml_fp16_t d;
} block_q6_K;

void dequantize_q6_K(const block_q6_K *b, float *dst) {
    float d = fp16_to_fp32(b->d);

    const uint8_t *ql = b->ql;
    const uint8_t *qh = b->qh;
    const int8_t  *sc = b->scales;

    for (int n = 0; n < 2; ++n) {
        const uint8_t *qlN = ql + n * 64;
        const uint8_t *qhN = qh + n * 32;
        const int8_t  *scN = sc + n * 8;
        float *dstN = dst + n * 128;

        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            int8_t dsc0 = scN[is + 0];
            int8_t dsc1 = scN[is + 2];
            int8_t dsc2 = scN[is + 4];
            int8_t dsc3 = scN[is + 6];

            uint8_t qhVal = qhN[l];
            int8_t q0 = (int8_t)((qlN[l + 0]  & 0xF) | (((qhVal >> 0) & 3) << 4)) - 32;
            int8_t q1 = (int8_t)((qlN[l + 32] & 0xF) | (((qhVal >> 2) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((qlN[l + 0]  >> 4)  | (((qhVal >> 4) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((qlN[l + 32] >> 4)  | (((qhVal >> 6) & 3) << 4)) - 32;

            dstN[l + 0]  = d * (float)dsc0 * (float)q0;
            dstN[l + 32] = d * (float)dsc1 * (float)q1;
            dstN[l + 64] = d * (float)dsc2 * (float)q2;
            dstN[l + 96] = d * (float)dsc3 * (float)q3;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 4) return 1;
    FILE *fin = fopen(argv[1], "rb");
    if (!fin) return 2;
    block_q6_K b;
    if (fread(&b, sizeof(block_q6_K), 1, fin) != 1) {
        fclose(fin);
        return 3;
    }
    fclose(fin);

    FILE *fdense = fopen(argv[2], "rb");
    if (!fdense) return 4;
    float denseX[256];
    if (fread(denseX, sizeof(float), 256, fdense) != 256) {
        fclose(fdense);
        return 5;
    }
    fclose(fdense);

    float out[256];
    dequantize_q6_K(&b, out);

    float dot = 0.0f;
    for (int i = 0; i < 256; ++i) {
        dot += out[i] * denseX[i];
    }

    FILE *fout = fopen(argv[3], "wb");
    if (!fout) return 6;
    fwrite(out, sizeof(float), 256, fout);
    fwrite(&dot, sizeof(float), 1, fout);
    fclose(fout);
    return 0;
}
`

func TestQ6_KVsCOracle(t *testing.T) {
	tmpDir := t.TempDir()
	cFile := filepath.Join(tmpDir, "q6_k.c")
	binFile := filepath.Join(tmpDir, "q6_k")
	if err := os.WriteFile(cFile, []byte(cQ6_KSource), 0644); err != nil {
		t.Fatalf("write c source: %v", err)
	}

	cmd := exec.Command("gcc", "-O2", "-std=c99", "-pedantic", "-Wall", "-Werror", "-o", binFile, cFile, "-lm")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("gcc compilation failed: %v, output: %s", err, out)
	}

	// Create test Q6_K block (210 bytes)
	rawBlock := make([]byte, 210)
	for i := 0; i < 128; i++ {
		rawBlock[i] = byte((i*19 + 5) & 0xFF) // ql
	}
	for i := 0; i < 64; i++ {
		rawBlock[128+i] = byte((i*29 + 17) & 0xFF) // qh
	}
	for i := 0; i < 16; i++ {
		rawBlock[192+i] = byte((i*7 - 31) & 0xFF) // scales (signed int8)
	}
	binary.LittleEndian.PutUint16(rawBlock[208:210], 0x3C00) // d = 1.0

	// Vecteur dense float32
	denseX := make([]float32, 256)
	for i := 0; i < 256; i++ {
		denseX[i] = float32(i%13 - 6)
	}
	denseBytes := make([]byte, 256*4)
	for i, v := range denseX {
		binary.LittleEndian.PutUint32(denseBytes[i*4:(i+1)*4], math.Float32bits(v))
	}

	inFile := filepath.Join(tmpDir, "in.bin")
	denseFile := filepath.Join(tmpDir, "dense.bin")
	outFile := filepath.Join(tmpDir, "out.bin")
	if err := os.WriteFile(inFile, rawBlock, 0644); err != nil {
		t.Fatalf("write in: %v", err)
	}
	if err := os.WriteFile(denseFile, denseBytes, 0644); err != nil {
		t.Fatalf("write dense: %v", err)
	}

	cmdRun := exec.Command(binFile, inFile, denseFile, outFile)
	if out, err := cmdRun.CombinedOutput(); err != nil {
		t.Fatalf("run C oracle failed: %v, output: %s", err, out)
	}

	cOutBytes, err := os.ReadFile(outFile)
	if err != nil {
		t.Fatalf("read out: %v", err)
	}
	if len(cOutBytes) != (256+1)*4 {
		t.Fatalf("cOut size %d != expected %d", len(cOutBytes), (256+1)*4)
	}

	var cOut [256]float32
	if err := binary.Read(bytes.NewReader(cOutBytes[0:1024]), binary.LittleEndian, &cOut); err != nil {
		t.Fatalf("read cOut: %v", err)
	}
	var cDot float32
	if err := binary.Read(bytes.NewReader(cOutBytes[1024:1028]), binary.LittleEndian, &cDot); err != nil {
		t.Fatalf("read cDot: %v", err)
	}

	// 1. Vérification bit-exacte stricte sans NaN sur chaque composante
	for i := 0; i < 256; i++ {
		unitX := make([]float32, 256)
		unitX[i] = 1.0
		goVal := tensor.DotQ6_K(rawBlock, unitX, 256)
		cVal := cOut[i]

		if math.IsNaN(float64(goVal)) || math.IsNaN(float64(cVal)) {
			t.Fatalf("NaN detected at index %d: Go=%f, C=%f", i, goVal, cVal)
		}
		// En IEEE 754, +0.0 et -0.0 ont des représentations binaires différentes (0x00000000 vs 0x80000000) mais sont numériquement identiques (+0.0 == -0.0).
		if goVal != cVal || (math.Float32bits(goVal) != math.Float32bits(cVal) && !(goVal == 0 && cVal == 0)) {
			t.Fatalf("Q6_K bit mismatch at index %d: Go=%f (0x%08x), C=%f (0x%08x)",
				i, goVal, math.Float32bits(goVal), cVal, math.Float32bits(cVal))
		}
	}

	// 2. Vérification du produit dense calculé directement par l'oracle C
	actualDenseDot := tensor.DotQ6_K(rawBlock, denseX, 256)
	if math.IsNaN(float64(actualDenseDot)) || math.IsNaN(float64(cDot)) {
		t.Fatalf("NaN detected in dense dot")
	}
	if math.Float32bits(actualDenseDot) != math.Float32bits(cDot) {
		t.Fatalf("Dense dot bit mismatch: Go=%f (0x%08x), C=%f (0x%08x)",
			actualDenseDot, math.Float32bits(actualDenseDot), cDot, math.Float32bits(cDot))
	}

	t.Logf("Q6_K: 256/256 weights AND dense dot product are 100%% BIT-EXACT against GCC -O2 C99 oracle!")
}

func TestQ6_K_HostileBlocks_VsCOracle(t *testing.T) {
	tmpDir := t.TempDir()
	cFile := filepath.Join(tmpDir, "q6_k.c")
	binFile := filepath.Join(tmpDir, "q6_k")
	if err := os.WriteFile(cFile, []byte(cQ6_KSource), 0644); err != nil {
		t.Fatalf("write c source: %v", err)
	}
	cmd := exec.Command("gcc", "-O2", "-std=c99", "-pedantic", "-Wall", "-Werror", "-o", binFile, cFile, "-lm")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("gcc compilation failed: %v, output: %s", err, out)
	}

	denseX := make([]float32, 256)
	for i := 0; i < 256; i++ {
		denseX[i] = 1.0
	}
	denseBytes := make([]byte, 256*4)
	for i, v := range denseX {
		binary.LittleEndian.PutUint32(denseBytes[i*4:(i+1)*4], math.Float32bits(v))
	}
	denseFile := filepath.Join(tmpDir, "dense.bin")
	if err := os.WriteFile(denseFile, denseBytes, 0644); err != nil {
		t.Fatalf("write dense: %v", err)
	}

	patterns := []struct {
		name string
		val  byte
	}{
		{"AllZero", 0x00},
		{"AllOne", 0xFF},
		{"AlternatingAA", 0xAA},
		{"Alternating55", 0x55},
	}

	for _, p := range patterns {
		raw := make([]byte, 210)
		for j := range raw {
			raw[j] = p.val
		}
		// fp16 scale d
		binary.LittleEndian.PutUint16(raw[208:210], 0x3C00) // d = 1.0

		inFile := filepath.Join(tmpDir, p.name+"_in.bin")
		outFile := filepath.Join(tmpDir, p.name+"_out.bin")
		if err := os.WriteFile(inFile, raw, 0644); err != nil {
			t.Fatalf("write in: %v", err)
		}

		cmdRun := exec.Command(binFile, inFile, denseFile, outFile)
		if out, err := cmdRun.CombinedOutput(); err != nil {
			t.Fatalf("run C oracle failed for pattern %s: %v, output: %s", p.name, err, out)
		}
		cOutBytes, err := os.ReadFile(outFile)
		if err != nil {
			t.Fatalf("read out: %v", err)
		}
		var cOut [256]float32
		if err := binary.Read(bytes.NewReader(cOutBytes[0:1024]), binary.LittleEndian, &cOut); err != nil {
			t.Fatalf("read cOut: %v", err)
		}
		for i := 0; i < 256; i++ {
			unitX := make([]float32, 256)
			unitX[i] = 1.0
			goVal := tensor.DotQ6_K(raw, unitX, 256)
			cVal := cOut[i]
			if math.IsNaN(float64(goVal)) || math.IsNaN(float64(cVal)) {
				t.Fatalf("[%s] NaN detected at %d", p.name, i)
			}
			if goVal != cVal || (math.Float32bits(goVal) != math.Float32bits(cVal) && !(goVal == 0 && cVal == 0)) {
				t.Fatalf("[%s] mismatch at %d: Go=%f (0x%08x), C=%f (0x%08x)",
					p.name, i, goVal, math.Float32bits(goVal), cVal, math.Float32bits(cVal))
			}
		}
	}
	t.Logf("Q6_K: Hostile block patterns (0x00, 0xFF, 0xAA, 0x55) validated 100%% bit-exact against GCC -O2!")
}
