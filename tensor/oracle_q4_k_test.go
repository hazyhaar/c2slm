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

const cQ4_KSource = `
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
    ggml_fp16_t d;
    ggml_fp16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K;

void dequantize_q4_K(const block_q4_K *b, float *dst) {
    float d = fp16_to_fp32(b->d);
    float dmin = fp16_to_fp32(b->dmin);

    uint8_t sc[8];
    uint8_t m[8];
    for (int j = 0; j < 4; ++j) {
        sc[j] = b->scales[j] & 63;
        m[j]  = b->scales[j+4] & 63;
        sc[j+4] = (b->scales[j+8] & 0xF) | ((b->scales[j] >> 6) << 4);
        m[j+4]  = (b->scales[j+8] >> 4)  | ((b->scales[j+4] >> 6) << 4);
    }

    const uint8_t *q = b->qs;
    for (int sb = 0; sb < 4; ++sb) {
        float d0 = d * (float)sc[2*sb + 0];
        float m0 = dmin * (float)m[2*sb + 0];
        float d1 = d * (float)sc[2*sb + 1];
        float m1 = dmin * (float)m[2*sb + 1];

        for (int l = 0; l < 32; ++l) {
            uint8_t v = q[sb*32 + l];
            dst[sb*64 + l]      = d0 * (float)(v & 0xF) - m0;
            dst[sb*64 + l + 32] = d1 * (float)(v >> 4)  - m1;
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 4) return 1;
    FILE *fin = fopen(argv[1], "rb");
    if (!fin) return 2;
    block_q4_K b;
    if (fread(&b, sizeof(block_q4_K), 1, fin) != 1) {
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
    dequantize_q4_K(&b, out);

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

func TestQ4_KVsCOracle(t *testing.T) {
	tmpDir := t.TempDir()
	cFile := filepath.Join(tmpDir, "q4_k.c")
	binFile := filepath.Join(tmpDir, "q4_k")
	if err := os.WriteFile(cFile, []byte(cQ4_KSource), 0644); err != nil {
		t.Fatalf("write c source: %v", err)
	}

	cmd := exec.Command("gcc", "-O2", "-std=c99", "-pedantic", "-Wall", "-Werror", "-o", binFile, cFile, "-lm")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("gcc compilation failed: %v, output: %s", err, out)
	}

	// Create test Q4_K block (144 bytes) avec couverture exhaustive des bits 6 et 7
	rawBlock := make([]byte, 144)
	binary.LittleEndian.PutUint16(rawBlock[0:2], 0x3C00) // d = 1.0
	binary.LittleEndian.PutUint16(rawBlock[2:4], 0x3800) // dmin = 0.5
	for i := 0; i < 12; i++ {
		rawBlock[4+i] = byte((i*37 + 193) & 0xFF)
	}
	for i := 0; i < 128; i++ {
		rawBlock[16+i] = byte((i*13 + 7) & 0xFF)
	}

	// Vecteur dense float32
	denseX := make([]float32, 256)
	for i := 0; i < 256; i++ {
		denseX[i] = float32(i%17 - 8)
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
		goVal := tensor.DotQ4_K(rawBlock, unitX, 256)
		cVal := cOut[i]

		if math.IsNaN(float64(goVal)) || math.IsNaN(float64(cVal)) {
			t.Fatalf("NaN detected at index %d: Go=%f, C=%f", i, goVal, cVal)
		}
		if math.Float32bits(goVal) != math.Float32bits(cVal) {
			t.Fatalf("Q4_K bit mismatch at index %d: Go=%f (0x%08x), C=%f (0x%08x)",
				i, goVal, math.Float32bits(goVal), cVal, math.Float32bits(cVal))
		}
	}

	// 2. Vérification du produit dense calculé directement par l'oracle C
	actualDenseDot := tensor.DotQ4_K(rawBlock, denseX, 256)
	if math.IsNaN(float64(actualDenseDot)) || math.IsNaN(float64(cDot)) {
		t.Fatalf("NaN detected in dense dot")
	}
	if math.Float32bits(actualDenseDot) != math.Float32bits(cDot) {
		t.Fatalf("Dense dot bit mismatch: Go=%f (0x%08x), C=%f (0x%08x)",
			actualDenseDot, math.Float32bits(actualDenseDot), cDot, math.Float32bits(cDot))
	}

	t.Logf("Q4_K: 256/256 weights AND dense dot product are 100%% BIT-EXACT against GCC -O2 C99 oracle!")
}

func TestQ4_K_HostileBlocks_VsCOracle(t *testing.T) {
	tmpDir := t.TempDir()
	cFile := filepath.Join(tmpDir, "q4_k.c")
	binFile := filepath.Join(tmpDir, "q4_k")
	if err := os.WriteFile(cFile, []byte(cQ4_KSource), 0644); err != nil {
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
		raw := make([]byte, 144)
		for j := range raw {
			raw[j] = p.val
		}
		// fp16 scales
		binary.LittleEndian.PutUint16(raw[0:2], 0x3C00) // d = 1.0
		binary.LittleEndian.PutUint16(raw[2:4], 0x3800) // dmin = 0.5

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
			goVal := tensor.DotQ4_K(raw, unitX, 256)
			cVal := cOut[i]
			if math.IsNaN(float64(goVal)) || math.IsNaN(float64(cVal)) {
				t.Fatalf("[%s] NaN detected at %d", p.name, i)
			}
			if math.Float32bits(goVal) != math.Float32bits(cVal) && !(goVal == 0 && cVal == 0) {
				t.Fatalf("[%s] mismatch at %d: Go=%f (0x%08x), C=%f (0x%08x)",
					p.name, i, goVal, math.Float32bits(goVal), cVal, math.Float32bits(cVal))
			}
		}
	}
	t.Logf("Q4_K: Hostile block patterns (0x00, 0xFF, 0xAA, 0x55) validated 100%% bit-exact against GCC -O2!")
}
