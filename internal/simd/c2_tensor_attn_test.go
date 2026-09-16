package simd

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"testing"
)

type attnFixture struct {
	hd       int
	pos      int
	seed     uint32
	stride   int
	q        []float32
	kbase    []float32
	vbase    []float32
	avScores []float32
	qkScale  float32
	qkGo     []float32
	avGo     []float32
}

func buildAttnFixture(hd, pos int, seed uint32) *attnFixture {
	stride := hd
	nk := (pos + 1) * stride
	state := seed
	next := func() float32 {
		state = state*1664525 + 1013904223
		return float32((state>>8)&0xFFFFFF) / float32(0x1000000)
	}
	nextSigned := func() float32 { return next()*2 - 1 }

	f := &attnFixture{
		hd:       hd,
		pos:      pos,
		seed:     seed,
		stride:   stride,
		q:        make([]float32, hd),
		kbase:    make([]float32, nk),
		vbase:    make([]float32, nk),
		avScores: make([]float32, pos+1),
		qkScale:  0.125,
	}
	for i := 0; i < hd; i++ {
		f.q[i] = nextSigned()
	}
	for i := 0; i < nk; i++ {
		f.kbase[i] = nextSigned()
	}
	for i := 0; i < nk; i++ {
		f.vbase[i] = 0.25 + 0.5*next()
	}

	f.qkGo = make([]float32, pos+1)
	C2_tensor_attn_qk(f.q, f.kbase, f.pos, f.stride, f.qkScale, f.qkGo, f.hd)

	for i := 0; i <= pos; i++ {
		f.avScores[i] = 0.25 + 0.5*next()
	}
	f.avGo = make([]float32, hd)
	C2_tensor_attn_av(f.avScores, f.vbase, f.pos, f.stride, f.avGo, f.hd)
	return f
}

func locateAttnSources(t *testing.T) (srcDir, oracleSrc, cSrc string) {
	t.Helper()
	for _, c := range []string{
		filepath.Join("..", "..", "c2simd", "sources", "c2archsimd"),
		filepath.Join("sources", "c2archsimd"),
		filepath.Join("/devhoros", "c2simd", "sources", "c2archsimd"),
	} {
		if _, err := os.Stat(filepath.Join(c, "test_attn_oracle.c")); err == nil {
			return c, filepath.Join(c, "test_attn_oracle.c"), filepath.Join(c, "c2_tensor_attn.c")
		}
	}
	t.Fatalf("CRITICAL GATE FAILURE: oracle C source introuvable")
	return "", "", ""
}

func compileAttnOracle(t *testing.T) string {
	t.Helper()
	srcDir, oracleSrc, cSrc := locateAttnSources(t)
	tmpBin := filepath.Join(t.TempDir(), "test_attn_oracle")
	args := []string{
		"-O2", "-mavx2", "-DNO_AVX2", "-fsanitize=address,undefined",
		"-I", srcDir, oracleSrc, cSrc, "-lm", "-o", tmpBin,
	}
	if out, err := exec.Command("gcc", args...).CombinedOutput(); err != nil {
		t.Fatalf("gcc compile failed: %v\n%s", err, string(out))
	}
	return tmpBin
}

func readFloat32s(t *testing.T, path string, n int) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read oracle output: %v", err)
	}
	if len(raw) != n*4 {
		t.Fatalf("oracle output size %d, want %d", len(raw), n*4)
	}
	out := make([]float32, n)
	if err := binary.Read(bytes.NewReader(raw), binary.LittleEndian, out); err != nil {
		t.Fatalf("decode oracle float32: %v", err)
	}
	return out
}

func maxRelDiff(got, want []float32) (float64, int) {
	maxDiff := 0.0
	maxIdx := -1
	for i := range got {
		ref := math.Abs(float64(want[i]))
		if ref < 1 {
			ref = 1
		}
		d := math.Abs(float64(got[i]-want[i])) / ref
		if d > maxDiff {
			maxDiff = d
			maxIdx = i
		}
	}
	return maxDiff, maxIdx
}

func TestAttnVsCOracle(t *testing.T) {
	bin := compileAttnOracle(t)
	const relTol = 1e-5

	cases := []struct {
		hd, pos int
		seed    uint32
	}{
		{64, 0, 7},
		{64, 1, 7},
		{64, 63, 11},
		{64, 255, 13},
		{64, 256, 17},
		{64, 2048, 19},
		{64, 16383, 31},
		{128, 512, 23},
		{128, 128, 29},
		{128, 8192, 37},
	}

	for _, tc := range cases {
		t.Run(strconv.Itoa(tc.hd)+"_"+strconv.Itoa(tc.pos), func(t *testing.T) {
			f := buildAttnFixture(tc.hd, tc.pos, tc.seed)

			outBin := filepath.Join(t.TempDir(), "oracle_out.bin")
			cmd := exec.Command(bin, outBin, strconv.Itoa(tc.hd), strconv.Itoa(tc.pos), strconv.FormatUint(uint64(tc.seed), 10))
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("oracle run failed: %v\n%s", err, string(out))
			}

			all := readFloat32s(t, outBin, tc.pos+1+tc.hd)
			oracleScores := all[:tc.pos+1]
			oracleOut := all[tc.pos+1:]

			dScores, idxS := maxRelDiff(f.qkGo, oracleScores)
			if dScores > relTol {
				t.Fatalf("QK rel diff %e > %e at p=%d (go=%g oracle=%g)", dScores, relTol, idxS, f.qkGo[idxS], oracleScores[idxS])
			}
			dOut, idxO := maxRelDiff(f.avGo, oracleOut)
			if dOut > relTol {
				t.Fatalf("AV rel diff %e > %e at d=%d (go=%g oracle=%g)", dOut, relTol, idxO, f.avGo[idxO], oracleOut[idxO])
			}
			t.Logf("hd=%d pos=%d QK rel=%e AV rel=%e [%s]", tc.hd, tc.pos, dScores, dOut, simdMode())
		})
	}
}

func TestAttnZeroAllocation(t *testing.T) {
	f := buildAttnFixture(64, 2048, 3)
	scores := make([]float32, f.pos+1)
	out := make([]float32, f.hd)

	allocsQK := testing.AllocsPerRun(20, func() {
		C2_tensor_attn_qk(f.q, f.kbase, f.pos, f.stride, f.qkScale, scores, f.hd)
	})
	if allocsQK != 0 {
		t.Errorf("AttnQK allocated %v objects, want 0", allocsQK)
	}

	allocsAV := testing.AllocsPerRun(20, func() {
		C2_tensor_attn_av(scores, f.vbase, f.pos, f.stride, out, f.hd)
	})
	if allocsAV != 0 {
		t.Errorf("AttnAV allocated %v objects, want 0", allocsAV)
	}
}

func simdMode() string {
	if SimdEnabled {
		return "AVX2 SIMD"
	}
	return "SCALAR FALLBACK"
}
