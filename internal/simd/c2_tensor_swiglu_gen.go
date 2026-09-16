//go:build goexperiment.simd && amd64

package simd

import (
	"math"
	"simd/archsimd"
)

// C2_tensor_swiglu computes out[i] = SiLU(gate[i]) * up[i] using vectorized AVX2 lanes.
func C2_tensor_swiglu(out, gate, up []float32, n int) {
	if !archsimd.X86.AVX2() {
		C2_tensor_swiglu_nosimd(out, gate, up, n)
		return
	}

	v123 := archsimd.BroadcastFloat32x8(math.Float32frombits(0x3f000000)) // 0.5
	v124 := archsimd.BroadcastFloat32x8(math.Float32frombits(0x3f800000)) // 1.0
	v104 := archsimd.BroadcastInt32x8(int32(127))
	vInv := archsimd.BroadcastFloat32x8(math.Float32frombits(0x3fb8aa3b)) // 1/ln2
	vLn2 := archsimd.BroadcastFloat32x8(math.Float32frombits(0x3f317218)) // ln2
	vC5 := archsimd.BroadcastFloat32x8(math.Float32frombits(0x3c088888))
	vC4 := archsimd.BroadcastFloat32x8(math.Float32frombits(0x3d2aaaab))
	vC3 := archsimd.BroadcastFloat32x8(math.Float32frombits(0x3e2aaaab))
	vZero := archsimd.BroadcastFloat32x8(0)
	vMin := archsimd.BroadcastInt32x8(int32(-126))

	i := 0
	for ; i+8 <= n; i += 8 {
		xv := archsimd.LoadFloat32x8(gate[i:])
		nx := xv.Neg()
		nf := nx.Mul(vInv)
		nPos := nf.Add(v123).ConvertToInt32()
		nNeg := nf.Sub(v123).ConvertToInt32()
		mask := nf.GreaterEqual(vZero)
		ni := nPos.IfElse(mask, nNeg).Max(vMin).Min(v104)
		niF := ni.ConvertToFloat32()
		r := nx.Sub(niF.Mul(vLn2))

		p := vC5.Mul(r).Add(vC4).Mul(r).Add(vC3).Mul(r).Add(v123).Mul(r).Add(v124).Mul(r).Add(v124)
		bits := ni.Add(v104).ShiftAllLeft(23)
		scale := bits.AsFloat32x8()
		ev := p.Mul(scale)
		den := v124.Add(ev)
		sig := v124.Div(den)
		silu := xv.Mul(sig)

		u := archsimd.LoadFloat32x8(up[i:])
		res := silu.Mul(u)
		res.Store(out[i:])
	}

	for ; i < n; i++ {
		g := float64(gate[i])
		silu := float32(g / (1.0 + math.Exp(-g)))
		out[i] = silu * up[i]
	}
}

// C2_tensor_swiglu_nosimd is the scalar fallback.
func C2_tensor_swiglu_nosimd(out, gate, up []float32, n int) {
	for i := 0; i < n; i++ {
		g := float64(gate[i])
		silu := float32(g / (1.0 + math.Exp(-g)))
		out[i] = silu * up[i]
	}
}
