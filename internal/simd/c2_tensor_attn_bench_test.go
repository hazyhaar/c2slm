package simd

import "testing"

func BenchmarkAttnHead16k(b *testing.B) {
	const hd = 64
	const pos = 16383
	f := buildAttnFixture(hd, pos, 41)
	scores := make([]float32, pos+1)
	out := make([]float32, hd)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		C2_tensor_attn_qk(f.q, f.kbase, pos, f.stride, f.qkScale, scores, hd)
		C2_tensor_attn_av(scores, f.vbase, pos, f.stride, out, hd)
	}
}
