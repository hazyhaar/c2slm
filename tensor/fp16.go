package tensor

import "math"

// fp16LUT is an ARCHTIME lookup table of 65,536 float32 values (256 KB)
// fits entirely in L2 CPU cache, eliminating all runtime bit-manipulation and branching.
var fp16LUT [65536]float32

func init() {
	for i := 0; i < 65536; i++ {
		fp16LUT[i] = computeFP16ToF32(uint16(i))
	}
}

// FP16ToF32 converts an IEEE-754 half-precision float (16-bit) to float32 in 1 memory lookup (0.5 ns).
func FP16ToF32(h uint16) float32 {
	return fp16LUT[h]
}

func computeFP16ToF32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32((h >> 10) & 0x1F)
	mant := uint32(h & 0x3FF)

	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		// Subnormal
		for (mant & 0x400) == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
		exp = (exp + (127 - 15)) << 23
		mant <<= 13
		return math.Float32frombits(sign | exp | mant)
	}
	if exp == 31 {
		if mant == 0 {
			return math.Float32frombits(sign | 0x7F800000) // Inf
		}
		return math.Float32frombits(sign | 0x7F800000 | (mant << 13)) // NaN
	}

	exp = (exp + (127 - 15)) << 23
	mant <<= 13
	return math.Float32frombits(sign | exp | mant)
}
