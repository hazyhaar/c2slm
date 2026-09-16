//go:build !goexperiment.simd || !amd64

package simd

// SimdEnabled indique que la compilation courante active le repli scalaire nosimd.
const SimdEnabled = false
