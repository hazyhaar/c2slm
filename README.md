# c2slm / nanogowen

[![Go Version](https://img.shields.io/badge/go-1.27+-00ADD8?style=flat&logo=go)](https://go.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-success.svg)](#)
[![Zero CGo](https://img.shields.io/badge/CGo-0%25-brightgreen.svg)](#)
[![Parity](https://img.shields.io/badge/oracle%20parity-100%25%20bit--exact-orange.svg)](#)

[🇫🇷 Documentation en français disponible ici](README.fr.md)

**c2slm** (and its companion CLI **nanogowen**) is a standalone, high-performance, **100% pure Go** inference engine for Small Language Models (SLMs), specifically tailored and optimized for **Qwen2.5** and **Qwen2** architectures using GGUF quantization (reference model: `Qwen2.5-0.5B-Instruct`).

---

## 1. Key Architectural Features

- **Zero External Dependencies:** Built strictly with standard Go (`go 1.27+`). No C++, CGo, Wasm, Python runtime, or shared libraries. The `go.mod` file has zero external module requirements.
- **Pure AVX2 Vector Micro-Kernels:** Mechanically generated from C99 algorithms via the `sgoiter` compiler using Go standard package `simd/archsimd` (`GOEXPERIMENT=simd`), with automatic runtime fallback to scalar Go on non-AVX2 hardware.
- **Vectorized Integer Arithmetic for Q4_K x Q8_K:**
  - Dynamic single-pass activation quantization ($F32 \rightarrow Q8\_K$) per projection vector.
  - Pure integer dot product leveraging `VPMADDUBSW` (`DotProductPairsSaturated`) and `VPMADDWD` (`DotProductPairs`), completely bypassing floating-point unpacking and YMM register pressure.
  - 100% bit-exact parity formally validated against GCC 13 `-O2` C99 binary oracles.
- **Zero Heap Allocations on Forward Pass (0 B/op):**
  - Static memory arena (`Arena`) and persistent 32-worker thread pool (`WorkerPool`).
  - Strict zero-allocation execution during autoregressive generation.
- **Native GGUF v2 / v3 Loader:**
  - Zero-copy direct memory-mapping (`mmap`) of weight files.
  - Supported quantizations: `Q4_K` (accelerated integer path), `Q8_0`, `Q5_0`, `Q6_K`, and `F32`.
- **Integrated Pure-Go ChatML Tokenizer & Interactive TUI:**
  - Fast regex-based token splitting with full 151,936-token BPE vocabulary support and special token handling.
  - Dedicated zero-dependency VT100/ANSI terminal user interface (`DECSTBM` scrolling region, anchored bottom input prompt, dynamic Unicode rune width rendering, instant Ctrl+C interruption preserving the KV-cache).
  - Dynamic repetition penalty filtering on recent logits to avoid autoregressive generation loops.

---

## 2. Physical Silicon Benchmarks

Measured on physical hardware (**Intel Core i9-14900K**, 24 cores / 32 threads, Linux 6.8, Go 1.27rc3):

| Component / Benchmark | Implementation | Latency / Throughput | Heap Allocations |
| :--- | :--- | :--- | :--- |
| **DotProduct Q4_K (896 cols)** | AVX2 Integer (`DotQ4_K_Q8_K`) | **1,111 ns/op** (2.0× faster than scalar) | **0 B/op, 0 allocs/op** |
| **DotProduct Q8_0 (32 cols)** | AVX2 Float (`DotQ8_0`) | **149 ns/op** (3.4× faster than scalar) | **0 B/op, 0 allocs/op** |
| **Dynamic Q8_K Quantization** | `QuantizeRowQ8_K` (896 cols) | **1,849 ns/op (1.94 GB/s)** | **0 B/op, 0 allocs/op** |
| **Vocabulary Projection (151k)** | `ComputeLogits` (32 workers) | **2.91 ms** | **0 B/op, 0 allocs/op** |
| **Complete Forward Pass (24 layers)** | `Forward` (1 token) | **38.6 ms** | **0 B/op, 0 allocs/op** |
| **End-to-End Generation (`nanogowen`)** | Qwen2.5-0.5B-Instruct-Q4_K_M | **3.79 tokens / second** | **Zero alloc continuous stream** |

---

## 3. Quickstart & CLI Usage

### Building `nanogowen`

```bash
# Build with SIMD intrinsics enabled
GOEXPERIMENT=simd go build -o bin/nanogowen ./cmd/nanogowen
```

### Running Inference

```bash
# Single prompt mode (one-shot batch execution):
./bin/nanogowen -model /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf -m "Explain Claude Shannon's information entropy."

# Interactive Full-Screen TUI Mode (anchored prompt, ANSI scrolling region):
./bin/nanogowen -model /path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### Interactive TUI Commands

While running in interactive mode, the following commands are available:
- `/stats` or `/context`: Inspect live KV-cache capacity and sequence length occupancy.
- `/clear` or `/reset`: Reset the KV cache and re-anchor the system prompt.
- `/help`: Display keyboard shortcuts and built-in commands.
- `Ctrl+C`: Abort ongoing assistant generation immediately without losing conversation history.
- `/exit` or `/quit`: Gracefully exit the session.

---

## 4. Go Library Usage

Add `github.com/hazyhaar/c2slm` to your Go project:

```go
package main

import (
	"fmt"
	"log"

	"github.com/hazyhaar/c2slm"
)

func main() {
	// Initialize engine with GGUF model path
	engine, err := c2slm.NewEngine("/path/to/qwen2.5-0.5b-instruct-q4_k_m.gguf")
	if err != nil {
		log.Fatalf("failed to initialize engine: %v", err)
	}
	defer engine.Close()

	// Format prompt using ChatML template
	prompt := "<|im_start|>user\nWhat is the speed of light in vacuum?<|im_end|>\n<|im_start|>assistant\n"
	tokens := engine.Tokenizer.Encode(prompt)

	fmt.Println("Assistant:")
	// Stream tokens with early termination support
	_ = engine.Generate(tokens, 128, []string{"<|im_end|>"}, func(token string) bool {
		fmt.Print(token)
		return true // continue streaming
	})
	fmt.Println()
}
```

---

## 5. Repository Structure

```
c2slm/
├── c2slm.go          # High-level public API (Engine, Generate, NewEngine)
├── c2slm_test.go     # End-to-end integration and generation tests
├── go.mod            # Self-contained Go module (zero external dependencies)
├── cmd/
│   └── nanogowen/    # Interactive REPL and streaming CLI executable
├── engine/           # Inference pipeline (Model, Forward, KVCache, WorkerPool, Arena)
├── gguf/             # Zero-copy binary GGUF v2/v3 parser (mmap)
├── internal/
│   └── simd/         # AVX2 vector micro-kernels and scalar fallbacks
├── tensor/           # Tensor math ops (RMSNorm, RoPE NeoX, SwiGLU, Softmax, GEMV)
└── tokenizer/        # Pure Go ChatML Byte-Pair Encoding (BPE) tokenizer
```

---

## 6. Testing & Parity Verification

Run the comprehensive test suite with race detection and SIMD instructions:

```bash
GOEXPERIMENT=simd go test -race -count=1 ./...
```

To run SIMD benchmarks:

```bash
GOEXPERIMENT=simd go test -bench=. -benchmem ./internal/simd/
```

---

## 7. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
