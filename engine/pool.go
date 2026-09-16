package engine

import (
	"runtime"
	"sync"

	"github.com/hazyhaar/c2slm/gguf"
	"github.com/hazyhaar/c2slm/tensor"
)

// attnParallelThreshold is the context length above which attention heads are
// distributed across the worker pool instead of being computed sequentially.
const attnParallelThreshold = 256

// WorkerPool provides pre-spawned worker goroutines to parallelize GEMV computations with 0 heap allocations
type WorkerPool struct {
	numWorkers int
	startChans []chan struct{}
	doneWg     sync.WaitGroup
	quitChan   chan struct{}

	// Flat task fields (strictly 0 allocation during dispatch)
	y      []float32
	weight []byte
	x      []float32
	q8k    []byte
	rows   int
	cols   int
	qtype  gguf.GGMLType

	// Attention task fields, also flat and 0 allocation during dispatch.
	attnMode    bool
	attnCache   *KVCache
	attnLayer   int
	attnPos     int
	attnScale   float32
	attnNumHead int
	attnQPerKV  int
	attnHeadDim int
	attnQ       []float32
	attnOut     []float32
	attnScratch []float32
}

// NewWorkerPool initializes persistent workers
func NewWorkerPool() *WorkerPool {
	nw := runtime.GOMAXPROCS(0)
	if nw < 1 {
		nw = 1
	}
	if nw > 32 {
		nw = 32
	}

	p := &WorkerPool{
		numWorkers:  nw,
		startChans:  make([]chan struct{}, nw),
		quitChan:    make(chan struct{}),
		q8k:         make([]byte, (16384/256)*tensor.BlockSizeQ8_K),
		attnScratch: make([]float32, nw*MaxContextLen),
	}

	for i := 0; i < nw; i++ {
		p.startChans[i] = make(chan struct{}, 1)
		go p.workerLoop(i)
	}

	return p
}

// Close gracefully terminates worker goroutines
func (p *WorkerPool) Close() {
	select {
	case <-p.quitChan:
		// already closed
	default:
		close(p.quitChan)
	}
}

// NumWorkers returns the number of active workers
func (p *WorkerPool) NumWorkers() int {
	return p.numWorkers
}

func (p *WorkerPool) workerLoop(workerID int) {
	for {
		select {
		case <-p.quitChan:
			return
		case <-p.startChans[workerID]:
			if p.attnMode {
				p.runAttnHeads(workerID)
				p.doneWg.Done()
				continue
			}

			startRow := (workerID * p.rows) / p.numWorkers
			endRow := ((workerID + 1) * p.rows) / p.numWorkers

			if startRow < endRow && startRow < p.rows {
				if endRow > p.rows {
					endRow = p.rows
				}
				switch p.qtype {
				case gguf.GGMLTypeQ8_0:
					tensor.GEMVQ8_0Range(p.y, p.weight, p.x, startRow, endRow, p.cols)
				case gguf.GGMLTypeQ5_0:
					tensor.GEMVQ5_0Range(p.y, p.weight, p.x, startRow, endRow, p.cols)
				case gguf.GGMLTypeQ6_K:
					tensor.GEMVQ6_K_Q8_KRange(p.y, p.weight, p.q8k, startRow, endRow, p.cols)
				case gguf.GGMLTypeQ4_K:
					tensor.GEMVQ4_K_Q8_KRange(p.y, p.weight, p.q8k, startRow, endRow, p.cols)
				}
			}
			p.doneWg.Done()
		}
	}
}

// ParallelGEMV executes GEMV across workers with 0 heap allocation
func (p *WorkerPool) ParallelGEMV(y []float32, weight []byte, x []float32, rows, cols int, qtype gguf.GGMLType) {
	if qtype == gguf.GGMLTypeQ4_K || qtype == gguf.GGMLTypeQ6_K {
		tensor.QuantizeRowQ8_K(x, p.q8k, cols)
	}

	if p.numWorkers <= 1 || rows < 256 {
		// Single-threaded path for small projections (ex: KV heads), avoiding goroutine wakeups
		switch qtype {
		case gguf.GGMLTypeQ8_0:
			tensor.GEMVQ8_0Range(y, weight, x, 0, rows, cols)
		case gguf.GGMLTypeQ5_0:
			tensor.GEMVQ5_0Range(y, weight, x, 0, rows, cols)
		case gguf.GGMLTypeQ6_K:
			tensor.GEMVQ6_K_Q8_KRange(y, weight, p.q8k, 0, rows, cols)
		case gguf.GGMLTypeQ4_K:
			tensor.GEMVQ4_K_Q8_KRange(y, weight, p.q8k, 0, rows, cols)
		}
		return
	}

	p.y = y
	p.weight = weight
	p.x = x
	p.rows = rows
	p.cols = cols
	p.qtype = qtype

	p.doneWg.Add(p.numWorkers)
	for i := 0; i < p.numWorkers; i++ {
		p.startChans[i] <- struct{}{}
	}
	p.doneWg.Wait()
}

// runAttnHeads computes the disjoint head range owned by one worker. Each head
// writes its own slice of out and uses a private scratch region, so no
// synchronization is required inside the worker.
func (p *WorkerPool) runAttnHeads(workerID int) {
	startHead := (workerID * p.attnNumHead) / p.numWorkers
	endHead := ((workerID + 1) * p.attnNumHead) / p.numWorkers
	if startHead >= p.attnNumHead {
		return
	}
	if endHead > p.attnNumHead {
		endHead = p.attnNumHead
	}

	base := workerID * MaxContextLen
	scratch := p.attnScratch[base : base+p.attnPos+1]

	for h := startHead; h < endHead; h++ {
		attentionHead(p.attnCache, p.attnLayer, p.attnPos, p.attnQ, p.attnOut, h, p.attnQPerKV, p.attnHeadDim, p.attnScale, scratch)
	}
}

// ParallelAttention distributes the query heads across the persistent workers
// once the context is long enough to amortize the wake-up cost. Below the
// threshold, or without a usable pool, the heads are computed sequentially.
func (p *WorkerPool) ParallelAttention(scratch []float32, cache *KVCache, layer, pos int, q, out []float32, numHeads, kvHeads, headDim int, scale float32) {
	qPerKV := numHeads / kvHeads

	if p == nil || p.numWorkers <= 1 || numHeads < 2 || pos < attnParallelThreshold {
		for h := 0; h < numHeads; h++ {
			attentionHead(cache, layer, pos, q, out, h, qPerKV, headDim, scale, scratch[:pos+1])
		}
		return
	}

	p.attnCache = cache
	p.attnLayer = layer
	p.attnPos = pos
	p.attnScale = scale
	p.attnNumHead = numHeads
	p.attnQPerKV = qPerKV
	p.attnHeadDim = headDim
	p.attnQ = q
	p.attnOut = out
	p.attnMode = true

	p.doneWg.Add(p.numWorkers)
	for i := 0; i < p.numWorkers; i++ {
		p.startChans[i] <- struct{}{}
	}
	p.doneWg.Wait()

	p.attnMode = false
}
