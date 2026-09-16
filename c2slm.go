package c2slm

import (
	"fmt"
	"strings"

	"github.com/hazyhaar/c2slm/engine"
	"github.com/hazyhaar/c2slm/tokenizer"
)

// Engine is the standalone pure Go SLM inference engine
type Engine struct {
	Model     *engine.Model
	Tokenizer *tokenizer.Tokenizer
	KVCache   *engine.KVCache
	Arena     *engine.Arena
}

// NewEngine initializes the inference engine from a GGUF file path
func NewEngine(modelPath string) (*Engine, error) {
	m, err := engine.LoadModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}

	tok, err := tokenizer.NewTokenizerFromGGUF(m.GGUFFile)
	if err != nil {
		_ = m.Close()
		return nil, fmt.Errorf("load tokenizer: %w", err)
	}

	cache := engine.NewKVCache(m.NumLayers, m.NumKVHeads, m.HeadDim, engine.MaxContextLen)
	arena := engine.NewArena(m)

	return &Engine{
		Model:     m,
		Tokenizer: tok,
		KVCache:   cache,
		Arena:     arena,
	}, nil
}

// Close frees memory-mapped resources and background workers
func (e *Engine) Close() error {
	if e.Arena != nil {
		e.Arena.Close()
	}
	if e.Model != nil {
		return e.Model.Close()
	}
	return nil
}

// Generate runs autoregressive inference for a given text prompt
func (e *Engine) Generate(prompt string, maxNewTokens int, stopStrings []string) (string, error) {
	return e.GenerateStream(prompt, maxNewTokens, stopStrings, nil)
}

// GenerateStream runs autoregressive inference with a token streaming callback
func (e *Engine) GenerateStream(prompt string, maxNewTokens int, stopStrings []string, onToken func(piece string) bool) (string, error) {
	promptTokens := e.Tokenizer.Encode(prompt)
	if len(promptTokens) == 0 {
		return "", fmt.Errorf("empty prompt tokens")
	}
	if len(promptTokens) >= engine.MaxContextLen {
		return "", fmt.Errorf("prompt (%d) exceeds max context (%d)",
			len(promptTokens), engine.MaxContextLen)
	}
	if len(promptTokens)+maxNewTokens > engine.MaxContextLen {
		maxNewTokens = engine.MaxContextLen - len(promptTokens)
	}

	// Reset KV cache for clean inference sequence
	e.KVCache.Reset()

	// 1. Ingest prompt tokens (prefill)
	var lastNormX []float32
	for pos, tokID := range promptTokens {
		lastNormX = engine.Forward(e.Model, e.KVCache, e.Arena, tokID, pos)
	}

	_, imEnd, eos := e.Tokenizer.SpecialTokenIDs()

	var generatedTokens []int32
	var generatedText strings.Builder
	currentPos := len(promptTokens)

	// 2. Autoregressive generation loop
	for step := 0; step < maxNewTokens; step++ {
		// Project logits for current state
		engine.ComputeLogits(e.Model, e.Arena, lastNormX, e.Arena.Logits, nil)

		// Greedy argmax selection
		bestID := int32(0)
		bestLogit := e.Arena.Logits[0]
		for id := 1; id < e.Model.VocabSize; id++ {
			if e.Arena.Logits[id] > bestLogit {
				bestLogit = e.Arena.Logits[id]
				bestID = int32(id)
			}
		}

		// Check end of sequence
		if bestID == imEnd || bestID == eos {
			break
		}

		generatedTokens = append(generatedTokens, bestID)
		piece := e.Tokenizer.Decode([]int32{bestID})
		generatedText.WriteString(piece)

		if onToken != nil {
			if !onToken(piece) {
				break
			}
		}

		// Check stop strings
		stopped := false
		curStr := generatedText.String()
		for _, s := range stopStrings {
			if strings.Contains(curStr, s) {
				stopped = true
				break
			}
		}
		if stopped {
			break
		}

		// Forward newly generated token
		lastNormX = engine.Forward(e.Model, e.KVCache, e.Arena, bestID, currentPos)
		currentPos++
	}

	return generatedText.String(), nil
}
