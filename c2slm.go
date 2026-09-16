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
	if e.KVCache != nil {
		_ = e.KVCache.Close()
	}
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

// GenerateStream runs autoregressive inference with a token streaming callback.
// It resets the KV cache and ingests the whole prompt before generation.
func (e *Engine) GenerateStream(prompt string, maxNewTokens int, stopStrings []string, onToken func(piece string) bool) (string, error) {
	promptTokens := e.Tokenizer.Encode(prompt)
	if len(promptTokens) == 0 {
		return "", fmt.Errorf("empty prompt tokens")
	}
	if len(promptTokens) >= engine.MaxContextLen {
		return "", fmt.Errorf("prompt (%d) exceeds max context (%d)",
			len(promptTokens), engine.MaxContextLen)
	}

	// Reset KV cache for clean inference sequence
	e.KVCache.Reset()

	lastNormX := e.IngestFrom(0, promptTokens)
	if remaining := e.KVCache.MaxTokens - e.KVCache.SeqLen; maxNewTokens > remaining {
		maxNewTokens = remaining
	}
	return e.generate(lastNormX, e.KVCache.SeqLen, maxNewTokens, stopStrings, onToken)
}

// IngestFrom feeds tokens into the KV cache starting at position pos, running
// engine.Forward only on the tokens that are not already materialized. When pos
// is behind the current cursor the cache is rewound in place (no memory copy);
// when pos is ahead it is clamped back to the cursor, since a gap cannot be
// attended to causally. The token fingerprint is recorded for every ingested
// position and SeqLen advances accordingly. The returned slice is the
// normalized hidden state of the last ingested token, aliasing the engine
// arena and therefore valid only until the next Forward call.
func (e *Engine) IngestFrom(pos int, tokens []int32) []float32 {
	if len(tokens) == 0 {
		return nil
	}
	if pos < 0 {
		pos = 0
	}
	if pos > e.KVCache.SeqLen {
		pos = e.KVCache.SeqLen
	}
	if pos < e.KVCache.SeqLen {
		e.KVCache.Rewind(e.KVCache.SeqLen - pos)
	}

	var lastNormX []float32
	for i, tokID := range tokens {
		p := pos + i
		if p >= e.KVCache.MaxTokens {
			break
		}
		lastNormX = engine.Forward(e.Model, e.KVCache, e.Arena, tokID, p)
		e.KVCache.StoreTokenID(p, tokID)
		e.KVCache.SeqLen = p + 1
	}
	return lastNormX
}

// GenerateStreamSession appends newPromptTokens to the ongoing conversation and
// generates a reply without ever recomputing the cached prefix. The KV cache is
// not reset, so the system prompt and the earlier turns (including any  thinking
// block) remain causally available. maxNewTokens is clamped to the remaining
// capacity MaxTokens - SeqLen.
func (e *Engine) GenerateStreamSession(newPromptTokens []int32, maxNewTokens int, stopStrings []string, onToken func(piece string) bool) (string, error) {
	if len(newPromptTokens) == 0 {
		return "", fmt.Errorf("empty prompt tokens")
	}
	if e.KVCache.SeqLen+len(newPromptTokens) >= engine.MaxContextLen {
		return "", fmt.Errorf("session prompt (%d) exceeds max context (%d)",
			e.KVCache.SeqLen+len(newPromptTokens), engine.MaxContextLen)
	}

	lastNormX := e.IngestFrom(e.KVCache.SeqLen, newPromptTokens)
	if remaining := e.KVCache.MaxTokens - e.KVCache.SeqLen; maxNewTokens > remaining {
		maxNewTokens = remaining
	}
	return e.generate(lastNormX, e.KVCache.SeqLen, maxNewTokens, stopStrings, onToken)
}

// generate is the shared autoregressive loop: it projects the logits from the
// last normalized hidden state, selects greedily, streams the decoded piece and
// forwards the accepted token at currentPos so the cache stays consistent for
// the next session turn.
func (e *Engine) generate(lastNormX []float32, currentPos, maxNewTokens int, stopStrings []string, onToken func(piece string) bool) (string, error) {
	_, imEnd, eos := e.Tokenizer.SpecialTokenIDs()

	var generatedText strings.Builder
	for step := 0; step < maxNewTokens; step++ {
		if currentPos >= e.KVCache.MaxTokens {
			break
		}

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
			_ = engine.Forward(e.Model, e.KVCache, e.Arena, bestID, currentPos)
			e.KVCache.StoreTokenID(currentPos, bestID)
			currentPos++
			e.KVCache.SeqLen = currentPos
			break
		}

		piece := e.Tokenizer.Decode([]int32{bestID})
		generatedText.WriteString(piece)

		if onToken != nil {
			if !onToken(piece) {
				// Persister le jeton courant dans le KV-cache avant l'arrêt
				lastNormX = engine.Forward(e.Model, e.KVCache, e.Arena, bestID, currentPos)
				e.KVCache.StoreTokenID(currentPos, bestID)
				currentPos++
				e.KVCache.SeqLen = currentPos
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

		// Forward newly generated token, extending the persisted cache
		lastNormX = engine.Forward(e.Model, e.KVCache, e.Arena, bestID, currentPos)
		e.KVCache.StoreTokenID(currentPos, bestID)
		currentPos++
		e.KVCache.SeqLen = currentPos
	}

	return generatedText.String(), nil
}
