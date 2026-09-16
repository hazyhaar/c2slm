package tokenizer

import (
	"fmt"
	"regexp"
	"strings"

	"github.com/hazyhaar/c2slm/gguf"
)

// Qwen2 / GPT2 byte-level pre-tokenization regex pattern (Go regexp compatible)
var qwenRegex = regexp.MustCompile(`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+`)

type mergePair struct {
	a, b string
}

// Tokenizer implements a byte-level BPE tokenizer for Qwen2
type Tokenizer struct {
	tokenToID     map[string]int32
	idToToken     []string
	merges        map[mergePair]int
	specialTokens map[string]int32
	byteToUnicode map[byte]rune
	unicodeToByte map[rune]byte
}

// NewTokenizerFromGGUF loads and initializes the BPE tokenizer from GGUF metadata
func NewTokenizerFromGGUF(gf *gguf.File) (*Tokenizer, error) {
	tokens, ok := gf.GetStringSlice("tokenizer.ggml.tokens")
	if !ok || len(tokens) == 0 {
		return nil, fmt.Errorf("tokenizer.ggml.tokens missing or empty")
	}

	mergesList, ok := gf.GetStringSlice("tokenizer.ggml.merges")
	if !ok {
		mergesList = nil
	}

	tok := &Tokenizer{
		tokenToID:     make(map[string]int32, len(tokens)),
		idToToken:     make([]string, len(tokens)),
		merges:        make(map[mergePair]int, len(mergesList)),
		specialTokens: make(map[string]int32),
		byteToUnicode: make(map[byte]rune, 256),
		unicodeToByte: make(map[rune]byte, 256),
	}

	initByteEncodings(tok.byteToUnicode, tok.unicodeToByte)

	for id, s := range tokens {
		id32 := int32(id)
		tok.tokenToID[s] = id32
		tok.idToToken[id] = s

		// Index special tokens (like <|im_start|>, <|im_end|>, etc.)
		if strings.HasPrefix(s, "<|") && strings.HasSuffix(s, "|>") {
			tok.specialTokens[s] = id32
		}
	}

	// Register specific standard Qwen special tokens if present
	specialNames := []string{
		"<|im_start|>",
		"<|im_end|>",
		"<|endoftext|>",
	}
	for _, name := range specialNames {
		if id, exists := tok.tokenToID[name]; exists {
			tok.specialTokens[name] = id
		}
	}

	for rank, m := range mergesList {
		parts := strings.Split(m, " ")
		if len(parts) == 2 {
			tok.merges[mergePair{parts[0], parts[1]}] = rank
		}
	}

	return tok, nil
}

// Encode converts a text string into a slice of token IDs
func (tok *Tokenizer) Encode(text string) []int32 {
	if text == "" {
		return nil
	}

	// Handle special tokens by splitting text into special and non-special segments
	var tokenIDs []int32
	remainder := text

	for len(remainder) > 0 {
		// Find earliest special token
		earliestIdx := -1
		earliestSpecial := ""
		for st := range tok.specialTokens {
			idx := strings.Index(remainder, st)
			if idx != -1 && (earliestIdx == -1 || idx < earliestIdx) {
				earliestIdx = idx
				earliestSpecial = st
			}
		}

		if earliestIdx == -1 {
			// No more special tokens in remainder
			tokenIDs = append(tokenIDs, tok.encodeTextChunk(remainder)...)
			break
		}

		if earliestIdx > 0 {
			// Encode regular text before the special token
			tokenIDs = append(tokenIDs, tok.encodeTextChunk(remainder[:earliestIdx])...)
		}

		// Append special token ID
		tokenIDs = append(tokenIDs, tok.specialTokens[earliestSpecial])
		remainder = remainder[earliestIdx+len(earliestSpecial):]
	}

	return tokenIDs
}

func (tok *Tokenizer) encodeTextChunk(chunk string) []int32 {
	var ids []int32
	matches := qwenRegex.FindAllString(chunk, -1)
	for _, match := range matches {
		encodedBytes := tok.bytesToUnicodeString([]byte(match))
		bpeTokens := tok.bpe(encodedBytes)
		for _, bpeTok := range bpeTokens {
			if id, ok := tok.tokenToID[bpeTok]; ok {
				ids = append(ids, id)
			}
		}
	}
	return ids
}

func (tok *Tokenizer) bpe(token string) []string {
	var word []string
	for _, r := range token {
		word = append(word, string(r))
	}
	if len(word) <= 1 {
		return word
	}

	for {
		minRank := -1
		var bestPair mergePair
		bestIdx := -1

		for i := 0; i < len(word)-1; i++ {
			pair := mergePair{word[i], word[i+1]}
			if rank, ok := tok.merges[pair]; ok {
				if minRank == -1 || rank < minRank {
					minRank = rank
					bestPair = pair
					bestIdx = i
				}
			}
		}

		if minRank == -1 || bestIdx == -1 {
			break
		}

		// Merge the best pair
		var newWord []string
		i := 0
		for i < len(word) {
			if i < len(word)-1 && word[i] == bestPair.a && word[i+1] == bestPair.b {
				newWord = append(newWord, bestPair.a+bestPair.b)
				i += 2
			} else {
				newWord = append(newWord, word[i])
				i++
			}
		}
		word = newWord
		if len(word) <= 1 {
			break
		}
	}

	return word
}

// Decode converts token IDs back into string text
func (tok *Tokenizer) Decode(ids []int32) string {
	var sb strings.Builder
	var byteBuf []byte

	for _, id := range ids {
		if id < 0 || int(id) >= len(tok.idToToken) {
			continue
		}
		s := tok.idToToken[id]
		if _, isSpecial := tok.specialTokens[s]; isSpecial {
			// Flush bytes if any
			if len(byteBuf) > 0 {
				sb.WriteString(string(byteBuf))
				byteBuf = byteBuf[:0]
			}
			sb.WriteString(s)
			continue
		}

		for _, r := range s {
			if b, ok := tok.unicodeToByte[r]; ok {
				byteBuf = append(byteBuf, b)
			} else {
				// Regular unicode rune
				if len(byteBuf) > 0 {
					sb.WriteString(string(byteBuf))
					byteBuf = byteBuf[:0]
				}
				sb.WriteRune(r)
			}
		}
	}

	if len(byteBuf) > 0 {
		sb.WriteString(string(byteBuf))
	}

	return sb.String()
}

func (tok *Tokenizer) bytesToUnicodeString(bytes []byte) string {
	var sb strings.Builder
	for _, b := range bytes {
		if r, ok := tok.byteToUnicode[b]; ok {
			sb.WriteRune(r)
		} else {
			sb.WriteRune(rune(b))
		}
	}
	return sb.String()
}

// initByteEncodings initializes standard GPT-2 byte to unicode mapping table
func initByteEncodings(b2u map[byte]rune, u2b map[rune]byte) {
	var bs []int
	for b := int('!'); b <= int('~'); b++ {
		bs = append(bs, b)
	}
	for b := int('¡'); b <= int('¬'); b++ {
		bs = append(bs, b)
	}
	for b := int('®'); b <= int('ÿ'); b++ {
		bs = append(bs, b)
	}

	cs := make([]int, len(bs))
	copy(cs, bs)

	n := 0
	for b := 0; b < 256; b++ {
		found := false
		for _, x := range bs {
			if x == b {
				found = true
				break
			}
		}
		if !found {
			bs = append(bs, b)
			cs = append(cs, 256+n)
			n++
		}
	}

	for i := 0; i < len(bs); i++ {
		b := byte(bs[i])
		r := rune(cs[i])
		b2u[b] = r
		u2b[r] = b
	}
}

// SpecialTokenIDs provides quick access to common control token IDs
func (tok *Tokenizer) SpecialTokenIDs() (imStart, imEnd, eos int32) {
	imStart = tok.specialTokens["<|im_start|>"]
	imEnd = tok.specialTokens["<|im_end|>"]
	eos = tok.specialTokens["<|endoftext|>"]
	return
}
