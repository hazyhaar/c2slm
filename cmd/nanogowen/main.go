package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/hazyhaar/c2slm"
)

const defaultModelPath = "/data/models/qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf"
const defaultSystemPrompt = "Tu es nanoGOqwen, un modèle de langage compact propulsé par un inféreur 100% pur Go sans runtime CGo ni Wasm. Réponds en français de manière claire et concise."

func main() {
	modelPath := flag.String("model", defaultModelPath, "Chemin vers le fichier de poids GGUF")
	promptMsg := flag.String("m", "", "Message ponctuel en ligne de commande (mode non-interactif)")
	systemPrompt := flag.String("system", defaultSystemPrompt, "Consigne système ChatML")
	maxTokens := flag.Int("max-tokens", 512, "Nombre maximal de nouveaux jetons à générer")
	flag.Parse()

	if _, err := os.Stat(*modelPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Erreur: fichier de modèle introuvable: %s\n", *modelPath)
		os.Exit(1)
	}

	fmt.Printf("Chargement du modèle pur Go (%s)...\n", *modelPath)
	t0 := time.Now()
	engine, err := c2slm.NewEngine(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Erreur lors du chargement du modèle: %v\n", err)
		os.Exit(1)
	}
	defer engine.Close()
	loadDuration := time.Since(t0)

	fmt.Printf("Modèle prêt en %v (mmap sans copie, %d couches, vocabulaire %d jetons).\n\n",
		loadDuration, engine.Model.NumLayers, engine.Model.VocabSize)

	stopStrings := []string{"<|im_end|>", "<|endoftext|>"}

	if *promptMsg != "" {
		runSinglePrompt(engine, *systemPrompt, *promptMsg, *maxTokens, stopStrings)
		return
	}

	runInteractiveREPL(engine, *systemPrompt, *maxTokens, stopStrings)
}

func formatChatML(system, user string) string {
	var b strings.Builder
	b.WriteString("<|im_start|>system\n")
	b.WriteString(system)
	b.WriteString("<|im_end|>\n<|im_start|>user\n")
	b.WriteString(user)
	b.WriteString("<|im_end|>\n<|im_start|>assistant\n")
	return b.String()
}

// formatSystemPrefix is the unique, immutable ChatML prefix materialized once
// at session start and never recomputed afterwards.
func formatSystemPrefix(system string, reg *c2slm.ToolRegistry) string {
	var b strings.Builder
	b.WriteString("<|im_start|>system\n")
	b.WriteString(system)
	if reg != nil && reg.Len() > 0 {
		b.WriteString("\n\n")
		b.WriteString(reg.SystemToolsBlock())
	}
	b.WriteString("<|im_end|>\n")
	return b.String()
}

// formatUserTurn is the per-turn delta: only the new user message and the
// assistant header are appended to the ongoing KV cache.
func formatUserTurn(user string) string {
	return "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
}

// prefillSystem materializes the system prompt in the KV cache from position 0.
func prefillSystem(engine *c2slm.Engine, system string, reg *c2slm.ToolRegistry) {
	tokens := engine.Tokenizer.Encode(formatSystemPrefix(system, reg))
	engine.IngestFrom(0, tokens)
}

func runSinglePrompt(engine *c2slm.Engine, system, user string, maxTokens int, stopStrings []string) {
	fullPrompt := formatChatML(system, user)
	fmt.Printf("[Utilisateur]: %s\n", user)
	fmt.Print("[nanoGOqwen]: ")

	tokenCount := 0
	genStart := time.Now()

	var m1, m2 runtime.MemStats
	runtime.ReadMemStats(&m1)

	_, err := engine.GenerateStream(fullPrompt, maxTokens, stopStrings, func(piece string) bool {
		fmt.Print(piece)
		tokenCount++
		return true
	})
	elapsed := time.Since(genStart)
	runtime.ReadMemStats(&m2)
	fmt.Println()

	if err != nil {
		fmt.Fprintf(os.Stderr, "\nErreur d'inférence: %v\n", err)
		return
	}

	tps := 0.0
	if elapsed.Seconds() > 0 && tokenCount > 0 {
		tps = float64(tokenCount) / elapsed.Seconds()
	}
	mallocs := m2.Mallocs - m1.Mallocs
	bytesAlloc := m2.TotalAlloc - m1.TotalAlloc
	fmt.Printf("\n---\n[%d jetons en %v (%.2f tok/s) | %d allocs (%d octets) streaming]\n",
		tokenCount, elapsed.Round(time.Millisecond), tps, mallocs, bytesAlloc)
}

func runInteractiveREPL(engine *c2slm.Engine, system string, maxTokens int, stopStrings []string) {
	// Initialisation des outils de codage, de fichiers et de recherche web
	toolReg := c2slm.NewToolRegistry()
	_, err := c2slm.RegisterCodingTools(toolReg, ".")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Avertissement outils: %v\n", err)
	}

	// Lancement IMMÉDIAT de l'interface TUI plein écran (démarrage instantané 0s)
	chatBox := NewChatBox()
	chatBox.InitScreen(engine.Model.NumLayers, engine.KVCache.MaxTokens)
	defer chatBox.ResetScreen()

	systemPrefilled := false
	ensureSystemPrefilled := func() {
		if !systemPrefilled {
			prefillSystem(engine, system, toolReg)
			systemPrefilled = true
		}
	}

	for {
		line, readErr := chatBox.ReadPrompt()
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				fmt.Println("\nFermeture de nanoGOqwen.")
			} else {
				fmt.Fprintf(os.Stderr, "\nErreur de saisie: %v\n", readErr)
			}
			break
		}
		if line == "" {
			continue
		}

		// Gestion des commandes slash
		switch strings.ToLower(line) {
		case "exit", "quit", "/exit", "/quit":
			fmt.Println("Fermeture de nanoGOqwen.")
			return
		case "/help":
			chatBox.PrintHelp()
			continue
		case "/clear", "/reset":
			engine.KVCache.Reset()
			systemPrefilled = false
			fmt.Printf("%s[KV Cache réinitialisé : contexte remis à zéro]%s\n", ansiGreen, ansiReset)
			continue
		case "/tools":
			fmt.Printf("\n%sOutils enregistrés (%d) :%s\n", ansiBold, toolReg.Len(), ansiReset)
			for _, def := range toolReg.Definitions() {
				fmt.Printf("  %s• %-14s%s : %s\n", ansiYellow, def.Name, ansiReset, def.Description)
			}
			fmt.Println()
			continue
		case "/stats", "/context":
			used := engine.KVCache.SeqLen
			total := engine.KVCache.MaxTokens
			pct := float64(used) / float64(total) * 100.0
			fmt.Printf("\n%sContexte KV Cache :%s %d / %d jetons (%.1f%% occupé)\n\n", ansiBold, ansiReset, used, total, pct)
			continue
		}

		ensureSystemPrefilled()
		deltaTokens := engine.Tokenizer.Encode(formatUserTurn(line))

		// Context saturation check
		if engine.KVCache.SeqLen+len(deltaTokens) >= engine.KVCache.MaxTokens {
			fmt.Printf("\n%s[contexte saturé: réancrage du préfixe système]%s\n", ansiYellow, ansiReset)
			engine.KVCache.Reset()
			prefillSystem(engine, system, toolReg)
			systemPrefilled = true
		}

		chatBox.PrintAssistantHeader()

		totalTokens := 0
		turnStart := time.Now()
		wasInterrupted := false

		// Boucle multi-tours pour supporter les appels d'outils successifs
		currentDelta := deltaTokens
		for round := 0; round < 6; round++ {
			toolScan := c2slm.NewToolScanner()
			var rawToolCall string
			hasToolCall := false

			interrupted, cancelInference := chatBox.BeginInference()

			_, err := engine.GenerateStreamSession(currentDelta, maxTokens, stopStrings, func(piece string) bool {
				if interrupted.Load() {
					return false
				}
				done, _, clean := toolScan.Push(piece)
				if clean != "" {
					fmt.Print(clean)
				}
				totalTokens++
				if done {
					rawToolCall = toolScan.ToolJSON()
					hasToolCall = true
					return false
				}
				return true
			})

			cancelInference()

			if interrupted.Load() {
				wasInterrupted = true
				// Scellement défensif d'imEnd dans le KV cache pour fermer proprement le tour assistant
				_, imEnd, _ := engine.Tokenizer.SpecialTokenIDs()
				engine.IngestFrom(engine.KVCache.SeqLen, []int32{imEnd})
				break
			}

			if err != nil {
				fmt.Fprintf(os.Stderr, "\nErreur d'inférence: %v\n", err)
				break
			}

			if !hasToolCall {
				break
			}

			// Exécution de l'outil appelé par le modèle
			call, parseErr := c2slm.ParseToolCall([]byte(rawToolCall))
			if parseErr != nil {
				fmt.Printf("\n[Erreur syntaxe tool_call: %v]\n", parseErr)
				break
			}

			chatBox.PrintToolCall(call.Name, string(call.Arguments))
			toolOutput, dispatchErr := toolReg.Dispatch(call.Name, call.Arguments)
			if dispatchErr != nil {
				toolOutput = fmt.Sprintf(`{"error": %q}`, dispatchErr.Error())
				fmt.Printf("%s⚠️  [Erreur outil: %v]%s\n", ansiRed, dispatchErr, ansiReset)
			} else {
				chatBox.PrintToolResult(toolOutput)
			}

			// Réinjection du résultat sous <tool_response> dans le KV Cache
			respChunk := fmt.Sprintf("<tool_call>\n%s\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n%s\n</tool_response><|im_end|>\n<|im_start|>assistant\n",
				strings.TrimSpace(rawToolCall), toolOutput)
			currentDelta = engine.Tokenizer.Encode(respChunk)
		}

		elapsed := time.Since(turnStart)
		chatBox.PrintAssistantFooter(wasInterrupted, totalTokens, elapsed, engine.KVCache.SeqLen, engine.KVCache.MaxTokens)
	}
}
