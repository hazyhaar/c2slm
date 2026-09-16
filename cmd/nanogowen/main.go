package main

import (
	"bufio"
	"flag"
	"fmt"
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
	fmt.Println("=== Session interactive nanoGOqwen (Pur Go) ===")
	fmt.Println("Tapez votre message et appuyez sur [Entrée]. Entrez 'exit' ou 'quit' pour fermer.")
	fmt.Println("--------------------------------------------------------------------------------")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\n> ")
		if !scanner.Scan() {
			break
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if line == "exit" || line == "quit" {
			fmt.Println("Fermeture de nanoGOqwen.")
			break
		}

		fullPrompt := formatChatML(system, line)
		fmt.Print("\n[nanoGOqwen]: ")

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
			fmt.Fprintf(os.Stderr, "Erreur d'inférence: %v\n", err)
			continue
		}

		tps := 0.0
		if elapsed.Seconds() > 0 && tokenCount > 0 {
			tps = float64(tokenCount) / elapsed.Seconds()
		}
		mallocs := m2.Mallocs - m1.Mallocs
		bytesAlloc := m2.TotalAlloc - m1.TotalAlloc
		fmt.Printf("[%d jetons en %v (%.2f tok/s) | %d allocs (%d octets)]\n",
			tokenCount, elapsed.Round(time.Millisecond), tps, mallocs, bytesAlloc)
	}
}
