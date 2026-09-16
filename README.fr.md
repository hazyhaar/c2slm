# c2slm / nanogowen

**Moteur d'inférence Transformer compact 100% Pur Go (Zero CGo, Zero Wasm, Zero Dependency)**  
Spécialisé pour l'architecture **Qwen2.5 / Qwen2** (modèle de référence physique : `Qwen2.5-0.5B-Instruct` quantifié GGUF).

---

## 1. Caractéristiques Principales

- **Zéro Dépendance Externe :** Compilable avec la toolchain Go standard (`go 1.27`). Aucun runtime C++, CGo, Wasm, Python ou bibliothèque partagée.
- **Micro-Noyaux Vectoriels AVX2 purs :** Transpilés mécaniquement depuis C99 via le compilateur souverain `sgoiter` et le package standard `simd/archsimd` (`GOEXPERIMENT=simd`), avec repli scalaire automatique si AVX2 est indisponible.
- **Arithmétique Entière Vectorisée Q4_K x Q8_K :**
  - Quantification dynamique de l'activation $F32 \rightarrow Q8\_K$ effectuée une seule fois par projection.
  - Produit scalaire entier pur sous `VPMADDUBSW` (`DotProductPairsSaturated`) et `VPMADDWD` (`DotProductPairs`) sans déballage flottant intermédiaire.
  - Parité bit-exacte à 100% prouvée formellement contre l'oracle binaire C99 GCC -O2.
- **Zéro Allocation Tas sur la Passe Forward (0 B/op) :**
  - Arène d'activation statique (`Arena`) et pool persistant de 32 workers statiques (`WorkerPool`).
  - Projection restreinte ARCHTIME pour l'échantillonnage de tokens ciblés en < 75 µs.
- **Format GGUF Natif v2 / v3 :**
  - Mmap direct du fichier de poids sans duplication mémoire.
  - Formats de quantification supportés : `Q4_K` (avec noyau entier Q8_K), `Q8_0`, `Q5_0`, `Q6_K`, `F32`.
- **Tokenizer BPE ChatML Intégré & Interface TUI Interactive :**
  - Découpage par expressions régulières et vocabulaire BPE de 151 936 jetons.
  - Interface plein écran VT100 / ANSI autonome (région de défilement `DECSTBM`, barre de saisie ancrée en bas, gestion des largeurs Unicode, interruption instantanée Ctrl+C sans perte de contexte).
  - Pénalité de répétition dynamique sur les logits récents pour éliminer les boucles autoregressives.

---

## 2. Métrologie Physique Réelle (Intel Core i9-14900K, 32 Threads)

Mesures issues de tests réels sous `testing.B` et inférence physique :

| Composant / Opération | Implémentation | Latence / Débit | Allocations Tas |
| :--- | :--- | :--- | :--- |
| **DotProduct Q4_K (896 cols)** | AVX2 Entier (`DotQ4_K_Q8_K`) | **1 111 ns/op** (gain 2,0× vs scalaire) | **0 B/op, 0 allocs/op** |
| **DotProduct Q8_0 (32 cols)** | AVX2 Flottant (`DotQ8_0`) | **149 ns/op** (gain 3,4× vs scalaire) | **0 B/op, 0 allocs/op** |
| **Quantification Q8_K (896 cols)** | `QuantizeRowQ8_K` dynamique | **1 849 ns/op (1,94 Go/s)** | **0 B/op, 0 allocs/op** |
| **Projection Vocabulaire (151k)** | `ComputeLogits` multi-cœurs | **2,91 ms** (parallélisé sur 32 threads) | **0 B/op, 0 allocs/op** |
| **Passe Forward Complète (24 couches)** | `Forward` (1 token) | **38,6 ms** | **0 B/op, 0 allocs/op** |
| **Inférence Bout-en-Bout (`nanogowen`)** | Qwen2.5-0.5B-Instruct-Q4_K_M | **3,79 tokens / seconde** | **Flux streaming continu** |

---

## 3. Installation & Utilisation

### Compilation du CLI `nanogowen`

```bash
GOEXPERIMENT=simd go build -o bin/nanogowen ./cmd/nanogowen
```

### Exécution du CLI

```bash
# Inférence ponctuelle en ligne de commande :
./bin/nanogowen -model /chemin/vers/qwen2.5-0.5b-instruct-q4_k_m.gguf -m "Explique la théorie de l'information de Shannon."

# Mode interactif plein écran TUI (barre ancrée, région de défilement ANSI) :
./bin/nanogowen -model /chemin/vers/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

### Commandes du Mode TUI Interactif

Pendant une session interactive, les commandes suivantes sont disponibles :
- `/stats` ou `/context` : affiche l'occupation et la capacité du KV-cache.
- `/clear` ou `/reset` : réinitialise le KV-cache et réancre le prompt système.
- `/help` : affiche l'aide et les raccourcis.
- `Ctrl+C` : interrompt la réponse de l'assistant sans perdre l'historique du dialogue.
- `/exit` ou `/quit` : quitte proprement la session.

### Utilisation comme Bibliothèque Go

```go
package main

import (
	"fmt"
	"log"

	"github.com/hazyhaar/c2slm"
)

func main() {
	engine, err := c2slm.NewEngine("/path/to/model.gguf")
	if err != nil {
		log.Fatal(err)
	}
	defer engine.Close()

	prompt := "<|im_start|>user\nBonjour !<|im_end|>\n<|im_start|>assistant\n"
	tokens := engine.Tokenizer.Encode(prompt)

	fmt.Println("Réponse du modèle :")
	_ = engine.Generate(tokens, 64, []string{"<|im_end|>"}, func(token string) bool {
		fmt.Print(token)
		return true
	})
	fmt.Println()
}
```

---

## 4. Architecture Interne

```
c2slm/
├── c2slm.go          # API publique haut-niveau (Engine, Generate, NewEngine)
├── c2slm_test.go     # Tests d'intégration bout-en-bout
├── go.mod            # Module autonome github.com/hazyhaar/c2slm (0 dépendance)
├── cmd/
│   └── nanogowen/    # CLI binaire interactif et streaming
├── engine/           # Moteur d'inférence (Model, Forward, KVCache, WorkerPool, Arena)
├── gguf/             # Parser binaire GGUF v2/v3 à mémoire projetée (mmap)
├── internal/
│   └── simd/         # Micro-noyaux vectoriels AVX2 transpilés sgoiter + repli scalaire
├── tensor/           # Opérations algébriques (RMSNorm, RoPE NeoX, SwiGLU, Softmax, GEMV)
└── tokenizer/        # BPE ChatML pur Go
```

---

## 5. Validation & Tests

```bash
# Exécution de la suite complète de tests sans dépendances externes :
GOEXPERIMENT=simd go test -race -count=1 ./...
```
