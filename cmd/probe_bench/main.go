package main

import (
	"flag"
	"fmt"
	"os"
	"runtime/pprof"
	"syscall"
	"time"

	"github.com/hazyhaar/c2slm"
	"github.com/hazyhaar/c2slm/engine"
	"github.com/hazyhaar/c2slm/tensor"
)

type RusageSnapshot struct {
	UserTime   time.Duration
	SysTime    time.Duration
	MaxRSS     int64 // KB
	MinorFault int64 // page reclaims (soft page faults)
	MajorFault int64 // page faults requiring disk I/O
	InBlocks   int64 // block input operations
	OutBlocks  int64 // block output operations
	VolCtxSw   int64 // voluntary context switches
	InvolCtxSw int64 // involuntary context switches
}

func getRusage() RusageSnapshot {
	var ru syscall.Rusage
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	return RusageSnapshot{
		UserTime:   time.Duration(ru.Utime.Sec)*time.Second + time.Duration(ru.Utime.Usec)*time.Microsecond,
		SysTime:    time.Duration(ru.Stime.Sec)*time.Second + time.Duration(ru.Stime.Usec)*time.Microsecond,
		MaxRSS:     ru.Maxrss,
		MinorFault: ru.Minflt,
		MajorFault: ru.Majflt,
		InBlocks:   ru.Inblock,
		OutBlocks:  ru.Oublock,
		VolCtxSw:   ru.Nvcsw,
		InvolCtxSw: ru.Nivcsw,
	}
}

type LayerProfiling struct {
	AttnNormDuration  time.Duration
	QKVGemvDuration   time.Duration
	QKNormDuration    time.Duration
	RoPEDuration      time.Duration
	KVStoreDuration   time.Duration
	AttentionDuration time.Duration
	OutGemvDuration   time.Duration
	Res1Duration      time.Duration
	FFNNormDuration   time.Duration
	FFNGateUpDuration time.Duration
	SwiGLUDuration    time.Duration
	FFNDownDuration   time.Duration
	Res2Duration      time.Duration
}

type StepProfiling struct {
	EmbedDuration     time.Duration
	LayersTotal       time.Duration
	FinalNormDuration time.Duration
	LMHeadDuration    time.Duration
	ArgmaxDuration    time.Duration
	LayerBreakdown    LayerProfiling
	TotalStepDuration time.Duration
}

func ProfileSingleStep(m *engine.Model, cache *engine.KVCache, a *engine.Arena, tokenID int32, pos int) StepProfiling {
	var prof StepProfiling
	tTotalStart := time.Now()

	// 1. Embed
	t0 := time.Now()
	m.EmbedToken(a.X, tokenID)
	prof.EmbedDuration = time.Since(t0)

	headDim := m.HeadDim
	scale := float32(1.0 / 8.0) // 1/sqrt(64) or 1/sqrt(128)
	if headDim == 128 {
		scale = float32(1.0 / 11.3137)
	}
	kvHeads := m.NumKVHeads

	tLayersStart := time.Now()
	for lIdx := 0; lIdx < m.NumLayers; lIdx++ {
		layer := &m.Layers[lIdx]

		// AttnNorm
		tNorm := time.Now()
		tensor.RMSNorm(a.NormX, a.X, layer.AttnNorm, m.RMSNormEps)
		prof.LayerBreakdown.AttnNormDuration += time.Since(tNorm)

		// QKV Projections
		tQKV := time.Now()
		engine.DispatchGEMV(a.Pool, a.Q, layer.AttnQ, a.NormX, m.EmbdLength, m.EmbdLength)
		tensor.AddBias(a.Q, layer.AttnQBias)
		engine.DispatchGEMV(a.Pool, a.K, layer.AttnK, a.NormX, kvHeads*headDim, m.EmbdLength)
		tensor.AddBias(a.K, layer.AttnKBias)
		engine.DispatchGEMV(a.Pool, a.V, layer.AttnV, a.NormX, kvHeads*headDim, m.EmbdLength)
		tensor.AddBias(a.V, layer.AttnVBias)
		prof.LayerBreakdown.QKVGemvDuration += time.Since(tQKV)

		// QK-Norm
		tQKNorm := time.Now()
		if len(layer.AttnQNorm) > 0 {
			for h := 0; h < m.NumHeads; h++ {
				qHead := a.Q[h*headDim : (h+1)*headDim]
				tensor.RMSNorm(qHead, qHead, layer.AttnQNorm, m.RMSNormEps)
			}
		}
		if len(layer.AttnKNorm) > 0 {
			for h := 0; h < kvHeads; h++ {
				kHead := a.K[h*headDim : (h+1)*headDim]
				tensor.RMSNorm(kHead, kHead, layer.AttnKNorm, m.RMSNormEps)
			}
		}
		prof.LayerBreakdown.QKNormDuration += time.Since(tQKNorm)

		// RoPE
		tRoPE := time.Now()
		if m.RoPETable != nil {
			m.RoPETable.Apply(a.Q, m.NumHeads, pos)
			m.RoPETable.Apply(a.K, kvHeads, pos)
		} else {
			tensor.RoPENeoX(a.Q, m.NumHeads, headDim, pos, m.RoPEBase)
			tensor.RoPENeoX(a.K, kvHeads, headDim, pos, m.RoPEBase)
		}
		prof.LayerBreakdown.RoPEDuration += time.Since(tRoPE)

		// KV-Store
		tKV := time.Now()
		cache.StoreKV(lIdx, pos, a.K, a.V)
		prof.LayerBreakdown.KVStoreDuration += time.Since(tKV)

		// Multi-Head Attention
		tAttn := time.Now()
		engine.DispatchAttention(a.Pool, a.AttnScores, cache, lIdx, pos, a.Q, a.AttnOut, m.NumHeads, kvHeads, headDim, scale)
		prof.LayerBreakdown.AttentionDuration += time.Since(tAttn)

		// Output projection W_o
		tOutGemv := time.Now()
		engine.DispatchGEMV(a.Pool, a.Down, layer.AttnOut, a.AttnOut, m.EmbdLength, m.EmbdLength)
		prof.LayerBreakdown.OutGemvDuration += time.Since(tOutGemv)

		// Residual 1
		tRes1 := time.Now()
		for i := 0; i < m.EmbdLength; i++ {
			a.X[i] += a.Down[i]
		}
		prof.LayerBreakdown.Res1Duration += time.Since(tRes1)

		// Pre-FFN RMSNorm
		tFFNNorm := time.Now()
		tensor.RMSNorm(a.NormX, a.X, layer.FFNNorm, m.RMSNormEps)
		prof.LayerBreakdown.FFNNormDuration += time.Since(tFFNNorm)

		// FFN Gate & Up
		tFFNGateUp := time.Now()
		engine.DispatchGEMV(a.Pool, a.Gate, layer.FFNGate, a.NormX, m.FFNDim, m.EmbdLength)
		engine.DispatchGEMV(a.Pool, a.Up, layer.FFNUp, a.NormX, m.FFNDim, m.EmbdLength)
		prof.LayerBreakdown.FFNGateUpDuration += time.Since(tFFNGateUp)

		// SwiGLU
		tSwiGLU := time.Now()
		tensor.SwiGLU(a.Gate, a.Gate, a.Up)
		prof.LayerBreakdown.SwiGLUDuration += time.Since(tSwiGLU)

		// FFN Down
		tFFNDown := time.Now()
		engine.DispatchGEMV(a.Pool, a.Down, layer.FFNDown, a.Gate, m.EmbdLength, m.FFNDim)
		prof.LayerBreakdown.FFNDownDuration += time.Since(tFFNDown)

		// Residual 2
		tRes2 := time.Now()
		for i := 0; i < m.EmbdLength; i++ {
			a.X[i] += a.Down[i]
		}
		prof.LayerBreakdown.Res2Duration += time.Since(tRes2)
	}
	prof.LayersTotal = time.Since(tLayersStart)

	// Final RMSNorm
	tFinalNorm := time.Now()
	tensor.RMSNorm(a.NormX, a.X, m.OutputNorm, m.RMSNormEps)
	prof.FinalNormDuration = time.Since(tFinalNorm)

	// LM Head Projection (VocabSize rows)
	tLMHead := time.Now()
	engine.ComputeLogits(m, a, a.NormX, a.Logits, nil)
	prof.LMHeadDuration = time.Since(tLMHead)

	// Greedy Argmax
	tArgmax := time.Now()
	bestID := int32(0)
	bestLogit := a.Logits[0]
	for id := 1; id < m.VocabSize; id++ {
		if a.Logits[id] > bestLogit {
			bestLogit = a.Logits[id]
			bestID = int32(id)
		}
	}
	_ = bestID
	prof.ArgmaxDuration = time.Since(tArgmax)

	prof.TotalStepDuration = time.Since(tTotalStart)
	return prof
}

func main() {
	modelPath := flag.String("model", "/data/models/qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf", "Path to GGUF model")
	numSteps := flag.Int("steps", 10, "Number of token steps to profile")
	cpuProfile := flag.String("cpuprofile", "", "Write cpu profile to file")
	flag.Parse()

	if *cpuProfile != "" {
		f, err := os.Create(*cpuProfile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Cannot create cpu profile: %v\n", err)
			os.Exit(1)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			fmt.Fprintf(os.Stderr, "Cannot start cpu profile: %v\n", err)
			os.Exit(1)
		}
		defer pprof.StopCPUProfile()
	}

	ruBeforeLoad := getRusage()
	tLoadStart := time.Now()
	eng, err := c2slm.NewEngine(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading engine: %v\n", err)
		os.Exit(1)
	}
	defer eng.Close()
	loadDuration := time.Since(tLoadStart)
	ruAfterLoad := getRusage()

	fmt.Printf("================================================================================\n")
	fmt.Printf("  SONDE KERNEL UNIX & PROFILING MATÉRIEL C2SLM\n")
	fmt.Printf("================================================================================\n")
	fmt.Printf("Modèle: %s\n", *modelPath)
	fmt.Printf("Architecture: %d couches, dim=%d, ffn_dim=%d, vocab=%d, heads=%d, kv_heads=%d, head_dim=%d\n",
		eng.Model.NumLayers, eng.Model.EmbdLength, eng.Model.FFNDim, eng.Model.VocabSize,
		eng.Model.NumHeads, eng.Model.NumKVHeads, eng.Model.HeadDim)
	fmt.Printf("CPUs physiques / Goroutines Pool: %d workers\n", eng.Arena.Pool.NumWorkers())
	fmt.Printf("Temps de chargement initial (mmap): %v\n", loadDuration)
	fmt.Printf("RSS après chargement: %d Mo (Soft Faults: %d, Hard Faults I/O: %d)\n\n",
		ruAfterLoad.MaxRSS/1024, ruAfterLoad.MinorFault-ruBeforeLoad.MinorFault, ruAfterLoad.MajorFault-ruBeforeLoad.MajorFault)

	// Warmup 1 step to trigger initial mmap page faults
	fmt.Println("--- 1. ÉPREUVE WARMUP (Détection des Page Faults & Accès Disque) ---")
	ruBeforeWarmup := getRusage()
	tWarmupStart := time.Now()
	_ = ProfileSingleStep(eng.Model, eng.KVCache, eng.Arena, 151644, 0)
	warmupDuration := time.Since(tWarmupStart)
	ruAfterWarmup := getRusage()

	fmt.Printf("Durée du premier token (Warmup): %v (%.2f tok/s)\n", warmupDuration, 1.0/warmupDuration.Seconds())
	fmt.Printf("Delta Kernel Probe Warmup:\n")
	fmt.Printf("  - Temps CPU Utilisateur : %v\n", ruAfterWarmup.UserTime-ruBeforeWarmup.UserTime)
	fmt.Printf("  - Temps CPU Système/IO  : %v\n", ruAfterWarmup.SysTime-ruBeforeWarmup.SysTime)
	fmt.Printf("  - Soft Page Faults (Reclaims) : +%d\n", ruAfterWarmup.MinorFault-ruBeforeWarmup.MinorFault)
	fmt.Printf("  - Hard Page Faults (Disk I/O) : +%d\n", ruAfterWarmup.MajorFault-ruBeforeWarmup.MajorFault)
	fmt.Printf("  - InBlocks (Lectures disque)  : +%d\n", ruAfterWarmup.InBlocks-ruBeforeWarmup.InBlocks)
	fmt.Printf("  - Changements contexte Volontaires   (I/O & Goroutines) : +%d\n", ruAfterWarmup.VolCtxSw-ruBeforeWarmup.VolCtxSw)
	fmt.Printf("  - Changements contexte Involontaires (Préemption CPU)    : +%d\n\n", ruAfterWarmup.InvolCtxSw-ruBeforeWarmup.InvolCtxSw)

	// Profiling across steps
	fmt.Printf("--- 2. BENCHMARK STRATIFIÉ SUR %d ÉTAPES DE GÉNÉRATION ---\n", *numSteps)
	var accum StepProfiling
	ruBeforeBench := getRusage()
	tBenchStart := time.Now()

	for step := 0; step < *numSteps; step++ {
		p := ProfileSingleStep(eng.Model, eng.KVCache, eng.Arena, 151644, step+1)
		accum.EmbedDuration += p.EmbedDuration
		accum.LayersTotal += p.LayersTotal
		accum.FinalNormDuration += p.FinalNormDuration
		accum.LMHeadDuration += p.LMHeadDuration
		accum.ArgmaxDuration += p.ArgmaxDuration
		accum.TotalStepDuration += p.TotalStepDuration

		accum.LayerBreakdown.AttnNormDuration += p.LayerBreakdown.AttnNormDuration
		accum.LayerBreakdown.QKVGemvDuration += p.LayerBreakdown.QKVGemvDuration
		accum.LayerBreakdown.QKNormDuration += p.LayerBreakdown.QKNormDuration
		accum.LayerBreakdown.RoPEDuration += p.LayerBreakdown.RoPEDuration
		accum.LayerBreakdown.KVStoreDuration += p.LayerBreakdown.KVStoreDuration
		accum.LayerBreakdown.AttentionDuration += p.LayerBreakdown.AttentionDuration
		accum.LayerBreakdown.OutGemvDuration += p.LayerBreakdown.OutGemvDuration
		accum.LayerBreakdown.Res1Duration += p.LayerBreakdown.Res1Duration
		accum.LayerBreakdown.FFNNormDuration += p.LayerBreakdown.FFNNormDuration
		accum.LayerBreakdown.FFNGateUpDuration += p.LayerBreakdown.FFNGateUpDuration
		accum.LayerBreakdown.SwiGLUDuration += p.LayerBreakdown.SwiGLUDuration
		accum.LayerBreakdown.FFNDownDuration += p.LayerBreakdown.FFNDownDuration
		accum.LayerBreakdown.Res2Duration += p.LayerBreakdown.Res2Duration
	}

	totalWall := time.Since(tBenchStart)
	ruAfterBench := getRusage()
	n := float64(*numSteps)

	avgTotal := accum.TotalStepDuration / time.Duration(*numSteps)
	tokPerSec := float64(*numSteps) / totalWall.Seconds()

	fmt.Printf("Débit Réel Moyen : %.2f tokens / seconde (Latence moyenne par token: %v)\n\n", tokPerSec, avgTotal)

	fmt.Printf("VENTILATION STRATIFIÉE PAR TOKEN (Moyenne sur %d passes):\n", *numSteps)
	fmt.Printf("%-35s : %10v (%5.1f%%)\n", "1. Embedding Lookup", accum.EmbedDuration/time.Duration(*numSteps), float64(accum.EmbedDuration)/float64(accum.TotalStepDuration)*100)
	fmt.Printf("%-35s : %10v (%5.1f%%)\n", "2. Couches Transformer (Total)", accum.LayersTotal/time.Duration(*numSteps), float64(accum.LayersTotal)/float64(accum.TotalStepDuration)*100)
	fmt.Printf("%-46s : %10v (%5.1f%%)\n", "   ├── AttnNorm (Pre-QKV)", accum.LayerBreakdown.AttnNormDuration/time.Duration(*numSteps), float64(accum.LayerBreakdown.AttnNormDuration)/float64(accum.TotalStepDuration)*100)
	fmt.Printf("%-46s : %10v (%5.1f%%)\n", "   ├── QKV Projections (GEMV)", accum.LayerBreakdown.QKVGemvDuration/time.Duration(*numSteps), float64(accum.LayerBreakdown.QKVGemvDuration)/float64(accum.TotalStepDuration)*100)
	if accum.LayerBreakdown.QKNormDuration > 0 {
		fmt.Printf("%-46s : %10v (%5.1f%%)\n", "   ├── QK-Norm (Per-Head Qwen3)", accum.LayerBreakdown.QKNormDuration/time.Duration(*numSteps), float64(accum.LayerBreakdown.QKNormDuration)/float64(accum.TotalStepDuration)*100)
	}
	fmt.Printf("%-46s : %10v (%5.1f%%)\n", "   ├── RoPE NeoX", accum.LayerBreakdown.RoPEDuration/time.Duration(*numSteps), float64(accum.LayerBreakdown.RoPEDuration)/float64(accum.TotalStepDuration)*100)
	fmt.Printf("%-46s : %10v (%5.1f%%)\n", "   ├── KV Cache Store", accum.LayerBreakdown.KVStoreDuration/time.Duration(*numSteps), float64(accum.LayerBreakdown.KVStoreDuration)/float64(accum.TotalStepDuration)*100)
	fmt.Printf("%-46s : %10v (%5.1f%%)\n", "   ├── Multi-Head Attn (Q@K + Softmax + Attn@V)", accum.LayerBreakdown.AttentionDuration/time.Duration(*numSteps), float64(accum.LayerBreakdown.AttentionDuration)/float64(accum.TotalStepDuration)*100)
	fmt.Printf("%-46s : %10v (%5.1f%%)\n", "   ├── Output Projection W_o (GEMV)", accum.LayerBreakdown.OutGemvDuration/time.Duration(*numSteps), float64(accum.LayerBreakdown.OutGemvDuration)/float64(accum.TotalStepDuration)*100)
	fmt.Printf("%-46s : %10v (%5.1f%%)\n", "   ├── Residual 1 Add", accum.LayerBreakdown.Res1Duration/time.Duration(*numSteps), float64(accum.LayerBreakdown.Res1Duration)/float64(accum.TotalStepDuration)*100)
	fmt.Printf("%-46s : %10v (%5.1f%%)\n", "   ├── FFNNorm (Pre-FFN)", accum.LayerBreakdown.FFNNormDuration/time.Duration(*numSteps), float64(accum.LayerBreakdown.FFNNormDuration)/float64(accum.TotalStepDuration)*100)
	fmt.Printf("%-46s : %10v (%5.1f%%)\n", "   ├── FFN Gate & Up Projections (GEMV)", accum.LayerBreakdown.FFNGateUpDuration/time.Duration(*numSteps), float64(accum.LayerBreakdown.FFNGateUpDuration)/float64(accum.TotalStepDuration)*100)
	fmt.Printf("%-46s : %10v (%5.1f%%)\n", "   ├── SwiGLU Activation (SiLU * Up)", accum.LayerBreakdown.SwiGLUDuration/time.Duration(*numSteps), float64(accum.LayerBreakdown.SwiGLUDuration)/float64(accum.TotalStepDuration)*100)
	fmt.Printf("%-46s : %10v (%5.1f%%)\n", "   ├── FFN Down Projection (GEMV)", accum.LayerBreakdown.FFNDownDuration/time.Duration(*numSteps), float64(accum.LayerBreakdown.FFNDownDuration)/float64(accum.TotalStepDuration)*100)
	fmt.Printf("%-46s : %10v (%5.1f%%)\n", "   └── Residual 2 Add", accum.LayerBreakdown.Res2Duration/time.Duration(*numSteps), float64(accum.LayerBreakdown.Res2Duration)/float64(accum.TotalStepDuration)*100)
	fmt.Printf("%-35s : %10v (%5.1f%%)\n", "3. Final Output RMSNorm", accum.FinalNormDuration/time.Duration(*numSteps), float64(accum.FinalNormDuration)/float64(accum.TotalStepDuration)*100)
	fmt.Printf("%-35s : %10v (%5.1f%%)\n", "4. LM Head Projection (Vocab 151k)", accum.LMHeadDuration/time.Duration(*numSteps), float64(accum.LMHeadDuration)/float64(accum.TotalStepDuration)*100)
	fmt.Printf("%-35s : %10v (%5.1f%%)\n", "5. Greedy Argmax Selection", accum.ArgmaxDuration/time.Duration(*numSteps), float64(accum.ArgmaxDuration)/float64(accum.TotalStepDuration)*100)
	fmt.Printf("--------------------------------------------------------------------------------\n")
	fmt.Printf("%-35s : %10v (100.0%%)\n\n", "TOTAL PAR JETON", avgTotal)

	fmt.Printf("SONDES SYSTÈME KERNEL EN RÉGIME PERMANENT (%d tokens):\n", *numSteps)
	fmt.Printf("  - Temps CPU Utilisateur : %v (soit %.1f%% de l'activité)\n",
		ruAfterBench.UserTime-ruBeforeBench.UserTime,
		float64(ruAfterBench.UserTime-ruBeforeBench.UserTime)/float64(totalWall)*100)
	fmt.Printf("  - Temps CPU Kernel / IO : %v\n", ruAfterBench.SysTime-ruBeforeBench.SysTime)
	fmt.Printf("  - Soft Page Faults      : +%d (%.1f par token)\n",
		ruAfterBench.MinorFault-ruBeforeBench.MinorFault, float64(ruAfterBench.MinorFault-ruBeforeBench.MinorFault)/n)
	fmt.Printf("  - Hard Page Faults (IO) : +%d\n", ruAfterBench.MajorFault-ruBeforeBench.MajorFault)
	fmt.Printf("  - InBlocks (Lectures)   : +%d\n", ruAfterBench.InBlocks-ruBeforeBench.InBlocks)
	fmt.Printf("  - Changements Contexte  : +%d Volontaires (%.0f/tok) | +%d Involontaires (%.0f/tok)\n",
		ruAfterBench.VolCtxSw-ruBeforeBench.VolCtxSw, float64(ruAfterBench.VolCtxSw-ruBeforeBench.VolCtxSw)/n,
		ruAfterBench.InvolCtxSw-ruBeforeBench.InvolCtxSw, float64(ruAfterBench.InvolCtxSw-ruBeforeBench.InvolCtxSw)/n)
	fmt.Printf("  - Mémoire Résidente RSS : %d Mo\n", ruAfterBench.MaxRSS/1024)
}
