package c2slm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// CodingTools fournit les capacités d'édition de code, d'exécution de commandes
// et de recherche web pour un agent autonome basé sur c2slm.
type CodingTools struct {
	WorkspaceRoot string
	HTTPClient    *http.Client
}

// NewCodingTools initialise l'ensemble d'outils de codage et recherche.
func NewCodingTools(workspaceRoot string) *CodingTools {
	if workspaceRoot == "" {
		workspaceRoot, _ = os.Getwd()
	}
	abs, err := filepath.Abs(workspaceRoot)
	if err == nil {
		workspaceRoot = abs
	}
	return &CodingTools{
		WorkspaceRoot: workspaceRoot,
		HTTPClient: &http.Client{
			Timeout: 15 * time.Second,
		},
	}
}

// RegisterCodingTools enregistre l'ensemble des outils de codage et recherche dans ToolRegistry.
func RegisterCodingTools(reg *ToolRegistry, workspaceRoot string) (*CodingTools, error) {
	ct := NewCodingTools(workspaceRoot)

	defs := []struct {
		def     ToolDefinition
		handler ToolHandler
	}{
		{
			def: ToolDefinition{
				Name:        "read_file",
				Description: "Lit un fichier texte (lignes optionnelles).",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"},"start_line":{"type":"integer"},"end_line":{"type":"integer"}},"required":["path"]}`),
			},
			handler: ct.ReadFile,
		},
		{
			def: ToolDefinition{
				Name:        "write_file",
				Description: "Crée ou écrase un fichier texte.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}`),
			},
			handler: ct.WriteFile,
		},
		{
			def: ToolDefinition{
				Name:        "patch_file",
				Description: "Remplace un bloc de texte cible dans un fichier.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"},"target_content":{"type":"string"},"replacement_content":{"type":"string"}},"required":["path","target_content","replacement_content"]}`),
			},
			handler: ct.PatchFile,
		},
		{
			def: ToolDefinition{
				Name:        "run_command",
				Description: "Exécute une commande bash et capture stdout/stderr.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}`),
			},
			handler: ct.RunCommand,
		},
		{
			def: ToolDefinition{
				Name:        "web_search",
				Description: "Recherche sur le web (DuckDuckGo).",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}`),
			},
			handler: ct.WebSearch,
		},
		{
			def: ToolDefinition{
				Name:        "fetch_url",
				Description: "Télécharge une page web (HTTP GET) et extrait son texte.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}`),
			},
			handler: ct.FetchURL,
		},
	}

	for _, d := range defs {
		if err := reg.Register(d.def, d.handler); err != nil {
			return nil, err
		}
	}
	return ct, nil
}

func (ct *CodingTools) resolvePath(p string) string {
	if filepath.IsAbs(p) {
		return filepath.Clean(p)
	}
	return filepath.Clean(filepath.Join(ct.WorkspaceRoot, p))
}

// ReadFile lit un fichier texte avec support de plage de lignes.
func (ct *CodingTools) ReadFile(args json.RawMessage) (string, error) {
	var req struct {
		Path      string `json:"path"`
		StartLine int    `json:"start_line"`
		EndLine   int    `json:"end_line"`
	}
	if err := json.Unmarshal(args, &req); err != nil {
		return "", fmt.Errorf("read_file: json invalide: %w", err)
	}
	target := ct.resolvePath(req.Path)
	f, err := os.Open(target)
	if err != nil {
		return "", fmt.Errorf("read_file open: %w", err)
	}
	defer f.Close()

	var b strings.Builder
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64<<10), 1<<20)

	lineNum := 0
	readBytes := 0
	const maxBytes = 32 << 10 // 32 Ko max

	for sc.Scan() {
		lineNum++
		if req.StartLine > 0 && lineNum < req.StartLine {
			continue
		}
		if req.EndLine > 0 && lineNum > req.EndLine {
			break
		}
		line := sc.Text()
		entry := fmt.Sprintf("%4d | %s\n", lineNum, line)
		if readBytes+len(entry) > maxBytes {
			b.WriteString("[... contenu tronqué à 32 Ko ...]\n")
			break
		}
		b.WriteString(entry)
		readBytes += len(entry)
	}
	if err := sc.Err(); err != nil {
		return "", fmt.Errorf("read_file scanner: %w", err)
	}
	return b.String(), nil
}

// WriteFile crée ou écrase un fichier.
func (ct *CodingTools) WriteFile(args json.RawMessage) (string, error) {
	var req struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(args, &req); err != nil {
		return "", fmt.Errorf("write_file: json invalide: %w", err)
	}
	target := ct.resolvePath(req.Path)
	if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
		return "", fmt.Errorf("write_file mkdir: %w", err)
	}
	if err := os.WriteFile(target, []byte(req.Content), 0644); err != nil {
		return "", fmt.Errorf("write_file write: %w", err)
	}
	return fmt.Sprintf("Fichier écrit avec succès (%d octets) : %s", len(req.Content), target), nil
}

// PatchFile remplace target_content par replacement_content.
func (ct *CodingTools) PatchFile(args json.RawMessage) (string, error) {
	var req struct {
		Path               string `json:"path"`
		TargetContent      string `json:"target_content"`
		ReplacementContent string `json:"replacement_content"`
	}
	if err := json.Unmarshal(args, &req); err != nil {
		return "", fmt.Errorf("patch_file: json invalide: %w", err)
	}
	target := ct.resolvePath(req.Path)
	data, err := os.ReadFile(target)
	if err != nil {
		return "", fmt.Errorf("patch_file read: %w", err)
	}
	str := string(data)
	count := strings.Count(str, req.TargetContent)
	if count == 0 {
		return "", fmt.Errorf("patch_file: target_content introuvable dans %s", target)
	}
	if count > 1 {
		return "", fmt.Errorf("patch_file: target_content apparaît %d fois (doit être unique)", count)
	}
	patched := strings.Replace(str, req.TargetContent, req.ReplacementContent, 1)
	if err := os.WriteFile(target, []byte(patched), 0644); err != nil {
		return "", fmt.Errorf("patch_file write: %w", err)
	}
	return fmt.Sprintf("Patch appliqué avec succès à %s", target), nil
}

// RunCommand exécute une commande shell bornée.
func (ct *CodingTools) RunCommand(args json.RawMessage) (string, error) {
	var req struct {
		Command        string `json:"command"`
		TimeoutSeconds int    `json:"timeout_seconds"`
	}
	if err := json.Unmarshal(args, &req); err != nil {
		return "", fmt.Errorf("run_command: json invalide: %w", err)
	}
	req.Command = strings.TrimSpace(req.Command)
	if req.Command == "" {
		return "", errors.New("run_command: commande vide")
	}
	timeout := 30 * time.Second
	if req.TimeoutSeconds > 0 {
		if req.TimeoutSeconds > 120 {
			req.TimeoutSeconds = 120
		}
		timeout = time.Duration(req.TimeoutSeconds) * time.Second
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "bash", "-c", req.Command)
	cmd.Dir = ct.WorkspaceRoot

	var outBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &outBuf

	err := cmd.Run()
	outBytes := outBuf.Bytes()
	const maxOutput = 16 << 10 // 16 Ko max
	truncated := false
	if len(outBytes) > maxOutput {
		outBytes = outBytes[:maxOutput]
		truncated = true
	}

	res := string(outBytes)
	if truncated {
		res += "\n[... sortie tronquée à 16 Ko ...]"
	}

	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return fmt.Sprintf("ERREUR: Délai dépassé (%v) pour la commande : %s\n%s", timeout, req.Command, res), nil
		}
		return fmt.Sprintf("ÉCHEC (code retour: %v) :\n%s", err, res), nil
	}
	if strings.TrimSpace(res) == "" {
		return "Commande exécutée avec succès (aucune sortie textuelle).", nil
	}
	return res, nil
}

// WebSearch interroge DuckDuckGo Instant Answer API.
func (ct *CodingTools) WebSearch(args json.RawMessage) (string, error) {
	var req struct {
		Query string `json:"query"`
	}
	if err := json.Unmarshal(args, &req); err != nil {
		return "", fmt.Errorf("web_search: json invalide: %w", err)
	}
	req.Query = strings.TrimSpace(req.Query)
	if req.Query == "" {
		return "", errors.New("web_search: query vide")
	}

	searchURL := fmt.Sprintf("https://api.duckduckgo.com/?q=%s&format=json&no_redirect=1&no_html=1", url.QueryEscape(req.Query))
	httpReq, err := http.NewRequest("GET", searchURL, nil)
	if err != nil {
		return "", err
	}
	httpReq.Header.Set("User-Agent", "c2slm-agent/1.0")

	resp, err := ct.HTTPClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("web_search http: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 128<<10))
	if err != nil {
		return "", fmt.Errorf("web_search read: %w", err)
	}

	var ddg struct {
		Heading      string `json:"Heading"`
		AbstractText string `json:"AbstractText"`
		AbstractURL  string `json:"AbstractURL"`
		Related      []struct {
			Text     string `json:"Text"`
			FirstURL string `json:"FirstURL"`
		} `json:"RelatedTopics"`
	}
	if err := json.Unmarshal(body, &ddg); err != nil {
		return string(body), nil
	}

	var b strings.Builder
	if ddg.Heading != "" {
		b.WriteString(fmt.Sprintf("### %s\n", ddg.Heading))
	}
	if ddg.AbstractText != "" {
		b.WriteString(fmt.Sprintf("%s\nSource: %s\n\n", ddg.AbstractText, ddg.AbstractURL))
	}
	count := 0
	for _, rel := range ddg.Related {
		if rel.Text != "" {
			b.WriteString(fmt.Sprintf("- %s (%s)\n", rel.Text, rel.FirstURL))
			count++
			if count >= 5 {
				break
			}
		}
	}
	if b.Len() == 0 {
		return fmt.Sprintf("Aucun résultat direct pour %q. Essayez d'utiliser fetch_url avec un lien spécifique.", req.Query), nil
	}
	return b.String(), nil
}

var scriptRegex = regexp.MustCompile(`(?is)<script[^>]*>.*?</script>`)
var styleRegex = regexp.MustCompile(`(?is)<style[^>]*>.*?</style>`)
var tagRegex = regexp.MustCompile(`<[^>]*>`)
var spaceRegex = regexp.MustCompile(`[ \t\r\f\v]{2,}`)

// FetchURL télécharge une page web ou un document brut et en extrait le texte épuré.
func (ct *CodingTools) FetchURL(args json.RawMessage) (string, error) {
	var req struct {
		URL string `json:"url"`
	}
	if err := json.Unmarshal(args, &req); err != nil {
		return "", fmt.Errorf("fetch_url: json invalide: %w", err)
	}
	req.URL = strings.TrimSpace(req.URL)
	if !strings.HasPrefix(req.URL, "http://") && !strings.HasPrefix(req.URL, "https://") {
		return "", errors.New("fetch_url: URL doit débuter par http:// ou https://")
	}

	httpReq, err := http.NewRequest("GET", req.URL, nil)
	if err != nil {
		return "", err
	}
	httpReq.Header.Set("User-Agent", "Mozilla/5.0 (compatible; c2slm-agent/1.0)")

	resp, err := ct.HTTPClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("fetch_url get: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 128<<10)) // 128 Ko max
	if err != nil {
		return "", fmt.Errorf("fetch_url read: %w", err)
	}

	raw := string(body)
	// Élimination des scripts et styles
	raw = scriptRegex.ReplaceAllString(raw, " ")
	raw = styleRegex.ReplaceAllString(raw, " ")
	// Élimination des balises HTML
	clean := tagRegex.ReplaceAllString(raw, " ")
	// Remplacement des entités HTML courantes
	clean = strings.ReplaceAll(clean, "&nbsp;", " ")
	clean = strings.ReplaceAll(clean, "&amp;", "&")
	clean = strings.ReplaceAll(clean, "&lt;", "<")
	clean = strings.ReplaceAll(clean, "&gt;", ">")
	clean = strings.ReplaceAll(clean, "&quot;", "\"")
	clean = strings.ReplaceAll(clean, "&#39;", "'")
	// Normalisation des espaces
	clean = spaceRegex.ReplaceAllString(clean, " ")
	clean = strings.TrimSpace(clean)

	const maxRes = 24 << 10 // 24 Ko max pour le contexte
	if len(clean) > maxRes {
		clean = clean[:maxRes] + "\n[... contenu tronqué à 24 Ko ...]"
	}
	return clean, nil
}
