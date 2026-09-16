package c2slm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCodingTools_FileOps(t *testing.T) {
	tmpDir := t.TempDir()
	reg := NewToolRegistry()
	ct, err := RegisterCodingTools(reg, tmpDir)
	if err != nil {
		t.Fatalf("RegisterCodingTools: %v", err)
	}

	// 1. WriteFile
	testPath := "sub/test.txt"
	testContent := "line 1\nline 2\nline 3\nline 4\nline 5\n"
	writeArgs, _ := json.Marshal(map[string]any{
		"path":    testPath,
		"content": testContent,
	})
	res, err := ct.WriteFile(writeArgs)
	if err != nil {
		t.Fatalf("WriteFile failed: %v", err)
	}
	if !strings.Contains(res, "35 octets") {
		t.Errorf("WriteFile unexpected output: %s", res)
	}

	// 2. ReadFile avec plage
	readArgs, _ := json.Marshal(map[string]any{
		"path":       testPath,
		"start_line": 2,
		"end_line":   4,
	})
	res, err = ct.ReadFile(readArgs)
	if err != nil {
		t.Fatalf("ReadFile failed: %v", err)
	}
	expectedRead := "   2 | line 2\n   3 | line 3\n   4 | line 4\n"
	if res != expectedRead {
		t.Errorf("ReadFile output mismatch: got %q, want %q", res, expectedRead)
	}

	// 3. PatchFile
	patchArgs, _ := json.Marshal(map[string]any{
		"path":                testPath,
		"target_content":      "line 3\n",
		"replacement_content": "line 3 patched\n",
	})
	res, err = ct.PatchFile(patchArgs)
	if err != nil {
		t.Fatalf("PatchFile failed: %v", err)
	}
	if !strings.Contains(res, "succès") {
		t.Errorf("PatchFile unexpected output: %s", res)
	}

	// Relire pour vérifier le patch
	fullReadArgs, _ := json.Marshal(map[string]any{
		"path": testPath,
	})
	res, err = ct.ReadFile(fullReadArgs)
	if err != nil {
		t.Fatalf("ReadFile after patch failed: %v", err)
	}
	if !strings.Contains(res, "line 3 patched") {
		t.Errorf("Patched content missing in file: %s", res)
	}
}

func TestCodingTools_RunCommand(t *testing.T) {
	tmpDir := t.TempDir()
	ct := NewCodingTools(tmpDir)

	cmdArgs, _ := json.Marshal(map[string]any{
		"command":         "echo 'hello c2slm'",
		"timeout_seconds": 5,
	})
	out, err := ct.RunCommand(cmdArgs)
	if err != nil {
		t.Fatalf("RunCommand failed: %v", err)
	}
	if !strings.Contains(out, "hello c2slm") {
		t.Errorf("RunCommand output unexpected: %s", out)
	}
}

func TestCodingTools_FetchURL(t *testing.T) {
	// Serveur HTTP de test retournant du HTML avec scripts et styles
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, `<!DOCTYPE html>
<html>
<head>
    <title>Doc Go 1.27</title>
    <style>
        body { font-family: sans-serif; }
        .hidden { display: none; }
    </style>
    <script>
        console.log("tracking and ad script to strip");
    </script>
</head>
<body>
    <h1>Documentation c2slm</h1>
    <p>Inféreur pur Go 1.27 &amp; SIMD.</p>
    <div>Support de <b>fetch_url</b> &lt;autonome&gt;!</div>
</body>
</html>`)
	}))
	defer server.Close()

	ct := NewCodingTools(".")
	fetchArgs, _ := json.Marshal(map[string]any{
		"url": server.URL,
	})
	text, err := ct.FetchURL(fetchArgs)
	if err != nil {
		t.Fatalf("FetchURL failed: %v", err)
	}

	if strings.Contains(text, "console.log") {
		t.Errorf("FetchURL did not strip script tags: %s", text)
	}
	if strings.Contains(text, "font-family") {
		t.Errorf("FetchURL did not strip style tags: %s", text)
	}
	if !strings.Contains(text, "Documentation c2slm") {
		t.Errorf("FetchURL missed title/h1: %s", text)
	}
	if !strings.Contains(text, "Inféreur pur Go 1.27 & SIMD.") {
		t.Errorf("FetchURL missed body text: %s", text)
	}
	if !strings.Contains(text, "<autonome>!") {
		t.Errorf("FetchURL did not decode html entities: %s", text)
	}
}
