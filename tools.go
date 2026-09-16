package c2slm

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
)

// ToolDefinition décrit un outil exposé au modèle sous la forme attendue par le
// gabarit ChatML Qwen3 (nom, description, schéma JSON des paramètres).
type ToolDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// ToolHandler exécute un outil à partir des arguments JSON fournis par le
// modèle et retourne le résultat textuel réinjecté dans la conversation.
type ToolHandler func(args json.RawMessage) (string, error)

// ErrUnknownTool est retourné par Dispatch lorsqu'aucun handler n'est enregistré
// sous le nom demandé.
var ErrUnknownTool = errors.New("c2slm: unknown tool")

// ToolRegistry détient les définitions et handlers des outils disponibles. Il
// est sûr en lecture concurrente ; l'enregistrement est sérialisé.
type ToolRegistry struct {
	mu       sync.RWMutex
	order    []string
	defs     map[string]ToolDefinition
	handlers map[string]ToolHandler
}

// NewToolRegistry crée un registre vide.
func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{
		defs:     make(map[string]ToolDefinition),
		handlers: make(map[string]ToolHandler),
	}
}

// Register ajoute un outil. Un nom vide ou un handler nil est refusé. Un nom
// déjà présent est remplacé, sans dupliquer l'ordre d'énumération.
func (r *ToolRegistry) Register(def ToolDefinition, handler ToolHandler) error {
	name := strings.TrimSpace(def.Name)
	if name == "" {
		return errors.New("c2slm: tool name must not be empty")
	}
	if handler == nil {
		return fmt.Errorf("c2slm: tool %q has no handler", name)
	}
	def.Name = name
	def.Description = strings.TrimSpace(def.Description)
	if len(def.Parameters) == 0 {
		def.Parameters = json.RawMessage(`{"type":"object","properties":{}}`)
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.defs[name]; !exists {
		r.order = append(r.order, name)
	}
	r.defs[name] = def
	r.handlers[name] = handler
	return nil
}

// Len retourne le nombre d'outils enregistrés.
func (r *ToolRegistry) Len() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.order)
}

// Definitions retourne les définitions dans l'ordre d'enregistrement.
func (r *ToolRegistry) Definitions() []ToolDefinition {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]ToolDefinition, 0, len(r.order))
	for _, name := range r.order {
		out = append(out, r.defs[name])
	}
	return out
}

// Handler retourne le handler associé à un nom.
func (r *ToolRegistry) Handler(name string) (ToolHandler, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	h, ok := r.handlers[name]
	return h, ok
}

// Dispatch exécute l'outil nommé avec les arguments fournis.
func (r *ToolRegistry) Dispatch(name string, args json.RawMessage) (string, error) {
	h, ok := r.Handler(name)
	if !ok {
		return "", fmt.Errorf("%w: %q", ErrUnknownTool, name)
	}
	if len(args) == 0 {
		args = json.RawMessage(`{}`)
	}
	return h(args)
}

// toolWireFormat est la représentation JSON d'un outil dans le bloc <tools>.
type toolWireFormat struct {
	Type     string       `json:"type"`
	Function toolWireFunc `json:"function"`
}

type toolWireFunc struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// SystemToolsBlock compose le bloc ChatML Qwen3 inséré dans le message système :
//
//	# Tools
//
//	...
//	<tools>
//	{"type":"function","function":{...}}
//	</tools>
//	...
//
// Le bloc est vide si le registre ne contient aucun outil.
func (r *ToolRegistry) SystemToolsBlock() string {
	defs := r.Definitions()
	if len(defs) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("# Tools\n\n")
	b.WriteString("You may call one or more functions to assist with the user query.\n\n")
	b.WriteString("You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n")
	for _, d := range defs {
		payload, err := json.Marshal(toolWireFormat{
			Type: "function",
			Function: toolWireFunc{
				Name:        d.Name,
				Description: d.Description,
				Parameters:  d.Parameters,
			},
		})
		if err != nil {
			continue
		}
		b.Write(payload)
		b.WriteByte('\n')
	}
	b.WriteString("</tools>\n\n")
	b.WriteString("For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n")
	b.WriteString("<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n")
	return b.String()
}

// ToolCall est un appel d'outil extrait du flux du modèle.
type ToolCall struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// ParseToolCall extrait un appel d'outil depuis la charge brute d'un bloc
// <tool_call>. Elle accepte un objet unique ou un tableau d'appels, et tolère
// des « arguments » encodés en chaîne JSON. Le premier appel exploitable est
// retourné.
func ParseToolCall(raw []byte) (ToolCall, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return ToolCall{}, errors.New("c2slm: empty tool call")
	}
	if trimmed[0] == '[' {
		var items []json.RawMessage
		if err := json.Unmarshal(trimmed, &items); err != nil {
			return ToolCall{}, fmt.Errorf("c2slm: invalid tool call array: %w", err)
		}
		var lastErr error
		for _, item := range items {
			call, err := ParseToolCall(item)
			if err == nil {
				return call, nil
			}
			lastErr = err
		}
		if lastErr == nil {
			lastErr = errors.New("c2slm: empty tool call array")
		}
		return ToolCall{}, lastErr
	}

	var wire struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal(trimmed, &wire); err != nil {
		return ToolCall{}, fmt.Errorf("c2slm: invalid tool call: %w", err)
	}
	if strings.TrimSpace(wire.Name) == "" {
		return ToolCall{}, errors.New("c2slm: tool call without name")
	}
	args := bytes.TrimSpace(wire.Arguments)
	if len(args) == 0 {
		args = []byte("{}")
	}
	// Certains modèles encodent les arguments en chaîne JSON : on déballe.
	if args[0] == '"' {
		var inner string
		if err := json.Unmarshal(args, &inner); err == nil {
			inner = strings.TrimSpace(inner)
			if inner != "" {
				args = []byte(inner)
			}
		}
	}
	return ToolCall{Name: strings.TrimSpace(wire.Name), Arguments: json.RawMessage(args)}, nil
}

// ToolRoundRecord retrace une invocation d'outil exécutée par la boucle
// agentique.
type ToolRoundRecord struct {
	Round  int             `json:"round"`
	Name   string          `json:"name"`
	Args   json.RawMessage `json:"args,omitempty"`
	Result string          `json:"result,omitempty"`
	Err    string          `json:"error,omitempty"`
}

// DefaultMaxToolRounds borne le nombre d'appels d'outil d'un tour agentique.
const DefaultMaxToolRounds = 8

// GenerateWithTools exécute un tour agentique complet : le modèle est invité
// avec le bloc d'outils, ses éventuels appels sont exécutés, leurs résultats
// réinjectés sous <tool_response>, et l'inférence reprend tant qu'un outil est
// demandé ou que le budget de tours n'est pas épuisé.
//
// Le texte visible (hors raisonnement et hors appels d'outil) est accumulé et
// retourné, ainsi que la trace des invocations.
func (e *Engine) GenerateWithTools(systemPrompt, userPrompt string, maxNewTokens, maxToolRounds int, reg *ToolRegistry) (string, []ToolRoundRecord, error) {
	return e.GenerateWithToolsStream(systemPrompt, userPrompt, maxNewTokens, maxToolRounds, reg, nil)
}

// GenerateWithToolsStream est la variante streaming de GenerateWithTools. Le
// callback onToken reçoit, au vol, uniquement le texte utilisateur épuré ; il
// retourne faux pour interrompre immédiatement l'inférence.
func (e *Engine) GenerateWithToolsStream(systemPrompt, userPrompt string, maxNewTokens, maxToolRounds int, reg *ToolRegistry, onToken func(cleanPiece string) bool) (string, []ToolRoundRecord, error) {
	if reg == nil {
		text, err := e.GenerateStream(buildToolsPrompt(systemPrompt, nil, userPrompt), maxNewTokens, nil, onToken)
		return text, nil, err
	}
	if maxToolRounds <= 0 {
		maxToolRounds = DefaultMaxToolRounds
	}

	transcript := buildToolsPrompt(systemPrompt, reg, userPrompt)
	scanner := NewToolScanner()

	var visible strings.Builder
	var records []ToolRoundRecord

	for round := 0; ; round++ {
		scanner.Reset()
		var rawCall string
		gotTool := false

		_, err := e.GenerateStream(transcript, maxNewTokens, nil, func(piece string) bool {
			done, _, clean := scanner.Push(piece)
			if clean != "" {
				visible.WriteString(clean)
				if onToken != nil && !onToken(clean) {
					return false
				}
			}
			if done {
				rawCall = scanner.ToolJSON()
				gotTool = true
				return false
			}
			return true
		})
		if err != nil {
			return visible.String(), records, err
		}
		if !gotTool {
			break
		}
		if len(records) >= maxToolRounds {
			break
		}

		rec := ToolRoundRecord{Round: round, Name: ""}
		call, parseErr := ParseToolCall([]byte(rawCall))
		var result string
		if parseErr != nil {
			rec.Err = parseErr.Error()
			result = errorToolResponse(parseErr)
		} else {
			rec.Name = call.Name
			rec.Args = call.Arguments
			out, dispatchErr := reg.Dispatch(call.Name, call.Arguments)
			if dispatchErr != nil {
				rec.Err = dispatchErr.Error()
				result = errorToolResponse(dispatchErr)
			} else {
				rec.Result = out
				result = out
			}
		}
		records = append(records, rec)
		transcript += toolCallTurn(rawCall, result)
	}

	return visible.String(), records, nil
}

// buildToolsPrompt assemble le prompt ChatML plat attendu par l'inféreur :
//
//	system\n<system>\n\n<bloc outils>\nuser\n<user>\nassistant\n
func buildToolsPrompt(systemPrompt string, reg *ToolRegistry, userPrompt string) string {
	var b strings.Builder
	b.Grow(len(systemPrompt) + len(userPrompt) + 640)
	b.WriteString("<|im_start|>system\n")
	if systemPrompt != "" {
		b.WriteString(systemPrompt)
	}
	if reg != nil {
		if block := reg.SystemToolsBlock(); block != "" {
			if systemPrompt != "" {
				b.WriteString("\n\n")
			}
			b.WriteString(block)
		}
	}
	b.WriteString("<|im_end|>\n<|im_start|>user\n")
	b.WriteString(userPrompt)
	b.WriteString("<|im_end|>\n<|im_start|>assistant\n")
	return b.String()
}

// toolCallTurn réinjecte l'appel d'outil et son résultat dans la transcription,
// selon la forme imposée par le gabarit : le résultat est un tour « user »
// encadré par <tool_response>, suivi de l'amorce « assistant » du tour suivant.
func toolCallTurn(rawCall, result string) string {
	var b strings.Builder
	b.Grow(len(rawCall) + len(result) + 96)
	b.WriteString(toolCallOpenTag)
	b.WriteByte('\n')
	b.WriteString(strings.TrimSpace(rawCall))
	b.WriteByte('\n')
	b.WriteString(toolCallCloseTag)
	b.WriteString("<|im_end|>\n<|im_start|>user\n<tool_response>\n")
	b.WriteString(result)
	b.WriteString("\n</tool_response><|im_end|>\n<|im_start|>assistant\n")
	return b.String()
}

func errorToolResponse(err error) string {
	payload, mErr := json.Marshal(struct {
		Error string `json:"error"`
	}{Error: err.Error()})
	if mErr != nil {
		return `{"error":"tool execution failed"}`
	}
	return string(payload)
}
