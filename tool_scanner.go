package c2slm

import "unsafe"

// ScanState est l'état de la machine à états finis qui découpe le flux de tokens
// du modèle en texte utilisateur, bloc de raisonnement et appel d'outil.
type ScanState uint8

const (
	// StText : texte ordinaire visible par l'usager.
	StText ScanState = iota
	// StThink : contenu de  thinking ... </think>, capté en interne.
	StThink
	// StToolJSON : charge JSON de <tool_call> ... </tool_call>, captée en interne.
	StToolJSON
)

// Balises ChatML Qwen3 reconnues par le scanner. Ce sont les formes exactes du
// gabarit Qwen3, sans espace de tête.
const (
	toolCallOpenTag  = "<tool_call>"
	toolCallCloseTag = "</tool_call>"
	thinkOpenTag     = "<" + "think" + ">"
	thinkCloseTag    = "<" + "/think" + ">"
)

// Capacités fixes. Toute la mémoire du scanner est préallouée : le chemin chaud
// (Push) ne touche jamais le tas.
const (
	maxToolJSONBytes = 4096
	maxThinkBytes    = 8192
	scannerOutBytes  = 8192
	tagPendingBytes  = 32
)

// ToolScanner découpe, token par token, le flux brut d'un modèle ChatML Qwen3.
//
// Invariant d'allocation : Push ne réalise aucune allocation sur le tas. Le
// cleanPiece retourné aliase le tampon interne du scanner (via unsafe.String) et
// n'est donc valide que jusqu'à l'appel suivant à Push ou à Reset. L'appelant
// doit le consommer (copie, écriture dans un builder, impression) avant de
// rappeler Push.
type ToolScanner struct {
	state      ScanState
	pending    [tagPendingBytes]byte
	pendingLen int

	jsonBuf  [maxToolJSONBytes]byte
	jsonLen  int
	thinkBuf [maxThinkBytes]byte
	thinkLen int
	out      [scannerOutBytes]byte
	outLen   int

	toolDone bool
	overflow bool
}

// NewToolScanner crée un scanner à l'état de texte, tampons préalloués.
func NewToolScanner() *ToolScanner {
	return &ToolScanner{}
}

// Reset replace le scanner dans son état initial sans libérer ni réallouer.
func (s *ToolScanner) Reset() {
	s.state = StText
	s.pendingLen = 0
	s.jsonLen = 0
	s.thinkLen = 0
	s.outLen = 0
	s.toolDone = false
	s.overflow = false
}

// State retourne l'état courant de la FSM.
func (s *ToolScanner) State() ScanState { return s.state }

// Overflow indique qu'un tampon borné (raisonnement ou JSON d'outil) a débordé.
// Le contenu excédentaire est alors abandonné, jamais réalloué.
func (s *ToolScanner) Overflow() bool { return s.overflow }

// ToolJSONBytes retourne la charge JSON capturée entre <tool_call> et
// </tool_call>. Le slice aliase le tampon interne : valide jusqu'au prochain
// Reset.
func (s *ToolScanner) ToolJSONBytes() []byte { return s.jsonBuf[:s.jsonLen] }

// ToolJSON copie la charge JSON capturée sous forme de chaîne.
func (s *ToolScanner) ToolJSON() string { return string(s.jsonBuf[:s.jsonLen]) }

// ThinkBytes retourne le contenu de raisonnement capté. Le slice aliase le
// tampon interne.
func (s *ToolScanner) ThinkBytes() []byte { return s.thinkBuf[:s.thinkLen] }

// ThinkText copie le contenu de raisonnement sous forme de chaîne.
func (s *ToolScanner) ThinkText() string { return string(s.thinkBuf[:s.thinkLen]) }

// Push consomme un fragment de texte produit par GenerateStream et retourne :
//
//   - done : vrai lorsque </tool_call> vient d'être rencontré ; la charge est
//     alors disponible via ToolJSON ;
//   - isTool : vrai lorsque le fragment appartient à un appel d'outil (capture
//     en cours) ou vient de le clôturer, plutôt qu'au texte utilisateur ;
//   - cleanPiece : le texte visible par l'usager émis par ce fragment, hors
//     raisonnement et hors charge d'outil.
func (s *ToolScanner) Push(piece string) (done bool, isTool bool, cleanPiece string) {
	s.outLen = 0
	s.toolDone = false

	for i := 0; i < len(piece); i++ {
		s.feed(piece[i])
	}

	if s.outLen > 0 {
		cleanPiece = unsafeString(s.out[:s.outLen])
	}
	return s.toolDone, s.state == StToolJSON || s.toolDone, cleanPiece
}

// feed traite un octet selon l'état courant. Un octet qui ne commence aucune
// balise potentielle est routé directement vers la destination de l'état ; tout
// « < » amorce une capture temporaire dans pending afin de reconnaître une
// balise coupée entre deux fragments.
func (s *ToolScanner) feed(b byte) {
	switch s.state {
	case StText:
		if s.pendingLen == 0 && b != '<' {
			s.appendOut(b)
			return
		}
		s.pushPending(b)
		s.resolve()
	case StThink:
		if s.pendingLen == 0 && b != '<' {
			s.appendThink(b)
			return
		}
		s.pushPending(b)
		s.resolve()
	case StToolJSON:
		if s.pendingLen == 0 && b != '<' {
			s.appendJSON(b)
			return
		}
		s.pushPending(b)
		s.resolve()
	}
}

func (s *ToolScanner) pushPending(b byte) {
	if s.pendingLen >= len(s.pending) {
		// Ne peut survenir qu'après un bug : les balises font au plus 11 octets.
		// On évacue un octet pour rester borné, sans réallouer.
		s.flushPendingFront()
	}
	s.pending[s.pendingLen] = b
	s.pendingLen++
}

// resolve examine pending tant qu'il est non vide. Trois issues :
//   - préfixe d'une balise candidate : attente du fragment suivant ;
//   - correspondance exacte : transition d'état ;
//   - non-correspondance : l'octet de tête est routé vers la destination de
//     l'état et l'analyse reprend, ce qui gère « < » littéral et « < » suivi
//     d'une autre balise.
func (s *ToolScanner) resolve() {
	for s.pendingLen > 0 {
		action, isPrefix := s.classifyPending()
		if action != tagNoMatch {
			s.pendingLen = 0
			s.applyAction(action)
			return
		}
		if isPrefix {
			return
		}
		s.flushPendingFront()
	}
}

type tagAction uint8

const (
	tagNoMatch tagAction = iota
	tagMatchedToolOpen
	tagMatchedThinkOpen
	tagMatchedToolClose
	tagMatchedThinkClose
)

// classifyPending confronte pending à l'ensemble des balises candidates de
// l'état courant. isPrefix vaut vrai tant que pending peut encore mener à une
// balise (y compris la correspondance exacte).
func (s *ToolScanner) classifyPending() (action tagAction, isPrefix bool) {
	p := s.pending[:s.pendingLen]
	switch s.state {
	case StText:
		if hasPrefixBytes(p, toolCallOpenTag) {
			if len(p) == len(toolCallOpenTag) {
				return tagMatchedToolOpen, true
			}
			return tagNoMatch, true
		}
		if hasPrefixBytes(p, thinkOpenTag) {
			if len(p) == len(thinkOpenTag) {
				return tagMatchedThinkOpen, true
			}
			return tagNoMatch, true
		}
	case StThink:
		if hasPrefixBytes(p, thinkCloseTag) {
			if len(p) == len(thinkCloseTag) {
				return tagMatchedThinkClose, true
			}
			return tagNoMatch, true
		}
	case StToolJSON:
		if hasPrefixBytes(p, toolCallCloseTag) {
			if len(p) == len(toolCallCloseTag) {
				return tagMatchedToolClose, true
			}
			return tagNoMatch, true
		}
	}
	return tagNoMatch, false
}

func (s *ToolScanner) applyAction(action tagAction) {
	switch action {
	case tagMatchedToolOpen:
		s.jsonLen = 0
		s.state = StToolJSON
	case tagMatchedThinkOpen:
		s.thinkLen = 0
		s.state = StThink
	case tagMatchedThinkClose:
		s.state = StText
	case tagMatchedToolClose:
		s.state = StText
		s.toolDone = true
	}
}

// flushPendingFront route l'octet de tête de pending vers la destination de
// l'état courant, puis décale pending d'un cran.
func (s *ToolScanner) flushPendingFront() {
	if s.pendingLen == 0 {
		return
	}
	b := s.pending[0]
	switch s.state {
	case StThink:
		s.appendThink(b)
	case StToolJSON:
		s.appendJSON(b)
	default:
		s.appendOut(b)
	}
	copy(s.pending[:], s.pending[1:s.pendingLen])
	s.pendingLen--
}

func (s *ToolScanner) appendOut(b byte) {
	if s.outLen >= len(s.out) {
		s.overflow = true
		return
	}
	s.out[s.outLen] = b
	s.outLen++
}

func (s *ToolScanner) appendThink(b byte) {
	if s.thinkLen >= len(s.thinkBuf) {
		s.overflow = true
		return
	}
	s.thinkBuf[s.thinkLen] = b
	s.thinkLen++
}

func (s *ToolScanner) appendJSON(b byte) {
	if s.jsonLen >= len(s.jsonBuf) {
		s.overflow = true
		return
	}
	s.jsonBuf[s.jsonLen] = b
	s.jsonLen++
}

// hasPrefixBytes teste si p est un préfixe de s (comparaison d'octets, sans
// allocation). p plus long que s rend faux.
func hasPrefixBytes(p []byte, s string) bool {
	if len(p) > len(s) {
		return false
	}
	for i := 0; i < len(p); i++ {
		if p[i] != s[i] {
			return false
		}
	}
	return true
}

// unsafeString convertit un slice en chaîne sans copie. Le résultat aliase le
// tampon sous-jacent : l'appelant ne doit pas le conserver au-delà de la
// prochaine mutation du buffer.
func unsafeString(b []byte) string {
	if len(b) == 0 {
		return ""
	}
	return unsafe.String(&b[0], len(b))
}
