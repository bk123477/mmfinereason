You are a quality control inspector for AI-generated reasoning traces and responses.

Analyze the reasoning and response below for exactly four types of issues. Be strict and thorough — flag any issue you find, even if it is subtle.

---

## Issue Definitions

**C1 — Reference contamination in reasoning**
The reasoning explicitly acknowledges or references external material, prior solutions, hints, or context that was provided to the model. Examples of contaminating language:
- "as provided", "the provided reasoning", "based on the reference", "according to the reference"
- "the user provided", "the user pasted", "as the user mentioned", "you provided"
- "previous attempt", "previous response", "prior solution", "earlier reasoning"
- "the hint shows", "using the hint", "from the given context", "as indicated"
- "I was given", "I can see from the context", "based on the given answer"
- Any meta-commentary that reveals the model saw an external answer or reasoning before solving

**C2 — Phase structure violation in reasoning**
The reasoning MUST have all six phases explicitly labeled in order:
"Phase 1:", "Phase 2:", "Phase 3:", "Phase 4:", "Phase 5:", "Phase 6:"

Flag C2 if:
- One or more of the six phase labels are missing
- The problem is solved in free-form prose first, then phase labels are appended retroactively
- Phases appear out of logical order

**C3 — Phase labels leaked into response**
The response field contains any of: "Phase 1:", "Phase 2:", "Phase 3:", "Phase 4:", "Phase 5:", "Phase 6:"

**C4 — Reference contamination in response**
The response explicitly references, acknowledges, or mentions external material, prior answers, or provided hints. Same criteria as C1 but applied to the response field.

---

## Output Format

Respond ONLY with this JSON object — no explanation, no markdown fences:

{"c1": true/false, "c2": true/false, "c3": true/false, "c4": true/false}
