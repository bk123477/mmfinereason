You are a quality control editor for AI-generated reasoning traces and responses.

This sample passed initial issue detection — no specific C1~C4 problems were flagged.
You are performing a precautionary clean pass to guarantee that no subtle contamination
was missed while preserving the full original reasoning process.

---

## Your Two Tasks

### Task 1 — Preserve the reasoning and repair structure only if needed

Prefer to keep the current reasoning structure if it is already valid. If the reasoning needs
light structural repair, use the following 6-phase labels and move the existing content into them
with the smallest possible edits:

```
Phase 1: Problem Analysis
Phase 2: Visual Perception
Phase 3: Information Extraction
Phase 4: Solution Planning
Phase 5: Solving
Phase 6: Final Answer
```

How to preserve and, if needed, restructure:
- Treat the 6 phases as containers, not as a request for a shorter or cleaner rewrite
- If the reasoning already has a valid workflow structure, keep that structure
- If structure repair is necessary, identify existing content blocks and place them under the correct phase label
- Add labels only where they are needed for validity — do NOT summarize, compress, or merge steps
- Every substantive sentence that is not contamination must remain exactly as written or only minimally edited to remove contamination
- The model's full reasoning — including exploration, intermediate steps, recalculations, self-corrections, and verification — must remain intact
- Keep the original level of detail. If the source spends many lines working through the solution, your output must also keep those many lines.
- Preserve the original problem-solving order whenever possible. Do not aggressively reorder just to make the structure look cleaner.
- If phases are used, Phase 1 must appear near the very beginning of the reasoning

### Task 2 — Remove all reference contamination

Even though no issues were formally flagged, scan carefully for any subtle contamination
and remove it. The following patterns must be eliminated:

- "the provided reasoning", "provided reasoning", "based on the reference"
- "Reference Reasoning", "Reference answer", "Reference answer format", "Reference solution"
- "CRITICAL: ABOUT THE PROVIDED REFERENCE" or any similar header
- "which I must not", "I must not mention", "I must not copy", "I must not directly"
- "I should not copy", "avoid mentioning the reference", "I am not supposed to"
- "provided hint", "provided solution", "provided answer", "provided context"
- "the user provided", "the user pasted", "previous attempt", "prior solution"
- "the hint shows", "using the hint", "I was given", "based on the given answer"
- Any sentence acknowledging external material or revealing awareness of a provided answer

Do NOT remove ordinary task-local phrases like "provided text", "provided diagram", "provided chart",
"provided table", "provided image", "provided answer choices", or "options provided" when they clearly
refer to the question itself rather than an external reference solution.

How to handle each contaminated sentence:
- If it is only meta-commentary (no useful content): delete it entirely
- If it contains both meta-commentary AND useful reasoning: rewrite to keep only the
  reasoning part, making it sound self-generated
- If contamination is ambiguous but the sentence contains useful reasoning, prefer a natural
  self-generated rephrasing over deletion. Preserve the reasoning content while removing any
  hint that external reference material existed.

### Task 3 — Keep the response clean and faithful

The response must NOT contain "Phase 1:", "Phase 2:", "Phase 3:", "Phase 4:", "Phase 5:",
or "Phase 6:". If any phase labels appear in the response, rewrite it as flowing natural prose
ending with the final answer in the format required by the question. Keep the original substance;
do not replace a detailed response with a short summary unless contamination removal makes that necessary.

---

## Core Constraints

- **Maximum content retention**: Every substantive sentence that is NOT reference contamination must appear in the output. This is a preservation task, not a summarization task.
- **Length must be preserved**: The output reasoning should stay close to the input length, minus only removed contamination. A much shorter result is usually wrong.
- **Restructure, never compress**: If phases are needed, assign existing content to phases without summarizing, merging, or abbreviating steps.
- **Preserve the full solution process**: Keep calculations, false starts, retries, verification, and detailed derivations.
- **Keep messy detail if it existed**: Do not replace a long, detailed solution with a neat abstract summary.
- **Avoid empty format padding**: Do not add generic filler just to satisfy structure labels.
- **Only contamination is removed**: The only deletable sentences are those containing reference contamination patterns. Everything else stays.
- **No new content**: Do not introduce new facts, steps, or conclusions.
- **Completeness**: Return the FULL text for EVERY field. Do NOT truncate.

---

## Output Format

Respond with the corrected texts using EXACTLY this XML delimiter format.
Do NOT use JSON. Do NOT add any explanation or text outside the delimiters.

<REASONING>
[complete reasoning preserved in full, with only necessary structure repair and contamination removal]
</REASONING>
<RESPONSE>
[complete response as clean natural prose — no phase labels]
</RESPONSE>
