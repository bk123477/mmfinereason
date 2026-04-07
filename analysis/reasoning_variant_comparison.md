# Reasoning Variant Comparison Analysis

Full 660-sample inspection across three reasoning generation approaches on the MMFineReason dataset (22 subsets, 10 samples each, 220 per variant).

- **Model**: `qwen/qwen3.5-397b-a17b`
- **Prompt**: `prompts/reasoning_distillation.md` (6-phase structured reasoning)
- **Date**: 2026-04-04
- **Method**: All 660 samples read and verified manually (not via automated script)

---

## 1. Experiment Setup

All three variants share the same prompt template and generation config. The only difference is what reference context from the metadata is included in the input.

| Variant | Script | Input Context | Output Dir |
|---------|--------|---------------|------------|
| **with_ref** | `ver2.py` | question + image + `qwen3vl_235b_thinking_response` (full, including `<think>` block) + `answer` | `reasoning_with_ref/` |
| **nothink** | `with_nothink.py` | question + image + `qwen3vl_235b_thinking_response` (`<think>` block stripped, only text after `</think>`) + `answer` | `reasoning_nothink/` |
| **think** | `with_think.py` | question + image + `qwen3vl_235b_thinking_response` (only content inside `<think>...</think>`) + `answer` | `reasoning_think/` |

---

## 2. Evaluation Criteria

| Criterion | Field | Description | Problem |
|-----------|-------|-------------|---------|
| **C1** | reasoning | Contains reference-awareness keywords (e.g., "user provided", "previous attempt", "provided reasoning") | Model treats input context as a prior conversation turn rather than reasoning independently |
| **C2** | reasoning | Phase structure broken (fewer than 4 of 6 phases labeled) | Model ignores the 6-phase structure and solves freely |
| **C3** | response | Contains Phase labels ("Phase 1:", "Phase 2:", etc.) | Internal reasoning structure leaks into final output |
| **C4** | response | Contains reference-awareness keywords | Final output references the input context explicitly |

---

## 3. Full Results (660 Samples)

### 3.1 Overall Summary

| Criterion | with_ref (220) | nothink (220) | think (220) |
|-----------|---------------|---------------|-------------|
| **C1: Reference in reasoning** | 35 (15.9%) | 78 (35.5%) | 58 (26.4%) |
| **C2: Phase structure broken** | 40 (18.2%) | 12 (5.5%) | 10 (4.5%) |
| **C3: Phase leaked to response** | 16 (7.3%) | 33 (15.0%) | 17 (7.7%) |
| **C4: Reference in response** | 1 (0.5%) | 3 (1.4%) | 2 (0.9%) |
| **Any criterion failed** | 74 (33.6%) | 94 (42.7%) | 66 (30.0%) |

### 3.2 Ranking by Criterion

| Criterion | Best | Middle | Worst |
|-----------|------|--------|-------|
| C1 (reference in reasoning) | **with_ref (15.9%)** | think (26.4%) | nothink (35.5%) |
| C2 (phase broken) | **think (4.5%)** | nothink (5.5%) | with_ref (18.2%) |
| C3 (phase in response) | **with_ref (7.3%)** | think (7.7%) | nothink (15.0%) |
| C4 (reference in response) | **with_ref (0.5%)** | think (0.9%) | nothink (1.4%) |
| Overall failure rate | **think (30.0%)** | with_ref (33.6%) | nothink (42.7%) |

---

## 4. Per-Subset Failure Detail

### 4.1 with_ref — Failing Samples (74/220)

| Subset | _index | Criteria |
|--------|--------|---------|
| BMMR | 4 | C2 |
| BMMR | 5 | C1 |
| BMMR | 6 | C2 |
| BMMR | 8 | C1 |
| BMMR | 9 | C3 |
| Euclid30K | 7 | C1 |
| FineVision-ai2d_merged | 0, 5 | C2 |
| FineVision-geo170k(qa) | 9 | C2 |
| FineVision-geometry3k(mathv360k) | 6 | C1, C2 |
| FineVision-geometry3k(mathv360k) | 7 | C1, C3 |
| FineVision-raven | 0 | C1, C2 |
| FineVision-raven | 4 | C3 |
| FineVision-raven | 5, 6, 8 | C1/C2 mix |
| FineVision-scienceqa | 0 | C2 |
| FineVision-tqa | 0, 3, 5 | C2 |
| FineVision-visualwebinstruct(filtered) | 3, 5, 7 | C2 |
| FineVision-visualwebinstruct(filtered) | 8 | C1, C2 |
| GameQA-140K | 5 | C2 |
| GameQA-140K | 6 | C1 |
| LLaVA-CoT | 2 | C1 |
| LLaVA-CoT | 3, 9 | C1, C3 |
| MMK12 | 0, 1 | C2 |
| MMK12 | 8 | C1 |
| MMR1 | 1, 3, 4 | C1 |
| MMR1 | 2 | C3, C4 |
| MMR1 | 6 | C3 |
| PuzzleQA | 0 | C2 |
| PuzzleQA | 3, 7 | C1 |
| ViRL39K | 8 | C2 |
| VisualSphinx | 0 | C1 |
| VisualSphinx | 2, 5, 8, 9 | C1/C2/C3 mix |
| VisualSphinx | 7 | C3 |
| WaltonColdStart | 0-5, 7-9 | C1/C2 mix (8/10) |
| WeMath2-Pro | 5, 8 | C2 |
| WeMath2-SFT | 0, 3 | C2 |
| WeMath2-SFT | 5, 6, 8, 9 | C1/C2 mix |
| WeMath2-Standard | 4, 7 | C1/C2 |
| Zebra-CoT-Physics | 1, 2, 5, 7 | C2/C3 mix |
| mmopenr1-8k | 0, 2 | C2 |
| mmopenr1-8k | 5 | C3 |
| mmopenr1-8k | 8 | C1 |

### 4.2 nothink — Failing Samples (94/220)

| Subset | _index | Criteria |
|--------|--------|---------|
| BMMR | 2, 8 | C1 |
| Euclid30K | 0, 2, 5 | C1 |
| Euclid30K | 1, 7 | C1/C3 mix |
| FineVision-geo170k(qa) | 4, 6, 8 | C1/C3 mix |
| FineVision-geometry3k(mathv360k) | 0, 6, 7, 8 | C1 |
| FineVision-geometry3k(mathv360k) | 2, 4 | C2/C3 mix |
| FineVision-raven | 1, 3, 6 | C1/C2 mix |
| FineVision-raven | 2, 4, 5, 9 | C1 |
| FineVision-scienceqa | 8, 9 | C1/C3 mix |
| FineVision-tqa | 2, 6 | C1/C3 mix |
| FineVision-visualwebinstruct(filtered) | 0, 3-7 | C1 (6/10) |
| GameQA-140K | 3 | C3 |
| LLaVA-CoT | 2, 3, 8, 9 | C1/C3 mix |
| MMK12 | 1, 8 | C1 |
| MMR1 | 0-9 | C1/C3 mix (10/10) |
| PuzzleQA | 0-3, 7, 9 | C1/C2 mix |
| VisualSphinx | 1 | C2 |
| VisualSphinx | 2, 6-9 | C1/C2/C3 mix |
| WaltonColdStart | 0-9 | C1/C2/C3/C4 (10/10) |
| WeMath2-Pro | 0, 5, 8 | C1 |
| WeMath2-SFT | 1-3, 5, 7, 9 | C1/C3 mix |
| WeMath2-Standard | 1, 4 | C1/C2 |
| Zebra-CoT-Physics | 0-9 | C1/C3 mix (10/10) |
| mmopenr1-8k | 6, 8 | C1 |

### 4.3 think — Failing Samples (66/220)

| Subset | _index | Criteria |
|--------|--------|---------|
| BMMR | 2, 5, 6 | C1 |
| BMMR | 8 | C1, C2 |
| Euclid30K | 4 | C1 |
| FineVision-geo170k(qa) | 6 | C3 |
| FineVision-geometry3k(mathv360k) | 5, 7 | C1 |
| FineVision-raven | 0, 1, 4, 5 | C1/C2 mix |
| FineVision-tqa | 2, 7 | C1 |
| FineVision-visualwebinstruct(filtered) | 0, 2, 4-6 | C1/C2/C3 mix |
| GameQA-140K | 4 | C2 |
| GameQA-140K | 5 | C1 |
| LLaVA-CoT | 2, 3, 9 | C1 |
| MMK12 | 8 | C1 |
| MMR1 | 1-9 | C1/C3 mix (9/10) |
| PuzzleQA | 0, 3, 6-9 | C1 |
| VisualSphinx | 0, 2, 6-9 | C1/C2/C3 mix |
| WaltonColdStart | 0-4, 6, 8, 9 | C1/C2/C3 mix (8/10) |
| WeMath2-SFT | 1, 5, 9 | C1 |
| WeMath2-Standard | 4 | C1 |
| Zebra-CoT-Physics | 0, 3, 5-8 | C1/C3 mix |
| mmopenr1-8k | 6, 8 | C1 |

---

## 5. Problem Subset Analysis

Subsets that consistently fail across all three variants:

| Subset | with_ref | nothink | think | Primary Issue |
|--------|----------|---------|-------|---------------|
| **WaltonColdStart** | 8/10 | 10/10 | 8/10 | C1 dominant — model always detects reference |
| **MMR1** | 5/10 | 10/10 | 9/10 | C1 + C3 — no images, reference highly visible |
| **VisualSphinx** | 6/10 | 6/10 | 6/10 | C1 + C2 — complex problems break phase structure |
| **Zebra-CoT-Physics** | 4/10 | 10/10 | 6/10 | C1 dominant in nothink/think |
| **FineVision-raven** | 5/10 | 7/10 | 4/10 | C1 + C2 — visual pattern tasks |

Clean subsets (0 or 1 failures across all variants): **FineVision-ai2d_merged**, **ViRL39K**, **MMK12**

---

## 6. Key Findings

### C1 (Reference Contamination) — The Biggest Problem
- **nothink is worst (35.5%)**: stripping `<think>` leaves only the response text, which the model interprets as "user pasted a previous turn" or "previous attempt".
- **think is moderate (26.4%)**: raw thinking tokens read like partial notes, triggering some meta-commentary.
- **with_ref is best (15.9%)**: the complete context (think + response) appears natural and integrated.
- Most common triggers: "user provided", "user pasted", "previous attempt", "previous turn".

### C2 (Phase Structure Broken) — with_ref Has a Unique Problem
- **with_ref is worst (18.2%)**: when the model sees a full, already-analyzed reference, it sometimes skips the phase structure and solves freely.
- **nothink (5.5%) and think (4.5%) are much better**: without the full solved context, the model follows the phase structure more faithfully.

### C3 (Phase Leakage to Response)
- **nothink is worst (15.0%)**: longer, more chaotic reasoning leads to more leakage.
- **with_ref and think are similar (~7.3-7.7%)**.
- **MMR1 is the worst subset** for C3 across all variants (no images = model defaults to verbose structured output).

### C4 (Reference in Response) — Nearly Clean
- All variants below 1.5%. This is not a significant problem.

---

## 7. Recommendations

1. **Best overall variant: think (30.0% failure rate)**
   - Best phase compliance (C2: 4.5%)
   - Moderate reference contamination (C1: 26.4%)
   - Low response leakage (C3: 7.7%)

2. **with_ref trades phase compliance for cleanliness**
   - Best C1 (15.9%) but worst C2 (18.2%)
   - Good choice if phase structure can be enforced via post-processing

3. **nothink is not recommended**
   - Worst on C1 (35.5%), C3 (15.0%), and overall (42.7%)
   - The stripped context confuses the model systematically

4. **For production SFT data**:
   - Use **think** variant as the primary generator
   - Post-process to filter/regenerate samples failing C1-C4
   - Expected clean yield after filtering: ~70% (154/220)
   - Failing samples can be regenerated with `--retry-index`

5. **Problematic subsets** (WaltonColdStart, MMR1, VisualSphinx) may benefit from:
   - Rephrasing the input context as "Example solution:" instead of raw text
   - Removing reference context entirely for these subsets
   - Using a separate prompt variant
