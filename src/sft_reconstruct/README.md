# sft_reconstruct

Async reasoning reconstruction pipeline for the full MMFineReason dataset (70 parquet files).

Takes the existing `qwen3vl_235b_thinking_response` column from each parquet and
restructures it into one of 10 reasoning workflows — no new reasoning generation,
no image access required.

> **When to use which script:**
> - `src/sft_reconstruct/sft_reconstruct.py` — **Full production run** over all 70 parquets.
>   Includes error log, `--retry-errors`, per-part isolated output, and append-only summary.
> - `src/sft_sample/reconstruct.py` — **Quick pilot test** on a small subset.
>   Lightweight (no error log), timestamped output dir, flush-on-every-success.

---

## How It Works

```
parquet row
  └─ qwen3vl_235b_thinking_response
       ├─ extract_think_block()   → existing reasoning
       └─ extract_response_block() → existing response
            │
            ▼
       LLM (reconstruct_prompt + reasoning_workflows)
            │
            ▼
       <WORKFLOW> / <REASONING> / <RESPONSE>
            │
            ▼
       dataset/reconstructed/NNNNN/train_NNNNN_partP.json
```

The LLM selects the most appropriate workflow from 10 options and restructures
the existing reasoning content into that workflow's section layout.
All original content is preserved — no summarization or omission.

---

## Quick Start

```bash
# Single parquet, part 0 of 4
python sft_reconstruct.py \
  --model deepseek/deepseek-v3.2 \
  --parquet /path/to/train-00000-of-00070.parquet \
  --part 0

# Run all 4 parts in parallel across 4 terminals
python sft_reconstruct.py --model MODEL --parquet train-00000-of-00070.parquet --part 0
python sft_reconstruct.py --model MODEL --parquet train-00000-of-00070.parquet --part 1
python sft_reconstruct.py --model MODEL --parquet train-00000-of-00070.parquet --part 2
python sft_reconstruct.py --model MODEL --parquet train-00000-of-00070.parquet --part 3
```

API key is read from `.env` at the project root — no `--api-key` needed:
```
OPENROUTER_API_KEY=sk-or-v1-...
```

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | OpenRouter model ID (e.g. `deepseek/deepseek-v3.2`) |
| `--parquet` | *(required)* | Path to parquet file |
| `--part` | `None` | Part index 0-based. Divides parquet into `--num-parts` equal chunks |
| `--num-parts` | `4` | Total number of parts when using `--part` |
| `--start-index` | `0` | Start row index (inclusive). Ignored when `--part` is set |
| `--end-index` | `None` | End row index (exclusive). Ignored when `--part` is set |
| `--workers` | `10` | Concurrent API calls |
| `--output-dir` | `dataset/reconstructed` | Root output directory |
| `--save-every` | `20` | Save results every N completions |
| `--max-tokens` | `131072` | Max tokens per API call |
| `--retry-index` | `None` | Re-run a single row index |
| `--retry-errors` | flag | Retry all indices in the error log for this part/range |
| `--price-input` | `0.26` | Input token price per 1M tokens in USD |
| `--price-output` | `0.38` | Output token price per 1M tokens in USD |
| `--reconstruct-prompt` | `prompts/reconstruct_prompt.md` | Path to reconstruction prompt |
| `--workflows` | `prompts/reasoning_workflows.md` | Path to workflow definitions (appended to prompt) |
| `--api-key` | `$OPENROUTER_API_KEY` | OpenRouter API key (optional if set in `.env`) |

---

## Output Structure

```
dataset/
└── reconstructed/
    └── 00000/                          ← parquet number
        ├── train_00000_part0.json      ← results (part 0)
        ├── train_00000_part1.json      ← results (part 1)
        ├── train_00000_part2.json      ← results (part 2)
        ├── train_00000_part3.json      ← results (part 3)
        └── logs/
            ├── train_00000_part0_errors.json
            ├── train_00000_part0_summary.json
            ├── train_00000_part1_errors.json
            ├── train_00000_part1_summary.json
            ...
```

### Result JSON schema (one item per row)

```jsonc
{
  "_index": 42,
  "_subset": "BMMR",
  "question": "...",
  "question_clean": "...",        // <image> tag removed
  "options": ["A", "B", "C", "D"],
  "answer": "B",
  "reasoning_original": "...",    // original <think> block
  "response_original": "...",     // original text after </think>
  "reasoning": "...",             // restructured reasoning
  "response": "...",              // restructured response
  "rc_workflow": "Structured Decomposition (SD)"
}
```

### Error log (`_errors.json`)

```jsonc
{
  "42": { "reason": "api_or_parse_failure", "timestamp": "2025-01-01T12:00:00" },
  "87": { "reason": "api_or_parse_failure", "timestamp": "2025-01-01T12:05:00" }
}
```

Errors are automatically cleared when a retry succeeds.

### Summary (`_summary.json`)

Append-only — each run adds one record to `runs[]`. Never overwrites.

```jsonc
{
  "runs": [
    {
      "start": "2025-01-01T10:00:00",
      "end":   "2025-01-01T11:30:00",
      "range": [0, 500],
      "processed": 498,
      "errors": 2,
      "tokens": { "prompt": 1200000, "completion": 800000, "total": 2000000 },
      "estimated_cost_usd": 0.616
    }
  ]
}
```

---

## Resume & Error Handling

### Automatic resume

Re-running with the same `--parquet` and `--part` args picks up where it left off:

```bash
# Interrupted at idx=350 — just re-run the same command
python sft_reconstruct.py --model MODEL --parquet train-00000-of-00070.parquet --part 0
# [RESUME] Already done: 350 sample(s) — skipping
```

### Retry failed samples

```bash
# Retry all indices recorded in the error log
python sft_reconstruct.py --model MODEL --parquet train-00000-of-00070.parquet --part 0 --retry-errors

# Retry a single specific index
python sft_reconstruct.py --model MODEL --parquet train-00000-of-00070.parquet --part 0 --retry-index 42
```

### Error / failure table

| Situation | Behaviour |
|---|---|
| Output JSON corrupted on load | Warns and starts fresh for that part |
| Error log corrupted on load | Warns and starts with empty error dict |
| API error (non-rate-limit) | Logs to `_errors.json`, continues with other samples |
| Rate limit (HTTP 429) | Retries up to 5× with 10/30/60/120/180 s backoff |
| XML delimiters missing in response | Logs to `_errors.json`, sample marked failed |
| Empty `qwen3vl_235b_thinking_response` | `[SKIP]` printed, sample skipped silently |
| Successful retry | Entry cleared from `_errors.json` automatically |

---

## Cost Tracking

Tokens and estimated cost are printed at the end of every run:

```
  Tokens — prompt: 1,234,567  completion: 890,123  total: 2,124,690
  Estimated cost: $0.6573  (input $0.26/1M, output $0.38/1M)
```

Also saved to `_summary.json` under `tokens` and `estimated_cost_usd`.

Override prices with `--price-input` / `--price-output` if using a different model.

---

## Prompts

| File | Role |
|---|---|
| `prompts/reconstruct_prompt.md` | System prompt: task description, core principles (VLM persona, content preservation, visual completeness), output format |
| `prompts/reasoning_workflows.md` | 10 workflow definitions appended to the prompt at runtime |

The two files are concatenated at startup — edit either without touching the script.
