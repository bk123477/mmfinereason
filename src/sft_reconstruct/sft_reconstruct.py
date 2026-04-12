"""
sft_reconstruct.py  —  Async reasoning reconstruction from MMFineReason parquet files

Reads a parquet file, takes the existing Qwen3-VL-235B thinking trace
(qwen3vl_235b_thinking_response), and restructures it into one of 10 reasoning
workflows — without generating new reasoning from scratch or requiring the image.

Input:   dataset/raw/.../train-NNNNN-of-00070.parquet
Output:  dataset/reconstructed/NNNNN/train_NNNNN_partP.json

Usage:
  python sft_reconstruct.py --model MODEL --parquet train-00000-of-00070.parquet --part 0
  python sft_reconstruct.py --model MODEL --parquet train-00000-of-00070.parquet --part 1
  python sft_reconstruct.py --model MODEL --parquet train-00000-of-00070.parquet --part 2
  python sft_reconstruct.py --model MODEL --parquet train-00000-of-00070.parquet --part 3
"""

import os
import re
import json
import base64
import asyncio
import argparse
from io import BytesIO
from datetime import datetime

import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

# ── Paths ─────────────────────────────────────────────────────

BASE_DIR               = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RECONSTRUCT_PROMPT_DEFAULT = os.path.join(BASE_DIR, "prompts", "reconstruct_prompt.md")
WORKFLOWS_DEFAULT          = os.path.join(BASE_DIR, "prompts", "reasoning_workflows.md")
OUTPUT_DIR_DEFAULT         = os.path.join(BASE_DIR, "dataset", "reconstructed")

MAX_RETRIES  = 5
RETRY_DELAYS = [10, 30, 60, 120, 180]

# ── Token counter ─────────────────────────────────────────────

class TokenCounter:
    def __init__(self):
        self.prompt     = 0
        self.completion = 0

    def add(self, usage):
        if usage:
            self.prompt     += getattr(usage, "prompt_tokens",     0)
            self.completion += getattr(usage, "completion_tokens", 0)

    @property
    def total(self):
        return self.prompt + self.completion

    def report(self, price_input: float = 0.0, price_output: float = 0.0):
        print(f"\n  Tokens — prompt: {self.prompt:,}  completion: {self.completion:,}  total: {self.total:,}")
        if price_input or price_output:
            cost = (self.prompt / 1_000_000) * price_input + \
                   (self.completion / 1_000_000) * price_output
            print(f"  Estimated cost: ${cost:.4f}  "
                  f"(input ${price_input}/1M, output ${price_output}/1M)")

# ── Helpers ───────────────────────────────────────────────────

def load_prompt(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_reconstruct_prompt(prompt_path: str, workflows_path: str) -> str:
    """Load the reconstruction prompt and append full workflow definitions."""
    prompt    = load_prompt(prompt_path)
    workflows = load_prompt(workflows_path)
    return prompt + "\n\n---\n\n## Full Workflow Reference\n\n" + workflows


def parquet_stem(parquet_path: str) -> str:
    """
    train-00000-of-00070.parquet  →  train_00000
    sft-00000-of-00193.parquet    →  sft_00000
    """
    name = os.path.basename(parquet_path)
    m = re.match(r"([A-Za-z]+)-(\d+)", name)
    return f"{m.group(1)}_{m.group(2)}" if m else name.replace(".parquet", "").replace("-", "_")


def extract_think_block(text: str) -> str:
    """Return content inside <think>...</think>."""
    if not text:
        return ""
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def extract_response_block(text: str) -> str:
    """Return content after </think> tag."""
    if not text:
        return ""
    m = re.search(r"</think>(.*)", text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


def parse_reconstruct_output(raw: str) -> dict | None:
    """Extract <WORKFLOW> and <REASONING>; parse everything after </REASONING> as response."""
    r_m  = re.search(r"<REASONING>(.*?)</REASONING>", raw, re.DOTALL)
    wf_m = re.search(r"<WORKFLOW>(.*?)</WORKFLOW>",    raw, re.DOTALL)
    if not r_m and not wf_m:
        return None
    response = None
    if r_m:
        response = raw[r_m.end():].strip()
    return {
        "reasoning": r_m.group(1).strip()  if r_m  else None,
        "response":  response,
        "workflow":  wf_m.group(1).strip() if wf_m else None,
    }


def response_has_phase_leakage(text: str) -> bool:
    """Detect templated phase-style leakage inside the response."""
    if not text:
        return False
    return any([
        re.search(r"(?im)^\s*#{2,}\s*phase\b", text) is not None,
        re.search(r"(?im)\bphase\s*\d+\b", text) is not None,
        re.search(r"(?m)^\[[^\n\]]+\]\s*$", text) is not None,
    ])


def reconstruct_output_is_valid(parsed: dict | None) -> bool:
    """Require both reasoning and response, with response free of phase leakage."""
    return len(get_reconstruct_output_issues(parsed)) == 0


def get_reconstruct_output_issues(parsed: dict | None) -> list[str]:
    """Return structured validation issues for parsed reconstruct output."""
    issues = []
    if not parsed:
        return ["parse_failure"]

    reasoning = (parsed.get("reasoning") or "").strip()
    response = (parsed.get("response") or "").strip()

    if not reasoning:
        issues.append("reasoning_empty")
    if not response:
        issues.append("response_empty")
    elif response_has_phase_leakage(response):
        issues.append("response_phase_leakage")

    return issues


def extract_image_info(img_val) -> tuple[dict | None, str | None]:
    """
    Extract image metadata and base64-encoded bytes from a parquet image value.
    Returns (image_info dict, image_b64 string).
    Supports HuggingFace dict format, PIL Image, and raw bytes.
    """
    if img_val is None:
        return None, None
    try:
        from PIL import Image as PILImage

        if isinstance(img_val, dict):
            # HuggingFace datasets format: {'bytes': b'...', 'path': '...'}
            raw_bytes = img_val.get("bytes")
            img_path  = img_val.get("path")
            if raw_bytes:
                img = PILImage.open(BytesIO(raw_bytes))
                return (
                    {
                        "width":  img.width,
                        "height": img.height,
                        "mode":   img.mode,
                        "format": img.format or "unknown",
                        "path":   img_path,
                    },
                    base64.b64encode(raw_bytes).decode("utf-8"),
                )

        elif hasattr(img_val, "save"):  # PIL Image object
            fmt = img_val.format or "PNG"
            buf = BytesIO()
            img_val.save(buf, format=fmt)
            raw_bytes = buf.getvalue()
            return (
                {
                    "width":  img_val.width,
                    "height": img_val.height,
                    "mode":   img_val.mode,
                    "format": fmt,
                    "path":   None,
                },
                base64.b64encode(raw_bytes).decode("utf-8"),
            )

        elif isinstance(img_val, (bytes, bytearray)):
            img = PILImage.open(BytesIO(img_val))
            return (
                {
                    "width":  img.width,
                    "height": img.height,
                    "mode":   img.mode,
                    "format": img.format or "unknown",
                    "path":   None,
                },
                base64.b64encode(bytes(img_val)).decode("utf-8"),
            )

    except Exception as e:
        return {"error": str(e)}, None

    return None, None


def row_to_sample(row: dict, row_idx: int, save_image: bool = False) -> dict:
    """Convert a parquet row to a sample dict.

    Args:
        save_image: When True, include image_info (metadata) and image_b64 (base64 bytes)
                    in the output. Default False — image fields are omitted entirely.
    """
    question = str(row.get("question", "") or "").replace("<image>", "").strip()
    source   = str(row.get("source", "unknown") or "unknown")

    sample = {"_index": row_idx, "_subset": source}
    for col, val in row.items():
        if col == "image":
            continue
        if isinstance(val, bool):
            sample[col] = bool(val)
        elif hasattr(val, "item"):       # numpy scalar
            sample[col] = val.item()
        elif val is None or val != val:  # nan check
            sample[col] = None
        else:
            sample[col] = val

    sample["question_clean"] = question

    if save_image:
        image_info, image_b64 = extract_image_info(row.get("image"))
        sample["image_info"] = image_info
        sample["image_b64"]  = image_b64

    return sample


def build_reconstruct_message(question: str, options, qwen_thinking: str) -> str:
    """Build the user message for the reconstruction API call."""
    options_str = ""
    if options:
        if isinstance(options, list):
            labels = list("ABCDEFGHIJ")
            options_str = "\n\n### Options:\n" + "\n".join(
                f"({labels[i]}) {opt}" for i, opt in enumerate(options) if i < len(labels)
            )
        else:
            options_str = f"\n\n### Options:\n{options}"

    return (
        f"### Question:\n{question}{options_str}\n\n"
        f"---\n\n"
        f"### Existing Reasoning:\n{qwen_thinking or '(empty)'}\n\n"
        f"---\n\n"
        f"Select the most appropriate workflow and restructure the reasoning and response. "
        f"Follow all rules in the system prompt exactly.\n\n"
        f"CONTENT REQUIREMENTS:\n"
        f"- This is reconstruction, not summarization.\n"
        f"- Preserve the full reasoning content in workflow-shaped form.\n"
        f"- Keep all substantial calculation steps, detours, option checks, self-corrections, and verification steps.\n"
        f"- If the original reasoning explored multiple attempts, the reconstructed reasoning must still include those attempts rather than collapsing them into one short solution.\n\n"
        f"FORMAT REQUIREMENTS:\n"
        f"- Output exactly one <WORKFLOW> block and one <REASONING> block.\n"
        f"- <REASONING> must be non-empty.\n"
        f"- After </REASONING>, continue with a non-empty final response in plain explanatory prose.\n"
        f"- Do not stop after </REASONING>.\n"
        f"- In the final response text after </REASONING>, do not use Phase, Step, bracketed subtitles, bullet lists, or workflow labels.\n"
    )


def build_response_retry_message(question: str, options, qwen_thinking: str) -> str:
    """Retry message used only when the response leaks phase/template structure."""
    base = build_reconstruct_message(question, options, qwen_thinking)
    return (
        base
        + "\n\nIMPORTANT CORRECTION FOR RETRY:\n"
        + "- The previous final response was malformed or missing.\n"
        + "- Rewrite both the reasoning and the final response so the output fully matches the required format.\n"
        + "- Do not shorten the reasoning into a summary.\n"
        + "- Preserve the full chain of reasoning in workflow-shaped form.\n"
        + "- After </REASONING>, write a non-empty plain-prose final response.\n"
        + "- Do not use 'Phase', 'Step', bracketed subtitles, bullet lists, or workflow labels in the final response.\n"
        + "- Make sure the final output still contains exactly one <WORKFLOW> block and one <REASONING> block.\n"
        + "- Keep the response explanatory and complete, but natural.\n"
    )


# ── Error log & summary ───────────────────────────────────────

def load_error_log(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[WARN] Error log corrupted ({e}), starting with empty log.")
        return {}


def save_error_log(errors: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)


def log_error(errors: dict, index: int, reason: str):
    errors[str(index)] = {"reason": reason, "timestamp": datetime.now().isoformat()}


def clear_error(errors: dict, index: int):
    errors.pop(str(index), None)


def append_summary(path: str, run_info: dict):
    """Append a run record and refresh cumulative totals."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            data = {"runs": []}
    else:
        data = {"runs": []}
    runs = data.get("runs", [])
    runs.append(run_info)
    data["runs"] = runs

    total_prompt = sum((r.get("tokens") or {}).get("prompt", 0) for r in runs)
    total_completion = sum((r.get("tokens") or {}).get("completion", 0) for r in runs)
    total_tokens = sum((r.get("tokens") or {}).get("total", 0) for r in runs)
    total_estimated_cost = sum(r.get("estimated_cost_usd", 0.0) or 0.0 for r in runs)
    total_processed = sum(r.get("processed", 0) for r in runs)
    total_errors = sum(r.get("errors", 0) for r in runs)

    data["aggregate"] = {
        "run_count": len(runs),
        "processed": total_processed,
        "errors": total_errors,
        "tokens": {
            "prompt": total_prompt,
            "completion": total_completion,
            "total": total_tokens,
        },
        "estimated_cost_usd": round(total_estimated_cost, 6),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Async API call ────────────────────────────────────────────

async def call_api_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    label: str,
    counter: TokenCounter,
) -> str | None:
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    max_tokens=max_tokens,
                    extra_body={"include_reasoning": False},
                )
                counter.add(resp.usage)
                return resp.choices[0].message.content or ""
            except Exception as e:
                err = str(e)
                is_rate = "429" in err or "rate" in err.lower()
                if is_rate and attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAYS[attempt]
                    print(f"\n  [RATE LIMIT] {label} — waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                    await asyncio.sleep(wait)
                else:
                    err_type = "api_rate_limit" if is_rate else "api_error"
                    print(f"\n  [API ERROR] {label} | type={err_type}: {e}")
                    return None
    return None


# ── Per-sample processing ─────────────────────────────────────

async def process_sample_async(
    row: dict,
    row_idx: int,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    reconstruct_prompt: str,
    max_tokens: int,
    counter: TokenCounter,
    save_image: bool = False,
) -> tuple[dict | None, str | None]:
    sample   = row_to_sample(row, row_idx, save_image=save_image)
    label    = f"{sample['_subset']}_{row_idx}"
    question = sample["question_clean"]
    options  = sample.get("options")

    raw_thinking = str(sample.get("qwen3vl_235b_thinking_response", "") or "")
    qwen_thinking = extract_think_block(raw_thinking)
    qwen_response = extract_response_block(raw_thinking)

    if not qwen_thinking:
        print(f"  [SKIP] {label}: empty thinking block")
        return None, "reasoning_source_empty"

    user_msg = build_reconstruct_message(question, options, qwen_thinking)
    raw = await call_api_async(
        client, semaphore, model, reconstruct_prompt, user_msg, max_tokens, label, counter
    )
    if raw is None:
        return None, "api_failure"

    parsed = parse_reconstruct_output(raw)
    issues = get_reconstruct_output_issues(parsed)
    if issues:
        retry_msg = build_response_retry_message(question, options, qwen_thinking)
        retry_raw = await call_api_async(
            client,
            semaphore,
            model,
            reconstruct_prompt,
            retry_msg,
            max_tokens,
            f"{label}:response-retry",
            counter,
        )
        if retry_raw is not None:
            retry_parsed = parse_reconstruct_output(retry_raw)
            retry_issues = get_reconstruct_output_issues(retry_parsed)
            if not retry_issues:
                parsed = retry_parsed
                issues = []

    if issues:
        raw_preview = (raw or "")[:200]
        print(
            f"\n  [WARN] Invalid reconstruct output for {label}"
            f" | issues={','.join(issues)} | raw: {raw_preview}"
        )
        return None, ",".join(issues)

    wf = parsed.get("workflow")
    if wf:
        print(f"\n  [WF] {label} → {wf}", flush=True)

    return ({
        **sample,
        "reasoning_original": qwen_thinking,
        "response_original":  qwen_response,
        "reasoning":          parsed["reasoning"],
        "response":           parsed["response"],
        "rc_workflow":        wf,
    }, None)


# ── Save ──────────────────────────────────────────────────────

async def save_results(results: dict, output_path: str, lock: asyncio.Lock):
    async with lock:
        sorted_list = sorted(results.values(), key=lambda r: r.get("_index", 0))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sorted_list, f, ensure_ascii=False, indent=2)


# ── Main ──────────────────────────────────────────────────────

async def async_main(args):
    if not args.api_key:
        print("[ERROR] No API key. Set OPENROUTER_API_KEY in .env or pass --api-key.")
        return

    df = pd.read_parquet(args.parquet)
    total_rows = len(df)
    print(f"Loaded parquet: {total_rows} rows")

    stem = parquet_stem(args.parquet)

    # Subdirectory per parquet: reconstructed/00000/
    m_num = re.search(r"_(\d+)$", stem)
    parquet_num   = m_num.group(1) if m_num else stem
    output_subdir = os.path.join(args.output_dir, parquet_num)
    log_dir       = os.path.join(output_subdir, "logs")
    os.makedirs(output_subdir, exist_ok=True)
    os.makedirs(log_dir,       exist_ok=True)

    # Determine index range — Priority: --part > --start-index/--end-index > full file
    if args.part is not None:
        n = args.num_parts
        p = args.part
        if p < 0 or p >= n:
            print(f"[ERROR] --part {p} out of range [0, {n})")
            return
        chunk = total_rows // n
        start = p * chunk
        end   = total_rows if p == n - 1 else (p + 1) * chunk
        output_path  = os.path.join(output_subdir, f"{stem}_part{p}.json")
        error_path   = os.path.join(log_dir,       f"{stem}_part{p}_errors.json")
        summary_path = os.path.join(log_dir,       f"{stem}_part{p}_summary.json")
    else:
        start = args.start_index
        end   = args.end_index if args.end_index is not None else total_rows
        end   = min(end, total_rows)
        has_range = (args.start_index != 0 or args.end_index is not None)
        suffix = f"_s{start}_e{end}" if has_range else ""
        output_path  = os.path.join(output_subdir, f"{stem}{suffix}.json")
        error_path   = os.path.join(log_dir,       f"{stem}{suffix}_errors.json")
        summary_path = os.path.join(log_dir,       f"{stem}{suffix}_summary.json")

    print(f"Processing rows [{start}, {end}) of {total_rows}")
    print(f"Output:    {output_path}")
    print(f"Error log: {error_path}")
    print(f"Workers:   {args.workers}")

    # Load existing results for resume
    results = {}
    done_indices = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for r in existing:
                results[r["_index"]] = r
            done_indices = set(results.keys())
            if done_indices:
                print(f"[RESUME] Already done: {len(done_indices)} sample(s) — skipping")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[WARN] Output file corrupted ({e}), starting fresh.")

    errors    = load_error_log(error_path)
    run_start = datetime.now()

    if errors:
        print(f"[RESUME] Error log: {len(errors)} previously failed index(es)")

    reconstruct_prompt = load_reconstruct_prompt(args.reconstruct_prompt, args.workflows)
    client    = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.workers)
    lock      = asyncio.Lock()
    counter   = TokenCounter()

    # --retry-index: single sample
    if args.retry_index is not None:
        if args.retry_index < 0 or args.retry_index >= total_rows:
            print(f"[ERROR] --retry-index {args.retry_index} out of range [0, {total_rows})")
            return
        print(f"[RETRY] Re-generating index={args.retry_index}")
        row = df.iloc[args.retry_index].to_dict()
        result, failure_reason = await process_sample_async(
            row, args.retry_index, client, semaphore, args.model,
            reconstruct_prompt, args.max_tokens, counter,
            save_image=args.save_image,
        )
        if result:
            results[args.retry_index] = result
            clear_error(errors, args.retry_index)
            status = "OK"
        else:
            log_error(errors, args.retry_index, failure_reason or "reconstruct_generation_failed")
            status = "ERR"
        await save_results(results, output_path, lock)
        save_error_log(errors, error_path)
        append_summary(summary_path, {
            "start":   run_start.isoformat(),
            "end":     datetime.now().isoformat(),
            "range":   [args.retry_index, args.retry_index + 1],
            "mode":    "retry-index",
            "processed": len(results),
            "errors":  len(errors),
            "tokens":  {"prompt": counter.prompt, "completion": counter.completion,
                        "total": counter.total},
            "estimated_cost_usd": round(
                (counter.prompt / 1_000_000) * args.price_input +
                (counter.completion / 1_000_000) * args.price_output, 6),
        })
        print(f"  → {status}  Saved → {output_path}")
        counter.report(args.price_input, args.price_output)
        return

    # --retry-errors: retry all indices in the error log
    if args.retry_errors:
        error_indices = sorted(int(k) for k in errors.keys())
        error_indices = [i for i in error_indices if start <= i < end and 0 <= i < total_rows]
        if not error_indices:
            print("[RETRY-ERRORS] No errors in log for this range.")
            return
        print(f"[RETRY-ERRORS] Retrying {len(error_indices)} failed index(es): "
              f"{error_indices[:10]}{'...' if len(error_indices) > 10 else ''}")
        for row_idx in error_indices:
            row = df.iloc[row_idx].to_dict()
            result, failure_reason = await process_sample_async(
                row, row_idx, client, semaphore, args.model,
                reconstruct_prompt, args.max_tokens, counter,
                save_image=args.save_image,
            )
            if result:
                results[row_idx] = result
                clear_error(errors, row_idx)
                print(f"  idx={row_idx} → OK ({result.get('rc_workflow', '?')})")
            else:
                log_error(errors, row_idx, failure_reason or "reconstruct_generation_failed")
                print(f"  idx={row_idx} → ERR ({failure_reason or 'reconstruct_generation_failed'})")
        await save_results(results, output_path, lock)
        save_error_log(errors, error_path)
        append_summary(summary_path, {
            "start":     run_start.isoformat(),
            "end":       datetime.now().isoformat(),
            "range":     [start, end],
            "mode":      "retry-errors",
            "processed": len(results),
            "errors":    len(errors),
            "tokens":    {"prompt": counter.prompt, "completion": counter.completion,
                          "total": counter.total},
            "estimated_cost_usd": round(
                (counter.prompt / 1_000_000) * args.price_input +
                (counter.completion / 1_000_000) * args.price_output, 6),
        })
        print(f"Done. Remaining errors: {len(errors)}")
        counter.report(args.price_input, args.price_output)
        return

    # Normal run
    pending_indices = [i for i in range(start, end) if i not in done_indices]
    if not pending_indices:
        print("[SKIP] All samples in range already processed.")
        return

    print(f"Pending: {len(pending_indices)} samples")
    completed = 0
    total_pending = len(pending_indices)

    async def process_one(row_idx: int):
        nonlocal completed
        row    = df.iloc[row_idx].to_dict()
        result, failure_reason = await process_sample_async(
            row, row_idx, client, semaphore, args.model,
            reconstruct_prompt, args.max_tokens, counter,
            save_image=args.save_image,
        )
        async with lock:
            completed += 1
            if result:
                results[row_idx] = result
                clear_error(errors, row_idx)
                wf = result.get("rc_workflow") or "?"
                print(
                    f"  [{completed:>5}/{total_pending}] idx={row_idx} "
                    f"({result.get('_subset','?')}) → OK  [{wf}]",
                    flush=True
                )
            else:
                log_error(errors, row_idx, failure_reason or "reconstruct_generation_failed")
                print(
                    f"  [{completed:>5}/{total_pending}] idx={row_idx} → FAILED ({failure_reason or 'reconstruct_generation_failed'})",
                    flush=True
                )

            if completed % args.save_every == 0:
                sorted_list = sorted(results.values(), key=lambda r: r.get("_index", 0))
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(sorted_list, f, ensure_ascii=False, indent=2)
                save_error_log(errors, error_path)

    await asyncio.gather(*[process_one(i) for i in pending_indices])

    # Final save
    await save_results(results, output_path, lock)
    save_error_log(errors, error_path)

    n_ok  = len(results)
    n_err = len(errors)

    cost_usd = (counter.prompt / 1_000_000) * args.price_input + \
               (counter.completion / 1_000_000) * args.price_output

    append_summary(summary_path, {
        "start":     run_start.isoformat(),
        "end":       datetime.now().isoformat(),
        "range":     [start, end],
        "processed": n_ok,
        "errors":    n_err,
        "tokens": {
            "prompt":     counter.prompt,
            "completion": counter.completion,
            "total":      counter.total,
        },
        "estimated_cost_usd": round(cost_usd, 6),
    })

    print(f"\nDone. {n_ok} OK, {n_err} failed → {output_path}")
    if errors:
        failed_list = sorted(int(k) for k in errors.keys())
        print(f"  Failed indices: {failed_list[:20]}{'...' if len(failed_list) > 20 else ''}")
        print(f"  Re-run with --retry-errors to retry them.")
    counter.report(args.price_input, args.price_output)


def main():
    parser = argparse.ArgumentParser(
        description="Async reasoning reconstruction from MMFineReason parquet"
    )
    parser.add_argument("--api-key",    default=os.environ.get("OPENROUTER_API_KEY"),
                        help="OpenRouter API key (default: $OPENROUTER_API_KEY)")
    parser.add_argument("--model",      required=True, help="OpenRouter model ID")
    parser.add_argument("--parquet",    required=True, help="Path to parquet file")
    parser.add_argument("--reconstruct-prompt", default=RECONSTRUCT_PROMPT_DEFAULT)
    parser.add_argument("--workflows",          default=WORKFLOWS_DEFAULT,
                        help="reasoning_workflows.md — appended to reconstruct prompt")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--workers",    type=int, default=10,
                        help="Concurrent API calls (default: 10)")
    parser.add_argument("--part",       type=int, default=None,
                        help="Part index (0-based). Divides parquet into --num-parts chunks.")
    parser.add_argument("--num-parts",  type=int, default=4,
                        help="Total number of parts (default: 4)")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Start row index (inclusive). Ignored when --part is set.")
    parser.add_argument("--end-index",   type=int, default=None,
                        help="End row index (exclusive). Ignored when --part is set.")
    parser.add_argument("--retry-index", type=int, default=None,
                        help="Re-generate only this single row index")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Retry all indices listed in the error log for this part/range")
    parser.add_argument("--save-every",  type=int, default=20,
                        help="Save results every N completions (default: 20)")
    parser.add_argument("--max-tokens",  type=int, default=131072,
                        help="Max tokens for reconstruction (default: 131072)")
    parser.add_argument("--save-image", action="store_true", default=False,
                        help="Include image_info and image_b64 (base64 bytes) in output JSON. "
                             "Default: off (image fields are omitted).")
    # Cost tracking
    parser.add_argument("--price-input",  type=float, default=0.26,
                        help="Input token price per 1M tokens in USD (default: 0.26)")
    parser.add_argument("--price-output", type=float, default=0.38,
                        help="Output token price per 1M tokens in USD (default: 0.38)")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
