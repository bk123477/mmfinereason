"""
reconstruct.py  —  Async reasoning reconstruction from metadata JSON files

Takes the existing Qwen3-VL-235B thinking traces (qwen3vl_235b_thinking_response)
from data/metadata/*.json and restructures them into one of 10 reasoning workflows.
No new reasoning generation, no image access required.

Input:   data/metadata/*_metadata.json  (22 files × 10 samples = 220 total)
Output:  post_filtering/reconstruct_YYYYMMDD_HHMMSS/

Usage:
  python reconstruct.py --model deepseek/deepseek-v3.2
  python reconstruct.py --model deepseek/deepseek-v3.2 --workers 5
  python reconstruct.py --model deepseek/deepseek-v3.2 --retry-index 42
"""

import os
import re
import json
import asyncio
import argparse
from datetime import datetime

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

# ── Paths ─────────────────────────────────────────────────────

BASE_DIR                   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
RECONSTRUCT_PROMPT_DEFAULT = os.path.join(BASE_DIR, "prompts", "reconstruct_prompt.md")
WORKFLOWS_DEFAULT          = os.path.join(BASE_DIR, "prompts", "reasoning_workflows.md")
METADATA_DIR_DEFAULT       = os.path.join(BASE_DIR, "data", "metadata")
OUT_BASE                   = os.path.join(BASE_DIR, "post_filtering")

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
    """Load reconstruction prompt and append full workflow definitions."""
    prompt    = load_prompt(prompt_path)
    workflows = load_prompt(workflows_path)
    return prompt + "\n\n---\n\n## Full Workflow Reference\n\n" + workflows


def load_all_metadata(metadata_dir: str) -> list[dict]:
    """
    Load all *_metadata.json files from metadata_dir.
    Returns a flat list of sample dicts, sorted by (_subset, _index).
    Skips download_summary.json and any non-metadata files.
    """
    samples = []
    for fname in sorted(os.listdir(metadata_dir)):
        if not fname.endswith("_metadata.json"):
            continue
        fpath = os.path.join(metadata_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                samples.extend(data)
            else:
                print(f"[WARN] {fname}: expected list, got {type(data).__name__}, skipping")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[WARN] {fname}: failed to load ({e}), skipping")
    return samples


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
        f"- Do not add new meta-commentary such as 'the existing reasoning concluded', 'the previous answer suggests', or 'the source material says'.\n"
        f"- Do not append a new reflective wrap-up paragraph unless that type of reflection already existed in the original reasoning.\n\n"
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
        + "- Do not add new meta-commentary such as 'the existing reasoning concluded' or 'the source material says'.\n"
        + "- Do not append a new reflective wrap-up paragraph unless that reflection already existed in the original reasoning.\n"
        + "- After </REASONING>, write a non-empty plain-prose final response.\n"
        + "- Do not use 'Phase', 'Step', bracketed subtitles, bullet lists, or workflow labels in the final response.\n"
        + "- Make sure the final output still contains exactly one <WORKFLOW> block and one <REASONING> block.\n"
        + "- Keep the response explanatory and complete, but natural.\n"
    )


def sample_key(sample: dict) -> str:
    """Unique key for resume deduplication: _subset + _index."""
    return f"{sample.get('_subset', 'unknown')}_{sample.get('_index', 0)}"


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


def log_error(errors: dict, key: str, reason: str, model_response: str | None = None):
    entry = {"reason": reason, "timestamp": datetime.now().isoformat()}
    if model_response:
        entry["model_response"] = model_response
    errors[key] = entry


def clear_error(errors: dict, key: str):
    errors.pop(key, None)


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
    total_ok = sum(r.get("ok", 0) for r in runs)
    total_failed = sum(r.get("failed", 0) for r in runs)

    data["aggregate"] = {
        "run_count": len(runs),
        "ok": total_ok,
        "failed": total_failed,
        "tokens": {
            "prompt": total_prompt,
            "completion": total_completion,
            "total": total_tokens,
        },
        "estimated_cost_usd": round(total_estimated_cost, 6),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _flush_results(results: dict, path: str):
    """Write results sorted by _subset + _index."""
    sorted_list = sorted(
        results.values(),
        key=lambda r: (r.get("_subset", ""), r.get("_index", 0))
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted_list, f, ensure_ascii=False, indent=2)

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
                    print(f"\n  [RATE LIMIT] {label} — waiting {wait}s "
                          f"(attempt {attempt+1}/{MAX_RETRIES})...")
                    await asyncio.sleep(wait)
                else:
                    err_type = "api_rate_limit" if is_rate else "api_error"
                    print(f"\n  [API ERROR] {label} | type={err_type}: {e}")
                    return None
    return None

# ── Per-sample processing ─────────────────────────────────────

async def process_sample_async(
    sample: dict,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    reconstruct_prompt: str,
    max_tokens: int,
    counter: TokenCounter,
) -> tuple[dict | None, str | None, str | None]:
    label    = sample_key(sample)
    question = str(sample.get("question", "") or "").replace("<image>", "").strip()
    options  = sample.get("options")

    raw_thinking  = str(sample.get("qwen3vl_235b_thinking_response", "") or "")
    qwen_thinking = extract_think_block(raw_thinking)
    qwen_response = extract_response_block(raw_thinking)

    if not qwen_thinking:
        print(f"  [SKIP] {label}: empty thinking block")
        return None, "reasoning_source_empty", None

    user_msg = build_reconstruct_message(question, options, qwen_thinking)
    raw = await call_api_async(
        client, semaphore, model, reconstruct_prompt,
        user_msg, max_tokens, label, counter
    )
    if raw is None:
        return None, "api_failure", None

    parsed = parse_reconstruct_output(raw)
    issues = get_reconstruct_output_issues(parsed)
    failure_raw = raw
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
            failure_raw = retry_raw
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
        return None, ",".join(issues), failure_raw

    wf = parsed.get("workflow")
    if wf:
        print(f"\n  [WF] {label} → {wf}", flush=True)

    return ({
        **sample,
        "question_clean":     question,
        "reasoning_original": qwen_thinking,
        "response_original":  qwen_response,
        "reasoning":          parsed["reasoning"],
        "response":           parsed["response"],
        "rc_workflow":        wf,
    }, None, None)

# ── Main ──────────────────────────────────────────────────────

async def async_main(args):
    if not args.api_key:
        print("[ERROR] No API key. Set OPENROUTER_API_KEY in .env or pass --api-key.")
        return

    # Load all metadata samples
    all_samples = load_all_metadata(args.metadata_dir)
    if not all_samples:
        print(f"[ERROR] No samples found in {args.metadata_dir}")
        return
    total = len(all_samples)
    print(f"Loaded {total} samples from {args.metadata_dir}")

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(OUT_BASE, f"reconstruct_{ts}")
    os.makedirs(output_dir, exist_ok=True)
    output_path  = os.path.join(output_dir, "metadata_reconstruct.json")
    error_path   = os.path.join(output_dir, "reconstruct_errors.json")
    summary_path = os.path.join(output_dir, "reconstruct_summary.json")

    print(f"Output dir : {output_dir}")
    print(f"Output file: {output_path}")
    print(f"Error log  : {error_path}")
    print(f"Workers    : {args.workers}")
    print(f"Model      : {args.model}")

    # Resume — load already-completed results
    results      = {}   # key → result dict
    done_keys    = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for r in existing:
                k = sample_key(r)
                results[k] = r
            done_keys = set(results.keys())
            if done_keys:
                print(f"[RESUME] Already done: {len(done_keys)} sample(s) — skipping")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[WARN] Output file corrupted ({e}), starting fresh.")

    counter   = TokenCounter()
    run_start = datetime.now()
    errors    = load_error_log(error_path)

    if errors:
        print(f"[RESUME] Error log: {len(errors)} previously failed sample(s)")

    reconstruct_prompt = load_reconstruct_prompt(args.reconstruct_prompt, args.workflows)
    client    = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.workers)
    lock      = asyncio.Lock()

    # --retry-index: re-run a single sample by position in the full list
    if args.retry_index is not None:
        if args.retry_index < 0 or args.retry_index >= total:
            print(f"[ERROR] --retry-index {args.retry_index} out of range [0, {total})")
            return
        target = all_samples[args.retry_index]
        label  = sample_key(target)
        print(f"[RETRY] {label}")
        result, failure_reason, failure_raw = await process_sample_async(
            target, client, semaphore, args.model,
            reconstruct_prompt, args.max_tokens, counter
        )
        if result:
            results[label] = result
            clear_error(errors, label)
            _flush_results(results, output_path)
            save_error_log(errors, error_path)
            append_summary(summary_path, {
                "mode": "retry-index",
                "start": run_start.isoformat(),
                "end": datetime.now().isoformat(),
                "target": label,
                "ok": 1,
                "failed": 0,
                "remaining_errors": len(errors),
                "tokens": {
                    "prompt": counter.prompt,
                    "completion": counter.completion,
                    "total": counter.total,
                },
                "estimated_cost_usd": round(
                    (counter.prompt / 1_000_000) * args.price_input +
                    (counter.completion / 1_000_000) * args.price_output, 6),
            })
            print(f"  → OK  [{result.get('rc_workflow','?')}]  Saved → {output_path}")
        else:
            log_error(
                errors,
                label,
                failure_reason or "reconstruct_generation_failed",
                failure_raw,
            )
            save_error_log(errors, error_path)
            append_summary(summary_path, {
                "mode": "retry-index",
                "start": run_start.isoformat(),
                "end": datetime.now().isoformat(),
                "target": label,
                "ok": 0,
                "failed": 1,
                "remaining_errors": len(errors),
                "tokens": {
                    "prompt": counter.prompt,
                    "completion": counter.completion,
                    "total": counter.total,
                },
                "estimated_cost_usd": round(
                    (counter.prompt / 1_000_000) * args.price_input +
                    (counter.completion / 1_000_000) * args.price_output, 6),
            })
            print("  → FAILED")
        counter.report(args.price_input, args.price_output)
        return

    # --retry-errors: retry all failed samples in the error log
    if args.retry_errors:
        retry_samples = [s for s in all_samples if sample_key(s) in errors]
        if not retry_samples:
            print("[RETRY-ERRORS] No errors in log.")
            return
        print(f"[RETRY-ERRORS] Retrying {len(retry_samples)} failed sample(s): "
              f"{[sample_key(s) for s in retry_samples[:10]]}{'...' if len(retry_samples) > 10 else ''}")
        for sample in retry_samples:
            key = sample_key(sample)
            result, failure_reason, failure_raw = await process_sample_async(
                sample, client, semaphore, args.model,
                reconstruct_prompt, args.max_tokens, counter
            )
            if result:
                results[key] = result
                clear_error(errors, key)
                print(f"  {key} → OK  [{result.get('rc_workflow','?')}]")
            else:
                log_error(
                    errors,
                    key,
                    failure_reason or "reconstruct_generation_failed",
                    failure_raw,
                )
                print(f"  {key} → ERR ({failure_reason or 'reconstruct_generation_failed'})")
        _flush_results(results, output_path)
        save_error_log(errors, error_path)
        append_summary(summary_path, {
            "mode": "retry-errors",
            "start": run_start.isoformat(),
            "end": datetime.now().isoformat(),
            "retried": len(retry_samples),
            "ok": sum(1 for s in retry_samples if sample_key(s) not in errors),
            "failed": sum(1 for s in retry_samples if sample_key(s) in errors),
            "remaining_errors": len(errors),
            "tokens": {
                "prompt": counter.prompt,
                "completion": counter.completion,
                "total": counter.total,
            },
            "estimated_cost_usd": round(
                (counter.prompt / 1_000_000) * args.price_input +
                (counter.completion / 1_000_000) * args.price_output, 6),
        })
        counter.report(args.price_input, args.price_output)
        return

    # Normal run — skip already-done
    pending = [s for s in all_samples if sample_key(s) not in done_keys]
    if not pending:
        print("[SKIP] All samples already processed.")
        return

    print(f"Pending: {len(pending)} samples\n")
    completed     = 0
    total_pending = len(pending)
    stats         = {"ok": 0, "failed": 0}

    async def process_one(sample: dict):
        nonlocal completed
        result, failure_reason, failure_raw = await process_sample_async(
            sample, client, semaphore, args.model,
            reconstruct_prompt, args.max_tokens, counter
        )
        key    = sample_key(sample)
        subset = sample.get("_subset", "?")
        async with lock:
            completed += 1
            if result:
                results[key] = result
                clear_error(errors, key)
                stats["ok"] += 1
                wf = result.get("rc_workflow") or "?"
                print(
                    f"  [{completed:>4}/{total_pending}] {key}"
                    f"  → OK  [{wf}]",
                    flush=True
                )
                # Flush after every success
                _flush_results(results, output_path)
                save_error_log(errors, error_path)
            else:
                log_error(
                    errors,
                    key,
                    failure_reason or "reconstruct_generation_failed",
                    failure_raw,
                )
                stats["failed"] += 1
                print(
                    f"  [{completed:>4}/{total_pending}] {key}"
                    f"  → FAILED ({failure_reason or 'reconstruct_generation_failed'})",
                    flush=True
                )

    await asyncio.gather(*[process_one(s) for s in pending])

    # Final flush + summary
    _flush_results(results, output_path)
    save_error_log(errors, error_path)

    run_info = {
        "metadata_dir": args.metadata_dir,
        "model":        args.model,
        "start":        run_start.isoformat(),
        "end":          datetime.now().isoformat(),
        "total":        total,
        "ok":           stats["ok"],
        "failed":       stats["failed"],
        "remaining_errors": len(errors),
        "tokens": {
            "prompt":     counter.prompt,
            "completion": counter.completion,
            "total":      counter.total,
        },
    }
    if args.price_input or args.price_output:
        cost = (counter.prompt / 1_000_000) * args.price_input + \
               (counter.completion / 1_000_000) * args.price_output
        run_info["estimated_cost_usd"] = round(cost, 6)

    append_summary(summary_path, run_info)

    print(f"\n{'='*60}")
    print(f"Done.  ok={stats['ok']}  failed={stats['failed']}")
    print(f"Saved  → {output_path}")
    if errors:
        failed_list = sorted(errors.keys())
        print(f"Errors → {error_path}")
        print(f"Remaining failed samples: {failed_list[:20]}{'...' if len(failed_list) > 20 else ''}")
        print("Re-run with --retry-errors to retry them.")
    print(f"Summary→ {summary_path}")
    counter.report(args.price_input, args.price_output)


def main():
    parser = argparse.ArgumentParser(
        description="Async reasoning reconstruction from metadata JSON files"
    )
    parser.add_argument("--api-key",     default=os.environ.get("OPENROUTER_API_KEY"),
                        help="OpenRouter API key (default: $OPENROUTER_API_KEY)")
    parser.add_argument("--model",       required=True, help="OpenRouter model ID")
    parser.add_argument("--metadata-dir", default=METADATA_DIR_DEFAULT,
                        help="Directory containing *_metadata.json files "
                             f"(default: data/metadata)")
    parser.add_argument("--reconstruct-prompt", default=RECONSTRUCT_PROMPT_DEFAULT)
    parser.add_argument("--workflows",          default=WORKFLOWS_DEFAULT,
                        help="reasoning_workflows.md — appended to reconstruct prompt")
    parser.add_argument("--output-dir",  default=None,
                        help="Output directory (default: post_filtering/reconstruct_YYYYMMDD_HHMMSS/)")
    parser.add_argument("--workers",     type=int, default=10,
                        help="Concurrent API calls (default: 10)")
    parser.add_argument("--retry-index", type=int, default=None,
                        help="Re-run a single sample by position index in the full list")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Retry all samples listed in reconstruct_errors.json")
    parser.add_argument("--max-tokens",  type=int, default=131072,
                        help="Max tokens per API call (default: 131072)")
    # Cost tracking
    parser.add_argument("--price-input",  type=float, default=0.26,
                        help="Input token price per 1M tokens in USD (default: 0.26)")
    parser.add_argument("--price-output", type=float, default=0.38,
                        help="Output token price per 1M tokens in USD (default: 0.38)")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
