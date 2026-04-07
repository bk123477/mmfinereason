"""
sft_postfilter.py  —  Async iterative post-filtering for parquet-based reasoning output

Step 1 (Detection):  LLM checks C1/C2/C4 + rule-based checks C3.
                     If no issues → sample saved as-is.
Step 2 (Rewrite):    Only for flagged samples. Uses XML delimiters to avoid JSON escape issues.
Step 3 (Re-detect):  Run detection again on the rewritten output.
                     If issues remain, repeat rewrite. Up to --max-iterations times.

Input:   dataset/reasoning/sft_XXXXX.json
Output:  dataset/post_filtering/sft_XXXXX.json

Usage:
  python sft_postfilter.py --api-key KEY --model MODEL --input dataset/reasoning/sft_00000.json
  # Split by index range:
  python sft_postfilter.py ... --start-index 0    --end-index 4588 --workers 5
  python sft_postfilter.py ... --start-index 4588 --end-index 9176 --workers 5
"""

import os
import re
import json
import asyncio
import argparse

from openai import AsyncOpenAI

# ── Paths ─────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DETECT_PROMPT_DEFAULT  = os.path.join(BASE_DIR, "prompts", "post_filter_detect_prompt.md")
REWRITE_PROMPT_DEFAULT = os.path.join(BASE_DIR, "prompts", "post_filter_rewrite_prompt.md")
OUTPUT_DIR_DEFAULT     = os.path.join(BASE_DIR, "dataset", "post_filtering")

MAX_RETRIES  = 5
RETRY_DELAYS = [10, 30, 60, 120, 180]
PHASE_LABELS = [f"Phase {i}:" for i in range(1, 7)]

# ── Helpers ───────────────────────────────────────────────────

def load_prompt(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def fix_invalid_escapes(text: str) -> str:
    # Fix LaTeX-style backslash sequences invalid in JSON (e.g. \alpha → \\alpha)
    return re.sub(r'\\([^"\\/bfnrtu])', lambda m: '\\\\' + m.group(1), text)


def parse_json_safe(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1:
        text = text[s:e+1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(fix_invalid_escapes(text))


def parse_rewrite_output(raw: str) -> dict | None:
    r_m = re.search(r"<REASONING>(.*?)</REASONING>", raw, re.DOTALL)
    p_m = re.search(r"<RESPONSE>(.*?)</RESPONSE>",  raw, re.DOTALL)
    if not r_m and not p_m:
        return None
    return {
        "reasoning": r_m.group(1).strip() if r_m else None,
        "response":  p_m.group(1).strip() if p_m else None,
    }


def rule_detect_c3(response: str) -> bool:
    return any(p in (response or "") for p in PHASE_LABELS)

# ── Message builders ──────────────────────────────────────────

def build_detect_message(question: str, options, reasoning: str, response: str) -> str:
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
        f"### Question:\n{question}{options_str}\n\n---\n"
        f"### Reasoning:\n{reasoning or '(empty)'}\n\n"
        f"### Response:\n{response or '(empty)'}\n\n---\n"
        f"Inspect the reasoning and response above. Output only the JSON detection result."
    )


def build_rewrite_message(question: str, options, reasoning: str, response: str,
                           answer: str, issues: dict) -> str:
    options_str = ""
    if options:
        if isinstance(options, list):
            labels = list("ABCDEFGHIJ")
            options_str = "\n\n### Options:\n" + "\n".join(
                f"({labels[i]}) {opt}" for i, opt in enumerate(options) if i < len(labels)
            )
        else:
            options_str = f"\n\n### Options:\n{options}"

    answer_str = f"\n\n### Ground Truth Answer:\n{answer}" if answer else ""
    flagged = []
    if issues.get("c1"): flagged.append("C1 — Reference contamination in reasoning")
    if issues.get("c2"): flagged.append("C2 — Phase structure violation in reasoning")
    if issues.get("c3"): flagged.append("C3 — Phase labels in response")
    if issues.get("c4"): flagged.append("C4 — Reference contamination in response")
    issue_list = "\n".join(f"  • {f}" for f in flagged)

    return (
        f"The following issues were detected and MUST be fixed:\n{issue_list}\n\n"
        f"### Question:\n{question}{options_str}{answer_str}\n\n---\n"
        f"### Reasoning (fix if needed):\n{reasoning or '(empty)'}\n\n"
        f"### Response (fix if needed):\n{response or '(empty)'}\n\n---\n"
        f"Apply ALL the fixing rules for the flagged issues. "
        f"Return the complete corrected output using the XML delimiter format. Do NOT truncate."
    )

# ── Async API call ────────────────────────────────────────────

async def call_api_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    label: str,
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
                return resp.choices[0].message.content or ""
            except Exception as e:
                err = str(e)
                is_rate = "429" in err or "rate" in err.lower()
                if is_rate and attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAYS[attempt]
                    print(f"\n  [RATE LIMIT] {label} — waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                    await asyncio.sleep(wait)
                else:
                    print(f"\n  [API ERROR] {label}: {e}")
                    return None
    return None

# ── Detection ─────────────────────────────────────────────────

async def detect_issues_async(
    client, semaphore, model, detect_prompt,
    question, options, reasoning, response, label
) -> dict | None:
    c3_rule = rule_detect_c3(response)
    user_msg = build_detect_message(question, options, reasoning, response)
    raw = await call_api_async(client, semaphore, model, detect_prompt, user_msg, 512, label + "/detect")
    if raw is None:
        return None
    try:
        llm = parse_json_safe(raw)
        return {
            "c1": bool(llm.get("c1", False)),
            "c2": bool(llm.get("c2", False)),
            "c3": c3_rule or bool(llm.get("c3", False)),
            "c4": bool(llm.get("c4", False)),
        }
    except Exception as e:
        print(f"\n  [WARN] Detection parse failed {label}: {e} | raw: {raw[:150]}")
        return {"c1": True, "c2": True, "c3": c3_rule, "c4": True}

# ── Rewrite ───────────────────────────────────────────────────

async def rewrite_sample_async(
    client, semaphore, model, rewrite_prompt,
    question, options, reasoning, response, answer, issues, label, max_tokens
) -> dict | None:
    user_msg = build_rewrite_message(question, options, reasoning, response, answer, issues)
    raw = await call_api_async(client, semaphore, model, rewrite_prompt, user_msg, max_tokens, label + "/rewrite")
    if raw is None:
        return None
    result = parse_rewrite_output(raw)
    if result is None:
        print(f"\n  [WARN] Rewrite delimiter not found for {label} | raw: {raw[:200]}")
        return None
    return {
        "reasoning": result["reasoning"] if result["reasoning"] else reasoning,
        "response":  result["response"]  if result["response"]  else response,
    }

# ── Per-sample processing ─────────────────────────────────────

async def process_sample_async(
    sample: dict,
    client, semaphore, model,
    detect_prompt, rewrite_prompt, max_tokens,
    max_iterations: int = 3,
) -> dict | None:
    idx      = sample.get("_index", 0)
    subset   = sample.get("_subset", "unknown")
    label    = f"{subset}_{idx}"
    question = sample.get("question_clean") or sample.get("question", "") or ""
    question = str(question).replace("<image>", "").strip()
    options  = sample.get("options")
    answer   = str(sample.get("answer", "") or "")
    reasoning = sample.get("reasoning") or ""
    response  = sample.get("response")  or ""

    if not reasoning and not response:
        return None

    # Step 1: Initial detection
    issues = await detect_issues_async(
        client, semaphore, model, detect_prompt,
        question, options, reasoning, response, label
    )
    if issues is None:
        return None

    if not any(issues.values()):
        return {
            **sample,
            "detected_issues":    issues,
            "reasoning_original": reasoning,
            "response_original":  response,
            "reasoning_modified": False,
            "response_modified":  False,
            "pf_iterations":      0,
        }

    # Iterative rewrite loop
    cur_reasoning = reasoning
    cur_response  = response
    final_issues  = issues
    iterations_done = 0

    for iteration in range(max_iterations):
        rewritten = await rewrite_sample_async(
            client, semaphore, model, rewrite_prompt,
            question, options, cur_reasoning, cur_response,
            answer, final_issues, f"{label}/iter{iteration+1}", max_tokens
        )

        if rewritten is None:
            return None  # rewrite failed — allow resume to retry

        cur_reasoning   = rewritten["reasoning"] or cur_reasoning
        cur_response    = rewritten["response"]  or cur_response
        iterations_done = iteration + 1

        # Re-detect
        re_issues = await detect_issues_async(
            client, semaphore, model, detect_prompt,
            question, options, cur_reasoning, cur_response,
            f"{label}/recheck{iteration+1}"
        )

        if re_issues is None:
            break  # recheck failed — keep current result

        final_issues = re_issues

        if not any(final_issues.values()):
            break  # clean after this iteration

    flags = "".join(k.upper() for k, v in issues.items() if v)

    return {
        **sample,
        "detected_issues":    issues,           # original detection
        "final_issues":       final_issues,     # issues after last recheck
        "reasoning_original": reasoning,
        "response_original":  response,
        "reasoning_modified": cur_reasoning != reasoning,
        "response_modified":  cur_response  != response,
        "reasoning":          cur_reasoning,
        "response":           cur_response,
        "pf_iterations":      iterations_done,
        "_pf_flags":          flags,
    }

# ── Save ──────────────────────────────────────────────────────

async def save_results(results: dict, output_path: str, lock: asyncio.Lock):
    async with lock:
        sorted_list = sorted(results.values(), key=lambda r: r.get("_index", 0))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sorted_list, f, ensure_ascii=False, indent=2)

# ── Main ──────────────────────────────────────────────────────

async def async_main(args):
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"[ERROR] Input not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        samples = [samples]

    total = len(samples)
    stem  = os.path.basename(input_path).replace(".json", "")
    output_path = os.path.join(args.output_dir, f"{stem}.json")
    os.makedirs(args.output_dir, exist_ok=True)

    start = args.start_index
    end   = args.end_index if args.end_index is not None else total
    end   = min(end, total)

    print(f"Input:   {input_path}  ({total} samples)")
    print(f"Output:  {output_path}")
    print(f"Range:   [{start}, {end})")
    print(f"Workers: {args.workers}")
    print(f"Max iter: {args.max_iterations}")

    # Load existing for resume
    done_indices = set()
    results = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for r in existing:
            results[r["_index"]] = r
        done_indices = set(results.keys())
        if done_indices:
            print(f"[RESUME] Already done: {len(done_indices)} sample(s)")

    detect_prompt  = load_prompt(args.detect_prompt)
    rewrite_prompt = load_prompt(args.rewrite_prompt)
    client    = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.workers)
    lock      = asyncio.Lock()

    # --retry-index
    if args.retry_index is not None:
        target = next((s for s in samples if s.get("_index") == args.retry_index), None)
        if target is None:
            print(f"[ERROR] _index {args.retry_index} not found.")
            return
        print(f"[RETRY] _index={args.retry_index}")
        result = await process_sample_async(
            target, client, semaphore, args.model,
            detect_prompt, rewrite_prompt, args.max_tokens,
            max_iterations=args.max_iterations,
        )
        if result:
            results[args.retry_index] = result
            await save_results(results, output_path, lock)
            flags = result.get("_pf_flags", "")
            status = f"issues [{flags}]" if flags else "clean"
            print(f"  → {status}  Saved → {output_path}")
        else:
            print("  → FAILED")
        return

    # Build pending
    range_samples  = [s for s in samples if start <= s.get("_index", 0) < end]
    pending = [s for s in range_samples if s.get("_index") not in done_indices]

    if not pending:
        print("[SKIP] All samples in range already processed.")
        return

    print(f"Pending: {len(pending)} samples")
    completed = 0
    total_pending = len(pending)
    stats = {"clean": 0, "rewritten": 0, "failed": 0}

    async def process_one(sample: dict):
        nonlocal completed
        idx    = sample.get("_index", 0)
        result = await process_sample_async(
            sample, client, semaphore, args.model,
            detect_prompt, rewrite_prompt, args.max_tokens,
            max_iterations=args.max_iterations,
        )
        async with lock:
            completed += 1
            if result is None:
                stats["failed"] += 1
                print(f"  [{completed:>5}/{total_pending}] idx={idx} → FAILED (will retry on resume)")
            else:
                results[idx] = result
                flags = result.get("_pf_flags", "")
                if flags:
                    stats["rewritten"] += 1
                    mod = []
                    if result.get("reasoning_modified"): mod.append("R")
                    if result.get("response_modified"):  mod.append("P")
                    tag = f"[{'+'.join(mod)}]" if mod else "[no change]"
                    print(f"  [{completed:>5}/{total_pending}] idx={idx} → [{flags}] rewritten {tag}")
                else:
                    stats["clean"] += 1
                    print(f"  [{completed:>5}/{total_pending}] idx={idx} → clean")

                if completed % args.save_every == 0:
                    sorted_list = sorted(results.values(), key=lambda r: r.get("_index", 0))
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(sorted_list, f, ensure_ascii=False, indent=2)

    await asyncio.gather(*[process_one(s) for s in pending])

    await save_results(results, output_path, lock)
    print(f"\nDone. clean={stats['clean']} rewritten={stats['rewritten']} failed={stats['failed']}")
    print(f"Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Async 2-step post-filtering for parquet-based reasoning output"
    )
    parser.add_argument("--api-key",       required=True)
    parser.add_argument("--model",         required=True)
    parser.add_argument("--input",         required=True,
                        help="Path to reasoning JSON (e.g. dataset/reasoning/sft_00000.json)")
    parser.add_argument("--detect-prompt",  default=DETECT_PROMPT_DEFAULT)
    parser.add_argument("--rewrite-prompt", default=REWRITE_PROMPT_DEFAULT)
    parser.add_argument("--output-dir",     default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--workers",        type=int, default=5,
                        help="Concurrent API calls for detection (default: 5)")
    parser.add_argument("--start-index",    type=int, default=0)
    parser.add_argument("--end-index",      type=int, default=None)
    parser.add_argument("--retry-index",    type=int, default=None)
    parser.add_argument("--save-every",     type=int, default=20)
    parser.add_argument("--max-tokens",     type=int, default=131072,
                        help="Max tokens for rewrite step (default: 131072)")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Max detect→rewrite iterations per sample (default: 3)")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
