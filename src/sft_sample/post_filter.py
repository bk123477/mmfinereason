"""
post_filter.py  —  Async iterative post-filtering pipeline

Step 1 (Detection):
  Send full reasoning + response to LLM.
  Model returns {c1, c2, c3, c4} booleans (tiny output → reliable).
  If no issues found → sample is saved as-is.

Step 2 (Rewrite, only when issues detected):
  Send the same reasoning + response with explicit issue list.
  Model rewrites ONLY what needs fixing, returns full corrected text.

Step 3 (Re-detect):
  Run detection again on the rewritten output.
  If issues still remain, repeat rewrite. Up to --max-iterations times.

Multiple samples are processed concurrently (--workers controls parallelism).

Quality issues:
  C1: Reference contamination in reasoning
  C2: Phase structure violation in reasoning
  C3: Phase labels leaked into response
  C4: Reference contamination in response

Usage:
  python post_filter.py \\
    --api-key YOUR_KEY \\
    --model deepseek/deepseek-v3.2 \\
    --input-dir data/reasoning_think/reasoning_think_20260406_215604 \\
    --workers 5

Output: post_filtering/post_filtering_YYYYMMDD_HHMMSS/
"""

import os
import re
import json
import asyncio
import argparse
from datetime import datetime
from openai import AsyncOpenAI

# ── Paths ────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

DETECT_PROMPT_DEFAULT  = os.path.join(BASE_DIR, "prompts", "post_filter_detect_prompt.md")
REWRITE_PROMPT_DEFAULT = os.path.join(BASE_DIR, "prompts", "post_filter_rewrite_prompt.md")
META_DIR               = os.path.join(BASE_DIR, "data", "metadata")
OUT_BASE               = os.path.join(BASE_DIR, "post_filtering")

MAX_RETRIES  = 5
RETRY_DELAYS = [10, 30, 60, 120, 180]
PHASE_LABELS = [f"Phase {i}:" for i in range(1, 7)]

# ── Prompt loader ────────────────────────────────────────────

def load_prompt(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ── JSON parsing ─────────────────────────────────────────────

def fix_invalid_escapes(text: str) -> str:
    return re.sub(r'\\([^"\\/bfnrtu])', lambda m: '\\\\' + m.group(1), text)


def parse_json_output(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(fix_invalid_escapes(text))

# ── Message builders ─────────────────────────────────────────

def build_options_str(options) -> str:
    if not options:
        return ""
    if isinstance(options, list):
        labels = list("ABCDEFGHIJ")
        return "\n\n### Options:\n" + "\n".join(
            f"({labels[i]}) {opt}" for i, opt in enumerate(options) if i < len(labels)
        )
    return f"\n\n### Options:\n{options}"


def build_detect_message(question: str, options, reasoning: str, response: str) -> str:
    return (
        f"### Question:\n{question}"
        f"{build_options_str(options)}"
        f"\n\n---\n"
        f"### Reasoning:\n{reasoning or '(empty)'}"
        f"\n\n### Response:\n{response or '(empty)'}"
        f"\n\n---\n"
        f"Inspect the reasoning and response above. "
        f"Output only the JSON detection result."
    )


def build_rewrite_message(
    question: str, options, reasoning: str, response: str, answer: str, issues: dict
) -> str:
    answer_str = f"\n\n### Ground Truth Answer:\n{answer}" if answer else ""
    flagged = []
    if issues.get("c1"): flagged.append("C1 — Reference contamination detected in reasoning")
    if issues.get("c2"): flagged.append("C2 — Phase structure violation detected in reasoning")
    if issues.get("c3"): flagged.append("C3 — Phase labels detected in response")
    if issues.get("c4"): flagged.append("C4 — Reference contamination detected in response")
    issue_list = "\n".join(f"  • {f}" for f in flagged)

    return (
        f"The following issues were detected and MUST be fixed:\n{issue_list}"
        f"\n\n### Question:\n{question}"
        f"{build_options_str(options)}"
        f"{answer_str}"
        f"\n\n---\n"
        f"### Reasoning (fix if needed):\n{reasoning or '(empty)'}"
        f"\n\n### Response (fix if needed):\n{response or '(empty)'}"
        f"\n\n---\n"
        f"Apply ALL the fixing rules for the flagged issues from your instructions. "
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

# ── Rule-based detection ──────────────────────────────────────

def rule_detect_c3(response: str) -> bool:
    return any(p in (response or "") for p in PHASE_LABELS)

# ── Detection ─────────────────────────────────────────────────

async def detect_issues_async(
    client, semaphore, model, detect_prompt,
    question, options, reasoning, response, label
) -> dict | None:
    c3_rule = rule_detect_c3(response)
    user_msg = build_detect_message(question, options, reasoning, response)
    raw = await call_api_async(
        client, semaphore, model, detect_prompt, user_msg, 512, label + "/detect"
    )
    if raw is None:
        return None
    try:
        llm = parse_json_output(raw)
        return {
            "c1": bool(llm.get("c1", False)),
            "c2": bool(llm.get("c2", False)),
            "c3": c3_rule or bool(llm.get("c3", False)),
            "c4": bool(llm.get("c4", False)),
        }
    except Exception as e:
        print(f"\n  [WARN] Detection parse failed for {label}: {e} | raw: {raw[:200]}")
        return {"c1": True, "c2": True, "c3": c3_rule, "c4": True}

# ── Rewrite ───────────────────────────────────────────────────

def parse_rewrite_output(raw: str) -> dict | None:
    r_match = re.search(r"<REASONING>(.*?)</REASONING>", raw, re.DOTALL)
    p_match = re.search(r"<RESPONSE>(.*?)</RESPONSE>",  raw, re.DOTALL)
    if not r_match and not p_match:
        return None
    return {
        "reasoning": r_match.group(1).strip() if r_match else None,
        "response":  p_match.group(1).strip() if p_match else None,
    }


async def rewrite_sample_async(
    client, semaphore, model, rewrite_prompt,
    question, options, reasoning, response, answer, issues, label, max_tokens
) -> dict | None:
    user_msg = build_rewrite_message(question, options, reasoning, response, answer, issues)
    raw = await call_api_async(
        client, semaphore, model, rewrite_prompt, user_msg, max_tokens, label + "/rewrite"
    )
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
    detect_prompt, rewrite_prompt,
    max_tokens, max_iterations,
    meta_by_index: dict,
) -> dict | None:
    idx        = sample.get("_index", 0)
    subset     = sample.get("_subset", "unknown")
    label      = f"{subset}_{idx}"
    question   = sample.get("question", "") or ""
    options    = sample.get("options")
    image_path = sample.get("image_path")
    reasoning  = sample.get("reasoning") or ""
    response   = sample.get("response") or ""
    source_run = sample.get("source_run", "")

    meta   = meta_by_index.get(idx, {})
    answer = meta.get("answer", "") or ""

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
            "_subset":            subset,
            "_index":             idx,
            "question":           question,
            "options":            options,
            "image_path":         image_path,
            "answer":             answer,
            "source_run":         source_run,
            "detected_issues":    issues,
            "reasoning_original": reasoning,
            "response_original":  response,
            "reasoning_modified": False,
            "response_modified":  False,
            "reasoning":          reasoning,
            "response":           response,
            "pf_iterations":      0,
        }

    # Iterative rewrite loop
    cur_reasoning   = reasoning
    cur_response    = response
    final_issues    = issues
    iterations_done = 0

    for iteration in range(max_iterations):
        rewritten = await rewrite_sample_async(
            client, semaphore, model, rewrite_prompt,
            question, options, cur_reasoning, cur_response,
            answer, final_issues, f"{label}/iter{iteration+1}", max_tokens
        )
        if rewritten is None:
            return None  # allow resume to retry

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
            break
        final_issues = re_issues
        if not any(final_issues.values()):
            break

    return {
        "_subset":            subset,
        "_index":             idx,
        "question":           question,
        "options":            options,
        "image_path":         image_path,
        "answer":             answer,
        "source_run":         source_run,
        "detected_issues":    issues,
        "final_issues":       final_issues,
        "reasoning_original": reasoning,
        "response_original":  response,
        "reasoning_modified": cur_reasoning != reasoning,
        "response_modified":  cur_response  != response,
        "reasoning":          cur_reasoning,
        "response":           cur_response,
        "pf_iterations":      iterations_done,
    }

# ── Main ─────────────────────────────────────────────────────

async def async_main(args):
    input_dir = args.input_dir
    if not os.path.isabs(input_dir):
        input_dir = os.path.join(BASE_DIR, input_dir)
    if not os.path.isdir(input_dir):
        print(f"[ERROR] Input directory not found: {input_dir}")
        return

    source_run = os.path.basename(input_dir)

    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(OUT_BASE, f"post_filtering_{ts}")
    elif not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(BASE_DIR, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    detect_prompt  = load_prompt(args.detect_prompt)
    rewrite_prompt = load_prompt(args.rewrite_prompt)
    client    = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.workers)

    print(f"Source run    : {source_run}")
    print(f"Input  dir    : {input_dir}")
    print(f"Output dir    : {args.output_dir}")
    print(f"Model         : {args.model}")
    print(f"Workers       : {args.workers}")
    print(f"Max tokens    : {args.max_tokens}")
    print(f"Max iterations: {args.max_iterations}")

    # Discover subset files
    if args.subsets:
        reason_files = [os.path.join(input_dir, f"{s}_reasoning.json") for s in args.subsets]
    else:
        reason_files = sorted(
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.endswith("_reasoning.json")
            and f not in ("reasoning_summary.json", "post_filter_summary.json")
        )

    if not reason_files:
        print("No reasoning files found.")
        return

    print(f"Found {len(reason_files)} subset file(s).\n")

    overall = {
        "source_run": source_run,
        "model": args.model,
        "subsets": {},
        "stats": {"total": 0, "clean": 0, "rewritten": 0, "failed": 0},
    }

    for reason_file in reason_files:
        if not os.path.exists(reason_file):
            print(f"[SKIP] Not found: {reason_file}")
            continue

        with open(reason_file, "r", encoding="utf-8") as f:
            samples = json.load(f)
        if not isinstance(samples, list):
            samples = [samples]
        if not samples:
            continue

        subset_name = samples[0].get(
            "_subset",
            os.path.basename(reason_file).replace("_reasoning.json", "")
        )
        for s in samples:
            s["source_run"] = source_run

        print(f"\n{'='*60}")
        print(f"Subset: {subset_name} ({len(samples)} samples)")
        print(f"{'='*60}")

        # Load metadata
        meta_by_index = {}
        meta_path = os.path.join(args.data_dir, f"{subset_name}_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                if isinstance(m, list):
                    meta_by_index = {item.get("_index", i): item for i, item in enumerate(m)}
            except Exception as e:
                print(f"  [WARN] Cannot load metadata: {e}")

        output_path = os.path.join(args.output_dir, f"{subset_name}_reasoning.json")

        # Resume: load already-done indices
        done_indices   = set()
        subset_results = {}
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for r in existing:
                subset_results[r["_index"]] = r
            done_indices = set(subset_results.keys())
            if done_indices:
                print(f"  [RESUME] Already done: {len(done_indices)} sample(s) — skipping")

        # --retry-index
        if args.retry_index is not None:
            target = next((s for s in samples if s.get("_index") == args.retry_index), None)
            if target is None:
                print(f"  [ERROR] _index {args.retry_index} not found in {subset_name}.")
                continue
            print(f"  [RETRY] _index={args.retry_index}", end=" ", flush=True)
            result = await process_sample_async(
                target, client, semaphore, args.model,
                detect_prompt, rewrite_prompt,
                args.max_tokens, args.max_iterations, meta_by_index,
            )
            if result:
                subset_results[args.retry_index] = result
                _flush_results(subset_results, output_path)
                print(f"  Saved → {output_path}")
            overall["subsets"][subset_name] = len(subset_results)
            continue

        pending = [s for s in samples if s.get("_index") not in done_indices]
        if not pending:
            print(f"  [SKIP] All samples already processed.")
            overall["subsets"][subset_name] = len(subset_results)
            continue

        # Async processing
        lock     = asyncio.Lock()
        stats    = {"clean": 0, "rewritten": 0, "failed": 0}
        completed = 0
        total_pending = len(pending)

        async def process_one(sample: dict):
            nonlocal completed
            idx    = sample.get("_index", 0)
            result = await process_sample_async(
                sample, client, semaphore, args.model,
                detect_prompt, rewrite_prompt,
                args.max_tokens, args.max_iterations, meta_by_index,
            )
            async with lock:
                completed += 1
                if result is None:
                    stats["failed"] += 1
                    print(f"  [{completed:>3}/{total_pending}] {subset_name}_{idx}  → FAILED")
                else:
                    subset_results[idx] = result
                    iters = result.get("pf_iterations", 0)
                    if iters == 0:
                        stats["clean"] += 1
                        print(f"  [{completed:>3}/{total_pending}] {subset_name}_{idx}  → clean")
                    else:
                        stats["rewritten"] += 1
                        mod = []
                        if result.get("reasoning_modified"): mod.append("R")
                        if result.get("response_modified"):  mod.append("P")
                        tag = f"[{'+'.join(mod)}]" if mod else "[no change]"
                        flags = "".join(
                            k.upper() for k, v in result.get("detected_issues", {}).items() if v
                        )
                        print(
                            f"  [{completed:>3}/{total_pending}] {subset_name}_{idx}"
                            f"  → [{flags}] rewritten {tag} ({iters} iter(s))"
                        )
                    # Save after every completed sample
                    _flush_results(subset_results, output_path)

        await asyncio.gather(*[process_one(s) for s in pending])

        overall["stats"]["total"]     += total_pending
        overall["stats"]["clean"]     += stats["clean"]
        overall["stats"]["rewritten"] += stats["rewritten"]
        overall["stats"]["failed"]    += stats["failed"]
        overall["subsets"][subset_name] = len(subset_results)

        print(
            f"  → clean: {stats['clean']} | rewritten: {stats['rewritten']}"
            f" | failed: {stats['failed']}"
        )
        print(f"  Saved → {output_path}")

    # Final summary
    s = overall["stats"]
    print(f"\n{'='*60}")
    print(
        f"Done.  total={s['total']}  clean={s['clean']}"
        f"  rewritten={s['rewritten']}  failed={s['failed']}"
    )
    summary_path = os.path.join(args.output_dir, "post_filter_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    print(f"Summary → {summary_path}")


def _flush_results(results: dict, path: str):
    sorted_list = sorted(results.values(), key=lambda r: r.get("_index", 0))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted_list, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Async iterative post-filter: detect issues, rewrite, re-detect until clean"
    )
    parser.add_argument("--api-key",        required=True)
    parser.add_argument("--model",          required=True,
                        help="OpenRouter model ID (e.g. deepseek/deepseek-v3.2)")
    parser.add_argument("--input-dir",      required=True,
                        help="Run directory with *_reasoning.json files")
    parser.add_argument("--detect-prompt",  default=DETECT_PROMPT_DEFAULT)
    parser.add_argument("--rewrite-prompt", default=REWRITE_PROMPT_DEFAULT)
    parser.add_argument("--data-dir",       default=META_DIR)
    parser.add_argument("--output-dir",     default=None,
                        help="Output dir (default: post_filtering/post_filtering_YYYYMMDD_HHMMSS/)")
    parser.add_argument("--subsets",        nargs="*", default=None)
    parser.add_argument("--resume",         action="store_true")
    parser.add_argument("--retry-index",    type=int, default=None,
                        help="Re-process only this _index (requires --subsets)")
    parser.add_argument("--workers",        type=int, default=5,
                        help="Max concurrent API calls (default: 5)")
    parser.add_argument("--max-tokens",     type=int, default=131072,
                        help="Max output tokens for rewrite step (default: 131072)")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Max detect→rewrite iterations per sample (default: 3)")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
