"""
post_filter.py  —  2-step post-filtering pipeline

Step 1 (Detection):
  Send full reasoning + response to LLM.
  Model returns {c1, c2, c3, c4} booleans (tiny output → reliable).
  If no issues found → sample is saved as-is (0 rewrites for clean samples).

Step 2 (Rewrite, only when issues detected):
  Send the same reasoning + response with explicit issue list.
  Model rewrites ONLY what needs fixing, returns full corrected text.

Quality issues:
  C1: Reference contamination in reasoning
  C2: Phase structure violation in reasoning
  C3: Phase labels leaked into response
  C4: Reference contamination in response

Usage:
  python post_filter.py \\
    --api-key YOUR_KEY \\
    --model deepseek/deepseek-v3.2 \\
    --input-dir data/reasoning_think/reasoning_think_20260406_215604

Output: post_filtering/post_filtering_YYYYMMDD_HHMMSS/
"""

import os
import re
import json
import time
import argparse
from datetime import datetime
from openai import OpenAI

# ── Paths ────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

DETECT_PROMPT_DEFAULT  = os.path.join(BASE_DIR, "prompts", "post_filter_detect_prompt.md")
REWRITE_PROMPT_DEFAULT = os.path.join(BASE_DIR, "prompts", "post_filter_rewrite_prompt.md")
META_DIR               = os.path.join(BASE_DIR, "data", "metadata")
OUT_BASE               = os.path.join(BASE_DIR, "post_filtering")

# ── Prompt loader ────────────────────────────────────────────

def load_prompt(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ── JSON parsing ─────────────────────────────────────────────

def fix_invalid_escapes(text: str) -> str:
    # Fix backslash sequences invalid in JSON but common in math/LaTeX.
    # e.g. \alpha -> \\alpha, \sigma -> \\sigma, \frac -> \\frac
    # Valid JSON escapes after backslash: " \ / b f n r t u(+4hex)
    # Everything else must be doubled.
    return re.sub(r'\\([^"\\/bfnrtu])', lambda m: '\\\\' + m.group(1), text)


def parse_json_output(text: str) -> dict:
    """
    Robustly parse a JSON object from LLM output.
    1) Strip markdown fences
    2) Extract outermost { ... }
    3) Try json.loads() directly
    4) If that fails with invalid escape, fix LaTeX backslashes and retry
    """
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
        # Second attempt: fix invalid escape sequences (LaTeX, etc.) then retry
        fixed = fix_invalid_escapes(text)
        return json.loads(fixed)  # raises if still broken

# ── Message builders ─────────────────────────────────────────

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
        f"### Question:\n{question}"
        f"{options_str}"
        f"\n\n---\n"
        f"### Reasoning:\n{reasoning or '(empty)'}"
        f"\n\n### Response:\n{response or '(empty)'}"
        f"\n\n---\n"
        f"Inspect the reasoning and response above. "
        f"Output only the JSON detection result."
    )


def build_rewrite_message(
    question: str, options, reasoning: str, response: str, answer: str,
    issues: dict
) -> str:
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

    # Build explicit issue list for the model
    flagged = []
    if issues.get("c1"): flagged.append("C1 — Reference contamination detected in reasoning")
    if issues.get("c2"): flagged.append("C2 — Phase structure violation detected in reasoning")
    if issues.get("c3"): flagged.append("C3 — Phase labels detected in response")
    if issues.get("c4"): flagged.append("C4 — Reference contamination detected in response")
    issue_list = "\n".join(f"  • {f}" for f in flagged)

    return (
        f"The following issues were detected and MUST be fixed:\n{issue_list}"
        f"\n\n### Question:\n{question}"
        f"{options_str}"
        f"{answer_str}"
        f"\n\n---\n"
        f"### Reasoning (fix if needed):\n{reasoning or '(empty)'}"
        f"\n\n### Response (fix if needed):\n{response or '(empty)'}"
        f"\n\n---\n"
        f"Apply ALL the fixing rules for the flagged issues from your instructions. "
        f"Return the complete corrected JSON. Do NOT truncate any text."
    )

# ── API call with retry ───────────────────────────────────────

MAX_RETRIES  = 5
RETRY_DELAYS = [10, 30, 60, 120, 180]


def call_api(client, model: str, system: str, user: str, max_tokens: int,
             label: str) -> str | None:
    """Call OpenRouter API with retry on rate-limit. Returns content string or None on failure."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
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
                print(
                    f"\n  [RATE LIMIT] {label} — waiting {wait}s "
                    f"(attempt {attempt+1}/{MAX_RETRIES})...",
                    end=" ", flush=True
                )
                time.sleep(wait)
                print("retrying...", end=" ", flush=True)
            else:
                print(f"\n  [API ERROR] {label}: {e}")
                return None
    return None

# ── Rule-based detection (C3 only) ───────────────────────────

PHASE_LABELS = [f"Phase {i}:" for i in range(1, 7)]


def rule_detect_c3(response: str) -> bool:
    # C3: any phase label present in response = violation (string match, never misses)
    return any(p in response for p in PHASE_LABELS)


# ── Step 1: Detection ─────────────────────────────────────────

def detect_issues(
    client, model: str, detect_prompt: str,
    question: str, options, reasoning: str, response: str,
    label: str, max_tokens: int = 512
) -> dict | None:
    # C3: rule-based (phase labels in response — deterministic, never misses)
    c3_rule = rule_detect_c3(response)

    # C1, C2, C4: LLM-based
    # C1/C4: require semantic understanding of reference contamination
    # C2: phase labels may all be present but appended retroactively after free-form solving
    user_msg = build_detect_message(question, options, reasoning, response)
    raw = call_api(client, model, detect_prompt, user_msg, max_tokens, label + "/detect")
    if raw is None:
        return None
    try:
        llm = parse_json_output(raw)
        return {
            "c1": bool(llm.get("c1", False)),
            "c2": bool(llm.get("c2", False)),
            "c3": c3_rule or bool(llm.get("c3", False)),  # rule OR LLM
            "c4": bool(llm.get("c4", False)),
        }
    except (json.JSONDecodeError, Exception) as e:
        print(f"\n  [WARN] Detection parse failed for {label}: {e} | raw: {raw[:200]}")
        # LLM failed — C3 rule result still valid; C1/C2/C4 unknown → conservative
        return {
            "c1": True,
            "c2": True,
            "c3": c3_rule,
            "c4": True,
        }

# ── Step 2: Rewrite ───────────────────────────────────────────

def parse_rewrite_output(raw: str) -> dict | None:
    # Extract content between XML delimiters — no JSON escaping issues
    r_match = re.search(r"<REASONING>(.*?)</REASONING>", raw, re.DOTALL)
    p_match = re.search(r"<RESPONSE>(.*?)</RESPONSE>",  raw, re.DOTALL)
    if not r_match and not p_match:
        return None
    return {
        "reasoning": r_match.group(1).strip() if r_match else None,
        "response":  p_match.group(1).strip() if p_match else None,
    }


def rewrite_sample(
    client, model: str, rewrite_prompt: str,
    question: str, options, reasoning: str, response: str, answer: str,
    issues: dict, label: str, max_tokens: int
) -> dict | None:
    user_msg = build_rewrite_message(question, options, reasoning, response, answer, issues)
    raw = call_api(client, model, rewrite_prompt, user_msg, max_tokens, label + "/rewrite")
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

def process_sample(
    client, model: str,
    detect_prompt: str, rewrite_prompt: str,
    sample: dict, meta_by_index: dict,
    gen_config: dict,
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
        print(f"  [SKIP] No reasoning/response for {label}")
        return None

    # ── Step 1: Detect ────────────────────────────────────────
    issues = detect_issues(
        client, model, detect_prompt,
        question, options, reasoning, response,
        label, max_tokens=512
    )

    if issues is None:
        # Detection API call failed entirely — skip to allow resume later
        return None

    any_issue = any(issues.values())

    if not any_issue:
        # Clean sample — save as-is, no rewrite needed
        print(f"→ clean")
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
        }

    # Print detected issues
    flags = "".join(k.upper() for k, v in issues.items() if v)
    print(f"→ issues [{flags}]", end=" ", flush=True)

    # ── Step 2: Rewrite ───────────────────────────────────────
    rewritten = rewrite_sample(
        client, model, rewrite_prompt,
        question, options, reasoning, response, answer,
        issues, label, max_tokens=gen_config["max_tokens"]
    )

    if rewritten is None:
        # Rewrite failed — skip so resume can retry
        print("→ REWRITE FAILED (will retry on resume)")
        return None

    new_reasoning = rewritten["reasoning"]
    new_response  = rewritten["response"]

    reasoning_modified = (new_reasoning != reasoning)
    response_modified  = (new_response  != response)

    mod_flags = []
    if reasoning_modified: mod_flags.append("R")
    if response_modified:  mod_flags.append("P")
    tag = f"[{'+'.join(mod_flags)}]" if mod_flags else "[no change]"
    print(f"→ rewritten {tag}")

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
        "reasoning_modified": reasoning_modified,
        "response_modified":  response_modified,
        "reasoning":          new_reasoning,
        "response":           new_response,
    }

# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="2-step post-filter: detect issues, then rewrite only flagged samples"
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
    # Generation config for rewrite step
    parser.add_argument("--temperature",    type=float, default=0)
    parser.add_argument("--top-p",         type=float, default=0.9)
    parser.add_argument("--max-tokens",    type=int,   default=131072,
                        help="Max output tokens for rewrite step (default: 131072)")
    args = parser.parse_args()

    # Resolve paths
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

    gen_config = {"max_tokens": args.max_tokens}

    detect_prompt  = load_prompt(args.detect_prompt)
    rewrite_prompt = load_prompt(args.rewrite_prompt)
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=args.api_key)

    print(f"Source run    : {source_run}")
    print(f"Input  dir    : {input_dir}")
    print(f"Output dir    : {args.output_dir}")
    print(f"Model         : {args.model}")
    print(f"Max tokens    : {args.max_tokens}")

    # Discover files
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

        # Auto-resume
        done_indices   = set()
        subset_results = []
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                subset_results = json.load(f)
            done_indices = {r.get("_index") for r in subset_results}
            if done_indices:
                print(f"  [RESUME] Already done: {len(done_indices)} sample(s) — skipping")

        # --retry-index
        if args.retry_index is not None:
            target = next((s for s in samples if s.get("_index") == args.retry_index), None)
            if target is None:
                print(f"  [ERROR] _index {args.retry_index} not found in {subset_name}.")
                continue
            print(f"  [RETRY] _index={args.retry_index}", end=" ", flush=True)
            result = process_sample(
                client, args.model,
                detect_prompt, rewrite_prompt,
                target, meta_by_index, gen_config
            )
            if result:
                subset_results = [r for r in subset_results if r.get("_index") != args.retry_index]
                subset_results.append(result)
                subset_results.sort(key=lambda r: r.get("_index", 0))
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(subset_results, f, ensure_ascii=False, indent=2)
                print(f"  Saved → {output_path}")
            overall["subsets"][subset_name] = len(subset_results)
            continue

        # Normal loop
        pending = [s for s in samples if s.get("_index") not in done_indices]
        if not pending:
            print(f"  [SKIP] All samples already processed.")
            overall["subsets"][subset_name] = len(subset_results)
            continue

        subset_clean    = 0
        subset_rewritten = 0
        subset_failed   = 0

        for i, sample in enumerate(pending):
            idx_label = sample.get("_index", i)
            print(f"  [{i+1:>3}/{len(pending)}] {subset_name}_{idx_label}", end="  ", flush=True)

            result = process_sample(
                client, args.model,
                detect_prompt, rewrite_prompt,
                sample, meta_by_index, gen_config
            )

            if result is None:
                subset_failed += 1
                continue

            if not any(result["detected_issues"].values()):
                subset_clean += 1
            else:
                subset_rewritten += 1

            subset_results.append(result)
            subset_results.sort(key=lambda r: r.get("_index", 0))
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(subset_results, f, ensure_ascii=False, indent=2)

        overall["stats"]["total"]     += len(pending)
        overall["stats"]["clean"]     += subset_clean
        overall["stats"]["rewritten"] += subset_rewritten
        overall["stats"]["failed"]    += subset_failed
        overall["subsets"][subset_name] = len(subset_results)

        print(
            f"  → clean: {subset_clean} | rewritten: {subset_rewritten} | failed: {subset_failed}"
        )
        print(f"  Saved → {output_path}")

    # Final summary
    s = overall["stats"]
    print(f"\n{'='*60}")
    print(f"Done.  total={s['total']}  clean={s['clean']}  rewritten={s['rewritten']}  failed={s['failed']}")
    summary_path = os.path.join(args.output_dir, "post_filter_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
