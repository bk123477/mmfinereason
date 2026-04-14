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
from datetime import datetime

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

# ── Paths ─────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DETECT_PROMPT_DEFAULT        = os.path.join(BASE_DIR, "prompts", "post_filter_detect_prompt.md")
REWRITE_PROMPT_DEFAULT       = os.path.join(BASE_DIR, "prompts", "post_filter_rewrite_prompt.md")
CLEAN_REWRITE_PROMPT_DEFAULT = os.path.join(BASE_DIR, "prompts", "post_filter_rewrite_clean_prompt.md")
WORKFLOWS_DEFAULT            = os.path.join(BASE_DIR, "prompts", "reasoning_workflows.md")
OUTPUT_DIR_DEFAULT           = os.path.join(BASE_DIR, "dataset", "post_filtering")

MAX_RETRIES  = 5
RETRY_DELAYS = [10, 30, 60, 120, 180]
PHASE_LABELS = [f"Phase {i}:" for i in range(1, 7)]

MODEL_PRICING_USD_PER_MTOK = {
    "qwen/qwen3.5-397b-a17b": {
        "input": 0.39,
        "output": 2.34,
        "source": "https://openrouter.ai/qwen/qwen3.5-397b-a17b",
    },
    "deepseek/deepseek-v3.2": {
        "input": 0.26,
        "output": 0.38,
        "source": "https://openrouter.ai/deepseek/deepseek-v3.2",
    },
}

# ── Rule-based detection constants ───────────────────────────

C1_HARD_PATTERNS = [
    # Direct reference mentions
    "Reference Reasoning",
    "Reference answer",
    "reference answer format",
    "Reference solution",
    "CRITICAL: ABOUT THE PROVIDED REFERENCE",
    "provided reasoning",
    "provided hint",
    "provided solution",
    "provided answer",
    "provided context",
    "provided reference",
    "the provided reasoning",
    # Self-aware hedging
    "which I must not",
    "I must not mention",
    "I must not copy",
    "I must not directly",
    "I should not copy",
    "I should not directly",
    "must not directly copy",
    "avoid mentioning the reference",
    "I am not supposed to",
    "I cannot directly use",
    "I need to avoid referencing",
    # Standard contamination phrases
    "as provided",
    "based on the reference",
    "according to the reference",
    "the user provided",
    "the user pasted",
    "as the user mentioned",
    "you provided",
    "previous attempt",
    "prior solution",
    "earlier reasoning",
    "the hint shows",
    "using the hint",
    "I was given",
    "based on the given answer",
]

C1_SAFE_PHRASES = [
    "provided text",
    "provided passage",
    "provided image",
    "provided diagram",
    "provided chart",
    "provided graph",
    "provided table",
    "provided figure",
    "provided options",
    "provided answer choices",
    "provided question",
    "provided information",
    "provided evidence",
    "options provided",
    "reference frame",
    "reference point",
    "reference angle",
    "year referenced",
]

C2_RETROACTIVE_THRESHOLD = 0.25

# ── Helpers ───────────────────────────────────────────────────

def load_prompt(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_rewrite_prompt(rewrite_path: str, workflows_path: str) -> str:
    """Load the rewrite prompt and append the full workflow definitions."""
    rewrite = load_prompt(rewrite_path)
    workflows = load_prompt(workflows_path)
    return rewrite + "\n\n---\n\n## Full Workflow Reference\n\n" + workflows


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
    r_m  = re.search(r"<REASONING>(.*?)</REASONING>", raw, re.DOTALL)
    p_m  = re.search(r"<RESPONSE>(.*?)</RESPONSE>",   raw, re.DOTALL)
    wf_m = re.search(r"<WORKFLOW>(.*?)</WORKFLOW>",    raw, re.DOTALL)
    if not r_m and not p_m:
        return None
    return {
        "reasoning": r_m.group(1).strip()  if r_m  else None,
        "response":  p_m.group(1).strip()  if p_m  else None,
        "workflow":  wf_m.group(1).strip() if wf_m else None,
    }


def rule_detect_c1(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    for safe in C1_SAFE_PHRASES:
        if safe in text_lower:
            text_lower = text_lower.replace(safe, "")
    for pat in C1_HARD_PATTERNS:
        if pat.lower() in text_lower:
            return True
    return False


def rule_detect_c2(reasoning: str) -> bool:
    """Return True if reasoning has no valid structured workflow.

    Accepts two formats:
      Format A (legacy 6-phase): all Phase 1~6 labels present and Phase 1 within first 25%
      Format B (bracket workflow): at least 4 [Section Name] headers, first one within first 25%
    """
    if not reasoning:
        return True

    # Format A: legacy 6-phase
    if all(p in reasoning for p in PHASE_LABELS):
        first_pos = reasoning.find("Phase 1:")
        if first_pos != -1 and first_pos / len(reasoning) <= C2_RETROACTIVE_THRESHOLD:
            return False

    # Format B: bracket workflow sections [Like This]
    bracket_sections = re.findall(r'(?:^|\n)\[([A-Z][^\]\n]{2,60})\]', reasoning)
    content_sections = [s for s in bracket_sections if not s.lower().startswith("workflow:")]
    if len(content_sections) >= 4:
        first_bracket_pos = reasoning.find("[" + content_sections[0] + "]")
        if first_bracket_pos != -1 and first_bracket_pos / len(reasoning) <= C2_RETROACTIVE_THRESHOLD:
            return False

    return True


def rule_detect_c3(response: str) -> bool:
    return any(p in (response or "") for p in PHASE_LABELS)


def empty_usage_stats() -> dict:
    return {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "by_stage": {},
    }


def extract_usage(resp) -> dict:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _read(obj, key: str) -> int:
        if obj is None:
            return 0
        if isinstance(obj, dict):
            return int(obj.get(key, 0) or 0)
        return int(getattr(obj, key, 0) or 0)

    prompt_tokens = _read(usage, "prompt_tokens")
    completion_tokens = _read(usage, "completion_tokens")
    total_tokens = _read(usage, "total_tokens") or (prompt_tokens + completion_tokens)
    return {
        "requests": 1,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def merge_usage(stats: dict, delta: dict | None, stage: str | None = None) -> dict:
    if not delta:
        return stats
    for key in ("requests", "prompt_tokens", "completion_tokens", "total_tokens"):
        stats[key] = int(stats.get(key, 0) or 0) + int(delta.get(key, 0) or 0)
    if stage:
        by_stage = stats.setdefault("by_stage", {})
        stage_stats = by_stage.setdefault(stage, {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        })
        for key in ("requests", "prompt_tokens", "completion_tokens", "total_tokens"):
            stage_stats[key] = int(stage_stats.get(key, 0) or 0) + int(delta.get(key, 0) or 0)
    return stats


def estimate_cost_usd(model: str, usage_stats: dict) -> dict:
    pricing = MODEL_PRICING_USD_PER_MTOK.get(model)
    if not pricing:
        return {
            "model": model,
            "pricing_found": False,
            "input_cost_usd": None,
            "output_cost_usd": None,
            "total_cost_usd": None,
            "pricing_source": None,
        }

    input_cost = usage_stats.get("prompt_tokens", 0) / 1_000_000 * pricing["input"]
    output_cost = usage_stats.get("completion_tokens", 0) / 1_000_000 * pricing["output"]
    return {
        "model": model,
        "pricing_found": True,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
        "pricing_source": pricing["source"],
        "pricing_usd_per_mtok": {
            "input": pricing["input"],
            "output": pricing["output"],
        },
    }

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
        f"Preserve the complete solution process. Keep all substantive reasoning steps, calculations, branches, retries, and verification. "
        f"Do not summarize or compress the reasoning just to fit the workflow structure. "
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
) -> tuple[str | None, dict]:
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
                return resp.choices[0].message.content or "", extract_usage(resp)
            except Exception as e:
                err = str(e)
                is_rate = "429" in err or "rate" in err.lower()
                if is_rate and attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAYS[attempt]
                    print(f"\n  [RATE LIMIT] {label} — waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                    await asyncio.sleep(wait)
                else:
                    print(f"\n  [API ERROR] {label}: {e}")
                    return None, empty_usage_stats()
    return None, empty_usage_stats()

# ── Detection ─────────────────────────────────────────────────

async def detect_issues_async(
    client, semaphore, model, detect_prompt,
    question, options, reasoning, response, label
) -> tuple[dict | None, dict]:
    # Rule-based detection (always runs, cannot be missed by LLM)
    c1_rule = rule_detect_c1(reasoning)
    c2_rule = rule_detect_c2(reasoning)
    c3_rule = rule_detect_c3(response)
    c4_rule = rule_detect_c1(response)

    user_msg = build_detect_message(question, options, reasoning, response)
    raw, usage = await call_api_async(client, semaphore, model, detect_prompt, user_msg, 512, label + "/detect")
    if raw is None:
        return None, usage
    try:
        llm = parse_json_safe(raw)
        result = {
            "c1": c1_rule or bool(llm.get("c1", False)),
            "c2": c2_rule or bool(llm.get("c2", False)),
            "c3": c3_rule or bool(llm.get("c3", False)),
            "c4": c4_rule or bool(llm.get("c4", False)),
        }
        for key, rule_val, llm_val in [
            ("c1", c1_rule, bool(llm.get("c1", False))),
            ("c2", c2_rule, bool(llm.get("c2", False))),
            ("c3", c3_rule, bool(llm.get("c3", False))),
            ("c4", c4_rule, bool(llm.get("c4", False))),
        ]:
            if rule_val and not llm_val:
                print(f"\n  [RULE] {label}: rule caught {key.upper()} that LLM missed")
        return result, usage
    except Exception as e:
        print(f"\n  [WARN] Detection parse failed {label}: {e} | raw: {raw[:150]}")
        return {
            "c1": c1_rule or True,
            "c2": c2_rule or True,
            "c3": c3_rule,
            "c4": c4_rule or True,
        }, usage

# ── Rewrite ───────────────────────────────────────────────────

async def rewrite_sample_async(
    client, semaphore, model, rewrite_prompt,
    question, options, reasoning, response, answer, issues, label, max_tokens
) -> tuple[dict | None, dict]:
    user_msg = build_rewrite_message(question, options, reasoning, response, answer, issues)
    raw, usage = await call_api_async(client, semaphore, model, rewrite_prompt, user_msg, max_tokens, label + "/rewrite")
    if raw is None:
        return None, usage
    result = parse_rewrite_output(raw)
    if result is None:
        print(f"\n  [WARN] Rewrite delimiter not found for {label} | raw: {raw[:200]}")
        return None, usage
    wf = result.get("workflow")
    if wf:
        print(f"\n  [WF] {label} → {wf}")
    return {
        "reasoning": result["reasoning"] if result["reasoning"] else reasoning,
        "response":  result["response"]  if result["response"]  else response,
        "workflow":  wf,
    }, usage

# ── Per-sample processing ─────────────────────────────────────

async def process_sample_async(
    sample: dict,
    client, semaphore, model,
    detect_prompt, rewrite_prompt, clean_rewrite_prompt,
    max_tokens, max_iterations: int = 3,
) -> tuple[dict | None, dict]:
    usage_stats = empty_usage_stats()
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
        return None, usage_stats

    # ── Step 1: Detection ────────────────────────────────────
    issues, usage = await detect_issues_async(
        client, semaphore, model, detect_prompt,
        question, options, reasoning, response, label
    )
    merge_usage(usage_stats, usage, "detect")
    if issues is None:
        return None, usage_stats

    # ── Path A: Clean — light 6-phase rewrite ────────────────
    if not any(issues.values()):
        answer_str = f"\n\n### Ground Truth Answer:\n{answer}" if answer else ""
        options_str = ""
        if options:
            if isinstance(options, list):
                labels = list("ABCDEFGHIJ")
                options_str = "\n\n### Options:\n" + "\n".join(
                    f"({labels[i]}) {opt}" for i, opt in enumerate(options) if i < len(labels)
                )
            else:
                options_str = f"\n\n### Options:\n{options}"
        user_msg = (
            f"### Question:\n{question}{options_str}{answer_str}\n\n---\n"
            f"### Reasoning:\n{reasoning or '(empty)'}\n\n"
            f"### Response:\n{response or '(empty)'}\n\n---\n"
            f"Preserve the full reasoning process with all calculations, detours, retries, dead ends, and verification steps. "
            f"Remove any reference contamination, but do not summarize, shorten, or collapse the solution. "
            f"If the reasoning already has a valid workflow structure, keep that structure. "
            f"If structure repair is needed, apply the lightest possible restructuring while retaining every substantive step in full. "
            f"Return the complete output."
        )
        raw, usage = await call_api_async(
            client, semaphore, model,
            clean_rewrite_prompt, user_msg, max_tokens, label + "/clean"
        )
        merge_usage(usage_stats, usage, "clean")
        if raw is None:
            return None, usage_stats

        result = parse_rewrite_output(raw)
        if result is None:
            print(f"\n  [WARN] Clean rewrite delimiter not found for {label} | raw: {raw[:200]}")
            return None, usage_stats

        new_reasoning = result["reasoning"] or reasoning
        new_response  = result["response"]  or response

        return {
            **sample,
            "detected_issues":    issues,
            "final_issues":       issues,
            "reasoning_original": reasoning,
            "response_original":  response,
            "reasoning_modified": new_reasoning != reasoning,
            "response_modified":  new_response  != response,
            "reasoning":          new_reasoning,
            "response":           new_response,
            "pf_iterations":      1,
            "_pf_flags":          "",
            "pf_workflow":        "clean-preserve",
        }, usage_stats

    # ── Path B: Flagged — diverse workflow rewrite ───────────
    cur_reasoning   = reasoning
    cur_response    = response
    final_issues    = issues
    iterations_done = 0
    chosen_workflow = None

    for iteration in range(max_iterations):
        rewritten, usage = await rewrite_sample_async(
            client, semaphore, model, rewrite_prompt,
            question, options, cur_reasoning, cur_response,
            answer, final_issues, f"{label}/iter{iteration+1}", max_tokens
        )
        merge_usage(usage_stats, usage, "rewrite")

        if rewritten is None:
            return None, usage_stats

        cur_reasoning   = rewritten["reasoning"] or cur_reasoning
        cur_response    = rewritten["response"]  or cur_response
        if rewritten.get("workflow"):
            chosen_workflow = rewritten["workflow"]
        iterations_done = iteration + 1

        re_issues, usage = await detect_issues_async(
            client, semaphore, model, detect_prompt,
            question, options, cur_reasoning, cur_response,
            f"{label}/recheck{iteration+1}"
        )
        merge_usage(usage_stats, usage, "recheck")

        if re_issues is None:
            break

        final_issues = re_issues

        if not any(final_issues.values()):
            break

    flags = "".join(k.upper() for k, v in issues.items() if v)

    return {
        **sample,
        "detected_issues":    issues,
        "final_issues":       final_issues,
        "reasoning_original": reasoning,
        "response_original":  response,
        "reasoning_modified": cur_reasoning != reasoning,
        "response_modified":  cur_response  != response,
        "reasoning":          cur_reasoning,
        "response":           cur_response,
        "pf_iterations":      iterations_done,
        "_pf_flags":          flags,
        "pf_workflow":        chosen_workflow,
    }, usage_stats

# ── Save ──────────────────────────────────────────────────────

async def save_results(results: dict, output_path: str, lock: asyncio.Lock):
    async with lock:
        sorted_list = sorted(results.values(), key=lambda r: r.get("_index", 0))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sorted_list, f, ensure_ascii=False, indent=2)

# ── Error log & summary ───────────────────────────────────────

def load_error_log(path: str) -> dict:
    """Load error log as {str(index): {reason, timestamp}}."""
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
    """Append a run record to the summary file (never overwrites existing runs)."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"runs": []}
    data["runs"].append(run_info)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ── Main ──────────────────────────────────────────────────────

async def async_main(args):
    if not args.api_key:
        print("[ERROR] No API key provided. Set OPENROUTER_API_KEY in .env or pass --api-key.")
        return

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"[ERROR] Input not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        samples = [samples]

    total = len(samples)
    input_stem = os.path.basename(input_path).replace(".json", "")

    # Base stem without any part/range suffix (e.g. train_00000_part0 → train_00000)
    base_stem = re.sub(r"(_part\d+|_s\d+_e\d+)$", "", input_stem)

    # Subdirectory per parquet: post_filtering/00000/
    m_num = re.search(r"_(\d+)$", base_stem)
    parquet_num   = m_num.group(1) if m_num else base_stem
    output_subdir = os.path.join(args.output_dir, parquet_num)
    log_dir       = os.path.join(output_subdir, "logs")
    os.makedirs(output_subdir, exist_ok=True)
    os.makedirs(log_dir,       exist_ok=True)

    # Determine index range
    # Priority: --part  >  --start-index/--end-index  >  full file
    if args.part is not None:
        n = args.num_parts
        p = args.part
        if p < 0 or p >= n:
            print(f"[ERROR] --part {p} out of range [0, {n})")
            return
        chunk = total // n
        start = p * chunk
        end   = total if p == n - 1 else (p + 1) * chunk
        output_path  = os.path.join(output_subdir, f"{base_stem}_part{p}.json")
        error_path   = os.path.join(log_dir,       f"{base_stem}_part{p}_errors.json")
        summary_path = os.path.join(log_dir,       f"{base_stem}_part{p}_summary.json")
    else:
        start = args.start_index
        end   = args.end_index if args.end_index is not None else total
        end   = min(end, total)
        has_range = (args.start_index != 0 or args.end_index is not None)
        if has_range:
            suffix = f"_s{start}_e{end}"
        else:
            # No range specified: preserve original stem (may already have _partN)
            suffix = re.sub(r"^" + re.escape(base_stem), "", input_stem)
        output_path  = os.path.join(output_subdir, f"{base_stem}{suffix}.json")
        error_path   = os.path.join(log_dir,       f"{base_stem}{suffix}_errors.json")
        summary_path = os.path.join(log_dir,       f"{base_stem}{suffix}_summary.json")

    print(f"Input:     {input_path}  ({total} samples)")
    print(f"Output:    {output_path}")
    print(f"Error log: {error_path}")
    print(f"Range:     [{start}, {end})")
    print(f"Workers:   {args.workers}")
    print(f"Max iter:  {args.max_iterations}")

    # Load existing for resume
    done_indices = set()
    results = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for r in existing:
                results[r["_index"]] = r
            done_indices = set(results.keys())
            if done_indices:
                print(f"[RESUME] Already done: {len(done_indices)} sample(s)")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[WARN] Output file corrupted ({e}), starting fresh from this file.")

    errors = load_error_log(error_path)
    if errors:
        print(f"[RESUME] Error log: {len(errors)} previously failed index(es)")

    run_start = datetime.now()
    usage_stats = empty_usage_stats()

    detect_prompt        = load_prompt(args.detect_prompt)
    rewrite_prompt       = load_rewrite_prompt(args.rewrite_prompt, args.workflows)
    clean_rewrite_prompt = load_prompt(args.clean_rewrite_prompt)
    client    = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.workers)
    lock      = asyncio.Lock()

    # --retry-index: single sample
    if args.retry_index is not None:
        target = next((s for s in samples if s.get("_index") == args.retry_index), None)
        if target is None:
            print(f"[ERROR] _index {args.retry_index} not found.")
            return
        print(f"[RETRY] _index={args.retry_index}")
        result, usage = await process_sample_async(
            target, client, semaphore, args.model,
            detect_prompt, rewrite_prompt, clean_rewrite_prompt,
            args.max_tokens, max_iterations=args.max_iterations,
        )
        merge_usage(usage_stats, usage)
        if result:
            results[args.retry_index] = result
            clear_error(errors, args.retry_index)
            await save_results(results, output_path, lock)
            save_error_log(errors, error_path)
            append_summary(summary_path, {
                "start": run_start.isoformat(),
                "end": datetime.now().isoformat(),
                "model": args.model,
                "range": [args.retry_index, args.retry_index + 1],
                "processed": len(results),
                "clean": 1 if not result.get("_pf_flags") else 0,
                "rewritten": 1 if result.get("_pf_flags") else 0,
                "errors": len(errors),
                "failed": 0,
                "mode": "retry-index",
                "token_usage": usage_stats,
                "cost_estimate": estimate_cost_usd(args.model, usage_stats),
            })
            flags = result.get("_pf_flags", "")
            status = f"issues [{flags}]" if flags else "clean"
            print(f"  → {status}  Saved → {output_path}")
        else:
            log_error(errors, args.retry_index, "api_or_parse_failure")
            save_error_log(errors, error_path)
            append_summary(summary_path, {
                "start": run_start.isoformat(),
                "end": datetime.now().isoformat(),
                "model": args.model,
                "range": [args.retry_index, args.retry_index + 1],
                "processed": len(results),
                "clean": 0,
                "rewritten": 0,
                "errors": len(errors),
                "failed": 1,
                "mode": "retry-index",
                "token_usage": usage_stats,
                "cost_estimate": estimate_cost_usd(args.model, usage_stats),
            })
            print("  → FAILED (logged to error log)")
        return

    # --retry-errors: retry all indices in the error log
    if args.retry_errors:
        error_indices = sorted(int(k) for k in errors.keys())
        error_indices = [i for i in error_indices if start <= i < end]
        if not error_indices:
            print("[RETRY-ERRORS] No errors in log for this range.")
            return
        print(f"[RETRY-ERRORS] Retrying {len(error_indices)} failed index(es): "
              f"{error_indices[:10]}{'...' if len(error_indices) > 10 else ''}")
        targets = {s.get("_index"): s for s in samples}
        for idx in error_indices:
            target = targets.get(idx)
            if target is None:
                print(f"  idx={idx} → NOT FOUND in input (skipping)")
                continue
            result, usage = await process_sample_async(
                target, client, semaphore, args.model,
                detect_prompt, rewrite_prompt, clean_rewrite_prompt,
                args.max_tokens, max_iterations=args.max_iterations,
            )
            merge_usage(usage_stats, usage)
            if result:
                results[idx] = result
                clear_error(errors, idx)
                flags = result.get("_pf_flags", "")
                status = f"[{flags}]" if flags else "clean"
                print(f"  idx={idx} → OK ({status})")
            else:
                log_error(errors, idx, "api_or_parse_failure")
                print(f"  idx={idx} → ERR (still failing)")
        await save_results(results, output_path, lock)
        save_error_log(errors, error_path)
        append_summary(summary_path, {
            "start": run_start.isoformat(),
            "end": datetime.now().isoformat(),
            "model": args.model,
            "range": [start, end],
            "processed": len(results),
            "clean": 0,
            "rewritten": len(error_indices) - len(errors),
            "errors": len(errors),
            "failed": len(errors),
            "mode": "retry-errors",
            "token_usage": usage_stats,
            "cost_estimate": estimate_cost_usd(args.model, usage_stats),
        })
        print(f"Done. Remaining errors: {len(errors)}")
        return

    # Build pending
    range_samples = [s for s in samples if start <= s.get("_index", 0) < end]
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
        result, usage = await process_sample_async(
            sample, client, semaphore, args.model,
            detect_prompt, rewrite_prompt, clean_rewrite_prompt,
            args.max_tokens, max_iterations=args.max_iterations,
        )
        async with lock:
            merge_usage(usage_stats, usage)
            completed += 1
            if result is None:
                stats["failed"] += 1
                log_error(errors, idx, "api_or_parse_failure")
                print(f"  [{completed:>5}/{total_pending}] idx={idx} → FAILED (logged)")
            else:
                results[idx] = result
                clear_error(errors, idx)
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
                    save_error_log(errors, error_path)

    await asyncio.gather(*[process_one(s) for s in pending])

    await save_results(results, output_path, lock)
    save_error_log(errors, error_path)

    append_summary(summary_path, {
        "start":     run_start.isoformat(),
        "end":       datetime.now().isoformat(),
        "model":     args.model,
        "range":     [start, end],
        "processed": len(results),
        "clean":     stats["clean"],
        "rewritten": stats["rewritten"],
        "errors":    len(errors),
        "failed":    stats["failed"],
        "token_usage": usage_stats,
        "cost_estimate": estimate_cost_usd(args.model, usage_stats),
    })

    print(f"\nDone. clean={stats['clean']} rewritten={stats['rewritten']} failed={stats['failed']}")
    print(f"Saved → {output_path}")
    if errors:
        failed_list = sorted(int(k) for k in errors.keys())
        print(f"  Failed indices: {failed_list[:20]}{'...' if len(failed_list) > 20 else ''}")
        print(f"  Re-run with --retry-errors to retry them.")

    # Cost summary
    prompt_tok = usage_stats.get("prompt_tokens", 0)
    compl_tok  = usage_stats.get("completion_tokens", 0)
    total_tok  = prompt_tok + compl_tok
    cost = (prompt_tok / 1_000_000) * args.price_input + \
           (compl_tok  / 1_000_000) * args.price_output
    print(f"\n  Tokens — prompt: {prompt_tok:,}  completion: {compl_tok:,}  total: {total_tok:,}")
    print(f"  Estimated cost: ${cost:.4f}  "
          f"(input ${args.price_input}/1M, output ${args.price_output}/1M)")


def main():
    parser = argparse.ArgumentParser(
        description="Async 2-step post-filtering for parquet-based reasoning output"
    )
    parser.add_argument("--api-key",       default=os.environ.get("OPENROUTER_API_KEY"),
                        help="OpenRouter API key (default: $OPENROUTER_API_KEY)")
    parser.add_argument("--model",         required=True)
    parser.add_argument("--input",         required=True,
                        help="Path to reasoning JSON (e.g. dataset/reasoning/sft_00000.json)")
    parser.add_argument("--detect-prompt",        default=DETECT_PROMPT_DEFAULT)
    parser.add_argument("--rewrite-prompt",       default=REWRITE_PROMPT_DEFAULT)
    parser.add_argument("--clean-rewrite-prompt", default=CLEAN_REWRITE_PROMPT_DEFAULT)
    parser.add_argument("--workflows",            default=WORKFLOWS_DEFAULT,
                        help="reasoning_workflows.md path — appended to rewrite prompt")
    parser.add_argument("--output-dir",     default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--workers",        type=int, default=5,
                        help="Concurrent API calls for detection (default: 5)")
    parser.add_argument("--part",           type=int, default=None,
                        help="Part index (0-based). Divides input into --num-parts equal chunks.")
    parser.add_argument("--num-parts",      type=int, default=4,
                        help="Total number of parts when using --part (default: 4)")
    parser.add_argument("--start-index",    type=int, default=0,
                        help="Start sample index (inclusive). Ignored when --part is set.")
    parser.add_argument("--end-index",      type=int, default=None,
                        help="End sample index (exclusive). Ignored when --part is set.")
    parser.add_argument("--retry-index",    type=int, default=None,
                        help="Re-process only this single _index")
    parser.add_argument("--retry-errors",   action="store_true",
                        help="Retry all indices listed in the error log for this part/range")
    parser.add_argument("--save-every",     type=int, default=20)
    parser.add_argument("--max-tokens",     type=int, default=131072,
                        help="Max tokens for rewrite step (default: 131072)")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Max detect→rewrite iterations per sample (default: 3)")
    # Cost tracking
    parser.add_argument("--price-input",  type=float, default=0.26,
                        help="Input token price per 1M tokens in USD (default: 0.26)")
    parser.add_argument("--price-output", type=float, default=0.38,
                        help="Output token price per 1M tokens in USD (default: 0.38)")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
