"""
sft_2step_pipeline.py  —  Async 2-step VLM→LLM pipeline for parquet-based SFT data

Step 1 (VLM generation):
  - Uses the same multimodal generation flow as sft_reason.py
  - Input: image + question + original qwen think block + answer
  - Output: new reasoning + response

Step 2 (LLM post-process):
  - Reconstructs the generated reasoning into an implicit workflow-shaped form
  - Removes reference/source mentions so the sample reads like a single model solved it directly
  - Polishes the response into natural explanatory prose

Input:   dataset/raw/.../train-NNNNN-of-00070.parquet
Output:  dataset/two_step_pipeline/NNNNN/train_NNNNN_partP.json
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
from PIL import Image
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

# ── Paths ─────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
TEMPLATE_DEFAULT = os.path.join(BASE_DIR, "prompts", "reasoning_distillation.md")
RECONSTRUCT_PROMPT_DEFAULT = os.path.join(BASE_DIR, "prompts", "reconstruct_prompt.md")
WORKFLOWS_DEFAULT = os.path.join(BASE_DIR, "prompts", "reasoning_workflows.md")
OUTPUT_DIR_DEFAULT = os.path.join(BASE_DIR, "dataset", "two_step_pipeline")

MAX_RETRIES = 5
RETRY_DELAYS = [10, 30, 60, 120, 180]

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

REFERENCE_HARD_PATTERNS = [
    "existing reasoning",
    "original reasoning",
    "provided reasoning",
    "provided thinking",
    "source material",
    "previous answer suggests",
    "existing response",
    "original response",
    "the reasoning concluded",
    "the previous answer concluded",
    "according to the existing",
    "based on the existing",
    "according to the original",
    "based on the original",
]

REFERENCE_SAFE_PHRASES = [
    "reference point",
    "reference angle",
    "reference frame",
    "potential energy reference",
    "for reference",
    "cross-reference",
]


# ── Generic helpers ───────────────────────────────────────────

def load_prompt(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_reconstruct_prompt(prompt_path: str, workflows_path: str) -> str:
    prompt = load_prompt(prompt_path)
    workflows = load_prompt(workflows_path)
    return prompt + "\n\n---\n\n## Full Workflow Reference\n\n" + workflows


def parquet_stem(parquet_path: str) -> str:
    name = os.path.basename(parquet_path)
    m = re.match(r"([A-Za-z]+)-(\d+)", name)
    return f"{m.group(1)}_{m.group(2)}" if m else name.replace(".parquet", "").replace("-", "_")


def extract_think_block(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def encode_image(image_field) -> str | None:
    if image_field is None:
        return None

    img_bytes = None
    if isinstance(image_field, dict):
        img_bytes = image_field.get("bytes")
    elif isinstance(image_field, bytes):
        img_bytes = image_field

    if img_bytes is not None:
        try:
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            buf = BytesIO()
            img.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            return None

    if hasattr(image_field, "save"):
        try:
            buf = BytesIO()
            image_field.convert("RGB").save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            return None

    return None


def row_to_sample(row, row_idx: int) -> dict:
    question = str(row.get("question", "") or "").replace("<image>", "").strip()
    source = str(row.get("source", "unknown") or "unknown")

    sample = {
        "_index": row_idx,
        "_subset": source,
    }
    for col, val in row.items():
        if col == "image":
            continue
        if isinstance(val, bool):
            sample[col] = bool(val)
        elif hasattr(val, "item"):
            sample[col] = val.item()
        elif val is None or val != val:
            sample[col] = None
        else:
            sample[col] = val

    sample["question_clean"] = question
    sample["_image_available"] = (
        row.get("image") is not None and
        isinstance(row.get("image"), dict) and
        row["image"].get("bytes") is not None
    )
    return sample


# ── Usage / cost helpers ─────────────────────────────────────

def empty_usage_stats() -> dict:
    return {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def empty_pipeline_usage() -> dict:
    return {
        "vlm": empty_usage_stats(),
        "llm": empty_usage_stats(),
    }


def extract_usage(resp) -> dict:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return empty_usage_stats()

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


def merge_usage(stats: dict, delta: dict | None) -> dict:
    if not delta:
        return stats
    for key in ("requests", "prompt_tokens", "completion_tokens", "total_tokens"):
        stats[key] = int(stats.get(key, 0) or 0) + int(delta.get(key, 0) or 0)
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


def estimate_pipeline_cost(vlm_model: str, llm_model: str, usage: dict) -> dict:
    vlm_cost = estimate_cost_usd(vlm_model, usage.get("vlm", {}))
    llm_cost = estimate_cost_usd(llm_model, usage.get("llm", {}))
    total_cost = 0.0
    for part in (vlm_cost, llm_cost):
        if part.get("total_cost_usd") is not None:
            total_cost += part["total_cost_usd"]
    return {
        "vlm": vlm_cost,
        "llm": llm_cost,
        "total_cost_usd": round(total_cost, 6),
    }


# ── VLM generation helpers ───────────────────────────────────

def build_vlm_user_content(
    template: str,
    question: str,
    options,
    base64_image: str | None,
    qwen_thinking: str,
    answer: str,
) -> list:
    options_str = ""
    if options:
        if isinstance(options, list):
            labels = list("ABCDEFGHIJ")
            options_str = "\n### Options:\n" + "\n".join(
                f"({labels[i]}) {opt}" for i, opt in enumerate(options) if i < len(labels)
            )
        else:
            options_str = f"\n### Options:\n{options}"

    content = []
    if base64_image:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        })

    body = (
        f"### Input Question:\n{question}"
        f"{options_str}"
        f"\n\n### Reasoning:\n{qwen_thinking or ''}"
        f"\n\n### Response:\n{answer or ''}"
        f"\n\n{template}"
        f"\n\n### Input Question:\n{question}"
        f"{options_str}"
    )
    content.append({"type": "text", "text": body})
    return content


async def call_vlm_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    content: list,
    gen_config: dict,
    label: str,
) -> tuple[str | None, str | None, dict]:
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    temperature=gen_config["temperature"],
                    top_p=gen_config["top_p"],
                    presence_penalty=gen_config["presence_penalty"],
                    max_tokens=gen_config["max_tokens"],
                    n=gen_config["n"],
                    extra_body={
                        "include_reasoning": True,
                        "top_k": gen_config["top_k"],
                        "min_p": gen_config["min_p"],
                        "repetition_penalty": gen_config["repetition_penalty"],
                    },
                )
                msg = resp.choices[0].message
                return getattr(msg, "reasoning", None), msg.content, extract_usage(resp)
            except Exception as e:
                err = str(e)
                is_rate = "429" in err or "rate" in err.lower()
                if is_rate and attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAYS[attempt]
                    print(f"\n  [RATE LIMIT] {label} — waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                    await asyncio.sleep(wait)
                else:
                    print(f"\n  [API ERROR] {label} | stage=vlm: {e}")
                    return None, None, empty_usage_stats()
    return None, None, empty_usage_stats()


# ── LLM reconstruction helpers ────────────────────────────────

def parse_reconstruct_output(raw: str) -> dict | None:
    """Extract <WORKFLOW> and <REASONING>; parse everything after </REASONING> as response."""
    r_m = re.search(r"<REASONING>(.*?)</REASONING>", raw, re.DOTALL)
    wf_m = re.search(r"<WORKFLOW>(.*?)</WORKFLOW>", raw, re.DOTALL)
    if not r_m and not wf_m:
        return None
    response = raw[r_m.end():].strip() if r_m else None
    return {
        "reasoning": r_m.group(1).strip() if r_m else None,
        "response": response,
        "workflow": wf_m.group(1).strip() if wf_m else None,
    }


def response_has_phase_leakage(text: str) -> bool:
    if not text:
        return False
    return any([
        re.search(r"(?im)^\s*#{1,6}\s*phase\b", text) is not None,
        re.search(r"(?im)\bphase\s*\d+\b", text) is not None,
        re.search(r"(?m)^\[[^\n\]]+\]\s*$", text) is not None,
    ])


def reasoning_has_template_leakage(text: str) -> bool:
    if not text:
        return False
    return any([
        re.search(r"(?m)^\[[^\n\]]+\]\s*$", text) is not None,
        re.search(r"(?im)^\s*#{1,6}\s*phase\b", text) is not None,
    ])


def has_reference_contamination(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    for safe in REFERENCE_SAFE_PHRASES:
        lowered = lowered.replace(safe, "")
    return any(pat in lowered for pat in REFERENCE_HARD_PATTERNS)


def get_post_issues(parsed: dict | None) -> list[str]:
    issues = []
    if not parsed:
        return ["parse_failure"]

    reasoning = (parsed.get("reasoning") or "").strip()
    response = (parsed.get("response") or "").strip()

    if not reasoning:
        issues.append("reasoning_empty")
    if not response:
        issues.append("response_empty")
    if reasoning and reasoning_has_template_leakage(reasoning):
        issues.append("reasoning_template_leakage")
    if response and response_has_phase_leakage(response):
        issues.append("response_phase_leakage")
    if reasoning and has_reference_contamination(reasoning):
        issues.append("reasoning_reference_contamination")
    if response and has_reference_contamination(response):
        issues.append("response_reference_contamination")

    return issues


def build_llm_rewrite_message(
    question: str,
    options,
    reasoning: str,
    response: str,
    answer: str,
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
    return (
        f"### Question:\n{question}{options_str}{answer_str}\n\n"
        f"---\n\n"
        f"### Generated Reasoning:\n{reasoning or '(empty)'}\n\n"
        f"### Generated Response:\n{response or '(empty)'}\n\n"
        f"---\n\n"
        f"Use the generated reasoning and response as source material, but rewrite them so the final sample reads like a single VLM directly observed the image and solved the problem itself.\n\n"
        f"CRITICAL GOALS:\n"
        f"- Remove all reference contamination and any hint that this came from prior reasoning, source material, previous answers, or post-processing.\n"
        f"- Reconstruct the reasoning using the most appropriate workflow as invisible internal scaffolding.\n"
        f"- Preserve the full substantive reasoning: calculations, detours, option checks, self-corrections, and verification.\n"
        f"- Do not summarize the reasoning into a short neat proof.\n"
        f"- Do not expose workflow headers, bracket subtitles, or phase labels in the final reasoning.\n"
        f"- After </REASONING>, write a polished plain-prose response that keeps the important observations and core logic, but feels like a natural final answer explanation rather than a template.\n"
        f"- The final response must not contain phase labels, bracket subtitles, or source/reconstruction mentions.\n"
        f"- Match the answer format the question expects.\n\n"
        f"FORMAT REQUIREMENTS:\n"
        f"- Output exactly one <WORKFLOW> block and one <REASONING> block.\n"
        f"- <REASONING> must be non-empty.\n"
        f"- After </REASONING>, continue with a non-empty final response in plain explanatory prose.\n"
        f"- Do not stop after </REASONING>.\n"
    )


def build_llm_retry_message(
    question: str,
    options,
    reasoning: str,
    response: str,
    answer: str,
    issues: list[str],
) -> str:
    base = build_llm_rewrite_message(question, options, reasoning, response, answer)
    return (
        base
        + "\n\nIMPORTANT CORRECTION FOR RETRY:\n"
        + f"- The previous output had these issues: {', '.join(issues)}.\n"
        + "- Rewrite the sample again.\n"
        + "- Keep the reasoning full and detailed; do not shorten it into a summary.\n"
        + "- Remove all source-oriented language such as 'existing reasoning', 'provided materials', or 'source material'.\n"
        + "- The reasoning must read like direct thinking about the image and question.\n"
        + "- The response must be polished plain prose and must not contain phase labels, bracket subtitles, or workflow/template markers.\n"
    )


async def call_llm_async(
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
                        {"role": "user", "content": user},
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
                    print(f"\n  [API ERROR] {label} | stage=llm: {e}")
                    return None, empty_usage_stats()
    return None, empty_usage_stats()


# ── Error / summary helpers ───────────────────────────────────

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


def log_error(errors: dict, index: int, reason: str, stage: str, raw_response: str | None = None):
    entry = {
        "reason": reason,
        "stage": stage,
        "timestamp": datetime.now().isoformat(),
    }
    if raw_response:
        entry["model_response"] = raw_response
    errors[str(index)] = entry


def clear_error(errors: dict, index: int):
    errors.pop(str(index), None)


def append_summary(path: str, run_info: dict):
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

    agg_vlm = empty_usage_stats()
    agg_llm = empty_usage_stats()
    total_processed = 0
    total_failed = 0
    total_cost = 0.0
    for run in runs:
        merge_usage(agg_vlm, (run.get("token_usage") or {}).get("vlm"))
        merge_usage(agg_llm, (run.get("token_usage") or {}).get("llm"))
        total_processed += int(run.get("processed", 0) or 0)
        total_failed += int(run.get("failed", 0) or 0)
        total_cost += float((run.get("cost_estimate") or {}).get("total_cost_usd", 0.0) or 0.0)

    data["aggregate"] = {
        "run_count": len(runs),
        "processed": total_processed,
        "failed": total_failed,
        "token_usage": {
            "vlm": agg_vlm,
            "llm": agg_llm,
        },
        "estimated_cost_usd": round(total_cost, 6),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Per-sample pipeline ───────────────────────────────────────

async def process_sample_async(
    row: dict,
    row_idx: int,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    vlm_model: str,
    llm_model: str,
    template: str,
    reconstruct_prompt: str,
    vlm_gen_config: dict,
    llm_max_tokens: int,
    loop: asyncio.AbstractEventLoop,
    llm_max_iterations: int = 2,
) -> tuple[dict | None, dict, str | None, str | None]:
    usage = empty_pipeline_usage()
    sample = row_to_sample(row, row_idx)
    label = f"{sample['_subset']}_{row_idx}"
    question = sample["question_clean"]
    options = sample.get("options")
    answer = str(sample.get("answer", "") or "")

    raw_thinking = str(sample.get("qwen3vl_235b_thinking_response", "") or "")
    qwen_thinking = extract_think_block(raw_thinking)

    # Step 1: VLM generation
    base64_image = None
    if sample.get("_image_available"):
        try:
            base64_image = await loop.run_in_executor(None, encode_image, row.get("image"))
        except Exception as e:
            print(f"  [WARN] {label} image encode error: {e}")

    content = build_vlm_user_content(template, question, options, base64_image, qwen_thinking, answer)
    gen_reasoning, gen_response, step_usage = await call_vlm_async(
        client, semaphore, vlm_model, content, vlm_gen_config, label + "/vlm"
    )
    merge_usage(usage["vlm"], step_usage)

    if not gen_reasoning or not gen_response:
        return None, usage, "vlm_empty_output", None

    # Step 2: LLM rewrite / polish
    current_reasoning = gen_reasoning
    current_response = gen_response
    last_raw = None
    last_issues = []
    chosen_workflow = None

    for iteration in range(llm_max_iterations):
        if iteration == 0:
            user_msg = build_llm_rewrite_message(question, options, current_reasoning, current_response, answer)
        else:
            user_msg = build_llm_retry_message(
                question, options, current_reasoning, current_response, answer, last_issues
            )

        raw, step_usage = await call_llm_async(
            client, semaphore, llm_model, reconstruct_prompt, user_msg, llm_max_tokens, f"{label}/llm{iteration+1}"
        )
        merge_usage(usage["llm"], step_usage)
        last_raw = raw
        if raw is None:
            continue

        parsed = parse_reconstruct_output(raw)
        last_issues = get_post_issues(parsed)
        if not last_issues:
            chosen_workflow = parsed.get("workflow")
            return {
                **sample,
                "reasoning_vlm": gen_reasoning,
                "response_vlm": gen_response,
                "reasoning": parsed["reasoning"],
                "response": parsed["response"],
                "pipeline_workflow": chosen_workflow,
                "pipeline_iterations": iteration + 1,
            }, usage, None, None

    reason = ",".join(last_issues) if last_issues else "llm_rewrite_failed"
    return None, usage, reason, last_raw


# ── Save helper ───────────────────────────────────────────────

async def save_results(results: dict, output_path: str, lock: asyncio.Lock):
    async with lock:
        sorted_list = sorted(results.values(), key=lambda r: r.get("_index", 0))
        clean = []
        for r in sorted_list:
            row = {k: v for k, v in r.items() if k != "_image_available"}
            clean.append(row)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)


# ── Main ──────────────────────────────────────────────────────

async def async_main(args):
    if not args.api_key:
        print("[ERROR] No API key provided. Set OPENROUTER_API_KEY in .env or pass --api-key.")
        return

    df = pd.read_parquet(args.parquet)
    total_rows = len(df)
    print(f"Loaded parquet: {total_rows} rows")

    stem = parquet_stem(args.parquet)
    m_num = re.search(r"_(\d+)$", stem)
    parquet_num = m_num.group(1) if m_num else stem
    output_subdir = os.path.join(args.output_dir, parquet_num)
    log_dir = os.path.join(output_subdir, "logs")
    os.makedirs(output_subdir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.part is not None:
        n = args.num_parts
        p = args.part
        if p < 0 or p >= n:
            print(f"[ERROR] --part {p} out of range [0, {n})")
            return
        chunk = total_rows // n
        start = p * chunk
        end = total_rows if p == n - 1 else (p + 1) * chunk
        output_path = os.path.join(output_subdir, f"{stem}_part{p}.json")
        error_path = os.path.join(log_dir, f"{stem}_part{p}_errors.json")
        summary_path = os.path.join(log_dir, f"{stem}_part{p}_summary.json")
    else:
        start = args.start_index
        end = args.end_index if args.end_index is not None else total_rows
        end = min(end, total_rows)
        has_range = (args.start_index != 0 or args.end_index is not None)
        suffix = f"_s{start}_e{end}" if has_range else ""
        output_path = os.path.join(output_subdir, f"{stem}{suffix}.json")
        error_path = os.path.join(log_dir, f"{stem}{suffix}_errors.json")
        summary_path = os.path.join(log_dir, f"{stem}{suffix}_summary.json")

    print(f"Processing rows [{start}, {end}) of {total_rows}")
    print(f"Output:    {output_path}")
    print(f"Error log: {error_path}")
    print(f"Workers:   {args.workers}")
    print(f"VLM model: {args.vlm_model}")
    print(f"LLM model: {args.llm_model}")

    # Resume
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
                print(f"[RESUME] Already done: {len(done_indices)} sample(s)")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[WARN] Output file corrupted ({e}), starting fresh from this file.")

    errors = load_error_log(error_path)
    if errors:
        print(f"[RESUME] Error log: {len(errors)} previously failed index(es)")

    run_start = datetime.now()
    template = load_prompt(args.template_path)
    reconstruct_prompt = load_reconstruct_prompt(args.reconstruct_prompt, args.workflows)
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.workers)
    lock = asyncio.Lock()
    loop = asyncio.get_event_loop()

    vlm_gen_config = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "n": args.n,
        "max_tokens": args.vlm_max_tokens,
    }

    usage_stats = empty_pipeline_usage()

    async def process_index(row_idx: int) -> tuple[dict | None, dict, str | None, str | None]:
        row = df.iloc[row_idx].to_dict()
        return await process_sample_async(
            row,
            row_idx,
            client,
            semaphore,
            args.vlm_model,
            args.llm_model,
            template,
            reconstruct_prompt,
            vlm_gen_config,
            args.llm_max_tokens,
            loop,
            llm_max_iterations=args.llm_max_iterations,
        )

    # retry-index
    if args.retry_index is not None:
        if args.retry_index < 0 or args.retry_index >= total_rows:
            print(f"[ERROR] --retry-index {args.retry_index} out of range [0, {total_rows})")
            return
        print(f"[RETRY] Re-processing index={args.retry_index}")
        result, usage, reason, raw = await process_index(args.retry_index)
        merge_usage(usage_stats["vlm"], usage.get("vlm"))
        merge_usage(usage_stats["llm"], usage.get("llm"))
        if result:
            results[args.retry_index] = result
            clear_error(errors, args.retry_index)
            status = "OK"
        else:
            log_error(errors, args.retry_index, reason or "pipeline_failed", stage="pipeline", raw_response=raw)
            status = f"ERR ({reason or 'pipeline_failed'})"
        await save_results(results, output_path, lock)
        save_error_log(errors, error_path)
        append_summary(summary_path, {
            "start": run_start.isoformat(),
            "end": datetime.now().isoformat(),
            "mode": "retry-index",
            "models": {"vlm": args.vlm_model, "llm": args.llm_model},
            "range": [args.retry_index, args.retry_index + 1],
            "processed": 1 if result else 0,
            "failed": 0 if result else 1,
            "errors": len(errors),
            "token_usage": usage_stats,
            "cost_estimate": estimate_pipeline_cost(args.vlm_model, args.llm_model, usage_stats),
        })
        print(f"  → {status}  Saved → {output_path}")
        return

    # retry-errors
    if args.retry_errors:
        error_indices = sorted(int(k) for k in errors.keys())
        error_indices = [i for i in error_indices if start <= i < end and 0 <= i < total_rows]
        if not error_indices:
            print("[RETRY-ERRORS] No errors in log for this range.")
            return
        print(f"[RETRY-ERRORS] Retrying {len(error_indices)} failed index(es): "
              f"{error_indices[:10]}{'...' if len(error_indices) > 10 else ''}")
        ok = 0
        failed = 0
        for row_idx in error_indices:
            result, usage, reason, raw = await process_index(row_idx)
            merge_usage(usage_stats["vlm"], usage.get("vlm"))
            merge_usage(usage_stats["llm"], usage.get("llm"))
            if result:
                results[row_idx] = result
                clear_error(errors, row_idx)
                ok += 1
                print(f"  idx={row_idx} → OK ({result.get('pipeline_workflow', '?')})")
            else:
                log_error(errors, row_idx, reason or "pipeline_failed", stage="pipeline", raw_response=raw)
                failed += 1
                print(f"  idx={row_idx} → ERR ({reason or 'pipeline_failed'})")
        await save_results(results, output_path, lock)
        save_error_log(errors, error_path)
        append_summary(summary_path, {
            "start": run_start.isoformat(),
            "end": datetime.now().isoformat(),
            "mode": "retry-errors",
            "models": {"vlm": args.vlm_model, "llm": args.llm_model},
            "range": [start, end],
            "processed": ok,
            "failed": failed,
            "errors": len(errors),
            "token_usage": usage_stats,
            "cost_estimate": estimate_pipeline_cost(args.vlm_model, args.llm_model, usage_stats),
        })
        print(f"Done. Remaining errors: {len(errors)}")
        return

    pending_indices = [i for i in range(start, end) if i not in done_indices]
    if not pending_indices:
        print("[SKIP] All samples in range already processed.")
        return

    print(f"Pending: {len(pending_indices)} samples")
    completed = 0
    total_pending = len(pending_indices)
    stats = {"ok": 0, "failed": 0}

    async def process_one(row_idx: int):
        nonlocal completed
        result, usage, reason, raw = await process_index(row_idx)
        async with lock:
            merge_usage(usage_stats["vlm"], usage.get("vlm"))
            merge_usage(usage_stats["llm"], usage.get("llm"))
            completed += 1
            if result:
                results[row_idx] = result
                clear_error(errors, row_idx)
                stats["ok"] += 1
                print(
                    f"  [{completed:>5}/{total_pending}] idx={row_idx} "
                    f"({result.get('_subset','?')}) → OK [{result.get('pipeline_workflow','?')}]",
                    flush=True
                )
            else:
                log_error(errors, row_idx, reason or "pipeline_failed", stage="pipeline", raw_response=raw)
                stats["failed"] += 1
                print(
                    f"  [{completed:>5}/{total_pending}] idx={row_idx} → FAILED ({reason or 'pipeline_failed'})",
                    flush=True
                )
            if completed % args.save_every == 0:
                await save_results(results, output_path, lock)
                save_error_log(errors, error_path)

    await asyncio.gather(*[process_one(i) for i in pending_indices])

    await save_results(results, output_path, lock)
    save_error_log(errors, error_path)
    append_summary(summary_path, {
        "start": run_start.isoformat(),
        "end": datetime.now().isoformat(),
        "models": {"vlm": args.vlm_model, "llm": args.llm_model},
        "range": [start, end],
        "processed": stats["ok"],
        "failed": stats["failed"],
        "errors": len(errors),
        "token_usage": usage_stats,
        "cost_estimate": estimate_pipeline_cost(args.vlm_model, args.llm_model, usage_stats),
    })

    print(f"\nDone. ok={stats['ok']} failed={stats['failed']}")
    print(f"Saved → {output_path}")
    if errors:
        failed_list = sorted(int(k) for k in errors.keys())
        print(f"  Failed indices: {failed_list[:20]}{'...' if len(failed_list) > 20 else ''}")
        print("  Re-run with --retry-errors to retry them.")


def main():
    parser = argparse.ArgumentParser(
        description="Async 2-step VLM→LLM pipeline for parquet-based SFT data"
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY"),
                        help="OpenRouter API key (default: $OPENROUTER_API_KEY)")
    parser.add_argument("--vlm-model", default="qwen/qwen3.5-397b-a17b",
                        help="VLM model for step 1 generation")
    parser.add_argument("--llm-model", default="deepseek/deepseek-v3.2",
                        help="LLM model for step 2 reconstruction/post-process")
    parser.add_argument("--parquet", required=True, help="Input parquet file")
    parser.add_argument("--template-path", default=TEMPLATE_DEFAULT,
                        help="reasoning_distillation template for step 1")
    parser.add_argument("--reconstruct-prompt", default=RECONSTRUCT_PROMPT_DEFAULT,
                        help="reconstruct system prompt for step 2")
    parser.add_argument("--workflows", default=WORKFLOWS_DEFAULT,
                        help="reasoning_workflows.md to append to reconstruct prompt")
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT,
                        help=f"Output directory (default: {OUTPUT_DIR_DEFAULT})")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent API calls (default: 8)")
    parser.add_argument("--part", type=int, default=None,
                        help="Split file into --num-parts chunks and process one part")
    parser.add_argument("--num-parts", type=int, default=4,
                        help="Number of equal parts for --part splitting (default: 4)")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Inclusive start row index")
    parser.add_argument("--end-index", type=int, default=None,
                        help="Exclusive end row index")
    parser.add_argument("--retry-index", type=int, default=None,
                        help="Retry a single row index")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Retry all failed indices from the error log")
    parser.add_argument("--save-every", type=int, default=20,
                        help="Flush output every N completed samples")
    parser.add_argument("--llm-max-iterations", type=int, default=2,
                        help="Max rewrite attempts for the LLM post-process")

    # Step 1 generation config
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--vlm-max-tokens", type=int, default=131072)
    parser.add_argument("--llm-max-tokens", type=int, default=32768)

    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
