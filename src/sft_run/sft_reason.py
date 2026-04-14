"""
sft_reason.py  —  Async reasoning extraction from HuggingFace parquet files

Reads a parquet file, generates reasoning + response for each sample using
the same pipeline as mmfinereason_with_think.py (think-block-only reference).

Input:   dataset/raw/sft-XXXXX-of-YYYYY.parquet
Output:  dataset/reasoning/sft_XXXXX.json

Speed:   asyncio + semaphore for concurrent API calls (~10-20x faster than sequential)

Usage:
  python sft_reason.py --api-key KEY --model MODEL --parquet dataset/raw/sft-00000-of-00193.parquet
  # Split across 2 terminals for 2x more speed:
  python sft_reason.py ... --start-index 0    --end-index 4588 --workers 10
  python sft_reason.py ... --start-index 4588 --end-index 9176 --workers 10
"""

import os
import re
import json
import asyncio
import argparse
import base64
from io import BytesIO
from datetime import datetime

import pandas as pd
from PIL import Image
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))

# ── Paths ─────────────────────────────────────────────────────

BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
TEMPLATE_DEFAULT = os.path.join(BASE_DIR, "prompts", "reasoning_distillation.md")
OUTPUT_DIR_DEFAULT = os.path.join(BASE_DIR, "dataset", "reasoning")

MAX_RETRIES  = 5
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

# ── Helpers ───────────────────────────────────────────────────

def load_template(path: str) -> str:
    if not os.path.exists(path):
        print(f"[WARN] Template not found: {path}")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parquet_stem(parquet_path: str) -> str:
    """
    sft-00000-of-00193.parquet   →  sft_00000
    train-00000-of-00070.parquet →  train_00000
    """
    name = os.path.basename(parquet_path)
    m = re.match(r"([A-Za-z]+)-(\d+)", name)
    return f"{m.group(1)}_{m.group(2)}" if m else name.replace(".parquet", "").replace("-", "_")


def extract_think_block(text: str) -> str:
    """Return only the content inside <think>...</think>."""
    if not text:
        return text
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def encode_image(image_field) -> str | None:
    """
    Accepts multiple image formats from parquet:
      - dict {'bytes': bytes, 'path': str|None}
      - raw bytes
      - PIL Image
    Returns base64-encoded JPEG string or None.
    """
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
        except Exception as e:
            print(f"  [WARN] Image encode failed: {e}")
            return None

    # Already a PIL Image
    if hasattr(image_field, "save"):
        try:
            buf = BytesIO()
            image_field.convert("RGB").save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"  [WARN] PIL image encode failed: {e}")
            return None

    return None


def build_user_content(template: str, question: str, options, base64_image: str | None,
                        qwen_thinking: str, answer: str) -> list:
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


def row_to_sample(row, row_idx: int) -> dict:
    """Convert a parquet row (dict) to a sample dict, preserving all metadata."""
    question = str(row.get("question", "") or "").replace("<image>", "").strip()
    source   = str(row.get("source", "unknown") or "unknown")

    # Serialize non-JSON-safe types
    sample = {
        "_index":  row_idx,
        "_subset": source,
    }
    for col, val in row.items():
        if col == "image":
            continue  # handled separately (binary)
        if isinstance(val, (bool,)):
            sample[col] = bool(val)
        elif hasattr(val, "item"):          # numpy scalar
            sample[col] = val.item()
        elif val is None or val != val:     # nan check
            sample[col] = None
        else:
            sample[col] = val

    sample["question_clean"] = question    # cleaned question without <image>
    sample["_image_available"] = (
        row.get("image") is not None and
        isinstance(row.get("image"), dict) and
        row["image"].get("bytes") is not None
    )
    return sample


def empty_usage_stats() -> dict:
    return {
        "requests": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
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

# ── Async API call ────────────────────────────────────────────

async def call_api_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    content: list,
    gen_config: dict,
    label: str,
) -> tuple[str | None, str | None, dict]:
    """Returns (reasoning, response, usage) or (None, None, empty usage) on failure."""
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
                        "top_k":             gen_config["top_k"],
                        "min_p":             gen_config["min_p"],
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
                    print(f"\n  [ERROR] {label}: {e}")
                    return None, None, empty_usage_stats()
    return None, None, empty_usage_stats()

# ── Per-sample async processing ───────────────────────────────

async def process_sample_async(
    row: dict,
    row_idx: int,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    template: str,
    gen_config: dict,
    loop: asyncio.AbstractEventLoop,
) -> tuple[dict, dict]:
    sample  = row_to_sample(row, row_idx)
    label   = f"{sample['_subset']}_{row_idx}"
    question = sample["question_clean"]
    options  = sample.get("options")
    answer   = str(sample.get("answer", "") or "")

    raw_thinking  = str(sample.get("qwen3vl_235b_thinking_response", "") or "")
    qwen_thinking = extract_think_block(raw_thinking)

    # Image encoding (CPU-bound → executor)
    base64_image = None
    if sample.get("_image_available"):
        try:
            base64_image = await loop.run_in_executor(
                None, encode_image, row.get("image")
            )
        except Exception as e:
            print(f"  [WARN] {label} image encode error: {e}")

    content = build_user_content(template, question, options, base64_image, qwen_thinking, answer)

    reasoning, response, usage = await call_api_async(client, semaphore, model, content, gen_config, label)

    result = dict(sample)
    result["reasoning"] = reasoning
    result["response"]  = response
    return result, usage

# ── Save helper ───────────────────────────────────────────────

async def save_results(results: dict, output_path: str, lock: asyncio.Lock):
    async with lock:
        sorted_list = sorted(results.values(), key=lambda r: r.get("_index", 0))
        # Remove binary-unsafe fields before saving
        clean = []
        for r in sorted_list:
            row = {k: v for k, v in r.items() if k != "_image_available"}
            clean.append(row)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)

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

    df = pd.read_parquet(args.parquet)
    total_rows = len(df)
    print(f"Loaded parquet: {total_rows} rows, columns: {df.columns.tolist()}")

    stem = parquet_stem(args.parquet)

    # Subdirectory per parquet: reasoning/00000/
    m_num = re.search(r"_(\d+)$", stem)
    parquet_num  = m_num.group(1) if m_num else stem
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
        if has_range:
            suffix = f"_s{start}_e{end}"
        else:
            suffix = ""
        output_path  = os.path.join(output_subdir, f"{stem}{suffix}.json")
        error_path   = os.path.join(log_dir,       f"{stem}{suffix}_errors.json")
        summary_path = os.path.join(log_dir,       f"{stem}{suffix}_summary.json")

    print(f"Processing rows [{start}, {end}) of {total_rows}")
    print(f"Output:    {output_path}")
    print(f"Error log: {error_path}")
    print(f"Workers (concurrent): {args.workers}")

    # Load existing results for resume
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
                print(f"[RESUME] Already done: {len(done_indices)} sample(s) — skipping")
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[WARN] Output file corrupted ({e}), starting fresh from this file.")

    errors = load_error_log(error_path)
    if errors:
        print(f"[RESUME] Error log: {len(errors)} previously failed index(es)")

    run_start = datetime.now()

    template  = load_template(args.template_path)
    client    = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.workers)
    lock      = asyncio.Lock()
    loop      = asyncio.get_event_loop()

    gen_config = {
        "temperature":        args.temperature,
        "top_p":              args.top_p,
        "top_k":              args.top_k,
        "min_p":              args.min_p,
        "presence_penalty":   args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "n":                  args.n,
        "max_tokens":         args.max_tokens,
    }
    usage_stats = empty_usage_stats()

    # --retry-index: single sample
    if args.retry_index is not None:
        if args.retry_index < 0 or args.retry_index >= total_rows:
            print(f"[ERROR] --retry-index {args.retry_index} out of range [0, {total_rows})")
            return
        print(f"[RETRY] Re-generating index={args.retry_index}")
        row = df.iloc[args.retry_index].to_dict()
        result, usage = await process_sample_async(
            row, args.retry_index, client, semaphore, args.model, template, gen_config, loop
        )
        merge_usage(usage_stats, usage)
        results[args.retry_index] = result
        if result.get("response"):
            clear_error(errors, args.retry_index)
            status = "OK"
        else:
            log_error(errors, args.retry_index, "empty_response")
            status = "ERR"
        await save_results(results, output_path, lock)
        save_error_log(errors, error_path)
        append_summary(summary_path, {
            "start": run_start.isoformat(),
            "end": datetime.now().isoformat(),
            "model": args.model,
            "range": [args.retry_index, args.retry_index + 1],
            "processed": len(results),
            "ok": 1 if result.get("response") else 0,
            "errors": len(errors),
            "mode": "retry-index",
            "token_usage": usage_stats,
            "cost_estimate": estimate_cost_usd(args.model, usage_stats),
        })
        print(f"  → {status}  Saved → {output_path}")
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
            result, usage = await process_sample_async(
                row, row_idx, client, semaphore, args.model, template, gen_config, loop
            )
            merge_usage(usage_stats, usage)
            results[row_idx] = result
            if result.get("response"):
                clear_error(errors, row_idx)
                print(f"  idx={row_idx} → OK")
            else:
                log_error(errors, row_idx, "empty_response")
                print(f"  idx={row_idx} → ERR (still failing)")
        await save_results(results, output_path, lock)
        save_error_log(errors, error_path)
        append_summary(summary_path, {
            "start": run_start.isoformat(),
            "end": datetime.now().isoformat(),
            "model": args.model,
            "range": [start, end],
            "processed": len(results),
            "ok": sum(1 for i in error_indices if i not in errors),
            "errors": len(errors),
            "mode": "retry-errors",
            "token_usage": usage_stats,
            "cost_estimate": estimate_cost_usd(args.model, usage_stats),
        })
        print(f"Done. Remaining errors: {len(errors)}")
        return

    # Build pending list
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
        result, usage = await process_sample_async(
            row, row_idx, client, semaphore, args.model, template, gen_config, loop
        )
        async with lock:
            merge_usage(usage_stats, usage)
            results[row_idx] = result
            completed += 1
            if result.get("response"):
                clear_error(errors, row_idx)
                status = "OK"
            else:
                log_error(errors, row_idx, "empty_response")
                status = "ERR"
            print(
                f"  [{completed:>5}/{total_pending}] idx={row_idx} "
                f"({result.get('_subset','?')}) → {status}",
                flush=True
            )
            # Incremental save every N completions
            if completed % args.save_every == 0:
                sorted_list = sorted(results.values(), key=lambda r: r.get("_index", 0))
                clean = [{k: v for k, v in r.items() if k != "_image_available"} for r in sorted_list]
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(clean, f, ensure_ascii=False, indent=2)
                save_error_log(errors, error_path)

    await asyncio.gather(*[process_one(i) for i in pending_indices])

    # Final save
    await save_results(results, output_path, lock)
    save_error_log(errors, error_path)

    n_ok  = sum(1 for r in results.values() if r.get("response"))
    n_err = len(errors)
    append_summary(summary_path, {
        "start":     run_start.isoformat(),
        "end":       datetime.now().isoformat(),
        "model":     args.model,
        "range":     [start, end],
        "processed": len(results),
        "ok":        n_ok,
        "errors":    n_err,
        "token_usage": usage_stats,
        "cost_estimate": estimate_cost_usd(args.model, usage_stats),
    })

    print(f"\nDone. {n_ok} OK, {n_err} failed → {output_path}")
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
        description="Async reasoning extraction from parquet (with_think pipeline)"
    )
    parser.add_argument("--api-key",       default=os.environ.get("OPENROUTER_API_KEY"),
                        help="OpenRouter API key (default: $OPENROUTER_API_KEY)")
    parser.add_argument("--model",         required=True, help="OpenRouter model ID")
    parser.add_argument("--parquet",       required=True, help="Path to parquet file")
    parser.add_argument("--template-path", default=TEMPLATE_DEFAULT)
    parser.add_argument("--output-dir",    default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--workers",       type=int, default=10,
                        help="Number of concurrent API calls (default: 10)")
    parser.add_argument("--part",          type=int, default=None,
                        help="Part index (0-based). Divides parquet into --num-parts equal chunks.")
    parser.add_argument("--num-parts",     type=int, default=4,
                        help="Total number of parts when using --part (default: 4)")
    parser.add_argument("--start-index",   type=int, default=0,
                        help="Start row index (inclusive). Ignored when --part is set.")
    parser.add_argument("--end-index",     type=int, default=None,
                        help="End row index (exclusive). Ignored when --part is set.")
    parser.add_argument("--retry-index",   type=int, default=None,
                        help="Re-generate only this single row index")
    parser.add_argument("--retry-errors",  action="store_true",
                        help="Retry all indices listed in the error log for this part/range")
    parser.add_argument("--save-every",    type=int, default=20,
                        help="Save results every N completions (default: 20)")
    # Generation config
    parser.add_argument("--temperature",        type=float, default=1.0)
    parser.add_argument("--top-p",              type=float, default=0.95)
    parser.add_argument("--top-k",              type=int,   default=20)
    parser.add_argument("--min-p",              type=float, default=0.0)
    parser.add_argument("--presence-penalty",   type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--n",                  type=int,   default=1)
    parser.add_argument("--max-tokens",         type=int,   default=81920)
    # Cost tracking
    parser.add_argument("--price-input",  type=float, default=0.26,
                        help="Input token price per 1M tokens in USD (default: 0.26)")
    parser.add_argument("--price-output", type=float, default=0.38,
                        help="Output token price per 1M tokens in USD (default: 0.38)")
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
