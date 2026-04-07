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

# ── Paths ─────────────────────────────────────────────────────

BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
TEMPLATE_DEFAULT = os.path.join(BASE_DIR, "prompts", "reasoning_distillation.md")
OUTPUT_DIR_DEFAULT = os.path.join(BASE_DIR, "dataset", "reasoning")

MAX_RETRIES  = 5
RETRY_DELAYS = [10, 30, 60, 120, 180]

# ── Helpers ───────────────────────────────────────────────────

def load_template(path: str) -> str:
    if not os.path.exists(path):
        print(f"[WARN] Template not found: {path}")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parquet_stem(parquet_path: str) -> str:
    """sft-00000-of-00193.parquet  →  sft_00000"""
    name = os.path.basename(parquet_path)
    m = re.match(r"sft-(\d+)", name)
    return f"sft_{m.group(1)}" if m else name.replace(".parquet", "")


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

# ── Async API call ────────────────────────────────────────────

async def call_api_async(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    content: list,
    gen_config: dict,
    label: str,
) -> tuple[str | None, str | None]:
    """Returns (reasoning, response) or (None, None) on failure."""
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
                return getattr(msg, "reasoning", None), msg.content

            except Exception as e:
                err = str(e)
                is_rate = "429" in err or "rate" in err.lower()
                if is_rate and attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAYS[attempt]
                    print(f"\n  [RATE LIMIT] {label} — waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})...")
                    await asyncio.sleep(wait)
                else:
                    print(f"\n  [ERROR] {label}: {e}")
                    return None, None
    return None, None

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
) -> dict:
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

    reasoning, response = await call_api_async(client, semaphore, model, content, gen_config, label)

    result = dict(sample)
    result["reasoning"] = reasoning
    result["response"]  = response
    return result

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

# ── Main ──────────────────────────────────────────────────────

async def async_main(args):
    df = pd.read_parquet(args.parquet)
    total_rows = len(df)
    print(f"Loaded parquet: {total_rows} rows, columns: {df.columns.tolist()}")

    stem = parquet_stem(args.parquet)
    output_path = os.path.join(args.output_dir, f"{stem}.json")
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine index range
    start = args.start_index
    end   = args.end_index if args.end_index is not None else total_rows
    end   = min(end, total_rows)

    print(f"Processing rows [{start}, {end}) of {total_rows}")
    print(f"Output: {output_path}")
    print(f"Workers (concurrent): {args.workers}")

    # Load existing results for resume
    done_indices = set()
    results = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for r in existing:
            results[r["_index"]] = r
        done_indices = set(results.keys())
        if done_indices:
            print(f"[RESUME] Already done: {len(done_indices)} sample(s) — skipping")

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

    # --retry-index: single sample
    if args.retry_index is not None:
        if args.retry_index < 0 or args.retry_index >= total_rows:
            print(f"[ERROR] --retry-index {args.retry_index} out of range [0, {total_rows})")
            return
        print(f"[RETRY] Re-generating index={args.retry_index}")
        row = df.iloc[args.retry_index].to_dict()
        result = await process_sample_async(
            row, args.retry_index, client, semaphore, args.model, template, gen_config, loop
        )
        results[args.retry_index] = result
        await save_results(results, output_path, lock)
        status = "OK" if result.get("response") else "ERR"
        print(f"  → {status}  Saved → {output_path}")
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
        result = await process_sample_async(
            row, row_idx, client, semaphore, args.model, template, gen_config, loop
        )
        async with lock:
            results[row_idx] = result
            completed += 1
            status = "OK" if result.get("response") else "ERR"
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

    await asyncio.gather(*[process_one(i) for i in pending_indices])

    # Final save
    await save_results(results, output_path, lock)
    print(f"\nDone. {len(results)} total results saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Async reasoning extraction from parquet (with_think pipeline)"
    )
    parser.add_argument("--api-key",       required=True)
    parser.add_argument("--model",         required=True, help="OpenRouter model ID")
    parser.add_argument("--parquet",       required=True, help="Path to parquet file")
    parser.add_argument("--template-path", default=TEMPLATE_DEFAULT)
    parser.add_argument("--output-dir",    default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--workers",       type=int, default=10,
                        help="Number of concurrent API calls (default: 10)")
    parser.add_argument("--start-index",   type=int, default=0,
                        help="Start row index (inclusive, default: 0)")
    parser.add_argument("--end-index",     type=int, default=None,
                        help="End row index (exclusive, default: all)")
    parser.add_argument("--retry-index",   type=int, default=None,
                        help="Re-generate only this single row index")
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
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
