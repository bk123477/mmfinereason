"""
mmfinereason_with_think.py

Identical to mmfinereason_sft_reasoning_ver2.py, with one difference:
  Only the content INSIDE <think>...</think> from qwen3vl_235b_thinking_response
  is used as input context. The text outside the tags is discarded.

Input context:
  question + image + qwen_thinking (think block only) + answer → new reasoning + new response

Output saved under: data/reasoning_think/reasoning_think_YYYYMMDD_HHMMSS/
"""

import os
import re
import json
import argparse
import base64
from datetime import datetime
from io import BytesIO
from PIL import Image
from openai import OpenAI

TEMPLATE_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../prompts/reasoning_distillation.md"
)


def get_script_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "../../"))


def load_prompt_template(path):
    if not os.path.exists(path):
        print(f"Warning: Template not found at {path}. Using empty string.")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_think_block(text):
    """Return only the content INSIDE <think>...</think>, discarding everything else."""
    if not text:
        return text
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def encode_image_path(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_user_content(template, question, options, base64_image,
                       qwen_thinking, answer):
    options_str = ""
    if options:
        if isinstance(options, list):
            labels = list("ABCDEFGHIJ")
            options_str = "\n### Options:\n" + "\n".join(
                f"({labels[i]}) {opt}" for i, opt in enumerate(options) if i < len(labels)
            )
        else:
            options_str = f"\n### Options:\n{options}"

    # Order: image → question+options → reasoning (think stripped) → response → template → question
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


def process_sample(client, model, template, sample, image_dir, gen_config):
    idx        = sample.get("_index", 0)
    subset     = sample.get("_subset", "unknown")
    image_path = sample.get("_image_path")

    question = sample.get("question", "") or sample.get("conversations", "")
    if isinstance(question, list):
        for turn in question:
            if isinstance(turn, dict) and turn.get("from") == "human":
                question = turn.get("value", "")
                break
        else:
            question = str(question)
    question = str(question).replace("<image>", "").strip()

    options = sample.get("options", None)
    answer  = sample.get("answer", "") or ""

    # Use only the content inside <think>...</think>
    raw_thinking  = sample.get("qwen3vl_235b_thinking_response", "") or ""
    qwen_thinking = extract_think_block(raw_thinking)

    base64_image = None
    if image_path and os.path.exists(image_path):
        try:
            base64_image = encode_image_path(image_path)
        except Exception as e:
            print(f"  Warning: Could not encode image {image_path}: {e}")

    content = build_user_content(
        template, question, options, base64_image,
        qwen_thinking, answer
    )

    try:
        response = client.chat.completions.create(
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
        msg         = response.choices[0].message
        output_text = msg.content
        reasoning   = getattr(msg, "reasoning", None)
    except Exception as e:
        print(f"  [ERROR] Sample {subset}_{idx}: {e}")
        output_text = str(e)
        reasoning   = None

    return {
        "_subset":    subset,
        "_index":     idx,
        "question":   question,
        "options":    options,
        "image_path": image_path,
        "reasoning":  reasoning,
        "response":   output_text,
    }


def main():
    dataset_root = get_script_root()

    parser = argparse.ArgumentParser(
        description="MMFineReason Reasoning (no-think) — strips <think> block from reference reasoning before use"
    )
    parser.add_argument("--api-key",       required=True, help="OpenRouter API Key")
    parser.add_argument("--model",         required=True, help="OpenRouter model ID")
    parser.add_argument("--template-path", default=TEMPLATE_DEFAULT,
                        help="Path to reasoning_distillation.md")
    parser.add_argument(
        "--data-dir",
        default=os.path.join(dataset_root, "data", "metadata"),
        help="Directory containing *_metadata.json files (default: data/metadata/)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: data/reasoning_think/reasoning_think_YYYYMMDD_HHMMSS/)",
    )
    parser.add_argument(
        "--subsets", nargs="*", default=None,
        help="Specific subsets to process (default: all found in --data-dir)",
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Number of samples per subset (default: all available)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-processed samples (auto-applied when output-dir already exists)",
    )
    parser.add_argument(
        "--retry-index", type=int, default=None,
        help="Re-generate only the sample with this _index (requires --subsets and --output-dir)",
    )
    # Generation config (Qwen3 recommended defaults)
    parser.add_argument("--temperature",        type=float, default=1.0)
    parser.add_argument("--top-p",              type=float, default=0.95)
    parser.add_argument("--top-k",              type=int,   default=20)
    parser.add_argument("--min-p",              type=float, default=0.0)
    parser.add_argument("--presence-penalty",   type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--n",                  type=int,   default=1)
    parser.add_argument("--max-tokens",         type=int,   default=81920)
    args = parser.parse_args()

    # Output dir
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(
            dataset_root, "data", "reasoning_think", f"reasoning_think_{timestamp}"
        )
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

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
    print(f"Generation config: {gen_config}")

    template = load_prompt_template(args.template_path)
    client   = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=args.api_key)

    # Discover metadata files
    if args.subsets:
        meta_files = [
            os.path.join(args.data_dir, f"{s}_metadata.json") for s in args.subsets
        ]
    else:
        meta_files = [
            os.path.join(args.data_dir, f)
            for f in os.listdir(args.data_dir)
            if f.endswith("_metadata.json") and f != "download_summary.json"
        ]

    if not meta_files:
        print("No metadata files found. Run the downloader first.")
        return

    print(f"Found {len(meta_files)} subset(s) to process.")
    overall_results = {}

    for meta_file in sorted(meta_files):
        if not os.path.exists(meta_file):
            print(f"[SKIP] Metadata file not found: {meta_file}")
            continue

        with open(meta_file, "r", encoding="utf-8") as f:
            samples = json.load(f)

        if not samples:
            print(f"[SKIP] Empty metadata: {meta_file}")
            continue

        subset_name = samples[0].get("_subset", os.path.basename(meta_file).replace("_metadata.json", ""))
        print(f"\n{'='*60}")
        print(f"Processing subset: {subset_name} ({len(samples)} samples)")
        print(f"{'='*60}")

        if args.samples is not None:
            samples = samples[: args.samples]

        image_dir   = os.path.join(dataset_root, "mmfinereason_images", subset_name)
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

        # --retry-index: single sample
        if args.retry_index is not None:
            target = next((s for s in samples if s.get("_index") == args.retry_index), None)
            if target is None:
                print(f"  [ERROR] _index {args.retry_index} not found in metadata.")
                continue
            print(f"  [RETRY] Re-generating _index={args.retry_index} ...")
            result = process_sample(client, args.model, template, target, image_dir, gen_config)
            status = "OK" if not str(result["response"]).startswith("Error") else "ERR"
            print(f"  → {status}")
            subset_results = [r for r in subset_results if r.get("_index") != args.retry_index]
            subset_results.append(result)
            subset_results.sort(key=lambda r: r.get("_index", 0))
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(subset_results, f, ensure_ascii=False, indent=2)
            print(f"  Saved → {output_path}")
            overall_results[subset_name] = len(subset_results)
            continue

        # Normal loop
        pending = [s for s in samples if s.get("_index") not in done_indices]
        if not pending:
            print(f"  [SKIP] All samples already processed.")
            overall_results[subset_name] = len(subset_results)
            continue

        for i, sample in enumerate(pending):
            print(f"  [{i+1}/{len(pending)}] {subset_name}_{sample.get('_index', i)}", end=" ", flush=True)
            result = process_sample(client, args.model, template, sample, image_dir, gen_config)
            subset_results.append(result)
            subset_results.sort(key=lambda r: r.get("_index", 0))
            status = "OK" if not str(result["response"]).startswith("Error") else "ERR"
            print(f"→ {status}")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(subset_results, f, ensure_ascii=False, indent=2)

        print(f"  Saved → {output_path}")
        overall_results[subset_name] = len(subset_results)

    summary_path = os.path.join(args.output_dir, "reasoning_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, ensure_ascii=False, indent=2)

    print(f"\nAll done. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
