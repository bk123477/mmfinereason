"""
judge.py  —  LLM-as-the-Judge: reconstructed vs post-filtered

Compares reasoning quality between two sources for the same questions:
  A:  post_filtering/reconstruct_*/     (sft_sample/reconstruct.py output)
  B:  post_filtering/post_filtering_*/  (sft_sample/post_filter.py output)

Matches pairs by _subset + _index, samples N items, judges each with an LLM.

Usage:
  python src/judge.py --model deepseek/deepseek-v3.2
  python src/judge.py --model deepseek/deepseek-v3.2 --n 20 --seed 42
  python src/judge.py --model deepseek/deepseek-v3.2 \\
      --dir-a post_filtering/reconstruct_20260408_191247 \\
      --dir-b post_filtering/post_filtering_20260408_093702
"""

import os
import re
import json
import random
import asyncio
import argparse
from datetime import datetime

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

# ── Paths ─────────────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DIR_A_DEFAULT = os.path.join(BASE_DIR, "post_filtering", "reconstruct_20260408_191247")
DIR_B_DEFAULT = os.path.join(BASE_DIR, "post_filtering", "post_filtering_20260408_093702")
OUT_DIR       = os.path.join(BASE_DIR, "src")

MAX_RETRIES  = 3
RETRY_DELAYS = [10, 30, 60]

# ── Judge prompt ──────────────────────────────────────────────

JUDGE_SYSTEM = """You are an expert evaluator of mathematical and scientific reasoning chains.
You will be given a question and two reasoning traces (A and B) that both attempt to solve it.
Evaluate each on three dimensions, then declare a winner.

Scoring rubric (1–5):
  logical_soundness   — Are the reasoning steps logically valid and free of errors?
  completeness        — Are all necessary steps shown without skipping key details?
  clarity_structure   — Is the reasoning well-organized and easy to follow?

Output ONLY the following XML block, with no additional text:
<SCORES>
A_logical_soundness: <1-5>
A_completeness: <1-5>
A_clarity_structure: <1-5>
B_logical_soundness: <1-5>
B_completeness: <1-5>
B_clarity_structure: <1-5>
</SCORES>
<WINNER>A|B|TIE</WINNER>
<EXPLANATION>One or two sentences justifying the winner.</EXPLANATION>"""


def build_judge_message(question: str, answer: str, reasoning_a: str, reasoning_b: str) -> str:
    def truncate(text: str, limit: int = 3000) -> str:
        return text[:limit] + "\n…[truncated]" if len(text) > limit else text

    return (
        f"### Question\n{question}\n\n"
        f"### Ground Truth Answer\n{answer or '(not provided)'}\n\n"
        f"---\n\n"
        f"### Reasoning A\n{truncate(reasoning_a) or '(empty)'}\n\n"
        f"---\n\n"
        f"### Reasoning B\n{truncate(reasoning_b) or '(empty)'}"
    )

# ── Data loading ──────────────────────────────────────────────

SKIP_FILES = {"reconstruct_summary.json", "post_filter_summary.json",
              "reasoning_summary.json", "download_summary.json"}


def load_dir(path: str) -> list[dict]:
    """Load all sample JSON files from a directory into a flat list.

    Handles two layouts:
      - Single file:   metadata_reconstruct.json
      - Multi-file:    BMMR_reasoning.json, MMK12_reasoning.json, …
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")

    items = []
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".json") or fname in SKIP_FILES:
            continue
        fpath = os.path.join(path, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                items.extend(data)
        except Exception as e:
            print(f"[WARN] Cannot load {fpath}: {e}")
    return items


def _pair_key(item: dict) -> str:
    """Match key: _subset + _index (works across both data sources)."""
    return f"{item.get('_subset', 'unknown')}_{item.get('_index', 0)}"


def build_index(items: list[dict]) -> dict[str, dict]:
    return {_pair_key(item): item for item in items}


def sample_matched_pairs(
    index_a: dict[str, dict],
    index_b: dict[str, dict],
    n: int,
    seed: int,
) -> list[tuple[dict, dict]]:
    """Return N randomly sampled (item_a, item_b) pairs matched by _subset+_index."""
    common = sorted(set(index_a) & set(index_b))
    if len(common) < n:
        print(f"[WARN] Only {len(common)} matched pairs available, sampling all.")
        n = len(common)
    rng = random.Random(seed)
    chosen = rng.sample(common, n)
    return [(index_a[k], index_b[k]) for k in sorted(chosen)]

# ── Parse judge output ────────────────────────────────────────

def parse_judge_output(raw: str) -> dict | None:
    scores_m = re.search(r"<SCORES>(.*?)</SCORES>", raw, re.DOTALL)
    winner_m = re.search(r"<WINNER>(.*?)</WINNER>",  raw, re.DOTALL)
    expl_m   = re.search(r"<EXPLANATION>(.*?)</EXPLANATION>", raw, re.DOTALL)

    if not scores_m or not winner_m:
        return None

    scores = {}
    for line in scores_m.group(1).strip().splitlines():
        line = line.strip()
        if ":" in line:
            key, val = line.split(":", 1)
            try:
                scores[key.strip()] = int(val.strip())
            except ValueError:
                pass

    return {
        "scores":      scores,
        "winner":      winner_m.group(1).strip(),
        "explanation": expl_m.group(1).strip() if expl_m else "",
    }

# ── Async API call ────────────────────────────────────────────

async def call_judge(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    user_msg: str,
    label: str,
) -> str | None:
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user",   "content": user_msg},
                    ],
                    max_tokens=512,
                    extra_body={"include_reasoning": False},
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                err = str(e)
                is_rate = "429" in err or "rate" in err.lower()
                if is_rate and attempt < MAX_RETRIES - 1:
                    wait = RETRY_DELAYS[attempt]
                    print(f"\n  [RATE LIMIT] {label} — waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"\n  [API ERROR] {label}: {e}")
                    return None
    return None

# ── Main ──────────────────────────────────────────────────────

async def async_main(args):
    if not args.api_key:
        print("[ERROR] No API key. Set OPENROUTER_API_KEY in .env or pass --api-key.")
        return

    print(f"Loading A: {args.dir_a}")
    print(f"Loading B: {args.dir_b}")
    items_a = load_dir(args.dir_a)
    items_b = load_dir(args.dir_b)
    print(f"  A: {len(items_a)} items  |  B: {len(items_b)} items")

    index_a = build_index(items_a)
    index_b = build_index(items_b)
    pairs   = sample_matched_pairs(index_a, index_b, args.n, args.seed)
    print(f"  Sampled {len(pairs)} matched pairs (seed={args.seed})\n")

    client    = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.workers)

    results   = []
    completed = 0
    lock      = asyncio.Lock()

    async def judge_one(item_a: dict, item_b: dict):
        nonlocal completed
        idx   = item_a["_index"]
        label = f"idx={idx} ({item_a.get('_subset','?')})"

        user_msg = build_judge_message(
            question   = item_a.get("question_clean") or item_a.get("question", ""),
            answer     = str(item_a.get("answer", "") or ""),
            reasoning_a= item_a.get("reasoning") or "",
            reasoning_b= item_b.get("reasoning") or "",
        )
        raw    = await call_judge(client, semaphore, args.model, user_msg, label)
        parsed = parse_judge_output(raw) if raw else None

        async with lock:
            completed += 1
            if parsed:
                winner = parsed["winner"]
                scores = parsed["scores"]
                avg_a  = round(sum(v for k,v in scores.items() if k.startswith("A_")) / 3, 2)
                avg_b  = round(sum(v for k,v in scores.items() if k.startswith("B_")) / 3, 2)
                print(f"  [{completed:>3}/{len(pairs)}] {label:30s}  "
                      f"A={avg_a:.1f}  B={avg_b:.1f}  winner={winner}")
            else:
                print(f"  [{completed:>3}/{len(pairs)}] {label:30s}  → PARSE FAILED")

            results.append({
                "_index":      item_a.get("_index", 0),
                "_subset":     item_a.get("_subset", ""),
                "rc_workflow": item_a.get("rc_workflow", ""),   # from reconstruct
                "pf_workflow": item_b.get("pf_workflow", ""),   # from post_filter
                "question":    item_a.get("question_clean") or item_a.get("question", "")[:120],
                "answer":      str(item_a.get("answer", "")),
                "judge":       parsed,
                "raw":         raw,
            })

    await asyncio.gather(*[judge_one(a, b) for a, b in pairs])

    # ── Aggregate stats ───────────────────────────────────────
    valid    = [r for r in results if r["judge"]]
    n_a      = sum(1 for r in valid if r["judge"]["winner"] == "A")
    n_b      = sum(1 for r in valid if r["judge"]["winner"] == "B")
    n_tie    = sum(1 for r in valid if r["judge"]["winner"] == "TIE")

    def avg_score(key):
        vals = [r["judge"]["scores"].get(key, 0) for r in valid if r["judge"]["scores"].get(key)]
        return round(sum(vals) / len(vals), 2) if vals else None

    stats = {
        "n_judged":     len(valid),
        "n_failed":     len(results) - len(valid),
        "winner_A":     n_a,
        "winner_B":     n_b,
        "tie":          n_tie,
        "avg_scores": {
            "A_logical_soundness":  avg_score("A_logical_soundness"),
            "A_completeness":       avg_score("A_completeness"),
            "A_clarity_structure":  avg_score("A_clarity_structure"),
            "B_logical_soundness":  avg_score("B_logical_soundness"),
            "B_completeness":       avg_score("B_completeness"),
            "B_clarity_structure":  avg_score("B_clarity_structure"),
        },
    }

    # ── Output ────────────────────────────────────────────────
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUT_DIR, f"judge_results_{ts}.json")
    output = {
        "meta": {
            "model":     args.model,
            "dir_a":     args.dir_a,
            "dir_b":     args.dir_b,
            "label_a":   "reconstructed (sft_sample/reconstruct.py)",
            "label_b":   "post-filtered (sft_sample/post_filter.py)",
            "n_samples": len(pairs),
            "seed":      args.seed,
            "timestamp": ts,
        },
        "stats":   stats,
        "samples": sorted(results, key=lambda r: r["_index"]),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── Print summary ─────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Judged: {len(valid)}/{len(pairs)}  |  Failed: {len(results)-len(valid)}")
    print(f"  Winner A (reconstructed):   {n_a:>3}  ({n_a/len(valid)*100:.0f}%)" if valid else "")
    print(f"  Winner B (post-filtered):   {n_b:>3}  ({n_b/len(valid)*100:.0f}%)" if valid else "")
    print(f"  Tie:                        {n_tie:>3}  ({n_tie/len(valid)*100:.0f}%)" if valid else "")
    print(f"\n  Avg scores (A = reconstructed | B = post-filtered):")
    s = stats["avg_scores"]
    print(f"    logical_soundness:  A={s['A_logical_soundness']}  B={s['B_logical_soundness']}")
    print(f"    completeness:       A={s['A_completeness']}  B={s['B_completeness']}")
    print(f"    clarity_structure:  A={s['A_clarity_structure']}  B={s['B_clarity_structure']}")
    print(f"\n  Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-the-Judge: reasoning vs reconstructed"
    )
    parser.add_argument("--api-key",  default=os.environ.get("OPENROUTER_API_KEY"),
                        help="OpenRouter API key (default: $OPENROUTER_API_KEY)")
    parser.add_argument("--model",    required=True, help="OpenRouter model ID for judge")
    parser.add_argument("--dir-a",    default=DIR_A_DEFAULT,
                        help="Directory A — reconstruct output (default: post_filtering/reconstruct_20260408_191247)")
    parser.add_argument("--dir-b",    default=DIR_B_DEFAULT,
                        help="Directory B — post-filter output (default: post_filtering/post_filtering_20260408_093702)")
    parser.add_argument("--n",        type=int, default=220,
                        help="Number of pairs to sample (default: 220 = all)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--workers",  type=int, default=5,
                        help="Concurrent judge calls (default: 5)")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
