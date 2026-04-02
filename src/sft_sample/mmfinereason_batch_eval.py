"""
Batch evaluation script for MMFineReason reasoning outputs.
Reads *_reasoning.json files and computes basic accuracy metrics
by comparing the model's final answer against the ground truth.
"""
import os
import json
import argparse
import re


def get_script_root():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "../../"))


def extract_final_answer(response_text):
    """Extract the content after 'Phase 7: Final Answer' or the last non-empty line."""
    if not response_text:
        return ""
    # Look for Phase 7 header
    match = re.search(r"Phase 7[:\s]+Final Answer[^\n]*\n(.+)", response_text, re.IGNORECASE | re.DOTALL)
    if match:
        answer_block = match.group(1).strip()
        # Return first non-empty line
        for line in answer_block.splitlines():
            line = line.strip().lstrip("-").strip()
            if line:
                return line
    # Fallback: last non-empty line
    lines = [l.strip() for l in response_text.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def normalize(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def evaluate_subset(results, subset_name):
    total = len(results)
    correct = 0
    skipped = 0

    for item in results:
        gt = item.get("answer") or item.get("gt_answer") or item.get("ground_truth")
        if gt is None:
            skipped += 1
            continue
        predicted = extract_final_answer(item.get("response", ""))
        if normalize(predicted) == normalize(gt):
            correct += 1

    evaluable = total - skipped
    accuracy = correct / evaluable if evaluable > 0 else None

    return {
        "subset": subset_name,
        "total": total,
        "evaluable": evaluable,
        "skipped_no_gt": skipped,
        "correct": correct,
        "accuracy": round(accuracy * 100, 2) if accuracy is not None else "N/A",
    }


def main():
    dataset_root = get_script_root()

    parser = argparse.ArgumentParser(
        description="Batch evaluation of MMFineReason reasoning outputs"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(dataset_root, "data", "reasoning"),
        help="Directory containing *_reasoning.json files (default: data/reasoning/)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(dataset_root, "data", "reasoning", "eval_results.json"),
        help="Path to save evaluation results JSON (default: data/reasoning/eval_results.json)",
    )
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=None,
        help="Specific subsets to evaluate (default: all *_reasoning.json in --data-dir)",
    )
    args = parser.parse_args()

    if args.subsets:
        reasoning_files = [
            os.path.join(args.data_dir, f"{s}_reasoning.json") for s in args.subsets
        ]
    else:
        reasoning_files = [
            os.path.join(args.data_dir, f)
            for f in os.listdir(args.data_dir)
            if f.endswith("_reasoning.json")
        ]

    if not reasoning_files:
        print("No reasoning files found. Run the reasoning script first.")
        return

    all_eval = []
    for rfile in sorted(reasoning_files):
        if not os.path.exists(rfile):
            print(f"[SKIP] Not found: {rfile}")
            continue
        with open(rfile, "r", encoding="utf-8") as f:
            results = json.load(f)
        subset_name = os.path.basename(rfile).replace("_reasoning.json", "")
        eval_result = evaluate_subset(results, subset_name)
        all_eval.append(eval_result)
        acc = eval_result["accuracy"]
        print(
            f"  {subset_name}: {eval_result['correct']}/{eval_result['evaluable']} correct"
            f" (acc={acc}%)"
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_eval, f, ensure_ascii=False, indent=2)
    print(f"\nEvaluation results saved to {args.output}")


if __name__ == "__main__":
    main()
