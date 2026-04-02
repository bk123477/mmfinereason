import os
import json
from flask import Flask, render_template, jsonify, send_from_directory, abort

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
META_DIR   = os.path.join(BASE_DIR, "data", "metadata")
REASON_DIR = os.path.join(BASE_DIR, "data", "reasoning")
IMAGE_BASE = os.path.join(BASE_DIR, "mmfinereason_images")


# ── Helpers ────────────────────────────────────────────────

def list_runs():
    """Return sorted list of timestamped run-folder names under REASON_DIR."""
    if not os.path.exists(REASON_DIR):
        return []
    runs = sorted(
        d for d in os.listdir(REASON_DIR)
        if os.path.isdir(os.path.join(REASON_DIR, d))
        and d.startswith("reasoning_")
    )
    return runs  # newest last; reverse for display (newest first)


def build_entry(r_item, meta_by_index):
    idx       = r_item.get("_index", 0)
    meta_item = meta_by_index.get(idx, {})

    image_abs = r_item.get("image_path") or meta_item.get("_image_path") or ""
    image_rel = ""
    if image_abs and os.path.exists(image_abs):
        try:
            image_rel = os.path.relpath(image_abs, IMAGE_BASE)
        except ValueError:
            image_rel = ""

    return {
        "index":         idx,
        "question":      r_item.get("question", ""),
        "options":       r_item.get("options"),
        "image_rel":     image_rel,
        "qwen_thinking": meta_item.get("qwen3vl_235b_thinking_response", ""),
        "answer":        meta_item.get("answer", ""),
        "reasoning":     r_item.get("reasoning") or "",
        "response":      r_item.get("response") or "",
    }


def load_subsets_for_run(run_name):
    """Load all subset entries from a specific run folder."""
    run_dir = os.path.join(REASON_DIR, run_name)
    if not os.path.isdir(run_dir):
        return []

    subsets = []
    reason_files = sorted(
        f for f in os.listdir(run_dir)
        if f.endswith("_reasoning.json") and f != "reasoning_summary.json"
    )
    for rf in reason_files:
        subset_name = rf.replace("_reasoning.json", "")
        reason_path = os.path.join(run_dir, rf)
        meta_path   = os.path.join(META_DIR, f"{subset_name}_metadata.json")

        try:
            with open(reason_path, "r", encoding="utf-8") as f:
                r_data = json.load(f)
            r_list = r_data if isinstance(r_data, list) else [r_data]
        except Exception as e:
            print(f"[WARN] Cannot load {reason_path}: {e}")
            continue

        meta_by_index = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                m_list = m_data if isinstance(m_data, list) else [m_data]
                meta_by_index = {item.get("_index", i): item for i, item in enumerate(m_list)}
            except Exception as e:
                print(f"[WARN] Cannot load {meta_path}: {e}")

        entries = [build_entry(r, meta_by_index) for r in r_list]
        subsets.append({"name": subset_name, "entries": entries})

    return subsets


# ── Routes ─────────────────────────────────────────────────

@app.route("/")
def index():
    runs = list(reversed(list_runs()))   # newest first
    return render_template("index.html", runs=runs)


@app.route("/api/runs")
def api_runs():
    runs = list(reversed(list_runs()))
    return jsonify(runs)


@app.route("/api/run/<run_name>/subsets")
def api_subsets(run_name):
    subsets = load_subsets_for_run(run_name)
    return jsonify([{"name": s["name"], "count": len(s["entries"])} for s in subsets])


@app.route("/api/run/<run_name>/subset/<int:s_idx>/sample/<int:e_idx>")
def api_entry(run_name, s_idx, e_idx):
    subsets = load_subsets_for_run(run_name)
    if s_idx < 0 or s_idx >= len(subsets):
        abort(404)
    entries = subsets[s_idx]["entries"]
    if e_idx < 0 or e_idx >= len(entries):
        abort(404)
    return jsonify(entries[e_idx])


@app.route("/api/run/<run_name>/stats")
def api_stats(run_name):
    """Return per-subset and total char-length statistics for reasoning & response."""
    subsets = load_subsets_for_run(run_name)
    rows = []
    total_r, total_p, total_n = 0, 0, 0

    for s in subsets:
        entries = s["entries"]
        n = len(entries)
        if n == 0:
            rows.append({
                "subset": s["name"], "count": 0,
                "avg_reasoning": None, "avg_response": None,
                "min_reasoning": None, "max_reasoning": None,
                "min_response":  None, "max_response":  None,
            })
            continue

        r_lens = [len(e.get("reasoning") or "") for e in entries]
        p_lens = [len(e.get("response")  or "") for e in entries]

        avg_r = round(sum(r_lens) / n)
        avg_p = round(sum(p_lens) / n)
        total_r += sum(r_lens)
        total_p += sum(p_lens)
        total_n += n

        rows.append({
            "subset":        s["name"],
            "count":         n,
            "avg_reasoning": avg_r,
            "avg_response":  avg_p,
            "min_reasoning": min(r_lens),
            "max_reasoning": max(r_lens),
            "min_response":  min(p_lens),
            "max_response":  max(p_lens),
        })

    total_row = {
        "subset":        "TOTAL",
        "count":         total_n,
        "avg_reasoning": round(total_r / total_n) if total_n else None,
        "avg_response":  round(total_p / total_n) if total_n else None,
        "min_reasoning": None,
        "max_reasoning": None,
        "min_response":  None,
        "max_response":  None,
    }
    return jsonify({"run": run_name, "subsets": rows, "total": total_row})


@app.route("/images/<path:filepath>")
def serve_image(filepath):
    directory = os.path.dirname(os.path.join(IMAGE_BASE, filepath))
    filename  = os.path.basename(filepath)
    return send_from_directory(directory, filename)


if __name__ == "__main__":
    print("MMFineReason Visualizer starting...")
    print("Open http://127.0.0.1:5002 in your browser.")
    app.run(debug=True, port=5002)
