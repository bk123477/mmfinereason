import os
import json
from flask import Flask, render_template, jsonify, send_from_directory, abort

app = Flask(__name__)

BASE_DIR              = os.path.dirname(os.path.abspath(__file__))
META_DIR              = os.path.join(BASE_DIR, "data", "metadata")
REASON_DIR            = os.path.join(BASE_DIR, "data", "reasoning")
REASON_WITH_REF_DIR   = os.path.join(BASE_DIR, "data", "reasoning_with_ref")
REASON_NOTHINK_DIR    = os.path.join(BASE_DIR, "data", "reasoning_nothink")
REASON_THINK_DIR      = os.path.join(BASE_DIR, "data", "reasoning_think")
IMAGE_BASE            = os.path.join(BASE_DIR, "mmfinereason_images")
POST_FILTERING_DIR    = os.path.join(BASE_DIR, "post_filtering")

ALL_REASON_DIRS = [
    REASON_DIR,
    REASON_WITH_REF_DIR,
    REASON_NOTHINK_DIR,
    REASON_THINK_DIR,
]


# ── Helpers ────────────────────────────────────────────────

def to_image_rel(image_path: str) -> str:
    """Convert an absolute or relative image_path to a path relative to IMAGE_BASE.

    Works on any machine regardless of the absolute prefix stored in the JSON,
    by extracting the portion starting from 'mmfinereason_images/'.
    """
    if not image_path:
        return ""
    # Already relative
    if not os.path.isabs(image_path):
        return image_path
    # Extract the relative portion after the image root folder name
    marker = "mmfinereason_images" + os.sep
    idx = image_path.find(marker)
    if idx != -1:
        return image_path[idx + len(marker):]
    # Fallback: try relpath if file happens to exist locally
    if os.path.exists(image_path):
        try:
            return os.path.relpath(image_path, IMAGE_BASE)
        except ValueError:
            pass
    return ""

def list_runs():
    """Return sorted list of timestamped run-folder names, newest first.
    Scans all reasoning variant directories."""
    runs = []
    for base_dir in ALL_REASON_DIRS:
        if not os.path.exists(base_dir):
            continue
        runs += [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
            and d.startswith("reasoning_")
        ]
    return sorted(runs, reverse=True)  # newest first


def build_entry(r_item, meta_by_index):
    idx       = r_item.get("_index", 0)
    meta_item = meta_by_index.get(idx, {})

    image_abs = r_item.get("image_path") or meta_item.get("_image_path") or ""
    image_rel = to_image_rel(image_abs)

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
    """Load all subset entries from a specific run folder.
    Searches both data/reasoning/ and data/reasoning_with_ref/."""
    run_dir = None
    for base_dir in ALL_REASON_DIRS:
        candidate = os.path.join(base_dir, run_name)
        if os.path.isdir(candidate):
            run_dir = candidate
            break
    if run_dir is None:
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


# ── Post-filtering helpers ─────────────────────────────────

def list_postfilter_runs():
    """Return sorted list of post-filtering run folder names, newest first."""
    if not os.path.exists(POST_FILTERING_DIR):
        return []
    runs = [
        d for d in os.listdir(POST_FILTERING_DIR)
        if os.path.isdir(os.path.join(POST_FILTERING_DIR, d))
        and d.startswith("post_filtering_")
    ]
    return sorted(runs, reverse=True)


def load_postfilter_run(run_name):
    """Load all subset entries from a post-filtering run folder."""
    run_dir = os.path.join(POST_FILTERING_DIR, run_name)
    if not os.path.isdir(run_dir):
        return []

    subsets = []
    reason_files = sorted(
        f for f in os.listdir(run_dir)
        if f.endswith("_reasoning.json")
        and f not in ("reasoning_summary.json", "post_filter_summary.json")
    )

    for rf in reason_files:
        subset_name = rf.replace("_reasoning.json", "")
        file_path   = os.path.join(run_dir, rf)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data if isinstance(data, list) else [data]
        except Exception as e:
            print(f"[WARN] Cannot load {file_path}: {e}")
            continue

        processed = []
        for item in entries:
            image_abs = item.get("image_path", "") or ""
            image_rel = to_image_rel(image_abs)

            processed.append({
                "index":              item.get("_index", 0),
                "question":           item.get("question", ""),
                "options":            item.get("options"),
                "image_rel":          image_rel,
                "answer":             item.get("answer", ""),
                "source_run":         item.get("source_run", ""),
                "reasoning_original": item.get("reasoning_original", ""),
                "response_original":  item.get("response_original", ""),
                "reasoning_modified": bool(item.get("reasoning_modified", False)),
                "response_modified":  bool(item.get("response_modified", False)),
                "reasoning":          item.get("reasoning", ""),
                "response":           item.get("response", ""),
            })

        subsets.append({"name": subset_name, "entries": processed})

    return subsets


# ── Post-filtering routes ──────────────────────────────────

@app.route("/api/postfilter/runs")
def api_postfilter_runs():
    return jsonify(list_postfilter_runs())


@app.route("/api/postfilter/run/<run_name>/subsets")
def api_postfilter_subsets(run_name):
    subsets = load_postfilter_run(run_name)
    return jsonify([{"name": s["name"], "count": len(s["entries"])} for s in subsets])


@app.route("/api/postfilter/run/<run_name>/subset/<int:s_idx>/sample/<int:e_idx>")
def api_postfilter_entry(run_name, s_idx, e_idx):
    subsets = load_postfilter_run(run_name)
    if s_idx < 0 or s_idx >= len(subsets):
        abort(404)
    entries = subsets[s_idx]["entries"]
    if e_idx < 0 or e_idx >= len(entries):
        abort(404)
    return jsonify(entries[e_idx])


@app.route("/api/postfilter/run/<run_name>/stats")
def api_postfilter_stats(run_name):
    """Return per-subset and total char-length statistics for post-filtered reasoning & response."""
    subsets = load_postfilter_run(run_name)
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
        "min_reasoning": None, "max_reasoning": None,
        "min_response":  None, "max_response":  None,
    }
    return jsonify({"run": run_name, "subsets": rows, "total": total_row})


if __name__ == "__main__":
    print("MMFineReason Visualizer starting...")
    print("Open http://127.0.0.1:5002 in your browser.")
    app.run(debug=True, port=5002)
