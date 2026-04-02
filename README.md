# MMFineReason — Reasoning Distillation Pipeline

Reasoning distillation pipeline for the [MMFineReason](https://huggingface.co/datasets/OpenDataArena/MMFineReason-Full-2.3M-Qwen3-VL-235B-Thinking) dataset.
Uses OpenRouter API (Qwen3-235B-Thinking) to generate structured 7-phase reasoning traces, with a Flask-based web viewer to inspect and compare results.

---

## Project Structure

```
mmfinereason/
├── app.py                          # Flask web viewer
├── templates/index.html            # Viewer UI (normal + compare mode)
├── prompts/
│   └── reasoning_distillation.md  # 7-phase reasoning prompt
├── src/sft_sample/
│   ├── mmfinereason_downloader.py  # Download samples from HuggingFace
│   ├── mmfinereason_sft_reasoning.py  # Generate reasoning via OpenRouter
│   └── mmfinereason_batch_eval.py  # Evaluate generated reasoning
├── data/
│   ├── metadata/                   # Downloaded sample metadata (per subset)
│   └── reasoning/
│       └── reasoning_YYYYMMDD_HHMMSS/  # Timestamped experiment runs
├── mmfinereason_images/            # Downloaded sample images (per subset)
├── requirements.txt
└── .gitignore
```

---

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/bk123477/mmfinereason.git
cd mmfinereason
```

### 2. Create and activate a conda environment

```bash
conda create -n mmfinereason python=3.11 -y
conda activate mmfinereason
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your OpenRouter API key

Create a `.env` file in the project root (excluded from git):

```bash
echo "OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx" > .env
```

Or export it directly in your shell:

```bash
export OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxxxxx"
```

> Get your key at [openrouter.ai/keys](https://openrouter.ai/keys)

---

## Usage

### Step 1 — Download samples from HuggingFace

Downloads 10 samples per subset from MMFineReason (22 subsets).
Saves images to `mmfinereason_images/` and metadata to `data/metadata/`.

```bash
python src/sft_sample/mmfinereason_downloader.py
```

Options:
```
--samples N      Samples per subset (default: 10)
```

---

### Step 2 — Generate reasoning traces

Calls OpenRouter API (Qwen3-235B-Thinking) and saves structured reasoning to a timestamped folder under `data/reasoning/`.

```bash
python src/sft_sample/mmfinereason_sft_reasoning.py \
  --api-key $OPENROUTER_API_KEY \
  --samples 10
```

Key options:
```
--api-key KEY          OpenRouter API key (or set OPENROUTER_API_KEY env var)
--samples N            Samples per subset to process (default: 10)
--subsets A B C        Process specific subsets only
--output-dir PATH      Custom output folder (auto-generates timestamped dir if omitted)
--resume               Resume from a previous run (skip already-processed samples)
--template PATH        Path to prompt template (default: prompts/reasoning_distillation.md)
```

Generation config (Qwen3.5 recommended):
```
--temperature 1.0
--top-p 0.95
--top-k 20
--min-p 0.0
--presence-penalty 1.5
--max-tokens 81920
```

**Resume after interruption (Ctrl+C):**
```bash
python src/sft_sample/mmfinereason_sft_reasoning.py \
  --api-key $OPENROUTER_API_KEY \
  --output-dir data/reasoning/reasoning_20260401_212153 \
  --resume
```

**Run a single subset:**
```bash
python src/sft_sample/mmfinereason_sft_reasoning.py \
  --api-key $OPENROUTER_API_KEY \
  --subsets BMMR
```

---

### Step 3 — Evaluate results (optional)

Extracts Phase 7 Final Answers and compares against ground truth.

```bash
python src/sft_sample/mmfinereason_batch_eval.py \
  --run-dir data/reasoning/reasoning_20260401_212153
```

---

### Step 4 — Launch the web viewer

```bash
python app.py
```

Open [http://127.0.0.1:5002](http://127.0.0.1:5002) in your browser.

**Features:**
- Browse all experiment runs via the Run dropdown (newest first)
- Navigate subsets with the Subset dropdown or `↑` / `↓` keys
- Navigate samples within a subset with `←` / `→` keys or ◀ ▶ buttons
- **Normal mode**: Left = Qwen3-VL-235B reference + ground truth, Right = generated reasoning + response
- **Compare mode** (`⇄ Compare`): Select two runs side-by-side to compare reasoning quality
- **📊 Stats**: View per-subset and total average character lengths for `reasoning` and `response`
- JSON files are reloaded on every request — no server restart needed after new data is generated

**Restart the server (if port is busy):**
```bash
lsof -ti:5002 | xargs kill -9 2>/dev/null; python app.py
```

---

## Data Flow

```
HuggingFace Dataset
       │
       ▼
mmfinereason_downloader.py
       │  saves images + metadata
       ▼
mmfinereason_images/<subset>/      ← sample images
data/metadata/<subset>_metadata.json
       │
       ▼
mmfinereason_sft_reasoning.py
       │  calls OpenRouter API
       ▼
data/reasoning/reasoning_YYYYMMDD_HHMMSS/<subset>_reasoning.json
       │
       ├──▶ mmfinereason_batch_eval.py  (accuracy evaluation)
       └──▶ app.py  (web viewer)
```

---

## Output JSON Format

Each `<subset>_reasoning.json` is a list of entries:

```json
{
  "_index": 0,
  "question": "...",
  "options": ["A", "B", "C", "D"],
  "image_path": "mmfinereason_images/BMMR/BMMR_0000.jpg",
  "reasoning": "<internal chain-of-thought from Qwen3 thinking tokens>",
  "response": "<final answer only>",
  "model": "qwen/qwen3-235b-a22b",
  "subset": "BMMR"
}
```

- `reasoning`: Model's internal `<think>` tokens (captured via `include_reasoning: true`)
- `response`: Final answer output (controlled by the prompt — should be concise)

---

## Notes

- API calls use `include_reasoning: true` via OpenRouter's extra body, which separates Qwen3.5's internal thinking from the response
- Results are saved **after every sample** so Ctrl+C never loses more than one sample's work
- Experiment runs are versioned by timestamp — old runs are never overwritten
