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
  --model qwen/qwen3-235b-a22b \
  --samples 10
```

---

#### Arguments

**필수 인자**

| 인자 | 설명 |
|------|------|
| `--api-key KEY` | OpenRouter API Key |
| `--model MODEL` | 사용할 모델 ID (예: `qwen/qwen3-235b-a22b`) |

**실행 범위 제어**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--samples N` | 전체 | subset당 처리할 샘플 수 |
| `--subsets A B C` | 전체 | 처리할 subset 이름 (공백으로 여러 개 지정 가능) |
| `--data-dir PATH` | `data/metadata/` | `*_metadata.json` 파일들이 있는 디렉토리 |

**출력 경로**

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--output-dir PATH` | `data/reasoning/reasoning_YYYYMMDD_HHMMSS/` | 결과 저장 폴더 (생략 시 타임스탬프 자동 생성) |
| `--template-path PATH` | `prompts/reasoning_distillation.md` | 사용할 reasoning 프롬프트 파일 경로 |

**재개 / 재생성**

| 인자 | 설명 |
|------|------|
| `--resume` | 이미 처리된 샘플 스킵 (기존 `--output-dir` 지정 시 자동 적용됨) |
| `--retry-index N` | 특정 `_index`의 샘플 하나만 재생성 (`--subsets`, `--output-dir`과 함께 사용) |

**Generation Config** (Qwen3 권장값이 기본값으로 설정됨)

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--temperature` | `1.0` | 샘플링 온도 |
| `--top-p` | `0.95` | Nucleus sampling |
| `--top-k` | `20` | Top-k sampling (OpenRouter extra_body) |
| `--min-p` | `0.0` | Min-p sampling (OpenRouter extra_body) |
| `--presence-penalty` | `1.5` | 반복 억제 (OpenAI 호환 파라미터) |
| `--repetition-penalty` | `1.0` | 반복 패널티 (OpenRouter extra_body) |
| `--n` | `1` | 샘플당 생성 개수 |
| `--max-tokens` | `81920` | 최대 출력 토큰 수 |

---

#### 사용 예시

**전체 subset 처리 (10샘플씩)**
```bash
python src/sft_sample/mmfinereason_sft_reasoning.py \
  --api-key $OPENROUTER_API_KEY \
  --model qwen/qwen3-235b-a22b \
  --samples 10
```

**특정 subset만 처리**
```bash
python src/sft_sample/mmfinereason_sft_reasoning.py \
  --api-key $OPENROUTER_API_KEY \
  --model qwen/qwen3-235b-a22b \
  --subsets BMMR Zebra-CoT-Physics
```

**중단된 run 이어서 실행 (Ctrl+C 후 재개)**
```bash
python src/sft_sample/mmfinereason_sft_reasoning.py \
  --api-key $OPENROUTER_API_KEY \
  --model qwen/qwen3-235b-a22b \
  --output-dir data/reasoning/reasoning_20260401_212153
# 기존 폴더 지정 시 완료된 샘플은 자동 스킵 (--resume 생략 가능)
```

**오류난 샘플 하나만 콕 집어 재생성**
```bash
python src/sft_sample/mmfinereason_sft_reasoning.py \
  --api-key $OPENROUTER_API_KEY \
  --model qwen/qwen3-235b-a22b \
  --output-dir data/reasoning/reasoning_20260401_212153 \
  --subsets BMMR \
  --retry-index 3
# _index=3인 샘플만 재생성 후 JSON 내 _index 순서 유지하여 저장
```

**다른 모델로 새 실험 run 생성 (웹 뷰어에서 비교 가능)**
```bash
python src/sft_sample/mmfinereason_sft_reasoning.py \
  --api-key $OPENROUTER_API_KEY \
  --model qwen/qwen3-30b-a3b \
  --samples 10
# 새 타임스탬프 폴더에 자동 저장 → 웹 뷰어 Compare 모드로 비교 가능
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
