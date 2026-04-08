# MMFineReason — Reasoning Distillation Pipeline

Reasoning distillation + post-filtering pipeline for the [MMFineReason-SFT-586K](https://huggingface.co/datasets/OpenDataArena/MMFineReason-SFT-586K-Qwen3-VL-235B-Thinking) dataset.  
Generates structured reasoning traces from Qwen3-VL-235B thinking blocks via OpenRouter API, then applies iterative quality filtering (C1~C4).

---

## Project Structure

```
mmfinereason/
├── src/
│   ├── sft_run/                        # Full dataset pipeline (586K)
│   │   ├── sft_reason.py               # Reasoning generation (parquet → JSON)
│   │   ├── sft_postfilter.py           # Quality detection & rewrite
│   │   └── README.md                   # Detailed usage for sft_run
│   └── sft_sample/                     # Pilot study scripts (small-scale)
│       ├── mmfinereason_downloader.py
│       ├── mmfinereason_sft_reasoning.py
│       ├── mmfinereason_with_think.py
│       ├── mmfinereason_with_nothink.py
│       ├── mmfinereason_batch_eval.py
│       └── post_filter.py
├── prompts/
│   ├── reasoning_distillation.md       # Reasoning generation prompt
│   ├── post_filter_detect_prompt.md    # C1~C4 detection prompt
│   ├── post_filter_rewrite_prompt.md   # Issue rewrite prompt (diverse workflows)
│   ├── post_filter_rewrite_clean_prompt.md  # Clean sample rewrite prompt
│   └── reasoning_workflows.md          # 10 reasoning workflow definitions
├── dataset/
│   └── raw/                            # Downloaded parquet files (not tracked)
│       └── MMFineReason-SFT-586K-Qwen3-VL-235B-Thinking/data/
├── app.py                              # Flask web viewer (pilot study)
├── templates/index.html
├── requirements.txt
└── .env                                # API key (not tracked)
```

---

## Environment Setup

```bash
git clone https://github.com/bk123477/mmfinereason.git
cd mmfinereason
pip install -r requirements.txt
```

`.env` 파일에 OpenRouter API key를 설정합니다 (git에 포함되지 않음):

```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxx
```

> API key는 [openrouter.ai/keys](https://openrouter.ai/keys)에서 발급

---

## Pipeline — Full Dataset (sft_run)

MMFineReason-SFT-586K 전체 데이터셋을 처리하는 메인 파이프라인입니다.  
70개 parquet 파일을 parquet × part 단위로 병렬 처리합니다.

### Step 1 — Reasoning Generation

```bash
# parquet 하나를 4개 터미널로 병렬 처리
python src/sft_run/sft_reason.py --model MODEL --parquet dataset/raw/.../train-00000-of-00070.parquet --part 0
python src/sft_run/sft_reason.py --model MODEL --parquet dataset/raw/.../train-00000-of-00070.parquet --part 1
python src/sft_run/sft_reason.py --model MODEL --parquet dataset/raw/.../train-00000-of-00070.parquet --part 2
python src/sft_run/sft_reason.py --model MODEL --parquet dataset/raw/.../train-00000-of-00070.parquet --part 3
```

출력: `dataset/reasoning/00000/train_00000_part{N}.json`

### Step 2 — Post-Filtering

```bash
python src/sft_run/sft_postfilter.py --model MODEL --input dataset/reasoning/00000/train_00000_part0.json
```

출력: `dataset/post_filtering/00000/train_00000_part0.json`

### Resume / Retry

```bash
# Ctrl+C 후 동일 명령어 재실행 → 완료된 index 자동 스킵
python src/sft_run/sft_reason.py --model MODEL --parquet ... --part 0

# 실패 index 재시도
python src/sft_run/sft_reason.py --model MODEL --parquet ... --part 0 --retry-errors
python src/sft_run/sft_reason.py --model MODEL --parquet ... --part 0 --retry-index 1234
```

자세한 사용법은 [`src/sft_run/README.md`](src/sft_run/README.md) 참조.

---

## Post-Filtering Quality Criteria (C1~C4)

| 코드 | 내용 | 처리 |
|------|------|------|
| C1 | reasoning에 reference 오염 (참조 답안 누출) | Rule-based (31개 패턴) + LLM |
| C2 | reasoning에 structured workflow 없음 | Rule-based (Format A/B) + LLM |
| C3 | response에 Phase 레이블 포함 | Rule-based |
| C4 | response에 reference 오염 | Rule-based + LLM |

- **Path A (clean)**: 이슈 없음 → 경량 구조 재정비
- **Path B (flagged)**: 이슈 있음 → 10가지 diverse reasoning workflow 중 선택하여 재작성, 최대 3회 반복 검증

---

## Output Directory Structure

```
dataset/
├── reasoning/
│   └── 00000/
│       ├── train_00000_part0.json
│       ├── train_00000_part1.json
│       └── logs/
│           ├── train_00000_part0_errors.json   # 실패 index 기록
│           └── train_00000_part0_summary.json  # 실행 이력 (append)
└── post_filtering/
    └── 00000/
        ├── train_00000_part0.json
        └── logs/
            ├── train_00000_part0_errors.json
            └── train_00000_part0_summary.json
```

---

## Pipeline — Pilot Study (sft_sample)

소규모 샘플 실험용 스크립트입니다. MMFineReason에서 subset별 샘플을 다운로드하고 reasoning을 생성합니다.

```bash
# 샘플 다운로드 (subset당 10개)
python src/sft_sample/mmfinereason_downloader.py

# Reasoning 생성
python src/sft_sample/mmfinereason_sft_reasoning.py \
  --model qwen/qwen3-235b-a22b \
  --samples 10

# 결과 시각화 (웹 뷰어)
python app.py
# → http://127.0.0.1:5002
```

---

## Prompts

| 파일 | 용도 |
|------|------|
| `reasoning_distillation.md` | Reasoning 생성 프롬프트 (Qwen3 thinking block 참조) |
| `post_filter_detect_prompt.md` | C1~C4 품질 검사 프롬프트 |
| `post_filter_rewrite_prompt.md` | Flagged 샘플 재작성 프롬프트 (10 workflows 포함) |
| `post_filter_rewrite_clean_prompt.md` | Clean 샘플 경량 재작성 프롬프트 |
| `reasoning_workflows.md` | 10가지 reasoning workflow 정의 |
