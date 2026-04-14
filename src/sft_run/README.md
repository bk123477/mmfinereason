# sft_run

MMFineReason 데이터셋의 reasoning 생성 및 post-filtering 파이프라인.

```
sft_reason.py       → 각 parquet 샘플에 대해 reasoning + response 생성
sft_postfilter.py   → 생성된 reasoning의 품질 검사 및 재작성
sft_2step_pipeline.py → VLM 생성 후 LLM 재구성/정제까지 한 번에 수행
```

---

## 파이프라인 흐름

```
dataset/raw/train-NNNNN-of-00070.parquet
        │
        ▼
  sft_reason.py   (OpenRouter API, multimodal)
        │
        ▼
dataset/reasoning/NNNNN/train_NNNNN_partP.json
        │
        ▼
  sft_postfilter.py   (detect C1~C4, rewrite)
        │
        ▼
dataset/post_filtering/NNNNN/train_NNNNN_partP.json

dataset/raw/train-NNNNN-of-00070.parquet
        │
        ▼
 sft_2step_pipeline.py
   Step 1: qwen/qwen3.5-397b-a17b (VLM generation)
   Step 2: deepseek/deepseek-v3.2 (LLM reconstruct / decontaminate)
        │
        ▼
dataset/two_step_pipeline/NNNNN/train_NNNNN_partP.json
```

---

## 출력 디렉토리 구조

```
dataset/
├── reasoning/
│   └── 00000/
│       ├── train_00000_part0.json     ← 결과
│       ├── train_00000_part1.json
│       ├── train_00000_part2.json
│       ├── train_00000_part3.json
│       └── logs/
│           ├── train_00000_part0_errors.json    ← 실패 index 기록
│           ├── train_00000_part0_summary.json   ← 실행 이력 (append)
│           └── ...
└── post_filtering/
    └── 00000/
        ├── train_00000_part0.json
        └── logs/
            ├── train_00000_part0_errors.json
            └── train_00000_part0_summary.json
└── two_step_pipeline/
    └── 00000/
        ├── train_00000_part0.json
        └── logs/
            ├── train_00000_part0_errors.json
            └── train_00000_part0_summary.json
```

---

## sft_reason.py

parquet 파일을 읽어 각 샘플에 대해 reasoning + response를 생성합니다.  
Qwen3-VL-235B의 thinking block을 참조 reasoning으로 사용하며, 이미지도 함께 전달합니다.

### API Key 설정

프로젝트 루트의 `.env`에 키가 저장되어 있어 `--api-key` 인자를 생략할 수 있습니다.

```bash
# .env
OPENROUTER_API_KEY=sk-or-v1-...
```

### 기본 사용법

```bash
python sft_reason.py \
  --model MODEL_ID \
  --parquet dataset/raw/MMFineReason-SFT-586K-Qwen3-VL-235B-Thinking/data/train-00000-of-00070.parquet \
  --part 0
```

### 4개 터미널로 병렬 처리

```bash
# Terminal 0
python sft_reason.py --model MODEL --parquet train-00000-of-00070.parquet --part 0

# Terminal 1
python sft_reason.py --model MODEL --parquet train-00000-of-00070.parquet --part 1

# Terminal 2
python sft_reason.py --model MODEL --parquet train-00000-of-00070.parquet --part 2

# Terminal 3
python sft_reason.py --model MODEL --parquet train-00000-of-00070.parquet --part 3
```

### Resume (자동)

동일한 명령어를 그대로 재실행하면 됩니다. part별로 파일이 분리되어 있어 다른 터미널 결과와 충돌하지 않습니다.

```bash
# Ctrl+C 후 재실행 — 완료된 index는 자동으로 스킵
python sft_reason.py --model MODEL --parquet train-00000-of-00070.parquet --part 0
```

### 실패 index 재시도

```bash
# error log에 기록된 index 전체 재시도
python sft_reason.py --model MODEL --parquet train-00000-of-00070.parquet --part 0 --retry-errors

# 단일 index 재시도
python sft_reason.py --model MODEL --parquet train-00000-of-00070.parquet --part 0 --retry-index 1234
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--parquet` | (필수) | parquet 파일 경로 |
| `--api-key` | `$OPENROUTER_API_KEY` | OpenRouter API key (.env에서 자동 로드) |
| `--model` | (필수) | OpenRouter model ID |
| `--part` | None | 파트 번호 (0-based). 미지정 시 전체 처리 |
| `--num-parts` | 4 | 전체 파트 수 |
| `--workers` | 10 | 동시 API 요청 수 |
| `--save-every` | 20 | N개 완료마다 중간 저장 |
| `--output-dir` | `dataset/reasoning` | 출력 루트 디렉토리 |
| `--template-path` | `prompts/reasoning_distillation.md` | 프롬프트 템플릿 |
| `--retry-index` | None | 단일 index 재생성 |
| `--retry-errors` | False | error log의 모든 index 재시도 |
| `--start-index` | 0 | 수동 범위 지정 (--part 미사용 시) |
| `--end-index` | None | 수동 범위 지정 (--part 미사용 시) |
| `--max-tokens` | 81920 | 최대 생성 토큰 수 |
| `--temperature` | 1.0 | |
| `--top-p` | 0.95 | |
| `--top-k` | 20 | |
| `--presence-penalty` | 1.5 | |

---

## sft_postfilter.py

생성된 reasoning JSON을 읽어 C1~C4 품질 기준으로 검사하고 필요시 재작성합니다.

### 품질 기준 (C1~C4)

| 코드 | 내용 | 탐지 방식 |
|------|------|-----------|
| C1 | reasoning에 reference 오염 | Rule-based (31개 패턴) + LLM |
| C2 | reasoning에 구조적 workflow 없음 | Rule-based (Format A/B) + LLM |
| C3 | response에 Phase 레이블 포함 | Rule-based |
| C4 | response에 reference 오염 | Rule-based + LLM |

- **Path A (clean)**: 이슈 없음 → 6-phase 경량 재구성
- **Path B (flagged)**: 이슈 있음 → 10가지 diverse workflow 중 선택하여 재작성 + 최대 3회 반복 검증

### 기본 사용법

```bash
python sft_postfilter.py \
  --model MODEL_ID \
  --input dataset/reasoning/00000/train_00000_part0.json \
  --part 0
```

### 4개 터미널로 병렬 처리

```bash
# Terminal 0
python sft_postfilter.py --model MODEL --input dataset/reasoning/00000/train_00000_part0.json --part 0

# Terminal 1
python sft_postfilter.py --model MODEL --input dataset/reasoning/00000/train_00000_part0.json --part 1

# Terminal 2
python sft_postfilter.py --model MODEL --input dataset/reasoning/00000/train_00000_part0.json --part 2

# Terminal 3
python sft_postfilter.py --model MODEL --input dataset/reasoning/00000/train_00000_part0.json --part 3
```

> **Note**: `--input`으로 `_part0.json` 파일을 주더라도 `--part`를 별도로 지정하면 해당 파일 내에서 다시 4분할합니다. 파일 하나를 그대로 처리하려면 `--part`를 생략하세요.

### Resume

```bash
# 동일 명령어 재실행으로 자동 이어하기
python sft_postfilter.py --model MODEL --input ... --part 0
```

### 실패 index 재시도

```bash
# error log 전체 재시도
python sft_postfilter.py --model MODEL --input ... --part 0 --retry-errors

# 단일 index 재시도
python sft_postfilter.py --model MODEL --input ... --part 0 --retry-index 1234
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--input` | (필수) | reasoning JSON 파일 경로 |
| `--api-key` | `$OPENROUTER_API_KEY` | OpenRouter API key (.env에서 자동 로드) |
| `--model` | (필수) | OpenRouter model ID |
| `--part` | None | 파트 번호 (0-based) |
| `--num-parts` | 4 | 전체 파트 수 |
| `--workers` | 5 | 동시 API 요청 수 |
| `--max-iterations` | 3 | 샘플당 최대 detect→rewrite 반복 횟수 |
| `--max-tokens` | 131072 | rewrite 최대 생성 토큰 수 |
| `--save-every` | 20 | N개 완료마다 중간 저장 |
| `--output-dir` | `dataset/post_filtering` | 출력 루트 디렉토리 |
| `--retry-index` | None | 단일 index 재처리 |
| `--retry-errors` | False | error log의 모든 index 재시도 |
| `--detect-prompt` | `prompts/post_filter_detect_prompt.md` | |
| `--rewrite-prompt` | `prompts/post_filter_rewrite_prompt.md` | |
| `--clean-rewrite-prompt` | `prompts/post_filter_rewrite_clean_prompt.md` | |
| `--workflows` | `prompts/reasoning_workflows.md` | rewrite prompt에 append되는 workflow 정의 |

---

## sft_2step_pipeline.py

123K 실험용 2-step 파이프라인입니다.

- **Step 1 (VLM)**: 이미지 + 질문 + 원본 think/answer를 보고 새 `reasoning_vlm`, `response_vlm` 생성
- **Step 2 (LLM)**: 생성 결과를 다시 읽고
  - reference/source 언급 제거
  - reasoning을 workflow 기반으로 자연스럽게 재구성
  - response를 설명형 prose로 다듬기

최종 결과는 “하나의 VLM이 직접 보고 reasoning/response를 생성한 것처럼” 읽히도록 설계되어 있습니다.

### 기본 사용법

```bash
python sft_2step_pipeline.py \
  --parquet dataset/raw/MMFineReason-SFT-123K/data/train-00000-of-00070.parquet \
  --part 0
```

### 권장 실행 예시

```bash
python sft_2step_pipeline.py \
  --vlm-model qwen/qwen3.5-397b-a17b \
  --llm-model deepseek/deepseek-v3.2 \
  --parquet dataset/raw/MMFineReason-SFT-123K/data/train-00000-of-00070.parquet \
  --output-dir dataset/two_step_pipeline_123k \
  --workers 8 \
  --part 0
```

### 4개 터미널 병렬 처리

```bash
# Terminal 0
python sft_2step_pipeline.py --parquet train-00000-of-00070.parquet --part 0

# Terminal 1
python sft_2step_pipeline.py --parquet train-00000-of-00070.parquet --part 1

# Terminal 2
python sft_2step_pipeline.py --parquet train-00000-of-00070.parquet --part 2

# Terminal 3
python sft_2step_pipeline.py --parquet train-00000-of-00070.parquet --part 3
```

### Resume

동일 명령어를 다시 실행하면 됩니다.

```bash
python sft_2step_pipeline.py --parquet ... --part 0
```

### 실패 index 재시도

```bash
# error log 전체 재시도
python sft_2step_pipeline.py --parquet ... --part 0 --retry-errors

# 단일 index 재시도
python sft_2step_pipeline.py --parquet ... --part 0 --retry-index 1234
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--parquet` | (필수) | 입력 parquet 파일 |
| `--api-key` | `$OPENROUTER_API_KEY` | OpenRouter API key |
| `--vlm-model` | `qwen/qwen3.5-397b-a17b` | 1단계 VLM 모델 |
| `--llm-model` | `deepseek/deepseek-v3.2` | 2단계 LLM 모델 |
| `--template-path` | `prompts/reasoning_distillation.md` | 1단계 생성 프롬프트 |
| `--reconstruct-prompt` | `prompts/reconstruct_prompt.md` | 2단계 재구성 프롬프트 |
| `--workflows` | `prompts/reasoning_workflows.md` | reconstruct prompt에 append되는 workflow 정의 |
| `--output-dir` | `dataset/two_step_pipeline` | 출력 루트 디렉토리 |
| `--workers` | `8` | 동시 API 요청 수 |
| `--part` | None | 파트 번호 (0-based) |
| `--num-parts` | `4` | 전체 파트 수 |
| `--start-index` | `0` | 수동 범위 지정 (--part 미사용 시) |
| `--end-index` | None | 수동 범위 지정 (--part 미사용 시) |
| `--retry-index` | None | 단일 index 재실행 |
| `--retry-errors` | False | error log의 모든 index 재시도 |
| `--save-every` | `20` | N개 완료마다 저장 |
| `--llm-max-iterations` | `2` | 2단계 재작성 최대 반복 수 |
| `--vlm-max-tokens` | `131072` | 1단계 최대 생성 토큰 수 |
| `--llm-max-tokens` | `32768` | 2단계 최대 생성 토큰 수 |

### 비용 집계

summary에는 두 단계 비용이 모두 누적됩니다.

- `token_usage.vlm`
- `token_usage.llm`
- `cost_estimate.vlm`
- `cost_estimate.llm`
- `cost_estimate.total_cost_usd`

123K용 새 데이터셋을 받을 예정이면 보통 아래 둘만 먼저 바꾸면 됩니다.

- 입력: `dataset/raw/MMFineReason-SFT-123K/data`
- 출력: `dataset/two_step_pipeline_123k`

아래 `scripts/run_2step_part.sh`, `scripts/retry_2step_errors_part.sh`도 같은 위치를 기준으로 만들어 두었습니다.

## 로그 파일

### `_errors.json`

실패한 index를 기록합니다. 성공하면 해당 항목이 자동으로 제거됩니다.

```json
{
  "1234": {"reason": "empty_response", "timestamp": "2026-04-08T10:23:45"},
  "5678": {"reason": "api_or_parse_failure", "timestamp": "2026-04-08T10:24:01"}
}
```

| reason | 발생 조건 |
|--------|-----------|
| `empty_response` | API가 빈 response 반환 (sft_reason) |
| `api_or_parse_failure` | API 오류 또는 XML 파싱 실패 (sft_postfilter) |

### `_summary.json`

실행할 때마다 결과를 append합니다. Resume해도 기존 기록이 유지됩니다.
각 run에는 OpenRouter 토큰 사용량과 비용 추정치도 함께 저장됩니다.

```json
{
  "runs": [
    {
      "start": "2026-04-08T09:00:00",
      "end":   "2026-04-08T11:30:00",
      "model": "qwen/qwen3.5-397b-a17b",
      "range": [0, 2100],
      "processed": 2100,
      "ok": 2098,
      "errors": 2,
      "token_usage": {
        "requests": 2100,
        "prompt_tokens": 123456789,
        "completion_tokens": 45678901,
        "total_tokens": 169135690
      },
      "cost_estimate": {
        "model": "qwen/qwen3.5-397b-a17b",
        "pricing_found": true,
        "input_cost_usd": 14.814815,
        "output_cost_usd": 13.70367,
        "total_cost_usd": 28.518485
      }
    }
  ]
}
```

- `sft_reason.py` 비용 추정은 `qwen/qwen3.5-397b-a17b` 가격표를 기준으로 계산합니다.
- `sft_postfilter.py` 비용 추정은 `deepseek/deepseek-v3.2` 가격표를 기준으로 계산합니다.
- `token_usage`는 OpenRouter 응답의 `usage` 필드를 합산한 값입니다.

---

## 에러 처리 요약

| 상황 | 처리 방식 |
|------|-----------|
| API rate limit (429) | 최대 5회 재시도, 대기 10→30→60→120→180초 |
| API 기타 오류 | 즉시 None 반환, error log 기록 |
| 이미지 인코딩 실패 | 경고 출력 후 이미지 없이 진행 |
| JSON parse 실패 (detection) | rule-based 결과만으로 폴백, 보수적으로 flagged 처리 |
| XML delimiter 미검출 (rewrite) | 경고 출력, None 반환, error log 기록 |
| Ctrl+C 중단 후 resume | 동일 명령어 재실행, 완료된 index 자동 스킵 |
| output JSON 파일 손상 | 경고 출력 후 해당 파일을 비운 상태로 재시작 |
| error log 파일 손상 | 경고 출력 후 빈 error log로 재시작 |
