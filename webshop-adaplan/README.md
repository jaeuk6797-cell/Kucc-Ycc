# WebShop × AdaPlan-H: 실패 시 확장 + 난이도 + 전략 탐색

AdaPlan-H식 계층 계획에 "막히면 계획을 더 나누는" 확장을 붙이고,
확장 횟수로 과제 난이도를 5단계로 매기고, 설정값을 자동 탐색하는 코드입니다.

## 파일

| 파일 | 내용 |
|---|---|
| `agent.py` | 계층 계획 생성, 막힘 감지, 확장(global/local/hybrid), 에피소드 실행 |
| `envs.py` | 실제 WebShop 어댑터 + 테스트용 가짜 쇼핑몰 |
| `llm.py` | Claude API(공식 SDK) / OpenAI 형식 LLM 호출 + 테스트용 가짜 LLM |
| `difficulty.py` | 확장 횟수 기반 5단계 난이도, 순위 상관 |
| `search.py` | 전략 탐색(파레토 선택, 교사 편집, 예산), 기준 전략과 신뢰구간 비교 |
| `run.py` | 실행 진입점 |
| `report.py` | 결과(trials.jsonl)를 표와 그래프로 정리 |

## 설치 없이 확인 (가짜 환경)

```bash
python run.py episode    --mock
python run.py difficulty --mock --tasks 1500:1600 --k 3
python run.py search     --mock --search 1500:1540 --select 1540:1570 --test 0:60 --max-runs 2000
```

가짜 환경 결과 숫자는 실제 성능과 무관합니다.

## 실제 실행

```bash
# 1) WebShop 설치 (저장소 README 참고). WebShop 폴더에서 실행하고 PYTHONPATH에 WebShop 폴더를 넣음
# 2-a) Claude API (기본값): ANTHROPIC_API_KEY 환경변수 필요
python run.py episode --session 1500 --model claude-haiku-5-5
# 2-b) vLLM 등 OpenAI 형식 서버
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
export OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY
python run.py episode --session 1500 --provider openai
# 3) 연결 확인 → 난이도 → 탐색
python run.py difficulty --tasks 1500:1600 --k 3
python run.py search --search 1500:1560 --select 1560:1590 --test 0:50
```

Claude 모델은 temperature를 바꿀 수 없어 `--temperature`가 무시됩니다. 생각 깊이는 `--effort`(에이전트)와
`--teacher-effort`(교사)로 조절합니다.

## 결과 보기 (표 + 그래프)

```bash
python report.py trials.jsonl                         # 전략별 표 + 그래프
python report.py trials.jsonl --tiers tiers.csv       # 난이도 단계별 성공률 추가
python report.py a.jsonl b.jsonl --labels 수정전,수정후 --out report_cmp   # 파일끼리 비교
```

`report/` 폴더에 다음 파일이 생깁니다.

| 파일 | 내용 |
|---|---|
| `summary.md`, `summary.csv` | 전략별 성공률(95% 윌슨 신뢰구간), 평균 보상, 토큰, 스텝, 확장. ★ = 파레토 전선 |
| `per_task.csv` | 과제 × 전략 성공률 |
| `success_rate.png` | 전략별 성공률과 신뢰구간 |
| `cost_vs_success.png` | 에피소드당 토큰 vs 성공률, 파레토 전선 |
| `expansions.png` | 확장 횟수 분포 |
| `success_by_tier.png` | 난이도 단계별 성공률 (`--tiers`를 줄 때) |

그룹 막대 그래프에는 성공률 상위 6개 전략만 그립니다(`--max-series`로 변경, baseline은 항상 포함).
색은 전략마다 고정이라, 일부만 그려도 같은 전략은 같은 색입니다.

## 단계(난이도) 규칙

기준 전략(`difficulty.REFERENCE`: 1레벨 시작, 최근 6스텝 중 막힘 신호 3번이면 확장, 최대 4회) 하나로만 매깁니다.

| 단계 | 기준 |
|---|---|
| 1 | 확장 없이 성공 |
| 2 | 1번 확장 후 성공 |
| 3 | 2번 확장 후 성공 |
| 4 | 3번 이상 확장 후 성공 |
| 5 | k번 중 과반 실패 |

## 알려진 한계

- 계획 모델은 학습된 AdaPlan-H 모델이 아니라 프롬프트 기반 LLM입니다. 시작 레벨은 모델이 고르지 않고 전략 설정값(`start_level`)으로 정합니다.
- WebShop은 상품을 사면 끝나므로, 잘못 사서 보상이 낮은 경우에는 확장이 일어나지 않습니다. 확장은 "막힘"(무효 행동, 또는 이미 본 화면으로 돌아온 행동)에서만 일어납니다.
- WebShop 텍스트 화면은 옵션을 클릭해도 바뀌지 않습니다. 그래서 에이전트가 고른 옵션을 따로 기억해
  프롬프트에 보여 주고, 새 옵션을 처음 누르는 것은 막힘 신호로 세지 않습니다.
- `WebShopEnv`의 메서드 이름과 세션 번호 분할(test 0~499 등)은 설치한 WebShop 버전에서 확인해야 합니다.
