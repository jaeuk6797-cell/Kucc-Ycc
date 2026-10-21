# WebShop × AdaPlan-H: 실패 시 확장 + 난이도 + 전략 탐색

AdaPlan-H식 계층 계획에 "막히면 계획을 더 나누는" 확장을 붙이고,
확장 횟수로 과제 난이도를 5단계로 매기고, 설정값을 자동 탐색하는 코드입니다.

## 파일

| 파일 | 내용 |
|---|---|
| `agent.py` | 계층 계획 생성, 막힘 감지, 확장(global/local/hybrid), 에피소드 실행 |
| `envs.py` | 실제 WebShop 어댑터 + 테스트용 가짜 쇼핑몰 |
| `llm.py` | OpenAI 형식 LLM 호출 + 테스트용 가짜 LLM |
| `difficulty.py` | 확장 횟수 기반 5단계 난이도, 순위 상관 |
| `search.py` | 전략 탐색(파레토 선택, 교사 편집, 예산), 기준 전략과 신뢰구간 비교 |
| `run.py` | 실행 진입점 |

## 설치 없이 확인 (가짜 환경)

```bash
python run.py episode    --mock
python run.py difficulty --mock --tasks 1500:1600 --k 3
python run.py search     --mock --search 1500:1540 --select 1540:1570 --test 0:60 --max-runs 2000
```

가짜 환경 결과 숫자는 실제 성능과 무관합니다.

## 실제 실행

```bash
# 1) WebShop 설치 (저장소 README 참고), 저장소 폴더 안에 이 파일들을 둠
# 2) 모델 서버
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
export OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY
pip install openai
# 3) 연결 확인 → 난이도 → 탐색
python run.py episode --session 1500
python run.py difficulty --tasks 1500:1600 --k 3
python run.py search --search 1500:1560 --select 1560:1590 --test 0:50
```

## 단계(난이도) 규칙

기준 전략(`difficulty.REFERENCE`: 1레벨 시작, 3연속 막힘 시 확장, 최대 4회) 하나로만 매깁니다.

| 단계 | 기준 |
|---|---|
| 1 | 확장 없이 성공 |
| 2 | 1번 확장 후 성공 |
| 3 | 2번 확장 후 성공 |
| 4 | 3번 이상 확장 후 성공 |
| 5 | k번 중 과반 실패 |

## 알려진 한계

- 계획 모델은 학습된 AdaPlan-H 모델이 아니라 프롬프트 기반 LLM입니다. 시작 레벨은 모델이 고르지 않고 전략 설정값(`start_level`)으로 정합니다.
- WebShop은 상품을 사면 끝나므로, 잘못 사서 보상이 낮은 경우에는 확장이 일어나지 않습니다. 확장은 "막힘"(무효 행동, 화면 변화 없음)에서만 일어납니다.
- `WebShopEnv`의 메서드 이름과 세션 번호 분할(test 0~499 등)은 설치한 WebShop 버전에서 확인해야 합니다.
