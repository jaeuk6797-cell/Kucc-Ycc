# ALFWorld × AdaPlan-H(M=3) + ADaPT

원 논문과 같이 M=3(초기 계층 계획 3레벨)으로 두고, 막힐 때 ADaPT식으로 막힌 단계만 하위 단계로 나누는 확장을 붙여 ALFWorld에서 성능을 비교한다. K는 논문에 값이 없고 temperature 0에서는 반복 결과가 같아서 1로 둔다 (`../alfworld/pilot_config.json`).

| 전략 | 내용 |
|---|---|
| `adaplan_M3` | 처음에 3레벨 계획을 만들고 그대로 실행 (확장 없음) |
| `adaplan_M3_adapt` | 같은 3레벨 계획 + 막힘 감지 시 그 단계만 하위 단계로 분해 (`scope=local`, 최대 4회) |

막힘 신호: 무효 명령, "Nothing happens.", 같은 관찰에서 같은 행동 반복이 최근 6스텝 중 3번.

## 환경 준비 (Python 3.11 또는 3.12 필요, 3.13은 textworld 오류)

```bash
python3.11 -m venv alfenv && ./alfenv/bin/pip install -r requirements.txt
export ALFWORLD_DATA=$HOME/alfworld_data && mkdir -p $ALFWORLD_DATA && ./alfenv/bin/alfworld-download
export ANTHROPIC_API_KEY=...
```

## 실행

```bash
python run.py compare --mock --n 3     # API 없이 흐름 확인
python run.py pilot --n 5              # 에피소드당 비용 실측과 견적 (먼저 이것부터)
python run.py compare --n 30           # 두 전략을 같은 과제 30개로 비교 (unseen 134개 중 무작위 추출)
```

- 비용 보호: `--budget`(기본 27달러)의 `--stop-frac`(기본 70%)에 닿으면 자동 중단. 같은 명령을 다시 실행하면 `results_cache.jsonl`에 있는 결과는 건너뛰고 이어서 진행.
- 호출별 토큰·비용은 `cost_calls.jsonl`에 기록.
- 과제는 고정 시드로 무작위 추출해 유형(6종)이 섞이게 하고, 두 전략을 과제마다 번갈아 돌려 짝 비교를 한다.
- 출력: 전략별 성공률(95% 윌슨 구간), 평균 스텝·확장·비용, 짝 비교(ADaPT만 성공/기준만 성공).
