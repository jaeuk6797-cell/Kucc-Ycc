#!/bin/bash
# ALFWorld 전략 탐색. WSL에서: bash alfworld_search.sh [workers] [seed]
# 설치/설정은 README의 "ALFWorld" 절 참고. 이 스크립트는 실제 API를 호출하니 돌리기 전에 확인할 것.
#   분할: valid_unseen 134개 → 탐색 0:40, 선택 40:70, 테스트 70:134
#   에피소드 모델 haiku, 교사 sonnet (run.py 기본값). 스텝 상한 30.
eval "$(grep '^export ANTHROPIC_API_KEY' ~/.bashrc | tr -d '\r' | tail -1)"
[ -n "$ANTHROPIC_API_KEY" ] || { echo "ANTHROPIC_API_KEY 없음"; exit 1; }
WORKERS=${1:-3}; SEED=${2:-0}
SRC=$(cd "$(dirname "$0")" && pwd)
OUT=$SRC/results/alfworld/seed${SEED}
mkdir -p "$OUT"
. ~/alf_venv/bin/activate
export ALFWORLD_DATA=$HOME/alfworld_data
python "$SRC/run.py" search --env alfworld --alf-split valid_unseen \
  --search 0:40 --select 40:70 --test 70:134 --max-steps 30 \
  --max-runs 300 --rounds 3 --confirm-k 2 --seed "$SEED" --workers "$WORKERS" \
  --trials-out "$OUT/trials.jsonl" --search-log "$OUT/search_log.json" --resume \
  >> "$OUT/stdout.log" 2>&1
grep -aE '^\[round|^\[confirm|^\[search|beats_baseline|test_skipped|Traceback|Error' "$OUT/stdout.log"
