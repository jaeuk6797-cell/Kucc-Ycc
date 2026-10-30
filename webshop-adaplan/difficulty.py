"""확장 횟수 기반 5단계 난이도.

기준 전략 하나(고정)로 각 과제를 k번 실행하고 아래 규칙으로 단계를 매긴다.
  1단계: 확장 없이 성공          2단계: 1번 확장 후 성공
  3단계: 2번 확장 후 성공        4단계: 3번 이상 확장 후 성공
  5단계: k번 중 과반이 실패 (최대 확장까지 해도 못 품)
성공한 실행들의 확장 횟수 '중앙값'을 써서 운에 의한 흔들림을 줄인다.

주의: 난이도는 반드시 '고정된 기준 전략'으로 한 번만 매기고, 모든 방법을 그 표로 비교할 것.
     평가할 방법 자신의 확장 횟수로 난이도를 매기면 순환 논리가 된다.
"""

from __future__ import annotations

import csv
import statistics
from collections import Counter

from agent import Strategy

# 난이도 측정용 기준 전략: 1레벨로 시작해 막힐 때마다 확장 (최대 4번 → 4단계까지 구분 가능)
REFERENCE = Strategy(sid="reference", start_level=1, max_level=3, trigger_n=3,
                     max_expand=4, scope="hybrid")


def tier_of(trials) -> int:
    ok = [t for t in trials if t.success]
    if len(ok) * 2 <= len(trials):        # 과반 실패
        return 5
    med = statistics.median(t.expansions for t in ok)
    return 1 + min(3, int(round(med)))


def assign_tiers(runner, tasks: list[int], k: int, ref: Strategy = REFERENCE,
                 out_csv: str | None = None) -> dict[int, int]:
    tiers: dict[int, int] = {}
    rows = []
    all_trials = runner.run_many([(ref, t, rep) for t in tasks for rep in range(k)])
    for i, t in enumerate(tasks):
        trials = all_trials[i * k:(i + 1) * k]
        tiers[t] = tier_of(trials)
        rows.append({
            "task": t, "tier": tiers[t],
            "succ_rate": sum(x.success for x in trials) / k,
            "mean_reward": round(sum(x.reward for x in trials) / k, 3),
            "expansions": ";".join(str(x.expansions) for x in trials),
            "steps": ";".join(str(x.steps) for x in trials),
        })
    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    dist = Counter(tiers.values())
    print("[difficulty] 단계별 과제 수:", {i: dist.get(i, 0) for i in range(1, 6)})
    return tiers


def spearman(xs: list[float], ys: list[float]) -> float:
    """순위 상관. 난이도 단계가 다른 기준(전문가 궤적 길이 등)과 같은 방향인지 확인할 때 쓴다."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for m in range(i, j + 1):
                r[order[m]] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (sx * sy) if sx and sy else 0.0
