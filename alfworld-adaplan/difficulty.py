"""에이전트와 무관한 과제 난이도 (멘토 피드백 Priority 2: 난이도 정의가 평가 대상 agent에 의존하면 순환적).

난이도는 agent를 한 번도 돌리지 않고 과제 파일(traj_data.json, receps.json)에서 바로 잰다.

  plan_len   전문가 계획의 상위 행동 수 (NoOp 제외). 과제가 요구하는 단계 수
  n_goto     전문가 계획에서 이동해야 하는 장소 수
  n_receps   장면의 수납장/가구(receptacle) 수. 물건을 찾아야 하는 탐색 공간 크기
  needs_tool 씻기/데우기/식히기 중 하나가 필요한지 (싱크대/전자레인지/냉장고)
  ttype      과제 유형 6종

단계(tier)는 (plan_len, n_receps) 순으로 정렬한 순위를 5등분한 값이다 (1 쉬움 ~ 5 어려움).
agent 결과는 단계를 매기는 데 쓰지 않고, 이 단계가 실제 성공률과 맞는지 확인하는 데만 쓴다.

  python difficulty.py features                    # 과제별 속성 표(difficulty_features.csv) 만들기 (API 불필요)
  python difficulty.py analyze --cache results_cache.jsonl   # 단계별 성공률 + 순위상관 + 전략 간 일치도
  python difficulty.py compare-tiers --cache results_cache.jsonl   # agent 비의존 단계 vs 기존 agent 의존(확장 횟수) 단계 비교
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

TOOL_ACTIONS = {"CleanObject", "HeatObject", "CoolObject"}


def game_features(game_file: str) -> dict:
    d = os.path.dirname(game_file)
    traj = json.load(open(os.path.join(d, "traj_data.json"), encoding="utf-8"))
    acts = [h["discrete_action"]["action"] for h in traj["plan"]["high_pddl"]]
    acts = [a for a in acts if a != "NoOp"]
    try:
        n_receps = len(json.load(open(os.path.join(d, "receps.json"), encoding="utf-8")))
    except FileNotFoundError:
        n_receps = 0
    return {"ttype": traj["task_type"], "plan_len": len(acts),
            "n_goto": sum(a == "GotoLocation" for a in acts),
            "n_receps": n_receps, "needs_tool": int(any(a in TOOL_ACTIONS for a in acts)),
            "scene": traj["scene"]["scene_num"]}


def assign_tiers(rows: list[dict], k: int = 5) -> None:
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["plan_len"], rows[i]["n_receps"]))
    for rank, i in enumerate(order):
        rows[i]["tier"] = rank * k // len(rows) + 1


def build_features(split: str, out: str) -> list[dict]:
    from envs import ALFWorldEnv
    env = ALFWorldEnv(split=split)
    rows = []
    for idx, gf in enumerate(env.game_files):
        rows.append({"task": idx, **game_features(gf)})
    assign_tiers(rows)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for t in range(i, j + 1):
            r[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a, b) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def analyze(features_csv: str, cache_path: str) -> None:
    feats = {int(r["task"]): r for r in csv.DictReader(open(features_csv, encoding="utf-8"))}
    # 같은 (전략, 과제)의 여러 반복/설정은 평균 성공률로 합친다
    succ: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for line in open(cache_path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            succ[r["sid"]][r["task"]].append(int(r["success"]))
    for sid, per in succ.items():
        tasks = sorted(t for t in per if t in feats)
        rate = {t: sum(per[t]) / len(per[t]) for t in tasks}
        print(f"\n[{sid}] 과제 {len(tasks)}개")
        print("  단계(1쉬움~5어려움)별 성공률:")
        by = defaultdict(list)
        for t in tasks:
            by[int(feats[t]["tier"])].append(rate[t])
        for tier in sorted(by):
            v = by[tier]
            print(f"    tier {tier}: {sum(v)/len(v):.0%} (n={len(v)})")
        for name in ("plan_len", "n_goto", "n_receps", "needs_tool"):
            rho = spearman([float(feats[t][name]) for t in tasks], [1 - rate[t] for t in tasks])
            print(f"  난이도 속성 {name:<10} vs 실패율 순위상관 = {rho:+.2f}")
        by_tt = defaultdict(list)
        for t in tasks:
            by_tt[feats[t]["ttype"]].append(rate[t])
        print("  유형별 성공률: " + ", ".join(f"{k.split('_')[0]}{'..' if len(k.split('_'))>1 else ''}:{sum(v)/len(v):.0%}"
                                          for k, v in sorted(by_tt.items())))
    sids = list(succ)
    if len(sids) >= 2:
        a, b = sids[0], sids[1]
        common = sorted(set(succ[a]) & set(succ[b]) & set(feats))
        rho = spearman([sum(succ[a][t]) / len(succ[a][t]) for t in common],
                       [sum(succ[b][t]) / len(succ[b][t]) for t in common])
        print(f"\n전략 간 일치도: {a} vs {b} 과제별 성공률 순위상관 = {rho:+.2f} (n={len(common)})")
        print("  난이도 속성이 두 전략 모두에서 같은 방향의 상관을 보이면 특정 agent에 종속되지 않는다는 근거가 된다.")


def agent_tier(rows: list[dict]) -> int:
    """기존 방식(webshop-adaplan/difficulty.py)과 같은 규칙: 확장 없이 성공=1, 1번 확장 후=2, 2번=3, 3번 이상=4, 과반 실패=5."""
    ok = [r for r in rows if r["success"]]
    if len(ok) * 2 <= len(rows):
        return 5
    ex = sorted(r["expansions"] for r in ok)
    med = ex[len(ex) // 2] if len(ex) % 2 else (ex[len(ex) // 2 - 1] + ex[len(ex) // 2]) / 2
    return 1 + min(3, int(round(med)))


def compare_tiers(features_csv: str, cache_path: str, ref_sid: str, other_sid: str) -> None:
    feats = {int(r["task"]): r for r in csv.DictReader(open(features_csv, encoding="utf-8"))}
    by: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for line in open(cache_path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            by[r["sid"]][r["task"]].append(r)
    ref, other = by.get(ref_sid, {}), by.get(other_sid, {})
    tasks = sorted(t for t in ref if t in feats)
    if not tasks:
        print(f"{ref_sid} 결과가 없습니다.")
        return
    static = [int(feats[t]["tier"]) for t in tasks]
    agent = [agent_tier(ref[t]) for t in tasks]
    print(f"기준 전략(agent 의존 단계 산출용): {ref_sid}, 과제 {len(tasks)}개")
    print("agent 의존 단계 분포:", {i: agent.count(i) for i in range(1, 6)})
    print("agent 비의존 단계 분포:", {i: static.count(i) for i in range(1, 6)})

    print("\n교차표 (행: agent 비의존 단계, 열: agent 의존 단계)")
    print("          " + "".join(f"  A{j}" for j in range(1, 6)))
    for i in range(1, 6):
        print(f"  S{i}     " + "".join(f"{sum(1 for s, a in zip(static, agent) if s == i and a == j):>4}" for j in range(1, 6)))
    same = sum(s == a for s, a in zip(static, agent)) / len(tasks)
    near = sum(abs(s - a) <= 1 for s, a in zip(static, agent)) / len(tasks)
    print(f"\n두 단계의 순위상관 = {spearman(static, agent):+.2f}   같은 단계 {same:.0%}, 1단계 이내 {near:.0%}")

    fail_ref = [1 - sum(r["success"] for r in ref[t]) / len(ref[t]) for t in tasks]
    print(f"\n{ref_sid} 실패율과의 상관: 비의존 단계 {spearman(static, fail_ref):+.2f}, 의존 단계 {spearman(agent, fail_ref):+.2f}"
          "  (의존 단계는 이 결과로 만든 것이라 높게 나오는 게 당연한 순환)")
    common = [t for t in tasks if t in other]
    if common:
        fail_other = [1 - sum(r["success"] for r in other[t]) / len(other[t]) for t in common]
        st = [int(feats[t]["tier"]) for t in common]
        ag = [agent_tier(ref[t]) for t in common]
        print(f"다른 전략({other_sid}) 실패율과의 상관(순환 없음): 비의존 단계 {spearman(st, fail_other):+.2f}, "
              f"의존 단계 {spearman(ag, fail_other):+.2f}")
        print("  -> 의존 단계가 다른 agent의 실패를 덜 설명하면, agent 의존 난이도가 그 agent에 묶여 있다는 증거")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["features", "analyze", "compare-tiers"])
    ap.add_argument("--ref-sid", default="adaplan_M3_adapt", help="agent 의존 단계를 만들 기준 전략")
    ap.add_argument("--other-sid", default="adaplan_M3", help="교차 검증용 다른 전략")
    ap.add_argument("--split", default="eval_out_of_distribution")
    ap.add_argument("--features", default="difficulty_features.csv")
    ap.add_argument("--cache", default="results_cache.jsonl")
    a = ap.parse_args()
    if a.cmd == "features":
        rows = build_features(a.split, a.features)
        tc = defaultdict(int)
        for r in rows:
            tc[r["tier"]] += 1
        print(f"{len(rows)}개 과제 속성 저장: {a.features}")
        print("단계별 과제 수:", dict(sorted(tc.items())))
    elif a.cmd == "analyze":
        analyze(a.features, a.cache)
    else:
        compare_tiers(a.features, a.cache, a.ref_sid, a.other_sid)
