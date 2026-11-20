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
  python difficulty.py effect --tiers 4 --split eval_out_of_distribution   # agent 비의존 4단계별 ADaPT 효과 (API 불필요)
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


def assign_tiers_tied(rows: list[dict], k: int) -> None:
    """(plan_len, n_receps)가 같은 과제는 같은 단계에 넣는 k등분 (assign_tiers는 동점을 과제 번호 순으로 갈라서 쓰지 않는다)."""
    keys = [(r["plan_len"], r["n_receps"]) for r in rows]
    order = sorted(set(keys))
    n = len(rows)
    below = 0
    tier_of = {}
    for key in order:
        cnt = keys.count(key)
        mid = below + (cnt - 1) / 2          # 이 묶음의 가운데 순위(0부터)
        tier_of[key] = int(mid * k / n) + 1
        below += cnt
    for r, key in zip(rows, keys):
        r["tier"] = tier_of[key]


def effect_by_tier(features_csv: str, cache_path: str, split: str, k: int, base_sid: str, treat_sid: str,
                   model: str, effort: str, max_steps: int) -> str:
    """agent 비의존 단계별로 (처치 - 기준) 성공률 차이를 과제 단위로 집계한다. 캐시는 서명으로 split을 걸러 섞이지 않게 한다."""
    import random
    from types import SimpleNamespace

    import run
    rows = list(csv.DictReader(open(features_csv, encoding="utf-8")))
    for r in rows:
        r["task"] = int(r["task"])
        r["plan_len"], r["n_receps"] = int(r["plan_len"]), int(r["n_receps"])
    assign_tiers_tied(rows, k)
    feats = {r["task"]: r for r in rows}
    a = SimpleNamespace(model=model, effort=effort, split=split, max_steps=max_steps, mock=False)
    sig = {sid: run.signature(a, run.STRATS[sid]) for sid in (base_sid, treat_sid)}
    succ = {sid: defaultdict(dict) for sid in sig}
    ttype_cache: dict[int, str] = {}
    for line in open(cache_path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r["sid"] in sig and r.get("sig") == sig[r["sid"]]:
                succ[r["sid"]][r["task"]][r.get("rep", 0)] = int(r["success"])
                if r.get("ttype"):
                    ttype_cache[r["task"]] = r["ttype"]
    # 속성표의 과제 번호가 실험 때의 과제 번호와 같은 게임인지 확인한다 (유형이 하나라도 다르면 순서가 다른 표)
    bad = [t for t, tt in ttype_cache.items() if t in feats and feats[t]["ttype"] != tt]
    if bad:
        return (f"\n[중단] 속성표({features_csv})의 과제 유형이 실험 결과와 {len(bad)}/{len(ttype_cache)}개 과제에서 다릅니다.\n"
                f"표가 다른 게임 순서로 만들어진 것입니다. 같은 컴퓨터에서 다시 만드세요:\n"
                f"  python difficulty.py features --split {split} --features {features_csv}")
    out = [f"\n난이도 단계별 효과: {treat_sid} - {base_sid} (split={split}, 단계 {k}개, 과제 단위, 과제 부트스트랩 95%)"]
    tasks = sorted(t for t in feats if t in succ[base_sid] and t in succ[treat_sid])
    if not tasks:
        out.append("두 전략 모두 결과가 있는 과제가 없습니다. (split/모델/effort/max_steps가 캐시와 같은지 확인)")
        return "\n".join(out)

    def mean_pair(t):
        reps = sorted(set(succ[base_sid][t]) & set(succ[treat_sid][t]))
        if not reps:
            return None
        return (sum(succ[base_sid][t][x] for x in reps) / len(reps), sum(succ[treat_sid][t][x] for x in reps) / len(reps))
    pairs = {t: mean_pair(t) for t in tasks}
    pairs = {t: p for t, p in pairs.items() if p}

    def boot(ts, iters=5000):
        diffs = [pairs[t][1] - pairs[t][0] for t in ts]
        if len(diffs) < 2:
            return None
        rng = random.Random(0)
        bs = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(iters))
        return sum(diffs) / len(diffs), bs[int(0.025 * iters)], bs[int(0.975 * iters) - 1]

    out.append("단계  과제수  plan_len  n_receps   기준   처치   차이 [95%]                 유형 구성")
    for tier in range(1, k + 1):
        ts = [t for t in pairs if feats[t]["tier"] == tier]
        if not ts:
            continue
        pl = [feats[t]["plan_len"] for t in ts]
        nr = [feats[t]["n_receps"] for t in ts]
        b = sum(pairs[t][0] for t in ts) / len(ts)
        tr = sum(pairs[t][1] for t in ts) / len(ts)
        ci = boot(ts)
        cis = f"{ci[0]:+.1%} [{ci[1]:+.1%}, {ci[2]:+.1%}]" if ci else "-"
        comp = defaultdict(int)
        for t in ts:
            comp[feats[t]["ttype"].split("_")[0] + ("_" + feats[t]["ttype"].split("_")[1] if feats[t]["ttype"].startswith("pick_") else "")] += 1
        out.append(f"{tier:>3}  {len(ts):>6}  {min(pl)}-{max(pl):<7} {min(nr)}-{max(nr):<7} {b:>5.0%} {tr:>5.0%}   {cis:<28}"
                   + ", ".join(f"{k2}:{v}" for k2, v in sorted(comp.items())))
    allts = list(pairs)
    ci = boot(allts)
    out.append(f"전체  {len(allts):>5}  {'':<9}{'':<9} "
               f"{sum(p[0] for p in pairs.values())/len(pairs):>5.0%} {sum(p[1] for p in pairs.values())/len(pairs):>5.0%}   "
               f"{ci[0]:+.1%} [{ci[1]:+.1%}, {ci[2]:+.1%}]")
    rho = spearman([feats[t]["tier"] for t in pairs], [1 - pairs[t][0] for t in pairs])
    out.append(f"\n단계(쉬움→어려움) vs 기준 실패율 순위상관 = {rho:+.2f}  (agent 비의존 난이도가 실제 어려움과 맞는지)")
    out.append("주의: plan_len은 과제 유형과 거의 겹쳐서 단계와 유형이 섞여 있음(위 유형 구성 참고). 단계별 과제 수가 작아 구간이 넓음.")
    return "\n".join(out)


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
    ap.add_argument("cmd", choices=["features", "analyze", "compare-tiers", "effect"])
    ap.add_argument("--ref-sid", default="adaplan_M3_adapt", help="agent 의존 단계를 만들 기준 전략")
    ap.add_argument("--other-sid", default="adaplan_M3", help="교차 검증용 다른 전략")
    ap.add_argument("--split", default="eval_out_of_distribution")
    ap.add_argument("--features", default="difficulty_features.csv")
    ap.add_argument("--cache", default=None, help="기본: $ADAPLAN_OUT/results_cache.jsonl, 없으면 ./results_cache.jsonl")
    ap.add_argument("--tiers", type=int, default=4, help="effect: 난이도 단계 수")
    ap.add_argument("--base-sid", default="adaplan_M3")
    ap.add_argument("--treat-sid", default="adaplan_M3_adapt")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-steps", type=int, default=40)
    a = ap.parse_args()
    a.cache = a.cache or os.path.join(os.environ.get("ADAPLAN_OUT") or ".", "results_cache.jsonl")
    if a.cmd == "features":
        rows = build_features(a.split, a.features)
        tc = defaultdict(int)
        for r in rows:
            tc[r["tier"]] += 1
        print(f"{len(rows)}개 과제 속성 저장: {a.features}")
        print("단계별 과제 수:", dict(sorted(tc.items())))
    elif a.cmd == "analyze":
        analyze(a.features, a.cache)
    elif a.cmd == "effect":
        text = effect_by_tier(a.features, a.cache, a.split, a.tiers, a.base_sid, a.treat_sid, a.model, a.effort, a.max_steps)
        print(text)
        from types import SimpleNamespace

        import run
        run.save_report(SimpleNamespace(mock=False, cmd="tiers", n=a.tiers, reps=0, seed=0, split=a.split, model=a.model,
                                        out_dir=os.environ.get("ADAPLAN_OUT"), push=False, cache=a.cache), text, "")
    else:
        compare_tiers(a.features, a.cache, a.ref_sid, a.other_sid)
