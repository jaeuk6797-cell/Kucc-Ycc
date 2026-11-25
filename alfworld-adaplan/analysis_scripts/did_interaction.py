import os, sys
from collections import defaultdict
from types import SimpleNamespace

sys.path.insert(0, os.path.expanduser("~/Kucc-Ycc/alfworld-adaplan"))
import run

cache, _ = run.load_cache("/mnt/d/kucc/results_cache.jsonl")
extra = {x.sid: x for x in run.EXTRA}
arms = {"M3": run.BASELINE, "M3a": run.ADAPT, "M1": extra["adaplan_M1"], "M1a": extra["adaplan_M1_adapt"]}


def load(split, steps, types):
    a = SimpleNamespace(model="claude-haiku-5-5", effort="low", split=split, max_steps=steps, mock=False, planner_model=None)
    sigs = {run.signature(a, s): (k, s.sid) for k, s in arms.items()}
    ok = run.type_filter(SimpleNamespace(types=types))
    res = {k: {} for k in arms}
    for (sid, t, rep, sig), r in cache.items():
        if sig in sigs and sigs[sig][1] == sid and ok(r.get("ttype")):
            res[sigs[sig][0]][(t, rep)] = int(r["success"])
    return res


def eff(res, f):
    keys = set.intersection(*[set(v) for v in res.values()])
    diffs = defaultdict(list)
    for t, rep in keys:
        diffs[t].append(f(lambda k: res[k][(t, rep)]))
    d, lo, hi = run.task_boot(diffs)
    return f"{d:+6.1%}p [{lo:+.1%}, {hi:+.1%}]", len(keys)


for name, split, steps, types in [
    ("A seen 전체 (40스텝)", "eval_in_distribution", 40, None),
    ("B seen 어려운 유형 (40스텝)", "eval_in_distribution", 40, "heat,look_at,clean,cool"),
    ("C unseen 더 어려운 조건 (20스텝)", "eval_out_of_distribution", 20, None),
]:
    res = load(split, steps, types)
    n = {k: len(v) for k, v in res.items()}
    print(f"\n=== {name}  에피소드 {n}")
    if min(n.values()) == 0:
        print("  (일부 전략 결과 없음, 건너뜀)")
        continue
    keys = set.intersection(*[set(v) for v in res.values()])
    print("  공통 에피소드", len(keys), " 성공률:", {k: f"{sum(res[k][x] for x in keys)/len(keys):.0%}" for k in arms})
    for label, f in [
        ("M 효과, ADaPT 꺼짐 (M1-M3)", lambda g: g("M1") - g("M3")),
        ("M 효과, ADaPT 켜짐 (M1a-M3a)", lambda g: g("M1a") - g("M3a")),
        ("ADaPT 효과, M=3 (M3a-M3)", lambda g: g("M3a") - g("M3")),
        ("ADaPT 효과, M=1 (M1a-M1)", lambda g: g("M1a") - g("M1")),
        ("상호작용 (M1a-M1)-(M3a-M3)", lambda g: (g("M1a") - g("M1")) - (g("M3a") - g("M3"))),
    ]:
        txt, k = eff(res, f)
        print(f"  {label:<32}{txt}  n={k}")
