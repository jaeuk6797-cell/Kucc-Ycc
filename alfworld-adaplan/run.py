"""ALFWorld에서 AdaPlan-H(M=3) 기준 vs M=3 + ADaPT식 확장을 비교한다.

  python run.py pilot   --n 3 --mock               # 어댑터/흐름 확인 (API 호출 없음)
  python run.py pilot   --n 5                      # 실제 모델로 에피소드당 비용 실측 + 견적
  python run.py compare --n 30                     # 두 전략을 같은 과제 n개에 돌려 성공률 비교

설정: alfworld_pilot_config.json (M=3, K=1). K 반복은 temperature 0에서 결과가 같아 의미가 없어 1로 둔다.
환경 준비: pip install alfworld (Python 3.11/3.12), alfworld-download, ALFWORLD_DATA 지정.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict

from agent import RunConfig, Strategy, Trial, run_episode
from cost import METER, BudgetExceeded

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "..", "alfworld", "pilot_config.json"), encoding="utf-8"))
M = CFG["M_values"][0]   # 3

# 비교할 두 전략. 둘 다 처음에 M=3 계획을 만든다.
BASELINE = Strategy(sid="adaplan_M3", start_level=M, max_level=M, trigger_n=0)
ADAPT = Strategy(sid="adaplan_M3_adapt", start_level=M, max_level=M, trigger_n=3,
                 max_expand=4, scope="local")


class RandomLLM:
    """API 없이 흐름만 확인하는 가짜 LLM. 계획은 고정 문장, 행동은 허용 명령 중 무작위."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def chat(self, role, system, user, temperature=0.0):
        from llm import Reply
        if role in ("plan", "refine"):
            return Reply("Step 1: Find the object.\nStep 2: Do the task.\nStep 3: Put it in place.", 10)
        cmds = user.split("# Available commands\n")[1].split(" | ")
        return Reply(f"Step: 1\nAction: {self.rng.choice(cmds)}", 10)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def build(args):
    from envs import ALFWorldEnv
    env = ALFWorldEnv(split=args.split)
    if args.mock:
        llm_for = lambda i: RandomLLM(i)                  # noqa: E731
    else:
        from llm import AnthropicLLM
        llm = AnthropicLLM(args.model, effort=args.effort)
        llm_for = lambda i: llm                           # noqa: E731
    return env, llm_for


def pick_tasks(num_games: int, n: int, seed: int) -> list[int]:
    """과제 순서가 유형별로 뭉쳐 있어서, 고정 시드로 무작위 추출해 유형이 섞이게 한다."""
    return sorted(random.Random(seed).sample(range(num_games), min(n, num_games)))


def load_cache(path):
    cache = {}
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                cache[(row["sid"], row["task"])] = row
    return cache


def run_one(env, llm_for, strat, task, cfg, cache, cache_path):
    key = (strat.sid, task)
    if key in cache:
        r = dict(cache[key])
        return Trial(**{k: r[k] for k in Trial.__dataclass_fields__})
    tr = run_episode(env, llm_for(task), strat, task, cfg)
    row = {"sid": strat.sid, "task": task, **asdict(tr)}
    cache[key] = row
    if cache_path:
        with open(cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    return tr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "compare"])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--split", default="eval_out_of_distribution",
                    help="eval_out_of_distribution(unseen 134) / eval_in_distribution(seen 140)")
    ap.add_argument("--n", type=int, default=5, help="과제 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--budget", type=float, default=CFG["budget_usd"])
    ap.add_argument("--stop-frac", type=float, default=CFG["stop_frac"])
    ap.add_argument("--cost-log", default="cost_calls.jsonl")
    ap.add_argument("--cache", default="results_cache.jsonl")
    args = ap.parse_args()

    METER.budget_usd, METER.stop_frac = args.budget, args.stop_frac
    METER.log_path = None if args.mock else args.cost_log
    cache_path = None if args.mock else args.cache
    env, llm_for = build(args)
    cfg = RunConfig(max_steps=args.max_steps)
    tasks = pick_tasks(env.num_games, args.n, args.seed)
    strategies = [BASELINE] if args.cmd == "pilot" else [BASELINE, ADAPT]
    cache = load_cache(cache_path)
    res: dict[str, dict[int, Trial]] = {s.sid: {} for s in strategies}
    try:
        for t in tasks:                       # 과제 단위로 두 전략을 번갈아 돌려 중단돼도 짝이 맞게 한다
            for s in strategies:
                tr = run_one(env, llm_for, s, t, cfg, cache, cache_path)
                res[s.sid][t] = tr
                print(f"[{s.sid}] task {t}: 성공={tr.success} 스텝={tr.steps} 확장={tr.expansions} "
                      f"입력={tr.in_tokens:,} 출력={tr.out_tokens:,} 비용=${tr.cost_usd:.4f}", flush=True)
    except BudgetExceeded as e:
        print(f"[예산 보호] 중단: {e}\n같은 명령을 다시 실행하면 캐시된 결과는 건너뛰고 이어서 진행합니다.")
    finally:
        print("[비용]", METER.summary())

    print(f"\n과제 {len(tasks)}개 (split={args.split}, seed={args.seed}), M={M}, K=1")
    for s in strategies:
        rs = list(res[s.sid].values())
        if not rs:
            continue
        k = sum(r.success for r in rs)
        lo, hi = wilson(k, len(rs))
        mean_cost = sum(r.cost_usd for r in rs) / len(rs)
        print(f"{s.sid}: 성공 {k}/{len(rs)} = {k/len(rs):.0%} (95% 윌슨 {lo:.0%}–{hi:.0%}), "
              f"평균 스텝 {sum(r.steps for r in rs)/len(rs):.1f}, 평균 확장 {sum(r.expansions for r in rs)/len(rs):.2f}, "
              f"에피소드당 ${mean_cost:.4f}")
    if args.cmd == "pilot" and res[BASELINE.sid]:
        mean = sum(r.cost_usd for r in res[BASELINE.sid].values()) / len(res[BASELINE.sid])
        usable = args.budget * args.stop_frac
        if mean:
            print(f"예산 ${args.budget}의 {args.stop_frac:.0%}(${usable:.2f}) 내 약 {int(usable/mean)} 에피소드 가능")
    if args.cmd == "compare":
        both = [t for t in tasks if t in res[BASELINE.sid] and t in res[ADAPT.sid]]
        b = sum(res[ADAPT.sid][t].success and not res[BASELINE.sid][t].success for t in both)
        c = sum(res[BASELINE.sid][t].success and not res[ADAPT.sid][t].success for t in both)
        print(f"짝 비교({len(both)}개): ADaPT만 성공 {b}, 기준만 성공 {c}")


if __name__ == "__main__":
    main()
