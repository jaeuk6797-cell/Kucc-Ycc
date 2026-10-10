"""전략 탐색 파이프라인 (가지치기 + 교사 편집 + 과제 분할 + 예산 상한).

흐름
  1) 전략 여러 개를 짧은 예산(k회)으로 실행
  2) 성공률-비용 파레토 전선 + 레벨별 대표 전략을 남김 (중간값 필터 대신)
  3) 생존 전략을 교사가 구조화된 편집(추가/삭제/파라미터 변경)으로 확장
  4) 라운드마다 k를 두 배로 늘려 노이즈를 줄임 (이전 실행 결과는 재사용)
  5) 탐색에 쓰지 않은 과제 묶음(select, test)으로 최종 확인하고 기준 전략과 신뢰구간 비교

실제 실험에서 바꿀 부분
  - MockEnv: ALFWorld/ScienceWorld 실행으로 교체 (run 메서드만 맞추면 됨)
  - RandomEditTeacher: LLM에 편집 목록(JSON)을 요청하고 같은 형식으로 파싱
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


# ---------------------------------------------------------------- 데이터 구조

@dataclass(frozen=True)
class Strategy:
    sid: str
    M: int = 3                     # 최대 레벨 수
    N: int = 5                     # 작업당 계획 수
    eps: float = 0.0               # p_best 동률 허용 오차
    lam: float = 0.0               # 길이 페널티 계수
    rules: tuple[str, ...] = ()    # 프롬프트 규칙 (교사가 추가/삭제)

    def key(self) -> tuple:
        """내용이 같은 전략을 중복으로 보지 않기 위한 식별 키 (sid는 무시)."""
        return (self.M, self.N, self.eps, self.lam, self.rules)


@dataclass(frozen=True)
class Trial:
    success: bool
    tokens: int
    seconds: float


class Environment(Protocol):
    def run(self, s: Strategy, task: str, rep: int) -> Trial: ...


class Teacher(Protocol):
    def propose(self, parents: list[Strategy], n: int, rng: random.Random) -> list[Strategy]: ...


class BudgetExhausted(RuntimeError):
    pass


class Budget:
    """실행 횟수 상한. 탐색 전체의 비용을 고정한다."""

    def __init__(self, max_runs: int):
        self.left = max_runs

    def spend(self) -> None:
        if self.left <= 0:
            raise BudgetExhausted
        self.left -= 1


# ---------------------------------------------------------------- 통계와 선택

@dataclass(frozen=True)
class Stat:
    succ: float   # 성공률
    cost: float   # 성공 궤적의 평균 토큰 (성공이 없으면 inf)
    n: int        # 실행 횟수


Runs = dict[str, dict[tuple[str, int], Trial]]   # sid -> {(task, rep): Trial}


def ensure_runs(runs: Runs, env: Environment, s: Strategy, tasks: list[str], k: int,
                budget: Budget) -> None:
    """필요한 (과제, 반복) 실행만 새로 한다. 이미 있는 결과는 재사용."""
    bucket = runs.setdefault(s.sid, {})
    for t in tasks:
        for rep in range(k):
            if (t, rep) not in bucket:
                budget.spend()
                bucket[(t, rep)] = env.run(s, t, rep)


def summarize(runs: Runs, sid: str, tasks: list[str], k: int) -> Stat:
    trials = [runs[sid][(t, r)] for t in tasks for r in range(k)]
    ok = [x for x in trials if x.success]
    succ = len(ok) / len(trials)
    cost = sum(x.tokens for x in ok) / len(ok) if ok else math.inf
    return Stat(succ=succ, cost=cost, n=len(trials))


def dominates(a: Stat, b: Stat) -> bool:
    """a가 b를 모든 면에서 같거나 낫고, 적어도 한 면에서 더 나은가."""
    return a.succ >= b.succ and a.cost <= b.cost and (a.succ > b.succ or a.cost < b.cost)


def pareto_front(stats: dict[str, Stat]) -> list[str]:
    return [
        sid for sid, a in stats.items()
        if not any(dominates(b, a) for other, b in stats.items() if other != sid)
    ]


def select_survivors(pool: list[Strategy], stats: dict[str, Stat], keep: int) -> list[Strategy]:
    """파레토 전선을 먼저 남기고, 레벨 수(M)별 최고 전략을 반드시 하나씩 보존한다."""
    by_sid = {s.sid: s for s in pool}
    front = set(pareto_front(stats))

    def rank(sid: str) -> tuple:
        return (sid not in front, -stats[sid].succ, stats[sid].cost)

    ordered = sorted(stats, key=rank)
    bucket_leaders: dict[int, str] = {}
    for sid in ordered:
        bucket_leaders.setdefault(by_sid[sid].M, sid)

    chosen = list(dict.fromkeys(list(bucket_leaders.values()) + ordered))
    chosen = chosen[: max(keep, len(bucket_leaders))]
    return [by_sid[sid] for sid in chosen]


def paired_ci(runs: Runs, a: Strategy, b: Strategy, tasks: list[str], k: int,
              n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """과제 단위 짝지은 부트스트랩으로 성공률 차이(a - b)의 95% 신뢰구간."""
    def per_task(s: Strategy) -> list[float]:
        return [sum(runs[s.sid][(t, r)].success for r in range(k)) / k for t in tasks]

    diffs = [x - y for x, y in zip(per_task(a), per_task(b))]
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n_boot)
    )
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


# ---------------------------------------------------------------- 탐색 루프

@dataclass(frozen=True)
class Config:
    max_runs: int = 3000       # 탐색 전체 실행 상한
    rounds: int = 3
    k0: int = 2                # 첫 라운드 반복 수 (이후 두 배씩)
    keep_frac: float = 0.5     # 라운드마다 남기는 비율
    children: int = 6          # 라운드마다 교사가 만드는 자식 수
    seed: int = 0


def run_search(env: Environment, teacher: Teacher, seeds: list[Strategy],
               tasks_search: list[str], cfg: Config) -> tuple[list[Strategy], Runs, list[dict]]:
    rng = random.Random(cfg.seed)
    budget = Budget(cfg.max_runs)
    runs: Runs = {}
    log: list[dict] = []
    pool = list(seeds)
    survivors = list(seeds)

    try:
        for r in range(cfg.rounds):
            k = cfg.k0 * 2**r
            for s in pool:
                ensure_runs(runs, env, s, tasks_search, k, budget)

            stats = {s.sid: summarize(runs, s.sid, tasks_search, k) for s in pool}
            survivors = select_survivors(pool, stats, keep=max(2, int(len(pool) * cfg.keep_frac)))
            log.append({
                "round": r, "k": k, "pool": len(pool), "budget_left": budget.left,
                "front": pareto_front(stats),
                "top": [(s.sid, round(stats[s.sid].succ, 3), round(stats[s.sid].cost, 1))
                        for s in survivors[:3]],
            })

            if r < cfg.rounds - 1:
                children = teacher.propose(survivors, cfg.children, rng)
                seen = {s.key() for s in survivors}
                fresh = [c for c in children if c.key() not in seen]
                pool = survivors + fresh
    except BudgetExhausted:
        log.append({"stopped": "budget_exhausted"})

    return survivors, runs, log


def confirm(env: Environment, cands: list[Strategy], baseline: Strategy,
            tasks_select: list[str], tasks_test: list[str], k: int, budget: Budget) -> dict:
    """선택 과제로 후보를 고르고, 테스트 과제에서 기준 전략과 짝지어 비교한다."""
    runs: Runs = {}
    for s in cands + [baseline]:
        ensure_runs(runs, env, s, tasks_select, k, budget)
    stats = {s.sid: summarize(runs, s.sid, tasks_select, k) for s in cands}
    winner = max(cands, key=lambda s: (stats[s.sid].succ, -stats[s.sid].cost))

    test_runs: Runs = {}
    for s in (winner, baseline):
        ensure_runs(test_runs, env, s, tasks_test, k, budget)
    lo, hi = paired_ci(test_runs, winner, baseline, tasks_test, k)
    return {
        "winner": asdict(winner),
        "select_stats": {sid: asdict(st) for sid, st in stats.items()},
        "test_succ_winner": summarize(test_runs, winner.sid, tasks_test, k).succ,
        "test_succ_baseline": summarize(test_runs, baseline.sid, tasks_test, k).succ,
        "diff_ci95": (round(lo, 3), round(hi, 3)),
        "beats_baseline": lo > 0,
    }


# ---------------------------------------------------------------- 교체 가능한 구성요소 (테스트용)

RULE_POOL = (
    "계획 1은 개괄만 쓴다",
    "단계는 '무엇'을 쓰고 '어떻게'는 피한다",
    "불필요한 탐색 순서를 줄인다",
    "마지막 레벨은 실행 가능한 행동 이름을 참고한다",
)


def make_sid(s_fields: tuple) -> str:
    return hashlib.md5(repr(s_fields).encode()).hexdigest()[:8]


class RandomEditTeacher:
    """테스트용 교사. 실제로는 LLM에 편집 목록(JSON)을 요청하고 같은 형식으로 바꾼다.

    편집 형식 예시: {"op": "add_rule", "rule": "..."} / {"op": "set", "field": "M", "value": 2}
    """

    def propose(self, parents: list[Strategy], n: int, rng: random.Random) -> list[Strategy]:
        children = []
        for _ in range(n):
            p = rng.choice(parents)
            op = rng.choice(["add_rule", "drop_rule", "M", "N", "eps", "lam"])
            M, N, eps, lam, rules = p.M, p.N, p.eps, p.lam, list(p.rules)
            if op == "add_rule":
                cand = [r for r in RULE_POOL if r not in rules]
                if cand:
                    rules.append(rng.choice(cand))
            elif op == "drop_rule" and rules:
                rules.pop(rng.randrange(len(rules)))
            elif op == "M":
                M = rng.choice([2, 3, 4])
            elif op == "N":
                N = rng.choice([3, 5, 7])
            elif op == "eps":
                eps = rng.choice([0.0, 0.05, 0.1])
            elif op == "lam":
                lam = rng.choice([0.0, 0.01, 0.05])
            fields = (M, N, eps, lam, tuple(rules))
            children.append(Strategy(sid=make_sid(fields), M=M, N=N, eps=eps, lam=lam,
                                     rules=tuple(rules)))
        return children


class MockEnv:
    """테스트용 환경. 실제로는 ALFWorld/ScienceWorld 실행으로 교체한다."""

    def run(self, s: Strategy, task: str, rep: int) -> Trial:
        rnd = random.Random(hashlib.md5(f"{s.sid}|{task}|{rep}".encode()).hexdigest())
        p = 0.6 + 0.05 * (s.M == 3) + 0.04 * ("불필요한 탐색 순서를 줄인다" in s.rules)
        p -= 0.01 * abs(s.N - 5)
        ok = rnd.random() < min(max(p, 0.0), 1.0)
        tokens = int(300 * s.M * s.N * (1 + 0.1 * len(s.rules)) * (0.8 + 0.4 * rnd.random()))
        return Trial(success=ok, tokens=tokens, seconds=tokens / 1000)


# ---------------------------------------------------------------- 실행 예시

def main() -> None:
    tasks = [f"task_{i:03d}" for i in range(120)]
    tasks_search, tasks_select, tasks_test = tasks[:60], tasks[60:90], tasks[90:]

    baseline = Strategy(sid="baseline", M=3, N=5)
    seeds = [
        baseline,
        Strategy(sid="m2", M=2, N=5),
        Strategy(sid="m4", M=4, N=5),
        Strategy(sid="n3", M=3, N=3),
    ]

    cfg = Config()
    env, teacher = MockEnv(), RandomEditTeacher()

    survivors, runs, log = run_search(env, teacher, seeds, tasks_search, cfg)
    result = confirm(env, survivors, baseline, tasks_select, tasks_test,
                     k=8, budget=Budget(cfg.max_runs))

    out = {"log": log, "result": result}
    Path("search_log.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    for row in log:
        print(row)
    print("결과:", {k: v for k, v in result.items() if k != "select_stats"})


if __name__ == "__main__":
    main()
