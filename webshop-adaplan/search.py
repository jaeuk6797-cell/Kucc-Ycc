"""전략 탐색 (strategy_search.py를 WebShop 에이전트용으로 옮긴 버전).

흐름은 이전과 같다.
  라운드마다 k회 실행 → 파레토 전선(성공률↑, 성공 시 토큰↓) + start_level별 대표 보존
  → 교사가 자식 전략 생성 → k 두 배 → 예산 소진 또는 라운드 종료
  → 선택 과제로 승자 결정 → 테스트 과제에서 기준 전략과 짝지은 부트스트랩 비교
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import asdict, dataclass, replace

from agent import RunConfig, Strategy, Trial, run_episode
from llm import LLM


# ---------------------------------------------------------------- 실행기

class TaskRunner:
    """(전략, 과제, 반복 번호) 하나를 실행한다. 과제 = WebShop 세션 번호.

    workers > 1 이면 run_many가 프로세스 여러 개로 나눠 돌린다. 각 프로세스는 build()로
    자기 환경과 LLM을 따로 만든다 (WebShop 환경과 Java 검색 엔진은 스레드끼리 공유하기 어려움).
    """

    def __init__(self, env, llm_factory, cfg: RunConfig, workers: int = 1, build=None):
        self.env = env
        self.llm_factory = llm_factory      # seed -> LLM (가짜 LLM은 seed로 재현성 확보)
        self.cfg = cfg
        self.workers = workers
        self.build = build                  # () -> (env, llm_factory, cfg), 피클 가능해야 함
        self._pool = None

    def run(self, s: Strategy, task: int, rep: int) -> Trial:
        seed = int(hashlib.md5(f"{s.key()}|{task}|{rep}".encode()).hexdigest()[:8], 16)
        return run_episode(self.env, self.llm_factory(seed), s, task, self.cfg)

    def run_many(self, jobs: list[tuple[Strategy, int, int]], on_done=None) -> list[Trial]:
        """여러 (전략, 과제, 반복)을 실행해 같은 순서로 결과를 돌려준다.
        on_done(job, trial)은 에피소드 하나가 끝날 때마다 (끝난 순서대로) 불린다."""
        results: list[Trial | None] = [None] * len(jobs)
        if not jobs:
            return []
        if self.workers <= 1 or self.build is None:
            for i, j in enumerate(jobs):
                results[i] = self.run(*j)
                if on_done:
                    on_done(j, results[i])
            return results
        from concurrent.futures import as_completed
        if self._pool is None:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            self._pool = ProcessPoolExecutor(self.workers, mp_context=multiprocessing.get_context("spawn"),
                                             initializer=_init_worker, initargs=(self.build,))
        futures = {self._pool.submit(_worker_run, j): i for i, j in enumerate(jobs)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            if on_done:
                on_done(jobs[i], results[i])
        return results


_WORKER: TaskRunner | None = None


def _init_worker(build) -> None:
    global _WORKER
    env, factory, cfg = build()
    _WORKER = TaskRunner(env, factory, cfg)


def _worker_run(job, attempts: int = 3) -> Trial:
    """에피소드 하나를 실행. API 오류 등으로 실패하면 잠시 쉬었다가 처음부터 다시 실행한다.
    끝내 실패하면 프로세스 사이에서 전달 가능한 RuntimeError로 바꿔 올린다
    (SDK 오류 객체는 피클이 안 돼서 그대로 올리면 프로세스 풀 전체가 깨진다)."""
    import time
    for i in range(attempts):
        try:
            return _WORKER.run(*job)
        except Exception as e:  # noqa: BLE001
            if i == attempts - 1:
                s, task, rep = job
                raise RuntimeError(f"{s.sid} task={task} rep={rep}: {type(e).__name__}: {e}") from None
            time.sleep(30 * (i + 1))


# ---------------------------------------------------------------- 예산, 통계, 선택

class BudgetExhausted(RuntimeError):
    pass


class Budget:
    def __init__(self, max_runs: int):
        self.left = max_runs

    def spend(self) -> None:
        if self.left <= 0:
            raise BudgetExhausted
        self.left -= 1


@dataclass(frozen=True)
class Stat:
    succ: float      # 성공률
    cost: float      # 성공한 에피소드의 평균 토큰 (성공 없으면 inf)
    steps: float     # 성공한 에피소드의 평균 행동 수
    expand: float    # 전체 평균 확장 횟수
    n: int


Runs = dict[str, dict[tuple[int, int], Trial]]


def ensure_runs(runs: Runs, runner: TaskRunner, strategies, tasks: list[int], k: int,
                budget: Budget, log_fn=None) -> None:
    """없는 (과제, 반복)만 실행한다. 전략 여러 개를 한 번에 넘기면 모든 작업을 한 줄로 세워
    프로세스가 쉬는 시간을 줄인다. 예산이 모자라면 할 수 있는 만큼 실행한 뒤 BudgetExhausted."""
    if isinstance(strategies, Strategy):
        strategies = [strategies]
    jobs, exhausted = [], False
    for s in strategies:
        bucket = runs.setdefault(s.sid, {})
        for t in tasks:
            for rep in range(k):
                if (t, rep) not in bucket:
                    try:
                        budget.spend()
                    except BudgetExhausted:
                        exhausted = True
                        break
                    jobs.append((s, t, rep))
            if exhausted:
                break
        if exhausted:
            break

    def done(job, trial):          # 에피소드가 끝나는 즉시 저장과 기록
        s, t, rep = job
        runs[s.sid][(t, rep)] = trial
        if log_fn:
            log_fn({"sid": s.sid, "task": t, "rep": rep, **asdict(trial)})

    runner.run_many(jobs, on_done=done)
    if exhausted:
        raise BudgetExhausted


def load_runs(path: str) -> Runs:
    """이전 실행 기록(jsonl)을 불러온다. --resume 으로 이어서 돌릴 때 쓴다."""
    runs: Runs = {}
    fields = set(Trial.__dataclass_fields__)
    with open(path, encoding="utf-8-sig") as f:      # 편집기가 붙인 BOM이 있어도 읽음
        for line in f:
            if line.strip():
                r = json.loads(line)
                runs.setdefault(r["sid"], {})[(r["task"], r["rep"])] = Trial(**{k: r[k] for k in fields})
    return runs


def restrict(runs: Runs, tasks: list[int]) -> Runs:
    """주어진 과제의 기록만 남긴 복사본."""
    ts = set(tasks)
    return {sid: {key: tr for key, tr in b.items() if key[0] in ts} for sid, b in runs.items()}


def summarize(runs: Runs, sid: str, tasks: list[int], k: int) -> Stat:
    trials = [runs[sid][(t, r)] for t in tasks for r in range(k)]
    ok = [x for x in trials if x.success]
    return Stat(
        succ=len(ok) / len(trials),
        cost=sum(x.tokens for x in ok) / len(ok) if ok else math.inf,
        steps=sum(x.steps for x in ok) / len(ok) if ok else math.inf,
        expand=sum(x.expansions for x in trials) / len(trials),
        n=len(trials),
    )


def dominates(a: Stat, b: Stat) -> bool:
    return a.succ >= b.succ and a.cost <= b.cost and (a.succ > b.succ or a.cost < b.cost)


def pareto_front(stats: dict[str, Stat]) -> list[str]:
    return [sid for sid, a in stats.items()
            if not any(dominates(b, a) for o, b in stats.items() if o != sid)]


def select_survivors(pool: list[Strategy], stats: dict[str, Stat], keep: int,
                     min_succ: float) -> list[Strategy]:
    """파레토 전선 우선, start_level별 1등 보존. 성공률이 min_succ 미만이면 전선에서 제외."""
    by_sid = {s.sid: s for s in pool}
    eligible = {sid: st for sid, st in stats.items() if st.succ >= min_succ} or stats
    front = set(pareto_front(eligible))
    ordered = sorted(stats, key=lambda sid: (sid not in front, -stats[sid].succ, stats[sid].cost))
    leaders: dict[int, str] = {}
    for sid in ordered:
        leaders.setdefault(by_sid[sid].start_level, sid)
    chosen = list(dict.fromkeys(list(leaders.values()) + ordered))[: max(keep, len(leaders))]
    return [by_sid[sid] for sid in chosen]


def paired_ci(runs: Runs, a: Strategy, b: Strategy, tasks: list[int], k: int,
              n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    def per_task(s: Strategy) -> list[float]:
        return [sum(runs[s.sid][(t, r)].success for r in range(k)) / k for t in tasks]

    diffs = [x - y for x, y in zip(per_task(a), per_task(b))]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


# ---------------------------------------------------------------- 교사

def make_sid(s: Strategy) -> str:
    return hashlib.md5(repr(s.key()).encode()).hexdigest()[:8]


RULE_POOL = (
    "Plan 1 gives only the outline.",
    "Mention every attribute in the instruction (color, size, price) explicitly.",
    "If no result matches, go to the next page before searching again.",
    "Check the selected options before buying.",
)

EDITABLE = {
    "start_level": [1, 2, 3],
    "trigger_n": [0, 2, 3, 4],
    "max_expand": [1, 2, 3, 4],
    "scope": ["global", "local", "hybrid"],
}


def apply_edit(p: Strategy, edit: dict) -> Strategy | None:
    """편집 하나를 적용한다. 형식이 틀리면 None."""
    op = edit.get("op")
    if op == "set" and edit.get("field") in EDITABLE and edit.get("value") in EDITABLE[edit["field"]]:
        c = replace(p, **{edit["field"]: edit["value"]})
    elif op == "add_rule" and isinstance(edit.get("rule"), str) and edit["rule"] not in p.rules:
        c = replace(p, rules=p.rules + (edit["rule"][:200],))
    elif op == "drop_rule" and edit.get("rule") in p.rules:
        c = replace(p, rules=tuple(r for r in p.rules if r != edit["rule"]))
    else:
        return None
    return replace(c, sid=make_sid(c))


class RandomEditTeacher:
    """테스트용 교사: 부모 하나에 무작위 편집 하나."""

    def propose(self, parents, n, rng, stats=None) -> list[Strategy]:
        out = []
        while len(out) < n:
            p = rng.choice(parents)
            kind = rng.choice(["set", "set", "add_rule", "drop_rule"])
            if kind == "set":
                f = rng.choice(list(EDITABLE))
                edit = {"op": "set", "field": f, "value": rng.choice(EDITABLE[f])}
            elif kind == "add_rule":
                edit = {"op": "add_rule", "rule": rng.choice(RULE_POOL)}
            else:
                edit = {"op": "drop_rule", "rule": rng.choice(p.rules) if p.rules else ""}
            c = apply_edit(p, edit)
            if c:
                out.append(c)
        return out


TEACHER_SYSTEM = (
    "You improve strategies for a hierarchical-planning web shopping agent. You see surviving "
    "strategies with their success rate, mean tokens and mean expansions. Find what the good "
    "ones have in common and propose edits. Reply with a JSON list only, each item: "
    '{"parent": "<sid>", "op": "set", "field": "<field>", "value": <value>} or '
    '{"parent": "<sid>", "op": "add_rule"|"drop_rule", "rule": "<text>"}. '
    f"Editable fields and allowed values: {json.dumps(EDITABLE)}."
)


class LLMTeacher:
    """LLM 교사: 생존 전략과 통계를 보여주고 편집 목록(JSON)을 받는다.
    형식이 틀린 편집은 버리고, 모자라면 무작위 편집으로 채운다."""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.fallback = RandomEditTeacher()

    def propose(self, parents, n, rng, stats=None) -> list[Strategy]:
        rows = [{**{k: v for k, v in asdict(p).items()},
                 **({"succ": round(stats[p.sid].succ, 3), "tokens": round(stats[p.sid].cost),
                     "expand": round(stats[p.sid].expand, 2)} if stats and p.sid in stats else {})}
                for p in parents]
        r = self.llm.chat("teacher", TEACHER_SYSTEM,
                          f"Strategies:\n{json.dumps(rows, ensure_ascii=False)}\nPropose {n} edits.",
                          temperature=0.7)
        by_sid = {p.sid: p for p in parents}
        out = []
        try:
            edits = json.loads(re.search(r"\[.*\]", r.text, re.S).group(0))
        except (AttributeError, json.JSONDecodeError):
            edits = []
        for e in edits if isinstance(edits, list) else []:
            if isinstance(e, dict) and e.get("parent") in by_sid:
                c = apply_edit(by_sid[e["parent"]], e)
                if c:
                    out.append(c)
        if len(out) < n:
            out += self.fallback.propose(parents, n - len(out), rng)
        return out[:n]


# ---------------------------------------------------------------- 탐색과 확인

@dataclass(frozen=True)
class SearchConfig:
    max_runs: int = 3000
    rounds: int = 3
    k0: int = 1
    keep_frac: float = 0.5
    children: int = 6
    min_succ: float = 0.0     # 이 성공률 미만은 파레토 전선에서 제외 (싸지만 실패만 하는 전략 방지)
    seed: int = 0


def run_search(runner, teacher, seeds, tasks, cfg: SearchConfig, log_fn=None, cache: Runs | None = None):
    """cache: 이전 실행 기록 (이어서 돌릴 때). 이미 있는 실행은 다시 하지 않고, 예산에서 미리 뺀다."""
    rng = random.Random(cfg.seed)
    budget = Budget(cfg.max_runs)
    runs: Runs = restrict(cache, tasks) if cache else {}
    budget.left -= sum(len(b) for b in runs.values())
    log: list[dict] = []
    pool, survivors = list(seeds), list(seeds)
    try:
        for r in range(cfg.rounds):
            k = cfg.k0 * 2**r
            ensure_runs(runs, runner, pool, tasks, k, budget, log_fn)    # 라운드의 전략 전체를 한 번에
            stats = {s.sid: summarize(runs, s.sid, tasks, k) for s in pool}
            survivors = select_survivors(pool, stats, max(2, int(len(pool) * cfg.keep_frac)),
                                         cfg.min_succ)
            log.append({"round": r, "k": k, "pool": len(pool), "budget_left": budget.left,
                        "front": pareto_front(stats),
                        "stats": {sid: asdict(st) for sid, st in stats.items()}})
            print(f"[round {r}] k={k} pool={len(pool)} budget_left={budget.left} "
                  f"top={[(s.sid, round(stats[s.sid].succ, 3)) for s in survivors[:3]]}", flush=True)
            if r < cfg.rounds - 1:
                children = teacher.propose(survivors, cfg.children, rng, stats)
                seen = {s.key() for s in pool}
                fresh = list({c.key(): c for c in children if c.key() not in seen}.values())
                pool = survivors + fresh
    except BudgetExhausted:
        log.append({"stopped": "budget_exhausted"})
        print("[search] 예산 소진으로 중단", flush=True)
    return survivors, runs, log


def confirm(runner, cands, baseline, tasks_select, tasks_test, k, budget, log_fn=None,
            cache: Runs | None = None) -> dict:
    """선택 과제로 후보 중 승자를 고르고, 테스트 과제에서 기준 전략과 짝지어 비교한다.
    기준 전략은 선택 과제에서 쓰이지 않으므로 돌리지 않는다."""
    runs: Runs = restrict(cache, tasks_select) if cache else {}
    ensure_runs(runs, runner, cands, tasks_select, k, budget, log_fn)
    stats = {s.sid: summarize(runs, s.sid, tasks_select, k) for s in cands}
    winner = max(cands, key=lambda s: (stats[s.sid].succ, -stats[s.sid].cost))
    print(f"[confirm] 선택 과제 승자: {winner.sid} succ={stats[winner.sid].succ:.3f}", flush=True)

    test: Runs = restrict(cache, tasks_test) if cache else {}
    ensure_runs(test, runner, [winner, baseline], tasks_test, k, budget, log_fn)
    lo, hi = paired_ci(test, winner, baseline, tasks_test, k)
    return {
        "winner": asdict(winner),
        "test_winner": asdict(summarize(test, winner.sid, tasks_test, k)),
        "test_baseline": asdict(summarize(test, baseline.sid, tasks_test, k)),
        "diff_ci95": (round(lo, 3), round(hi, 3)),
        "beats_baseline": lo > 0,
    }
