"""실행 진입점.

  python run.py episode    --session 1500            # 에피소드 하나 (환경·모델 연결 확인용)
  python run.py difficulty --tasks 1500:1600 --k 3    # 5단계 난이도 표 만들기
  python run.py search     --search 1500:1560 --select 1560:1590 --test 0:50
                                                      # 전략 탐색 + 기준 전략과 비교
  python run.py eval       --tasks 0:20 --k 3 --label after --trials-out after.jsonl
                                                      # 한 전략을 과제×반복으로 실행해 jsonl로 저장
                                                      # (report.py 로 바로 표·그래프)

--mock 을 붙이면 가짜 쇼핑몰 + 가짜 LLM으로 돈다 (설치 없이 코드 흐름 확인용).
실제 실행 예:
  OPENAI_BASE_URL=http://localhost:8000/v1 python run.py episode --session 1500 \
      --model meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from functools import partial

from agent import RunConfig, Strategy, run_episode
from difficulty import assign_tiers, informative_tasks, load_tiers, spearman
from search import (Budget, LLMTeacher, RandomEditTeacher, SearchConfig, TaskRunner, confirm,
                    load_runs, run_search, summarize)


def task_range(text: str) -> list[int]:
    a, b = text.split(":")
    return list(range(int(a), int(b)))


def build(args, need_env: bool = True):
    """need_env=False: 병렬 실행의 부모 프로세스용. 환경은 각 작업 프로세스가 만든다."""
    if args.mock:
        from envs import FakeShop
        from llm import ScriptedLLM
        env = FakeShop()
        factory = ScriptedLLM
        teacher_llm = None
    else:
        from envs import WebShopEnv
        env = WebShopEnv(num_products=args.num_products) if need_env else None
        if args.provider == "anthropic":
            from llm import AnthropicLLM
            llm = AnthropicLLM(args.model or "claude-haiku-5-5", effort=args.effort)
            teacher_llm = AnthropicLLM(args.teacher_model, effort=args.teacher_effort) \
                if args.teacher_model else None
        else:
            from llm import OpenAICompatLLM
            llm = OpenAICompatLLM(args.model or "meta-llama/Llama-3.1-8B-Instruct",
                                  base_url=args.base_url)
            teacher_llm = OpenAICompatLLM(args.teacher_model, base_url=args.teacher_base_url) \
                if args.teacher_model else None
        factory = lambda seed: llm          # noqa: E731  실제 LLM은 하나를 계속 재사용
    cfg = RunConfig(max_steps=args.max_steps, success_threshold=args.success_threshold,
                    act_temperature=args.temperature)
    return env, factory, teacher_llm, cfg


def build_worker(args):
    """병렬 실행용 프로세스가 자기 환경과 LLM을 만들 때 쓴다 (교사는 필요 없음)."""
    env, factory, _, cfg = build(args)
    return env, factory, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["episode", "difficulty", "search", "eval"])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic",
                    help="anthropic: Claude API / openai: vLLM 등 OpenAI 형식 서버")
    ap.add_argument("--model", default=None,
                    help="에피소드 실행용. 비우면 anthropic=claude-haiku-5-5(싸게), openai=Llama-3.1-8B-Instruct")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"],
                    help="Claude 생각 깊이 (에이전트 행동용)")
    ap.add_argument("--teacher-effort", default="high",
                    choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--teacher-model", default=None,
                    help="교사 모델. 비우면 anthropic=claude-sonnet-5-5, openai=무작위 편집 교사")
    ap.add_argument("--random-teacher", action="store_true", help="anthropic에서도 무작위 편집 교사 사용")
    ap.add_argument("--teacher-base-url", default=None)
    ap.add_argument("--num-products", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--success-threshold", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=1,
                    help="동시에 돌릴 에피소드 수 (프로세스 하나당 메모리 약 650MB)")
    # episode
    ap.add_argument("--session", type=int, default=1500)
    # difficulty
    ap.add_argument("--tasks", default="1500:1600")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="tiers.csv")
    # eval
    ap.add_argument("--label", default="demo", help="eval 결과의 sid 칸에 쓸 이름")
    ap.add_argument("--trials-out", default=None,
                    help="실행 기록 jsonl (기본: eval=eval_trials.jsonl, search=trials.jsonl)")
    # search
    ap.add_argument("--search", default="1500:1560")
    ap.add_argument("--select", default="1560:1590")
    ap.add_argument("--test", default="0:50")
    ap.add_argument("--max-runs", type=int, default=3000, help="탐색 실행 상한 (확인 단계는 별도로 같은 상한)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--confirm-k", type=int, default=2)
    # 예산 절약
    ap.add_argument("--tiers", default=None,
                    help="난이도 표(tiers.csv). 주면 search/select 과제에서 1·5단계를 줄인다")
    ap.add_argument("--edge-keep", type=int, default=10, help="1단계·5단계에서 각각 남길 과제 수")
    ap.add_argument("--no-race", action="store_true", help="중간 탈락(레이싱) 끄기")
    ap.add_argument("--race-chunk", type=int, default=5, help="과제 몇 개마다 중간 비교할지")
    ap.add_argument("--race-z", type=float, default=2.0, help="마지막 라운드 탈락 기준 (클수록 신중)")
    ap.add_argument("--race-z-early", type=float, default=1.5, help="앞 라운드 탈락 기준 (더 공격적)")
    ap.add_argument("--children", type=int, default=4, help="라운드마다 교사가 만드는 자식 전략 수")
    ap.add_argument("--confirm-top", type=int, default=3, help="선택 과제에서 확인할 상위 후보 수")
    ap.add_argument("--no-skip-test", action="store_true", help="승자가 기준 전략보다 낮아도 테스트를 돌린다")
    ap.add_argument("--seed", type=int, default=0, help="탐색 시드 (교사 편집의 무작위성)")
    ap.add_argument("--search-log", default="search_log.json")
    ap.add_argument("--resume", action="store_true",
                    help="search/eval: --trials-out 의 기존 기록을 불러와 이미 한 실행은 건너뛰고 이어서 기록")
    args = ap.parse_args()

    parallel = args.workers > 1 and args.cmd != "episode"
    env, factory, teacher_llm, cfg = build(args, need_env=not parallel)
    runner = TaskRunner(env, factory, cfg, workers=args.workers if parallel else 1,
                        build=partial(build_worker, args))

    if args.cmd == "episode":
        s = Strategy(sid="demo")
        print(json.dumps(asdict(run_episode(env, factory(0), s, args.session, cfg)), indent=2))

    elif args.cmd == "eval":
        # 기본 전략 하나를 과제×반복으로 실행. 끝나는 즉시 한 줄씩 써서 중간에 멈춰도 결과가 남는다.
        s = Strategy(sid=args.label)
        out = args.trials_out or "eval_trials.jsonl"
        done_before = load_runs(out).get(args.label, {}) if args.resume and os.path.exists(out) else {}
        jobs = [(s, t, rep) for t in task_range(args.tasks) for rep in range(args.k)
                if (t, rep) not in done_before]
        print(f"[eval {args.label}] 실행할 에피소드 {len(jobs)}개 (이미 있음 {len(done_before)}개)", flush=True)
        with open(out, "a" if args.resume else "w", encoding="utf-8") as f:
            def write(job, trial):
                _, t, rep = job
                f.write(json.dumps({"sid": args.label, "task": t, "rep": rep, **asdict(trial)}) + "\n")
                f.flush()
                print(f"[eval {args.label}] task {t} rep {rep}: success={trial.success} "
                      f"reward={trial.reward:.2f} steps={trial.steps} expand={trial.expansions} "
                      f"tokens={trial.tokens}", flush=True)
            runner.run_many(jobs, on_done=write)
        print(f"[eval] 저장: {out}")

    elif args.cmd == "difficulty":
        tasks = task_range(args.tasks)
        tiers = assign_tiers(runner, tasks, args.k, out_csv=args.out)
        if args.mock:   # 가짜 쇼핑몰은 숨은 난이도(세션 % 4)를 알고 있으므로 측정이 맞는지 확인
            rho = spearman([tiers[t] for t in tasks], [t % 4 for t in tasks])
            print(f"[difficulty] 숨은 난이도와의 순위 상관: {rho:.3f}")
        print(f"[difficulty] 저장: {args.out}")

    else:
        baseline = Strategy(sid="baseline", start_level=3, trigger_n=0)   # 원래 방식: 3레벨, 확장 없음
        seeds = [
            baseline,
            Strategy(sid="L1_hybrid", start_level=1, trigger_n=3, scope="hybrid"),
            Strategy(sid="L2_local", start_level=2, trigger_n=3, scope="local"),
            Strategy(sid="L1_global", start_level=1, trigger_n=2, scope="global"),
        ]
        teacher = LLMTeacher(teacher_llm) if teacher_llm else RandomEditTeacher()
        out = args.trials_out or "trials.jsonl"
        cache = load_runs(out) if args.resume and os.path.exists(out) else None
        if cache:
            print(f"[search] 이전 기록 {sum(len(b) for b in cache.values())}개를 불러와 이어서 실행", flush=True)
        log_file = open(out, "a" if args.resume else "w", encoding="utf-8")

        def log_fn(row):
            log_file.write(json.dumps(row) + "\n")
            log_file.flush()

        search_tasks, select_tasks = task_range(args.search), task_range(args.select)
        if args.tiers:      # 정보 없는 과제(항상 성공/항상 실패)를 줄여 같은 예산으로 더 많이 비교
            tiers = load_tiers(args.tiers)
            n0 = len(search_tasks) + len(select_tasks)
            search_tasks = informative_tasks(search_tasks, tiers, args.edge_keep)
            select_tasks = informative_tasks(select_tasks, tiers, args.edge_keep)
            print(f"[tiers] 과제 {n0} → {len(search_tasks) + len(select_tasks)}개 "
                  f"(탐색 {len(search_tasks)}, 선택 {len(select_tasks)}; 테스트는 전부 유지)", flush=True)

        scfg = SearchConfig(max_runs=args.max_runs, rounds=args.rounds, seed=args.seed,
                            race=not args.no_race, race_chunk=args.race_chunk, race_z=args.race_z,
                            race_z_early=args.race_z_early, children=args.children)
        survivors, search_runs, log = run_search(runner, teacher, seeds, search_tasks, scfg, log_fn,
                                       cache=cache, protect={baseline.sid})
        k_last = scfg.k0 * 2 ** (scfg.rounds - 1)
        search_stats = {sid: summarize(search_runs, sid, search_tasks, k_last) for sid in search_runs}
        result = confirm(runner, survivors, baseline, select_tasks,
                         task_range(args.test), args.confirm_k, Budget(args.max_runs), log_fn,
                         cache=cache, top=args.confirm_top, search_stats=search_stats,
                         skip_margin=9.0 if args.no_skip_test else 0.05)
        log_file.close()
        with open(args.search_log, "w", encoding="utf-8") as f:
            json.dump({"log": log, "result": result}, f, ensure_ascii=False, indent=2, default=str)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
