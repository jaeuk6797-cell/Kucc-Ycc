"""실행 진입점.

  python run.py episode    --session 1500            # 에피소드 하나 (환경·모델 연결 확인용)
  python run.py difficulty --tasks 1500:1600 --k 3    # 5단계 난이도 표 만들기
  python run.py search     --search 1500:1560 --select 1560:1590 --test 0:50
                                                      # 전략 탐색 + 기준 전략과 비교

--mock 을 붙이면 가짜 쇼핑몰 + 가짜 LLM으로 돈다 (설치 없이 코드 흐름 확인용).
실제 실행 예:
  OPENAI_BASE_URL=http://localhost:8000/v1 python run.py episode --session 1500 \
      --model meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from agent import RunConfig, Strategy, run_episode
from difficulty import assign_tiers, spearman
from search import (Budget, LLMTeacher, RandomEditTeacher, SearchConfig, TaskRunner, confirm,
                    run_search)


def task_range(text: str) -> list[int]:
    a, b = text.split(":")
    return list(range(int(a), int(b)))


def build(args):
    if args.mock:
        from envs import FakeShop
        from llm import ScriptedLLM
        env = FakeShop()
        factory = ScriptedLLM
        teacher_llm = None
    else:
        from envs import WebShopEnv
        from llm import OpenAICompatLLM
        env = WebShopEnv(num_products=args.num_products)
        llm = OpenAICompatLLM(args.model, base_url=args.base_url)
        factory = lambda seed: llm          # noqa: E731  실제 LLM은 하나를 계속 재사용
        teacher_llm = OpenAICompatLLM(args.teacher_model, base_url=args.teacher_base_url) \
            if args.teacher_model else None
    cfg = RunConfig(max_steps=args.max_steps, success_threshold=args.success_threshold,
                    act_temperature=args.temperature)
    return env, factory, teacher_llm, cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["episode", "difficulty", "search"])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--teacher-model", default=None, help="비우면 무작위 편집 교사 사용")
    ap.add_argument("--teacher-base-url", default=None)
    ap.add_argument("--num-products", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--success-threshold", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.7)
    # episode
    ap.add_argument("--session", type=int, default=1500)
    # difficulty
    ap.add_argument("--tasks", default="1500:1600")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="tiers.csv")
    # search
    ap.add_argument("--search", default="1500:1560")
    ap.add_argument("--select", default="1560:1590")
    ap.add_argument("--test", default="0:50")
    ap.add_argument("--max-runs", type=int, default=3000)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--confirm-k", type=int, default=2)
    args = ap.parse_args()

    env, factory, teacher_llm, cfg = build(args)
    runner = TaskRunner(env, factory, cfg)

    if args.cmd == "episode":
        s = Strategy(sid="demo")
        print(json.dumps(asdict(run_episode(env, factory(0), s, args.session, cfg)), indent=2))

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
        log_file = open("trials.jsonl", "w", encoding="utf-8")
        log_fn = lambda row: log_file.write(json.dumps(row) + "\n")   # noqa: E731

        scfg = SearchConfig(max_runs=args.max_runs, rounds=args.rounds)
        survivors, _, log = run_search(runner, teacher, seeds, task_range(args.search), scfg, log_fn)
        result = confirm(runner, survivors, baseline, task_range(args.select),
                         task_range(args.test), args.confirm_k, Budget(args.max_runs), log_fn)
        log_file.close()
        with open("search_log.json", "w", encoding="utf-8") as f:
            json.dump({"log": log, "result": result}, f, ensure_ascii=False, indent=2, default=str)
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
