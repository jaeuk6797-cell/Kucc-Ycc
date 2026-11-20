"""ALFWorld에서 AdaPlan-H(M=3) 기준 vs M=3 + ADaPT식 확장을 비교한다.

  python run.py pilot   --n 3 --mock               # 어댑터/흐름 확인 (API 호출 없음)
  python run.py pilot   --n 5                      # 실제 모델로 에피소드당 비용 실측 + 견적
  python run.py compare --n 134                    # 두 전략을 같은 과제에 돌려 성공률 비교 (134 = unseen 전체)
  python run.py show --task 130 --sid adaplan_M3_adapt   # 저장된 한 에피소드의 행동 기록과 계획을 출력

설정: ../alfworld/pilot_config.json (M=3, K=1). 같은 과제의 반복 실행은 --reps로 늘린다
(Claude는 temperature를 바꿀 수 없어서 반복해도 결과가 달라질 수 있고, 그 흔들림을 재는 데 쓴다).
환경 준비: pip install alfworld (Python 3.11/3.12), alfworld-download, ALFWORLD_DATA 지정.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
import traceback
import contextlib
import io
import subprocess
import multiprocessing as mp
import atexit
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from collections import defaultdict
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
# 비교군(분리 실험). 기존 두 전략의 서명은 그대로라 쌓인 캐시를 계속 쓴다.
#  - M 스윕: 계획 깊이 M을 1/2/5로 바꾼 기준 전략 (M=3은 위 BASELINE)
#  - adaplan_M1_adapt: M을 대충(1) 정하고 막힐 때만 국소 분해 -> "M 튜닝 없이도 되는가"
#  - adaplan_M3_adapt_nohint: 미방문 장소 힌트를 뺀 ADaPT -> 힌트의 몫을 분리
EXTRA = [Strategy(sid=f"adaplan_M{m}", start_level=m, max_level=m, trigger_n=0) for m in (1, 2, 5)] + [
    Strategy(sid="adaplan_M1_adapt", start_level=1, max_level=1, trigger_n=3, max_expand=4, scope="local"),
    Strategy(sid="adaplan_M3_adapt_nohint", start_level=M, max_level=M, trigger_n=3, max_expand=4,
             scope="local", hint=False),
]
CODE_VERSION = "4"   # 에이전트 프롬프트/막힘 규칙을 바꾸면 올려서 예전 캐시를 무효화한다

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


# ---------------------------------------------------------------- 통계

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b: int, c: int) -> float:
    """짝 비교의 정확 이항 검정(양측). b, c = 서로 결과가 갈린 쌍의 수."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired_diff_ci(pairs: list[tuple[int, int]], iters: int = 5000, seed: int = 0):
    """(기준, ADaPT) 성공(0/1) 짝들에서 성공률 차이(ADaPT-기준)의 95% 부트스트랩 구간."""
    if not pairs:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(pairs)
    d = sum(p[1] - p[0] for p in pairs) / n
    boots = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(sum(pairs[i][1] - pairs[i][0] for i in idx) / n)
    boots.sort()
    return d, boots[int(0.025 * iters)], boots[int(0.975 * iters)]


# ---------------------------------------------------------------- 실행 보조

def make_env(args):
    if os.environ.get("ADAPLAN_FAKE_ENV"):                # 테스트용: ALFWorld 없이 흐름만 확인
        from test_core import FakeEnv
        return FakeEnv()
    from envs import ALFWorldEnv
    return ALFWorldEnv(split=args.split)


def make_llm_for(args):
    if args.mock:
        return lambda i: RandomLLM(i)                     # noqa: E731
    from llm import AnthropicLLM
    llm = AnthropicLLM(args.model, effort=args.effort)
    return lambda i: llm                                  # noqa: E731


def pick_tasks(num_games: int, n: int, seed: int) -> list[int]:
    """과제 순서가 유형별로 뭉쳐 있어서, 고정 시드로 무작위 추출해 유형이 섞이게 한다."""
    return sorted(random.Random(seed).sample(range(num_games), min(n, num_games)))


def signature(args, strat: Strategy) -> str:
    """캐시 키: 결과에 영향을 주는 설정이 하나라도 다르면 다른 결과로 본다."""
    raw = json.dumps({"model": args.model if not args.mock else "mock", "effort": args.effort,
                      "split": args.split, "max_steps": args.max_steps, "code": CODE_VERSION,
                      "strategy": strat.key()}, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()[:10]


def load_cache(path):
    cache, spent = {}, 0.0
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                row = json.loads(line)
                cache[(row["sid"], row["task"], row.get("rep", 0), row.get("sig"))] = row
                spent += row.get("cost_usd", 0.0)
    arch = archive_path(path)
    if arch and os.path.exists(arch):          # 보관함으로 옮긴 결과도 이미 쓴 비용이므로 예산 계산에는 포함
        for line in open(arch, encoding="utf-8"):
            if line.strip():
                spent += json.loads(line).get("cost_usd", 0.0)
    return cache, spent


def archive_path(path):
    return os.path.splitext(path)[0] + "_archive.jsonl" if path else None


def prune(args) -> None:
    """캐시에서 기준(adaplan_M3)과 ADaPT(adaplan_M3_adapt)만 남기고 나머지 행은 보관 파일로 옮긴다. 백업을 먼저 만들고, 비용은 보관 파일로 계속 센다."""
    path = args.cache
    if not os.path.exists(path):
        raise SystemExit(f"캐시 파일이 없습니다: {path}")
    rows = [l for l in open(path, encoding="utf-8") if l.strip()]
    keep, move = [], []
    for l in rows:
        (keep if json.loads(l)["sid"] in (BASELINE.sid, ADAPT.sid) else move).append(l)
    print(f"전체 {len(rows)}행 → 남김 {len(keep)}행(기준·ADaPT), 보관함으로 이동 {len(move)}행")
    if not move:
        return
    shutil.copy2(path, path + ".bak")
    with open(archive_path(path), "a", encoding="utf-8") as f:
        f.writelines(move)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(keep)
    os.replace(tmp, path)
    print(f"백업 {path}.bak, 보관 {archive_path(path)} (누적 비용 계산에는 계속 포함)")



class DiskFull(RuntimeError):
    pass


TMP_WORK: str | None = None      # ALFWorld가 임시 파일(libdownward.so 복사본 등)을 쌓는 전용 폴더


def use_private_tmp(prefix: str, root: str | None = None) -> str:
    """임시 파일을 이 프로세스 전용 폴더로 모으고, 종료 시 지운다. (ALFWorld는 환경을 만들 때마다 임시 폴더를 남긴다)"""
    global TMP_WORK
    if root:
        os.makedirs(root, exist_ok=True)
    d = tempfile.mkdtemp(prefix=prefix, dir=root or None)
    tempfile.tempdir = d
    os.environ["TMPDIR"] = d
    TMP_WORK = d
    atexit.register(shutil.rmtree, d, True)
    return d


def cleanup_tmp(keep: int = 1) -> None:
    """에피소드가 끝날 때마다 전용 폴더의 낡은 임시 폴더를 지운다 (가장 최근 keep개만 남김)."""
    if not TMP_WORK:
        return
    try:
        subs = sorted((os.path.join(TMP_WORK, x) for x in os.listdir(TMP_WORK) if x.startswith("tmp")),
                      key=os.path.getmtime)
        for p in subs[:-keep] if keep else subs:
            shutil.rmtree(p, ignore_errors=True)
    except OSError:
        pass


STRATS = {x.sid: x for x in (BASELINE, ADAPT, *EXTRA)}
_W: dict = {}          # 작업 프로세스가 쓰는 환경/LLM/설정


def make_row(args, strat, task, rep, tr, ttype):
    return {"sid": strat.sid, "task": task, "rep": rep, "sig": signature(args, strat), "ttype": ttype,
            "ts": time.time(), **asdict(tr)}


def trial_from_row(r) -> Trial:
    return Trial(**{k: r[k] for k in Trial.__dataclass_fields__ if k in r})


def run_job(env, llm_for, cfg, args, t, rep, sids):
    """과제 하나(같은 rep)에서 전략들을 연달아 돌린다. (sid, row 또는 None, 오류 문자열 또는 None) 목록을 돌려준다."""
    out = []
    for sid in sids:
        s = STRATS[sid]
        try:
            tr = run_episode(env, llm_for(t * 1000 + rep), s, t, cfg)
            out.append((sid, make_row(args, s, t, rep, tr, env.task_type(t)), None))
        except BudgetExceeded:
            raise
        except Exception as ex:          # API 오류 등: 이 에피소드만 건너뜀 (캐시에 안 남아 다음 실행에서 재시도)
            out.append((sid, None, f"{type(ex).__name__}: {ex}"))
        finally:
            cleanup_tmp()
    return t, rep, out


def worker_init(args, shared):
    """작업 프로세스 시작 시 한 번: 자기만의 임시 작업 폴더에서 환경을 만든다.
    (ALFWorld가 환경을 만들 때 외부 플래너가 작업 폴더에 임시 파일을 쓰므로, 폴더를 나눠야 서로 안 부딪힌다.)"""
    import envs  # noqa: F401  (폴더를 옮기기 전에 import)
    if os.environ.get("ADAPLAN_FAKE_ENV"):
        import test_core  # noqa: F401
    os.chdir(use_private_tmp("alfw_", getattr(args, "tmp_root", None)))
    METER.shared = shared
    METER.budget_usd, METER.stop_frac = args.budget, args.stop_frac
    METER.log_path = None if args.mock else args.cost_log
    _W.update(env=make_env(args), llm_for=make_llm_for(args), cfg=RunConfig(max_steps=args.max_steps), args=args)


def proc_job(t, rep, sids):
    return run_job(_W["env"], _W["llm_for"], _W["cfg"], _W["args"], t, rep, sids)


def show(args) -> None:
    cache, _ = load_cache(args.cache)
    rows = [r for (sid, t, rep, _), r in cache.items() if t == args.task and (not args.sid or sid == args.sid)]
    if not rows:
        print("해당 기록이 없습니다.")
    for r in sorted(rows, key=lambda r: (r["sid"], r.get("rep", 0), r.get("ts", 0))):
        print(f"\n=== {r['sid']} task {r['task']} rep {r.get('rep', 0)} 유형={r.get('ttype')} "
              f"성공={r['success']} 스텝={r['steps']} 확장={r['expansions']} 비용=${r['cost_usd']:.4f}")
        for e in r.get("traj", []):
            if "expand" in e:
                print(f"  -- t={e['t']}: 계획 확장({e['expand']}), 막힌 단계 {e['plan_step']}")
            else:
                flag = " [막힘]" if e["stuck"] else ""
                print(f"  t={e['t']:>2} (plan {e['plan_step']}) {e['action']}{'' if e['valid'] else ' [무효]'}{flag} -> {e['obs']}")
        print("  최종 계획:\n" + "\n".join("    " + l for l in r.get("plan", "").splitlines()))


def summarize(args, tasks, strategies, res, ttypes) -> None:
    keys = [(t, rep) for t in tasks for rep in range(args.reps)]
    print(f"\n과제 {len(tasks)}개 x 반복 {args.reps} (split={args.split}, seed={args.seed}), M={M}, K=1")
    for s in strategies:
        rs = [res[s.sid][k] for k in keys if k in res[s.sid]]
        if not rs:
            continue
        k = sum(r.success for r in rs)
        lo, hi = wilson(k, len(rs))
        cost = sum(r.cost_usd for r in rs)
        print(f"{s.sid}: 성공 {k}/{len(rs)} = {k/len(rs):.0%} (95% 윌슨 {lo:.0%}–{hi:.0%}), "
              f"평균 스텝 {sum(r.steps for r in rs)/len(rs):.1f}, 평균 확장 {sum(r.expansions for r in rs)/len(rs):.2f}, "
              f"에피소드당 ${cost/len(rs):.4f}, 성공당 ${cost/k if k else float('inf'):.4f}")
    if len(strategies) >= 2 and strategies[0] is BASELINE and strategies[1] is ADAPT:
        both = [k for k in keys if k in res[BASELINE.sid] and k in res[ADAPT.sid]]
        pairs = [(int(res[BASELINE.sid][k].success), int(res[ADAPT.sid][k].success)) for k in both]
        b = sum(1 for p in pairs if p == (0, 1))
        c = sum(1 for p in pairs if p == (1, 0))
        d, lo, hi = paired_diff_ci(pairs)
        print(f"\n짝 비교({len(both)}쌍): ADaPT만 성공 {b}, 기준만 성공 {c}, 정확검정(McNemar) p={mcnemar_exact(b, c):.3f}")
        print(f"성공률 차이(ADaPT-기준) {d:+.1%} (95% 부트스트랩 {lo:+.1%} ~ {hi:+.1%})")
        # 과제 유형별
        by = defaultdict(list)
        for k, p in zip(both, pairs):
            by[ttypes.get(k[0], "?")].append(p)
        print("\n유형별 (기준 성공 / ADaPT 성공 / 쌍 수):")
        for tt, ps in sorted(by.items()):
            print(f"  {tt:<28} {sum(p[0] for p in ps):>3} / {sum(p[1] for p in ps):>3} / {len(ps)}")
        exp = [res[ADAPT.sid][k].expansions for k in both]
        fired = sum(1 for e in exp if e > 0)
        print(f"\nADaPT 확장이 한 번이라도 발동한 에피소드: {fired}/{len(exp)}")
        # 확장 발동 여부별 분해. 주의: 확장 여부는 ADaPT 실행이 막혔는지에 따라 정해지는 결과 변수라서,
        # "확장 없음" 구간은 ADaPT가 잘 풀린 실행만 남는 선택 편향이 있다. 잡음 크기나 효과 추정으로 쓸 수 없고 참고용이다.
        # 잡음 크기는 같은 전략을 두 번 돌린 결과(아래 '같은 전략 재실행 일치도')로 잰다.
        print("\n확장 발동 여부별 (쌍 수 / 기준 성공 / ADaPT 성공 / ADaPT만 성공 / 기준만 성공):")
        for label, flag in (("확장 발동   ", True), ("확장 발동 안 함", False)):
            ks = [k for k in both if (res[ADAPT.sid][k].expansions > 0) == flag]
            bs = [int(res[BASELINE.sid][k].success) for k in ks]
            as_ = [int(res[ADAPT.sid][k].success) for k in ks]
            only_a = sum(1 for x, y in zip(bs, as_) if (x, y) == (0, 1))
            only_b = sum(1 for x, y in zip(bs, as_) if (x, y) == (1, 0))
            print(f"  {label:<10} {len(ks):>4} / {sum(bs):>4} / {sum(as_):>4} / {only_a:>3} / {only_b:>3}")
    for s in strategies:                      # 같은 전략을 같은 과제에 두 번 돌렸을 때 결과가 얼마나 갈리는가 = 실행 간 잡음
        rp = [(int(res[s.sid][(t, 0)].success), int(res[s.sid][(t, 1)].success))
              for t in tasks if (t, 0) in res[s.sid] and (t, 1) in res[s.sid]]
        if len(rp) >= 10:
            diff = sum(1 for a, b in rp if a != b)
            print(f"\n같은 전략 재실행 일치도 {s.sid}: {len(rp)}개 과제 중 두 실행의 성공 여부가 다른 것 {diff}개 ({diff/len(rp):.0%})"
                  f" [1회차만 성공 {sum(1 for a, b in rp if (a, b) == (1, 0))}, 2회차만 성공 {sum(1 for a, b in rp if (a, b) == (0, 1))}]")
    if args.cmd == "pilot" and res[BASELINE.sid]:
        rs = list(res[BASELINE.sid].values())
        mean = sum(r.cost_usd for r in rs) / len(rs)
        usable = args.budget * args.stop_frac
        if mean:
            print(f"예산 ${args.budget}의 {args.stop_frac:.0%}(${usable:.2f}) 내 약 {int(usable/mean)} 에피소드 가능")


def save_report(args, text: str, extra: str = "") -> str | None:
    """요약을 results/ 아래 마크다운으로 저장하고, --push면 커밋·푸시를 시도한다 (실패해도 실험은 영향 없음)."""
    if args.mock:
        return None
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(getattr(args, "out_dir", None) or here, "results")
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{args.cmd}_{stamp}_n{args.n}_r{args.reps}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {args.cmd} 결과 ({stamp})\n\n"
                f"명령: n={args.n} reps={args.reps} seed={args.seed} split={args.split} model={args.model}\n\n"
                f"```\n{text.strip()}\n```\n{extra}")
    print(f"[저장] {path}")
    if getattr(args, "push", False):
        try:
            rel = os.path.relpath(path, here)
            cache_rel = os.path.relpath(os.path.abspath(args.cache), here)
            targets = [rel] + ([cache_rel] if os.path.exists(os.path.abspath(args.cache)) else [])
            subprocess.run(["git", "add", *targets], cwd=here, check=True, timeout=60)
            subprocess.run(["git", "commit", "-m", f"results: {args.cmd} n={args.n} reps={args.reps} ({stamp})"],
                           cwd=here, check=True, timeout=60)
            subprocess.run(["git", "push"], cwd=here, check=True, timeout=120)
            print("[푸시] 완료")
        except Exception as e:  # 인증 실패 등. 파일은 남아 있으니 나중에 수동 push
            print(f"[푸시 실패] {type(e).__name__}: {e}  (결과 파일은 저장됨)")
    return path


def extra_arms(args) -> list:
    if not args.arms:
        return []
    names = [x.sid for x in EXTRA] if args.arms == "all" else [x.strip() for x in args.arms.split(",") if x.strip()]
    bad = [n for n in names if n not in STRATS]
    if bad:
        raise SystemExit(f"알 수 없는 전략: {bad}. 가능: {[x.sid for x in EXTRA]}")
    return [STRATS[n] for n in names if n not in (BASELINE.sid, ADAPT.sid)]


def task_boot(diffs: dict, iters: int = 5000, seed: int = 0):
    """과제별 평균 차이의 과제 단위 부트스트랩 (같은 과제의 반복은 묶어서 한 단위로 본다)."""
    tasks = sorted(diffs)
    means = [sum(diffs[t]) / len(diffs[t]) for t in tasks]
    rng = random.Random(seed)
    bs = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(iters))
    return sum(means) / len(means), bs[int(0.025 * iters)], bs[int(0.975 * iters) - 1]


def ablate(args) -> None:
    """API 호출 없이 캐시에서 모든 전략을 기준(adaplan_M3)·ADaPT와 과제 단위로 비교한다."""
    cache, _ = load_cache(args.cache)
    arms = [BASELINE, ADAPT] + [x for x in EXTRA if not args.arms or args.arms == "all" or x.sid in args.arms.split(",") or x.sid == "adaplan_M1"]
    sigs = {signature(args, s): s for s in arms}
    res = {s.sid: {} for s in arms}
    for (sid, t, rep, sig), r in cache.items():
        if sig in sigs and sigs[sig].sid == sid:
            res[sid][(t, rep)] = int(r["success"])
    done = [k for v in res.values() for k in v]
    if done:
        args.n, args.reps = len({t for t, _ in done}), max(rep for _, rep in done) + 1     # 결과 파일 이름용
    lines = [f"\n전략 비교 (split={args.split}, 과제 단위 평균, 과제 부트스트랩 95%)",
             f"{'전략':<26}{'에피소드':>8}{'성공률':>8}   {'vs adaplan_M3':<38}{'vs adaplan_M3_adapt':<38}vs adaplan_M1"]
    def cmp(a, b):
        ks = sorted(set(res[a]) & set(res[b]))
        diffs = defaultdict(list)
        for t, rep in ks:
            diffs[t].append(res[a][(t, rep)] - res[b][(t, rep)])
        if len(diffs) < 5:
            return "      -"
        d, lo, hi = task_boot(diffs)
        return f"{d:+6.1%}p [{lo:+.1%}, {hi:+.1%}] n={len(ks)}"
    for s in arms:
        rs = res[s.sid]
        if not rs:
            lines.append(f"{s.sid:<26}{0:>8}")
            continue
        rate = sum(rs.values()) / len(rs)
        base = "" if s is BASELINE else cmp(s.sid, BASELINE.sid)
        adp = "" if s is ADAPT else cmp(s.sid, ADAPT.sid)
        m1 = "" if s.sid == "adaplan_M1" else cmp(s.sid, "adaplan_M1")
        lines.append(f"{s.sid:<26}{len(rs):>8}{rate:>8.0%}   {base:<38}{adp:<38}{m1}")
    text = "\n".join(lines) + "\n"
    print(text)
    save_report(args, text, "")


def report(args) -> None:
    """API 호출 없이 results_cache.jsonl에서 현재 코드 버전의 결과를 다시 요약한다 (중단된 실행, 이어서 한 실행 포함)."""
    cache, _ = load_cache(args.cache)
    strategies = [BASELINE, ADAPT]
    sigs = {signature(args, s): s for s in strategies}
    res: dict[str, dict[tuple[int, int], Trial]] = {s.sid: {} for s in strategies}
    ttypes: dict[int, str] = {}
    for (sid, t, rep, sig), r in cache.items():
        if sig in sigs and sigs[sig].sid == sid:
            res[sid][(t, rep)] = Trial(**{k: r[k] for k in Trial.__dataclass_fields__ if k in r})
            ttypes[t] = r.get("ttype", "?")
    keys = sorted(set(res[BASELINE.sid]) & set(res[ADAPT.sid]))       # 두 전략 모두 끝난 (task, rep)만
    if not keys:
        print("현재 코드 버전(CODE_VERSION=%s)·설정의 결과가 캐시에 없습니다." % CODE_VERSION)
        return
    args.n = len({t for t, _ in keys})
    args.reps = max(rep for _, rep in keys) + 1
    for sid in res:
        res[sid] = {k: v for k, v in res[sid].items() if k in set(keys)}
    tasks = sorted({t for t, _ in keys})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summarize(args, tasks, strategies, res, ttypes)
    text = buf.getvalue()
    print(text, end="")
    save_report(args, text)


def table(args) -> None:
    """과제별 전체 표 (API 호출 없음). 화면에 출력하고 results/table_*.csv로 저장한다. O=성공 X=실패 -=아직 없음."""
    import csv
    cache, _ = load_cache(args.cache)
    sigs = {signature(args, s): s.sid for s in (BASELINE, ADAPT)}
    R: dict[tuple[str, int, int], dict] = {}
    for (sid, t, rep, sig), r in cache.items():
        if sigs.get(sig) == sid:
            R[(sid, t, rep)] = r
    tasks = sorted({t for (_, t, _) in R})
    if not tasks:
        print("현재 코드 버전의 결과가 캐시에 없습니다.")
        return
    reps = max(rep for (_, _, rep) in R) + 1
    mark = lambda sid, t, rep: "-" if (sid, t, rep) not in R else ("O" if R[(sid, t, rep)]["success"] else "X")   # noqa: E731
    def rate(sid, t):
        v = [int(R[(sid, t, rep)]["success"]) for rep in range(reps) if (sid, t, rep) in R]
        return sum(v) / len(v) if v else None
    head = ["task", "유형"] + [f"기준{r + 1}" for r in range(reps)] + [f"ADaPT{r + 1}" for r in range(reps)] + \
           [f"확장{r + 1}" for r in range(reps)] + ["기준평균", "ADaPT평균", "차이"]
    rows = []
    for t in tasks:
        tt = next((R[k].get("ttype") for k in R if k[1] == t), "?")
        rb, ra = rate(BASELINE.sid, t), rate(ADAPT.sid, t)
        exp = [str(R[(ADAPT.sid, t, rep)]["expansions"]) if (ADAPT.sid, t, rep) in R else "-" for rep in range(reps)]
        rows.append([t, tt] + [mark(BASELINE.sid, t, r) for r in range(reps)] + [mark(ADAPT.sid, t, r) for r in range(reps)] + exp +
                    ["" if rb is None else f"{rb:.2f}", "" if ra is None else f"{ra:.2f}",
                     "" if rb is None or ra is None else f"{ra - rb:+.2f}"])
    widths = [max(len(str(x)) for x in col) for col in zip(head, *rows)]
    line = lambda r: "  ".join(str(x).ljust(w) for x, w in zip(r, widths))                                     # noqa: E731
    print(line(head))
    for r in rows:
        print(line(r))
    both = [(rate(BASELINE.sid, t), rate(ADAPT.sid, t)) for t in tasks]
    both = [(b, a) for b, a in both if b is not None and a is not None]
    if both:
        print(f"\n과제 {len(both)}개: 기준 평균 {sum(b for b, _ in both) / len(both):.1%}, "
              f"ADaPT 평균 {sum(a for _, a in both) / len(both):.1%}, "
              f"ADaPT가 더 나은 과제 {sum(a > b for b, a in both)}개, 같은 과제 {sum(a == b for b, a in both)}개, "
              f"기준이 더 나은 과제 {sum(a < b for b, a in both)}개")
    out_dir = os.path.join(args.out_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:      # utf-8-sig: 엑셀에서 한글이 안 깨지게
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)
    print(f"[저장] {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pilot", "compare", "show", "report", "table", "ablate", "prune"])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model", default="claude-haiku-5-5")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--split", default="eval_out_of_distribution",
                    help="eval_out_of_distribution(unseen 134) / eval_in_distribution(seen 140)")
    ap.add_argument("--n", type=int, default=5, help="과제 수")
    ap.add_argument("--reps", type=int, default=1, help="같은 과제의 반복 횟수 (모델 비결정성 측정용)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=40)
    ap.add_argument("--budget", type=float, default=CFG["budget_usd"])
    ap.add_argument("--stop-frac", type=float, default=CFG["stop_frac"])
    ap.add_argument("--cost-log", default=None, help="기본: <out-dir>/cost_calls.jsonl")
    ap.add_argument("--cache", default=None, help="기본: <out-dir>/results_cache.jsonl")
    ap.add_argument("--out-dir", default=os.environ.get("ADAPLAN_OUT"),
                    help="결과(results/), 캐시, 비용 로그를 모을 폴더. 환경변수 ADAPLAN_OUT으로도 지정 (예: /mnt/d/kucc)")
    ap.add_argument("--tmp-root", default=os.environ.get("ADAPLAN_TMP"),
                    help="ALFWorld 임시 파일을 쌓을 폴더. 환경변수 ADAPLAN_TMP으로도 지정 (예: /mnt/d/kucc_tmp)")
    ap.add_argument("--arms", default=None,
                    help="compare/ablate: 추가로 돌릴 전략 이름(쉼표). 예: adaplan_M1,adaplan_M2. 'all'이면 비교군 전부")
    ap.add_argument("--workers", type=int, default=1, help="동시에 돌릴 작업 프로세스 수 (권장 3, API 속도 제한 주의)")
    ap.add_argument("--push", action="store_true", help="결과 파일을 git commit + push (실패해도 무시)")
    ap.add_argument("--task", type=int, default=None, help="show: 과제 번호")
    ap.add_argument("--sid", default=None, help="show: 전략 이름")
    args = ap.parse_args()
    out = os.path.abspath(args.out_dir) if args.out_dir else HERE
    if args.out_dir:
        os.makedirs(out, exist_ok=True)
    args.out_dir = out
    args.cache = os.path.abspath(args.cache or os.path.join(out, "results_cache.jsonl"))
    args.cost_log = os.path.abspath(args.cost_log or os.path.join(out, "cost_calls.jsonl"))
    if args.tmp_root:
        args.tmp_root = os.path.abspath(args.tmp_root)

    if args.cmd == "show":
        show(args)
        return
    if args.cmd == "report":
        report(args)
        return
    if args.cmd == "table":
        table(args)
        return
    if args.cmd == "ablate":
        ablate(args)
        return
    if args.cmd == "prune":
        prune(args)
        return

    cache_path = None if args.mock else args.cache
    cache, spent = load_cache(cache_path)
    METER.budget_usd, METER.stop_frac = args.budget, args.stop_frac
    METER.cost = spent                      # 재시작해도 지금까지 쓴 비용을 이어서 센다
    METER.log_path = None if args.mock else args.cost_log
    if args.workers <= 1:
        use_private_tmp("alfw_main_", getattr(args, "tmp_root", None))
    env = make_env(args)
    llm_for = make_llm_for(args)
    cfg = RunConfig(max_steps=args.max_steps)
    tasks = pick_tasks(env.num_games, args.n, args.seed)
    ttypes = {t: env.task_type(t) for t in tasks}
    strategies = [BASELINE] if args.cmd == "pilot" else [BASELINE, ADAPT] + extra_arms(args)
    res: dict[str, dict[tuple[int, int], Trial]] = {s.sid: {} for s in strategies}
    total, done_n, failed, t_start = len(tasks) * args.reps * len(strategies), 0, [], time.time()
    if spent:
        print(f"[예산] 이전 실행까지 누적 ${spent:.4f} 사용 (한도 ${args.budget * args.stop_frac:.2f})")
    shared = mp.get_context("spawn").Value("d", spent) if args.workers > 1 else None
    cur_cost = lambda: shared.value if shared is not None else METER.cost      # noqa: E731

    def show_done(sid, t, rep, tr, cached):
        nonlocal done_n
        res[sid][(t, rep)] = tr
        done_n += 1
        eta = (time.time() - t_start) / max(done_n, 1) * (total - done_n)
        print(f"[{done_n}/{total}{' 캐시' if cached else ''}] {sid} task {t}({ttypes[t]}): "
              f"성공={tr.success} 스텝={tr.steps} 확장={tr.expansions} "
              f"입력={tr.in_tokens:,} 출력={tr.out_tokens:,} 비용=${tr.cost_usd:.4f} "
              f"| 누적 ${cur_cost():.3f} | 남은 약 {eta/60:.0f}분", flush=True)

    def handle(t, rep, out):
        for sid, row, err in out:
            if err is not None:
                if "No space left" in err or "AuthenticationError" in err or "PermissionDenied" in err:
                    raise DiskFull(err)            # 디스크 부족·키 오류는 나머지도 전부 실패하므로 즉시 멈춘다
                failed.append((sid, t, rep))
                print(f"[오류] {sid} task {t} rep {rep}: {err}", flush=True)
                continue
            if cache_path:
                with open(cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            cache[(sid, t, rep, row["sig"])] = row
            show_done(sid, t, rep, trial_from_row(row), False)

    pending = []                              # 캐시에 없는 (과제, 반복, 전략들)
    for t in tasks:
        for rep in range(args.reps):
            todo = []
            for s in strategies:
                hit = cache.get((s.sid, t, rep, signature(args, s)))
                if hit is not None:
                    show_done(s.sid, t, rep, trial_from_row(hit), True)
                else:
                    todo.append(s.sid)
            if todo:
                pending.append((t, rep, todo))

    ex = None
    try:
        if args.workers <= 1:
            for t, rep, sids in pending:
                handle(*run_job(env, llm_for, cfg, args, t, rep, sids))
        else:
            ex = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                     initializer=worker_init, initargs=(args, shared))
            futs = [ex.submit(proc_job, t, rep, sids) for t, rep, sids in pending]
            for fut in as_completed(futs):
                handle(*fut.result())          # BudgetExceeded는 여기서 올라온다
    except BudgetExceeded as e:
        print(f"[예산 보호] 중단: {e}\n같은 명령을 다시 실행하면 캐시된 결과는 건너뛰고 이어서 진행합니다.")
    except DiskFull as e:
        print(f"[치명적 오류] 중단: {e}\n디스크 공간(No space) 또는 API 키(Authentication) 문제입니다. 해결한 뒤 같은 명령을 다시 실행하면 이어서 진행합니다.")
    finally:
        if ex is not None:
            ex.shutdown(wait=True, cancel_futures=True)
        print("[비용]", METER.summary() if shared is None else f"누적 ${cur_cost():.4f} (병렬 {args.workers}개)")
    if failed:
        print(f"[주의] 오류로 건너뛴 에피소드 {len(failed)}개 (같은 명령으로 다시 실행하면 재시도): {failed[:5]}")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summarize(args, tasks, strategies, res, ttypes)
    text = buf.getvalue()
    print(text, end="")
    save_report(args, text, "\n비용: " + (METER.summary() if shared is None else f"누적 ${cur_cost():.4f}") + "\n")


if __name__ == "__main__":
    main()
