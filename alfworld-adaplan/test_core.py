"""ALFWorld 없이 돌아가는 핵심 로직 테스트.  python test_core.py"""

from types import SimpleNamespace

import run
from agent import RunConfig, Strategy, is_valid, parse_action, run_episode
from llm import Reply


def test_parse_and_valid():
    assert parse_action("Step: 2\nAction: go to desk 1") == (2, "go to desk 1")
    assert parse_action("nonsense") == (1, None)
    assert is_valid("Go To Desk 1", {"commands": ["go to desk 1"]})
    assert not is_valid("take mug", {"commands": ["go to desk 1"]})


def test_stats():
    lo, hi = run.wilson(3, 5)
    assert 0.2 < lo < 0.3 and 0.85 < hi < 0.9
    assert run.mcnemar_exact(0, 0) == 1.0
    assert abs(run.mcnemar_exact(0, 5) - 0.0625) < 1e-9      # 양측 정확검정
    assert run.mcnemar_exact(3, 3) == 1.0
    d, lo, hi = run.paired_diff_ci([(0, 1)] * 10)
    assert d == 1.0 and lo == 1.0 and hi == 1.0


def test_signature_changes_with_settings():
    a = SimpleNamespace(model="m", effort="low", split="s", max_steps=40, mock=False)
    b = SimpleNamespace(model="m", effort="low", split="s", max_steps=50, mock=False)
    assert run.signature(a, run.BASELINE) == run.signature(a, run.BASELINE)
    assert run.signature(a, run.BASELINE) != run.signature(b, run.BASELINE)
    assert run.signature(a, run.BASELINE) != run.signature(a, run.ADAPT)


class FakeEnv:
    def __init__(self):
        self.t = 0

    def reset(self, idx):
        self.t = 0
        return "You are in a room.", "put a mug in cabinet"

    def available(self):
        return {"commands": ["look", "go to cabinet 1"]}

    def step(self, a):
        self.t += 1
        return "Nothing happens.", 0.0, False

    def task_type(self, idx):
        return "fake"


class StuckLLM:
    def chat(self, role, system, user, temperature=0.0):
        if role in ("plan", "refine"):
            return Reply("Step 1: a\nStep 2: b", 5)
        return Reply("Step: 1\nAction: look", 5)


def test_expansion_and_trace():
    tr = run_episode(FakeEnv(), StuckLLM(), run.ADAPT, 0, RunConfig(max_steps=30))
    assert tr.expansions == 4 and not tr.success
    assert any("expand" in e for e in tr.traj) and tr.plan
    base = run_episode(FakeEnv(), StuckLLM(), run.BASELINE, 0, RunConfig(max_steps=30))
    assert base.expansions == 0 and base.steps == 30


class RecordLLM(StuckLLM):
    def __init__(self):
        self.refine_prompts = []

    def chat(self, role, system, user, temperature=0.0):
        if role == "refine":
            self.refine_prompts.append(user)
        return super().chat(role, system, user, temperature)


def test_refine_note_lists_unvisited_places():
    class Env(FakeEnv):
        def available(self):
            return {"commands": ["look", "go to cabinet 1", "go to desk 2"]}
    llm = RecordLLM()
    run_episode(Env(), llm, run.ADAPT, 0, RunConfig(max_steps=12))
    assert llm.refine_prompts, "막혀서 분해가 한 번은 불려야 함"
    assert "Not yet visited: cabinet 1, desk 2" in llm.refine_prompts[0]


def test_meter_per_thread_and_total():
    import threading
    from cost import CostMeter
    m = CostMeter(budget_usd=100)
    got = {}

    def work(i):
        b = m.snapshot()
        for _ in range(50):
            m.add("claude-haiku-5-5", "act", 1000, 100)
        got[i] = m.delta(b)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    one = 50 * (1000 * 0.10 + 100 * 0.50) / 1e6
    assert all(abs(d[2] - one) < 1e-9 and d[0] == 50_000 and d[1] == 5_000 for d in got.values())
    assert m.calls == 200 and abs(m.cost - 4 * one) < 1e-9


def test_parallel_runs_match_serial():
    from concurrent.futures import ThreadPoolExecutor
    class Env(FakeEnv):
        pass
    def one(i):
        return run_episode(Env(), StuckLLM(), run.ADAPT, i, RunConfig(max_steps=20)).steps
    serial = [one(i) for i in range(6)]
    with ThreadPoolExecutor(3) as ex:
        par = list(ex.map(one, range(6)))
    assert serial == par


def test_process_pool_matches_serial():
    """작업 프로세스(spawn, 각자 임시 폴더)로 돌린 결과가 한 프로세스로 돌린 결과와 같다."""
    import os
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    os.environ["ADAPLAN_FAKE_ENV"] = "1"
    args = SimpleNamespace(mock=True, model="m", effort="low", split="s", max_steps=15, budget=27.0,
                           stop_frac=0.7, cost_log="x.jsonl", workers=2)
    sids = [run.BASELINE.sid, run.ADAPT.sid]
    cfg = RunConfig(max_steps=15)
    serial = {}
    for t in range(4):
        _, _, out = run.run_job(FakeEnv(), lambda i: run.RandomLLM(i), cfg, args, t, 0, sids)
        serial[t] = [(sid, r["steps"], r["expansions"]) for sid, r, err in out]
    ctx = mp.get_context("spawn")
    shared = ctx.Value("d", 0.0)
    cwd = os.getcwd()
    with ProcessPoolExecutor(2, mp_context=ctx, initializer=run.worker_init, initargs=(args, shared)) as ex:
        futs = [ex.submit(run.proc_job, t, 0, sids) for t in range(4)]
        par = {t: [(sid, r["steps"], r["expansions"]) for sid, r, err in out]
               for t, _, out in (f.result() for f in futs)}
    assert os.getcwd() == cwd and par == serial


def test_private_tmp_cleanup():
    import os, tempfile
    old = tempfile.tempdir
    d = run.use_private_tmp("alfw_test_")
    try:
        for i in range(4):
            p = os.path.join(d, f"tmp{i}")
            os.makedirs(p)
            os.utime(p, (1000 + i, 1000 + i))
        run.cleanup_tmp(keep=1)
        assert sorted(os.listdir(d)) == ["tmp3"]
    finally:
        tempfile.tempdir = old
        run.TMP_WORK = None
        import shutil
        shutil.rmtree(d, ignore_errors=True)


def test_existing_signatures_unchanged_by_hint_field():
    """hint 필드를 추가해도 기존 두 전략의 캐시 서명은 그대로여야 쌓인 결과를 쓸 수 있다."""
    import hashlib, json
    a = SimpleNamespace(model="m", effort="low", split="s", max_steps=40, mock=False)
    for s in (run.BASELINE, run.ADAPT):
        old_key = (s.start_level, s.max_level, s.trigger_n, s.max_expand, s.scope, s.rules)
        raw = json.dumps({"model": "m", "effort": "low", "split": "s", "max_steps": 40, "code": run.CODE_VERSION,
                          "strategy": old_key}, sort_keys=True, default=str)
        assert run.signature(a, s) == hashlib.md5(raw.encode()).hexdigest()[:10]
    sigs = {run.signature(a, s) for s in (run.BASELINE, run.ADAPT, *run.EXTRA)}
    assert len(sigs) == 2 + len(run.EXTRA)       # 비교군마다 서명이 다름


def test_nohint_arm_omits_unvisited_and_instruction():
    class Env(FakeEnv):
        def available(self):
            return {"commands": ["look", "go to cabinet 1", "go to desk 2"]}

    class SysLLM(RecordLLM):
        def __init__(self):
            super().__init__()
            self.refine_systems = []

        def chat(self, role, system, user, temperature=0.0):
            if role == "refine":
                self.refine_systems.append(system)
            return super().chat(role, system, user, temperature)
    nh = next(s for s in run.EXTRA if s.sid == "adaplan_M3_adapt_nohint")
    llm = SysLLM()
    run_episode(Env(), llm, nh, 0, RunConfig(max_steps=12))
    assert llm.refine_prompts and "Not yet visited" not in llm.refine_prompts[0]
    assert "Not yet visited" not in llm.refine_systems[0]
    llm2 = SysLLM()
    run_episode(Env(), llm2, run.ADAPT, 0, RunConfig(max_steps=12))
    assert "Not yet visited" in llm2.refine_systems[0]


def test_m_sweep_plan_depth():
    """start_level이 실제로 계획 깊이를 바꾼다 (M=1/2/5 기준은 확장 없이 그 깊이로 시작)."""
    for m in (1, 2, 5):
        s = next(x for x in run.EXTRA if x.sid == f"adaplan_M{m}")
        tr = run_episode(FakeEnv(), StuckLLM(), s, 0, RunConfig(max_steps=6))
        assert tr.plan.count("<plan ") == m and tr.expansions == 0
    m1a = next(x for x in run.EXTRA if x.sid == "adaplan_M1_adapt")
    tr = run_episode(FakeEnv(), StuckLLM(), m1a, 0, RunConfig(max_steps=20))
    assert tr.plan.count("<plan ") == 1 and tr.expansions > 0


def test_ablate_runs_from_cache(tmp_path, capsys):
    import json
    cache = tmp_path / "c.jsonl"
    a = SimpleNamespace(model="m", effort="low", split="s", max_steps=40, mock=False, cache=str(cache),
                        arms="all", n=6, reps=1, cmd="ablate", out_dir=str(tmp_path), seed=0, push=False)
    rows = []
    for t in range(6):
        for s in (run.BASELINE, run.ADAPT, *run.EXTRA):
            rows.append({"sid": s.sid, "task": t, "rep": 0, "sig": run.signature(a, s),
                         "success": (s.trigger_n > 0 and t % 2 == 0) or t == 5, "cost_usd": 0.0})
    cache.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    run.ablate(a)
    out = capsys.readouterr().out
    assert "adaplan_M1_adapt" in out and "adaplan_M3_adapt_nohint" in out and "+" in out


def test_prune_keeps_m3_and_counts_spend(tmp_path):
    import json
    c = tmp_path / "c.jsonl"
    rows = [{"sid": s, "task": 0, "rep": 0, "sig": "x", "success": True, "cost_usd": 0.01}
            for s in ("adaplan_M3", "adaplan_M3_adapt", "adaplan_M3_adapt_nohint", "adaplan_M1", "adaplan_M5", "adaplan_M1_adapt")]
    c.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    _, before = run.load_cache(str(c))
    run.prune(SimpleNamespace(cache=str(c)))
    cache, after = run.load_cache(str(c))
    assert {k[0] for k in cache} == {"adaplan_M3", "adaplan_M3_adapt", "adaplan_M3_adapt_nohint"}
    assert abs(before - after) < 1e-12 and (tmp_path / "c.jsonl.bak").exists()
    run.prune(SimpleNamespace(cache=str(c)))      # 두 번째 실행은 아무것도 안 옮김


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
