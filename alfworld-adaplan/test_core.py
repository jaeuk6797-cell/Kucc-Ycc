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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
