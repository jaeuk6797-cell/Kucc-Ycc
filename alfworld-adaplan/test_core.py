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
    a.arms = None                                   # 기본값: M3 계열만, M1 열 없음
    run.ablate(a)
    out2 = capsys.readouterr().out
    assert "adaplan_M3_adapt_nohint" in out2 and "adaplan_M1" not in out2


def test_pick_tasks_pool_and_types_filter(tmp_path, capsys):
    import json
    assert run.pick_tasks(10, 3, 0) == run.pick_tasks(10, 3, 0, None)          # pool 없으면 기존과 동일
    got = run.pick_tasks(100, 5, 0, pool=[7, 8, 9, 40])
    assert set(got) <= {7, 8, 9, 40} and len(got) == 4                           # pool보다 많이 요청해도 pool 크기까지만
    f = run.type_filter(SimpleNamespace(types="heat, look_at"))
    assert f("pick_heat_then_place_in_recep") and f("look_at_obj_in_light") and not f("pick_two_obj_and_place")
    assert run.type_filter(SimpleNamespace(types=None))("anything")
    cache = tmp_path / "c.jsonl"
    a = SimpleNamespace(model="m", effort="low", split="s", max_steps=40, mock=False, cache=str(cache),
                        arms="adaplan_M1", n=6, reps=1, cmd="ablate", out_dir=str(tmp_path), seed=0, push=False,
                        types="heat")
    rows = []
    for t in range(6):
        for s in (run.BASELINE, run.ADAPT, *[x for x in run.EXTRA if x.sid == "adaplan_M1"]):
            rows.append({"sid": s.sid, "task": t, "rep": 0, "sig": run.signature(a, s), "success": t < 3, "cost_usd": 0.01,
                         "ttype": "pick_heat_then_place_in_recep" if t < 3 else "pick_two_obj_and_place"})
    cache.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    run.ablate(a)
    out = capsys.readouterr().out
    assert "유형=heat" in out and "$/성공" in out
    line = next(l for l in out.splitlines() if l.startswith("adaplan_M1"))
    assert line.split()[1] == "3" and "100%" in line and "0.0100" in line          # heat 3개만, 전부 성공, 성공당 0.01


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


def test_effect_by_tier(tmp_path):
    import csv, json
    import difficulty
    feats = tmp_path / "f.csv"
    rows = [{"task": i, "ttype": "pick_and_place_simple", "plan_len": 4 + i // 4, "n_goto": 2, "n_receps": 20, "needs_tool": 0,
             "scene": 1, "tier": 1} for i in range(16)]
    with open(feats, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    a = SimpleNamespace(model="claude-haiku-5-5", effort="low", split="s", max_steps=40, mock=False)
    cache = tmp_path / "c.jsonl"
    lines = []
    for t in range(16):
        for rep in range(2):
            lines.append({"sid": "adaplan_M3", "task": t, "rep": rep, "sig": run.signature(a, run.BASELINE), "success": t < 8})
            lines.append({"sid": "adaplan_M3_adapt", "task": t, "rep": rep, "sig": run.signature(a, run.ADAPT), "success": t < 12})
        # 다른 split의 같은 task 번호 결과가 섞이면 안 됨
        lines.append({"sid": "adaplan_M3", "task": t, "rep": 0, "sig": "other", "success": True})
    cache.write_text("\n".join(json.dumps(r) for r in lines), encoding="utf-8")
    text = difficulty.effect_by_tier(str(feats), str(cache), "s", 4, "adaplan_M3", "adaplan_M3_adapt", "claude-haiku-5-5", "low", 40)
    assert "전체" in text and "+25.0%" in text           # 16개 중 4개(0.25)만 새로 성공
    rows2 = [{"plan_len": 5, "n_receps": 20} for _ in range(10)]
    difficulty.assign_tiers_tied(rows2, 4)
    assert len({r["tier"] for r in rows2}) == 1          # 같은 값은 같은 단계


def test_effect_aborts_when_feature_table_order_differs(tmp_path):
    import csv, json
    import difficulty
    feats = tmp_path / "f.csv"
    rows = [{"task": i, "ttype": "look_at_obj_in_light", "plan_len": 4, "n_goto": 2, "n_receps": 10, "needs_tool": 0,
             "scene": 1, "tier": 1} for i in range(6)]
    with open(feats, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    a = SimpleNamespace(model="claude-haiku-5-5", effort="low", split="s", max_steps=40, mock=False)
    cache = tmp_path / "c.jsonl"
    lines = [{"sid": s.sid, "task": t, "rep": 0, "sig": run.signature(a, s), "success": True, "ttype": "pick_two_obj_and_place"}
             for t in range(6) for s in (run.BASELINE, run.ADAPT)]
    cache.write_text("\n".join(json.dumps(r) for r in lines), encoding="utf-8")
    text = difficulty.effect_by_tier(str(feats), str(cache), "s", 4, "adaplan_M3", "adaplan_M3_adapt", "claude-haiku-5-5", "low", 40)
    assert "[중단]" in text


def test_diagnose_runs(tmp_path, capsys):
    import json
    a = SimpleNamespace(model="m", effort="low", split="s", max_steps=40, mock=False, cache=str(tmp_path / "c.jsonl"),
                        arms="adaplan_M1", n=2, reps=1, cmd="diagnose", out_dir=str(tmp_path), seed=0, push=False)
    rows = []
    for s in (run.BASELINE, run.EXTRA[0]):
        for t in range(2):
            rows.append({"sid": s.sid, "task": t, "rep": 0, "sig": run.signature(a, s), "success": t == 0, "steps": 5,
                         "in_tokens": 100, "plan": "<plan 1>\nStep 1: x\n</plan 1>",
                         "traj": [{"valid": True, "stuck": False}, {"valid": False, "stuck": True}]})
    (tmp_path / "c.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    run.diagnose(a)
    out = capsys.readouterr().out
    assert "adaplan_M1" in out and "50.0%" in out


def test_planner_router_roles_and_signature():
    from llm import RoleRouter
    calls = []

    class Tag:
        def __init__(self, name):
            self.name = name

        def chat(self, role, system, user, temperature=0.0):
            calls.append((self.name, role))
            return Reply("Step 1: a\nStep 2: b" if role in ("plan", "refine") else "Step: 1\nAction: look", 3)
    router = RoleRouter(Tag("main"), Tag("planner"))
    run_episode(FakeEnv(), router, run.ADAPT, 0, RunConfig(max_steps=12))
    assert all(n == "planner" for n, r in calls if r in ("plan", "refine"))
    assert all(n == "main" for n, r in calls if r == "act")
    assert any(r == "refine" for _, r in calls)
    a = SimpleNamespace(model="m", effort="low", split="s", max_steps=40, mock=False)
    b = SimpleNamespace(model="m", effort="low", split="s", max_steps=40, mock=False, planner_model="llama3.1:8b")
    c = SimpleNamespace(model="m", effort="low", split="s", max_steps=40, mock=False, planner_model=None)
    assert run.signature(a, run.BASELINE) == run.signature(c, run.BASELINE)       # 지정 안 하면 기존 서명 그대로
    assert run.signature(a, run.BASELINE) != run.signature(b, run.BASELINE)


def test_register_price():
    import cost
    cost.register_price("llama-test", 0.05, 0.08)
    assert abs(cost.call_cost("llama-test", 1_000_000, 1_000_000) - 0.13) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
