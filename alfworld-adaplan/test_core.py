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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
