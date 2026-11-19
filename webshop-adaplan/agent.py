"""계층 계획 + 실패 시 확장 에이전트.

한 에피소드의 흐름
  1) 계획 모델이 레벨 1부터 start_level까지 계층 계획을 만든다 (AdaPlan-H 형식: 위 레벨일수록 개괄적).
  2) 실행 모델이 계획을 보며 행동을 고른다.
  3) 최근 2*trigger_n 스텝 중 막힘 신호(무효 행동, 또는 이미 본 화면으로 돌아온 행동)가
     trigger_n번 이상이면 '막힘'으로 보고 계획을 확장한다.
       - global : 다음 레벨을 하나 더 만든다 (앞 레벨들을 조건으로, 논문 식 p_i = f(u, p_{i-1}, ..., p_1))
       - local  : 지금 막힌 단계 하나만 하위 단계로 나눈다 (ADaPT 방식)
       - hybrid : 레벨을 max_level까지 늘린 뒤에는 local로 전환
  4) 상품을 사거나(click[buy now]) 최대 스텝에 도달하면 끝난다.
     최대 스텝은 확장 전후를 합친 값이라, 확장하지 않는 기준 전략과 공정하게 비교된다.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from cost import METER
from llm import LLM


# ---------------------------------------------------------------- 전략과 결과

@dataclass(frozen=True)
class Strategy:
    sid: str
    start_level: int = 1          # 처음에 만들 계획 레벨 수
    max_level: int = 3            # global 확장으로 늘릴 수 있는 최대 레벨 (논문 M=3)
    trigger_n: int = 3            # 최근 2n 스텝 중 막힘 신호가 n번이면 확장 (0이면 확장 안 함 = 기준 전략)
    max_expand: int = 4           # 에피소드당 최대 확장 횟수
    scope: str = "hybrid"         # global / local / hybrid
    rules: tuple[str, ...] = ()   # 계획 프롬프트에 붙일 규칙 (교사가 추가/삭제)

    def key(self) -> tuple:
        """sid를 뺀 실제 설정값. 같은 내용의 전략을 중복으로 보지 않기 위해 쓴다."""
        return (self.start_level, self.max_level, self.trigger_n, self.max_expand,
                self.scope, self.rules)


@dataclass(frozen=True)
class Trial:
    success: bool
    reward: float
    tokens: int
    steps: int          # 실제로 고른 행동 수 (무효 행동 포함)
    expansions: int     # 계획 확장 횟수
    seconds: float
    in_tokens: int = 0    # 비용 계량기 기준 입력 토큰 (캐시 포함)
    out_tokens: int = 0   # 출력 토큰 (생각 토큰 포함)
    cost_usd: float = 0.0


@dataclass
class Plan:
    levels: list[list[str]] = field(default_factory=list)          # levels[i] = i+1 레벨의 단계 목록
    local: dict[int, list[str]] = field(default_factory=dict)      # 막힌 단계 번호 -> 하위 단계

    def render(self) -> str:
        parts = []
        for i, steps in enumerate(self.levels, 1):
            body = "\n".join(f"Step {j}: {s}" for j, s in enumerate(steps, 1))
            parts.append(f"<plan {i}>\n{body}\n</plan {i}>")
        for step_no, subs in sorted(self.local.items()):
            body = "\n".join(f"  {step_no}.{j} {s}" for j, s in enumerate(subs, 1))
            parts.append(f"<refinement of step {step_no}>\n{body}\n</refinement>")
        return "\n".join(parts)


# ---------------------------------------------------------------- 프롬프트

# 계획 모델이 없는 기능(필터, 장바구니 등)을 계획하지 않도록 사이트 구조를 알려 준다
SITE_GUIDE = (
    "The website has only three kinds of pages. "
    "(1) Search page: type a query with search[...]. "
    "(2) Results page: a list of products with their ids, names and prices; 'next >' and "
    "'< prev' change pages, 'back to search' returns to the search page. "
    "(3) Product page: option buttons (such as size or color), 'description', 'features', "
    "'reviews', '< prev', and 'buy now'. "
    "There are no filters and no cart. To finish, open a matching product, click each required "
    "option, then click 'buy now'. Buying ends the task, so buy the best match you have found "
    "rather than searching forever."
)

PLAN_SYSTEM = (
    "You are a planner for a web shopping agent. Plans are hierarchical: plan 1 is a coarse "
    "outline, and each next level refines the previous one with more concrete conditions and "
    "details while keeping the same overall goal. " + SITE_GUIDE + " Write only numbered lines "
    "in the form 'Step k: ...'."
)

REFINE_SYSTEM = (
    "You are a planner for a web shopping agent. The agent got stuck on one step of its plan. "
    "Break that single step into 2-4 smaller concrete sub-steps based on the current page. "
    + SITE_GUIDE + " Write only numbered lines in the form 'Step k: ...'."
)

ACT_SYSTEM = (
    "You are a web shopping agent. Follow the plan. Choose exactly one action.\n"
    "Valid actions: search[<query>] when the search bar is available, or click[<button>] "
    "where <button> is one of the clickables.\n" + SITE_GUIDE + "\n"
    "Answer in exactly two lines:\nStep: <number of the step you are working on in the most "
    "detailed plan>\nAction: <action>"
)


def _steps(text: str) -> list[str]:
    """'Step k: ...' 줄만 뽑아 단계 목록으로 만든다."""
    found = re.findall(r"Step\s*\d+\s*[:.]\s*(.+)", text)
    return [s.strip() for s in found] or [text.strip()[:200]]


def make_level(llm: LLM, instruction: str, plan: Plan, rules: tuple[str, ...],
               note: str = "") -> int:
    """계획에 레벨 하나를 추가하고 사용한 토큰 수를 돌려준다."""
    level = len(plan.levels) + 1
    user = f"# Instruction\n{instruction}\n"
    if plan.levels:
        user += f"# Previous levels\n{plan.render()}\n"
    if rules:
        user += "# Rules\n" + "\n".join(f"- {r}" for r in rules) + "\n"
    if note:
        user += f"# Situation\n{note}\n"
    user += f"# Write plan {level}" + (" (more detailed than the previous level)" if level > 1 else "")
    r = llm.chat("plan", PLAN_SYSTEM, user)
    plan.levels.append(_steps(r.text))
    return r.tokens


def refine_step(llm: LLM, instruction: str, plan: Plan, step_no: int, note: str) -> int:
    """막힌 단계 하나를 하위 단계로 나눈다."""
    user = (f"# Instruction\n{instruction}\n# Plan\n{plan.render()}\n"
            f"# Stuck step\nStep {step_no}\n# Situation\n{note}\n# Sub-steps for step {step_no}")
    r = llm.chat("refine", REFINE_SYSTEM, user)
    plan.local[step_no] = _steps(r.text)
    return r.tokens


def parse_action(text: str) -> tuple[int, str | None]:
    m_step = re.search(r"Step:\s*(\d+)", text)
    m_act = re.search(r"Action:\s*((?:search|click)\[.*?\])", text, re.IGNORECASE)
    step_no = int(m_step.group(1)) if m_step else 1
    return step_no, (m_act.group(1) if m_act else None)


def is_valid(action: str, avail: dict) -> bool:
    a = action.strip().lower()
    if a.startswith("search["):
        return bool(avail.get("has_search_bar"))
    if a.startswith("click["):
        return a[6:-1] in [c.lower() for c in avail.get("clickables", [])]
    return False


def _short(obs: str, n: int = 300) -> str:
    return obs if len(obs) <= n else obs[:n] + " ..."


# ---------------------------------------------------------------- 에피소드 실행

@dataclass(frozen=True)
class RunConfig:
    max_steps: int = 20               # 확장 전후 합산 최대 행동 수
    success_threshold: float = 1.0    # 보상이 이 값 이상이면 성공
    act_temperature: float = 0.0      # 같은 과제를 여러 번 돌릴 거면 0보다 크게
    history_len: int = 6              # 행동 프롬프트에 넣을 최근 기록 수
    obs_chars: int = 4000             # 현재 화면을 이 길이까지 보여 줌 (결과 페이지는 최대 약 2200자)


def run_episode(env, llm: LLM, s: Strategy, session: int, cfg: RunConfig) -> Trial:
    t0 = time.time()
    meter0 = METER.snapshot()
    obs, instruction = env.reset(session)
    plan = Plan()
    tokens = 0
    for _ in range(s.start_level):
        tokens += make_level(llm, instruction, plan, s.rules)

    history: list[tuple[str, str]] = []
    marks: list[bool] = []        # 스텝마다 막힘 신호가 있었는지 (확장하면 비운다)
    seen = {obs}                  # 이번 에피소드에서 이미 본 화면
    steps = expansions = 0
    reward, done = 0.0, False

    while steps < cfg.max_steps and not done:
        avail = env.available()
        recent = "\n".join(f"> {a}\n{o}" for a, o in history[-cfg.history_len:]) or "(none)"
        user = (f"# Instruction\n{instruction}\n# Plan\n{plan.render()}\n"
                f"# Recent history\n{recent}\n# Current observation\n{_short(obs, cfg.obs_chars)}\n"
                f"# Available actions\nSearch bar: {'yes' if avail.get('has_search_bar') else 'no'}\n"
                f"Clickables: {' | '.join(avail.get('clickables', []))}")
        r = llm.chat("act", ACT_SYSTEM, user, temperature=cfg.act_temperature)
        tokens += r.tokens
        steps += 1
        step_no, action = parse_action(r.text)

        # -- 막힘 신호: 무효 행동, 또는 이미 본 화면으로 돌아온 행동
        #    (화면이 안 바뀐 경우도 포함. 검색→상품→검색으로 맴도는 반복도 잡는다)
        if action is None or not is_valid(action, avail):
            marks.append(True)
            history.append((action or r.text.strip()[:60], "Invalid action."))
        else:
            new_obs, reward, done = env.step(action)
            marks.append(new_obs in seen)
            seen.add(new_obs)
            history.append((action, _short(new_obs)))
            obs = new_obs

        # -- 확장: 최근 2*trigger_n 스텝 중 막힘 신호가 trigger_n번 이상이면
        stuck = sum(marks[-2 * s.trigger_n:]) if s.trigger_n else 0
        if (not done and s.trigger_n and stuck >= s.trigger_n and expansions < s.max_expand):
            note = f"The agent is stuck on step {step_no}. Last actions:\n" + \
                   "\n".join(f"> {a} -> {o[:80]}" for a, o in history[-s.trigger_n:])
            add_level = s.scope == "global" or (s.scope == "hybrid" and len(plan.levels) < s.max_level)
            if add_level and len(plan.levels) < s.max_level:
                tokens += make_level(llm, instruction, plan, s.rules, note)
            elif s.scope in ("local", "hybrid"):
                tokens += refine_step(llm, instruction, plan, step_no, note)
            else:
                continue          # global인데 이미 최대 레벨: 더 확장할 수 없음
            expansions += 1
            marks.clear()

    d_in, d_out, d_cost = METER.delta(meter0)
    return Trial(success=reward >= cfg.success_threshold, reward=reward, tokens=tokens,
                 steps=steps, expansions=expansions, seconds=round(time.time() - t0, 3),
                 in_tokens=d_in, out_tokens=d_out, cost_usd=round(d_cost, 6))
