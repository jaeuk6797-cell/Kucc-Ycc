"""LLM 호출부.

- OpenAICompatLLM : 실제 실험용. vLLM이나 OpenAI API처럼 OpenAI 형식을 지원하는 서버에 요청한다.
- ScriptedLLM     : 테스트용 가짜 LLM. 서버 없이 전체 코드 흐름을 확인할 때 쓴다.

모든 LLM은 chat(role, system, user, temperature) -> Reply 형식을 따른다.
role은 "plan"(계획 생성), "refine"(국소 분해), "act"(행동 선택), "teacher"(전략 편집) 중 하나다.
실제 LLM은 role을 무시하고, 가짜 LLM만 role을 보고 응답을 고른다.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
from dataclasses import dataclass
from typing import Protocol

from cost import METER


@dataclass(frozen=True)
class Reply:
    text: str
    tokens: int   # 입력 + 출력 토큰 수


class LLM(Protocol):
    def chat(self, role: str, system: str, user: str, temperature: float = 0.0) -> Reply: ...


# ---------------------------------------------------------------- 실제 LLM

class OpenAICompatLLM:
    """OpenAI 형식 서버에 요청한다.

    vLLM 예시:  vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000
               OPENAI_BASE_URL=http://localhost:8000/v1  OPENAI_API_KEY=EMPTY
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 max_tokens: int = 512):
        from openai import OpenAI   # 실제 실행할 때만 필요하므로 여기서 불러온다

        self.client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        )
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, role: str, system: str, user: str, temperature: float = 0.0) -> Reply:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        text = r.choices[0].message.content or ""
        u = r.usage
        tokens = u.prompt_tokens + u.completion_tokens if u else (len(system) + len(user) + len(text)) // 4
        if u:
            METER.add("local", role, u.prompt_tokens, u.completion_tokens)
        return Reply(text, tokens)


class AnthropicLLM:
    """Claude API에 공식 SDK로 요청한다.  ANTHROPIC_API_KEY 환경변수를 읽는다.

    현재 Claude 모델은 temperature를 기본값 외로 바꾸면 400을 내므로 temperature는 보내지 않는다.
    같은 과제를 반복할 때의 다양성은 모델 자체의 비결정성에서 나온다.
    생각(thinking)은 끌 수 없으므로 effort로 깊이와 비용을 조절한다.
    """

    # 거절(refusal) 시 서버가 다른 모델로 다시 실행하는 기능을 지원하는 모델
    FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-opus-5", "claude-fable-5-1"}

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", max_tokens: int = 4096):
        import anthropic   # 실제 실행할 때만 필요하므로 여기서 불러온다

        self.client = anthropic.Anthropic(max_retries=6)   # 병렬 실행 시 429/529 대비
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens   # 생각 토큰도 여기에 포함. 비용 상한을 위해 4096으로 제한

    @staticmethod
    def _content(user: str):
        """고정 앞부분(지시문+계획)을 캐시 블록으로 분리. 최소 길이 미만이면 서버가 캐시하지 않아 추가 비용은 없다."""
        marker = "# Recent history"
        if marker in user:
            head, tail = user.split(marker, 1)
            return [{"type": "text", "text": head, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": marker + tail}]
        return user

    def chat(self, role: str, system: str, user: str, temperature: float = 0.0) -> Reply:
        METER.check()   # 예산 한도(기본 70%)에 닿았으면 여기서 멈춘다
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": self._content(user)}],
            output_config={"effort": self.effort},
        )
        if self.model in self.FALLBACK_MODELS:
            r = self.client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        else:
            r = self.client.messages.create(**kwargs)

        text = "" if r.stop_reason == "refusal" else \
            "".join(b.text for b in r.content if b.type == "text")
        u = r.usage
        METER.add(self.model, role, u.input_tokens, u.output_tokens,
                  getattr(u, "cache_read_input_tokens", 0) or 0,
                  getattr(u, "cache_creation_input_tokens", 0) or 0)
        return Reply(text, u.input_tokens + u.output_tokens)


# ---------------------------------------------------------------- 가짜 LLM (테스트용)

class ScriptedLLM:
    """서버 없이 코드 흐름을 시험하기 위한 가짜 LLM.

    - 계획이 자세할수록(레벨 수, 국소 분해 수) 올바른 행동을 고를 확률이 오른다.
    - 한 번 헤매기 시작하면 계획이 바뀔 때까지 계속 헤매는 경향이 있다.
      → 막힘 감지와 계획 확장이 실제로 작동하는지 확인할 수 있다.
    결과 숫자는 실제 성능과 무관하다.
    """

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.confused = False
        self.last_plan_hash = ""

    def chat(self, role: str, system: str, user: str, temperature: float = 0.0) -> Reply:
        if role == "plan":
            text = self._plan(user)
        elif role == "refine":
            text = ("Step 1: Re-read the instruction and list the required attributes.\n"
                    "Step 2: Pick only actions that appear in the clickable list.\n"
                    "Step 3: Continue with the next step of the plan.")
        elif role == "act":
            text = self._act(user)
        else:
            text = "[]"
        return Reply(text, (len(system) + len(user) + len(text)) // 4)

    @staticmethod
    def _plan(user: str) -> str:
        level = len(re.findall(r"<plan \d+>", user)) + 1
        steps = ["Search for the product with the key attributes.",
                 "Open the result that matches the color and category.",
                 "Select the required size option.",
                 "Buy the product."]
        if level >= 2:
            steps.insert(2, "If no result matches, go to the next results page.")
        if level >= 3:
            steps.insert(4, "Check that the price is below the limit before buying.")
        return "\n".join(f"Step {i}: {s}" for i, s in enumerate(steps, 1))

    def _act(self, user: str) -> str:
        plan_part = user.split("# Current observation")[0]
        plan_hash = hashlib.md5(plan_part.split("# Recent history")[0].encode()).hexdigest()
        if plan_hash != self.last_plan_hash:          # 계획이 바뀌면 헤매던 상태가 풀린다
            self.confused = False
            self.last_plan_hash = plan_hash

        levels = len(re.findall(r"<plan \d+>", user))
        refinements = user.count("<refinement")
        skill = min(0.95, 0.70 + 0.07 * levels + 0.08 * refinements)

        if self.confused:
            self.confused = self.rng.random() < 0.85
        elif self.rng.random() > skill:
            self.confused = True
        if self.confused:
            return "Step: 1\nAction: click[a product that looks right]"   # 존재하지 않는 버튼 → 무효 행동

        return f"Step: 1\nAction: {self._good_action(user)}"

    @staticmethod
    def _good_action(user: str) -> str:
        instr = re.search(r"# Instruction\n(.+)", user).group(1).lower()
        m = re.search(r"i need a (\w+) (\w+) in size (\w+)", instr)
        color, category, size = m.groups() if m else ("", "", "")
        obs = user.split("# Current observation", 1)[1]
        clickables = re.search(r"Clickables: (.*)", user).group(1).split(" | ")

        if "Search bar: yes" in user:
            return f"search[{color} {category}]"
        if "Total results" in obs:
            for asin, name in re.findall(r"(b\d{5}) \[SEP\] ([^\[]+?) \[SEP\] \$", obs):
                if color in name.lower() and category in name.lower():
                    return f"click[{asin}]"
            return "click[next >]" if "next >" in clickables else "click[back to search]"
        if "buy now" in clickables:
            if f"size={size}" not in obs and size in clickables:
                return f"click[{size}]"
            return "click[buy now]"
        return "click[back to search]"
