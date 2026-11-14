"""쇼핑 환경.

- WebShopEnv : 실제 WebShop (princeton-nlp/WebShop) 텍스트 환경을 감싼 어댑터
- ALFWorldEnv: ALFWorld 가정 환경 (자유 형식 명령, 허용 명령 목록 제공)
- FakeShop   : 테스트용 가짜 쇼핑몰. 행동 형식(search[...], click[...])과 화면 구조를 흉내 낸다.

두 환경 모두 같은 세 메서드를 제공한다.
  reset(session)  -> (관찰 텍스트, 지시문)
  step(action)    -> (관찰 텍스트, 보상 0~1, 종료 여부)
  available()     -> {"has_search_bar": bool, "clickables": [str, ...]}
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------- 실제 WebShop

class WebShopEnv:
    """WebShop 텍스트 환경 어댑터.

    WebShop 저장소를 설치한 뒤(./setup.sh), 그 저장소 폴더 안에서 실행해야 한다.
    아래 메서드 이름(get_instruction_text, get_available_actions)은 WebShop 버전에 따라
    다를 수 있으니 처음 한 번 `python run.py episode --session 0`으로 확인할 것.
    """

    def __init__(self, num_products: int | None = None):
        import gym
        from web_agent_site.envs import WebAgentTextEnv  # noqa: F401  (gym에 환경 등록)

        self.env = gym.make("WebAgentTextEnv-v0", observation_mode="text",
                            num_products=num_products)

    def reset(self, session: int) -> tuple[str, str]:
        obs, _ = self.env.reset(session=session)
        return obs, self.env.get_instruction_text()

    def step(self, action: str) -> tuple[str, float, bool]:
        obs, reward, done, _ = self.env.step(action)
        return obs, float(reward or 0.0), bool(done)

    def available(self) -> dict:
        return self.env.get_available_actions()


# ---------------------------------------------------------------- ALFWorld

class ALFWorldEnv:
    """ALFWorld 텍스트 환경(TextWorld 판) 어댑터. 과제 번호 = 정렬된 게임 파일 목록의 인덱스.

    ALFWORLD_DATA 환경변수(기본 ~/.cache/alfworld)의 json_2.1.1/<split> 아래에서 게임을 모은다.
    split: train(6374 파일, 풀 수 있는 것만) / valid_seen / valid_unseen / valid_train.
    행동은 자유 형식 명령이고 매 스텝 허용 명령 목록(admissible commands)이 주어진다.
    보상은 성공하면 1, 아니면 0 (ALFWorld는 중간 보상이 없다).
    """

    domain = "alfworld"

    def __init__(self, split: str = "valid_unseen", data_dir: str | None = None):
        import json
        import os

        root = os.path.expandvars(data_dir or os.environ.get("ALFWORLD_DATA", "~/.cache/alfworld"))
        root = os.path.join(os.path.expanduser(root), "json_2.1.1", split)
        files = []
        for d, _, names in os.walk(root):
            if "traj_data.json" not in names or "movable" in d or "Sliced" in d:
                continue
            gf = os.path.join(d, "game.tw-pddl")
            if not os.path.exists(gf):
                continue
            with open(gf, encoding="utf-8") as f:
                if not json.load(f).get("solvable", False):
                    continue
            files.append(gf)
        self.game_files = sorted(files)
        if not self.game_files:
            raise FileNotFoundError(f"ALFWorld 게임을 찾지 못함: {root} (alfworld-download 를 했는지 확인)")
        self.env = None
        self.infos: dict = {}

    def reset(self, session: int) -> tuple[str, str]:
        import re

        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        if self.env is not None:
            self.env.close()
        gf = self.game_files[session]
        infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env_id = textworld.gym.register_games([gf], infos, batch_size=None,
                                              wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
        self.env = textworld.gym.make(env_id)
        obs, self.infos = self.env.reset()
        obs = re.sub(r"^-=.*?=-\s*", "", obs, flags=re.S).strip()
        m = re.search(r"Your task is to:\s*(.+)", obs)
        return obs, (m.group(1).strip() if m else obs)

    def step(self, action: str) -> tuple[str, float, bool]:
        obs, _, done, self.infos = self.env.step(action)
        won = bool(self.infos.get("won"))
        return obs.strip(), 1.0 if won else 0.0, bool(done) or won

    def available(self) -> dict:
        cmds = [c for c in self.infos.get("admissible_commands", []) if c != "help"]
        return {"has_search_bar": False, "free_text": True, "clickables": cmds}


# ---------------------------------------------------------------- 가짜 쇼핑몰 (테스트용)

CATEGORIES = ["shirt", "shoes", "bag", "lamp", "mug", "jacket", "pillow", "headphones"]
COLORS = ["red", "blue", "black", "white", "green", "gray"]
SIZES = ["small", "medium", "large"]


@dataclass
class Product:
    asin: str
    category: str
    color: str
    sizes: list[str]
    price: float

    @property
    def name(self) -> str:
        return f"{self.color.title()} {self.category.title()}"


@dataclass
class _State:
    page: str = "search"                    # search / results / item
    results: list[Product] = field(default_factory=list)
    page_no: int = 0
    item: Product | None = None
    selected_size: str | None = None


class FakeShop:
    """세션 번호마다 정해진 목표 상품이 있는 가짜 쇼핑몰.

    세션 번호 % 4 = 숨은 난이도: 목표 상품이 검색 결과 몇 번째 페이지에 있는지.
    (실제 실험의 난이도 측정이 이 숨은 값을 잘 맞히는지 확인하는 용도)
    """

    PER_PAGE = 10

    def __init__(self, n_products: int = 400, seed: int = 0):
        rng = random.Random(seed)
        self.products = [
            Product(f"b{i:05d}", rng.choice(CATEGORIES), rng.choice(COLORS),
                    sorted(rng.sample(SIZES, rng.randint(1, 3))), round(rng.uniform(10, 90), 2))
            for i in range(n_products)
        ]
        self.s = _State()

    # -- 과제 설정
    def reset(self, session: int) -> tuple[str, str]:
        rng = random.Random(10_000 + session)
        self.target = rng.choice(self.products)
        self.goal_size = rng.choice(self.target.sizes)
        self.max_price = round(self.target.price + rng.uniform(1, 10), 2)
        self.hardness = session % 4
        self.instruction = (f"i need a {self.target.color} {self.target.category} in size "
                            f"{self.goal_size}, and price lower than {self.max_price:.2f} dollars")
        self.s = _State()
        return self._render(), self.instruction

    # -- 화면
    def _render(self) -> str:
        head = f"Instruction: [SEP] {self.instruction}"
        if self.s.page == "search":
            return f"WebShop [SEP] {head} [SEP] Search"
        if self.s.page == "results":
            start = self.s.page_no * self.PER_PAGE
            items = self.s.results[start:start + self.PER_PAGE]
            rows = " [SEP] ".join(f"{p.asin} [SEP] {p.name} [SEP] ${p.price}" for p in items)
            return (f"{head} [SEP] Back to Search [SEP] Page {self.s.page_no + 1} "
                    f"(Total results: {len(self.s.results)}) [SEP] Next > [SEP] {rows}")
        p = self.s.item
        sel = f"size={self.s.selected_size}" if self.s.selected_size else "none"
        return (f"{head} [SEP] Back to Search [SEP] < Prev [SEP] size [SEP] {' [SEP] '.join(p.sizes)}"
                f" [SEP] {p.name} [SEP] Price: ${p.price} [SEP] Selected: {sel} [SEP] Buy Now")

    def available(self) -> dict:
        if self.s.page == "search":
            return {"has_search_bar": True, "clickables": ["search"]}
        if self.s.page == "results":
            start = self.s.page_no * self.PER_PAGE
            items = [p.asin for p in self.s.results[start:start + self.PER_PAGE]]
            more = start + self.PER_PAGE < len(self.s.results)
            return {"has_search_bar": False,
                    "clickables": ["back to search"] + (["next >"] if more else []) + items}
        return {"has_search_bar": False,
                "clickables": ["back to search", "< prev", *self.s.item.sizes, "buy now"]}

    # -- 행동
    def step(self, action: str) -> tuple[str, float, bool]:
        action = action.strip().lower()
        if action.startswith("search[") and self.s.page == "search":
            self._search(action[7:-1])
        elif action.startswith("click["):
            target = action[6:-1]
            if target not in self.available()["clickables"]:
                return self._render(), 0.0, False          # 무효 클릭: 화면 그대로
            if target == "buy now":
                return self._render(), self._reward(), True
            self._click(target)
        return self._render(), 0.0, False

    def _search(self, query: str) -> None:
        words = set(query.split())
        others = [p for p in self.products
                  if p is not self.target and words & {p.color, p.category}
                  and not (p.color == self.target.color and p.category == self.target.category)]
        others.sort(key=lambda p: -len(words & {p.color, p.category}))
        pos = min(len(others), self.hardness * self.PER_PAGE + 3)
        self.s = _State(page="results", results=others[:pos] + [self.target] + others[pos:])

    def _click(self, target: str) -> None:
        s = self.s
        if target == "back to search":
            self.s = _State()
        elif target == "next >":
            s.page_no += 1
        elif target == "< prev":
            s.page, s.item, s.selected_size = "results", None, None
        elif s.page == "results":
            s.page = "item"
            s.item = next(p for p in s.results if p.asin == target)
        elif s.page == "item" and target in SIZES:
            s.selected_size = target

    def _reward(self) -> float:
        p = self.s.item
        checks = [p.category == self.target.category, p.color == self.target.color,
                  self.s.selected_size == self.goal_size, p.price <= self.max_price]
        return sum(checks) / len(checks)
