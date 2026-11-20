"""ALFWorld 텍스트 환경 어댑터 (TextWorld 기반, THOR 없이 CPU로 동작).

  reset(idx)  -> (관찰 텍스트, 지시문)       idx = 평가 split 안의 게임 번호
  step(cmd)   -> (관찰 텍스트, 보상 0/1, 종료 여부)   보상은 과제를 끝내면 1
  available() -> {"commands": [허용되는 명령 목록]}

실행 전에 ALFWORLD_DATA 환경변수가 데이터 폴더(alfworld-download로 받은 곳)를 가리켜야 한다.
Python 3.11/3.12 필요 (3.13에서는 textworld가 동작하지 않음).
"""

from __future__ import annotations

import os
import re

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alfworld_config.yaml")


class ALFWorldEnv:
    def __init__(self, split: str = "eval_out_of_distribution", config_path: str = DEFAULT_CONFIG):
        import yaml
        from alfworld.agents.environment import get_environment

        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self._env = get_environment(cfg["env"]["type"])(cfg, train_eval=split)
        self.game_files = list(self._env.game_files)   # 게임 번호 -> 파일 (고정 순서)
        self.num_games = len(self.game_files)
        self._tw = None
        self._cmds: list[str] = []

    def task_type(self, idx: int) -> str:
        """게임 폴더 이름 앞부분 = 과제 유형 (pick_and_place_simple, look_at_obj_in_light 등 6종)."""
        return os.path.basename(os.path.dirname(os.path.dirname(self.game_files[idx]))).split("-")[0]

    def reset(self, idx: int) -> tuple[str, str]:
        self._env.game_files = [self.game_files[idx]]
        self._env.num_games = 1
        self._tw = self._env.init_env(batch_size=1)
        obs, info = self._tw.reset()
        self._cmds = list(info["admissible_commands"][0])
        text = obs[0]
        m = re.search(r"Your task is to:\s*(.+)", text)
        goal = m.group(1).strip() if m else ""
        body = text.split("Your task is to:")[0].replace("-= Welcome to TextWorld, ALFRED! =-", "").strip()
        return body, goal

    def step(self, action: str) -> tuple[str, float, bool]:
        obs, _scores, dones, info = self._tw.step([action])
        self._cmds = list(info["admissible_commands"][0])
        won = bool(info.get("won", [False])[0])
        return obs[0], 1.0 if won else 0.0, bool(dones[0]) or won

    def available(self) -> dict:
        return {"commands": self._cmds}


class ChainEnv:
    """게임 k개를 연속으로 풀게 하는 합성 환경 (긴 과제 흉내). 한 세계가 아니라 독립된 게임을 순서대로 여는 방식이다.

    - 합성 번호 c는 시드로 섞은 게임 목록을 k개씩 끊은 c번째 묶음 (같은 시드·풀이면 항상 같음)
    - 지시문에 k개 목표를 한꺼번에 주고, 한 게임이 끝나면 다음 게임으로 자동 전환(관찰에 새 목표가 나옴)
    - 보상 = 푼 게임 수 / k (전부 풀면 1.0), 모두 끝났을 때만 종료. 성공 기준은 전부 풀기
    - 게임이 성공 없이 끝나면(환경 자체 종료) 그 게임은 실패로 치고 다음으로 넘어감
    """

    def __init__(self, base, k: int, seed: int = 0, pool=None):
        import random
        self.base, self.k = base, k
        pool = list(range(base.num_games)) if pool is None else list(pool)
        perm = random.Random(seed).sample(pool, len(pool))
        self.chains = [perm[i * k:(i + 1) * k] for i in range(len(pool) // k)]
        self.num_games = len(self.chains)
        self._games: list[int] = []
        self._i = self._solved = 0

    def task_type(self, idx: int) -> str:
        return f"chain{self.k}"

    def reset(self, idx: int) -> tuple[str, str]:
        self._games = list(self.chains[idx])
        goals = [self.base.reset(g)[1] for g in self._games]          # 지시문에 쓸 목표만 먼저 훑음
        body, goal = self.base.reset(self._games[0])
        self._i = self._solved = 0
        lines = "\n".join(f"{i}. {g}" for i, g in enumerate(goals, 1))
        instruction = (f"Complete ALL {self.k} tasks below, in this order. When a task ends, the next one starts "
                       f"automatically in a new room.\n{lines}")
        return f"[Task 1/{self.k}] {body}\nYour task is to: {goal}", instruction

    def step(self, action: str) -> tuple[str, float, bool]:
        obs, r, done = self.base.step(action)
        if not done:
            return obs, self._solved / self.k, False
        won = r >= 1.0
        self._solved += int(won)
        self._i += 1
        if self._i >= self.k:
            return obs, self._solved / self.k, True
        body, goal = self.base.reset(self._games[self._i])
        head = f"[Task {self._i}/{self.k} {'completed' if won else 'ended without success'}. Task {self._i + 1}/{self.k} starts now in a new room.]"
        return f"{head}\n{body}\nYour task is to: {goal}", self._solved / self.k, False

    def available(self) -> dict:
        return self.base.available()
