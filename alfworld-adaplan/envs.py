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

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "alfworld_config.yaml")


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
