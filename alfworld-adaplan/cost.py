"""API 비용 계량기 + 예산 보호.

- 모든 LLM 호출의 입력/출력/캐시 토큰을 합산해 비용(USD)을 계산한다.
- 예산의 stop_frac(기본 70%)에 닿으면 BudgetExceeded를 내서 실험을 멈춘다. 나머지는 재실행용 예비.
- 100K 초과 프롬프트(5배 단가 구간)가 나오면 횟수를 센다.

단가는 USD / 1M 토큰 (Claude Haiku 5.5, 2026-10 기준). 모델이 바뀌면 PRICES만 고친다.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field

LONG_THRESHOLD = 100_000   # 이 길이를 넘는 프롬프트는 전체가 비싼 단가

PRICES = {
    "claude-haiku-5-5": {
        "short": {"in": 0.10, "out": 0.50, "cache_read": 0.01, "cache_write": 0.125},
        "long":  {"in": 0.50, "out": 2.50, "cache_read": 0.05, "cache_write": 0.625},
    },
}
DEFAULT_MODEL = "claude-haiku-5-5"


class BudgetExceeded(RuntimeError):
    pass


def call_cost(model: str, in_tok: int, out_tok: int, cache_read: int = 0, cache_write: int = 0) -> float:
    table = PRICES.get(model, PRICES[DEFAULT_MODEL])
    prompt = in_tok + cache_read + cache_write
    p = table["long" if prompt > LONG_THRESHOLD else "short"]
    return (in_tok * p["in"] + out_tok * p["out"]
            + cache_read * p["cache_read"] + cache_write * p["cache_write"]) / 1e6


@dataclass
class CostMeter:
    budget_usd: float = 27.0
    stop_frac: float = 0.7
    in_tokens: int = 0
    out_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cost: float = 0.0
    calls: int = 0
    over_100k: int = 0
    log_path: str | None = None
    _lock: object = field(default_factory=threading.Lock, repr=False)
    _tl: object = field(default_factory=threading.local, repr=False)   # 스레드별 누적(에피소드 비용 계산용)

    def check(self) -> None:
        """호출 직전에 부른다. 예산 한도에 닿았으면 멈춘다."""
        limit = self.budget_usd * self.stop_frac
        if self.cost >= limit:
            raise BudgetExceeded(
                f"누적 ${self.cost:.3f} >= 한도 ${limit:.2f} (예산 ${self.budget_usd} x {self.stop_frac:.0%})")

    def add(self, model: str, role: str, in_tok: int, out_tok: int,
            cache_read: int = 0, cache_write: int = 0) -> float:
        c = call_cost(model, in_tok, out_tok, cache_read, cache_write)
        long_prompt = in_tok + cache_read + cache_write > LONG_THRESHOLD
        tl = self._tl
        tl.inp = getattr(tl, "inp", 0) + in_tok + cache_read + cache_write
        tl.out = getattr(tl, "out", 0) + out_tok
        tl.cost = getattr(tl, "cost", 0.0) + c
        with self._lock:
            self.in_tokens += in_tok
            self.out_tokens += out_tok
            self.cache_read += cache_read
            self.cache_write += cache_write
            self.cost += c
            self.calls += 1
            self.over_100k += long_prompt
            if self.log_path:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"model": model, "role": role, "in": in_tok, "out": out_tok,
                                        "cache_read": cache_read, "cache_write": cache_write,
                                        "cost": round(c, 6), "over_100k": long_prompt}) + "\n")
        return c

    def snapshot(self) -> tuple[int, int, float]:
        """이 스레드가 지금까지 쓴 (입력, 출력, 비용). 병렬 실행에서도 에피소드 비용이 섞이지 않는다."""
        tl = self._tl
        return getattr(tl, "inp", 0), getattr(tl, "out", 0), getattr(tl, "cost", 0.0)

    def delta(self, before: tuple[int, int, float]) -> tuple[int, int, float]:
        now = self.snapshot()
        return now[0] - before[0], now[1] - before[1], now[2] - before[2]

    def summary(self) -> str:
        return (f"호출 {self.calls}회 / 입력 {self.in_tokens:,} (캐시읽기 {self.cache_read:,}) / "
                f"출력 {self.out_tokens:,} / 누적 ${self.cost:.4f} / 100K 초과 {self.over_100k}회")


METER = CostMeter()
