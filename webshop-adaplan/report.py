"""실험 결과를 표와 그래프로 정리한다.

  python report.py trials.jsonl                          # 전략별 표 + 그래프
  python report.py trials.jsonl --tiers tiers.csv        # 난이도 단계별 성공률도
  python report.py before.jsonl after.jsonl --labels 수정전,수정후 --out report_cmp
                                                          # 여러 파일 비교 (파일마다 라벨)

입력 한 줄 = 실행 한 번 (run.py search가 쓰는 trials.jsonl 형식):
  {"sid": ..., "task": ..., "rep": ..., "success": ..., "reward": ..., "tokens": ...,
   "steps": ..., "expansions": ...}

출력 (--out 폴더, 기본 report/)
  summary.csv / summary.md   전략별 성공률(95% 신뢰구간), 보상, 토큰, 스텝, 확장
  per_task.csv               과제 × 전략 성공률
  success_rate.png           전략별 성공률 + 95% 신뢰구간
  cost_vs_success.png        에피소드당 토큰 vs 성공률, 파레토 전선 표시
  expansions.png             확장 횟수 분포 (전략별)
  success_by_tier.png        난이도 단계별 성공률 (--tiers 를 줄 때)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict

# 색: 기본 팔레트의 범주형 순서 (색은 순위가 아니라 전략에 고정)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


# ---------------------------------------------------------------- 읽기와 집계

def load(paths: list[str], labels: list[str] | None) -> list[dict]:
    rows = []
    for i, path in enumerate(paths):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if labels:
                        r["sid"] = labels[i]
                    rows.append(r)
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """이항 비율의 95% 윌슨 신뢰구간. 표본이 작을 때도 0~1을 벗어나지 않는다."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def summarize(rows: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["sid"]].append(r)
    out = []
    for sid, rs in by.items():
        n, k = len(rs), sum(bool(r["success"]) for r in rs)
        lo, hi = wilson(k, n)
        ok = [r for r in rs if r["success"]]
        out.append({
            "sid": sid, "n": n, "success": k,
            "succ_rate": k / n, "ci_lo": lo, "ci_hi": hi,
            "mean_reward": sum(r["reward"] for r in rs) / n,
            "tokens_per_episode": sum(r["tokens"] for r in rs) / n,
            "tokens_per_success": sum(r["tokens"] for r in rs) / k if k else math.inf,
            "mean_steps": sum(r["steps"] for r in rs) / n,
            "mean_expansions": sum(r["expansions"] for r in rs) / n,
        })
    return out


def pareto(summary: list[dict]) -> set[str]:
    """성공률은 높을수록, 에피소드당 토큰은 낮을수록 좋다."""
    def dom(a, b):
        return (a["succ_rate"] >= b["succ_rate"] and a["tokens_per_episode"] <= b["tokens_per_episode"]
                and (a["succ_rate"] > b["succ_rate"] or a["tokens_per_episode"] < b["tokens_per_episode"]))
    return {a["sid"] for a in summary if not any(dom(b, a) for b in summary if b is not a)}


# ---------------------------------------------------------------- 표

def fmt(x: float, d: int = 0) -> str:
    return "-" if x == math.inf else f"{x:,.{d}f}"


def write_tables(summary: list[dict], rows: list[dict], out: str) -> str:
    keys = list(summary[0])
    with open(os.path.join(out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)

    front = pareto(summary)
    head = ("| 전략 | 실행 수 | 성공률 (95% CI) | 평균 보상 | 토큰/에피소드 | 토큰/성공 | 평균 스텝 | 평균 확장 |\n"
            "|---|---:|---|---:|---:|---:|---:|---:|\n")
    body = "".join(
        f"| {s['sid']}{' ★' if s['sid'] in front else ''} | {s['n']} | "
        f"{s['succ_rate']:.0%} ({s['ci_lo']:.0%}–{s['ci_hi']:.0%}) | {s['mean_reward']:.3f} | "
        f"{fmt(s['tokens_per_episode'])} | {fmt(s['tokens_per_success'])} | "
        f"{s['mean_steps']:.1f} | {s['mean_expansions']:.2f} |\n"
        for s in summary)
    md = head + body + "\n★ = 파레토 전선 (성공률↑, 토큰↓ 기준으로 더 나은 전략이 없음)\n"
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)

    # 과제 × 전략 성공률 (반복 실행 평균)
    sids = [s["sid"] for s in summary]
    cell = defaultdict(list)
    for r in rows:
        cell[(r["task"], r["sid"])].append(bool(r["success"]))
    tasks = sorted({r["task"] for r in rows})
    with open(os.path.join(out, "per_task.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", *sids])
        for t in tasks:
            w.writerow([t, *[f"{sum(v) / len(v):.2f}" if (v := cell.get((t, s))) else "" for s in sids]])
    return md


# ---------------------------------------------------------------- 그래프

def _style(plt):
    # 한글 글꼴: WSL/리눅스에서는 Windows 글꼴 폴더의 맑은 고딕을 직접 등록한다
    from matplotlib import font_manager
    for path in ("/mnt/c/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgun.ttf",
                 "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
    plt.rcParams.update({
        "font.family": ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    })


def color_map(summary: list[dict]) -> dict[str, str]:
    """입력 순서대로 색을 고정. 8개를 넘는 전략은 회색으로 묶는다."""
    return {s["sid"]: SERIES[i] if i < len(SERIES) else MUTED for i, s in enumerate(summary)}


def pick_series(summary: list[dict], k: int) -> list[dict]:
    """그룹 막대 그래프용: 성공률 상위 k개 (baseline은 항상 포함). 원래 순서와 색은 유지한다."""
    top = sorted(summary, key=lambda s: (-s["succ_rate"], s["tokens_per_episode"]))[:k]
    keep = {s["sid"] for s in top} | {s["sid"] for s in summary if s["sid"] == "baseline"}
    return [s for s in summary if s["sid"] in keep]


def _legend(ax, n: int) -> None:
    if n > 1:
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)


def plot_success(plt, summary, colors, out):
    data = sorted(summary, key=lambda s: s["succ_rate"])
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(data) + 1.6))
    y = range(len(data))
    ax.barh(y, [s["succ_rate"] for s in data], height=0.55,
            color=[colors[s["sid"]] for s in data], zorder=2)
    ax.errorbar([s["succ_rate"] for s in data], y,
                xerr=[[s["succ_rate"] - s["ci_lo"] for s in data], [s["ci_hi"] - s["succ_rate"] for s in data]],
                fmt="none", ecolor=INK2, elinewidth=1.2, capsize=4, zorder=3)
    for i, s in enumerate(data):
        ax.text(min(s["ci_hi"] + 0.02, 0.98), i, f"{s['succ_rate']:.0%}  (n={s['n']})",
                va="center", fontsize=9, color=INK2)
    ax.set_yticks(list(y), [s["sid"] for s in data])
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}" if v <= 1 else ""))
    ax.grid(axis="y", visible=False)
    ax.set_title("전략별 성공률 (막대: 95% 신뢰구간)", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "success_rate.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cost(plt, summary, colors, out):
    front = pareto(summary)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    pts = sorted((s for s in summary if s["sid"] in front), key=lambda s: s["tokens_per_episode"])
    if len(pts) > 1:
        ax.plot([s["tokens_per_episode"] for s in pts], [s["succ_rate"] for s in pts],
                color=MUTED, linewidth=1.5, linestyle="--", zorder=1, label="파레토 전선")
    for s in summary:
        on = s["sid"] in front
        ax.scatter(s["tokens_per_episode"], s["succ_rate"], s=90 if on else 60,
                   color=colors[s["sid"]], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(s["sid"] + (" ★" if on else ""), (s["tokens_per_episode"], s["succ_rate"]),
                    xytext=(7, 5), textcoords="offset points", fontsize=9, color=INK2)
    ax.set_xlabel("에피소드당 평균 토큰")
    ax.set_ylabel("성공률")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylim(-0.03, 1.05)
    if len(pts) > 1:
        ax.legend(frameon=False, loc="lower right")
    ax.set_title("비용 대비 성공률 (왼쪽 위일수록 좋음, ★ = 파레토 전선)", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "cost_vs_success.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_expansions(plt, summary, rows, colors, out):
    sids = [s["sid"] for s in summary]
    max_e = max(int(r["expansions"]) for r in rows)
    xs = list(range(max_e + 1))
    width = 0.8 / len(sids)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, sid in enumerate(sids):
        rs = [r for r in rows if r["sid"] == sid]
        share = [sum(int(r["expansions"]) == e for r in rs) / len(rs) for e in xs]
        ax.bar([x + (i - (len(sids) - 1) / 2) * width for x in xs], share, width=width * 0.92,
               color=colors[sid], label=sid, zorder=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.set_xlabel("에피소드당 계획 확장 횟수")
    ax.set_ylabel("에피소드 비율")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="x", visible=False)
    _legend(ax, len(sids))
    ax.set_title("확장 횟수 분포", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "expansions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tiers(plt, summary, rows, colors, tiers_csv, out) -> str:
    with open(tiers_csv, encoding="utf-8") as f:
        tier = {int(r["task"]): int(r["tier"]) for r in csv.DictReader(f)}
    sids = [s["sid"] for s in summary]
    levels = [1, 2, 3, 4, 5]
    rate, count = {}, {}
    for sid in sids:
        for lv in levels:
            rs = [r for r in rows if r["sid"] == sid and tier.get(int(r["task"])) == lv]
            count[(sid, lv)] = len(rs)
            rate[(sid, lv)] = sum(bool(r["success"]) for r in rs) / len(rs) if rs else None
    width = 0.8 / len(sids)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for i, sid in enumerate(sids):
        xs = [lv + (i - (len(sids) - 1) / 2) * width for lv in levels]
        ax.bar(xs, [rate[(sid, lv)] or 0 for lv in levels], width=width * 0.92,
               color=colors[sid], label=sid, zorder=2)
    labels = [f"{lv}단계\n(과제 {sum(1 for t in tier.values() if t == lv)})" for lv in levels]
    ax.set_xticks(levels, labels)
    ax.set_ylabel("성공률")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="x", visible=False)
    _legend(ax, len(sids))
    ax.set_title("난이도 단계별 성공률 (단계는 기준 전략으로 매김)", loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "success_by_tier.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    head = "| 전략 | " + " | ".join(f"{lv}단계" for lv in levels) + " |\n|---|" + "---:|" * 5 + "\n"
    body = "".join(
        f"| {sid} | " + " | ".join("-" if rate[(sid, lv)] is None
                                    else f"{rate[(sid, lv)]:.0%} (n={count[(sid, lv)]})" for lv in levels) + " |\n"
        for sid in sids)
    return "\n## 난이도 단계별 성공률\n\n" + head + body


# ---------------------------------------------------------------- 진입점

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="trials.jsonl 형식 파일 (여러 개 가능)")
    ap.add_argument("--labels", default=None, help="파일별 라벨, 쉼표로 구분 (주면 sid 대신 사용)")
    ap.add_argument("--tiers", default=None, help="run.py difficulty 가 만든 tiers.csv")
    ap.add_argument("--out", default="report")
    ap.add_argument("--max-series", type=int, default=6,
                    help="그룹 막대 그래프에 보여 줄 전략 수 (성공률 상위, baseline은 항상 포함)")
    ap.add_argument("--no-plots", action="store_true", help="표만 만들기 (matplotlib 없어도 됨)")
    args = ap.parse_args()

    labels = args.labels.split(",") if args.labels else None
    if labels and len(labels) != len(args.files):
        ap.error("--labels 개수가 파일 개수와 달라요")
    os.makedirs(args.out, exist_ok=True)

    rows = load(args.files, labels)
    summary = summarize(rows)
    md = write_tables(summary, rows, args.out)

    if not args.no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _style(plt)
        colors = color_map(summary)          # 전체 기준으로 색을 정해, 일부만 그려도 색이 안 바뀜
        shown = pick_series(summary, args.max_series)
        plot_success(plt, summary, colors, args.out)
        plot_cost(plt, summary, colors, args.out)
        plot_expansions(plt, shown, rows, colors, args.out)
        if args.tiers:
            md += plot_tiers(plt, shown, rows, colors, args.tiers, args.out)
            with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
                f.write(md)

    print(md)
    print(f"[report] 저장: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
