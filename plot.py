"""读取 results/*.json，画学习曲线（多种子均值 ± 标准差）并输出汇总表 results/summary.md。"""
import glob
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ENVS = {"CartPole-v1": 475, "LunarLander-v3": 200}  # 常用“解决”阈值
ALGOS = ["dqn", "a2c", "ppo"]


def smooth_on_grid(curve, grid, window=20):
    """在每个网格步数处，取此前最近 window 个回合回报的均值。"""
    steps = np.array([s for s, _ in curve]); rets = np.array([r for _, r in curve])
    out = []
    for g in grid:
        k = np.searchsorted(steps, g, side="right")
        out.append(rets[max(0, k - window):k].mean() if k > 0 else np.nan)
    return np.array(out)


def main():
    runs = defaultdict(list)
    for f in sorted(glob.glob("results/*.json")):
        env, algo, seed = f.split("/")[-1][:-5].rsplit("_", 2)
        runs[(env, algo)].append(json.load(open(f)))
    fig, axes = plt.subplots(1, len(ENVS), figsize=(12, 4.5))
    lines = ["| 环境 | 算法 | 种子数 | 最后 20 回合平均回报 (均值 ± 标准差) | 训练中峰值(20回合滑动均值, 种子平均) | 达到阈值(20回合滑动均值)所需步数 | 单次耗时(s) |",
             "|---|---|---|---|---|---|---|"]
    for ax, (env, thr) in zip(axes, ENVS.items()):
        algos = ALGOS + sorted(a for e, a in runs if e == env and a not in ALGOS)
        for algo in algos:
            rs = runs[(env, algo)]
            total = min(r["curve"][-1][0] for r in rs)  # 每个算法只画到自己的步数预算
            grid = np.linspace(total / 100, total, 100)
            ys = np.array([smooth_on_grid(r["curve"], grid) for r in rs])
            m, s = np.nanmean(ys, 0), np.nanstd(ys, 0)
            ax.plot(grid, m, label=f"{algo.upper()} ({len(rs)} seeds)")
            ax.fill_between(grid, m - s, m + s, alpha=0.2)
            finals = [np.mean([x for _, x in r["curve"][-20:]]) for r in rs]
            reach, peaks = [], []
            for r in rs:
                y = smooth_on_grid(r["curve"], [st for st, _ in r["curve"]])
                peaks.append(np.max(y[19:]))
                hit = [st for (st, _), v in zip(r["curve"], y) if v >= thr]
                reach.append(f"{hit[0] // 1000}k" if hit else "未达到")
            secs = np.mean([r["seconds"] for r in rs])
            lines.append(f"| {env} | {algo.upper()} | {len(rs)} | {np.mean(finals):.1f} ± {np.std(finals):.1f} | {np.mean(peaks):.1f} | "
                         f"{' / '.join(reach)} | {secs:.0f} |")
        ax.axhline(thr, ls="--", c="gray", lw=0.8)
        ax.set_title(env); ax.set_xlabel("env steps"); ax.set_ylabel("episode return"); ax.legend()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.tight_layout(); plt.savefig("learning_curves.png", dpi=120)
    open("results/summary.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print("已保存 learning_curves.png, results/summary.md")


if __name__ == "__main__":
    main()
