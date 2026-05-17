"""
Plot progress chart from results.jsonl.

Shows each experiment's mean_return, the running best as a green staircase,
discarded runs as gray dots, and a blue dashed line at the 200 pass bar.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PASS_BAR = 200.0


def load_runs(results_file: str) -> list[dict]:
    runs = []
    with open(results_file) as f:
        for line in f:
            runs.append(json.loads(line))
    return runs


def plot(results_file: str = "results.jsonl", output: str = "progress.png"):
    runs = load_runs(results_file)
    if not runs:
        print("No entries found in", results_file)
        return

    n = len(runs)
    run_ids = [r["run_id"] for r in runs]
    rets = [r["mean_return"] for r in runs]
    descriptions = [r.get("description", "") for r in runs]

    running_best = []
    best_so_far = float("-inf")
    kept = []
    for i, r in enumerate(rets):
        if r > best_so_far:
            best_so_far = r
            kept.append(i)
        running_best.append(best_so_far)

    discarded = [i for i in range(n) if i not in kept]

    fig, ax = plt.subplots(figsize=(14, 6))

    if discarded:
        ax.scatter(
            [run_ids[i] for i in discarded],
            [rets[i] for i in discarded],
            color="#cccccc", s=40, zorder=2, label="Discarded",
        )

    kept_x = [run_ids[i] for i in kept]
    kept_y = [rets[i] for i in kept]
    ax.scatter(kept_x, kept_y, color="#2ecc71", s=60, zorder=4, label="Kept")

    stair_x, stair_y = [], []
    for k in kept:
        x, y = run_ids[k], rets[k]
        if stair_x:
            stair_x.append(x)
            stair_y.append(stair_y[-1])
        stair_x.append(x)
        stair_y.append(y)
    if stair_x:
        stair_x.append(run_ids[-1])
        stair_y.append(stair_y[-1])
    ax.plot(stair_x, stair_y, color="#2ecc71", linewidth=1.5,
            zorder=3, label="Running best")

    ax.axhline(PASS_BAR, color="#3498db", linestyle="--", linewidth=1.2,
               label=f"Pass bar ({PASS_BAR:.0f})")

    for i in kept:
        desc = descriptions[i]
        if len(desc) > 50:
            desc = desc[:47] + "..."
        ax.annotate(
            desc,
            (run_ids[i], rets[i]),
            textcoords="offset points",
            xytext=(8, -12),
            fontsize=7,
            color="#2ecc71",
            rotation=25,
            ha="left",
        )

    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel("Mean Return (higher is better)", fontsize=12)
    ax.set_title(
        f"Lunar Lander Progress: {n} Experiments, {len(kept)} Kept Improvements",
        fontsize=13,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=150)
    print(f"Saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.jsonl")
    parser.add_argument("--output", default="progress.png")
    args = parser.parse_args()
    plot(args.results, args.output)
