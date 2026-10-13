"""Compare centrality metrics on the same fixed-N, fixed-M ratio sweep graphs."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

from centrality import METRICS
from network_core import NULL_MODEL_CHOICES
from ratio_sweep import DEFAULT_RATIOS, run_sweep


METRIC_LABELS = {
    "degree": "Degree (DC)",
    "harmonic": "Harmonic (HC)",
    "betweenness": "Betweenness (BC)",
    "pagerank": "PageRank (PR)",
}

def write_comparison_plot(output: Path, rows: list[dict[str, float | str]], metrics: tuple[str, ...]) -> bool:
    try:
        os.environ.setdefault(
            "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "node_position_pvalue_matplotlib")
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plots = (
        ("bonf_recall", "Recall", (-0.04, 1.04)),
        ("bonf_precision", "Precision", (-0.04, 1.04)),
        ("bonf_f1", "F1 score", (-0.04, 1.04)),
    )
    for metric in metrics:
        metric_rows = [row for row in rows if row["metric"] == metric]
        x_values = [float(row["ratio"]) for row in metric_rows]
        for axis, (key, label, limits) in zip(axes.flat, plots):
            axis.plot(x_values, [float(row[key]) for row in metric_rows], marker="o", label=METRIC_LABELS[metric])
            axis.set_ylabel(label)
            if limits is not None:
                axis.set_ylim(*limits)
    for axis in axes.flat:
        axis.set_xlabel("Hub / ordinary edge probability ratio")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)
    return True


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--m", type=int, default=350)
    parser.add_argument("--hubs", type=int, default=3)
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--null-graphs", type=int)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--null-model", choices=NULL_MODEL_CHOICES, default="gnm")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1")
    if args.null_graphs is not None and args.null_graphs < 1:
        parser.error("--null-graphs must be positive")
    null_graphs = args.null_graphs if args.null_graphs is not None else 1000
    args.output = args.output or Path("experiment_results") / (
        "metric_comparison_per_target" if args.null_model == "gnm"
        else f"metric_comparison_{args.null_model}"
    )
    metrics = tuple(dict.fromkeys(args.metrics))
    ratios = tuple(args.ratios)
    args.output.mkdir(parents=True, exist_ok=True)

    combined: list[dict[str, float | str]] = []
    for metric in metrics:
        print(f"Running {METRIC_LABELS[metric]}...")
        summary = run_sweep(
            args.n,
            args.m,
            args.hubs,
            ratios,
            args.repetitions,
            null_graphs,
            args.alpha,
            args.seed,
            args.output / metric,
            metric,
            quiet=True,
            detail_output=False,
            save_edges=metric == "degree",
            null_model=args.null_model,
        )
        combined.extend({"metric": metric, **row} for row in summary)

    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)

    lines = [
        "# 中心性指标比较",
        "",
        f"固定 N={args.n}、M={args.m}、植入中心节点数={args.hubs}；每个倍率重复 {args.repetitions} 次。",
        (f"每种指标使用相同的目标图随机种子；每张目标图独立生成 {null_graphs} 张 GNM 零模型图，"
         "四种指标共享该目标图对应的零图随机种子。"
         if args.null_model == "gnm" else
         f"每种指标使用相同的目标图随机种子；零模型为 {args.null_model}，"
         f"每张目标图各模拟 {null_graphs} 张零模型图，汇总全部节点分数估计 p 值。"),
        f"以下均使用 Bonferroni 校正 p 值 < {args.alpha}（等价于未校正 p 值 < {args.alpha}/{args.n}）。",
        "",
    ]
    lines.extend(
        [
            f"召回率、精确率和 F1 先逐图计算，再对每个倍率的 {args.repetitions} 张图取平均（图级 macro 汇总）。",
            "若某张图没有报告节点，其精确率和 F1 记为 0。",
            "平均报告数和平均误报数均按每张图计算。",
            "",
            "| 倍率 | 中心性指标 | 召回率 | 精确率 | F1 | 平均报告数 | 平均误报数 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for ratio in ratios:
        for metric in metrics:
            row = next(row for row in combined if row["metric"] == metric and row["ratio"] == ratio)
            lines.append(
                f"| {ratio:g} | {METRIC_LABELS[metric]} | {row['bonf_recall']:.1%} | "
                f"{row['bonf_precision']:.1%} | {row['bonf_f1']:.1%} | "
                f"{row['bonf_mean_reported']:.2f} | {row['bonf_mean_fp']:.2f} |"
            )
    lines.append("")
    lines.append("各指标目录保留逐次记录和逐节点校正 p 值；共用目标图的边列表位于 degree 目录。")
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_written = write_comparison_plot(args.output, combined, metrics)
    print(f"Comparison saved to {args.output}; plot written: {plot_written}")


if __name__ == "__main__":
    main()
