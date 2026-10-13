"""Compare exact GNM degree p-values with the simulated DC null on paired graphs."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import tempfile
import time
from bisect import bisect_left
from functools import lru_cache
from pathlib import Path

from centrality import degree
from network_core import Network, build_null_scores, possible_edges, score_network
from ratio_sweep import DEFAULT_RATIOS, classification_rates, hub_edge_distribution


@lru_cache(maxsize=None)
def hypergeometric_tails(n: int, m: int) -> tuple[float, ...]:
    """P(D >= d) for a node's degree D in the uniform G(n, m) model."""
    population = math.comb(n, 2)
    incident_edges = n - 1
    other_edges = population - incident_edges
    if not 0 <= m <= population:
        raise ValueError("m must be between 0 and the number of possible edges")

    low = max(0, m - other_edges)
    high = min(incident_edges, m)
    denominator = math.comb(population, m)
    tails = [1.0] * n
    upper_tail = 0.0
    for d in range(high, low - 1, -1):
        upper_tail += (
            math.comb(incident_edges, d) * math.comb(other_edges, m - d) / denominator
        )
        tails[d] = min(1.0, upper_tail)
    return tuple(tails)


def target_graphs(
    n: int, m: int, k: int, ratios: tuple[float, ...], repetitions: int, seed: int
):
    """Reproduce the paired target graphs from ratio_sweep.run_sweep."""
    distributions = {ratio: hub_edge_distribution(n, m, k, ratio) for ratio in ratios}
    candidates = possible_edges(n)
    for repetition in range(1, repetitions + 1):
        hubs = frozenset(random.Random(f"{seed}:{repetition}:hubs").sample(range(n), k))
        hub_candidates = []
        ordinary_candidates = []
        for u, v in candidates:
            (hub_candidates if u in hubs or v in hubs else ordinary_candidates).append((u, v))
        random.Random(f"{seed}:{repetition}:hub_edges").shuffle(hub_candidates)
        random.Random(f"{seed}:{repetition}:ordinary_edges").shuffle(ordinary_candidates)
        quantile = random.Random(f"{seed}:{repetition}:hub_edge_count").random()
        for ratio in ratios:
            model = distributions[ratio]
            hub_edge_count = model.sample(quantile)
            edges = tuple(hub_candidates[:hub_edge_count] + ordinary_candidates[:m - hub_edge_count])
            name = f"ratio_{ratio:g}_rep{repetition:02d}"
            yield ratio, repetition, Network(
                name, n, edges, hubs, m, k, model.hub_probability, model.ordinary_probability
            )


def result_row(ratio: float, repetition: int, network: Network, method: str, reports: set[int]):
    tp = len(reports & network.hubs)
    fp = len(reports) - tp
    fn = len(network.hubs) - tp
    recall, precision, f1 = classification_rates(tp, fp, fn)
    return {
        "ratio": ratio,
        "repetition": repetition,
        "graph": network.name,
        "method": method,
        "reported": len(reports),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(output: Path, rows: list[dict[str, float | str]]) -> bool:
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
    measures = (("recall", "Recall"), ("precision", "Precision"), ("f1", "F1 score"))
    for method, label, style, options in (
        (
            "hypergeometric",
            "Exact hypergeometric",
            "o-",
            {"color": "tab:blue", "markersize": 10, "markerfacecolor": "none", "zorder": 3},
        ),
        (
            "simulated",
            "Simulated GNM",
            "s--",
            {"color": "tab:orange", "markersize": 5, "zorder": 4},
        ),
    ):
        method_rows = [row for row in rows if row["method"] == method]
        ratios = [float(row["ratio"]) for row in method_rows]
        for axis, (key, ylabel) in zip(axes, measures):
            axis.plot(
                ratios,
                [float(row[key]) for row in method_rows],
                style,
                label=label,
                **options,
            )
            axis.set_ylabel(ylabel)
    for axis in axes:
        axis.set_xlabel("Hub / ordinary edge probability ratio")
        axis.set_ylim(-0.04, 1.04)
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.svg")
    plt.close(fig)
    return True


def run_experiment(args: argparse.Namespace) -> tuple[list[dict[str, float | str]], dict[str, float]]:
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not 0 < args.alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    ratios = tuple(args.ratios)
    if len(set(ratios)) != len(ratios):
        raise ValueError("ratios must be distinct")

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tails = hypergeometric_tails(args.n, args.m)
    exact_null_seconds = time.perf_counter() - started

    started = time.perf_counter()
    simulated_null = build_null_scores(
        args.n,
        args.m,
        args.null_graphs,
        random.Random(f"{args.seed}:null:{args.n}:{args.m}"),
        "degree",
    )
    simulated_null_seconds = time.perf_counter() - started
    exact_threshold = next(
        (d for d, tail in enumerate(tails) if args.n * tail < args.alpha), None
    )
    simulated_threshold = next(
        (
            d
            for d in range(args.n)
            if args.n
            * (len(simulated_null) - bisect_left(simulated_null, d) + 1)
            / (len(simulated_null) + 1)
            < args.alpha
        ),
        None,
    )

    score_seconds = {"hypergeometric": 0.0, "simulated": 0.0}
    replicate_rows: list[dict[str, object]] = []
    differing_nodes = 0
    for ratio, repetition, network in target_graphs(
        args.n, args.m, args.hubs, ratios, args.repetitions, args.seed
    ):
        started = time.perf_counter()
        degrees = degree(network.n, network.edges)
        exact_reports = {
            node
            for node, value in enumerate(degrees)
            if min(1.0, network.n * tails[int(value)]) < args.alpha
        }
        score_seconds["hypergeometric"] += time.perf_counter() - started

        started = time.perf_counter()
        simulated_results = score_network(
            network, simulated_null, args.alpha, "degree", bonferroni_only=True
        )
        simulated_reports = {
            int(row["node"])
            for row in simulated_results
            if row["predicted_hub_bonferroni"]
        }
        score_seconds["simulated"] += time.perf_counter() - started
        differing_nodes += len(exact_reports ^ simulated_reports)

        replicate_rows.append(
            result_row(ratio, repetition, network, "hypergeometric", exact_reports)
        )
        replicate_rows.append(result_row(ratio, repetition, network, "simulated", simulated_reports))

    summary: list[dict[str, float | str]] = []
    for ratio in ratios:
        for method in ("hypergeometric", "simulated"):
            selected = [
                row for row in replicate_rows if row["ratio"] == ratio and row["method"] == method
            ]
            summary.append(
                {
                    "ratio": ratio,
                    "method": method,
                    "recall": sum(float(row["recall"]) for row in selected) / args.repetitions,
                    "precision": sum(float(row["precision"]) for row in selected) / args.repetitions,
                    "f1": sum(float(row["f1"]) for row in selected) / args.repetitions,
                    "mean_reported": sum(int(row["reported"]) for row in selected) / args.repetitions,
                    "mean_fp": sum(int(row["false_positives"]) for row in selected) / args.repetitions,
                }
            )

    write_csv(args.output / "summary.csv", summary)
    write_csv(args.output / "replicates.csv", replicate_rows)
    plot_written = write_plot(args.output, summary)
    timings = {
        "hypergeometric_null_seconds": exact_null_seconds,
        "simulated_null_seconds": simulated_null_seconds,
        "hypergeometric_score_seconds": score_seconds["hypergeometric"],
        "simulated_score_seconds": score_seconds["simulated"],
        "differing_nodes": differing_nodes,
    }
    lines = [
        "# 度中心性：精确超几何分布与 GNM 模拟比较",
        "",
        f"固定 N={args.n}、M={args.m}、中心节点数={args.hubs}；每个倍率模拟 {args.repetitions} 张配对目标图。",
        f"模拟方案使用 {args.null_graphs} 张 GNM 零模型图；超几何方案不生成零模型图。",
        f"两种方案均按校正 p 值 < {args.alpha} 判定中心节点。",
        "本次显著性度数门槛：超几何 "
        f"{'无' if exact_threshold is None else f'≥{exact_threshold}'}；GNM 模拟 "
        f"{'无' if simulated_threshold is None else f'≥{simulated_threshold}'}。",
        "召回率、精确率和 F1 先逐图计算，再对同倍率目标图取平均；单图无报告时，精确率和 F1 记为 0。",
        "",
        "| 倍率 | 方法 | 召回率 | 精确率 | F1 | 平均报告数 | 平均误报数 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        method = "精确超几何" if row["method"] == "hypergeometric" else "GNM 模拟"
        lines.append(
            f"| {row['ratio']:g} | {method} | {row['recall']:.1%} | {row['precision']:.1%} | "
            f"{row['f1']:.1%} | {row['mean_reported']:.2f} | {row['mean_fp']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 耗时",
            "",
            "以下仅计零分布准备及目标图节点判定；两种方法共用目标图生成过程。",
            "",
            f"- 超几何尾概率表：{exact_null_seconds:.4f} 秒；GNM 模拟零分布：{simulated_null_seconds:.4f} 秒。",
            f"- 对全部目标图判定：超几何 {score_seconds['hypergeometric']:.4f} 秒；GNM 模拟 {score_seconds['simulated']:.4f} 秒。",
            f"- 两种方法判定不同的节点数：{differing_nodes}。",
            "",
            f"对比图已生成：{plot_written}。",
        ]
    )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, timings


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--m", type=int, default=350)
    parser.add_argument("--hubs", type=int, default=3)
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--null-graphs", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("experiment_results/ratio_sweep_fast"))
    args = parser.parse_args()
    _, timings = run_experiment(args)
    print(f"Comparison saved to {args.output}")
    print(
        "Null preparation: hypergeometric "
        f"{timings['hypergeometric_null_seconds']:.4f}s, "
        f"simulated GNM {timings['simulated_null_seconds']:.4f}s"
    )


if __name__ == "__main__":
    main()
