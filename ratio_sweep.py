"""Measure how planted-hub detection changes with the hub edge-probability ratio.

The target graphs have exactly N vertices, M edges, and K designated hubs.
Run ``python ratio_sweep.py`` for the default 10 ratios and 50 paired graph
replicates per ratio.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import tempfile
from bisect import bisect_left
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

from centrality import METRICS
from network_core import (
    NULL_MODEL_CHOICES, Network, build_null_distribution, possible_edges, score_network,
)


DEFAULT_RATIOS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0)


@dataclass(frozen=True)
class HubEdgeDistribution:
    """Number of hub-related edges in a graph conditioned on exactly M edges."""

    counts: tuple[int, ...]
    cumulative_probabilities: tuple[float, ...]
    ordinary_probability: float
    hub_probability: float

    def sample(self, uniform_draw: float) -> int:
        return self.counts[bisect_left(self.cumulative_probabilities, uniform_draw)]


def _log_choose(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hub_edge_distribution(n: int, m: int, k: int, ratio: float) -> HubEdgeDistribution:
    """Calibrate a fixed-M two-class graph to the requested marginal probability ratio.

    If H is the number of hub-related edges, P(H=h) is proportional to
    C(B,h) C(A,M-h) w**h. At w=1 this is exactly the GNM distribution. We
    choose w so E[H]/B divided by E[M-H]/A equals ``ratio``.
    """
    if not 1 <= k <= n - 2:
        raise ValueError("k must leave at least two ordinary nodes")
    total_pairs = math.comb(n, 2)
    if not 0 < m < total_pairs:
        raise ValueError("m must be between 1 and the number of possible edges - 1")
    if not math.isfinite(ratio) or ratio < 1:
        raise ValueError("ratio must be at least 1")

    ordinary_pairs = math.comb(n - k, 2)
    hub_pairs = total_pairs - ordinary_pairs
    ordinary_probability = m / (ordinary_pairs + ratio * hub_pairs)
    hub_probability = ratio * ordinary_probability
    if hub_probability >= 1:
        raise ValueError("requested ratio and M require a hub edge probability below 1")

    target_hub_edges = hub_pairs * hub_probability
    lowest = max(0, m - ordinary_pairs)
    highest = min(hub_pairs, m)
    counts = tuple(range(lowest, highest + 1))
    base_log_weights = tuple(
        _log_choose(hub_pairs, h) + _log_choose(ordinary_pairs, m - h)
        for h in counts
    )

    def weighted_distribution(log_odds: float) -> tuple[float, tuple[float, ...]]:
        log_weights = tuple(base + h * log_odds for h, base in zip(counts, base_log_weights))
        offset = max(log_weights)
        weights = tuple(math.exp(value - offset) for value in log_weights)
        total = sum(weights)
        probabilities = tuple(weight / total for weight in weights)
        mean = sum(h * probability for h, probability in zip(counts, probabilities))
        return mean, probabilities

    if ratio == 1:
        log_odds = 0.0
    else:
        lower, upper = 0.0, 1.0
        while weighted_distribution(upper)[0] < target_hub_edges:
            upper *= 2
        for _ in range(55):
            middle = (lower + upper) / 2
            if weighted_distribution(middle)[0] < target_hub_edges:
                lower = middle
            else:
                upper = middle
        log_odds = (lower + upper) / 2

    mean, probabilities = weighted_distribution(log_odds)
    if abs(mean - target_hub_edges) > 1e-7:
        raise ArithmeticError("failed to calibrate the requested edge probability ratio")
    cumulative = list(accumulate(probabilities))
    cumulative[-1] = 1.0
    return HubEdgeDistribution(counts, tuple(cumulative), ordinary_probability, hub_probability)


def target_graphs(
    n: int, m: int, k: int, ratios: tuple[float, ...], repetitions: int, seed: int
):
    """Reproduce the paired target graphs used by run_sweep."""
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


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def classification_rates(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return recall, precision, and F1; undefined precision is set to zero."""
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return recall, precision, f1


def write_plot(output: Path, summary: list[dict[str, float]]) -> bool:
    try:
        os.environ.setdefault(
            "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "node_position_pvalue_matplotlib")
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    ratios = [row["ratio"] for row in summary]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax in axes:
        ax.set_xticks(ratios)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Hub / ordinary edge probability ratio")

    axes[0].plot(ratios, [row["bonf_recall"] for row in summary], "o-")
    axes[0].set_ylabel("Planted-hub recall")
    axes[0].set_ylim(-0.04, 1.04)

    axes[1].plot(ratios, [row["bonf_precision"] for row in summary], "o-")
    axes[1].set_ylabel("Precision")
    axes[1].set_ylim(-0.04, 1.04)

    axes[2].plot(ratios, [row["bonf_f1"] for row in summary], "o-")
    axes[2].set_ylabel("F1 score")
    axes[2].set_ylim(-0.04, 1.04)

    fig.tight_layout()
    fig.savefig(output / "ratio_sweep.png", dpi=180)
    fig.savefig(output / "ratio_sweep.svg")
    plt.close(fig)
    return True


def run_sweep(
    n: int,
    m: int,
    k: int,
    ratios: tuple[float, ...],
    repetitions: int,
    null_graphs: int,
    alpha: float,
    seed: int,
    output: Path,
    metric: str = "degree",
    quiet: bool = False,
    detail_output: bool = True,
    save_edges: bool = True,
    null_model: str = "gnm",
) -> list[dict[str, float]]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if len(set(ratios)) != len(ratios):
        raise ValueError("ratios must be distinct")
    if null_model not in NULL_MODEL_CHOICES:
        raise ValueError(f"unknown null model '{null_model}'")

    distributions = {ratio: hub_edge_distribution(n, m, k, ratio) for ratio in ratios}
    output.mkdir(parents=True, exist_ok=True)
    shared_null = None
    if null_model == "gnm":
        # The GNM generator uses only n and m; give it a graph with m edges.
        reference_network = Network(
            "null_reference", n, possible_edges(n)[:m], frozenset(), m, 0, None, None
        )
        shared_null = build_null_distribution(
            reference_network, null_graphs, random.Random(f"{seed}:null:{n}:{m}"), metric,
            null_model,
        )

    replicate_rows: list[dict[str, object]] = []
    node_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    for repetition in range(1, repetitions + 1):
        # The same random hubs, edge order, and H quantile are reused across
        # ratios within a replicate, giving a paired comparison.
        hubs = frozenset(random.Random(f"{seed}:{repetition}:hubs").sample(range(n), k))
        hub_candidates = []
        ordinary_candidates = []
        for u, v in possible_edges(n):
            (hub_candidates if u in hubs or v in hubs else ordinary_candidates).append((u, v))
        random.Random(f"{seed}:{repetition}:hub_edges").shuffle(hub_candidates)
        random.Random(f"{seed}:{repetition}:ordinary_edges").shuffle(ordinary_candidates)
        hub_count_quantile = random.Random(f"{seed}:{repetition}:hub_edge_count").random()

        for ratio in ratios:
            model = distributions[ratio]
            hub_edge_count = model.sample(hub_count_quantile)
            edges = tuple(hub_candidates[:hub_edge_count] + ordinary_candidates[:m - hub_edge_count])
            graph_name = f"ratio_{ratio:g}_rep{repetition:02d}"
            network = Network(
                graph_name, n, edges, hubs, m, k, model.hub_probability, model.ordinary_probability
            )
            null_scores = shared_null or build_null_distribution(
                network, null_graphs,
                random.Random(f"{seed}:null:{n}:{m}:{repetition}:{ratio:g}:{null_model}"),
                metric, null_model,
            )
            results = score_network(network, null_scores, alpha, metric, bonferroni_only=True)
            bonf_reported = sum(bool(row["predicted_hub_bonferroni"]) for row in results)
            bonf_tp = sum(
                bool(row["predicted_hub_bonferroni"] and row["true_hub"]) for row in results
            )
            bonf_fp = bonf_reported - bonf_tp
            bonf_fn = k - bonf_tp
            bonf_recall, bonf_precision, bonf_f1 = classification_rates(
                bonf_tp, bonf_fp, bonf_fn
            )
            replicate_rows.append(
                {
                    "ratio": ratio,
                    "repetition": repetition,
                    "graph": graph_name,
                    "hubs": ";".join(map(str, sorted(hubs))),
                    "hub_related_edges": hub_edge_count,
                    "ordinary_edges": m - hub_edge_count,
                    "bonf_reported": bonf_reported,
                    "bonf_true_positives": bonf_tp,
                    "bonf_false_positives": bonf_fp,
                    "bonf_false_negatives": bonf_fn,
                    "bonf_recall": bonf_recall,
                    "bonf_precision": bonf_precision,
                    "bonf_f1": bonf_f1,
                }
            )
            node_rows.extend({"graph": graph_name, "ratio": ratio, **row} for row in results)
            if save_edges:
                edge_rows.extend({"graph": graph_name, "source": u, "target": v} for u, v in edges)

    summary: list[dict[str, float]] = []
    for ratio in ratios:
        rows = [row for row in replicate_rows if row["ratio"] == ratio]
        bonf_fp = sum(int(row["bonf_false_positives"]) for row in rows)
        model = distributions[ratio]
        summary.append(
            {
                "ratio": ratio,
                "ordinary_probability": model.ordinary_probability,
                "hub_probability": model.hub_probability,
                "bonf_recall": sum(float(row["bonf_recall"]) for row in rows) / repetitions,
                "bonf_precision": sum(float(row["bonf_precision"]) for row in rows) / repetitions,
                "bonf_f1": sum(float(row["bonf_f1"]) for row in rows) / repetitions,
                "bonf_mean_reported": sum(int(row["bonf_reported"]) for row in rows) / repetitions,
                "bonf_mean_fp": bonf_fp / repetitions,
            }
        )

    write_csv(output / "replicates.csv", replicate_rows, list(replicate_rows[0]))
    write_csv(output / "node_results.csv", node_rows, list(node_rows[0]))
    if save_edges:
        write_csv(output / "network_edges.csv", edge_rows, ["graph", "source", "target"])
    if not detail_output:
        return summary

    write_csv(output / "summary.csv", summary, list(summary[0]))

    md_columns = [
        ("倍率", "ratio", "g"),
        ("普通边概率", "ordinary_probability", ".4f"),
        ("中心边概率", "hub_probability", ".4f"),
        ("Bonferroni检出率", "bonf_recall", ".1%"),
        ("Bonferroni精确率", "bonf_precision", ".1%"),
        ("Bonferroni F1", "bonf_f1", ".1%"),
        ("Bonferroni平均报告数", "bonf_mean_reported", ".2f"),
        ("Bonferroni平均误报数", "bonf_mean_fp", ".2f"),
    ]
    lines = [
        f"# 固定 N、M、中心节点数的连边概率倍率实验",
        "",
        f"固定 N={n}、M={m}、中心节点数={k}；每个倍率生成 {repetitions} 张图。"
        + (f"所有目标图共用 {null_graphs} 张 GNM({n},{m}) 图构成的经验零分布。"
           if null_model == "gnm" else
           f"零模型：{null_model}；每张目标图生成 {null_graphs} 张零模型图，并汇总全部节点分数估计 p 值。"),
        f"中心性指标：{metric}。",
        f"只报告 Bonferroni 校正结果：校正 p 值 < {alpha}（等价于未校正 p 值 < {alpha}/{n}）。",
        "每张目标图恰好有 M 条边。倍率指中心相关边与普通边的边际连边概率之比；倍率 1 对应 GNM。",
        "同一重复编号在不同倍率下共用中心节点及随机边顺序，以便比较倍率变化。",
        "",
        "| " + " | ".join(name for name, _, _ in md_columns) + " |",
        "| " + " | ".join("---" for _ in md_columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(format(row[key], fmt) for _, key, fmt in md_columns) + " |"
        for row in summary
    )
    lines.extend(
        [
            "",
            "召回率、精确率和 F1 先逐图计算，再对本倍率的图取平均；"
            "单图没有报告任何节点时，精确率和 F1 记为 0。",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    plot_written = write_plot(output, summary)
    if not quiet:
        print("\n".join(lines))
        print(f"Results saved to {output}; plot written: {plot_written}")
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--m", type=int, default=350)
    parser.add_argument("--hubs", type=int, default=3)
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--null-graphs", type=int)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", choices=METRICS, default="degree")
    parser.add_argument("--null-model", choices=NULL_MODEL_CHOICES, default="gnm")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1")
    if args.null_graphs is not None and args.null_graphs < 1:
        parser.error("--null-graphs must be positive")
    null_graphs = args.null_graphs if args.null_graphs is not None else 1000
    output = args.output or Path("experiment_results") / (
        ("ratio_sweep" if args.metric == "degree" else f"ratio_sweep_{args.metric}")
        if args.null_model == "gnm" else f"ratio_sweep_{args.metric}_{args.null_model}"
    )
    run_sweep(
        args.n,
        args.m,
        args.hubs,
        tuple(args.ratios),
        args.repetitions,
        null_graphs,
        args.alpha,
        args.seed,
        output,
        args.metric,
        null_model=args.null_model,
    )


if __name__ == "__main__":
    main()
