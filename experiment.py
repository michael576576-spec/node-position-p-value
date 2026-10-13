"""Empirical centrality p-values for simulated network hubs.

Run ``python experiment.py`` to evaluate 10 negative and 10 positive parameter
sets, with 50 independently seeded graphs per setting.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from centrality import METRICS
from network_core import NULL_MODEL_CHOICES, Network, detect_hubs, gnm_edges, possible_edges


def make_negative(name: str, n: int, m: int, rng: random.Random) -> Network:
    return Network(name, n, gnm_edges(n, m, rng), frozenset(), m, 0, None, None)


def make_positive(
    name: str,
    n: int,
    target_m: int,
    hub_count: int,
    probability_ratio: float,
    rng: random.Random,
) -> Network:
    """Draw edges independently, with a larger probability for hub-incident edges.

    ``target_m`` is the expected edge count. The realized edge count may differ;
    inference fits the selected null using the realized graph.
    """
    candidates = possible_edges(n)
    if not 1 <= hub_count < n:
        raise ValueError("hub_count must be between 1 and n - 1")
    if probability_ratio <= 1:
        raise ValueError("probability_ratio must exceed 1")
    if not 0 < target_m < len(candidates):
        raise ValueError("target_m must be between 1 and the number of possible edges")

    hubs = frozenset(rng.sample(range(n), hub_count))
    ordinary_pairs = (n - hub_count) * (n - hub_count - 1) // 2
    hub_pairs = len(candidates) - ordinary_pairs
    background_probability = target_m / (ordinary_pairs + probability_ratio * hub_pairs)
    hub_probability = probability_ratio * background_probability
    if hub_probability > 1:
        raise ValueError("parameters imply a hub edge probability above 1")

    edges = tuple(
        (u, v)
        for u, v in candidates
        if rng.random() < (hub_probability if u in hubs or v in hubs else background_probability)
    )
    return Network(
        name,
        n,
        edges,
        hubs,
        target_m,
        hub_count,
        hub_probability,
        background_probability,
    )


# Ten distinct parameter settings per class. Positive settings vary hub number
# and edge probability ratio; both classes vary N and target M.
SPECS = (
    (60, 90, 1, 3.0),
    (65, 120, 2, 4.0),
    (70, 150, 3, 5.0),
    (75, 180, 1, 6.0),
    (80, 220, 4, 3.5),
    (90, 280, 2, 5.0),
    (100, 350, 5, 4.0),
    (110, 450, 3, 6.0),
    (125, 600, 4, 5.5),
    (150, 900, 5, 7.0),
)

SUMMARY_COLUMNS = [
    "网络名称",
    "节点数",
    "平均边数",
    "正例节点数",
    "正例节点连边概率",
    "普通节点连边概率",
    "报告的中心节点数",
]

COMPARISON_COLUMNS = [
    "网络名称",
    "节点数",
    "真实中心节点数",
    "校正前平均报告数",
    "校正前平均检出数",
    "校正前平均误报数",
    "Bonferroni后平均报告数",
    "Bonferroni后平均检出数",
    "Bonferroni后平均误报数",
]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    seed: int,
    null_graphs: int,
    alpha: float,
    output: Path,
    repetitions: int = 50,
    metric: str = "degree",
    null_model: str = "gnm",
) -> None:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")

    summary_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    node_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    replicate_rows: dict[str, list[dict[str, object]]] = {"negative": [], "positive": []}

    for kind in ("negative", "positive"):
        for index, (n, target_m, hub_count, ratio) in enumerate(SPECS, start=1):
            name = f"{kind}_{index:02d}"
            reported_counts: list[int] = []
            true_positive_counts: list[int] = []
            bonferroni_reported_counts: list[int] = []
            bonferroni_true_positive_counts: list[int] = []
            observed_edges: list[int] = []
            hub_probability: float | None = None
            background_probability: float | None = None

            for repetition in range(1, repetitions + 1):
                graph_seed = f"{seed}:{name}:{repetition}:graph"
                null_seed = f"{seed}:{name}:{repetition}:null"
                replicate_name = f"{name}_rep{repetition:02d}"
                graph_rng = random.Random(graph_seed)
                network = (
                    make_negative(replicate_name, n, target_m, graph_rng)
                    if kind == "negative"
                    else make_positive(replicate_name, n, target_m, hub_count, ratio, graph_rng)
                )
                results = detect_hubs(
                    network, null_graphs, alpha, random.Random(null_seed), metric, null_model
                )
                reported = sum(bool(row["predicted_hub"]) for row in results)
                true_positives = sum(
                    bool(row["predicted_hub"] and row["true_hub"]) for row in results
                )
                bonferroni_reported = sum(
                    bool(row["predicted_hub_bonferroni"]) for row in results
                )
                bonferroni_true_positives = sum(
                    bool(row["predicted_hub_bonferroni"] and row["true_hub"])
                    for row in results
                )
                reported_counts.append(reported)
                true_positive_counts.append(true_positives)
                bonferroni_reported_counts.append(bonferroni_reported)
                bonferroni_true_positive_counts.append(bonferroni_true_positives)
                observed_edges.append(network.m)
                hub_probability = network.hub_probability
                background_probability = network.background_probability
                replicate_rows[kind].append(
                    {
                        "parameter_set": name,
                        "replicate": repetition,
                        "graph_seed": graph_seed,
                        "null_seed": null_seed,
                        "observed_m": network.m,
                        "reported_hubs": reported,
                        "true_positives": true_positives,
                        "false_positives": reported - true_positives,
                        "bonferroni_reported_hubs": bonferroni_reported,
                        "bonferroni_true_positives": bonferroni_true_positives,
                        "bonferroni_false_positives": (
                            bonferroni_reported - bonferroni_true_positives
                        ),
                    }
                )
                node_rows.extend({"network": replicate_name, **row} for row in results)
                edge_rows.extend(
                    {"network": replicate_name, "source": u, "target": v}
                    for u, v in network.edges
                )

            summary_rows.append(
                {
                    "网络名称": name,
                    "节点数": n,
                    "平均边数": f"{sum(observed_edges) / repetitions:.2f}",
                    "正例节点数": hub_count if kind == "positive" else 0,
                    "正例节点连边概率": (
                        f"{hub_probability:.6f}" if hub_probability is not None else "null"
                    ),
                    "普通节点连边概率": (
                        f"{background_probability:.6f}"
                        if background_probability is not None
                        else "null"
                    ),
                    "报告的中心节点数": f"{sum(reported_counts) / repetitions:.2f}",
                }
            )
            comparison_rows.append(
                {
                    "网络名称": name,
                    "节点数": n,
                    "真实中心节点数": hub_count if kind == "positive" else 0,
                    "校正前平均报告数": f"{sum(reported_counts) / repetitions:.2f}",
                    "校正前平均检出数": f"{sum(true_positive_counts) / repetitions:.2f}",
                    "校正前平均误报数": (
                        f"{(sum(reported_counts) - sum(true_positive_counts)) / repetitions:.2f}"
                    ),
                    "Bonferroni后平均报告数": (
                        f"{sum(bonferroni_reported_counts) / repetitions:.2f}"
                    ),
                    "Bonferroni后平均检出数": (
                        f"{sum(bonferroni_true_positive_counts) / repetitions:.2f}"
                    ),
                    "Bonferroni后平均误报数": (
                        f"{(sum(bonferroni_reported_counts) - sum(bonferroni_true_positive_counts)) / repetitions:.2f}"
                    ),
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "network_summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(output / "bonferroni_comparison.csv", comparison_rows, COMPARISON_COLUMNS)
    for kind, rows in replicate_rows.items():
        write_csv(output / f"{kind}_replicates.csv", rows, list(rows[0]))
    write_csv(output / "node_results.csv", node_rows, list(node_rows[0]))
    write_csv(output / "network_edges.csv", edge_rows, ["network", "source", "target"])
    markdown_lines = [
        "| " + " | ".join(SUMMARY_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in SUMMARY_COLUMNS) + " |",
    ]
    markdown_lines.extend(
        "| " + " | ".join(str(row[column]) for column in SUMMARY_COLUMNS) + " |"
        for row in summary_rows
    )
    markdown_table = "\n".join(markdown_lines) + (
        f"\n\n中心性指标：{metric}。每组参数独立模拟 {repetitions} 张图。“平均边数”和“报告的中心节点数”"
        "均为这些图的平均值；GNM 按每张图的实际边数构造零分布，其他零模型按实际图拟合。\n"
    )
    markdown_table += (
        f"\n零模型：{null_model}；每张目标图使用 {null_graphs} 张零模型图。"
        + ("每张零模型图的全部节点分数参与 p 值估计。" if null_model != "gnm" else "")
        + "\n"
    )
    (output / "network_summary.md").write_text(markdown_table, encoding="utf-8")
    comparison_lines = [
        "| " + " | ".join(COMPARISON_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COMPARISON_COLUMNS) + " |",
    ]
    comparison_lines.extend(
        "| " + " | ".join(str(row[column]) for column in COMPARISON_COLUMNS) + " |"
        for row in comparison_rows
    )
    negative_rows = replicate_rows["negative"]
    positive_rows = replicate_rows["positive"]
    total_planted = repetitions * sum(spec[2] for spec in SPECS)
    comparison_notes = (
        f"\n\n中心性指标：{metric}。每组独立模拟 {repetitions} 张图；原始阈值为 p < {alpha}，"
        f"Bonferroni 阈值为 p < {alpha}/N。\n\n"
        "负例：无误报的网络数由 "
        f"{sum(row['reported_hubs'] == 0 for row in negative_rows)}/{len(negative_rows)} "
        f"变为 {sum(row['bonferroni_reported_hubs'] == 0 for row in negative_rows)}/{len(negative_rows)}；"
        f"误报节点总数由 {sum(int(row['false_positives']) for row in negative_rows)} "
        f"降为 {sum(int(row['bonferroni_false_positives']) for row in negative_rows)}。\n\n"
        f"正例：{total_planted} 个植入节点中，检出数由 "
        f"{sum(int(row['true_positives']) for row in positive_rows)} "
        f"变为 {sum(int(row['bonferroni_true_positives']) for row in positive_rows)}；"
        f"误报节点总数由 {sum(int(row['false_positives']) for row in positive_rows)} "
        f"降为 {sum(int(row['bonferroni_false_positives']) for row in positive_rows)}。\n"
    )
    comparison_notes += f"\n零模型：{null_model}；每张目标图模拟 {null_graphs} 张零模型图。\n"
    (output / "bonferroni_comparison.md").write_text(
        "\n".join(comparison_lines) + comparison_notes, encoding="utf-8"
    )
    print(markdown_table)
    print(f"Wrote summary, Bonferroni comparison, and detailed results to {output}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--metric", choices=METRICS, default="degree")
    parser.add_argument("--null-model", choices=NULL_MODEL_CHOICES, default="gnm")
    parser.add_argument("--null-graphs", type=int, help="null graphs per network")
    parser.add_argument("--repetitions", type=int, default=50, help="graphs per setting")
    parser.add_argument("--alpha", type=float, default=0.05, help="p-value threshold")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1")
    if args.null_graphs is not None and args.null_graphs < 1:
        parser.error("--null-graphs must be positive")
    null_graphs = args.null_graphs if args.null_graphs is not None else 100
    output = args.output or (
        (Path("experiment_results/basic_experiment")
         if args.metric == "degree" else Path("experiment_results") / f"{args.metric}_experiment")
        if args.null_model == "gnm" else
        Path("experiment_results") / f"{args.metric}_{args.null_model}_experiment"
    )
    run_experiment(
        args.seed, null_graphs, args.alpha, output, args.repetitions, args.metric,
        args.null_model,
    )


if __name__ == "__main__":
    main()
