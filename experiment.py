"""Degree-based empirical p-values for simulated network hubs.

Run ``python experiment.py`` to evaluate 10 negative and 10 positive parameter
sets, with 50 independently seeded graphs per setting.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Network:
    name: str
    n: int
    edges: tuple[tuple[int, int], ...]
    hubs: frozenset[int]
    target_m: int
    hub_count: int
    hub_probability: float | None
    background_probability: float | None

    @property
    def m(self) -> int:
        return len(self.edges)


@lru_cache(maxsize=None)
def possible_edges(n: int) -> tuple[tuple[int, int], ...]:
    if n < 2:
        raise ValueError("n must be at least 2")
    return tuple((u, v) for u in range(n) for v in range(u + 1, n))


def gnm_edges(n: int, m: int, rng: random.Random) -> tuple[tuple[int, int], ...]:
    """Sample G(n,m): every simple graph with n vertices and m edges is equally likely."""
    candidates = possible_edges(n)
    if not 0 <= m <= len(candidates):
        raise ValueError(f"m must be between 0 and {len(candidates)}")
    return tuple(rng.sample(candidates, m))


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
    inference always fits its G(n,m) null using the realized count.
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


def degrees(n: int, edges: tuple[tuple[int, int], ...]) -> list[int]:
    result = [0] * n
    for u, v in edges:
        result[u] += 1
        result[v] += 1
    return result


def detect_hubs(
    network: Network, null_graphs: int, alpha: float, rng: random.Random
) -> list[dict[str, int | float | bool]]:
    """Compare each degree with all degrees pooled from G(n, observed m) graphs."""
    if null_graphs < 1:
        raise ValueError("null_graphs must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    null_degrees: list[int] = []
    for _ in range(null_graphs):
        null_degrees.extend(degrees(network.n, gnm_edges(network.n, network.m, rng)))
    null_degrees.sort()

    observed_degrees = degrees(network.n, network.edges)
    rows = []
    for position, node in enumerate(
        sorted(range(network.n), key=lambda u: (-observed_degrees[u], u)), start=1
    ):
        degree = observed_degrees[node]
        # Upper tail includes ties. The added 1 avoids an estimated p-value of zero.
        at_least_as_large = len(null_degrees) - bisect_left(null_degrees, degree)
        p_value = (at_least_as_large + 1) / (len(null_degrees) + 1)
        rows.append(
            {
                "position": position,
                "node": node,
                "degree": degree,
                "p_value": p_value,
                "predicted_hub": p_value < alpha,
                "true_hub": node in network.hubs,
            }
        )
    return rows


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


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(seed: int, null_graphs: int, alpha: float, output: Path, repetitions: int = 50) -> None:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")

    summary_rows: list[dict[str, object]] = []
    node_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    replicate_rows: dict[str, list[dict[str, object]]] = {"negative": [], "positive": []}

    for kind in ("negative", "positive"):
        for index, (n, target_m, hub_count, ratio) in enumerate(SPECS, start=1):
            name = f"{kind}_{index:02d}"
            reported_counts: list[int] = []
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
                results = detect_hubs(network, null_graphs, alpha, random.Random(null_seed))
                reported = sum(bool(row["predicted_hub"]) for row in results)
                true_positives = sum(
                    bool(row["predicted_hub"] and row["true_hub"]) for row in results
                )
                reported_counts.append(reported)
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

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "network_summary.csv", summary_rows, SUMMARY_COLUMNS)
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
        f"\n\n每组参数独立模拟 {repetitions} 张图。“平均边数”和“报告的中心节点数”"
        "均为这些图的平均值；每张图的零分布按其实际边数构造。\n"
    )
    (output / "network_summary.md").write_text(markdown_table, encoding="utf-8")
    print(markdown_table)
    print(f"Wrote summary tables and detailed results to {output}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--null-graphs", type=int, default=100, help="GNM null graphs per network")
    parser.add_argument("--repetitions", type=int, default=50, help="graphs per setting")
    parser.add_argument("--alpha", type=float, default=0.05, help="p-value threshold")
    parser.add_argument("--output", type=Path, default=Path("experiment_results"))
    args = parser.parse_args()
    run_experiment(args.seed, args.null_graphs, args.alpha, args.output, args.repetitions)


if __name__ == "__main__":
    main()
