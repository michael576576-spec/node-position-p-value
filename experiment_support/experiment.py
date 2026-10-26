"""Compare centrality metrics and null models on simulated GNP networks.

Run ``python -m experiment_support.experiment``. Defaults follow
``experiment.md``: 10 graphs per setting, alpha 0.01, and 100 null graphs per
target. The experiment supports GNM, configuration, and Chung-Lu nulls.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any

from experiment_support.simulated_method_comparison import (
    ALPHA,
    METRIC_ORDER,
    NULL_GRAPHS,
    PILOT_SETTINGS,
    REPETITIONS,
    SETTINGS,
    empirical_bonferroni_results,
    generate_target,
)
from experiment_support.null_models import NULL_MODELS


METRIC_LABELS = {
    "degree": "DC",
    "pagerank": "PR",
    "betweenness": "BC",
    "closeness": "CC",
    "gpg": "GPG",
}
SUMMARY_FIELDS = [
    "setting",
    "class",
    "replicate",
    "network",
    "N",
    "M",
    "p",
    "planted_hubs",
    "null_model",
    "method",
    "metric",
    "reported_central_nodes",
    "reported_non_central_nodes",
    "false_positives",
    "accuracy",
    "method_runtime_seconds",
]
AGGREGATE_FIELDS = [
    "setting",
    "class",
    "N",
    "p",
    "null_model",
    "method",
    "metric",
    "graphs",
    "mean_reported_central_nodes",
    "mean_reported_non_central_nodes",
    "mean_false_positives",
    "mean_accuracy",
    "mean_method_runtime_seconds",
]


def _evaluate_decisions(
    setting: str,
    graph_class: str,
    replicate: int,
    network_name: str,
    n: int,
    m: int,
    edge_probability: float,
    hubs: frozenset[int],
    method: str,
    metric: str,
    reported: list[bool],
    runtime_seconds: float,
    null_model: str,
) -> dict[str, Any]:
    reported_count = sum(reported)
    true_positives = sum(predicted and node in hubs for node, predicted in enumerate(reported))
    false_positives = reported_count - true_positives
    true_negatives = (n - len(hubs)) - false_positives
    return {
        "setting": setting,
        "class": graph_class,
        "replicate": replicate,
        "network": network_name,
        "N": n,
        "M": m,
        "p": edge_probability,
        "planted_hubs": len(hubs),
        "null_model": null_model,
        "method": method,
        "metric": metric,
        "reported_central_nodes": reported_count,
        "reported_non_central_nodes": n - reported_count,
        "false_positives": false_positives,
        "accuracy": (true_positives + true_negatives) / n,
        "method_runtime_seconds": runtime_seconds,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["setting"], row["class"], row["N"], row["p"], row["null_model"],
            row["method"], row["metric"]
        )
        groups.setdefault(key, []).append(row)

    aggregate_rows = []
    for key, group in groups.items():
        setting, graph_class, n, p, null_model, method, metric = key
        record: dict[str, Any] = {
            "setting": setting,
            "class": graph_class,
            "N": n,
            "p": p,
            "null_model": null_model,
            "method": method,
            "metric": metric,
            "graphs": len(group),
            "mean_reported_central_nodes": fmean(
                row["reported_central_nodes"] for row in group
            ),
            "mean_reported_non_central_nodes": fmean(
                row["reported_non_central_nodes"] for row in group
            ),
            "mean_false_positives": fmean(row["false_positives"] for row in group),
            "mean_accuracy": fmean(row["accuracy"] for row in group),
            "mean_method_runtime_seconds": fmean(
                row["method_runtime_seconds"] for row in group
            ),
        }
        aggregate_rows.append(record)
    return aggregate_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            if value is None:
                values.append("NA")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def run_experiment(
    seed: int = 42,
    null_graphs: int = NULL_GRAPHS,
    alpha: float = ALPHA,
    repetitions: int = REPETITIONS,
    output: Path = Path("experiment_results/benchmark_n500_n1000_five_metrics_v1"),
    save_node_results: bool = False,
    null_models: tuple[str, ...] = ("gnm",),
    settings: tuple[Any, ...] = SETTINGS,
) -> None:
    if null_graphs < 1 or repetitions < 1:
        raise ValueError("null_graphs and repetitions must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    unknown_models = set(null_models) - NULL_MODELS.keys()
    if not null_models or unknown_models:
        raise ValueError(f"null_models must be selected from {', '.join(NULL_MODELS)}")
    settings_path = output / "settings.json"
    if output.exists() and any(output.iterdir()):
        if not settings_path.exists():
            raise ValueError(f"output contains unversioned results: {output}")
        previous_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        if previous_settings.get("comparison_version") != "five_metrics_v1":
            raise ValueError(f"output contains results from another comparison: {output}")
    output.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    node_writer = None
    node_file = None
    if save_node_results:
        node_file = (output / "node_results.csv").open("w", newline="", encoding="utf-8-sig")
        node_fields = [
            "network", "node", "null_model", "method", "metric", "centrality_score", "p_value", "adjusted_p_value",
            "reported", "true_hub",
        ]
        node_writer = csv.DictWriter(node_file, fieldnames=node_fields, extrasaction="ignore")
        node_writer.writeheader()

    try:
        for setting in settings:
            graph_class = "positive" if setting.positive else "negative"
            for replicate in range(1, repetitions + 1):
                network = generate_target(setting, replicate, seed, 6.0)
                print(
                    f"[{setting.name} {graph_class} {replicate}/{repetitions}] "
                    f"N={network.n}, M={network.m}",
                    flush=True,
                )

                for null_model in null_models:
                    print(f"  [proposed / {null_model}]", flush=True)
                    started = time.perf_counter()
                    proposed = empirical_bonferroni_results(
                        network,
                        null_graphs,
                        alpha,
                        random.Random(f"{seed}:{network.name}:{null_model}"),
                        null_model,
                    )
                    elapsed = time.perf_counter() - started
                    for metric in METRIC_ORDER:
                        rows = proposed[metric]
                        metric_label = METRIC_LABELS[metric]
                        summary_rows.append(
                            _evaluate_decisions(
                                setting.name, graph_class, replicate, network.name, network.n,
                                network.m, setting.p, network.hubs, "Proposed", metric_label,
                                [bool(row["reported"]) for row in rows], elapsed, null_model,
                            )
                        )
                        if node_writer:
                            node_writer.writerows(
                                {
                                    **row,
                                    "method": "Proposed",
                                    "metric": metric_label,
                                }
                                for row in rows
                            )

    finally:
        if node_file:
            node_file.close()

    aggregate_rows = _aggregate(summary_rows)
    _write_csv(output / "network_summary.csv", summary_rows, SUMMARY_FIELDS)
    _write_csv(output / "aggregate_summary.csv", aggregate_rows, AGGREGATE_FIELDS)
    (output / "aggregate_summary.md").write_text(
        "# Simulated method comparison\n\n"
        f"Alpha = {alpha}; repetitions per setting = {repetitions}; "
        f"null graphs per target for the proposed method = {null_graphs}; "
        f"null models = {', '.join(null_models)}.\n\n"
        "The table reports mean nodes classified as central, mean nodes classified as non-central, "
        "mean false positives, mean accuracy, and mean method runtime. Classification metrics "
        "are averaged across graphs within each setting.\n\n"
        + _markdown_table(aggregate_rows, AGGREGATE_FIELDS)
        + "\n",
        encoding="utf-8",
    )
    (output / "settings.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "alpha": alpha,
                "repetitions": repetitions,
                "proposed_null_models": list(null_models),
                "comparison_version": "five_metrics_v1",
                "null_graphs_per_target": null_graphs,
                "positive_hub_ratio": 6,
                "positive_hub_fraction": 0.01,
                "metrics": list(METRIC_ORDER),
                "settings": [setting.__dict__ for setting in settings],
                "pilot": settings == PILOT_SETTINGS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote simulation comparison results to {output}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--null-graphs", type=int, default=NULL_GRAPHS)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument(
        "--pilot", action="store_true",
        help="use four 200-node positive/negative settings for a quick pilot",
    )
    parser.add_argument(
        "--null-models", nargs="+", choices=tuple(NULL_MODELS),
        default=("gnm",),
        help="null models for the centrality metrics",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiment_results/benchmark_n500_n1000_five_metrics_v1"),
    )
    parser.add_argument(
        "--save-node-results", action="store_true",
        help="write per-node p-values (can produce a large CSV)",
    )
    args = parser.parse_args()
    if args.null_graphs < 1 or args.repetitions < 1:
        parser.error("--null-graphs and --repetitions must be positive")
    if not 0 < args.alpha < 1:
        parser.error("--alpha must be between 0 and 1")
    run_experiment(
        args.seed, args.null_graphs, args.alpha, args.repetitions, args.output,
        args.save_node_results, tuple(args.null_models),
        PILOT_SETTINGS if args.pilot else SETTINGS,
    )


if __name__ == "__main__":
    main()
