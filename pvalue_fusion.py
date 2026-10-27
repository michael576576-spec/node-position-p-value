"""Empirically calibrate an equal-weight combination of five centrality p-values.

Run with ``python pvalue_fusion.py --suite all``. Each target
uses independent reference (A) and calibration (B) null-graph streams.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiment_support.centrality import METRICS, compute_multiple_scores_igraph
from experiment_support.network_core import Network
from experiment_support.null_models import NULL_MODELS
from experiment_support.ratio_sweep import classification_rates, target_graphs
from experiment_support.real_network_experiment import gnm_edges_scalable, load_edgelist
from experiment_support.simulated_method_comparison import SETTINGS, generate_target
from experiment_support.simulation_data import _gnm_edges_for_simulation


METRIC_ORDER = tuple(METRICS)
MODELS = ("gnm", "configuration", "chung_lu")
RATIOS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
VERSION = "equal_weight_fusion_v3"


@dataclass(frozen=True)
class Case:
    suite: str
    name: str
    group: str
    graph_class: str
    model: str
    alpha: float
    reference_seed: str
    calibration_seed: str
    target_sha256: str
    network: Network | None = None
    source: str | None = None


def upper_tail_p(reference: np.ndarray, values: np.ndarray,
                 *, already_sorted: bool = False) -> np.ndarray:
    """Smoothed pooled-node right tail, counting ties as exceedances."""
    if len(reference) == 0:
        raise ValueError("reference scores cannot be empty")
    sorted_reference = np.asarray(reference, dtype=float)
    if not already_sorted:
        sorted_reference = np.sort(sorted_reference)
    values = np.asarray(values, dtype=float)
    if not np.isfinite(sorted_reference).all() or not np.isfinite(values).all():
        raise ValueError("centrality and fusion scores must be finite")
    return (len(sorted_reference) - np.searchsorted(sorted_reference, values, side="left") + 1) / (len(sorted_reference) + 1)


def fusion_score(p_values: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    contributions = {metric: -2.0 * np.log(p_values[metric]) for metric in METRIC_ORDER}
    return sum(contributions.values()), contributions


def edge_digest(network: Network) -> str:
    digest = hashlib.sha256()
    digest.update(f"{network.n}:{network.m}:".encode("ascii"))
    for u, v in network.edges:
        digest.update(f"{u},{v};".encode("ascii"))
    return digest.hexdigest()


def make_cases(suite: str, limit: int | None = None) -> list[Case]:
    cases: list[Case] = []
    if suite == "benchmark":
        seed = 42
        targets = [
            (setting, generate_target(setting, repetition, seed, 6.0))
            for setting in SETTINGS for repetition in range(1, 11)
        ]
        for model in MODELS:
            for setting, network in targets:
                cases.append(Case(
                    suite, network.name, setting.name,
                    "positive" if setting.positive else "negative", model, 0.01,
                    f"{seed}:{network.name}:{model}",
                    f"{seed}:fusion-cal:benchmark:{network.name}:{model}",
                    edge_digest(network), network=network,
                ))
    elif suite == "strength":
        seed = 42
        targets = list(target_graphs(100, 350, 6, RATIOS, 10, seed))
        for model in MODELS:
            for ratio, repetition, network in targets:
                cases.append(Case(
                    suite, network.name, f"ratio_{ratio:g}", "positive", model, 0.01,
                    f"{seed}:null:{network.n}:{network.m}:{repetition}:{ratio:g}:{model}",
                    f"{seed}:fusion-cal:strength:{network.name}:{model}",
                    edge_digest(network), network=network,
                ))
    elif suite == "real":
        seed = 20261014
        sources = sorted(Path("network").glob("*.txt"), key=lambda path: path.stem.lower())
        if not sources:
            raise ValueError("no real-network edge lists found in network/*.txt")
        source_hashes = [(source, hashlib.sha256(source.read_bytes()).hexdigest()) for source in sources]
        for model in MODELS:
            for source, source_sha in source_hashes:
                cases.append(Case(
                    suite, source.stem, "real", "unlabeled", model, 0.01,
                    f"{seed}:{model}:{source.stem}:{source_sha}",
                    f"{seed}:fusion-cal:real:{model}:{source.stem}:{source_sha}",
                    source_sha, source=str(source),
                ))
    else:
        raise ValueError(f"unknown suite: {suite}")
    return cases[:limit] if limit is not None else cases


def target_data(case: Case) -> tuple[Network, list[str]]:
    if case.network is not None:
        if edge_digest(case.network) != case.target_sha256:
            raise ValueError(f"target graph changed: {case.name}")
        return case.network, [str(node) for node in range(case.network.n)]
    if case.source is None:
        raise ValueError(f"missing target source: {case.name}")
    source = Path(case.source)
    if hashlib.sha256(source.read_bytes()).hexdigest() != case.target_sha256:
        raise ValueError(f"real-network source changed: {source}")
    labels, edges, _ = load_edgelist(source)
    n = len(labels)
    return Network(case.name, n, edges, frozenset(), len(edges), 0, None, None), labels


def null_edges(case: Case, network: Network, rng: random.Random,
               offsets: list[int]) -> tuple[tuple[int, int], ...]:
    if case.model != "gnm":
        return NULL_MODELS[case.model](network.n, network.edges, rng)
    if case.suite == "benchmark":
        return _gnm_edges_for_simulation(network.n, network.m, rng)
    if case.suite == "real":
        return gnm_edges_scalable(network.n, network.m, rng, offsets)
    return NULL_MODELS[case.model](network.n, network.edges, rng)


def analyze_case(case: Case, reference_graphs: int, calibration_graphs: int) -> dict:
    network, labels = target_data(case)
    n = network.n
    observed = {
        metric: np.asarray(values, dtype=float)
        for metric, values in compute_multiple_scores_igraph(n, network.edges, METRIC_ORDER).items()
    }
    offsets = [node * (2 * n - node - 1) // 2 for node in range(n)]
    reference_rng = random.Random(case.reference_seed)
    reference_values: dict[str, list[float]] = {metric: [] for metric in METRIC_ORDER}
    started = time.perf_counter()
    for _ in range(reference_graphs):
        edges = null_edges(case, network, reference_rng, offsets)
        scores = compute_multiple_scores_igraph(n, edges, METRIC_ORDER)
        for metric in METRIC_ORDER:
            reference_values[metric].extend(scores[metric])
    reference = {
        metric: np.sort(np.asarray(reference_values[metric], dtype=float))
        for metric in METRIC_ORDER
    }
    expected = n * reference_graphs
    if any(len(values) != expected for values in reference.values()):
        raise ArithmeticError("wrong number of reference node scores")
    observed_p = {
        metric: upper_tail_p(reference[metric], observed[metric], already_sorted=True)
        for metric in METRIC_ORDER
    }
    target_score, contributions = fusion_score(observed_p)

    calibration_rng = random.Random(case.calibration_seed)
    calibration_scores = []
    for _ in range(calibration_graphs):
        edges = null_edges(case, network, calibration_rng, offsets)
        scores = compute_multiple_scores_igraph(n, edges, METRIC_ORDER)
        null_p = {
            metric: upper_tail_p(reference[metric], np.asarray(scores[metric], dtype=float), already_sorted=True)
            for metric in METRIC_ORDER
        }
        calibration_scores.append(fusion_score(null_p)[0])
    calibration = np.sort(np.concatenate(calibration_scores))
    if len(calibration) != n * calibration_graphs:
        raise ArithmeticError("wrong number of calibration node scores")
    ensemble_p = upper_tail_p(calibration, target_score, already_sorted=True)
    adjusted = np.minimum(1.0, n * ensemble_p)
    order = sorted(range(n), key=lambda node: (-float(target_score[node]), node))
    ranks = {node: rank for rank, node in enumerate(order, 1)}
    seconds = time.perf_counter() - started
    rows = []
    for node in range(n):
        row = {
            "suite": case.suite, "group": case.group, "class": case.graph_class,
            "graph": case.name, "null_model": case.model, "N": n, "M": network.m,
            "node": labels[node], "node_index": node, "rank": ranks[node],
            "fusion_score": float(target_score[node]),
            "fusion_p": float(ensemble_p[node]), "fusion_bonferroni_p": float(adjusted[node]),
            "fusion_reported": bool(adjusted[node] < case.alpha),
            "true_hub": node in network.hubs if case.suite != "real" else None,
        }
        for metric in METRIC_ORDER:
            row[f"score_{metric}"] = float(observed[metric][node])
            row[f"p_{metric}"] = float(observed_p[metric][node])
            row[f"contribution_{metric}"] = float(contributions[metric][node])
        rows.append(row)
    return {
        "case": {key: getattr(case, key) for key in (
            "suite", "name", "group", "graph_class", "model", "alpha",
            "reference_seed", "calibration_seed", "target_sha256",
        )},
        "N": n, "M": network.m, "hubs": sorted(network.hubs),
        "reference_graphs": reference_graphs, "calibration_graphs": calibration_graphs,
        "seconds": seconds, "nodes": rows,
    }


def checkpoint_path(root: Path, case: Case) -> Path:
    return root / "checkpoints" / case.model / f"{case.name}.json"


def run_case(case: Case, reference_graphs: int, calibration_graphs: int,
             signature: str, path: Path) -> str:
    result = analyze_case(case, reference_graphs, calibration_graphs)
    result["settings_signature"] = signature
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)
    return f"{case.name}/{case.model}"


def _ranking_quality(ordered_nodes: list[dict], hubs: set[int]) -> tuple[float, float]:
    k = len(hubs)
    if not k:
        return math.nan, math.nan
    hits = 0
    precision_sum = 0.0
    for position, row in enumerate(ordered_nodes, 1):
        if int(row["node_index"]) in hubs:
            hits += 1
            precision_sum += hits / position
    precision_at_k = sum(int(row["node_index"]) in hubs for row in ordered_nodes[:k]) / k
    return precision_sum / k, precision_at_k


def summarize_case(payload: dict) -> list[dict]:
    nodes = payload["nodes"]
    n = payload["N"]
    hubs = set(payload["hubs"])
    case = payload["case"]
    results = []
    for method in ("fusion", *METRIC_ORDER):
        if method == "fusion":
            ordered = sorted(nodes, key=lambda row: (row["rank"], row["node_index"]))
            reported = [row for row in nodes if row["fusion_reported"]]
        else:
            ordered = sorted(nodes, key=lambda row: (-row[f"score_{method}"], row["node_index"]))
            reported = [row for row in nodes if n * row[f"p_{method}"] < case["alpha"]]
        ap, precision_at_k = _ranking_quality(ordered, hubs)
        if case["suite"] == "real":
            tp = fp = recall = precision = f1 = ""
        else:
            tp = sum(int(row["node_index"]) in hubs for row in reported)
            fp = len(reported) - tp
            recall, precision, f1 = classification_rates(tp, fp, len(hubs) - tp)
        results.append({
            "suite": case["suite"], "group": case["group"], "class": case["graph_class"],
            "graph": case["name"], "null_model": case["model"], "N": n, "M": payload["M"],
            "planted_hubs": len(hubs) if case["suite"] != "real" else "",
            "method": method, "reported_nodes": len(reported),
            "true_positives": tp, "false_positives": fp,
            "recall": recall, "precision": precision, "f1": f1,
            "average_precision": ap if math.isfinite(ap) else "",
            "precision_at_k": precision_at_k if math.isfinite(precision_at_k) else "",
            "seconds_per_target": payload["seconds"],
        })
    return results


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (row["suite"], row["group"], row["class"], row["null_model"], row["method"])
        groups.setdefault(key, []).append(row)
    result = []
    for (suite, group, graph_class, model, method), selected in groups.items():
        item = {
            "suite": suite, "group": group, "class": graph_class,
            "null_model": model, "method": method, "graphs": len(selected),
            "mean_reported_nodes": statistics.mean(row["reported_nodes"] for row in selected),
            "graphs_with_any_report": sum(row["reported_nodes"] > 0 for row in selected),
        }
        for key in ("false_positives", "recall", "precision", "f1", "average_precision", "precision_at_k"):
            values = [float(row[key]) for row in selected if isinstance(row[key], (int, float)) and math.isfinite(row[key])]
            item[f"mean_{key}"] = statistics.mean(values) if values else ""
        result.append(item)
    return result


def summarize_suite(root: Path, cases: list[Case], signature: str) -> None:
    node_rows = []
    graph_rows = []
    for case in cases:
        path = checkpoint_path(root, case)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("settings_signature") != signature or payload["case"]["target_sha256"] != case.target_sha256:
            raise ValueError(f"checkpoint differs from experiment: {path}")
        node_rows.extend(payload["nodes"])
        graph_rows.extend(summarize_case(payload))
    write_csv(root / "node_results.csv", node_rows)
    write_csv(root / "graph_summary.csv", graph_rows)
    aggregate_rows = aggregate(graph_rows)
    write_csv(root / "aggregate_summary.csv", aggregate_rows)
    if cases[0].suite == "real":
        write_csv(root / "top_nodes.csv", sorted(
            (row for row in node_rows if row["rank"] <= 10),
            key=lambda row: (row["graph"].lower(), row["null_model"], row["rank"]),
        ))
    alpha = cases[0].alpha
    calibration_graphs = payload["calibration_graphs"]
    lines = [
        f"# Equal-weight empirical p-value fusion: {cases[0].suite}", "",
        f"Target/model cases: {len(cases)}. Target-node rows: {len(node_rows)}.",
        f"Reference A: {payload['reference_graphs']} graphs per case; calibration B: {calibration_graphs} independent graphs per case.",
        "Metrics: DC, PR, BC, CC, GPG. Individual p = (1 + A-node scores >= score) / (1 + A-node count).",
        "Fusion score S = -2 sum(log(individual p)); fusion p = (1 + B-node scores >= S) / (1 + B-node count).",
        f"Reported nodes satisfy N * fusion p < {alpha}; only target nodes are ranked.",
        "Real-network data have no hub labels; classification metrics are blank." if cases[0].suite == "real" else
        "AP and Precision@K measure ranking quality; recall, precision and F1 use Bonferroni-significant reports.",
        "",
        "## Fusion results by setting", "",
    ]
    if cases[0].suite == "strength":
        lines.insert(8, "Configuration-model DC compares against pooled null nodes; it is not degree significance conditional on a fixed node.")
    if cases[0].suite == "real":
        lines.extend(["| Model | Graphs | Mean reports | Graphs with reports |", "| --- | ---: | ---: | ---: |"])
        for row in aggregate_rows:
            if row["method"] == "fusion":
                lines.append(f"| {row['null_model']} | {row['graphs']} | {row['mean_reported_nodes']:.3f} | {row['graphs_with_any_report']} |")
    else:
        lines.extend(["| Setting | Model | Graphs | Mean reports | Mean recall | Mean AP |", "| --- | --- | ---: | ---: | ---: | ---: |"])
        for row in aggregate_rows:
            if row["method"] == "fusion":
                ap = row["mean_average_precision"]
                lines.append(f"| {row['group']} | {row['null_model']} | {row['graphs']} | {row['mean_reported_nodes']:.3f} | {row['mean_recall']:.3f} | {ap:.3f} |" if ap != "" else
                             f"| {row['group']} | {row['null_model']} | {row['graphs']} | {row['mean_reported_nodes']:.3f} | {row['mean_recall']:.3f} | — |")
    lines.extend(["", "See aggregate_summary.csv for the five single-metric comparisons and graph_summary.csv for each target."])
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(suite: str, output: Path, reference_graphs: int,
              calibration_graphs: int, workers: int, limit: int | None) -> None:
    cases = make_cases(suite, limit)
    root = output / suite
    settings = {
        "version": VERSION, "suite": suite, "metrics": METRIC_ORDER,
        "reference_graphs": reference_graphs, "calibration_graphs": calibration_graphs,
        "pooling": "all_null_nodes_per_target", "tail": "greater_or_equal",
        "node_correction": "Bonferroni", "cases": [
            (case.name, case.model, case.group, case.alpha, case.target_sha256,
             case.reference_seed, case.calibration_seed) for case in cases
        ],
    }
    canonical = json.dumps(settings, ensure_ascii=False, sort_keys=True)
    signature = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    settings_path = root / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != json.loads(canonical):
        raise ValueError(f"settings differ from existing fusion run: {root}")
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    pending = []
    for case in cases:
        path = checkpoint_path(root, case)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("settings_signature") != signature or payload["case"]["target_sha256"] != case.target_sha256:
                raise ValueError(f"checkpoint differs from experiment: {path}")
        else:
            pending.append((case, path))
    print(f"[{suite}] target/model cases={len(cases)}, pending={len(pending)}", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_case, case, reference_graphs,
                                       calibration_graphs, signature, path)
                       for case, path in pending]
            for completed, future in enumerate(as_completed(futures), 1):
                name = future.result()
                if completed % 5 == 0 or completed == len(pending):
                    print(f"[{suite}] {completed}/{len(pending)}: {name}", flush=True)
    summarize_suite(root, cases, signature)
    print(f"[{suite}] results: {root}", flush=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "benchmark", "strength", "real"), default="all")
    parser.add_argument("--output", type=Path, default=Path("experiment_results/equal_weight_p_fusion_a500_b500_v3"))
    parser.add_argument("--reference-graphs", type=int, default=500)
    parser.add_argument("--calibration-graphs", type=int, default=500)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, help="run only the first cases of each selected suite")
    args = parser.parse_args()
    if args.reference_graphs < 1 or args.calibration_graphs < 1 or args.workers < 1:
        parser.error("graph counts and workers must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    suites = ("benchmark", "strength", "real") if args.suite == "all" else (args.suite,)
    for suite in suites:
        run_suite(suite, args.output, args.reference_graphs,
                  args.calibration_graphs, args.workers, args.limit)


if __name__ == "__main__":
    main()
