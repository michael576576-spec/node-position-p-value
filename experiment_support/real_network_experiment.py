"""GNM centrality significance on cleaned real-network edge lists.

Input files are whitespace-separated edge lists. The first two columns are node
labels; further columns (for example weights) are ignored. Every graph becomes
an undirected, unweighted simple graph before inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiment_support.centrality import METRICS, compute_multiple_scores_igraph


LABELS = {"degree": "DC", "pagerank": "PR",
          "betweenness": "BC", "closeness": "CC", "gpg": "GPG"}


def load_edgelist(path: Path) -> tuple[list[str], tuple[tuple[int, int], ...], dict]:
    labels: set[str] = set()
    undirected: set[tuple[str, str]] = set()
    seen_directed: set[tuple[str, str]] = set()
    raw_edges = self_loops = duplicate_edges = reversed_edges = extra_columns = 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split()
            if not parts or parts[0].startswith(("#", "%", "//")):
                continue
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_number}: expected at least two columns")
            raw_edges += 1
            extra_columns += len(parts) > 2
            u, v = parts[:2]
            labels.update((u, v))
            if u == v:
                self_loops += 1
                continue
            edge = (u, v) if u < v else (v, u)
            if edge in undirected:
                duplicate_edges += 1
                reversed_edges += (v, u) in seen_directed
            else:
                undirected.add(edge)
            seen_directed.add((u, v))
    node_labels = sorted(labels)
    mapping = {label: index for index, label in enumerate(node_labels)}
    edges = tuple(sorted((mapping[u], mapping[v]) for u, v in undirected))
    info = {
        "network": path.stem, "source": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "nodes": len(node_labels), "edges": len(edges),
        "raw_edge_rows": raw_edges, "self_loops_removed": self_loops,
        "duplicate_or_reciprocal_rows_removed": duplicate_edges,
        "of_which_reciprocal_rows": reversed_edges,
        "rows_with_extra_columns_ignored": extra_columns,
    }
    return node_labels, edges, info


def gnm_edges_scalable(n: int, m: int, rng: random.Random,
                       offsets: list[int]) -> tuple[tuple[int, int], ...]:
    """Uniform simple G(n,m), without materializing all possible edges."""
    total = n * (n - 1) // 2
    sampled = rng.sample(range(total), m)
    result = []
    for rank in sampled:
        u = bisect_right(offsets, rank) - 1
        v = u + 1 + rank - offsets[u]
        result.append((u, v))
    return tuple(result)


def analyze_network(
    source: Path, metrics: tuple[str, ...], budget: int, alpha: float,
    seed: int,
) -> dict:
    labels, edges, cleaning = load_edgelist(source)
    n, m = len(labels), len(edges)
    if n < 2 or not 0 <= m <= n * (n - 1) // 2:
        raise ValueError(f"invalid cleaned graph: {source}")
    observed = {
        metric: np.asarray(scores)
        for metric, scores in compute_multiple_scores_igraph(n, edges, metrics).items()
    }
    counts = {metric: np.zeros(n, dtype=np.int64) for metric in metrics}
    offsets = [u * (2 * n - u - 1) // 2 for u in range(n)]
    rng = random.Random(f"{seed}:gnm:{source.stem}:{cleaning['sha256']}")
    denominator = budget * n + 1
    threshold = alpha / n
    started = time.perf_counter()
    for _ in range(budget):
        null_edges = gnm_edges_scalable(n, m, rng, offsets)
        null_scores = compute_multiple_scores_igraph(n, null_edges, metrics)
        for metric in metrics:
            reference = np.sort(np.asarray(null_scores[metric]))
            counts[metric] += n - np.searchsorted(
                reference, observed[metric], side="left"
            )
    seconds = time.perf_counter() - started
    reports = {}
    for metric in metrics:
        ids = np.flatnonzero((counts[metric] + 1) / denominator < threshold)
        reports[metric] = [
            {"node": labels[node], "score": float(observed[metric][node]),
             "p_value": float((counts[metric][node] + 1) / denominator),
             "bonferroni_p": float(min(1.0, n * (counts[metric][node] + 1) / denominator))}
            for node in ids
        ]
        reports[metric].sort(key=lambda row: (row["p_value"], -row["score"], row["node"]))
    return {
        "cleaning": cleaning, "null_graphs": budget, "seconds": seconds,
        "reports": reports,
    }


def compute_checkpoint(source: Path, metrics: tuple[str, ...], budget: int,
                       alpha: float, seed: int, checkpoint: Path) -> tuple[str, int, int, float]:
    data = analyze_network(source, metrics, budget, alpha, seed)
    temporary = checkpoint.with_name(f"{checkpoint.stem}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, checkpoint)
    cleaning = data["cleaning"]
    return source.stem, cleaning["nodes"], cleaning["edges"], data["seconds"]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(output: Path, names: list[str], metrics: tuple[str, ...],
              settings: dict) -> None:
    cleaning_rows = []
    summary_rows = []
    node_rows = []
    for name in names:
        data = json.loads((output / "checkpoints" / f"{name}.json").read_text(encoding="utf-8"))
        cleaning = data["cleaning"]
        cleaning_rows.append(cleaning)
        for metric in metrics:
            reports = data["reports"][metric]
            summary_rows.append({
                "network": name, "metric": LABELS[metric],
                "nodes": cleaning["nodes"], "edges": cleaning["edges"],
                "reported_nodes": len(reports),
                "null_graphs_generated": data["null_graphs"],
                "seconds_per_network": data["seconds"],
            })
            node_rows.extend({"network": name, "metric": LABELS[metric], **row}
                             for row in reports)
    write_csv(output / "cleaning.csv", cleaning_rows, list(cleaning_rows[0]))
    write_csv(output / "summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(output / "significant_nodes.csv", node_rows,
              ["network", "metric", "node", "score", "p_value", "bonferroni_p"])
    lines = [
        "# 真实网络上的中心性显著性检验", "",
        f"零模型：GNM；每图独立生成固定 {settings['budget']} 张零图；"
        f"Bonferroni α={settings['alpha']}。",
        "边表按无向、无权简单图处理：取前两列为端点，忽略权重列，合并反向/重复边，删除自环。",
        "文件未提供真实中心节点标签，不能计算 Recall、Precision、F1 或分类准确率。",
        "报告数表示校正后显著的节点数量；节点明细与 p 值见 `significant_nodes.csv`。",
        "运行时间是整张图所有所选指标的合计，不应按指标相加。", "",
        "| 网络 | 节点 | 边 | 零图 | 秒 | "
        + " | ".join(f"{LABELS[metric]} 报告数" for metric in metrics) + " |",
        "| --- | ---: | ---: | ---: | ---: | "
        + " | ".join("---:" for _ in metrics) + " |",
    ]
    for name in names:
        selected = [row for row in summary_rows if row["network"] == name]
        first = selected[0]
        lines.append(
            f"| {name} | {first['nodes']} | {first['edges']} | "
            f"{first['null_graphs_generated']} | {first['seconds_per_network']:.3f} | "
            + " | ".join(str(row["reported_nodes"]) for row in selected) + " |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "node_position_pvalue_matplotlib")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-dir", type=Path, default=Path("network"))
    parser.add_argument("--names", nargs="+", help="File stems to run; default: all .txt files")
    parser.add_argument("--max-nodes", type=int,
                        help="Run only files with at most this many observed nodes")
    parser.add_argument("--metrics", nargs="+", choices=tuple(METRICS),
                        default=tuple(METRICS))
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--output", type=Path,
                        default=Path("experiment_results/real_network_five_metrics_gnm_b100_v1"))
    args = parser.parse_args()
    if args.budget < 1 or args.workers < 1 or not 0 < args.alpha < 1:
        parser.error("budget and workers must be positive; alpha must be between 0 and 1")
    paths = sorted(args.network_dir.glob("*.txt"), key=lambda path: path.stem.lower())
    if args.names:
        requested = set(args.names)
        found = {path.stem for path in paths}
        missing = requested - found
        if missing:
            parser.error(f"unknown network names: {', '.join(sorted(missing))}")
        paths = [path for path in paths if path.stem in requested]
    if args.max_nodes is not None:
        if args.max_nodes < 2:
            parser.error("max-nodes must be at least 2")
        paths = [path for path in paths if len(load_edgelist(path)[0]) <= args.max_nodes]
    if not paths:
        parser.error("no network files selected")
    metrics = tuple(dict.fromkeys(args.metrics))
    settings = {
        "names": [path.stem for path in paths], "metrics": metrics,
        "budget": args.budget, "alpha": args.alpha, "seed": args.seed,
        "null_model": "GNM", "sampling": "fixed_budget_no_early_stop",
        "metric_version": "five_metrics_v1",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    settings_path = args.output / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text(encoding="utf-8")) != json.loads(json.dumps(settings)):
        raise ValueError(f"existing output has different settings: {args.output}")
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    pending = []
    for index, path in enumerate(paths, start=1):
        checkpoint = checkpoint_dir / f"{path.stem}.json"
        if checkpoint.exists():
            data = json.loads(checkpoint.read_text(encoding="utf-8"))
            if data["cleaning"]["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f"source changed since checkpoint: {path}")
            print(f"[{index}/{len(paths)}] {path.stem}: cached", flush=True)
            continue
        pending.append((path, checkpoint))
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(compute_checkpoint, path, metrics, args.budget,
                                       args.alpha, args.seed, checkpoint)
                       for path, checkpoint in pending]
            for finished, future in enumerate(as_completed(futures), start=1):
                name, n, m, seconds = future.result()
                print(f"[{len(paths) - len(pending) + finished}/{len(paths)}] "
                      f"{name}: n={n}, m={m}, null={args.budget}, seconds={seconds:.2f}", flush=True)
    summarize(args.output, [path.stem for path in paths], metrics, settings)
    print(f"Report: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
