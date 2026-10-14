"""Compare centrality reports on real graphs across null models and budgets.

Each input edge list is converted to an undirected, unweighted simple graph.
For each target graph and null model, the smaller null sample is a prefix of
the larger one. All four centralities share each generated null graph.
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np

from experiment_support.centrality import METRICS, compute_multiple_scores_igraph
from experiment_support.null_models import configuration_edges, chung_lu_edges
from experiment_support.real_network_experiment import LABELS, gnm_edges_scalable, load_edgelist


MODELS = ("gnm", "configuration", "chung_lu")


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_path(output: Path, model: str, name: str) -> Path:
    return output / "checkpoints" / model / f"{name}.json"


def degrees(n: int, edges: tuple[tuple[int, int], ...]) -> np.ndarray:
    endpoints = np.asarray(edges, dtype=np.int64).reshape(-1)
    return np.bincount(endpoints, minlength=n)


def run_one(source: Path, model: str, metrics: tuple[str, ...],
            budgets: tuple[int, ...], alpha: float, seed: int,
            checkpoint: Path) -> str:
    labels, edges, cleaning = load_edgelist(source)
    n, m = len(labels), len(edges)
    if n < 2:
        raise ValueError(f"{source}: fewer than two observed nodes")
    observed_started = time.perf_counter()
    observed = {
        metric: np.asarray(values)
        for metric, values in compute_multiple_scores_igraph(n, edges, metrics).items()
    }
    observed_seconds = time.perf_counter() - observed_started
    observed_degrees = degrees(n, edges) if model == "configuration" else None
    offsets = [u * (2 * n - u - 1) // 2 for u in range(n)] if model == "gnm" else []
    rng = random.Random(f"{seed}:{model}:{source.name}:{cleaning['sha256']}")
    counts = {metric: np.zeros(n, dtype=np.int64) for metric in metrics}
    snapshots: dict[str, dict] = {}
    edge_sum = 0
    started = time.perf_counter()
    for graph_number in range(1, budgets[-1] + 1):
        if model == "gnm":
            null_edges = gnm_edges_scalable(n, m, rng, offsets)
        elif model == "configuration":
            null_edges = configuration_edges(n, edges, rng)
            if not np.array_equal(degrees(n, null_edges), observed_degrees):
                raise ArithmeticError(f"configuration degree sequence changed: {source}")
        else:
            null_edges = chung_lu_edges(n, edges, rng)
        edge_sum += len(null_edges)
        null_scores = compute_multiple_scores_igraph(n, null_edges, metrics)
        for metric in metrics:
            reference = np.sort(np.asarray(null_scores[metric]))
            counts[metric] += n - np.searchsorted(
                reference, observed[metric], side="left"
            )
        if graph_number in budgets:
            denominator = graph_number * n + 1
            cutoff = alpha / n
            reports = {}
            for metric in metrics:
                selected = np.flatnonzero((counts[metric] + 1) / denominator < cutoff)
                reports[metric] = [
                    {
                        "node": labels[node],
                        "score": float(observed[metric][node]),
                        "p_value": float((counts[metric][node] + 1) / denominator),
                        "bonferroni_p": float(min(
                            1.0, n * (counts[metric][node] + 1) / denominator
                        )),
                    }
                    for node in selected
                ]
                reports[metric].sort(
                    key=lambda item: (item["p_value"], -item["score"], item["node"])
                )
            snapshots[str(graph_number)] = {
                "seconds": time.perf_counter() - started,
                "mean_null_edges": edge_sum / graph_number,
                "reports": reports,
            }
    payload = {
        "network": source.stem, "model": model, "cleaning": cleaning,
        "observed_seconds": observed_seconds,
        "budgets": budgets, "metrics": metrics, "snapshots": snapshots,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(f"{checkpoint.stem}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, checkpoint)
    return source.stem


def average(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def summarize(output: Path, names: list[str], models: tuple[str, ...],
              metrics: tuple[str, ...], budgets: tuple[int, ...],
              alpha: float) -> None:
    ordered_models = tuple(model for model in MODELS if model in models)
    ordered_metrics = tuple(metric for metric in LABELS if metric in metrics)
    summary_rows: list[dict] = []
    report_rows: list[dict] = []
    cleaning_rows: list[dict] = []
    report_sets: dict[tuple[str, str, int, str], set[str]] = {}
    for name in names:
        for model in models:
            data = json.loads(checkpoint_path(output, model, name).read_text(encoding="utf-8"))
            if (data["network"] != name or data["model"] != model
                    or tuple(data["budgets"]) != budgets
                    or tuple(data["metrics"]) != metrics):
                raise ValueError(f"checkpoint mismatch for {name}/{model}")
            cleaning = data["cleaning"]
            if model == models[0]:
                cleaning_rows.append(cleaning)
            for budget in budgets:
                snapshot = data["snapshots"][str(budget)]
                for metric in metrics:
                    reports = snapshot["reports"][metric]
                    report_sets[(name, model, budget, metric)] = {
                        item["node"] for item in reports
                    }
                    summary_rows.append({
                        "network": name, "model": model, "budget": budget,
                        "metric": LABELS[metric], "metric_key": metric,
                        "nodes": cleaning["nodes"], "edges": cleaning["edges"],
                        "reported_nodes": len(reports),
                        "reported_fraction": len(reports) / cleaning["nodes"],
                        "mean_null_edges": snapshot["mean_null_edges"],
                        "null_seconds": snapshot["seconds"],
                        "observed_seconds": data["observed_seconds"],
                    })
                    report_rows.extend({
                        "network": name, "model": model, "budget": budget,
                        "metric": LABELS[metric], **item,
                    } for item in reports)
    write_csv(output / "cleaning.csv", cleaning_rows, list(cleaning_rows[0]))
    write_csv(output / "summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(output / "significant_nodes.csv", report_rows,
              ["network", "model", "budget", "metric", "node", "score",
               "p_value", "bonferroni_p"])

    budget_comparison: list[dict] = []
    for small_budget, large_budget in combinations(budgets, 2):
        for name in names:
            for model in models:
                for metric in metrics:
                    small = report_sets[(name, model, small_budget, metric)]
                    large = report_sets[(name, model, large_budget, metric)]
                    budget_comparison.append({
                        "network": name, "model": model, "metric": LABELS[metric],
                        "small_budget": small_budget, "large_budget": large_budget,
                        "same_report_set": small == large,
                        "small_only": len(small - large),
                        "large_only": len(large - small),
                        "intersection": len(small & large),
                        "union": len(small | large),
                    })
    if budget_comparison:
        write_csv(output / "budget_comparison.csv", budget_comparison,
                  list(budget_comparison[0]))

    model_comparison: list[dict] = []
    for name in names:
        for budget in budgets:
            for metric in metrics:
                for model_a, model_b in combinations(ordered_models, 2):
                    first = report_sets[(name, model_a, budget, metric)]
                    second = report_sets[(name, model_b, budget, metric)]
                    union = first | second
                    model_comparison.append({
                        "network": name, "budget": budget, "metric": LABELS[metric],
                        "model_a": model_a, "model_b": model_b,
                        "reported_a": len(first), "reported_b": len(second),
                        "intersection": len(first & second),
                        "union": len(union),
                        "jaccard": len(first & second) / len(union) if union else 1.0,
                        "a_only": len(first - second),
                        "b_only": len(second - first),
                    })
    if model_comparison:
        write_csv(output / "model_comparison.csv", model_comparison,
                  list(model_comparison[0]))

    lines = [
        "# 真实网络：零模型与零图数量对比", "",
        f"目标图 {len(names)} 张；中心性：{', '.join(LABELS[metric] for metric in metrics)}；"
        f"零图数量：{', '.join(map(str, budgets))}；Bonferroni α={alpha}。",
        "每张目标图先合并反向/重复边、删除自环并忽略权重列，随后按无向无权简单图处理。",
        "每个“网络 × 零模型”独立生成零图；较小预算使用同一零图序列的前缀。"
        "同一张零图的全部节点分数进入该图对应指标的经验零分布。",
        "配置模型用保持每个标号节点度数的双边交换，逐张检查度序列；"
        "Chung–Lu 图的边数可以变化。",
        "输入文件没有真实中心节点标签，不能计算 Recall、Precision、F1 或分类准确率。"
        "报告数只表示相对所选零模型显著的节点数量。",
        "耗时记录在 `summary.csv` 中，是该网络/模型计算所有所选中心性指标的累计零图时间，"
        "不能按指标相加。", "",
        "## 各组合平均结果", "",
        "| 零模型 | 零图数 | 指标 | 平均报告比例 | 有报告的网络数 |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for model in ordered_models:
        for metric in ordered_metrics:
            for budget in budgets:
                selected = [row for row in summary_rows
                            if row["model"] == model and row["budget"] == budget
                            and row["metric_key"] == metric]
                lines.append(
                    f"| {model} | {budget} | {LABELS[metric]} | "
                    f"{average(selected, 'reported_fraction'):.3%} | "
                    f"{sum(row['reported_nodes'] > 0 for row in selected)} |"
                )
    if budget_comparison:
        lines += [
            "", "## 零图数量改变后的报告集稳定性", "",
            "| 零模型 | 小预算 | 大预算 | 指标 | 完全一致的网络数 | 小预算独有报告总数 | 大预算独有报告总数 |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
        for model in ordered_models:
            for metric in ordered_metrics:
                for small_budget, large_budget in combinations(budgets, 2):
                    selected = [row for row in budget_comparison
                                if row["small_budget"] == small_budget
                                and row["large_budget"] == large_budget
                                and row["model"] == model
                                and row["metric"] == LABELS[metric]]
                    lines.append(
                        f"| {model} | {small_budget} | {large_budget} | {LABELS[metric]} | "
                        f"{sum(row['same_report_set'] for row in selected)}/{len(selected)} | "
                        f"{sum(row['small_only'] for row in selected)} | "
                        f"{sum(row['large_only'] for row in selected)} |"
                    )
    if model_comparison:
        lines += [
            "", "## 同预算下零模型报告集的差异", "",
            "| 零模型 A | 零模型 B | 零图数 | 指标 | 完全一致的网络数 | 非空并集的平均 Jaccard |",
            "| --- | --- | ---: | --- | ---: | ---: |",
        ]
        for model_a, model_b in combinations(ordered_models, 2):
            for metric in ordered_metrics:
                for budget in budgets:
                    selected = [row for row in model_comparison
                                if row["budget"] == budget
                                and row["metric"] == LABELS[metric]
                                and row["model_a"] == model_a
                                and row["model_b"] == model_b]
                    nonempty = [row for row in selected if row["union"]]
                    jaccard = (f"{average(nonempty, 'jaccard'):.3f}"
                               if nonempty else "null")
                    lines.append(
                        f"| {model_a} | {model_b} | {budget} | {LABELS[metric]} | "
                        f"{sum(row['union'] == row['intersection'] for row in selected)}"
                        f"/{len(selected)} | {jaccard} |"
                    )
    lines += [
        "", "各网络的节点数、边数、报告数和时间见 `summary.csv`；"
        "具体节点及校正前后 p 值见 `significant_nodes.csv`；"
        "同预算下不同零模型的报告集交并情况见 `model_comparison.csv`。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "node_position_pvalue_matplotlib")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network-dir", type=Path, default=Path("network"))
    parser.add_argument("--names", nargs="+", help="File stems; default: all .txt files")
    parser.add_argument("--max-nodes", type=int,
                        help="Include only graphs with at most this many observed nodes")
    parser.add_argument("--metrics", nargs="+", choices=tuple(METRICS),
                        default=tuple(METRICS))
    parser.add_argument("--null-models", nargs="+", choices=MODELS,
                        default=MODELS)
    parser.add_argument("--budgets", nargs="+", type=int, default=(100, 1000))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        default=Path("experiment_results/real_network_model_budget_comparison"))
    args = parser.parse_args()
    budgets = tuple(args.budgets)
    metrics = tuple(dict.fromkeys(args.metrics))
    models = tuple(dict.fromkeys(args.null_models))
    if not budgets or any(value < 1 for value in budgets) or tuple(sorted(set(budgets))) != budgets:
        parser.error("budgets must be distinct positive integers in increasing order")
    if not 0 < args.alpha < 1 or args.workers < 1:
        parser.error("alpha must be between 0 and 1; workers must be positive")
    paths = sorted(args.network_dir.glob("*.txt"), key=lambda path: path.stem.lower())
    if args.names:
        requested = set(args.names)
        missing = requested - {path.stem for path in paths}
        if missing:
            parser.error(f"unknown network names: {', '.join(sorted(missing))}")
        paths = [path for path in paths if path.stem in requested]
    if args.max_nodes is not None:
        if args.max_nodes < 2:
            parser.error("max-nodes must be at least 2")
        paths = [path for path in paths if len(load_edgelist(path)[0]) <= args.max_nodes]
    if not paths:
        parser.error("no network files selected")
    source_hashes = {path.stem: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in paths}
    settings = {
        "names": [path.stem for path in paths], "source_hashes": source_hashes,
        "metrics": metrics, "null_models": models, "budgets": budgets,
        "alpha": args.alpha, "seed": args.seed,
        "sampling": "fixed_budget_shared_prefix_no_early_stop",
        "backend": "igraph",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    settings_path = args.output / "settings.json"
    serialized = json.loads(json.dumps(settings))
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != serialized:
            raise ValueError(f"output contains a different experiment: {args.output}")
    else:
        settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    pending = [
        (path, model, checkpoint_path(args.output, model, path.stem))
        for path in paths for model in models
        if not checkpoint_path(args.output, model, path.stem).exists()
    ]
    print(f"Networks: {len(paths)}; tasks pending: {len(pending)}", flush=True)
    if args.workers == 1:
        for index, (path, model, checkpoint) in enumerate(pending, start=1):
            run_one(path, model, metrics, budgets, args.alpha, args.seed, checkpoint)
            print(f"Completed {index}/{len(pending)}: {path.stem}/{model}", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_one, path, model, metrics, budgets,
                                       args.alpha, args.seed, checkpoint)
                       for path, model, checkpoint in pending]
            for index, future in enumerate(as_completed(futures), start=1):
                future.result()
                print(f"Completed {index}/{len(futures)}", flush=True)
    summarize(args.output, [path.stem for path in paths], models, metrics,
              budgets, args.alpha)
    print(f"Report: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
