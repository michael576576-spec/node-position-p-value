"""Compare null models, budgets and centralities on 20 negative and 20 positive graphs.

Every target graph has its own parameter setting. Each target/model
uses one null-graph stream; smaller budgets are prefixes of the largest one.
Only Bonferroni-adjusted p-values and decisions are produced.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiment_support.centrality import METRICS, compute_multiple_scores_igraph
from experiment_support.experiment import SPECS, make_negative, make_positive
from experiment_support.null_models import NULL_MODELS
from experiment_support.ratio_sweep import classification_rates


MODELS = ("gnm", "configuration", "chung_lu")
METRIC_ORDER = tuple(METRICS)
METRIC_LABELS = {"degree": "DC", "lac": "LAC", "eigenvector": "EC",
                 "betweenness": "BC", "closeness": "CC", "gpg": "GPG"}

NEGATIVE_SPECS = (
    (52, 78), (56, 95), (58, 105), (62, 115), (64, 130),
    (68, 145), (72, 160), (74, 175), (78, 200), (82, 225),
    (86, 245), (88, 265), (92, 290), (96, 315), (98, 340),
    (102, 365), (106, 400), (112, 470), (120, 540), (135, 700),
)
POSITIVE_SPECS = SPECS + (
    (55, 75, 1, 2.0), (60, 105, 2, 4.5),
    (70, 125, 1, 5.0), (75, 210, 3, 3.0),
    (85, 250, 5, 4.5), (95, 320, 3, 3.5),
    (105, 390, 4, 5.0), (115, 500, 2, 6.0),
    (130, 680, 6, 4.0), (145, 820, 4, 6.5),
)


def target_graphs(seed: int):
    for index, (n, target_m) in enumerate(NEGATIVE_SPECS, start=1):
        name = f"negative_{index:02d}"
        rng = random.Random(f"{seed}:{name}:graph")
        yield "negative", name, make_negative(name, n, target_m, rng)
    for index, (n, target_m, hub_count, ratio) in enumerate(POSITIVE_SPECS, start=1):
        name = f"positive_{index:02d}"
        rng = random.Random(f"{seed}:{name}:graph")
        yield "positive", name, make_positive(name, n, target_m, hub_count, ratio, rng)


def checkpoint_path(output: Path, model: str, graph: str) -> Path:
    return output / "checkpoints" / model / f"{graph}.json"


def degree_sequence(n: int, edges: tuple[tuple[int, int], ...]) -> np.ndarray:
    endpoints = np.asarray(edges, dtype=np.int64).reshape(-1)
    return np.bincount(endpoints, minlength=n)


def run_target(kind: str, setting: str, network,
               model: str, metrics: tuple[str, ...], budgets: tuple[int, ...],
               alpha: float, seed: int, checkpoint: Path) -> str:
    n = network.n
    observed_started = time.perf_counter()
    observed = {
        metric: np.asarray(scores)
        for metric, scores in compute_multiple_scores_igraph(
            n, network.edges, metrics
        ).items()
    }
    observed_seconds = time.perf_counter() - observed_started
    rng = random.Random(f"{seed}:{network.name}:{model}:null")
    generator = NULL_MODELS[model]
    expected_degrees = degree_sequence(n, network.edges) if model == "configuration" else None
    counts = {metric: np.zeros(n, dtype=np.int64) for metric in metrics}
    budget_set = set(budgets)
    snapshots: dict[str, dict] = {}
    edge_sum = 0
    started = time.perf_counter()

    for graph_number in range(1, budgets[-1] + 1):
        null_edges = generator(n, network.edges, rng)
        if expected_degrees is not None and not np.array_equal(
            degree_sequence(n, null_edges), expected_degrees
        ):
            raise ArithmeticError(f"configuration changed labelled degrees: {network.name}")
        edge_sum += len(null_edges)
        null_scores = compute_multiple_scores_igraph(n, null_edges, metrics)
        for metric in metrics:
            reference = np.sort(np.asarray(null_scores[metric]))
            counts[metric] += n - np.searchsorted(reference, observed[metric], side="left")
        if graph_number in budget_set:
            reports = {}
            for metric in metrics:
                adjusted = np.minimum(
                    1.0, n * (counts[metric] + 1) / (graph_number * n + 1)
                )
                selected = np.flatnonzero(adjusted < alpha)
                reports[metric] = [
                    {"node": int(node), "bonferroni_p": float(adjusted[node])}
                    for node in selected
                ]
            snapshots[str(graph_number)] = {
                "reports": reports,
                "null_seconds": time.perf_counter() - started,
                "mean_null_edges": edge_sum / graph_number,
            }

    payload = {
        "kind": kind, "setting": setting,
        "graph": network.name, "model": model,
        "n": n, "m": network.m, "hubs": sorted(network.hubs),
        "hub_probability": network.hub_probability,
        "ordinary_probability": network.background_probability,
        "metrics": metrics, "budgets": budgets,
        "observed_seconds": observed_seconds, "snapshots": snapshots,
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(f"{checkpoint.stem}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, checkpoint)
    return f"{network.name}/{model}"


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(output: Path, targets: list, models: tuple[str, ...],
              metrics: tuple[str, ...], budgets: tuple[int, ...], alpha: float) -> None:
    """Average each graph's false-report fraction and classification rates."""
    negative_count = sum(kind == "negative" for kind, _, _ in targets)
    positive_count = sum(kind == "positive" for kind, _, _ in targets)
    if negative_count != 20 or positive_count != 20:
        raise ValueError("expected exactly 20 negative and 20 positive target graphs")
    aggregates = {
        (model, metric, budget): {
            "negative_report_rate_sum": 0.0, "recall_sum": 0.0,
            "precision_sum": 0.0, "f1_sum": 0.0,
            "total_tp": 0, "total_fp": 0, "total_fn": 0,
        }
        for model in models for metric in metrics for budget in budgets
    }
    for kind, setting, network in targets:
        hubs = set(network.hubs)
        for model in models:
            checkpoint = checkpoint_path(output, model, network.name)
            data = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (data["kind"] != kind or data["setting"] != setting
                    or data["graph"] != network.name or data["model"] != model
                    or data["n"] != network.n or data["m"] != network.m
                    or data["hubs"] != sorted(hubs)
                    or tuple(data["metrics"]) != metrics
                    or tuple(data["budgets"]) != budgets):
                raise ValueError(f"checkpoint mismatch: {checkpoint}")
            for metric in metrics:
                for budget in budgets:
                    reported = {
                        item["node"]
                        for item in data["snapshots"][str(budget)]["reports"][metric]
                    }
                    aggregate = aggregates[(model, metric, budget)]
                    if kind == "negative":
                        aggregate["negative_report_rate_sum"] += len(reported) / network.n
                        aggregate["total_fp"] += len(reported)
                    else:
                        tp = len(reported & hubs)
                        fp = len(reported - hubs)
                        fn = len(hubs) - tp
                        recall, precision, f1 = classification_rates(
                            tp, fp, fn
                        )
                        aggregate["recall_sum"] += recall
                        aggregate["precision_sum"] += precision
                        aggregate["f1_sum"] += f1
                        aggregate["total_tp"] += tp
                        aggregate["total_fp"] += fp
                        aggregate["total_fn"] += fn

    rows = []
    micro_rows = []
    micro_1000_rows = []
    for model in models:
        for metric in metrics:
            for budget in budgets:
                aggregate = aggregates[(model, metric, budget)]
                rows.append({
                    "零模型": model,
                    "中心性指标": METRIC_LABELS[metric],
                    "零图数": budget,
                    "负例误检率": aggregate["negative_report_rate_sum"] / negative_count,
                    "正例Recall": aggregate["recall_sum"] / positive_count,
                    "正例Precision": aggregate["precision_sum"] / positive_count,
                    "正例F1": aggregate["f1_sum"] / positive_count,
                })
    columns = list(rows[0])
    write_csv(output / "summary.csv", rows, columns)
    lines = [
        "# 基础正负例实验汇总",
        "",
        f"共 20 张负例图、20 张正例图；每张图使用不同的参数组合。"
        f"使用 {', '.join(models)} 零模型、"
        f"{', '.join(METRIC_LABELS[m] for m in metrics)} 中心性指标，"
        f"零图数为 {', '.join(map(str, budgets))}。",
        "**负例误检率**＝先计算每张负例图的报告节点数 / 该图节点数，"
        "再对 20 张负例图取算术平均。"
        "正例 Recall、Precision、F1 先逐图计算，再对 20 张正例图取算术平均；"
        "无报告时 Precision 和 F1 记为 0。",
        f"使用零图中全部节点的中心性分数构造经验零分布，"
        f"只计算 Bonferroni 校正 p 值，以校正后 p < {alpha} 报告节点。"
        "同一张目标图的不同零图预算共享一条零图序列。",
        "",
        "| " + " | ".join(columns) + " |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['零模型']} | {row['中心性指标']} | {row['零图数']} | "
            f"{row['负例误检率']:.3%} | {row['正例Recall']:.1%} | "
            f"{row['正例Precision']:.1%} | {row['正例F1']:.1%} |"
        )
    lines += [
        "", "## 正负例合并：节点级整体指标", "",
        "先合并 40 张图的节点预测，累计 TP、FP、FN，再计算一次 Recall、Precision、F1。"
        "负例没有真实中心节点，只可能贡献 FP；因此这三项指标是整个测试集的微平均，"
        "与上表的正例逐图平均值口径不同。", "",
        "| 零模型 | 中心性指标 | 零图数 | TP | FP | FN | 整体Recall | 整体Precision | 整体F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        for metric in metrics:
            for budget in budgets:
                aggregate = aggregates[(model, metric, budget)]
                tp = aggregate["total_tp"]
                fp = aggregate["total_fp"]
                fn = aggregate["total_fn"]
                recall, precision, f1 = classification_rates(tp, fp, fn)
                micro_row = (
                    f"| {model} | {METRIC_LABELS[metric]} | {budget} | "
                    f"{tp} | {fp} | {fn} | {recall:.1%} | {precision:.1%} | {f1:.1%} |"
                )
                micro_rows.append(micro_row)
                if budget == 1000:
                    micro_1000_rows.append(micro_row)
    lines.extend(micro_rows)
    if 1000 in budgets:
        lines += [
            "", "## 正负例合并：仅 1000 张零图", "",
            "| 零模型 | 中心性指标 | 零图数 | TP | FP | FN | 整体Recall | 整体Precision | 整体F1 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        lines.extend(micro_1000_rows)
        lines += [
            "", "## 逐图平均：仅 1000 张零图", "",
            "正例的 Recall、Precision、F1 分别先对每张正例图计算，"
            "再对 20 张图取算术平均。负例误检率先逐图计算报告数/节点数，"
            "再对 20 张图取算术平均。", "",
            "| 零模型 | 中心性指标 | 零图数 | 负例平均误检率 | 正例平均Recall | 正例平均Precision | 正例平均F1 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            if row["零图数"] == 1000:
                lines.append(
                    f"| {row['零模型']} | {row['中心性指标']} | {row['零图数']} | "
                    f"{row['负例误检率']:.3%} | {row['正例Recall']:.1%} | "
                    f"{row['正例Precision']:.1%} | {row['正例F1']:.1%} |"
                )
        lines += [
            "", "## 1000 张零图：全部指标汇总", "",
            "逐图平均列与整体列分别使用上文定义的计算口径。", "",
            "| 零模型 | 中心性指标 | 零图数 | 负例平均误检率 | 正例平均Recall | 正例平均Precision | 正例平均F1 | TP | FP | FN | 整体Recall | 整体Precision | 整体F1 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            if row["零图数"] != 1000:
                continue
            model = row["零模型"]
            metric = next(key for key in metrics if METRIC_LABELS[key] == row["中心性指标"])
            aggregate = aggregates[(model, metric, 1000)]
            tp = aggregate["total_tp"]
            fp = aggregate["total_fp"]
            fn = aggregate["total_fn"]
            recall, precision, f1 = classification_rates(tp, fp, fn)
            lines.append(
                f"| {model} | {row['中心性指标']} | 1000 | "
                f"{row['负例误检率']:.3%} | {row['正例Recall']:.1%} | "
                f"{row['正例Precision']:.1%} | {row['正例F1']:.1%} | "
                f"{tp} | {fp} | {fn} | {recall:.1%} | {precision:.1%} | {f1:.1%} |"
            )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=int, nargs="+", default=(100, 500, 1000))
    parser.add_argument("--null-models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--metrics", nargs="+", choices=METRIC_ORDER, default=METRIC_ORDER)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path,
                        default=Path("experiment_results/basic_40_unique_graphs_aggregate"))
    args = parser.parse_args()
    budgets = tuple(args.budgets)
    models = tuple(model for model in MODELS if model in args.null_models)
    metrics = tuple(metric for metric in METRIC_ORDER if metric in args.metrics)
    if args.workers < 1 or not 0 < args.alpha < 1:
        parser.error("workers must be positive; alpha must be between 0 and 1")
    if not budgets or any(value < 1 for value in budgets) or tuple(sorted(set(budgets))) != budgets:
        parser.error("budgets must contain distinct positive integers in increasing order")
    if len(args.null_models) != len(set(args.null_models)) or len(args.metrics) != len(set(args.metrics)):
        parser.error("null-models and metrics must not contain duplicates")
    targets = list(target_graphs(args.seed))
    settings = {
        "seed": args.seed,
        "negative_specs": NEGATIVE_SPECS, "positive_specs": POSITIVE_SPECS,
        "target_graphs_per_setting": 1,
        "models": models, "metrics": metrics,
        "budgets": budgets, "alpha": args.alpha,
        "sampling": "per_target_model_shared_prefix_all_null_nodes",
        "p_values": "bonferroni_only", "backend": "igraph",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    settings_path = args.output / "settings.json"
    normalized = json.loads(json.dumps(settings))
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != normalized:
            raise ValueError(f"output contains a different experiment: {args.output}")
    else:
        settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2),
                                 encoding="utf-8")

    pending = [
        (kind, setting, network, model,
         checkpoint_path(args.output, model, network.name))
        for kind, setting, network in targets for model in models
        if not checkpoint_path(args.output, model, network.name).exists()
    ]
    print(f"Target graphs: {len(targets)}; pending target/model tasks: {len(pending)}", flush=True)
    if args.workers == 1:
        for index, (kind, setting, network, model, checkpoint) in enumerate(pending, 1):
            run_target(kind, setting, network, model,
                       metrics, budgets, args.alpha, args.seed, checkpoint)
            if index % 10 == 0 or index == len(pending):
                print(f"Completed {index}/{len(pending)}", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_target, kind, setting, network,
                                model, metrics, budgets, args.alpha, args.seed, checkpoint)
                for kind, setting, network, model, checkpoint in pending
            ]
            for index, future in enumerate(as_completed(futures), 1):
                future.result()
                if index % 10 == 0 or index == len(futures):
                    print(f"Completed {index}/{len(futures)}", flush=True)
    summarize(args.output, targets, models, metrics, budgets, args.alpha)
    print(f"Report: {args.output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
