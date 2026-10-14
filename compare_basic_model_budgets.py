"""Compare null models and budgets on selected positive/negative settings.

The target graphs and their seeds come from experiment.py. Each target/model
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
from itertools import combinations
from pathlib import Path

import numpy as np

from experiment_support.centrality import METRICS, compute_multiple_scores_igraph
from experiment_support.experiment import SPECS, make_negative, make_positive
from experiment_support.null_models import NULL_MODELS
from experiment_support.ratio_sweep import classification_rates


MODELS = ("gnm", "configuration", "chung_lu")
METRIC_ORDER = tuple(METRICS)
METRIC_LABELS = {"degree": "DC", "harmonic": "HC", "betweenness": "BC", "pagerank": "PR"}
DEFAULT_NEGATIVE_SETTINGS = (1, 5, 10)
DEFAULT_POSITIVE_SETTINGS = (1, 3, 5, 7, 10)


def target_graphs(seed: int, repetitions: int,
                  negative_settings: tuple[int, ...], positive_settings: tuple[int, ...]):
    for kind in ("negative", "positive"):
        for index, (n, target_m, hub_count, ratio) in enumerate(SPECS, start=1):
            if index not in (negative_settings if kind == "negative" else positive_settings):
                continue
            setting = f"{kind}_{index:02d}"
            for repetition in range(1, repetitions + 1):
                name = f"{setting}_rep{repetition:02d}"
                rng = random.Random(f"{seed}:{setting}:{repetition}:graph")
                network = (
                    make_negative(name, n, target_m, rng) if kind == "negative"
                    else make_positive(name, n, target_m, hub_count, ratio, rng)
                )
                yield kind, setting, repetition, network


def checkpoint_path(output: Path, model: str, graph: str) -> Path:
    return output / "checkpoints" / model / f"{graph}.json"


def degree_sequence(n: int, edges: tuple[tuple[int, int], ...]) -> np.ndarray:
    endpoints = np.asarray(edges, dtype=np.int64).reshape(-1)
    return np.bincount(endpoints, minlength=n)


def run_target(kind: str, setting: str, repetition: int, network,
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
        "kind": kind, "setting": setting, "repetition": repetition,
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


def mean(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def summarize(output: Path, targets: list, models: tuple[str, ...],
              metrics: tuple[str, ...], budgets: tuple[int, ...], alpha: float) -> None:
    replicate_rows: list[dict] = []
    node_rows: list[dict] = []
    budget_rows: list[dict] = []
    setting_details: dict[str, dict] = {}

    for kind, setting, repetition, network in targets:
        setting_details[setting] = {
            "kind": kind, "setting": setting, "n": network.n,
            "target_m": network.target_m, "hubs": len(network.hubs),
            "hub_probability": network.hub_probability,
            "ordinary_probability": network.background_probability,
        }
        for model in models:
            checkpoint = checkpoint_path(output, model, network.name)
            data = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (data["kind"] != kind or data["setting"] != setting
                    or data["repetition"] != repetition
                    or data["graph"] != network.name or data["model"] != model
                    or data["n"] != network.n or data["m"] != network.m
                    or data["hubs"] != sorted(network.hubs)
                    or tuple(data["metrics"]) != metrics
                    or tuple(data["budgets"]) != budgets):
                raise ValueError(f"checkpoint mismatch: {checkpoint}")
            hubs = set(network.hubs)
            for metric in metrics:
                report_sets: dict[int, set[int]] = {}
                for budget in budgets:
                    snapshot = data["snapshots"][str(budget)]
                    reports = snapshot["reports"][metric]
                    reported = {item["node"] for item in reports}
                    report_sets[budget] = reported
                    tp = len(reported & hubs)
                    fp = len(reported - hubs)
                    if kind == "positive":
                        recall, precision, f1 = classification_rates(
                            tp, fp, len(hubs) - tp
                        )
                    else:
                        recall = precision = f1 = None
                    replicate_rows.append({
                        "kind": kind, "setting": setting, "repetition": repetition,
                        "graph": network.name, "model": model,
                        "metric": METRIC_LABELS[metric], "metric_key": metric,
                        "budget": budget, "n": network.n, "m": network.m,
                        "hubs": len(hubs), "reported": len(reported),
                        "tp": tp, "fp": fp, "no_report": int(not reported),
                        "recall": recall, "precision": precision, "f1": f1,
                        "mean_null_edges": snapshot["mean_null_edges"],
                        "null_seconds": snapshot["null_seconds"],
                        "observed_seconds": data["observed_seconds"],
                    })
                    node_rows.extend({
                        "kind": kind, "setting": setting,
                        "graph": network.name, "model": model,
                        "metric": METRIC_LABELS[metric], "budget": budget,
                        "node": item["node"],
                        "true_hub": int(item["node"] in hubs),
                        "bonferroni_p": item["bonferroni_p"],
                    } for item in reports)
                for small, large in combinations(budgets, 2):
                    first, second = report_sets[small], report_sets[large]
                    budget_rows.append({
                        "kind": kind, "setting": setting,
                        "graph": network.name, "model": model,
                        "metric": METRIC_LABELS[metric],
                        "small_budget": small, "large_budget": large,
                        "same_report_set": first == second,
                        "small_only": len(first - second),
                        "large_only": len(second - first),
                        "intersection": len(first & second),
                    })

    write_csv(output / "replicates.csv", replicate_rows, list(replicate_rows[0]))
    write_csv(output / "significant_nodes.csv", node_rows,
              ["kind", "setting", "graph", "model", "metric", "budget",
               "node", "true_hub", "bonferroni_p"])
    if budget_rows:
        write_csv(output / "budget_comparison.csv", budget_rows, list(budget_rows[0]))

    setting_rows: list[dict] = []
    overall_rows: list[dict] = []
    for kind in ("negative", "positive"):
        selected_settings = [setting for setting, data in setting_details.items()
                             if data["kind"] == kind]
        for setting in selected_settings + ([None] if kind == "negative" else []):
            for model in models:
                for metric in metrics:
                    for budget in budgets:
                        selected = [row for row in replicate_rows
                                    if row["kind"] == kind
                                    and (setting is None or row["setting"] == setting)
                                    and row["model"] == model
                                    and row["metric_key"] == metric
                                    and row["budget"] == budget]
                        detail = setting_details[setting] if setting is not None else None
                        summary = {
                            "kind": kind, "setting": setting or "ALL",
                            "model": model, "metric": METRIC_LABELS[metric],
                            "budget": budget, "graphs": len(selected),
                            "n": detail["n"] if detail else None,
                            "target_m": detail["target_m"] if detail else None,
                            "mean_observed_m": mean(selected, "m"),
                            "hubs": detail["hubs"] if detail else None,
                            "hub_probability": detail["hub_probability"] if detail else None,
                            "ordinary_probability": detail["ordinary_probability"] if detail else None,
                            "mean_reported": mean(selected, "reported"),
                            "mean_tp": mean(selected, "tp") if kind == "positive" else None,
                            "mean_fp": mean(selected, "fp"),
                            "no_report_fraction": mean(selected, "no_report") if kind == "negative" else None,
                            "mean_recall": mean(selected, "recall") if kind == "positive" else None,
                            "mean_precision": mean(selected, "precision") if kind == "positive" else None,
                            "mean_f1": mean(selected, "f1") if kind == "positive" else None,
                            "mean_null_seconds": mean(selected, "null_seconds"),
                        }
                        (setting_rows if setting is not None else overall_rows).append(summary)

    write_csv(output / "setting_summary.csv", setting_rows, list(setting_rows[0]))
    write_csv(output / "overall_summary.csv", overall_rows, list(overall_rows[0]))

    lines = [
        "# 精简正负例：零模型与零图数量对比", "",
        f"{sum(data['kind'] == 'negative' for data in setting_details.values())} 组负例、"
        f"{sum(data['kind'] == 'positive' for data in setting_details.values())} 组正例，"
        f"每组 {len(targets) // len(setting_details)} 张目标图；"
        f"零模型：{', '.join(models)}；指标：{', '.join(METRIC_LABELS[m] for m in metrics)}；"
        f"零图数：{', '.join(map(str, budgets))}。",
        "同一目标图用于全部零模型；每个目标图/模型只生成一条零图序列，"
        "较小预算使用该序列的前缀。每张零图的全部节点分数进入对应指标的零分布。",
        f"只计算 Bonferroni 校正 p 值并以 p < {alpha} 报告节点；"
        "不计算或输出原始 p 值。无报告时，正例图的 Precision 和 F1 记为 0。",
        "负例没有真实中心节点，其 Recall、Precision 和 F1 不定义；"
        "以平均误报数和无报告图占比评价。",
        "配置模型严格保持每个标号节点的度数。配置模型和 Chung–Lu 的汇总零分布"
        "衡量相对随机节点整体的极端程度，不能解释为固定节点条件下的 p 值。",
        "",
        "## 负例整体结果", "",
        "| 零模型 | 指标 | 零图数 | 平均误报数/图 | 无报告图占比 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for model in models:
        for metric in metrics:
            for budget in budgets:
                row = next(row for row in overall_rows
                           if row["model"] == model
                           and row["metric"] == METRIC_LABELS[metric]
                           and row["budget"] == budget)
                lines.append(
                    f"| {model} | {METRIC_LABELS[metric]} | {budget} | "
                    f"{row['mean_fp']:.3f} | {row['no_report_fraction']:.1%} |"
                )
    lines += [
        "", "## 正例：按参数组分别报告", "",
        "不同正例组的 N、M、中心节点数和强度不同，因此不计算跨组平均分类分数。", "",
        "| 参数组 | N | 目标 M | 中心数 | 中心边概率 | 普通边概率 | 零模型 | 指标 | 零图数 | 平均报告数 | 平均检出数 | 平均误报数 | Recall | Precision | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in setting_rows:
        if row["kind"] != "positive":
            continue
        lines.append(
            f"| {row['setting']} | {row['n']} | {row['target_m']} | {row['hubs']} | "
            f"{row['hub_probability']:.4f} | {row['ordinary_probability']:.4f} | "
            f"{row['model']} | {row['metric']} | {row['budget']} | "
            f"{row['mean_reported']:.3f} | {row['mean_tp']:.3f} | {row['mean_fp']:.3f} | "
            f"{row['mean_recall']:.1%} | {row['mean_precision']:.1%} | {row['mean_f1']:.1%} |"
        )
    lines += [
        "", "各参数组详见 `setting_summary.csv`，逐图结果见 `replicates.csv`，"
        "校正后显著节点及其校正 p 值见 `significant_nodes.csv`，"
        "预算间报告集合差异见 `budget_comparison.csv`。",
    ]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--negative-settings", type=int, nargs="+", default=DEFAULT_NEGATIVE_SETTINGS)
    parser.add_argument("--positive-settings", type=int, nargs="+", default=DEFAULT_POSITIVE_SETTINGS)
    parser.add_argument("--budgets", type=int, nargs="+", default=(100, 500, 1000))
    parser.add_argument("--null-models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--metrics", nargs="+", choices=METRIC_ORDER, default=METRIC_ORDER)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path,
                        default=Path("experiment_results/basic_reduced_model_budget_comparison"))
    args = parser.parse_args()
    budgets = tuple(args.budgets)
    models = tuple(model for model in MODELS if model in args.null_models)
    metrics = tuple(metric for metric in METRIC_ORDER if metric in args.metrics)
    if args.repetitions < 1 or args.workers < 1 or not 0 < args.alpha < 1:
        parser.error("repetitions/workers must be positive; alpha must be between 0 and 1")
    if len(budgets) < 2 or any(value < 1 for value in budgets) or tuple(sorted(set(budgets))) != budgets:
        parser.error("budgets must contain at least two distinct positive integers in increasing order")
    if len(args.null_models) != len(set(args.null_models)) or len(args.metrics) != len(set(args.metrics)):
        parser.error("null-models and metrics must not contain duplicates")
    for name, values, expected in (
        ("negative-settings", args.negative_settings, 3),
        ("positive-settings", args.positive_settings, 5),
    ):
        if len(values) != expected or len(set(values)) != expected or any(index < 1 or index > len(SPECS) for index in values):
            parser.error(f"{name} must contain {expected} distinct indices from 1 to {len(SPECS)}")

    negative_settings = tuple(sorted(args.negative_settings))
    positive_settings = tuple(sorted(args.positive_settings))
    targets = list(target_graphs(args.seed, args.repetitions, negative_settings, positive_settings))
    settings = {
        "seed": args.seed, "repetitions": args.repetitions,
        "specs": SPECS, "negative_settings": negative_settings,
        "positive_settings": positive_settings, "models": models, "metrics": metrics,
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
        (kind, setting, repetition, network, model,
         checkpoint_path(args.output, model, network.name))
        for kind, setting, repetition, network in targets for model in models
        if not checkpoint_path(args.output, model, network.name).exists()
    ]
    print(f"Target graphs: {len(targets)}; pending target/model tasks: {len(pending)}", flush=True)
    if args.workers == 1:
        for index, (kind, setting, repetition, network, model, checkpoint) in enumerate(pending, 1):
            run_target(kind, setting, repetition, network, model,
                       metrics, budgets, args.alpha, args.seed, checkpoint)
            if index % 10 == 0 or index == len(pending):
                print(f"Completed {index}/{len(pending)}", flush=True)
    elif pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_target, kind, setting, repetition, network,
                                model, metrics, budgets, args.alpha, args.seed, checkpoint)
                for kind, setting, repetition, network, model, checkpoint in pending
            ]
            for index, future in enumerate(as_completed(futures), 1):
                future.result()
                if index % 10 == 0 or index == len(futures):
                    print(f"Completed {index}/{len(futures)}", flush=True)
    summarize(args.output, targets, models, metrics, budgets, args.alpha)
    print(f"Report: {args.output / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
