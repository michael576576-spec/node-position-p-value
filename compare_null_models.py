"""Run a paired hub-strength sweep for GNM, configuration, and Chung-Lu nulls.

The default experiment uses 50 target graphs per ratio, 1000 null graphs per
target graph and model, and all six centrality metrics. Completed target
graphs are checkpointed, so repeating the same command resumes the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from experiment_support.centrality import METRICS, compute_multiple_scores_igraph
from experiment_support.compare_metrics import METRIC_LABELS, write_comparison_plot
from experiment_support.network_core import Network
from experiment_support.null_models import NULL_MODELS
from experiment_support.ratio_sweep import classification_rates, target_graphs


MODELS = ("gnm", "configuration", "chung_lu")
METRIC_ORDER = tuple(METRICS)
BACKEND = "igraph-1.0.0"
SUMMARY_FIELDS = (
    "model", "metric", "ratio", "ordinary_probability", "hub_probability",
    "bonf_recall", "bonf_precision", "bonf_f1", "bonf_mean_reported", "bonf_mean_fp",
)
ALL_METHOD_FIELDS = (
    "model", "metric", "decision_rule", "ratio", "ordinary_probability",
    "hub_probability", "recall", "precision", "f1", "mean_reported", "mean_fp",
)


def write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_path(root: Path, model: str, network: Network) -> Path:
    return root / "checkpoints" / model / f"{network.name}.json"


def run_target(
    model: str, ratio: float, repetition: int, network: Network,
    null_graphs: int, alpha: float, seed: int, path: Path,
) -> str:
    """Generate one shared collection of null graphs and score all five metrics."""
    metrics = METRIC_ORDER
    generator = NULL_MODELS[model]
    rng = random.Random(
        f"{seed}:null:{network.n}:{network.m}:{repetition}:{ratio:g}:{model}"
    )
    pooled: dict[str, list[float]] = {metric: [] for metric in metrics}
    observed = compute_multiple_scores_igraph(network.n, network.edges, metrics)
    if model == "configuration":
        # Every switch preserves the labelled degree sequence exactly.
        pooled["degree"] = observed["degree"] * null_graphs
        sampled_metrics = tuple(metric for metric in metrics if metric != "degree")
    else:
        sampled_metrics = metrics

    for _ in range(null_graphs):
        null_edges = generator(network.n, network.edges, rng)
        scores = compute_multiple_scores_igraph(network.n, null_edges, sampled_metrics)
        for metric in sampled_metrics:
            pooled[metric].extend(scores[metric])

    results: dict[str, dict] = {}
    for metric in metrics:
        values = pooled[metric]
        if len(values) != network.n * null_graphs:
            raise ArithmeticError(f"wrong null score count for {model}/{metric}")
        reference = np.sort(np.asarray(values))
        scores = np.asarray(observed[metric])
        upper_tail_counts = len(reference) - np.searchsorted(reference, scores, side="left")
        adjusted_p = np.minimum(
            1.0, network.n * (upper_tail_counts + 1) / (len(reference) + 1)
        )
        order = sorted(range(network.n), key=lambda node: (-scores[node], node))
        node_rows = [
            {
                "position": position,
                "node": node,
                "degree": int(observed["degree"][node]),
                "metric": metric,
                "centrality_score": float(scores[node]),
                "bonferroni_p_value": float(adjusted_p[node]),
                "predicted_hub_bonferroni": bool(adjusted_p[node] < alpha),
                "true_hub": node in network.hubs,
            }
            for position, node in enumerate(order, start=1)
        ]
        reported = [row for row in node_rows if row["predicted_hub_bonferroni"]]
        tp = sum(bool(row["true_hub"]) for row in reported)
        fp = len(reported) - tp
        recall, precision, f1 = classification_rates(tp, fp, len(network.hubs) - tp)
        results[metric] = {
            "reported": len(reported), "true_positives": tp, "false_positives": fp,
            "recall": recall, "precision": precision, "f1": f1,
            "nodes": node_rows,
        }

    payload = {
        "model": model, "ratio": ratio, "repetition": repetition,
        "graph": network.name, "hubs": sorted(network.hubs),
        "ordinary_probability": network.background_probability,
        "hub_probability": network.hub_probability,
        "null_graphs": null_graphs, "scores_per_metric": network.n * null_graphs,
        "backend": BACKEND,
        "metrics": results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)
    return network.name


def summarize_model(
    root: Path, model: str, targets: list[tuple[float, int, Network]],
    repetitions: int, null_graphs: int, alpha: float,
) -> list[dict]:
    replicate_rows: list[dict] = []
    node_rows: list[dict] = []
    probabilities: dict[float, tuple[float, float]] = {}
    for ratio, repetition, network in targets:
        payload = json.loads(checkpoint_path(root, model, network).read_text(encoding="utf-8"))
        if (payload["model"], payload["ratio"], payload["repetition"], payload["graph"]) != (
            model, ratio, repetition, network.name
        ) or payload["scores_per_metric"] != network.n * null_graphs or payload.get("backend") != BACKEND:
            raise ValueError(f"checkpoint does not match current experiment: {network.name}")
        probabilities[ratio] = (payload["ordinary_probability"], payload["hub_probability"])
        for metric in METRIC_ORDER:
            result = payload["metrics"][metric]
            replicate_rows.append({
                "model": model, "metric": metric, "ratio": ratio,
                "repetition": repetition, "graph": network.name,
                "hubs": ";".join(map(str, payload["hubs"])),
                "bonf_reported": result["reported"],
                "bonf_true_positives": result["true_positives"],
                "bonf_false_positives": result["false_positives"],
                "bonf_recall": result["recall"],
                "bonf_precision": result["precision"], "bonf_f1": result["f1"],
            })
            node_rows.extend(
                {"model": model, "graph": network.name, "ratio": ratio, **row}
                for row in result["nodes"]
            )

    ratios = tuple(dict.fromkeys(ratio for ratio, _, _ in targets))
    summary: list[dict] = []
    for metric in METRIC_ORDER:
        for ratio in ratios:
            selected = [row for row in replicate_rows if row["metric"] == metric and row["ratio"] == ratio]
            if len(selected) != repetitions:
                raise ArithmeticError(f"missing repeats for {model}/{metric}/{ratio:g}")
            ordinary_probability, hub_probability = probabilities[ratio]
            summary.append({
                "model": model, "metric": metric, "ratio": ratio,
                "ordinary_probability": ordinary_probability,
                "hub_probability": hub_probability,
                "bonf_recall": sum(row["bonf_recall"] for row in selected) / repetitions,
                "bonf_precision": sum(row["bonf_precision"] for row in selected) / repetitions,
                "bonf_f1": sum(row["bonf_f1"] for row in selected) / repetitions,
                "bonf_mean_reported": sum(row["bonf_reported"] for row in selected) / repetitions,
                "bonf_mean_fp": sum(row["bonf_false_positives"] for row in selected) / repetitions,
            })

    directory = root / model
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / "comparison.csv", summary, SUMMARY_FIELDS)
    write_csv(directory / "replicates.csv", replicate_rows, tuple(replicate_rows[0]))
    write_csv(directory / "node_results.csv", node_rows, tuple(node_rows[0]))
    write_comparison_plot(directory, summary, METRIC_ORDER)
    lines = [
        f"# {model}：中心结构强度实验", "",
        f"每个倍率 {repetitions} 张配对目标图；每张目标图 {null_graphs} 张零模型图；"
        f"每张图汇总全部节点分数；Bonferroni 校正后 p < {alpha}。",
        "无报告时，图级精确率和 F1 记为 0。",
        "GNM 节点可交换；每张目标图独立生成零图并汇总全部节点分数。" if model == "gnm" else
        "配置模型严格保持各节点度数；Chung–Lu 根据目标图度数设定连边概率。"
        "汇总分数衡量相对随机节点整体的尾部概率，不是固定节点条件下的 p 值。"
        "此处使用 Bonferroni 阈值作统一比较，不据此声称逐节点误报率控制。", "",
        "| 倍率 | 指标 | 召回率 | 精确率 | F1 | 平均报告数 | 平均误报数 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ratio in ratios:
        for metric in METRIC_ORDER:
            row = next(row for row in summary if row["ratio"] == ratio and row["metric"] == metric)
            lines.append(
                f"| {ratio:g} | {METRIC_LABELS[metric]} | {row['bonf_recall']:.1%} | "
                f"{row['bonf_precision']:.1%} | {row['bonf_f1']:.1%} | "
                f"{row['bonf_mean_reported']:.2f} | {row['bonf_mean_fp']:.2f} |"
            )
    (directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def centrality_as_all_method(row: dict) -> dict:
    return {
        "model": row["model"], "metric": row["metric"],
        "decision_rule": "Bonferroni", "ratio": row["ratio"],
        "ordinary_probability": row["ordinary_probability"],
        "hub_probability": row["hub_probability"],
        "recall": row["bonf_recall"], "precision": row["bonf_precision"],
        "f1": row["bonf_f1"], "mean_reported": row["bonf_mean_reported"],
        "mean_fp": row["bonf_mean_fp"],
    }


def comparison_plot(root: Path, rows: list[dict], ratios: tuple[float, ...]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "node_position_pvalue_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = (("gnm", "GNM"), ("configuration", "Configuration"), ("chung_lu", "Chung–Lu"))
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    measures = (("recall", "Recall"), ("precision", "Precision"), ("f1", "F1 Score"))
    for i, (model, model_label) in enumerate(models):
        for j, (key, title) in enumerate(measures):
            axis = axes[i, j]
            metrics = METRIC_ORDER
            for metric in metrics:
                selected = [row for row in rows if row["model"] == model and row["metric"] == metric]
                if not selected:
                    continue
                by_ratio = {float(row["ratio"]): float(row[key]) for row in selected}
                label = METRIC_LABELS[metric]
                axis.plot(ratios, [by_ratio[ratio] for ratio in ratios], marker="o", label=label)
            axis.set_title(f"{model_label} — {title}")
            axis.set_ylim(0, 1.04)
            axis.set_xticks(ratios)
            axis.tick_params(axis="x", labelbottom=True)
            axis.set_xlabel("Central strength (hub / ordinary edge probability ratio)")
            axis.set_ylabel(title)
            axis.grid(alpha=0.25)
            axis.legend()
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=180)
    fig.savefig(root / "comparison.svg")
    plt.close(fig)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--m", type=int, default=350)
    parser.add_argument("--hubs", type=int, default=6)
    parser.add_argument("--ratios", type=float, nargs="+", default=(1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0))
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--null-graphs", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, default=Path("experiment_results/strength_n100_m350_k6_five_metrics_v1"))
    args = parser.parse_args()
    matplotlib_dir = Path(tempfile.gettempdir()) / "node_position_pvalue_matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    if args.repetitions < 1 or args.null_graphs < 1 or args.workers < 1:
        parser.error("repetitions, null-graphs, and workers must be positive")
    if not 0 < args.alpha < 1:
        parser.error("alpha must be between 0 and 1")
    ratios = tuple(args.ratios)
    if len(ratios) != len(set(ratios)):
        parser.error("ratios must be distinct")

    targets = list(target_graphs(args.n, args.m, args.hubs, ratios, args.repetitions, args.seed))
    settings = {
        "n": args.n, "m": args.m, "hubs": args.hubs, "ratios": ratios,
        "repetitions": args.repetitions, "null_graphs": args.null_graphs,
        "alpha": args.alpha, "seed": args.seed, "models": MODELS, "metrics": METRIC_ORDER,
        "pooling": "all_nodes_per_target", "backend": BACKEND,
        "comparison_version": "five_metrics_v1",
        "version": 7,
    }
    if args.output.exists() and any(args.output.iterdir()) and not (args.output / "settings.json").exists():
        raise ValueError(f"output contains unversioned results: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    settings_path = args.output / "settings.json"
    if settings_path.exists():
        existing = json.loads(settings_path.read_text(encoding="utf-8"))
        current = json.loads(json.dumps(settings))
        fixed_keys = (
            "n", "m", "hubs", "repetitions", "null_graphs", "alpha", "seed",
            "models", "metrics", "pooling", "backend", "comparison_version",
        )
        if (
            any(existing.get(key) != current[key] for key in fixed_keys)
            or existing.get("ratios") != current["ratios"][:len(existing.get("ratios", []))]
            or existing.get("version") != 7
        ):
            raise ValueError(f"settings differ from existing run in {args.output}")
        if existing != current:
            settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

    pending = [
        (model, ratio, repetition, network, checkpoint_path(args.output, model, network))
        for model in MODELS for ratio, repetition, network in targets
        if not checkpoint_path(args.output, model, network).exists()
        or json.loads(checkpoint_path(args.output, model, network).read_text(encoding="utf-8")).get("backend") != BACKEND
    ]
    print(f"Target graphs: {len(targets)}; pending model/graph pairs: {len(pending)}", flush=True)
    started = time.perf_counter()
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    run_target, model, ratio, repetition, network,
                    args.null_graphs, args.alpha, args.seed, path,
                )
                for model, ratio, repetition, network, path in pending
            ]
            for finished, future in enumerate(as_completed(futures), start=1):
                future.result()
                if finished % 10 == 0 or finished == len(pending):
                    print(f"Completed {finished}/{len(pending)}; elapsed {time.perf_counter()-started:.1f}s", flush=True)

    summaries = []
    for model in MODELS:
        summary = summarize_model(args.output, model, targets, args.repetitions, args.null_graphs, args.alpha)
        summaries.extend(summary)
    combined = summaries
    write_csv(args.output / "comparison.csv", combined, SUMMARY_FIELDS)
    all_methods = [centrality_as_all_method(row) for row in combined]
    write_csv(args.output / "comparison_all_methods.csv", all_methods, ALL_METHOD_FIELDS)
    gnm_all_methods = [row for row in all_methods if row["model"] == "gnm"]
    write_csv(args.output / "gnm" / "comparison_all_methods.csv", gnm_all_methods, ALL_METHOD_FIELDS)
    write_comparison_plot(args.output / "gnm", [row for row in combined if row["model"] == "gnm"], METRIC_ORDER)
    lines = [
        "# Null model and centrality comparison", "",
        f"N={args.n}, M={args.m}, planted hubs={args.hubs}; "
        f"{args.repetitions} target graphs per strength and {args.null_graphs} null graphs per target/model.",
        f"Five centrality metrics use Bonferroni p < {args.alpha}.",
        "Only the centrality metrics are included in this comparison.", "",
        "| Strength | Null model | Metric | Recall | Precision | F1 | Mean reported | Mean false positives |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ratio in ratios:
        for model in MODELS:
            for metric in METRIC_ORDER:
                row = next(row for row in combined if row["model"] == model and row["metric"] == metric and float(row["ratio"]) == ratio)
                lines.append(
                    f"| {ratio:g} | {model} | {METRIC_LABELS[metric]} | "
                    f"{float(row['bonf_recall']):.1%} | {float(row['bonf_precision']):.1%} | "
                    f"{float(row['bonf_f1']):.1%} | {float(row['bonf_mean_reported']):.2f} | "
                    f"{float(row['bonf_mean_fp']):.2f} |"
                )
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    comparison_plot(args.output, all_methods, ratios)
    print(f"Saved combined comparison to {args.output}; elapsed {time.perf_counter()-started:.1f}s")


if __name__ == "__main__":
    main()
