"""Run a paired hub-strength sweep for configuration and Chung-Lu nulls.

The default experiment uses 50 target graphs per ratio, 1000 null graphs per
target graph and model, and all four centrality metrics. Completed target
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

from centrality import METRICS, compute_multiple_scores_igraph, compute_scores
from compare_metrics import METRIC_LABELS, write_comparison_plot
from network_core import Network, NullDistribution, score_network
from null_models import NULL_MODELS
from ratio_sweep import DEFAULT_RATIOS, classification_rates, target_graphs


MODELS = ("configuration", "chung_lu")
METRIC_ORDER = tuple(METRICS)
BACKEND = "igraph-1.0.0"
SUMMARY_FIELDS = (
    "model", "metric", "ratio", "ordinary_probability", "hub_probability",
    "bonf_recall", "bonf_precision", "bonf_f1", "bonf_mean_reported", "bonf_mean_fp",
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
    """Generate one shared collection of null graphs and score all four metrics."""
    metrics = METRIC_ORDER
    generator = NULL_MODELS[model]
    rng = random.Random(
        f"{seed}:null:{network.n}:{network.m}:{repetition}:{ratio:g}:{model}"
    )
    pooled: dict[str, list[float]] = {metric: [] for metric in metrics}
    if model == "configuration":
        # Every switch preserves the multiset of degrees exactly.
        pooled["degree"] = compute_scores(network.n, network.edges, "degree") * null_graphs
        sampled_metrics = tuple(metric for metric in metrics if metric != "degree")
    else:
        sampled_metrics = metrics

    for _ in range(null_graphs):
        null_edges = generator(network.n, network.edges, rng)
        scores = compute_multiple_scores_igraph(network.n, null_edges, sampled_metrics)
        for metric in sampled_metrics:
            pooled[metric].extend(scores[metric])

    observed = compute_multiple_scores_igraph(network.n, network.edges, metrics)
    results: dict[str, dict] = {}
    for metric in metrics:
        values = pooled[metric]
        if len(values) != network.n * null_graphs:
            raise ArithmeticError(f"wrong null score count for {model}/{metric}")
        values.sort()
        node_rows = score_network(
            network, NullDistribution(model, values), alpha, metric,
            bonferroni_only=True, observed_scores=observed[metric],
        )
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
        "无报告时，图级精确率和 F1 记为 0。配置模型严格保持各节点度数；"
        "汇总分数得到的是相对随机节点整体的尾部概率。",
        "由于不同节点的零分布并不相同，这一汇总尾概率不是固定节点条件下的 p 值；"
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


def comparison_plot(root: Path, rows: list[dict], ratios: tuple[float, ...]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "node_position_pvalue_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = (("gnm", "GNM"), ("configuration", "Configuration"), ("chung_lu", "Chung–Lu"))
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    measures = (("bonf_recall", "Recall"), ("bonf_precision", "Precision"), ("bonf_f1", "F1 Score"))
    for i, (model, model_label) in enumerate(models):
        for j, (key, title) in enumerate(measures):
            axis = axes[i, j]
            for metric in METRIC_ORDER:
                selected = [row for row in rows if row["model"] == model and row["metric"] == metric]
                if not selected:
                    continue
                by_ratio = {float(row["ratio"]): float(row[key]) for row in selected}
                axis.plot(ratios, [by_ratio[ratio] for ratio in ratios], marker="o", label=METRIC_LABELS[metric])
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


def baseline_rows(args: argparse.Namespace) -> list[dict]:
    pilot_path = args.output / "gnm" / "comparison.csv"
    defaults_match = (
        args.n == 100 and args.m == 350 and args.hubs == 3
        and tuple(args.ratios) == DEFAULT_RATIOS and args.repetitions == 50
        and args.null_graphs == 1000 and args.alpha == 0.05 and args.seed == 42
    )
    path = pilot_path if pilot_path.is_file() else (
        Path("experiment_results/metric_comparison/comparison.csv") if defaults_match else pilot_path
    )
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    expected = {(metric, ratio) for metric in METRIC_ORDER for ratio in args.ratios}
    actual = {(row["metric"], float(row["ratio"])) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise ValueError("GNM baseline has unexpected metrics or ratios")
    return [{"model": "gnm", **row} for row in rows]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--m", type=int, default=350)
    parser.add_argument("--hubs", type=int, default=3)
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--null-graphs", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, default=Path("experiment_results/null_model_comparison"))
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
        "pooling": "all_nodes", "backend": BACKEND, "version": 2,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    settings_path = args.output / "settings.json"
    if settings_path.exists():
        existing = json.loads(settings_path.read_text(encoding="utf-8"))
        if existing != json.loads(json.dumps(settings)):
            previous = {key: value for key, value in existing.items() if key not in {"backend", "version"}}
            current = {
                key: value for key, value in json.loads(json.dumps(settings)).items()
                if key not in {"backend", "version"}
            }
            if previous != current or existing.get("version") != 1:
                raise ValueError(f"settings differ from existing run in {args.output}")
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
    baseline = baseline_rows(args)
    combined = baseline + summaries
    write_csv(args.output / "comparison.csv", combined, SUMMARY_FIELDS)
    lines = [
        "# 零模型与中心性指标比较", "",
        f"固定 N={args.n}、M={args.m}、中心节点数={args.hubs}；每倍率 {args.repetitions} 张目标图；"
        f"每张目标图使用 {args.null_graphs} 张配置模型或 Chung–Lu 零模型图。",
        "每张零模型图全部节点的中心性分数参与经验尾概率计算。",
        f"Bonferroni 校正 p < {args.alpha}。各模型使用相同目标图和植入中心节点。",
        "配置模型严格保持各节点度数；其汇总 p 值衡量相对随机节点整体的极端程度，"
        "不能解释为固定该节点度数后的逐节点 p 值。",
        "Bonferroni 阈值用于统一比较；对这两种汇总零分布，不据此声称严格的逐节点误报率控制。", "",
        "| 倍率 | 零模型 | 指标 | 召回率 | 精确率 | F1 | 平均报告数 | 平均误报数 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if baseline:
        lines.insert(6, "GNM 行来自已有的同参数四指标实验。")
    for ratio in ratios:
        for model in (("gnm",) if baseline else ()) + MODELS:
            for metric in METRIC_ORDER:
                row = next(row for row in combined if row["model"] == model and row["metric"] == metric and float(row["ratio"]) == ratio)
                lines.append(
                    f"| {ratio:g} | {model} | {METRIC_LABELS[metric]} | "
                    f"{float(row['bonf_recall']):.1%} | {float(row['bonf_precision']):.1%} | "
                    f"{float(row['bonf_f1']):.1%} | {float(row['bonf_mean_reported']):.2f} | "
                    f"{float(row['bonf_mean_fp']):.2f} |"
                )
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    comparison_plot(args.output, combined, ratios)
    print(f"Saved combined comparison to {args.output}; elapsed {time.perf_counter()-started:.1f}s")


if __name__ == "__main__":
    main()
