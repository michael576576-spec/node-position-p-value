"""Scan hub strength and hub count across several fixed-(N,M) scenarios.

Each target graph is shared by three null models and six centrality metrics.
Only Bonferroni-adjusted decisions are used. Completed graph/model pairs are
checkpointed for resumption.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from compare_null_models import BACKEND, METRIC_LABELS, METRIC_ORDER, MODELS, run_target
from experiment_support.ratio_sweep import DEFAULT_RATIOS, target_graphs


def parse_scenario(value: str) -> tuple[int, int]:
    try:
        n, m = (int(part) for part in value.split(":"))
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("scenario must be N:M, e.g. 100:350") from error
    if n < 3 or not 0 < m < math.comb(n, 2):
        raise argparse.ArgumentTypeError("scenario requires N >= 3 and 0 < M < N*(N-1)/2")
    return n, m


def checkpoint_path(root: Path, model: str, graph: str) -> Path:
    return root / "checkpoints" / model / f"{graph}.json"


def build_targets(scenarios, ratios, hub_counts, baseline_hubs, count_ratio,
                  sweeps, repetitions, seed):
    targets = []
    for scenario_index, (n, m) in enumerate(scenarios, start=1):
        combinations = []
        if "strength" in sweeps:
            combinations.extend((baseline_hubs, ratio) for ratio in ratios)
        if "count" in sweeps:
            combinations.extend((hubs, count_ratio) for hubs in hub_counts)
        for hubs, ratio_value in dict.fromkeys(combinations):
            for ratio, repetition, network in target_graphs(
                n, m, hubs, (ratio_value,), repetitions, seed
            ):
                name = f"s{scenario_index:02d}_k{hubs}_r{ratio:g}_rep{repetition:02d}"
                targets.append((scenario_index, ratio, hubs, repetition,
                                replace(network, name=name)))
    return targets


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def summarize(root: Path, targets, scenarios, ratios, hub_counts,
              baseline_hubs, count_ratio, sweeps, repetitions, null_graphs, alpha):
    replicate_rows = []
    grouped = {}
    for scenario_index, ratio, hubs, repetition, network in targets:
        for model in MODELS:
            path = checkpoint_path(root, model, network.name)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (payload.get("graph") != network.name
                    or payload.get("model") != model
                    or payload.get("ratio") != ratio
                    or payload.get("repetition") != repetition
                    or payload.get("hubs") != sorted(network.hubs)
                    or payload.get("null_graphs") != null_graphs
                    or payload.get("backend") != BACKEND):
                raise ValueError(f"checkpoint does not match experiment: {path}")
            for metric in METRIC_ORDER:
                result = payload["metrics"][metric]
                row = {
                    "scenario": scenario_index, "n": network.n, "m": network.m,
                    "hubs": hubs, "ratio": ratio, "repetition": repetition,
                    "graph": network.name, "model": model,
                    "metric": METRIC_LABELS[metric], "metric_key": metric,
                    "ordinary_probability": payload["ordinary_probability"],
                    "hub_probability": payload["hub_probability"],
                    "reported": result["reported"],
                    "tp": result["true_positives"],
                    "fp": result["false_positives"],
                    "no_report": int(result["reported"] == 0),
                    "recall": result["recall"],
                    "precision": result["precision"], "f1": result["f1"],
                }
                replicate_rows.append(row)
                grouped.setdefault((scenario_index, hubs, ratio, model, metric), []).append(row)

    write_csv(root / "replicates.csv", replicate_rows, list(replicate_rows[0]))
    summary_rows = []
    for scenario_index, (n, m) in enumerate(scenarios, start=1):
        for sweep, points in (
            ("strength", [(baseline_hubs, ratio, ratio) for ratio in ratios]),
            ("count", [(hubs, count_ratio, hubs) for hubs in hub_counts]),
        ):
            if sweep not in sweeps:
                continue
            for hubs, ratio, x in points:
                for model in MODELS:
                    for metric in METRIC_ORDER:
                        rows = grouped[(scenario_index, hubs, ratio, model, metric)]
                        if len(rows) != repetitions:
                            raise ArithmeticError(f"missing replicates: {scenario_index}/{hubs}/{ratio}/{model}/{metric}")
                        summary = {
                            "scenario": scenario_index, "sweep": sweep,
                            "n": n, "m": m, "hubs": hubs, "ratio": ratio,
                            "x": x, "model": model, "metric": METRIC_LABELS[metric],
                            "ordinary_probability": rows[0]["ordinary_probability"],
                            "hub_probability": rows[0]["hub_probability"],
                            "graphs": len(rows),
                        }
                        for key in ("reported", "tp", "fp", "no_report",
                                    "recall", "precision", "f1"):
                            values = [float(row[key]) for row in rows]
                            avg = sum(values) / len(values)
                            summary[f"mean_{key}"] = avg
                            if key in ("recall", "precision", "f1"):
                                variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1) if len(values) > 1 else 0.0
                                half_width = 1.96 * math.sqrt(variance / len(values))
                                summary[f"{key}_ci_low"] = max(0.0, avg - half_width)
                                summary[f"{key}_ci_high"] = min(1.0, avg + half_width)
                        summary_rows.append(summary)

    write_csv(root / "summary.csv", summary_rows, list(summary_rows[0]))
    lines = [
        ("# 中心结构：中心节点数量" if sweeps == ("count",) else
         "# 中心结构：中心强度" if sweeps == ("strength",) else
         "# 中心结构：强度与中心节点数量"), "",
        f"{len(scenarios)} 组 (N,M)：{', '.join(f'({n},{m})' for n, m in scenarios)}；"
        f"每个参数点 {repetitions} 张目标图，每张目标图/零模型 {null_graphs} 张零图。",
        (f"强度扫描固定 K={baseline_hubs}。" if "strength" in sweeps else "")
        + (f"数量扫描固定概率比={count_ratio:g}。" if "count" in sweeps else ""),
        "各场景分别汇总，三种零模型使用相同目标图。DC、LAC、EC、BC、CC、GPG 共享各模型的零图。",
        f"只使用 Bonferroni 校正 p 值，阈值为 {alpha}；无报告时 Precision 和 F1 记为 0。",
        "Recall、Precision、F1 先逐图计算再求平均；阴影为均值的近似 95% 区间（正态标准误，截断到 0–1）。",
        "固定 N、M 和概率比时，增加 K 会降低中心边及普通边的边际概率；"
        "数量扫描反映这些约束下的整体变化，不能单独归因于 K。两种概率见 summary.csv。",
        "配置模型保持每个标号节点的度数；配置模型与 Chung–Lu 的全节点汇总零分布"
        "不等同于固定节点条件下的零分布。", "",
    ]
    for scenario_index, (n, m) in enumerate(scenarios, start=1):
        directory = root / f"scenario_{scenario_index:02d}_n{n}_m{m}"
        directory.mkdir(parents=True, exist_ok=True)
        links = []
        for sweep in sweeps:
            selected = [row for row in summary_rows if row["scenario"] == scenario_index and row["sweep"] == sweep]
            write_csv(directory / f"{sweep}.csv", selected, list(selected[0]))
            plot_sweep(directory / f"{sweep}.png", selected, sweep)
            plot_sweep(directory / f"{sweep}.svg", selected, sweep)
            label = "强度" if sweep == "strength" else "数量"
            links.extend((
                f"[{label}趋势图](scenario_{scenario_index:02d}_n{n}_m{m}/{sweep}.png)",
                f"[{label}数据](scenario_{scenario_index:02d}_n{n}_m{m}/{sweep}.csv)",
            ))
        lines += [
            f"## 场景 {scenario_index}：N={n}, M={m}", "",
            " · ".join(links), "",
        ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def plot_sweep(path: Path, rows: list[dict], sweep: str) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "node_position_pvalue_matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, sharey=True)
    for model_index, model in enumerate(MODELS):
        for measure_index, (key, label) in enumerate((
            ("recall", "Recall"), ("precision", "Precision"), ("f1", "F1 Score")
        )):
            axis = axes[model_index, measure_index]
            for metric in METRIC_ORDER:
                selected = sorted((row for row in rows if row["model"] == model and row["metric"] == METRIC_LABELS[metric]), key=lambda row: row["x"])
                x = [row["x"] for row in selected]
                line, = axis.plot(x, [row[f"mean_{key}"] for row in selected], marker="o", label=METRIC_LABELS[metric])
                axis.fill_between(x, [row[f"{key}_ci_low"] for row in selected],
                                  [row[f"{key}_ci_high"] for row in selected],
                                  color=line.get_color(), alpha=0.08)
            axis.set_title(f"{model} — {label}")
            axis.set_ylim(0, 1.04)
            axis.set_xticks(sorted({row["x"] for row in rows}))
            axis.set_xlabel("Hub / ordinary edge probability ratio" if sweep == "strength" else "Number of planted hubs")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
            axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180 if path.suffix == ".png" else None)
    plt.close(fig)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=parse_scenario, nargs="+",
                        default=[(60, 90), (100, 350), (150, 900)])
    parser.add_argument("--sweeps", nargs="+", choices=("strength", "count"),
                        default=("strength", "count"))
    parser.add_argument("--ratios", type=float, nargs="+", default=DEFAULT_RATIOS)
    parser.add_argument("--hub-counts", type=int, nargs="+", default=(1, 2, 3, 5, 8))
    parser.add_argument("--baseline-hubs", type=int, default=3)
    parser.add_argument("--count-ratio", type=float, default=4.0)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--null-graphs", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path,
                        default=Path("experiment_results/central_structure_sweeps"))
    args = parser.parse_args()
    scenarios = tuple(args.scenarios)
    sweeps = tuple(sweep for sweep in ("strength", "count") if sweep in args.sweeps)
    ratios = tuple(args.ratios)
    hub_counts = tuple(args.hub_counts)
    if len(scenarios) != len(set(scenarios)):
        parser.error("scenarios must be distinct")
    if len(args.sweeps) != len(set(args.sweeps)):
        parser.error("sweeps must not contain duplicates")
    if len(ratios) != len(set(ratios)) or not ratios or any(not math.isfinite(value) or value < 1 for value in ratios):
        parser.error("ratios must be distinct finite values >= 1")
    if len(hub_counts) != len(set(hub_counts)) or not hub_counts or any(k < 1 for k in hub_counts):
        parser.error("hub-counts must be distinct positive integers")
    if "strength" in sweeps and "count" in sweeps and (
        args.baseline_hubs not in hub_counts or args.count_ratio not in ratios
    ):
        parser.error("for both sweeps, baseline-hubs must be in hub-counts and count-ratio in ratios")
    if not math.isfinite(args.count_ratio) or args.count_ratio < 1:
        parser.error("count-ratio must be finite and >= 1")
    if args.repetitions < 1 or args.null_graphs < 1 or args.workers < 1 or not 0 < args.alpha < 1:
        parser.error("repetitions, null-graphs and workers must be positive; alpha must be in (0,1)")
    used_hubs = (hub_counts if "count" in sweeps else (args.baseline_hubs,))
    if any(max(used_hubs) > n - 2 for n, _ in scenarios):
        parser.error("every hub count must leave at least two ordinary nodes")
    try:
        targets = build_targets(scenarios, ratios, hub_counts, args.baseline_hubs,
                                args.count_ratio, sweeps, args.repetitions, args.seed)
    except (ValueError, ArithmeticError) as error:
        parser.error(str(error))
    settings = {
        "scenarios": scenarios, "sweeps": sweeps,
        "ratios": ratios, "hub_counts": hub_counts,
        "baseline_hubs": args.baseline_hubs, "count_ratio": args.count_ratio,
        "repetitions": args.repetitions, "null_graphs": args.null_graphs,
        "alpha": args.alpha, "seed": args.seed, "models": MODELS,
        "metrics": METRIC_ORDER, "pooling": "all_nodes_per_target",
        "backend": BACKEND, "p_values": "bonferroni_only", "version": 1,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    settings_path = args.output / "settings.json"
    normalized = json.loads(json.dumps(settings))
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != normalized:
            raise ValueError(f"settings differ from existing run: {args.output}")
    else:
        settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    pending = [
        (model, ratio, repetition, network, checkpoint_path(args.output, model, network.name))
        for _, ratio, _, repetition, network in targets for model in MODELS
        if not checkpoint_path(args.output, model, network.name).exists()
    ]
    print(f"Target graphs: {len(targets)}; pending graph/model tasks: {len(pending)}", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_target, model, ratio, repetition, network,
                                       args.null_graphs, args.alpha, args.seed, path)
                       for model, ratio, repetition, network, path in pending]
            for finished, future in enumerate(as_completed(futures), start=1):
                future.result()
                if finished % 10 == 0 or finished == len(futures):
                    print(f"Completed {finished}/{len(futures)}", flush=True)
    summarize(args.output, targets, scenarios, ratios, hub_counts,
              args.baseline_hubs, args.count_ratio, sweeps, args.repetitions,
              args.null_graphs, args.alpha)
    print(f"Report: {args.output / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
