"""GNM hub-strength sweep: N=1000, M=6000, K=10, eight ratios."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from compare_null_models import (
    ALL_METHOD_FIELDS,
    BACKEND,
    BASELINE_METHODS,
    METRIC_ORDER,
    SUMMARY_FIELDS,
    baseline_checkpoint_path,
    centrality_as_all_method,
    checkpoint_path,
    run_baselines_target,
    run_target,
    summarize_baselines,
    summarize_model,
    write_csv,
)
from experiment_support.compare_metrics import write_comparison_plot
from experiment_support.ratio_sweep import target_graphs


N = 1000
M = 6000
HUBS = 10
RATIOS = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0)
REPETITIONS = 10
NULL_GRAPHS = 100
ALPHA = 0.01
SEED = 42
OUTPUT = Path("experiment_results/central_strength_gnm_n1000_m6000_k10_b100_alpha001")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    settings = {
        "n": N, "m": M, "hubs": HUBS, "ratios": RATIOS,
        "repetitions": REPETITIONS, "null_graphs": NULL_GRAPHS,
        "alpha": ALPHA, "seed": SEED, "models": ("gnm",),
        "metrics": METRIC_ORDER, "gnm_baselines": BASELINE_METHODS,
        "baseline_rules": {"HTC": "Bonferroni <=", "SigEP": "BH <=, exact joint tail"},
        "pooling": "all_nodes_per_target", "backend": BACKEND,
    }
    settings_path = output / "settings.json"
    if settings_path.exists():
        if json.loads(settings_path.read_text(encoding="utf-8")) != json.loads(json.dumps(settings)):
            raise ValueError(f"settings differ from existing run: {output}")
    else:
        settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    targets = list(target_graphs(N, M, HUBS, RATIOS, REPETITIONS, SEED))
    baseline_pending = [
        (ratio, repetition, network, baseline_checkpoint_path(output, network))
        for ratio, repetition, network in targets
        if not baseline_checkpoint_path(output, network).exists()
    ]
    gnm_pending = [
        (ratio, repetition, network, checkpoint_path(output, "gnm", network))
        for ratio, repetition, network in targets
        if not checkpoint_path(output, "gnm", network).exists()
    ]
    print(
        f"Target graphs: {len(targets)}; pending HTC/SigEP: {len(baseline_pending)}; "
        f"pending GNM null-model tasks: {len(gnm_pending)}",
        flush=True,
    )
    started = time.perf_counter()
    if baseline_pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_baselines_target, ratio, repetition, network, ALPHA, path)
                for ratio, repetition, network, path in baseline_pending
            ]
            for finished, future in enumerate(as_completed(futures), 1):
                future.result()
                if finished % 10 == 0 or finished == len(futures):
                    print(f"HTC/SigEP {finished}/{len(futures)}; elapsed {time.perf_counter()-started:.1f}s", flush=True)
    if gnm_pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    run_target, "gnm", ratio, repetition, network,
                    NULL_GRAPHS, ALPHA, SEED, path,
                )
                for ratio, repetition, network, path in gnm_pending
            ]
            for finished, future in enumerate(as_completed(futures), 1):
                future.result()
                if finished % 5 == 0 or finished == len(futures):
                    print(f"GNM {finished}/{len(futures)}; elapsed {time.perf_counter()-started:.1f}s", flush=True)

    centrality = summarize_model(output, "gnm", targets, REPETITIONS, NULL_GRAPHS, ALPHA)
    baselines = summarize_baselines(output, targets, RATIOS, REPETITIONS, ALPHA)
    write_csv(output / "comparison.csv", centrality, SUMMARY_FIELDS)
    all_methods = [centrality_as_all_method(row) for row in centrality] + baselines
    write_csv(output / "comparison_all_methods.csv", all_methods, ALL_METHOD_FIELDS)
    write_csv(output / "gnm" / "comparison_all_methods.csv", all_methods, ALL_METHOD_FIELDS)
    plot_rows = centrality + [
        {
            **row,
            "bonf_recall": row["recall"], "bonf_precision": row["precision"],
            "bonf_f1": row["f1"],
        }
        for row in baselines
    ]
    write_comparison_plot(output, plot_rows, METRIC_ORDER + BASELINE_METHODS)
    write_comparison_plot(output / "gnm", plot_rows, METRIC_ORDER + BASELINE_METHODS)
    (output / "report.md").write_text(
        "# GNM 中心节点强度实验\n\n"
        f"固定 N={N}、M={M}、植入节点数={HUBS}，总体边密度为 {M / (N * (N - 1) / 2):.6%}。"
        f"每个强度 {REPETITIONS} 张配对目标图，每张目标图生成 {NULL_GRAPHS} 张 GNM 参考图；"
        f"显著性水平 alpha={ALPHA}，随机种子={SEED}。\n\n"
        "强度为中心相关边与普通边的边际连边概率比，固定总边数 M。"
        "四项中心性用经验 p 值和 Bonferroni <；HTC 用 Bonferroni <=；"
        "SigEP 用条件超几何联合尾概率与 BH <=。\n\n"
        "逐图先计算召回率、精确率和 F1，再对同一强度的图取平均；"
        "若某图没有报告节点，精确率和 F1 记为 0。\n\n"
        "结果： [六方法趋势图](comparison.png) · "
        "[汇总数据](comparison_all_methods.csv) · "
        "[逐图数据](gnm/replicates.csv) · "
        "[HTC/SigEP 逐图数据](gnm/baseline_replicates.csv)\n",
        encoding="utf-8",
    )
    print(f"Saved results to {output}; elapsed {time.perf_counter()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
