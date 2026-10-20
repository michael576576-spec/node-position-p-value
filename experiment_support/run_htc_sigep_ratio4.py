"""Run HTC and exact SigEP on ten G(5000, 0.01) graphs with 50 fourfold hubs."""

from __future__ import annotations

import json
import time
from pathlib import Path

from experiment_support.baseline_methods import (
    benjamini_hochberg,
    bonferroni,
    htc_p_values,
    sigep_p_values,
)
from experiment_support.experiment import (
    AGGREGATE_FIELDS,
    SUMMARY_FIELDS,
    _aggregate,
    _evaluate_decisions,
    _write_csv,
)
from experiment_support.simulated_method_comparison import SimulationSetting, generate_target


OUTPUT = Path("experiment_results/simulated_htc_sigep_n5000_p001_h50_ratio4")
SETTING = SimulationSetting("G7", 5000, 0.01, True)
SEED = 42
REPETITIONS = 10
ALPHA = 0.01
HUB_PROBABILITY_RATIO = 4.0


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for replicate in range(1, REPETITIONS + 1):
        network = generate_target(SETTING, replicate, SEED, HUB_PROBABILITY_RATIO)
        if len(network.hubs) != 50:
            raise ValueError("expected exactly 50 planted hubs")
        for method, metric, p_value_function, decision_function in (
            ("HTC", "neighbor_degree_sum", htc_p_values, bonferroni),
            (
                "SigEP",
                "degree+clustering",
                lambda graph: sigep_p_values(graph, mode="exact"),
                lambda p_values, alpha: benjamini_hochberg(p_values, alpha, strict=False),
            ),
        ):
            start = time.perf_counter()
            p_values = p_value_function(network)
            reported = decision_function(p_values, ALPHA)
            elapsed = time.perf_counter() - start
            row = _evaluate_decisions(
                SETTING.name,
                "positive",
                replicate,
                network.name,
                network.n,
                network.m,
                SETTING.p,
                network.hubs,
                method,
                metric,
                reported,
                elapsed,
                "gnm",
            )
            rows.append(row)
            print(
                f"rep={replicate:02d} M={network.m} {method}: "
                f"TP={row['reported_central_nodes'] - row['false_positives']} "
                f"FP={row['false_positives']} "
                f"reported={row['reported_central_nodes']}",
                flush=True,
            )

    _write_csv(OUTPUT / "network_summary.csv", rows, SUMMARY_FIELDS)
    _write_csv(OUTPUT / "aggregate_summary.csv", _aggregate(rows), AGGREGATE_FIELDS)
    (OUTPUT / "settings.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "repetitions": REPETITIONS,
                "alpha": ALPHA,
                "n": SETTING.n,
                "p": SETTING.p,
                "planted_hubs": 50,
                "hub_edge_probability_ratio": HUB_PROBABILITY_RATIO,
                "hub_edge_probability": SETTING.p * HUB_PROBABILITY_RATIO,
                "null_model": "gnm",
                "sigep_implementation": "conditional_hypergeometric_joint_tail",
                "sigep_bh_rule": "<=",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved results to {OUTPUT}")


if __name__ == "__main__":
    main()
