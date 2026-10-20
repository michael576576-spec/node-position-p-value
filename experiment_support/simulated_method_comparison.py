"""Simulation setup and empirical p-values for the method comparison."""

from __future__ import annotations

import random
from dataclasses import dataclass

from experiment_support.centrality import compute_multiple_scores_igraph
from experiment_support.network_core import Network
from experiment_support.null_models import NULL_MODELS
from experiment_support.simulation_data import _gnm_edges_for_simulation, make_gnp_network


ALPHA = 0.01
NULL_GRAPHS = 100
REPETITIONS = 10
METRIC_ORDER = ("degree", "harmonic", "betweenness", "pagerank")


@dataclass(frozen=True)
class SimulationSetting:
    name: str
    n: int
    p: float
    positive: bool


SETTINGS = tuple(
    SimulationSetting(f"G{index}", n, p, positive)
    for index, (n, p, positive) in enumerate(
        (
            (1000, 0.01, False),
            (1000, 0.005, False),
            (5000, 0.01, False),
            (5000, 0.005, False),
            (1000, 0.01, True),
            (1000, 0.005, True),
            (5000, 0.01, True),
            (5000, 0.005, True),
        ),
        start=1,
    )
)

# A compact pilot that keeps the expected average degree near the full-scale
# settings while reducing each graph to 200 nodes.
PILOT_SETTINGS = (
    SimulationSetting("pilot_negative_p050", 200, 0.05, False),
    SimulationSetting("pilot_negative_p025", 200, 0.025, False),
    SimulationSetting("pilot_positive_p050", 200, 0.05, True),
    SimulationSetting("pilot_positive_p025", 200, 0.025, True),
)


def generate_target(
    setting: SimulationSetting, repetition: int, seed: int, hub_probability_ratio: float = 10.0
) -> Network:
    """Generate one independent target graph for a setting and replicate."""
    rng = random.Random(f"{seed}:{setting.name}:{repetition}:target")
    hub_count = round(setting.n * 0.01) if setting.positive else 0
    return make_gnp_network(
        f"{setting.name}_rep{repetition:02d}",
        setting.n,
        setting.p,
        hub_count,
        rng,
        hub_probability_ratio,
    )


def empirical_bonferroni_results(
    network: Network,
    null_graphs: int,
    alpha: float,
    rng: random.Random,
    null_model: str = "gnm",
) -> dict[str, list[dict[str, int | float | bool | str]]]:
    """Compute pooled-node empirical tails and Bonferroni decisions.

    For each metric, all nodes from each independently generated null graph
    enter the reference distribution. GNM graphs preserve the target edge
    count; configuration and Chung-Lu graphs are fitted to the target graph.
    This streams each null graph and retains only exceedance counts, avoiding
    an O(N * B) score matrix.
    """
    if null_graphs < 1:
        raise ValueError("null_graphs must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if null_model not in NULL_MODELS:
        raise ValueError(
            f"unknown null model '{null_model}'; choose from {', '.join(NULL_MODELS)}"
        )

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("This comparison requires NumPy: python -m pip install numpy") from exc

    observed = compute_multiple_scores_igraph(network.n, network.edges, METRIC_ORDER)
    observed_arrays = {metric: np.asarray(observed[metric], dtype=float) for metric in METRIC_ORDER}
    exceedances = {metric: np.zeros(network.n, dtype=np.int64) for metric in METRIC_ORDER}
    for _ in range(null_graphs):
        if null_model == "gnm":
            edges = _gnm_edges_for_simulation(network.n, network.m, rng)
        else:
            edges = NULL_MODELS[null_model](network.n, network.edges, rng)
        null_scores = compute_multiple_scores_igraph(network.n, edges, METRIC_ORDER)
        for metric in METRIC_ORDER:
            reference = np.sort(np.asarray(null_scores[metric], dtype=float))
            exceedances[metric] += network.n - np.searchsorted(
                reference, observed_arrays[metric], side="left"
            )

    total_reference = network.n * null_graphs
    results: dict[str, list[dict[str, int | float | bool | str]]] = {}
    for metric in METRIC_ORDER:
        rows = []
        for node, score in enumerate(observed[metric]):
            p_value = (int(exceedances[metric][node]) + 1) / (total_reference + 1)
            adjusted_p = min(1.0, network.n * p_value)
            rows.append(
                {
                    "network": network.name,
                    "node": node,
                    "null_model": null_model,
                    "metric": metric,
                    "centrality_score": score,
                    "p_value": p_value,
                    "adjusted_p_value": adjusted_p,
                    "reported": adjusted_p <= alpha,
                    "true_hub": node in network.hubs,
                }
            )
        results[metric] = rows
    return results

