"""Shared graph model and empirical centrality p-value inference."""

from __future__ import annotations

import random
from bisect import bisect_left
from dataclasses import dataclass

from experiment_support.centrality import compute_scores
from experiment_support.null_models import NULL_MODELS, gnm_edges, possible_edges

NULL_MODEL_CHOICES = tuple(NULL_MODELS)


@dataclass(frozen=True)
class Network:
    name: str
    n: int
    edges: tuple[tuple[int, int], ...]
    hubs: frozenset[int]
    target_m: int
    hub_count: int
    hub_probability: float | None
    background_probability: float | None

    @property
    def m(self) -> int:
        return len(self.edges)


def build_null_scores(
    n: int, m: int, null_graphs: int, rng: random.Random, metric: str = "degree"
) -> list[float]:
    """Pool and sort the selected centrality scores from G(n,m) graphs."""
    if null_graphs < 1:
        raise ValueError("null_graphs must be positive")
    null_scores: list[float] = []
    for _ in range(null_graphs):
        null_scores.extend(compute_scores(n, gnm_edges(n, m, rng), metric))
    null_scores.sort()
    return null_scores


@dataclass(frozen=True)
class NullDistribution:
    """Sorted centrality scores pooled across all nodes in null graphs."""

    model: str
    scores: list[float]


def build_null_distribution(
    network: Network,
    null_graphs: int,
    rng: random.Random,
    metric: str = "degree",
    model: str = "gnm",
) -> NullDistribution:
    """Fit a null to a target graph and simulate centrality scores."""
    if model not in NULL_MODEL_CHOICES:
        raise ValueError(f"unknown null model '{model}'; choose from {', '.join(NULL_MODEL_CHOICES)}")
    if null_graphs < 1:
        raise ValueError("null_graphs must be positive")
    if model == "gnm":
        return NullDistribution("gnm", build_null_scores(network.n, network.m, null_graphs, rng, metric))

    if model == "configuration" and metric == "degree":
        degrees = compute_scores(network.n, network.edges, "degree")
        return NullDistribution(model, sorted(degrees * null_graphs))

    generate = NULL_MODELS[model]
    scores: list[float] = []
    for _ in range(null_graphs):
        scores.extend(compute_scores(network.n, generate(network.n, network.edges, rng), metric))
    scores.sort()
    return NullDistribution(model, scores)


def score_network(
    network: Network,
    null_scores: list[float] | NullDistribution,
    alpha: float,
    metric: str = "degree",
    bonferroni_only: bool = False,
    observed_scores: list[float] | None = None,
) -> list[dict[str, int | float | bool | str]]:
    """Compare scores to the selected model's sorted null distribution."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    pooled_scores = null_scores.scores if isinstance(null_scores, NullDistribution) else null_scores
    if not pooled_scores:
        raise ValueError("null_scores cannot be empty")

    observed_scores = observed_scores if observed_scores is not None else compute_scores(
        network.n, network.edges, metric
    )
    if len(observed_scores) != network.n:
        raise ValueError("observed scores must match network size")
    observed_degrees = observed_scores if metric == "degree" else compute_scores(
        network.n, network.edges, "degree"
    )
    rows = []
    for position, node in enumerate(
        sorted(range(network.n), key=lambda u: (-observed_scores[u], u)), start=1
    ):
        score = observed_scores[node]
        # Upper tail includes ties. The added 1 avoids an estimated p-value of zero.
        reference = pooled_scores
        at_least_as_large = len(reference) - bisect_left(reference, score)
        bonferroni_p_value = min(
            1.0, network.n * (at_least_as_large + 1) / (len(reference) + 1)
        )
        row = {
                "position": position,
                "node": node,
                "degree": int(observed_degrees[node]),
                "metric": metric,
                "centrality_score": score,
                "bonferroni_p_value": bonferroni_p_value,
                "predicted_hub_bonferroni": bonferroni_p_value < alpha,
                "true_hub": node in network.hubs,
        }
        if not bonferroni_only:
            p_value = (at_least_as_large + 1) / (len(reference) + 1)
            row["p_value"] = p_value
            row["predicted_hub"] = p_value < alpha
        rows.append(row)
    return rows


def detect_hubs(
    network: Network, null_graphs: int, alpha: float, rng: random.Random,
    metric: str = "degree", model: str = "gnm"
) -> list[dict[str, int | float | bool | str]]:
    """Compare scores with a freshly simulated selected null model."""
    return score_network(
        network,
        build_null_distribution(network, null_graphs, rng, metric, model),
        alpha,
        metric,
    )
