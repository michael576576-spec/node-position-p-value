"""Shared graph model and empirical centrality p-value inference."""

from __future__ import annotations

import random
from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache

from centrality import compute_scores


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


@lru_cache(maxsize=None)
def possible_edges(n: int) -> tuple[tuple[int, int], ...]:
    if n < 2:
        raise ValueError("n must be at least 2")
    return tuple((u, v) for u in range(n) for v in range(u + 1, n))


def gnm_edges(n: int, m: int, rng: random.Random) -> tuple[tuple[int, int], ...]:
    """Sample G(n,m): every simple graph with n vertices and m edges is equally likely."""
    candidates = possible_edges(n)
    if not 0 <= m <= len(candidates):
        raise ValueError(f"m must be between 0 and {len(candidates)}")
    return tuple(rng.sample(candidates, m))


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


def score_network(
    network: Network, null_scores: list[float], alpha: float, metric: str = "degree"
) -> list[dict[str, int | float | bool | str]]:
    """Compare node scores with a shared, sorted G(n,m) null distribution."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if not null_scores:
        raise ValueError("null_scores cannot be empty")

    observed_scores = compute_scores(network.n, network.edges, metric)
    observed_degrees = observed_scores if metric == "degree" else compute_scores(
        network.n, network.edges, "degree"
    )
    rows = []
    for position, node in enumerate(
        sorted(range(network.n), key=lambda u: (-observed_scores[u], u)), start=1
    ):
        score = observed_scores[node]
        # Upper tail includes ties. The added 1 avoids an estimated p-value of zero.
        at_least_as_large = len(null_scores) - bisect_left(null_scores, score)
        p_value = (at_least_as_large + 1) / (len(null_scores) + 1)
        bonferroni_p_value = min(1.0, p_value * network.n)
        rows.append(
            {
                "position": position,
                "node": node,
                "degree": int(observed_degrees[node]),
                "metric": metric,
                "centrality_score": score,
                "p_value": p_value,
                "predicted_hub": p_value < alpha,
                "bonferroni_p_value": bonferroni_p_value,
                "predicted_hub_bonferroni": bonferroni_p_value < alpha,
                "true_hub": node in network.hubs,
            }
        )
    return rows


def detect_hubs(
    network: Network, null_graphs: int, alpha: float, rng: random.Random, metric: str = "degree"
) -> list[dict[str, int | float | bool | str]]:
    """Compare scores with a freshly simulated G(n, observed m) null."""
    return score_network(
        network,
        build_null_scores(network.n, network.m, null_graphs, rng, metric),
        alpha,
        metric,
    )
