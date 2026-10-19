"""GNP negative and planted-hub target graphs for the method comparison."""

from __future__ import annotations

import random
from math import floor, isqrt, log1p

from experiment_support.network_core import Network
from experiment_support.null_models import Edge


def _edge_from_index(n: int, index: int) -> Edge:
    """Map a compact upper-triangle edge index to its endpoint pair."""
    discriminant = (2 * n - 1) ** 2 - 8 * index
    left = ((2 * n - 1) - isqrt(discriminant)) // 2
    row_start = left * (2 * n - left - 1) // 2
    while row_start > index:
        left -= 1
        row_start = left * (2 * n - left - 1) // 2
    while left + 1 < n and (left + 1) * (2 * n - left - 2) // 2 <= index:
        left += 1
        row_start = left * (2 * n - left - 1) // 2
    return left, left + 1 + index - row_start


def _gnm_edges_for_simulation(n: int, m: int, rng: random.Random) -> tuple[Edge, ...]:
    """Sample G(n,m) without materializing all possible node pairs."""
    total = n * (n - 1) // 2
    if n < 2 or not 0 <= m <= total:
        raise ValueError("invalid n or m for G(n,m)")
    return tuple(_edge_from_index(n, index) for index in rng.sample(range(total), m))


def _gnp_edges(n: int, p: float, rng: random.Random) -> tuple[Edge, ...]:
    """Sample G(n,p) by geometric skipping, without scanning all node pairs."""
    if n < 2 or not 0 <= p <= 1:
        raise ValueError("invalid n or p for G(n,p)")
    if p == 0:
        return ()
    if p == 1:
        raise ValueError("p=1 is not supported for sparse simulation generation")
    total = n * (n - 1) // 2
    log_q = log1p(-p)
    edges = []
    index = 0
    while True:
        index += floor(log1p(-rng.random()) / log_q)
        if index >= total:
            break
        edges.append(_edge_from_index(n, index))
        index += 1
    return tuple(edges)


def make_gnp_network(
    name: str,
    n: int,
    p: float,
    hub_count: int,
    rng: random.Random,
    hub_probability_ratio: float = 10.0,
) -> Network:
    """Generate a G(n,p) negative or a graph with planted high-degree hubs.

    Base edges are sampled from G(n,p). For a positive graph, each edge
    incident to at least one planted hub receives an independent additional
    chance of being added so that its final probability is exactly
    ``hub_probability_ratio * p``.
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if not 0 < p < 1:
        raise ValueError("p must be between 0 and 1")
    if not 0 <= hub_count < n:
        raise ValueError("hub_count must be between 0 and n - 1")
    if hub_probability_ratio < 1 or hub_probability_ratio * p > 1:
        raise ValueError("hub probability ratio must produce a probability <= 1")

    hubs = frozenset(rng.sample(range(n), hub_count)) if hub_count else frozenset()
    ordered_hubs = tuple(sorted(hubs))
    edges: set[Edge] = set(_gnp_edges(n, p, rng))
    if hubs and hub_probability_ratio > 1:
        extra_probability = (hub_probability_ratio * p - p) / (1.0 - p)
        ordinary_nodes = [node for node in range(n) if node not in hubs]
        for hub in ordered_hubs:
            for node in ordinary_nodes:
                edge = (min(hub, node), max(hub, node))
                if edge not in edges and rng.random() < extra_probability:
                    edges.add(edge)
        for index, left in enumerate(ordered_hubs):
            for right in ordered_hubs[index + 1 :]:
                edge = (min(left, right), max(left, right))
                if edge not in edges and rng.random() < extra_probability:
                    edges.add(edge)

    ordered_edges = tuple(sorted(edges))
    return Network(
        name=name,
        n=n,
        edges=ordered_edges,
        hubs=hubs,
        target_m=round(
            (n - hub_count) * (n - hub_count - 1) // 2 * p
            + (n * (n - 1) // 2 - (n - hub_count) * (n - hub_count - 1) // 2)
            * hub_probability_ratio * p
        ),
        hub_count=hub_count,
        hub_probability=hub_probability_ratio * p if hub_count else None,
        background_probability=p if hub_count else None,
    )
