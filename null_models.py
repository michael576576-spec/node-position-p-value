"""Simple-graph null models used by centrality inference."""

from __future__ import annotations

import random
from collections.abc import Callable
from functools import lru_cache

from centrality import Edge


@lru_cache(maxsize=None)
def possible_edges(n: int) -> tuple[Edge, ...]:
    if n < 2:
        raise ValueError("n must be at least 2")
    return tuple((u, v) for u in range(n) for v in range(u + 1, n))


def gnm_edges(n: int, m: int, rng: random.Random) -> tuple[Edge, ...]:
    """Uniform simple G(n,m) graph."""
    candidates = possible_edges(n)
    if not 0 <= m <= len(candidates):
        raise ValueError(f"m must be between 0 and {len(candidates)}")
    return tuple(rng.sample(candidates, m))


def _gnm_from_reference(n: int, edges: tuple[Edge, ...], rng: random.Random) -> tuple[Edge, ...]:
    return gnm_edges(n, len(edges), rng)


def _degrees(n: int, edges: tuple[Edge, ...]) -> list[int]:
    degrees = [0] * n
    for u, v in edges:
        degrees[u] += 1
        degrees[v] += 1
    return degrees


def configuration_edges(n: int, edges: tuple[Edge, ...], rng: random.Random) -> tuple[Edge, ...]:
    """Switch edges in a simple graph, preserving every labelled node degree exactly.

    This is a degree-preserving switch-chain sample, not a claim of an exact
    uniform draw from all simple graphs with this degree sequence.
    """
    if len(edges) < 2:
        return edges
    current = [tuple(sorted(edge)) for edge in edges]
    present = set(current)
    if len(present) != len(current) or any(u == v or not 0 <= u < n or not 0 <= v < n for u, v in current):
        raise ValueError("configuration model requires a simple graph on nodes 0..n-1")

    # Attempts, rather than successful switches, keep work bounded on dense or
    # highly constrained degree sequences. Rejections leave degrees unchanged.
    for _ in range(10 * len(current)):
        i, j = rng.sample(range(len(current)), 2)
        a, b = current[i]
        c, d = current[j]
        if len({a, b, c, d}) != 4:
            continue
        if rng.randrange(2):
            replacements = (tuple(sorted((a, c))), tuple(sorted((b, d))))
        else:
            replacements = (tuple(sorted((a, d))), tuple(sorted((b, c))))
        if replacements[0] in present or replacements[1] in present:
            continue
        present.remove(current[i])
        present.remove(current[j])
        current[i], current[j] = replacements
        present.update(replacements)
    return tuple(current)


def chung_lu_edges(n: int, edges: tuple[Edge, ...], rng: random.Random) -> tuple[Edge, ...]:
    """Independent-edge simple Chung-Lu graph fitted to observed degrees.

    p_uv = min(d_u*d_v / sum(d), 1). Its degrees and edge count can vary.
    In a simple graph these expected degrees only approximate the input ones.
    """
    degrees = _degrees(n, edges)
    total_degree = sum(degrees)
    if total_degree == 0:
        return ()
    return tuple(
        (u, v)
        for u in range(n)
        for v in range(u + 1, n)
        if rng.random() < min(degrees[u] * degrees[v] / total_degree, 1.0)
    )


NULL_MODELS: dict[str, Callable[[int, tuple[Edge, ...], random.Random], tuple[Edge, ...]]] = {
    "gnm": _gnm_from_reference,
    "configuration": configuration_edges,
    "chung_lu": chung_lu_edges,
}
