"""Interchangeable node-centrality scores for undirected, unweighted graphs."""

from __future__ import annotations

from collections.abc import Callable


Edge = tuple[int, int]
MetricFunction = Callable[[int, tuple[Edge, ...]], list[float]]


def degree(n: int, edges: tuple[Edge, ...]) -> list[float]:
    scores = [0.0] * n
    for u, v in edges:
        scores[u] += 1.0
        scores[v] += 1.0
    return scores


def _networkx_graph(n: int, edges: tuple[Edge, ...], metric: str):
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError(
            f"Metric '{metric}' requires NetworkX. Install it with: python -m pip install networkx"
        ) from exc

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(edges)
    return nx, graph


def harmonic(n: int, edges: tuple[Edge, ...]) -> list[float]:
    nx, graph = _networkx_graph(n, edges, "harmonic")
    values = nx.harmonic_centrality(graph)
    return [float(values[node]) for node in range(n)]


def betweenness(n: int, edges: tuple[Edge, ...]) -> list[float]:
    nx, graph = _networkx_graph(n, edges, "betweenness")
    values = nx.betweenness_centrality(graph, normalized=True, weight=None)
    return [float(values[node]) for node in range(n)]


def pagerank(n: int, edges: tuple[Edge, ...]) -> list[float]:
    nx, graph = _networkx_graph(n, edges, "pagerank")
    values = nx.pagerank(graph, alpha=0.85, weight=None)
    return [float(values[node]) for node in range(n)]


METRICS: dict[str, MetricFunction] = {
    "degree": degree,
    "harmonic": harmonic,
    "betweenness": betweenness,
    "pagerank": pagerank,
}


def compute_scores(n: int, edges: tuple[Edge, ...], metric: str) -> list[float]:
    try:
        scorer = METRICS[metric]
    except KeyError as exc:
        raise ValueError(f"unknown metric '{metric}'; choose from {', '.join(METRICS)}") from exc
    return scorer(n, edges)
