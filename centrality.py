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


def compute_multiple_scores(
    n: int, edges: tuple[Edge, ...], metrics: tuple[str, ...]
) -> dict[str, list[float]]:
    """Score one graph with several metrics while constructing NetworkX once."""
    unknown = set(metrics) - METRICS.keys()
    if unknown:
        raise ValueError(f"unknown metrics: {', '.join(sorted(unknown))}")
    scores: dict[str, list[float]] = {}
    if "degree" in metrics:
        scores["degree"] = degree(n, edges)
    if any(metric != "degree" for metric in metrics):
        nx, graph = _networkx_graph(n, edges, "multiple")
        if "harmonic" in metrics:
            values = nx.harmonic_centrality(graph)
            scores["harmonic"] = [float(values[node]) for node in range(n)]
        if "betweenness" in metrics:
            values = nx.betweenness_centrality(graph, normalized=True, weight=None)
            scores["betweenness"] = [float(values[node]) for node in range(n)]
        if "pagerank" in metrics:
            values = nx.pagerank(graph, alpha=0.85, weight=None)
            scores["pagerank"] = [float(values[node]) for node in range(n)]
    return scores


def compute_multiple_scores_igraph(
    n: int, edges: tuple[Edge, ...], metrics: tuple[str, ...]
) -> dict[str, list[float]]:
    """Compute the same four unweighted metrics with igraph's compiled routines."""
    try:
        import igraph as ig
    except ImportError as exc:
        raise RuntimeError("This sweep requires igraph: python -m pip install igraph") from exc
    unknown = set(metrics) - METRICS.keys()
    if unknown:
        raise ValueError(f"unknown metrics: {', '.join(sorted(unknown))}")
    graph = ig.Graph(n=n, edges=edges, directed=False)
    scores: dict[str, list[float]] = {}
    if "degree" in metrics:
        scores["degree"] = [float(value) for value in graph.degree()]
    if "harmonic" in metrics:
        scores["harmonic"] = list(graph.harmonic_centrality(normalized=False))
    if "betweenness" in metrics:
        scale = 2 / ((n - 1) * (n - 2)) if n > 2 else 0.0
        scores["betweenness"] = [value * scale for value in graph.betweenness(directed=False)]
    if "pagerank" in metrics:
        scores["pagerank"] = list(graph.pagerank(damping=0.85, directed=False))
    return scores
