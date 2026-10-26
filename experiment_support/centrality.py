"""Interchangeable node-centrality scores for undirected, unweighted graphs."""

from __future__ import annotations

from collections.abc import Callable
import math
import warnings


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


def betweenness(n: int, edges: tuple[Edge, ...]) -> list[float]:
    nx, graph = _networkx_graph(n, edges, "betweenness")
    values = nx.betweenness_centrality(graph, normalized=True, weight=None)
    return [float(values[node]) for node in range(n)]


def pagerank(n: int, edges: tuple[Edge, ...]) -> list[float]:
    nx, graph = _networkx_graph(n, edges, "pagerank")
    values = nx.pagerank(graph, alpha=0.85, tol=1e-12, max_iter=1000)
    return [float(values[node]) for node in range(n)]


def closeness(n: int, edges: tuple[Edge, ...]) -> list[float]:
    nx, graph = _networkx_graph(n, edges, "closeness")
    values = nx.closeness_centrality(graph, wf_improved=True)
    return [float(values[node]) for node in range(n)]


def _geometric_potential_gain(adjacency, eigenvalue: float) -> list[float]:
    """A(I - 0.5 A/lambda_max)^-1 1, evaluated by a convergent series."""
    import numpy as np

    n = adjacency.shape[0]
    if eigenvalue <= 0 or not math.isfinite(eigenvalue):
        return [0.0] * n
    delta = 0.5 / eigenvalue
    vector = np.ones(n, dtype=float)
    for _ in range(80):
        updated = 1.0 + delta * (adjacency @ vector)
        if np.max(np.abs(updated - vector)) <= 1e-12 * max(1.0, float(np.max(updated))):
            vector = updated
            break
        vector = updated
    return [float(value) for value in adjacency @ vector]


def gpg(n: int, edges: tuple[Edge, ...]) -> list[float]:
    _, graph = _networkx_graph(n, edges, "gpg")
    if not edges:
        return [0.0] * n
    import scipy.sparse.linalg as spla
    import networkx as nx

    adjacency = nx.to_scipy_sparse_array(graph, nodelist=range(n), dtype=float, format="csr")
    eigenvalue = float(spla.eigsh(adjacency, k=1, which="LA", return_eigenvectors=False)[0])
    return _geometric_potential_gain(adjacency, eigenvalue)


METRICS: dict[str, MetricFunction] = {
    "degree": degree,
    "pagerank": pagerank,
    "betweenness": betweenness,
    "closeness": closeness,
    "gpg": gpg,
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
    """Score one graph with several metrics using the NetworkX implementations."""
    unknown = set(metrics) - METRICS.keys()
    if unknown:
        raise ValueError(f"unknown metrics: {', '.join(sorted(unknown))}")
    scores: dict[str, list[float]] = {}
    for metric in metrics:
        scores[metric] = METRICS[metric](n, edges)
    return scores


def compute_multiple_scores_igraph(
    n: int, edges: tuple[Edge, ...], metrics: tuple[str, ...]
) -> dict[str, list[float]]:
    """Compute the five selected unweighted metrics with igraph and SciPy."""
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
    if "pagerank" in metrics:
        scores["pagerank"] = [float(value) for value in graph.pagerank(directed=False, damping=0.85)]
    eigenvalue = 0.0
    if "gpg" in metrics:
        if graph.ecount():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Some eigenvector centralities are nearly zero", category=RuntimeWarning)
                eigenvalue = graph.eigenvector_centrality(return_eigenvalue=True)[1]
    if "betweenness" in metrics:
        scale = 2 / ((n - 1) * (n - 2)) if n > 2 else 0.0
        scores["betweenness"] = [value * scale for value in graph.betweenness(directed=False)]
    if "closeness" in metrics:
        components = graph.connected_components()
        membership = components.membership
        sizes = components.sizes()
        values = graph.closeness(normalized=True)
        scores["closeness"] = [float(values[node] * (sizes[membership[node]] - 1) / (n - 1))
                               if n > 1 and sizes[membership[node]] > 1 and math.isfinite(values[node]) else 0.0
                               for node in range(n)]
    if "gpg" in metrics:
        scores["gpg"] = _geometric_potential_gain(graph.get_adjacency_sparse(), float(eigenvalue))
    return scores
