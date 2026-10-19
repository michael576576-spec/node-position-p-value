"""HTC and SigEP p-value calculations from the supplied papers.

Both methods use a simple undirected graph and the G(N, M) null model. HTC
uses the paper's approximate degree-neighbor-sum distribution and a
Bonferroni decision rule. SigEP ports the p-value calculation and BH decision
semantics from the authors' MIT-licensed implementation at
https://github.com/yanganggood/sigep, using log-combinations for numerical
stability and adapting it to the project's explicit node set (including
isolates).
"""

from __future__ import annotations

from math import floor, sqrt

from experiment_support.network_core import Network


def _numeric_tools():
    try:
        import numpy as np
        from scipy.special import gammaln, logsumexp
    except ImportError as exc:
        raise RuntimeError(
            "HTC/SigEP require NumPy and SciPy: python -m pip install numpy scipy"
        ) from exc
    return np, gammaln, logsumexp


def _log_choose_array(n: int | object, k: object, np, gammaln):
    n_array, k_array = np.broadcast_arrays(np.asarray(n, dtype=float), np.asarray(k, dtype=float))
    result = np.full(n_array.shape, -np.inf, dtype=float)
    valid = (n_array >= 0) & (k_array >= 0) & (k_array <= n_array)
    result[valid] = (
        gammaln(n_array[valid] + 1)
        - gammaln(k_array[valid] + 1)
        - gammaln(n_array[valid] - k_array[valid] + 1)
    )
    return result


def _network_statistics(network: Network) -> tuple[list[int], list[int], list[int], list[int]]:
    """Return degree, neighbor-degree-sum, triangle, and wedge counts per node."""
    try:
        import igraph as ig
    except ImportError as exc:
        raise RuntimeError("HTC/SigEP require igraph: python -m pip install igraph") from exc

    graph = ig.Graph(n=network.n, edges=network.edges, directed=False)
    degrees = [int(value) for value in graph.degree()]
    adjacency = graph.get_adjlist()
    neighbor_degree_sums = [sum(degrees[v] for v in neighbors) for neighbors in adjacency]
    clustering = graph.transitivity_local_undirected(mode="zero")
    triangles: list[int] = []
    wedges: list[int] = []
    for degree, coefficient in zip(degrees, clustering):
        wedge_count = degree * (degree - 1) // 2
        wedges.append(wedge_count)
        triangles.append(int(round(float(coefficient) * wedge_count)))
    return degrees, neighbor_degree_sums, triangles, wedges


def _htc_mass(n: int, m: int, score: int, log_total_graphs: float, np, gammaln, logsumexp) -> float:
    """Approximate probability mass from HTC's equation (8)."""
    if score == 0:
        log_count = float(_log_choose_array((n - 1) * (n - 2) // 2, m, np, gammaln))
        return float(np.exp(log_count - log_total_graphs)) if np.isfinite(log_count) else 0.0
    if score == 1:
        log_count = (
            np.log(n - 1)
            + _log_choose_array((n - 2) * (n - 3) // 2, m - 1, np, gammaln)
        )
        return float(np.exp(log_count - log_total_graphs)) if np.isfinite(log_count) else 0.0

    # Include focal degree 1 as well as degree >= 2. For score >= 2, the
    # degree-1 case occurs when the focal node has one neighbor whose own
    # degree is ``score``; omitting it undercounts the HTC mass.
    degree = np.arange(1, min(n - 1, score, m) + 1, dtype=np.int64)
    neighbor_pairs = degree * (degree - 1) // 2 + degree * (n - 1 - degree)
    untouched_pairs = (n - 1 - degree) * (n - 2 - degree) // 2
    terms = (
        _log_choose_array(n - 1, degree, np, gammaln)
        + _log_choose_array(neighbor_pairs, score - degree, np, gammaln)
        + _log_choose_array(untouched_pairs, m - score, np, gammaln)
    )
    log_count = float(logsumexp(terms)) if terms.size else -np.inf
    return float(np.exp(log_count - log_total_graphs)) if np.isfinite(log_count) else 0.0


def htc_p_values(network: Network) -> list[float]:
    """Return HTC analytical p-value approximations for all nodes.

    Distinct observed neighbor-degree sums share the same cumulative
    distribution calculation. This evaluates the paper's approximate
    distribution (equations 3-8), rather than simulating null graphs.
    """
    n, m = network.n, network.m
    np, gammaln, logsumexp = _numeric_tools()
    total_pairs = n * (n - 1) // 2
    log_total = float(_log_choose_array(total_pairs, m, np, gammaln))
    if not np.isfinite(log_total):
        raise ValueError("invalid G(n, m) parameters for HTC")
    _, scores, _, _ = _network_statistics(network)
    max_score = max(scores, default=0)

    # The approximate masses are shared by all vertices in this graph.
    lower_tail: list[float] = [0.0] * (max_score + 1)
    cumulative = 0.0
    for score in range(max_score + 1):
        cumulative += _htc_mass(n, m, score, log_total, np, gammaln, logsumexp)
        lower_tail[score] = cumulative
    return [
        min(1.0, max(0.0, 1.0 - lower_tail[score - 1])) if score else 1.0
        for score in scores
    ]


def sigep_p_values(network: Network) -> list[float]:
    """Return the official SigEP upper bounds, including its N multiplier.

    For clustering coefficient zero, Equation (1)'s degree tail is used;
    otherwise Equation (10)'s joint degree/clustering upper bound is used.
    The authors' implementation multiplies both cases by N before BH.
    """
    n, m = network.n, network.m
    np, gammaln, logsumexp = _numeric_tools()
    total_pairs = n * (n - 1) // 2
    log_total = float(_log_choose_array(total_pairs, m, np, gammaln))
    if not np.isfinite(log_total):
        raise ValueError("invalid G(n, m) parameters for SigEP")

    degrees, _, triangles, wedges = _network_statistics(network)
    degree_tail_cache: dict[int, float] = {}
    p_values = []
    for degree, triangle_count, wedge_count in zip(degrees, triangles, wedges):
        if degree < 2 or wedge_count == 0 or triangle_count == 0:
            # With observed clustering coefficient zero, the joint event
            # reduces to the degree upper tail under G(n, m) (SigEP Eq. 1).
            if degree not in degree_tail_cache:
                d_values = np.arange(degree, min(n - 1, m) + 1, dtype=np.int64)
                log_terms = (
                    _log_choose_array(n - 1, d_values, np, gammaln)
                    + _log_choose_array(total_pairs - (n - 1), m - d_values, np, gammaln)
                    - log_total
                )
                log_tail = float(logsumexp(log_terms)) if log_terms.size else -np.inf
                degree_tail_cache[degree] = (
                    float(np.exp(log_tail)) if np.isfinite(log_tail) else 0.0
                )
            # The official implementation multiplies every node's upper-bound
            # p-value by N before applying BH, including the c = 0 case.
            p_values.append(n * degree_tail_cache[degree])
            continue

        coefficient_numerator = triangle_count
        coefficient_denominator = wedge_count
        discriminant = (2.0 - triangle_count / wedge_count) ** 2 + (
            8.0 * (triangle_count / wedge_count) * m
        )
        coefficient = triangle_count / wedge_count
        b = min(
            n - 1,
            m,
            floor((coefficient - 2.0 + sqrt(discriminant)) / (2.0 * coefficient)),
        )
        d_values = np.arange(degree, b + 1, dtype=np.int64)
        pairs_among_neighbors = d_values * (d_values - 1) // 2
        required_local_edges = (
            coefficient_numerator * pairs_among_neighbors + coefficient_denominator - 1
        ) // coefficient_denominator
        available_non_star = total_pairs - (n - 1) - required_local_edges
        remaining_edges = m - d_values - required_local_edges
        log_terms = (
            _log_choose_array(n - 1, d_values, np, gammaln)
            + _log_choose_array(pairs_among_neighbors, required_local_edges, np, gammaln)
            + _log_choose_array(available_non_star, remaining_edges, np, gammaln)
            - log_total
        )
        log_probability = (
            float(np.log(n) + logsumexp(log_terms)) if log_terms.size else -np.inf
        )
        p_values.append(float(np.exp(log_probability)) if np.isfinite(log_probability) else 0.0)
    return p_values


def benjamini_hochberg(p_values: list[float], alpha: float) -> list[bool]:
    """Return SigEP's BH discoveries at the requested FDR level."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    ordered = sorted(enumerate(p_values), key=lambda item: (item[1], item[0]))
    largest = 0
    count = len(ordered)
    for rank, (_, p_value) in enumerate(ordered, start=1):
        # Match the strict inequality used by the authors' reference code.
        if p_value < alpha * rank / count:
            largest = rank
    reported = [False] * count
    for node, _ in ordered[:largest]:
        reported[node] = True
    return reported


def benjamini_hochberg_adjusted(p_values: list[float]) -> list[float]:
    """Return BH adjusted p-values (q-values), in the original node order."""
    count = len(p_values)
    if count == 0:
        return []
    ordered = sorted(enumerate(p_values), key=lambda item: (item[1], item[0]))
    adjusted = [1.0] * count
    running_minimum = 1.0
    for rank in range(count, 0, -1):
        node, p_value = ordered[rank - 1]
        running_minimum = min(running_minimum, p_value * count / rank)
        adjusted[node] = min(1.0, running_minimum)
    return adjusted


def bonferroni(p_values: list[float], alpha: float) -> list[bool]:
    """Return Bonferroni discoveries at family-wise level alpha."""
    threshold = alpha / len(p_values) if p_values else 0.0
    return [p_value <= threshold for p_value in p_values]
