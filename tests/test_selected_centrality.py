"""Check the six selected metrics on a small graph with an isolate."""

import unittest

import numpy as np

from experiment_support.centrality import METRICS, compute_multiple_scores, compute_multiple_scores_igraph


class SelectedCentralityTests(unittest.TestCase):
    def test_lac_closeness_and_geometric_gain(self):
        edges = ((0, 1), (1, 2), (2, 0), (2, 3))
        scores = compute_multiple_scores_igraph(5, edges, tuple(METRICS))
        self.assertEqual(tuple(scores), tuple(METRICS))
        np.testing.assert_allclose(scores["lac"], (1, 1, 2 / 3, 0, 0))
        np.testing.assert_allclose(scores["closeness"], (.5625, .5625, .75, .45, 0))
        self.assertEqual(scores["eigenvector"][4], 0.0)
        self.assertEqual(scores["gpg"][4], 0.0)

        adjacency = np.zeros((5, 5))
        for u, v in edges:
            adjacency[u, v] = adjacency[v, u] = 1
        eigenvalue = np.linalg.eigvalsh(adjacency)[-1]
        expected = adjacency @ np.linalg.solve(
            np.eye(5) - (0.5 / eigenvalue) * adjacency, np.ones(5)
        )
        np.testing.assert_allclose(scores["gpg"], expected, rtol=1e-10, atol=1e-10)

    def test_python_and_igraph_backends_agree_on_connected_graph(self):
        edges = ((0, 1), (1, 2), (2, 0), (2, 3))
        requested = tuple(METRICS)
        python_scores = compute_multiple_scores(4, edges, requested)
        igraph_scores = compute_multiple_scores_igraph(4, edges, requested)
        for metric in requested:
            with self.subTest(metric=metric):
                np.testing.assert_allclose(
                    python_scores[metric], igraph_scores[metric], rtol=1e-7, atol=1e-8
                )


if __name__ == "__main__":
    unittest.main()
