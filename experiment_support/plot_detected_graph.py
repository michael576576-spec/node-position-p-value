"""Plot a planted positive graph and highlight a detected planted hub."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from experiment_support.experiment import make_positive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--graph-index", type=int, default=4)
    parser.add_argument("--model", default="chung_lu")
    parser.add_argument("--metric", default="betweenness")
    parser.add_argument("--budget", type=int, default=1000)
    args = parser.parse_args()

    settings = json.loads((args.results / "settings.json").read_text(encoding="utf-8"))
    name = f"positive_{args.graph_index:02d}"
    n, target_m, hub_count, ratio = settings["positive_specs"][args.graph_index - 1]
    network = make_positive(
        name, n, target_m, hub_count, ratio,
        random.Random(f"{settings['seed']}:{name}:graph"),
    )
    checkpoint = json.loads(
        (args.results / "checkpoints" / args.model / f"{name}.json").read_text(encoding="utf-8")
    )
    if (checkpoint["n"] != network.n or checkpoint["m"] != network.m
            or set(checkpoint["hubs"]) != network.hubs):
        raise ValueError("reconstructed graph does not match the experiment checkpoint")
    reports = checkpoint["snapshots"][str(args.budget)]["reports"][args.metric]
    true_positives = [report for report in reports if report["node"] in network.hubs]
    if len(true_positives) != 1:
        raise ValueError(f"expected exactly one detected planted hub, got {true_positives}")
    selected = true_positives[0]["node"]
    adjusted_p = true_positives[0]["bonferroni_p"]

    graph = nx.Graph()
    graph.add_nodes_from(range(network.n))
    graph.add_edges_from(network.edges)
    positions = nx.spring_layout(graph, seed=42, iterations=350)
    neighbors = set(graph.neighbors(selected))
    incident_edges = [edge for edge in graph.edges if selected in edge]
    other_edges = [edge for edge in graph.edges if selected not in edge]

    fig, ax = plt.subplots(figsize=(11.5, 9.5), facecolor="white")
    nx.draw_networkx_edges(
        graph, positions, edgelist=other_edges, ax=ax,
        edge_color="#9aa8b8", alpha=0.35, width=0.85,
    )
    nx.draw_networkx_edges(
        graph, positions, edgelist=incident_edges, ax=ax,
        edge_color="#e85d48", alpha=0.68, width=1.35,
    )
    nx.draw_networkx_nodes(
        graph, positions, nodelist=sorted(set(graph) - neighbors - {selected}), ax=ax,
        node_color="#d6dee8", edgecolors="white", linewidths=0.6, node_size=150,
    )
    nx.draw_networkx_nodes(
        graph, positions, nodelist=sorted(neighbors), ax=ax,
        node_color="#61a5d8", edgecolors="white", linewidths=0.7, node_size=175,
    )
    nx.draw_networkx_nodes(
        graph, positions, nodelist=[selected], ax=ax,
        node_color="#c51b35", edgecolors="#641126", linewidths=2.1, node_size=1000,
    )
    nx.draw_networkx_labels(
        graph, positions, labels={selected: str(selected)}, ax=ax,
        font_size=15, font_weight="bold", font_color="white",
    )
    ax.set_title(
        f"{name}: detected planted hub under Chung–Lu + BC",
        loc="left", fontsize=17, fontweight="bold", color="#172235", pad=18,
    )
    ax.text(
        0.01, 0.99,
        f"N = {network.n}   M = {network.m}   Detected node = {selected}   "
        f"Degree = {graph.degree(selected)}   Bonferroni p = {adjusted_p:.4f}",
        transform=ax.transAxes, va="top", fontsize=11, color="#334155",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.55"},
    )
    ax.scatter([], [], s=230, c="#c51b35", edgecolors="#641126", label="Detected planted hub")
    ax.scatter([], [], s=100, c="#61a5d8", edgecolors="white", label="Its neighbors")
    ax.scatter([], [], s=100, c="#d6dee8", edgecolors="white", label="Other nodes")
    ax.legend(loc="lower left", frameon=False, fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()

    stem = f"{name}_{args.model}_{args.metric}_{args.budget}"
    for suffix in ("png", "svg"):
        path = args.results / f"{stem}.{suffix}"
        fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
