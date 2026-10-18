"""Plot a real-network node reported only by betweenness under degree-aware nulls."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from experiment_support.real_network_experiment import load_edgelist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--node", default="37")
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--remove-node", action="store_true")
    parser.add_argument("--hide-node-label", action="store_true")
    args = parser.parse_args()

    labels, edges, _ = load_edgelist(args.source)
    graph = nx.Graph()
    graph.add_nodes_from(labels)
    graph.add_edges_from((labels[u], labels[v]) for u, v in edges)
    if args.node not in graph:
        raise ValueError(f"node {args.node} not present in {args.source}")

    with (args.results / "significant_nodes.csv").open(encoding="utf-8-sig", newline="") as file:
        reports = list(csv.DictReader(file))
    relevant = [
        row for row in reports
        if row["network"] == args.source.stem and row["node"] == args.node
        and int(row["budget"]) == args.budget
        and row["model"] in {"configuration", "chung_lu"}
    ]
    found = {(row["model"], row["metric"]): row for row in relevant}
    if set(found) != {("configuration", "BC"), ("chung_lu", "BC")}:
        raise ValueError(f"node is not BC-only under both null models: {set(found)}")

    bc = nx.betweenness_centrality(graph, normalized=True)
    degree_rank = 1 + sum(graph.degree(node) > graph.degree(args.node) for node in graph)
    bc_rank = 1 + sum(score > bc[args.node] for score in bc.values())
    communities = sorted(nx.community.greedy_modularity_communities(graph), key=len, reverse=True)
    community_of = {node: index for index, group in enumerate(communities) for node in group}
    positions = nx.spring_layout(graph, seed=41, iterations=500)

    incident = [edge for edge in graph.edges if args.node in edge]
    others = [edge for edge in graph.edges if args.node not in edge]
    remaining = graph.copy()
    remaining.remove_node(args.node)
    palette = ["#85b6d2", "#9ccfa6", "#e7c888", "#c6b7da", "#b7ced1"]
    colors = [palette[community_of[node] % len(palette)] for node in graph if node != args.node]

    fig, ax = plt.subplots(figsize=(10, 7.5), facecolor="white")
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.13, top=0.74)
    nx.draw_networkx_edges(
        graph, positions, edgelist=others, ax=ax,
        edge_color="#7d8b99", alpha=0.32, width=0.75,
    )
    if not args.remove_node:
        nx.draw_networkx_edges(
            graph, positions, edgelist=incident, ax=ax,
            edge_color="#dc4255", alpha=0.9, width=1.7,
        )
    nx.draw_networkx_nodes(
        graph, positions, nodelist=[node for node in graph if node != args.node], ax=ax,
        node_color=colors, edgecolors="white", linewidths=0.7, node_size=155,
    )
    if not args.remove_node:
        nx.draw_networkx_nodes(
            graph, positions, nodelist=[args.node], ax=ax,
            node_color="#c51b35", edgecolors="#671124", linewidths=1.6, node_size=400,
        )
        if not args.hide_node_label:
            nx.draw_networkx_labels(
                graph, positions, labels={args.node: args.node}, ax=ax,
                font_size=15, font_weight="bold", font_color="white",
            )
    p_config = float(found[("configuration", "BC")]["bonferroni_p"])
    p_chung = float(found[("chung_lu", "BC")]["bonferroni_p"])
    if args.remove_node:
        title = f"{args.source.stem} network | node {args.node} removed"
        annotation = (
            f"Remaining N = {remaining.number_of_nodes()}   M = {remaining.number_of_edges()}\n"
            f"Connected components = {nx.number_connected_components(remaining)}; "
            f"original node degree = {graph.degree(args.node)}\n"
            "Layout and community colors match the original plot"
        )
    else:
        title = (
            f"{args.source.stem} network | BC-only reported node"
            if args.hide_node_label else
            f"{args.source.stem} network | BC-only reported node {args.node}"
        )
        annotation = (
            f"N = {graph.number_of_nodes()}   M = {graph.number_of_edges()}   "
            f"degree = {graph.degree(args.node)} (rank {degree_rank})   BC rank = {bc_rank}\n"
            f"Adjusted p: configuration = {p_config:.4f}, Chung–Lu = {p_chung:.4f}\n"
            "DC, HC and PR did not report the highlighted node"
        )
    fig.text(0.052, 0.96, title, ha="left", va="top", fontsize=15.5,
             fontweight="bold", color="#172235")
    fig.text(
        0.052, 0.875, annotation, ha="left", va="top", fontsize=9.5,
        color="#334155",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.55"},
    )
    if not args.remove_node:
        ax.scatter([], [], s=120, c="#c51b35", edgecolors="#671124", label="BC-only reported node")
    ax.scatter([], [], s=120, c="#85b6d2", edgecolors="white", label="Other nodes (community colors)")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower left",
               bbox_to_anchor=(0.045, 0.015), frameon=False, fontsize=9)
    ax.set_axis_off()
    xs = [position[0] for position in positions.values()]
    ys = [position[1] for position in positions.values()]
    ax.set_xlim(min(xs) - 0.12, max(xs) + 0.12)
    ax.set_ylim(min(ys) - 0.12, max(ys) + 0.12)

    if args.remove_node:
        stem = f"{args.source.stem}_without_node_{args.node}_{args.budget}"
    elif args.hide_node_label:
        stem = f"{args.source.stem}_bc_only_node_{args.node}_{args.budget}_no_label"
    else:
        stem = f"{args.source.stem}_bc_only_node_{args.node}_{args.budget}"
    for suffix in ("png", "svg"):
        path = args.results / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
