from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import networkx as nx
from loguru import logger
from pyvis.network import Network

from wip_analysis.hwm.product_level.aggregate_paths_stats import (
    get_paths_with_stats_for_products,
)


class PlotPathsNoDataError(ValueError):
    """Raised when no path clusters are available for plotting."""


class TransitionGraph:
    def __init__(self, primary_paths: list[tuple[list[str], int]]):
        """
        Args:
            primary_paths: list of (primary_path, lot_cnt) pairs
        """

        def name_to_color(category):
            import colorsys
            import hashlib

            # generate determined color via md5
            h = int(hashlib.md5(str(category).encode()).hexdigest(), 16)
            hue = (h % 360) / 360.0
            r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.65)
            return f"#{int(r * 255):02X}{int(g * 255):02X}{int(b * 255):02X}"

        def generate_colormap(categories):
            return {cat: name_to_color(cat) for cat in categories}

        # construct the graph
        edges = []

        for i, (primary_path, lot_cnt) in enumerate(primary_paths):
            for j in range(1, len(primary_path)):
                edge = (primary_path[j - 1], primary_path[j], {"cnt": lot_cnt, "cluster": i})
                edges.append(edge)

        G = nx.MultiDiGraph(edges)

        self.G = G
        self.primary_paths = primary_paths
        self.color_map = generate_colormap(range(len(primary_paths)))

    def plot(self, path: str | Path, distinguish: bool = False, align: bool = False) -> None:
        """
        Args:
            path: path to save the html
            distinguish: distinguish the difference in lot cnts or not
            align: align the main path horizontally
        """
        Path(path).write_text(
            self.render_html(align=align, distinguish=distinguish),
            encoding="utf-8",
        )

    def render_html(self, align: bool = False, distinguish: bool = False) -> str:
        net = Network(height="800px", width="100%", notebook=True, directed=True)
        net.from_nx(self.G)

        for edge in net.edges:
            edge["title"] = edge.get("cnt")
            edge["color"] = self.color_map[edge.get("cluster")]

        self._adjust_edge_width(net, distinguish=distinguish)
        self._highlight_start_end_nodes(net)

        if align:
            self._align(net)

        return net.generate_html(notebook=False)

    def _adjust_edge_width(self, net, min_width=0.1, max_width=10, distinguish=False):
        """
        Args:
            net (Network): Network
            min_width (int): minimum width of all edges (no effect when distinguish=True)
            max_width (int): maxmimum width of all edges (no effect when distinguish=True)
            distinguish (bool): distinguish the difference in lot cnts or not

        Returns:
            pyvis.network
        """
        weights = [edge.get("cnt") for edge in net.edges]
        min_weight = min(weights)
        max_weight = max(weights)

        for edge in net.edges:
            weight = edge.get("cnt")
            if not distinguish:
                # linear proj
                width = ((weight - min_weight + 1) / (max_weight - min_weight + 1)) * (
                    max_width - min_width
                ) + min_width
            else:
                width = weight
            edge["width"] = width

    def _align(self, net: Network, x_gap=200, y_fixed=0):
        """
        Align nodes according to the first primary_path. Other nodes remain free for physics layout.

        Args:
            net (Network): a pyvis Network object (from self.G)
            x_gap (int): horizontal spacing between nodes in the main path
            y_fixed (int): y-coordinate for the main path
        """
        if not self.primary_paths:
            return

        main_path, _ = self.primary_paths[0]

        # assign coordinate for nodes on main_path
        node_pos = {node: (i * x_gap, y_fixed) for i, node in enumerate(main_path)}

        for node in net.nodes:
            node_id = node["id"]
            if node_id in node_pos:
                x, y = node_pos[node_id]
                node["x"] = x
                node["y"] = y
                node["fixed"] = {"x": True, "y": True}

    def _highlight_start_end_nodes(self, net: Network, border_width=5):
        """
        Highlight start and end nodes of each path by increasing border width and
        setting border color to match the path color.

        Args:
            net (Network): a pyvis Network object (from self.G)
            border_width (int): border width for start/end nodes (default: 5)
        """
        # Map each node to the path indices it belongs to (as start or end)
        # If a node belongs to multiple paths, we'll use the first path's color
        node_to_path_idx = {}

        for path_idx, (primary_path, _) in enumerate(self.primary_paths):
            if len(primary_path) > 0:
                start_node = primary_path[0]
                end_node = primary_path[-1]

                # Record path index for start node (use first occurrence if multiple)
                if start_node not in node_to_path_idx:
                    node_to_path_idx[start_node] = path_idx

                # Record path index for end node (use first occurrence if multiple)
                if end_node not in node_to_path_idx:
                    node_to_path_idx[end_node] = path_idx

        # Set border width and color for start and end nodes
        for node in net.nodes:
            node_id = node["id"]
            if node_id in node_to_path_idx:
                path_idx = node_to_path_idx[node_id]
                node["borderWidth"] = border_width
                node["borderWidthSelected"] = border_width
                node["color"] = {"border": self.color_map[path_idx]}


def _primary_paths_from_clusters(
    clusters: Sequence[Mapping[str, Any]],
) -> list[tuple[list[str], int]]:
    primary_paths: list[tuple[list[str], int]] = []
    for cluster in clusters:
        path = cluster.get("path", [])
        if not path:
            continue
        primary_paths.append((list(path), int(cluster.get("lot_count", 0))))

    primary_paths.sort(key=lambda item: item[1], reverse=True)
    return primary_paths


def build_paths_plot_html(
    product_id: str,
    snapshot_start: datetime,
    snapshot_end: datetime,
    *,
    align: bool = True,
    rules: Mapping[str, Any] | None = None,
) -> str:
    rules_dict = dict(rules or {})
    if not rules_dict.get("clustering_method"):
        raise ValueError("build_paths_plot_html requires clustering_method in rules")

    path_stats = get_paths_with_stats_for_products(
        snapshot_start=snapshot_start,
        snapshot_end=snapshot_end,
        product_ids=[product_id],
        rules=rules_dict,
    )

    primary_paths = _primary_paths_from_clusters(path_stats.get(product_id, []))
    if not primary_paths:
        raise PlotPathsNoDataError(
            f"No path clusters found for product {product_id} "
            f"between {snapshot_start} and {snapshot_end}"
        )

    graph = TransitionGraph(primary_paths)
    return graph.render_html(align=align)


def plot_paths(
    pid: str,
    snapshot_start: datetime,
    snapshot_end: datetime,
    align: bool = True,
    sol_timeout: float = 300.0,
    **cluster_params,
):
    rules = dict(cluster_params)
    rules["sol_timeout"] = sol_timeout
    if not rules.get("clustering_method"):
        raise ValueError("plot_paths requires clustering_method in cluster_params")

    request_pids = None if pid == "ALL" else [pid]
    path_stats = get_paths_with_stats_for_products(
        snapshot_start=snapshot_start,
        snapshot_end=snapshot_end,
        product_ids=request_pids,
        rules=rules,
    )

    target_pid = pid if pid != "ALL" else next(iter(path_stats), None)
    if target_pid is None:
        logger.warning("plot_paths found no products for pid={}", pid)
        return

    primary_paths = _primary_paths_from_clusters(path_stats.get(target_pid, []))
    if not primary_paths:
        logger.warning("plot_paths found no paths for pid={}", target_pid)
        return

    graph = TransitionGraph(primary_paths)
    output_dir = Path("outputs/primary_paths")
    output_dir.mkdir(parents=True, exist_ok=True)
    graph.plot(output_dir / f"{target_pid.replace('/', '_')}.html", align=align)


if __name__ == "__main__":
    pid = "UcXDW7FuxUqS8fDrT6fWTQ=="
    plot_paths(
        pid,
        snapshot_start=datetime(2024, 1, 1, 0, 0, 0),
        snapshot_end=datetime(2024, 4, 1, 23, 59, 59),
        clustering_method="hdbscan",
        # distance_metric="jaccard",
    )
