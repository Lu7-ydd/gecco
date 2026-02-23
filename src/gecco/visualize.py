from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

from .tree import ModuleNode


def _iter_nodes_with_depth(root: ModuleNode):
    stack: list[tuple[ModuleNode, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        yield node, depth
        for child in reversed(node.children):
            stack.append((child, depth + 1))


def _layout_tree(root: ModuleNode) -> tuple[dict[str, tuple[float, float]], dict[str, int], list[tuple[str, str]]]:
    pos: dict[str, tuple[float, float]] = {}
    depth_map: dict[str, int] = {}
    edges: list[tuple[str, str]] = []
    leaves: list[ModuleNode] = []

    def collect(node: ModuleNode, depth: int) -> None:
        depth_map[node.node_id] = depth
        if node.is_leaf():
            leaves.append(node)
        for child in node.children:
            edges.append((node.node_id, child.node_id))
            collect(child, depth + 1)

    collect(root, 0)
    if not leaves:
        leaves = [root]

    leaf_x = {node.node_id: i for i, node in enumerate(leaves)}

    def assign_x(node: ModuleNode) -> float:
        if node.is_leaf():
            x = float(leaf_x[node.node_id])
        else:
            child_x = [assign_x(child) for child in node.children]
            x = float(np.mean(child_x))
        pos[node.node_id] = (x, -float(depth_map[node.node_id]))
        return x

    assign_x(root)
    return pos, depth_map, edges


def _build_display_ids(root: ModuleNode) -> dict[str, str]:
    display: dict[str, str] = {root.node_id: "root"}

    def assign(node: ModuleNode, label: str) -> None:
        for idx, child in enumerate(node.children, start=1):
            if label == "root":
                child_label = f"M{idx}"
            else:
                child_label = f"{label}{idx}"
            display[child.node_id] = child_label
            assign(child, child_label)

    assign(root, "root")
    return display


def plot_module_tree(root: ModuleNode, output_path: str | Path | None = None) -> None:
    pos, depth_map, edges = _layout_tree(root)
    display_ids = _build_display_ids(root)

    nodes = list(_iter_nodes_with_depth(root))
    max_depth = max(depth_map.values()) if depth_map else 0
    n_leaves = sum(1 for node, _ in nodes if node.is_leaf())

    fig_w = max(10.0, 1.2 * max(3, n_leaves))
    fig_h = max(6.0, 2.5 + 1.3 * max_depth)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    ax.set_facecolor("#f8fafc")

    for src, dst in edges:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]
        ax.plot([x1, x2], [y1, y2], color="#94a3b8", linewidth=1.8, alpha=0.9, zorder=1)

    cmap = plt.cm.get_cmap("YlGnBu", max_depth + 1 if max_depth >= 0 else 1)

    xs: list[float] = []
    ys: list[float] = []
    sizes: list[float] = []
    colors: list[tuple[float, float, float, float]] = []

    for node, depth in nodes:
        x, y = pos[node.node_id]
        n_genes = len(node.genes)
        xs.append(x)
        ys.append(y)
        sizes.append(380.0 + 80.0 * np.sqrt(n_genes + 1))
        colors.append(cmap(depth))

    ax.scatter(xs, ys, s=sizes, c=colors, edgecolor="#0f172a", linewidth=1.0, zorder=2)

    for node, depth in nodes:
        x, y = pos[node.node_id]
        n_genes = len(node.genes)
        node_label = display_ids.get(node.node_id, node.node_id)
        label = f"{node_label}\nN={n_genes}"
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=8,
            color="#0f172a",
            zorder=3,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
    ax.set_title("Hierarchical Gene Module Tree", fontsize=14, weight="bold", color="#0f172a")
    ax.set_axis_off()
    
    if output_path is not None:
        fig.savefig(output_path, dpi=260, bbox_inches='tight')
        print(f"图已保存至: {output_path}")
    
    plt.show()


def _wrap_genes(genes: list[str], width: int) -> list[str]:
    if not genes:
        return ["-"]
    text = ", ".join(genes)
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    return lines if lines else ["-"]


def plot_node_genes(
    root: ModuleNode,
    output_path: str | Path | None = None,
    *,
    gene_source: str = "direct",
    include_root: bool = True,
    wrap_width: int = 120,
) -> None:
    if gene_source not in {"direct", "total"}:
        raise ValueError("gene_source must be 'direct' or 'total'")

    rows: list[dict[str, object]] = []
    for node, depth in _iter_nodes_with_depth(root):
        if not include_root and node.node_id == "root":
            continue
        genes = sorted(node.total_genes() if gene_source == "total" else node.genes)
        lines = _wrap_genes(genes, width=wrap_width)
        rows.append(
            {
                "node_id": node.node_id,
                "depth": depth,
                "n_genes": len(genes),
                "lines": lines,
            }
        )

    if not rows:
        rows = [{"node_id": "root", "depth": 0, "n_genes": 0, "lines": ["-"]}]

    total_line_units = 3
    for row in rows:
        total_line_units += max(1, len(row["lines"])) + 1

    fig_h = max(6.0, total_line_units * 0.33)
    fig, ax = plt.subplots(figsize=(22, fig_h), constrained_layout=True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, float(total_line_units))
    ax.axis("off")

    y = float(total_line_units - 1)

    ax.add_patch(Rectangle((0.0, y - 0.8), 1.0, 1.1, color="#0f172a", alpha=0.95))
    ax.text(0.01, y - 0.15, "Node", color="white", fontsize=11, weight="bold", va="top")
    ax.text(0.14, y - 0.15, "Depth", color="white", fontsize=11, weight="bold", va="top")
    ax.text(0.20, y - 0.15, "#Genes", color="white", fontsize=11, weight="bold", va="top")
    ax.text(0.28, y - 0.15, f"Genes ({gene_source})", color="white", fontsize=11, weight="bold", va="top")
    y -= 1.5

    for idx, row in enumerate(rows):
        lines = row["lines"]
        n_lines = max(1, len(lines))

        if idx % 2 == 0:
            ax.add_patch(Rectangle((0.0, y - n_lines + 0.05), 1.0, n_lines + 0.45, color="#f1f5f9", alpha=0.9))

        depth = int(row["depth"])
        indent = "  " * depth
        node_label = f"{indent}{row['node_id']}"

        ax.text(0.01, y - 0.1, node_label, color="#0f172a", fontsize=10, family="monospace", va="top")
        ax.text(0.14, y - 0.1, str(row["depth"]), color="#334155", fontsize=10, va="top")
        ax.text(0.20, y - 0.1, str(row["n_genes"]), color="#334155", fontsize=10, va="top")
        ax.text(0.28, y - 0.1, "\n".join(lines), color="#1e293b", fontsize=9, family="monospace", va="top")

        y -= n_lines + 1.0

    ax.set_title("Node-to-Gene Membership", loc="left", fontsize=14, weight="bold", color="#0f172a", pad=12)
    
    if output_path is not None:
        fig.savefig(output_path, dpi=230, bbox_inches='tight')
        print(f"图已保存至: {output_path}")
    
    plt.show()


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out
