from __future__ import annotations

import textwrap
from functools import wraps
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch, Rectangle
from scipy import sparse

from .tree import ModuleNode
from .relationships import binarize_expression
from .utils import select_module_genes


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


def _with_arial(plot):
    """Resolve Arial explicitly, then preserve it in all rendered/exported text."""
    @wraps(plot)
    def wrapped(*args, **kwargs):
        try:
            font_manager.findfont("Arial", fallback_to_default=False)
        except ValueError:
            for path in font_manager.findSystemFonts():
                if Path(path).name.lower().startswith("arial"):
                    font_manager.fontManager.addfont(path)
            font_manager.findfont("Arial", fallback_to_default=False)
        families = ["Arial"]
        if any(f.name == "Arial Unicode MS" for f in font_manager.fontManager.ttflist):
            families.append("Arial Unicode MS")
        with plt.rc_context({"font.family": families, "pdf.fonttype": 42, "svg.fonttype": "path"}):
            return plot(*args, **kwargs)
    return wrapped


def _finish(fig, output_path, show):
    if output_path is not None:
        fig.savefig(output_path, dpi=260, bbox_inches="tight")
    if show:
        plt.show()


@_with_arial
def plot_module_tree(
    root: ModuleNode, output_path: str | Path | None = None, *, show: bool = True
):
    pos, depth_map, edges = _layout_tree(root)

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

    cmap = plt.get_cmap("YlGnBu", max_depth + 1)

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
        label = f"{node.node_id}\nN={n_genes}"
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
    
    _finish(fig, output_path, show)
    return fig, ax


def _wrap_genes(genes: list[str], width: int) -> list[str]:
    if not genes:
        return ["-"]
    text = ", ".join(genes)
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    return lines if lines else ["-"]


@_with_arial
def plot_node_genes(
    root: ModuleNode,
    output_path: str | Path | None = None,
    *,
    gene_source: str = "direct",
    include_root: bool = True,
    wrap_width: int = 120,
    show: bool = True,
):
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

        ax.text(0.01, y - 0.1, node_label, color="#0f172a", fontsize=10, va="top")
        ax.text(0.14, y - 0.1, str(row["depth"]), color="#334155", fontsize=10, va="top")
        ax.text(0.20, y - 0.1, str(row["n_genes"]), color="#334155", fontsize=10, va="top")
        ax.text(0.28, y - 0.1, "\n".join(lines), color="#1e293b", fontsize=9, va="top")

        y -= n_lines + 1.0

    ax.set_title("Node-to-Gene Membership", loc="left", fontsize=14, weight="bold", color="#0f172a", pad=12)
    
    _finish(fig, output_path, show)
    return fig, ax


def ensure_output_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


@_with_arial
def plot_module_heatmap(
    adata,
    root: ModuleNode,
    node_ids: list[str] | None = None,
    output_path: str | Path | None = None,
    *,
    gene_source: str = "total",
    groupby: str | list[str] | None = None,
    cell_order: list[str] | None = None,
    scale: str = "zscore",
    zscore_limit: float | None = 3.0,
    binarize_quantile: float | None = None,
    show: bool = True,
):
    """Gene rows in tree order, cell columns in observed or explicitly supplied order.

    ``groupby`` stably orders cells by one or more ``adata.obs`` columns; an
    explicit ``cell_order`` (cell names, optionally a subset) takes precedence.
    ``scale`` is ``zscore`` (per gene across plotted cells), ``raw`` (adata.X
    as supplied), or ``binary`` (expression > 0 by default; an explicit
    ``binarize_quantile`` uses quantiles computed across all input cells).
    ``zscore_limit`` caps the color range, not expression values; use ``None``
    for the full range. Colorbar arrows indicate values outside the range.
    No clustering is applied, so the displayed pattern can be checked directly.
    """
    genes = select_module_genes(root, node_ids, gene_source=gene_source)
    if not genes:
        raise ValueError("The selected modules contain no genes")
    group_cols = [groupby] if isinstance(groupby, str) else list(groupby or [])
    if cell_order is None:
        cell_order = adata.obs.sort_values(group_cols, kind="stable").index if group_cols else adata.obs_names
    selected = adata[cell_order, genes]
    x = selected.X.toarray() if sparse.issparse(selected.X) else np.asarray(selected.X)
    x = x.astype(float)
    if scale == "zscore":
        std = x.std(axis=0)
        std[std == 0] = 1
        x = (x - x.mean(axis=0)) / std
        limit = zscore_limit if zscore_limit is not None else max(float(np.abs(x).max()), 1e-12)
        image_options = {"cmap": "RdBu_r", "vmin": -limit, "vmax": limit}
        color_label = "Expression (gene z-score)"
    elif scale == "binary":
        source = adata[:, genes].X
        source = source.toarray() if sparse.issparse(source) else np.asarray(source)
        binary = binarize_expression(source, binarize_quantile)
        x = binary[adata.obs_names.get_indexer(selected.obs_names)]
        image_options = {"cmap": "Blues", "vmin": 0, "vmax": 1}
        color_label = "Expression (binary)"
    elif scale == "raw":
        image_options = {"cmap": "viridis"}
        color_label = "Expression (adata.X)"
    else:
        raise ValueError("scale must be 'zscore', 'raw', or 'binary'")

    fig, ax = plt.subplots(figsize=(11, max(3, 1.5 + 0.28 * len(genes))), constrained_layout=True)
    im = ax.imshow(x.T, aspect="auto", interpolation="nearest", **image_options)
    owner = {g: node.node_id for node, _ in _iter_nodes_with_depth(root) for g in node.genes}
    ax.set_yticks(range(len(genes)), [f"{g}  ({owner[g]})" for g in genes], fontsize=9)
    row_modules = [owner[g] for g in genes]
    for i in range(1, len(genes)):
        if row_modules[i] != row_modules[i - 1]:
            ax.axhline(i - 0.5, color="white", lw=1)
    if group_cols:
        labels = selected.obs[group_cols].astype(str).agg(" / ".join, axis=1).to_numpy()
        starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
        ends = np.r_[starts[1:], len(labels)]
        centers = (starts + ends - 1) / 2
        crowded = len(starts) > 10 or (len(starts) > 1 and np.diff(centers).min() < len(labels) * 0.04)
        if crowded:
            categories = list(dict.fromkeys(labels))
            cmap = plt.get_cmap("tab20", len(categories))
            colors = {label: cmap(i) for i, label in enumerate(categories)}
            for start, end in zip(starts, ends):
                ax.axvspan(start - 0.5, end - 0.5, ymin=-0.025, ymax=-0.005,
                           color=colors[labels[start]], linewidth=0, clip_on=False)
            ax.set_xticks([])
            ax.legend(handles=[Patch(color=colors[label], label=label) for label in categories],
                      loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3,
                      frameon=False, fontsize=9)
        else:
            ax.set_xticks(centers, labels[starts], rotation=45, ha="right")
        for start in starts[1:]:
            ax.axvline(start - 0.5, color="#64748b", lw=0.8)
        ax.set_xlabel("Cells grouped by " + " / ".join(group_cols), labelpad=14 if crowded else 4)
    else:
        ax.set_xticks([])
        ax.set_xlabel(f"Cells (n={len(cell_order)})")
    ax.set_ylabel("Gene (module ID)")
    ax.set_title("Selected gene modules")
    extend = "neither"
    if scale == "zscore":
        lower, upper = bool(x.min() < -limit), bool(x.max() > limit)
        extend = "both" if lower and upper else "min" if lower else "max" if upper else "neither"
    fig.colorbar(im, ax=ax, label=color_label, shrink=0.75, extend=extend)
    _finish(fig, output_path, show)
    return fig, ax


@_with_arial
def plot_relationship_heatmap(
    pre,
    root: ModuleNode,
    node_ids: list[str] | None = None,
    output_path: str | Path | None = None,
    *,
    gene_source: str = "total",
    show_gene_names: bool | None = None,
    show: bool = True,
):
    """Plot the selected gene-gene matrix in tree order, without aggregation.

    Up to 40 genes, ticks show gene names. Larger matrices use module labels
    at block centers; every matrix entry still represents one gene pair.
    ``show_gene_names`` overrides this automatic labeling.
    """
    genes = select_module_genes(root, node_ids, gene_source=gene_source)
    if not genes:
        raise ValueError("The selected modules contain no genes")
    gene_index = {g: i for i, g in enumerate(pre.gene_names)}
    indices = [gene_index[g] for g in genes]
    values = pre.phi[np.ix_(indices, indices)].copy()
    np.fill_diagonal(values, np.nan)
    n_genes = len(genes)
    owner = {g: node.node_id for node, _ in _iter_nodes_with_depth(root) for g in node.genes}
    modules = np.array([owner[g] for g in genes])
    starts = np.r_[0, np.flatnonzero(modules[1:] != modules[:-1]) + 1]
    ends = np.r_[starts[1:], n_genes]
    major = {g: child.node_id for child in root.children for g in child.total_genes()}

    fig, ax = plt.subplots(
        figsize=(max(5, min(10, 0.38 * n_genes)), max(4, min(9, 0.34 * n_genes))),
        constrained_layout=True,
    )
    im = ax.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    for start in starts[1:]:
        width = 1.4 if major.get(genes[start]) != major.get(genes[start - 1]) else 0.6
        ax.axvline(start - 0.5, color="white", lw=width)
        ax.axhline(start - 0.5, color="white", lw=width)
    if show_gene_names is None:
        show_gene_names = n_genes <= 40
    if show_gene_names:
        ax.set_xticks(range(n_genes), genes, rotation=90, fontsize=9)
        ax.set_yticks(range(n_genes), genes, fontsize=9)
        axis_label = f"Genes (n={n_genes})"
    else:
        centers = (starts + ends - 1) / 2
        ax.set_xticks(centers, modules[starts], fontsize=10)
        ax.set_yticks(centers, modules[starts], fontsize=10)
        axis_label = f"Genes grouped by module (n={n_genes})"
    ax.set_xlabel(axis_label)
    ax.set_ylabel(axis_label)
    ax.set_title("Gene-gene relationships in tree order")
    fig.colorbar(im, ax=ax, label="Relationship", shrink=0.75)
    _finish(fig, output_path, show)
    return fig, ax
