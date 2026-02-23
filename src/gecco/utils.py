from __future__ import annotations

import re
from collections import defaultdict

import pandas as pd

from .tree import ModuleNode
def add_depth_labels_to_obs(adata, result, *, include_true=False):
    """
    把 result.cell_assignments 里的层级标签写入 adata.obs。
    默认写入 depth1/depth2/...；可选写入 true_depth1/true_depth2/...。
    """
    ca = result.cell_assignments.reindex(adata.obs_names)

    def _depth_key(col):
        m = re.match(r"^depth(\d+)$", col)
        return int(m.group(1)) if m else 10**9

    depth_cols = sorted([c for c in ca.columns if re.match(r"^depth\d+$", c)], key=_depth_key)
    if not depth_cols:
        raise ValueError("result.cell_assignments 里没有 depth 列")

    adata.obs[depth_cols] = ca[depth_cols].astype("object")

    if include_true:
        true_cols = sorted([c for c in ca.columns if re.match(r"^true_depth\d+$", c)], key=lambda x:
int(x[10:]))
        if true_cols:
            adata.obs[true_cols] = ca[true_cols].astype("object")
            



def _iter_nodes_with_depth_and_path(root: ModuleNode):
    stack: list[tuple[ModuleNode, int, list[str]]] = [(root, 0, ["root"])]
    while stack:
        node, depth, path = stack.pop()
        yield node, depth, path
        for child in reversed(node.children):
            stack.append((child, depth + 1, path + [child.node_id]))


def add_tree_labels_to_var(
    adata,
    tree_root: ModuleNode,
    *,
    gene_source: str = "direct",
    include_root: bool = True,
) -> None:
    """
    给每个基因打树层级标签，写到 adata.var。
    标签语义与 plot_node_genes 一致（gene_source / include_root 同义）。

    输出列：
    - depth{d}: 第 d 层节点ID（动态列，不再固定 depth1~depth3）
    - tree_path: 基因所在节点路径（多路径用 ';' 拼接）
    - tree_depth: 基因出现的最深层级（未命中为 0）
    """
    if gene_source not in {"direct", "total"}:
        raise ValueError("gene_source must be 'direct' or 'total'")

    # 清理旧 depth 列，避免历史列残留。
    old_depth_cols = [c for c in adata.var.columns if re.match(r"^depth\d+$", str(c))]
    if old_depth_cols:
        adata.var.drop(columns=old_depth_cols, inplace=True)

    gene_depth_nodes: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    gene_paths: dict[str, set[str]] = defaultdict(set)
    depth_levels: set[int] = set()

    for node, depth, path in _iter_nodes_with_depth_and_path(tree_root):
        if not include_root and node.node_id == "root":
            continue

        genes = node.total_genes() if gene_source == "total" else node.genes
        if not genes:
            continue

        depth_levels.add(depth)
        path_str = "->".join(path)
        for g in genes:
            gene_depth_nodes[g][depth].add(node.node_id)
            gene_paths[g].add(path_str)

    genes = adata.var_names.astype(str)
    sorted_depths = sorted(depth_levels)

    for d in sorted_depths:
        col = f"depth{d}"
        adata.var[col] = pd.Series(
            [
                "|".join(sorted(gene_depth_nodes.get(g, {}).get(d, set())))
                if gene_depth_nodes.get(g, {}).get(d)
                else "NA"
                for g in genes
            ],
            index=adata.var_names,
            dtype="object",
        )

    adata.var["tree_path"] = pd.Series(
        [
            ";".join(sorted(gene_paths.get(g, set()))) if gene_paths.get(g) else "NA"
            for g in genes
        ],
        index=adata.var_names,
        dtype="object",
    )

    adata.var["tree_depth"] = pd.Series(
        [
            max(gene_depth_nodes.get(g, {}).keys()) if gene_depth_nodes.get(g) else 0
            for g in genes
        ],
        index=adata.var_names,
        dtype="int64",
    )
