from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

from .config import GeCCoConfig
from .preprocess import DataPreprocessor, PreprocessResult
from .tree import ModuleNode, ModuleTreeBuilder, post_process
from .utils import module_table


@dataclass(slots=True)
class PipelineResult:
    tree_root: ModuleNode
    preprocess: PreprocessResult
    cell_assignments: pd.DataFrame
    module_scores: pd.DataFrame
    insertion_trace: pd.DataFrame
    metrics: dict[str, float]
    inserted_genes: list[str]
    noise_genes: list[str]

    @property
    def module_table(self) -> pd.DataFrame:
        return module_table(self.tree_root)


class GeCCoPipeline:
    def __init__(self, config: GeCCoConfig | None = None):
        self.config = config or GeCCoConfig()

    @staticmethod
    def _to_dense(adata: ad.AnnData) -> np.ndarray:
        return adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (x - mean) / std

    def _module_score(self, x_z, genes, gene_to_idx):
        idx = [gene_to_idx[g] for g in sorted(genes)]
        return x_z[:, idx].mean(axis=1) if idx else np.zeros(x_z.shape[0])

    def _collect_module_scores(self, x_z, root, gene_to_idx) -> pd.DataFrame:
        rows = {}
        stack = [root]
        while stack:
            node = stack.pop()
            if node.node_id != "root":
                rows[node.node_id] = self._module_score(x_z, node.genes or node.total_genes(), gene_to_idx)
            stack.extend(reversed(node.children))
        return pd.DataFrame(rows, index=np.arange(x_z.shape[0]))

    def _assign_cells(self, x, root, gene_to_idx, max_depth=None):
        """Keep the original within-parent z scoring, using each node's direct markers."""
        n_cells = x.shape[0]
        depth_cols = {}
        current_labels = np.full(n_cells, "root", dtype=object)
        current_map = {"root": root}
        depth = 1
        while current_map and (max_depth is None or depth <= max_depth):
            # Cells that reached a leaf retain it while other branches expand.
            next_labels = current_labels.copy()
            next_map = {}
            for parent_label, parent_node in current_map.items():
                mask = current_labels == parent_label
                if not np.any(mask) or not parent_node.children:
                    continue
                children = parent_node.children
                x_sub_z = self._zscore(x[mask])
                score_mat = np.column_stack([
                    self._module_score(x_sub_z, n.genes or n.total_genes(), gene_to_idx)
                    for n in children
                ])
                best = np.argmax(score_mat, axis=1)
                indices = np.flatnonzero(mask)
                for i, child in enumerate(children):
                    sel = best == i
                    if np.any(sel):
                        next_labels[indices[sel]] = child.node_id
                        next_map[child.node_id] = child
            if not next_map:
                break
            depth_cols[f"depth{depth}"] = next_labels.copy()
            current_labels, current_map = next_labels, next_map
            depth += 1
        return depth_cols

    def run(
        self, adata: ad.AnnData, pre: PreprocessResult | None = None, *,
        genes: Sequence[str] | None = None,
        relationship_matrix: pd.DataFrame | np.ndarray | None = None,
        fdr_matrix: pd.DataFrame | np.ndarray | None = None,
        major_key: str | None = None, subtype_key: str | None = None,
    ) -> PipelineResult:
        if pre is None:
            pre = DataPreprocessor(self.config).run(adata, genes, relationship_matrix, fdr_matrix)
        elif genes is not None or relationship_matrix is not None or fdr_matrix is not None:
            raise ValueError("Supply either pre or genes/relationship_matrix/fdr_matrix")
        if self.config.test and pre.fdr is None:
            raise ValueError("test=True requires preprocessing with statistical tests enabled")
        if pre.cell_names is not None and not np.array_equal(pre.cell_names, adata.obs_names):
            raise ValueError("pre and adata must have the same cells in the same order")

        builder = ModuleTreeBuilder(
            config=self.config, gene_names=pre.gene_names, phi=pre.phi, fdr=pre.fdr,
            connectivity=pre.connectivity, valid_mask=pre.valid_gene_mask,
            n_expressed=pre.binary_expr.sum(axis=0),
        )
        tree_res = builder.build()
        root = tree_res.root
        trace = pd.DataFrame(tree_res.insertion_trace, columns=[
            "step", "gene", "gene_index", "connectivity", "rule", "target_node_id",
            "path", "accepted", "is_noise", "changed_nodes", "split_parent_id",
            "split_child_ids", "reassigned_existing_genes",
        ])

        if self.config.pp_enabled:
            valid = pre.valid_gene_mask
            root = post_process(
                phi=pre.phi[np.ix_(valid, valid)], gene_names=pre.gene_names[valid],
                n_depth1=self.config.pp_n_depth1, n_depth2=self.config.pp_n_depth2,
                branching=self.config.pp_branching, phi_threshold=self.config.phi_threshold,
                pos_pair_frac=self.config.pp_pos_pair_frac,
            )
        final_genes = root.total_genes()
        inserted = [g for g in pre.gene_names if g in final_genes]
        noise = [g for g in pre.gene_names if g not in final_genes]
        trace["in_final_tree"] = trace["gene"].isin(final_genes)
        # Trace node IDs describe the incremental tree; post mode rebuilds the topology.
        trace["stage"] = "incremental"

        selected_adata = adata[:, pre.gene_names]
        x = self._to_dense(selected_adata)
        gene_to_idx = {g: i for i, g in enumerate(pre.gene_names)}
        module_scores = self._collect_module_scores(self._zscore(x), root, gene_to_idx)
        module_scores.index = adata.obs_names
        max_assign_depth = self.config.pp_max_assign_depth if self.config.pp_enabled else None
        cell_dict = self._assign_cells(x, root, gene_to_idx, max_depth=max_assign_depth)
        cell_assignments = pd.DataFrame(cell_dict, index=adata.obs_names)
        metrics = {
            "n_valid_genes": float(pre.valid_gene_mask.sum()),
            "n_inserted_genes": float(len(inserted)),
            "n_noise_genes": float(len(noise)),
            "n_major_pred": float(cell_assignments["depth1"].nunique()) if "depth1" in cell_dict else 0.0,
        }
        for key, depth, label in [(major_key, "depth1", "major"), (subtype_key, "depth2", "subtype")]:
            if key is not None and depth in cell_dict:
                truth = adata.obs[key].astype(str)
                metrics[f"{label}_ari"] = float(adjusted_rand_score(truth, cell_dict[depth]))
                cell_assignments[f"true_{depth}"] = truth
        return PipelineResult(root, pre, cell_assignments, module_scores, trace, metrics, inserted, noise)
