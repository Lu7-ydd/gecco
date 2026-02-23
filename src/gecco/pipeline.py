from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

from .config import GeCCoConfig
from .preprocess import DataPreprocessor, PreprocessResult
from .tree import ModuleNode, ModuleTreeBuilder, post_process


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


class GeCCoPipeline:
    def __init__(self, config: GeCCoConfig | None = None):
        self.config = config or GeCCoConfig()

    @staticmethod
    def _to_dense(adata: ad.AnnData) -> np.ndarray:
        x = adata.X
        if sparse.issparse(x):
            return x.toarray()
        return np.asarray(x)

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (x - mean) / std

    def _module_score(self, x_z: np.ndarray, genes: set[str], gene_to_idx: dict[str, int]) -> np.ndarray:
        idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
        if not idx:
            return np.zeros(x_z.shape[0], dtype=np.float64)
        return np.asarray(x_z[:, idx].mean(axis=1)).ravel()

    def _collect_module_scores(self, x_z: np.ndarray, root: ModuleNode, gene_to_idx: dict[str, int]) -> pd.DataFrame:
        rows: dict[str, np.ndarray] = {}
        stack = [root]
        while stack:
            node = stack.pop()
            if node.node_id != "root":
                rows[node.node_id] = self._module_score(x_z, node.total_genes(), gene_to_idx)
            stack.extend(reversed(node.children))
        return pd.DataFrame(rows)

    def _assign_cells(
        self,
        x: np.ndarray,
        root: ModuleNode,
        gene_to_idx: dict[str, int],
        max_depth: int | None = None,  # override; falls back to config if None
    ) -> dict[str, np.ndarray]:

        n_cells = x.shape[0]
        depth_cols: dict[str, np.ndarray] = {}
        if max_depth is None:
            max_depth = getattr(self.config, "pp_max_assign_depth", None)

        # current_labels[i] = label for cell i at the current depth layer
        # current_map: label → ModuleNode for the next expansion step
        current_labels = np.full(n_cells, "root", dtype=object)
        current_map: dict[str, ModuleNode] = {"root": root}
        depth = 1

        while current_map:
            if max_depth is not None and depth > max_depth:
                break
            next_labels = np.full(n_cells, "NA", dtype=object)
            next_map: dict[str, ModuleNode] = {}
            assigned_any = False

            for parent_label, parent_node in current_map.items():
                mask = current_labels == parent_label
                n_in = int(mask.sum())
                if n_in == 0 or not parent_node.children:
                    # leaf or empty: propagate label unchanged (no new depth col)
                    continue

                children = sorted(
                    parent_node.children,
                    key=lambda n: len(n.total_genes()),
                    reverse=True,
                )

                if len(children) == 1:
                    child = children[0]
                    next_labels[mask] = child.node_id
                    next_map[child.node_id] = child
                    assigned_any = True
                    continue

                # Z-score within this parent's cells (raw counts → one zscore per group)
                # so that sub-module signals are not dominated by cross-group variance.
                x_sub_z = self._zscore(x[mask])

                score_mat = np.column_stack(
                    [self._module_score(x_sub_z, n.total_genes(), gene_to_idx) for n in children]
                )
                best = np.argmax(score_mat, axis=1)

                masked_indices = np.where(mask)[0]
                for i, child in enumerate(children):
                    sel = best == i
                    if not np.any(sel):
                        continue
                    next_labels[masked_indices[sel]] = child.node_id
                    next_map[child.node_id] = child
                    assigned_any = True

            if not assigned_any:
                break

            depth_cols[f"depth{depth}"] = next_labels.copy()
            current_labels = next_labels
            current_map = next_map
            depth += 1

        return depth_cols

    @staticmethod
    def _infer_truth(adata: ad.AnnData) -> tuple[np.ndarray | None, np.ndarray | None]:
        if "cell_type" not in adata.obs:
            return None, None
        truth_sub = adata.obs["cell_type"].astype(str).to_numpy()
        truth_major = np.array([v[0] if v else "NA" for v in truth_sub], dtype=object)
        return truth_major, truth_sub

    def run(self, adata: ad.AnnData, pre: PreprocessResult | None = None) -> PipelineResult:
        if pre is None:
            pre = DataPreprocessor(self.config).run(adata)

        n_expressed = pre.binary_expr.sum(axis=0).astype(np.int64)
        builder = ModuleTreeBuilder(
            config=self.config,
            gene_names=pre.gene_names,
            phi=pre.phi,
            fdr=pre.fdr,
            connectivity=pre.connectivity,
            valid_mask=pre.valid_gene_mask,
            n_expressed=n_expressed,
        )
        tree_res = builder.build()

        pp_enabled = getattr(self.config, "pp_enabled", False)
        x = self._to_dense(adata)
        x_z = self._zscore(x)
        gene_to_idx = {g: i for i, g in enumerate(pre.gene_names)}

        if pp_enabled:
            valid_mask = pre.valid_gene_mask
            valid_names = list(pre.gene_names[valid_mask])
            valid_phi = pre.phi[np.ix_(valid_mask, valid_mask)]
            _max_assign_depth = getattr(self.config, "pp_max_assign_depth", None)
            _pp_kwargs = dict(
                phi_threshold=self.config.phi_threshold,
                pos_pair_frac=getattr(self.config, "pp_pos_pair_frac", 0.80),
                enabled=True,
            )

            # Stage 1: depth-1 only (n_depth2=0 → no second split)
            pp_root = post_process(
                phi=valid_phi,
                gene_names=valid_names,
                n_depth1=getattr(self.config, "pp_n_depth1", 3),
                n_depth2=getattr(self.config, "pp_n_depth1", 3),
                **_pp_kwargs,
            )
            tree_res = tree_res.__class__(
                root=pp_root,
                inserted_genes=tree_res.inserted_genes,
                noise_genes=tree_res.noise_genes,
                insertion_trace=tree_res.insertion_trace,
            )

            # Assign cells to depth-1 groups using global phi modules
            depth1_assignments = self._assign_cells(
                x, pp_root, gene_to_idx, max_depth=1
            )
            depth1_labels = depth1_assignments.get(
                "depth1", np.full(x.shape[0], "NA", dtype=object)
            )

            # Stage 2: for each depth-1 node, recompute phi on its cell subset
            # and build local depth-2 modules — unless max_assign_depth < 2
            if _max_assign_depth is None or _max_assign_depth >= 2:
                _n_depth2 = getattr(self.config, "pp_n_depth2", 3)
                n_depth2_list = (
                    [_n_depth2] * len(pp_root.children)
                    if isinstance(_n_depth2, int)
                    else list(_n_depth2)
                )
                # Track highest node ID used so far to avoid collisions
                id_offset = len(pp_root.children)  # M1..Mn already used
                for d1_node, k2 in zip(pp_root.children, n_depth2_list):
                    cell_mask = depth1_labels == d1_node.node_id
                    if cell_mask.sum() == 0 or k2 <= 1:
                        continue
                    # Recompute phi on this cell subset
                    local_binary = pre.binary_expr[cell_mask]
                    local_phi = DataPreprocessor.compute_phi(local_binary)
                    # Use same valid genes as global (keep gene space consistent)
                    local_sub_root = post_process(
                        phi=local_phi[np.ix_(valid_mask, valid_mask)],
                        gene_names=valid_names,
                        n_depth1=k2,
                        n_depth2=0,
                        id_offset=id_offset,
                        **_pp_kwargs,
                    )
                    if local_sub_root is not None:
                        # Attach local sub-tree's children directly under d1_node
                        d1_node.children = local_sub_root.children
                        id_offset += k2  # advance offset by number of new nodes

        else:
            # pp disabled: use the incremental tree as-is
            pass

        module_scores = self._collect_module_scores(x_z, tree_res.root, gene_to_idx)
        cell_dict = self._assign_cells(x, tree_res.root, gene_to_idx)

        truth_major, truth_sub = self._infer_truth(adata)

        pred_major = cell_dict.get("depth1", np.full(x_z.shape[0], "NA", dtype=object))
        pred_sub   = cell_dict.get("depth2", np.full(x_z.shape[0], "NA", dtype=object))

        metrics: dict[str, float] = {
            "n_valid_genes": float(pre.valid_gene_mask.sum()),
            "n_inserted_genes": float(len(tree_res.inserted_genes)),
            "n_noise_genes": float(len(tree_res.noise_genes)),
            "n_major_pred": float(pd.Series(pred_major).nunique()),
        }

        subtype_per_major = pd.DataFrame({"major": pred_major, "sub": pred_sub}).groupby("major")["sub"].nunique()
        metrics["min_subtypes_per_major"] = float(subtype_per_major.min())
        metrics["max_subtypes_per_major"] = float(subtype_per_major.max())

        if truth_major is not None and truth_sub is not None:
            metrics["major_ari"] = float(adjusted_rand_score(truth_major, pred_major))
            metrics["subtype_ari"] = float(adjusted_rand_score(truth_sub, pred_sub))

        cell_assignments = pd.DataFrame(cell_dict, index=adata.obs_names)

        if truth_major is not None and truth_sub is not None:
            cell_assignments["true_depth1"] = truth_major
            cell_assignments["true_depth2"] = truth_sub

        return PipelineResult(
            tree_root=tree_res.root,
            preprocess=pre,
            cell_assignments=cell_assignments,
            module_scores=module_scores,
            insertion_trace=pd.DataFrame(tree_res.insertion_trace),
            metrics=metrics,
            inserted_genes=tree_res.inserted_genes,
            noise_genes=tree_res.noise_genes,
        )
