from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .config import GeCCoConfig
from .relationships import (
    binarize_expression,
    binary_correlation,
    binary_fdr,
    compute_relationship_matrix,
)


@dataclass(slots=True)
class PreprocessResult:
    gene_names: np.ndarray
    binary_expr: np.ndarray
    phi: np.ndarray
    fdr: np.ndarray | None
    valid_gene_mask: np.ndarray
    connectivity: np.ndarray
    cell_names: np.ndarray | None = None

    @property
    def valid_gene_names(self) -> np.ndarray:
        return self.gene_names[self.valid_gene_mask]

    @property
    def relationship_matrix(self) -> np.ndarray:
        """``phi`` is retained as a compatibility name for any relationship method."""
        return self.phi


class DataPreprocessor:
    def __init__(self, config: GeCCoConfig):
        self.config = config

    @staticmethod
    def _to_dense(adata: ad.AnnData) -> np.ndarray:
        x = adata.X
        return x.toarray() if sparse.issparse(x) else np.asarray(x)

    def binarize(self, adata: ad.AnnData) -> np.ndarray:
        return binarize_expression(self._to_dense(adata), self.config.binarize_quantile)

    # Preserve the original callable helpers used in existing notebooks.
    compute_phi = staticmethod(binary_correlation)
    compute_fdr = staticmethod(binary_fdr)

    def filter_valid_genes(self, phi: np.ndarray, fdr: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        related = (np.abs(phi) >= self.config.phi_threshold) & (phi != 0)
        if self.config.test:
            if fdr is None:
                raise ValueError("test=True requires an FDR matrix")
            related &= fdr <= self.config.fdr_threshold
        np.fill_diagonal(related, False)
        # Isolated selected genes remain visible to the builder and its trace.
        return np.ones(phi.shape[0], dtype=bool), related.sum(axis=1)

    @staticmethod
    def _align_matrix(matrix: pd.DataFrame | np.ndarray, genes: np.ndarray, label: str) -> np.ndarray:
        if isinstance(matrix, pd.DataFrame):
            if not matrix.index.is_unique or not matrix.columns.is_unique:
                raise ValueError(f"{label} gene labels must be unique")
            matrix = matrix.loc[genes, genes].to_numpy()
        values = np.array(matrix, dtype=np.float64, copy=True)
        if values.shape != (len(genes), len(genes)):
            raise ValueError(f"{label} shape must match the selected genes")
        if not np.isfinite(values).all() or not np.allclose(values, values.T):
            raise ValueError(f"{label} must be finite and symmetric")
        np.fill_diagonal(values, 0.0)
        return values

    def run(
        self,
        adata: ad.AnnData,
        genes: Sequence[str] | None = None,
        relationship_matrix: pd.DataFrame | np.ndarray | None = None,
        fdr_matrix: pd.DataFrame | np.ndarray | None = None,
    ) -> PreprocessResult:
        gene_names = adata.var_names.to_numpy(copy=True) if genes is None else np.asarray(list(genes))
        if not len(gene_names) or len(set(gene_names)) != len(gene_names):
            raise ValueError("Select at least one gene, with no duplicate names")
        x = self._to_dense(adata[:, gene_names])
        binary = binarize_expression(x, self.config.binarize_quantile)

        if relationship_matrix is None:
            if fdr_matrix is not None:
                raise ValueError("fdr_matrix is only used with a custom relationship_matrix")
            phi, fdr = compute_relationship_matrix(
                x, self.config.relationship_method, self.config.test, self.config.binarize_quantile
            )
        else:
            phi = self._align_matrix(relationship_matrix, gene_names, "relationship_matrix")
            if self.config.test and fdr_matrix is None:
                raise ValueError("A custom relationship_matrix with test=True requires fdr_matrix")
            fdr = self._align_matrix(fdr_matrix, gene_names, "fdr_matrix") if self.config.test else None
            if fdr is not None and np.any((fdr < 0) | (fdr > 1)):
                raise ValueError("fdr_matrix values must be between 0 and 1")

        valid_mask, connectivity = self.filter_valid_genes(phi, fdr)
        return PreprocessResult(gene_names, binary, phi, fdr, valid_mask, connectivity, adata.obs_names.to_numpy(copy=True))
