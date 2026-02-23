from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
from scipy import sparse
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import fdrcorrection

from .config import GeCCoConfig


@dataclass(slots=True)
class PreprocessResult:
    gene_names: np.ndarray
    binary_expr: np.ndarray
    phi: np.ndarray
    fdr: np.ndarray
    valid_gene_mask: np.ndarray
    connectivity: np.ndarray

    @property
    def valid_gene_names(self) -> np.ndarray:
        return self.gene_names[self.valid_gene_mask]


class DataPreprocessor:
    def __init__(self, config: GeCCoConfig):
        self.config = config

    @staticmethod
    def _to_dense(adata: ad.AnnData) -> np.ndarray:
        x = adata.X
        if sparse.issparse(x):
            return x.toarray()
        return np.asarray(x)

    def binarize(self, adata: ad.AnnData) -> np.ndarray:
        x = self._to_dense(adata)
        thresholds = np.quantile(x, self.config.binarize_quantile, axis=0)
        return (x > thresholds).astype(np.int8)

    @staticmethod
    def compute_phi(binary_expr: np.ndarray) -> np.ndarray:
        n_cells, _ = binary_expr.shape
        x = binary_expr.astype(np.int64)

        n11 = x.T @ x
        n1 = x.sum(axis=0)

        n10 = n1[:, None] - n11
        n01 = n1[None, :] - n11
        n00 = n_cells - n11 - n10 - n01

        numerator = n11 * n00 - n10 * n01
        denom = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))

        phi = np.zeros_like(numerator, dtype=np.float64)
        mask = denom > 0
        phi[mask] = numerator[mask] / denom[mask]
        np.fill_diagonal(phi, 0.0)
        return phi

    @staticmethod
    def compute_fdr(binary_expr: np.ndarray) -> np.ndarray:
        _, n_genes = binary_expr.shape
        x = binary_expr.astype(np.int8)
        iu = np.triu_indices(n_genes, 1)

        pvals = np.ones(len(iu[0]), dtype=np.float64)
        for idx, (i, j) in enumerate(zip(iu[0], iu[1], strict=False)):
            g1 = x[:, i]
            g2 = x[:, j]
            a = int(np.sum((g1 == 1) & (g2 == 1)))
            b = int(np.sum((g1 == 1) & (g2 == 0)))
            c = int(np.sum((g1 == 0) & (g2 == 1)))
            d = int(np.sum((g1 == 0) & (g2 == 0)))
            _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            pvals[idx] = p

        _, qvals = fdrcorrection(pvals, alpha=0.05, method="indep")

        fdr = np.ones((n_genes, n_genes), dtype=np.float64)
        fdr[iu] = qvals
        fdr[(iu[1], iu[0])] = qvals
        np.fill_diagonal(fdr, 0.0)
        return fdr

    def filter_valid_genes(self, phi: np.ndarray, fdr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sig = (np.abs(phi) >= self.config.phi_threshold) & (fdr <= self.config.fdr_threshold)
        np.fill_diagonal(sig, False)
        connectivity = sig.sum(axis=1)
        valid = connectivity >= 1
        return valid, connectivity

    def run(self, adata: ad.AnnData) -> PreprocessResult:
        binary = self.binarize(adata)
        phi = self.compute_phi(binary)
        fdr = self.compute_fdr(binary)
        valid_mask, connectivity = self.filter_valid_genes(phi, fdr)
        return PreprocessResult(
            gene_names=adata.var_names.to_numpy(copy=True),
            binary_expr=binary,
            phi=phi,
            fdr=fdr,
            valid_gene_mask=valid_mask,
            connectivity=connectivity,
        )
