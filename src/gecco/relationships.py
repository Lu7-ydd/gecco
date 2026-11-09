"""Gene-gene relationships, independent of tree construction.

New methods can be computed elsewhere and passed to the pipeline as a labelled
DataFrame. Positive scores mean co-expression; negative scores mean exclusion.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta, fisher_exact
from statsmodels.stats.multitest import fdrcorrection


def binarize_expression(expression: np.ndarray, quantile: float | None = None) -> np.ndarray:
    """Use expression > 0 by default; an explicit quantile preserves binary columns."""
    x = np.asarray(expression)
    if quantile is None:
        return (x > 0).astype(np.int8)
    thresholds = np.quantile(x, quantile, axis=0)
    thresholds[np.all((x == 0) | (x == 1), axis=0)] = 0
    return (x > thresholds).astype(np.int8)


def pearson_correlation(expression: np.ndarray) -> np.ndarray:
    """Column-wise Pearson correlation; constant genes have zero relationships."""
    x = np.asarray(expression, dtype=np.float64)
    centered = x - x.mean(axis=0)
    norm = np.linalg.norm(centered, axis=0)
    scaled = np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 0)
    correlation = np.clip(scaled.T @ scaled, -1.0, 1.0)
    correlation[np.abs(correlation) <= np.finfo(np.float64).eps] = 0.0
    np.fill_diagonal(correlation, 0.0)
    return correlation


def binary_correlation(binary_expr: np.ndarray) -> np.ndarray:
    """Compute phi from counts, preserving exact zero for independent pairs."""
    x = np.asarray(binary_expr, dtype=np.int64)
    n_cells = x.shape[0]
    expressed = x.sum(axis=0)
    numerator = n_cells * (x.T @ x) - expressed[:, None] * expressed[None, :]
    variance = expressed.astype(np.float64) * (n_cells - expressed)
    denominator = np.sqrt(variance[:, None] * variance[None, :])
    correlation = np.divide(numerator, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    np.fill_diagonal(correlation, 0.0)
    return correlation


def _adjust_pairs(pvalues: np.ndarray, n_genes: int) -> np.ndarray:
    upper = np.triu_indices(n_genes, 1)
    fdr = np.ones((n_genes, n_genes), dtype=np.float64)
    if pvalues.size:
        fdr[upper] = fdrcorrection(pvalues, method="indep")[1]
        fdr[(upper[1], upper[0])] = fdr[upper]
    np.fill_diagonal(fdr, 0.0)
    return fdr


def binary_fdr(binary_expr: np.ndarray) -> np.ndarray:
    """Two-sided Fisher exact tests, BH-adjusted over unique gene pairs."""
    x = np.asarray(binary_expr, dtype=np.int64)
    n_cells, n_genes = x.shape
    both = x.T @ x
    expressed = x.sum(axis=0)
    pvalues = []
    for i, j in zip(*np.triu_indices(n_genes, 1)):
        a = int(both[i, j])
        b, c = int(expressed[i] - a), int(expressed[j] - a)
        pvalues.append(fisher_exact([[a, b], [c, n_cells - a - b - c]])[1])
    return _adjust_pairs(np.asarray(pvalues), n_genes)


def pearson_fdr(expression: np.ndarray, correlation: np.ndarray | None = None) -> np.ndarray:
    """Two-sided Pearson tests, BH-adjusted over unique gene pairs."""
    x = np.asarray(expression, dtype=np.float64)
    n_cells, n_genes = x.shape
    if n_cells < 2:
        raise ValueError("Pearson tests require at least two cells")
    if correlation is None:
        correlation = pearson_correlation(x)
    upper = np.triu_indices(n_genes, 1)
    if n_cells == 2:
        pvalues = np.ones(len(upper[0]))
    else:
        distribution = beta(n_cells / 2 - 1, n_cells / 2 - 1, loc=-1, scale=2)
        pvalues = 2 * distribution.cdf(-np.abs(correlation[upper]))
    return _adjust_pairs(pvalues, n_genes)


def compute_relationship_matrix(
    expression: np.ndarray,
    method: str = "binary",
    test: bool = False,
    binarize_quantile: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute a relationship matrix and optional FDR matrix from cells × genes.

    ``binary`` uses expression > 0, phi and optional Fisher tests. An explicit
    ``binarize_quantile`` uses per-gene quantiles, preserving existing binary columns.
    ``pearson`` uses the supplied expression values and optional Pearson tests.
    No hypothesis tests are run when ``test=False``.
    """
    if method == "binary":
        binary = binarize_expression(expression, binarize_quantile)
        return binary_correlation(binary), binary_fdr(binary) if test else None
    if method == "pearson":
        correlation = pearson_correlation(expression)
        return correlation, pearson_fdr(expression, correlation) if test else None
    raise ValueError("relationship_method must be 'binary' or 'pearson'; pass custom scores as relationship_matrix")
