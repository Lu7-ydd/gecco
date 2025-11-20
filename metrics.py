import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings

def binarize_expression(X, threshold=0.5):
    """
    Projects continuous transcriptomic manifold onto a Boolean hypercube.
    
    Matches LaTeX Section 2.2:
    B_{gi} = 1 if X_{gi} > tau else 0
    
    Parameters
    ----------
    X : np.ndarray or scipy.sparse matrix
        Normalized expression matrix (Shape: n_cells x n_genes).
    threshold : float
        The threshold tau (default 0.5 TPM).
        
    Returns
    -------
    B : np.ndarray (bool)
        Binary state matrix.
    """
    # Handle sparse input efficiency if necessary, but for correlation usually dense is needed eventually
    if hasattr(X, "toarray"):
        X = X.toarray()
        
    # Binarize
    B = (X > threshold).astype(int)
    return B

def compute_phi_matrix(B):
    """
    Computes the pairwise Phi coefficient matrix using vectorized covariance.
    
    Matches LaTeX Section 2.2:
    phi_{ij} = Cov(I_i, I_j) / sqrt(Var(I_i)Var(I_j))
    
    Parameters
    ----------
    B : np.ndarray
        Binary matrix (n_cells x n_genes).
        
    Returns
    -------
    phi_matrix : np.ndarray
        Symmetric matrix of range [-1, 1].
        NaNs will appear for genes with 0 variance (all 0s or all 1s).
    """
    # Check for constant genes (variance = 0) to avoid division by zero warnings
    # In binary data, var = p(1-p). If p=0 or p=1, var=0.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # np.corrcoef implements exactly the Pearson correlation formula
        # rowvar=False implies columns are variables (genes)
        phi_matrix = np.corrcoef(B, rowvar=False)
    
    # Replace NaNs (caused by constant genes) with 0
    phi_matrix = np.nan_to_num(phi_matrix, nan=0.0)
    
    # Remove self-correlation (diagonal = 1) to avoid trivial edges later
    np.fill_diagonal(phi_matrix, 0)
    
    return phi_matrix

def compute_edge_statistics(B, gene_names, phi_threshold=0.1, fdr_alpha=0.05):
    """
    Performs statistical inference on regulatory edges.
    
    Workflow:
    1. Compute global Phi matrix.
    2. Pre-filter edges where |phi| < threshold (Optimization).
    3. Perform One-tailed Fisher's Exact Test on remaining edges.
    4. Apply Benjamini-Hochberg FDR correction.
    
    Parameters
    ----------
    B : np.ndarray
        Binary matrix (n_cells x n_genes).
    gene_names : list-like
        Names of genes corresponding to columns of B.
    phi_threshold : float
        Minimum absolute effect size (default 0.1).
    fdr_alpha : float
        Significance level for FDR (default 0.05).
        
    Returns
    -------
    valid_edges : pd.DataFrame
        DataFrame containing significant edges. 
        Columns: ['source', 'target', 'phi', 'p_value', 'p_adj', 'sign']
    """
    n_cells = B.shape[0]
    n_genes = B.shape[1]
    
    # 1. Compute Phi Matrix
    print(f"Computing Phi matrix for {n_genes} genes...")
    phi_matrix = compute_phi_matrix(B)
    
    # 2. Pre-filtering: Get indices where |phi| >= threshold
    # We only look at upper triangle to avoid duplicates (i,j) and (j,i)
    # triu_indices(k=1) excludes the diagonal
    rows, cols = np.where(np.triu(np.abs(phi_matrix), k=1) >= phi_threshold)
    
    print(f"Testing significance for {len(rows)} potential regulatory edges...")
    
    results = []
    
    # Pre-calculate column sums (number of active cells per gene) for contingency table
    # gene_sums[i] = a + c
    gene_counts = np.sum(B, axis=0)
    
    # 3. Fisher's Exact Test (Iterative but optimized)
    for i, j in zip(rows, cols):
        phi_val = phi_matrix[i, j]
        
        # Construct Contingency Table
        #       j=1     j=0
        # i=1   a       b
        # i=0   c       d
        
        # a: count where both are 1 (dot product of column i and j)
        a = np.dot(B[:, i], B[:, j])
        
        # Marginals
        row_total = gene_counts[i] # a + b
        col_total = gene_counts[j] # a + c
        
        b = row_total - a
        c = col_total - a
        d = n_cells - (a + b + c)
        
        table = [[a, b], [c, d]]
        
        # Hypothesis testing based on sign of phi
        # If phi > 0: Alt = 'greater' (Positive association)
        # If phi < 0: Alt = 'less' (Negative association)
        alternative = 'greater' if phi_val > 0 else 'less'
        
        _, p_val = stats.fisher_exact(table, alternative=alternative)
        
        results.append({
            'source_idx': i,
            'target_idx': j,
            'source': gene_names[i],
            'target': gene_names[j],
            'phi': phi_val,
            'p_value': p_val
        })
    
    if not results:
        return pd.DataFrame(columns=['source', 'target', 'phi', 'p_value', 'p_adj', 'sign'])

    edges_df = pd.DataFrame(results)
    
    # 4. FDR Correction (Benjamini-Hochberg)
    # Note: We correct based on the number of tests performed, 
    # or ideally, the total number of possible pairs? 
    # Standard practice in rigorous network reconstruction is correcting for tested hypotheses.
    _, p_adj, _, _ = multipletests(edges_df['p_value'], alpha=fdr_alpha, method='fdr_bh')
    edges_df['p_adj'] = p_adj
    
    # 5. Final Filtering
    valid_edges = edges_df[edges_df['p_adj'] < fdr_alpha].copy()
    
    # Add 'sign' column for easy usage in tree building
    valid_edges['sign'] = valid_edges['phi'].apply(lambda x: 1 if x > 0 else -1)
    
    print(f"Found {len(valid_edges)} significant edges (FDR < {fdr_alpha}).")
    
    return valid_edges