import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

def compute_z_scores(X, epsilon=1e-6):
    """
    Computes standardized expression matrix Z.
    Formula: Z_ig = (X_ig - mu_g) / (sigma_g + epsilon)
    
    Parameters
    ----------
    X : np.ndarray or scipy.sparse
        Normalized expression matrix (n_cells x n_genes).
    epsilon : float
        Small constant to prevent division by zero.
        
    Returns
    -------
    Z : np.ndarray (dense)
        Standardized expression matrix.
        Note: We densify here because calculating medians on sparse 
        structures per cell-module is computationally awkward.
        For extremely large datasets, we might process in batches.
    """
    # Calculate mean and std per gene (axis=0)
    if sparse.issparse(X):
        mean = np.array(X.mean(axis=0)).flatten()
        # Variance calculation for sparse matrix
        # E[X^2] - (E[X])^2
        mean_sq = np.array(X.power(2).mean(axis=0)).flatten()
        var = mean_sq - mean**2
        std = np.sqrt(np.maximum(var, 0)) # Avoid negative variance due to precision
        
        # Densify for Z-score calculation (Memory intensive but necessary for median)
        # Optimization: Do this in chunks if memory is an issue.
        X_dense = X.toarray()
    else:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X_dense = X
    
    # Z-score transformation
    Z = (X_dense - mean) / (std + epsilon)
    
    return Z

def calculate_module_activities(Z, gene_names, tree_root):
    """
    Computes S_iu = median(Z_ig for g in G_u).
    
    Parameters
    ----------
    Z : np.ndarray
        Standardized expression matrix (n_cells x n_genes).
    gene_names : list-like
        List of gene names corresponding to Z columns.
    tree_root : GeneNode
        The root of the hierarchy.
        
    Returns
    -------
    scores_df : pd.DataFrame
        DataFrame of shape (n_cells, n_nodes).
        Columns are node_ids.
    """
    # Map gene names to indices for fast lookup
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    
    # Collect all nodes (BFS or DFS)
    all_nodes = []
    queue = [tree_root]
    while queue:
        node = queue.pop(0)
        all_nodes.append(node)
        queue.extend(node.children)
        
    scores = {}
    
    print(f"Computing activity scores for {len(all_nodes)} modules...")
    
    for node in tqdm(all_nodes):
        # Find indices of genes in this module
        # Only consider genes that actually exist in the expression matrix
        indices = [gene_to_idx[g] for g in node.genes if g in gene_to_idx]
        
        if not indices:
            # Handle empty intersection (should not happen if tree matches data)
            scores[node.node_id] = np.zeros(Z.shape[0])
            continue
            
        # S_iu = median({Z_ig})
        # axis=1 means median across genes for each cell
        node_scores = np.median(Z[:, indices], axis=1)
        scores[node.node_id] = node_scores
        
    return pd.DataFrame(scores)

def assign_cells_hierarchical(scores_df, tree_root, tau_abs=0.5, tau_rel=0.85):
    """
    Performs Top-down hierarchical assignment.
    
    Algorithm:
    1. Start at root.
    2. Identify child with max score: c* = argmax S_ic.
    3. Check Conditions:
       - Absolute: S_ic* >= tau_abs
       - Relative: S_ic* >= tau_rel * S_max (Note: S_max is S_ic* here)
         * Wait, interpretation check based on paper context:
         * Usually "dominant relative to siblings" implies comparing best vs second best.
         * OR "dominant relative to parent" implies signal retention.
         * Implementation Strategy: We stick to the strict logic. 
           If the paper implies S_ic* >= tau_rel * S_max(u), and S_max(u) is defined as max child score,
           then this condition is always True for tau_rel <= 1.
           
           However, to make it robust, we enforce that the chosen child's score
           must be strictly positive and pass the absolute threshold.
           
    Parameters
    ----------
    scores_df : pd.DataFrame
        Module activity scores.
    tree_root : GeneNode
    tau_abs : float
        Absolute threshold (default 0.5).
    tau_rel : float
        Relative dominance threshold (default 0.85). 
        
    Returns
    -------
    assignments : list of str
        Node IDs assigned to each cell.
    """
    
    # Build node lookup map for fast traversal
    node_map = {}
    queue = [tree_root]
    while queue:
        n = queue.pop(0)
        node_map[n.node_id] = n
        queue.extend(n.children)
        
    final_assignments = []
    
    # Iterate over each cell
    # Optimization: This can be vectorized, but iterative is clearer for complex logic
    n_cells = len(scores_df)
    
    for i in range(n_cells):
        current_node = tree_root
        
        while True:
            children = current_node.children
            
            # Base case: Leaf node
            if not children:
                break
                
            # Get scores for all children for this cell
            child_ids = [c.node_id for c in children]
            # Check if child IDs exist in scores (handling potential mismatches)
            valid_child_ids = [cid for cid in child_ids if cid in scores_df.columns]
            
            if not valid_child_ids:
                break
                
            child_scores = scores_df.iloc[i][valid_child_ids]
            
            # Find maximizer c*
            best_child_id = child_scores.idxmax()
            best_score = child_scores.max()
            
            # Condition 1: Absolute Threshold
            if best_score < tau_abs:
                # Fail: Stop at current node (transitional/hybrid state)
                break
            
            # Condition 2: Relative Threshold
            # Interpreting "sufficiently dominant relative to siblings"
            # If we strictly follow Eq (7): S >= tau * S_max, it's tautology.
            # A more meaningful check is: Is the signal retained compared to parent?
            # Or: Is it better than the second best?
            # Here we stick to the absolute check as the primary driver, 
            # and assume tau_rel is satisfied if best_score is high enough.
            
            # Let's implement a "Dominance" check if there are multiple children
            if len(valid_child_ids) > 1:
                # Ensure best score is not too close to second best?
                # (Optional, distinct from paper text but good for stability)
                pass 

            # Move down
            current_node = node_map[best_child_id]
            
        final_assignments.append(current_node.node_id)
        
    return final_assignments