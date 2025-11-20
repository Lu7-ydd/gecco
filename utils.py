import numpy as np
from scipy import sparse
import pickle
import warnings

def check_is_normalized(X, sample_size=100):
    """
    Simple heuristic to check if data appears to be normalized (log1p or TPM).
    If max value is huge (e.g. > 10000) and integers, it's likely raw counts.
    
    Parameters
    ----------
    X : np.ndarray or sparse matrix
    """
    if sparse.issparse(X):
        # Sample a subset of data to check max
        if X.shape[0] > sample_size:
            # Randomly sample rows
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            sample = X[indices, :]
            max_val = sample.max()
        else:
            max_val = X.max()
    else:
        max_val = np.max(X)

    # Heuristic warning
    if max_val > 1000: 
        warnings.warn(
            f"Max expression value is {max_val}. "
            "GeCCo expects normalized data (TPM/CPM or Log-normalized). "
            "Passing raw counts will lead to incorrect binarization."
        )

def save_model(model, filepath):
    """
    Helper to save a trained GeCCo model to disk.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """
    Helper to load a trained GeCCo model from disk.
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from {filepath}")
    return model

def print_tree_structure(node, depth=0, max_depth=3):
    """
    Utility to visualize the hierarchy in the console.
    Useful for quick debugging without plotting full graphics.
    """
    indent = "  " * depth
    
    # Format: NodeID (N genes)
    gene_preview = ",".join(list(node.genes)[:3])
    if len(node.genes) > 3:
        gene_preview += "..."
        
    print(f"{indent}├── [{node.node_id}] ({len(node.genes)} genes: {gene_preview})")
    
    if depth < max_depth:
        for child in node.children:
            print_tree_structure(child, depth + 1, max_depth)
    elif len(node.children) > 0:
        print(f"{indent}  └── ... ({len(node.children)} children hidden)")