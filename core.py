import scanpy as sc
from .metrics import binarize_expression, compute_edge_statistics
from .hierarchy import GeneHierarchy
from .scoring import compute_z_scores, calculate_module_activities, assign_cells_hierarchical

class GeCCo:
    def __init__(self, 
                 bin_threshold=0.5, 
                 phi_threshold=0.1, 
                 fdr=0.05,
                 assign_tau_abs=0.5,
                 assign_tau_rel=0.85):
        """
        GeCCo: Gene Co-expression Constructed identity framework.
        """
        # Parameters
        self.bin_threshold = bin_threshold
        self.phi_threshold = phi_threshold
        self.fdr = fdr
        self.tau_abs = assign_tau_abs
        self.tau_rel = assign_tau_rel
        
        # State
        self.hierarchy = None
        self.module_scores = None
        self.cell_assignments = None
        
    def fit(self, adata, layer=None):
        """
        Build the gene hierarchy from data.
        """
        # 1. Get Expression Data
        if layer:
            X = adata.layers[layer]
        else:
            X = adata.X
            
        gene_names = adata.var_names.tolist()
        
        # 2. Binarize
        print("Step 1: Binarizing expression data...")
        B = binarize_expression(X, threshold=self.bin_threshold)
        
        # 3. Compute Edges (Phi & Fisher)
        print("Step 2: Computing regulatory edges...")
        edges_df = compute_edge_statistics(
            B, 
            gene_names, 
            phi_threshold=self.phi_threshold, 
            fdr_alpha=self.fdr
        )
        
        # 4. Build Hierarchy
        print("Step 3: Constructing gene module hierarchy...")
        self.hierarchy = GeneHierarchy(edges_df)
        self.hierarchy.build()
        
        return self
        
    def transform(self, adata, layer=None):
        """
        Assign cells to the hierarchy.
        """
        if self.hierarchy is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        if layer:
            X = adata.layers[layer]
        else:
            X = adata.X
            
        gene_names = adata.var_names.tolist()
        
        # 1. Compute Z-scores
        print("Step 4: Computing Z-scores...")
        Z = compute_z_scores(X)
        
        # 2. Compute Module Activity
        print("Step 5: Computing module activity scores...")
        self.module_scores = calculate_module_activities(Z, gene_names, self.hierarchy.root)
        
        # 3. Assign Cells
        print("Step 6: Assigning cells to hierarchy...")
        assignments = assign_cells_hierarchical(
            self.module_scores,
            self.hierarchy.root,
            tau_abs=self.tau_abs,
            tau_rel=self.tau_rel
        )
        
        self.cell_assignments = assignments
        return assignments
    
    def fit_transform(self, adata, layer=None):
        self.fit(adata, layer)
        return self.transform(adata, layer)