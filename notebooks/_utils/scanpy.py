import scanpy as sc

def scanpy_pp(adata, n_hvg = 2000, use_hvg = True, n_neighbors = 10, n_pcs = 4):
    print("---begin normalization---")
    sc.pp.normalize_total(adata, target_sum=1e4)
    print("---begin log---")
    sc.pp.log1p(adata)
    print("---begin hvg---")
    sc.pp.highly_variable_genes(
        adata,
        layer="counts",
        n_top_genes=n_hvg,
        min_mean=0.0125,
        max_mean=3,
        min_disp=0.5,
        flavor="seurat_v3",
    )
    print("---begin pca---")
    sc.pp.pca(
        adata,
        svd_solver='arpack',
        use_highly_variable = use_hvg, # in scanpy, default is None, which equivalent to True if hvg is calculated, and False otherwise
    )
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)