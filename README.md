# GeCCo

**Gene-First Identity Construction for Robust Cell Identification in Single-Cell Transcriptomics**

GeCCo is a gene-first hierarchical modeling framework for single-cell transcriptomics.  
It builds a gene-module tree from Boolean gene-gene coupling, then assigns cell identities on top of that hierarchy to improve global-local consistency. 

## Core Idea

1. Binarize expression profiles and compute `phi` correlation with Fisher test + BH correction.
2. Filter valid genes and build a hierarchical module tree using incremental insertion rules (R1-R4).
3. Assign cells into `depth1/depth2/...` labels based on module-level scores.
4. Output module structure, insertion trace, cell assignments, and visualizations.

## Installation

```bash
cd GeCCo
pip install -e .
```

## Quick Start

```python
import anndata as ad
from gecco import GeCCoConfig, GeCCoPipeline
from gecco.visualize import plot_module_tree, plot_node_genes

adata = ad.read_h5ad("datasets/adata_sim.h5ad")

config = GeCCoConfig(
    phi_threshold=0.3,
    fdr_threshold=0.05,
    max_depth=5,
)
result = GeCCoPipeline(config).run(adata)

print(result.metrics)
print(result.cell_assignments.head())

plot_module_tree(result.tree_root, "outputs/module_tree.png")
plot_node_genes(result.tree_root, "outputs/node_genes.png")
```

## Repository Structure

- `src/gecco/`: Core implementation (`preprocess`, `tree`, `pipeline`, `visualize`)
- `datasets/`: Example input data
- `outputs/`: Example figures
- `notebooks/`: Reproducible analysis and visualization notebooks


## Paper

`Gene-First Identity Construction for Robust Cell Identification in Single-Cell Transcriptomics`  
(Luqi Yang, Zhenwei Huang, Jinpu Cai, Hongyi Xin)
