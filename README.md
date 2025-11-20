# gecco
GeCCo (Gene Co-expression Constructed identity) is a mathematically rigorous framework for single-cell transcriptomics analysis that shifts the paradigm from ad hoc clustering to programmatic cell typing.

Unlike conventional cell-centric methods (e.g., Seurat, Scanpy) that measure distances in a fixed global feature space, GeCCo constructs cell identities as emergent entities anchored in pair-dependent Hilbert subspaces. By organizing gene-gene relationships into a topologically constrained hierarchy (based on Boolean regulatory logic), GeCCo guarantees hierarchical consistency and resolves subtle transitional states that global metrics often obscure.

🧬 Gene-Centric Philosophy: Identities are defined by the co-activation of specific gene programs, not just geometric proximity in a UMAP.
📐 Hilbert Subspace Metric: Dynamically selects the appropriate feature subspace for comparing any two cells, respecting the biological context of the comparison.
🌳 Topologically Constrained Hierarchy: Uses a greedy algorithm with strict regulatory constraints (Synergy vs. Antagonism) to build a conflict-free gene module tree.
🛡️ Robust to Batch Effects: By focusing on correlation structure ($\phi$-coefficient) rather than absolute expression magnitudes, GeCCo preserves identity definitions across batches and tissues.
🔌 Scikit-learn Compatible API: Designed to integrate seamlessly with scanpy and anndata workflows.

## Installation

Currently, GeCCo is in the research preview phase. You can install it directly from the source:

```bash
# Clone the repository
git clone [https://github.com/yourusername/GeCCo.git](https://github.com/yourusername/GeCCo.git)

# Navigate to the directory
cd GeCCo

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install .

