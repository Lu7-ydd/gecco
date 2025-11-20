import uuid
import pandas as pd
import networkx as nx

class GeneNode:
    """
    Represents a node in the Gene Co-expression Hierarchy.
    Each node corresponds to a Hilbert subspace defined by its gene set.
    """
    def __init__(self, genes=None, node_id=None, parent=None):
        self.node_id = node_id if node_id else str(uuid.uuid4())[:8]
        self.genes = set(genes) if genes else set()
        self.children = []
        self.parent = parent

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def is_root(self):
        return self.parent is None

    def add_child(self, node):
        node.parent = self
        self.children.append(node)

    def remove_child(self, node):
        if node in self.children:
            self.children.remove(node)
            node.parent = None

    def __repr__(self):
        return f"Node(id={self.node_id}, genes={len(self.genes)}, children={len(self.children)})"


class GeneHierarchy:
    """
    Topologically-constrained Gene Hierarchy Construction.
    Implements the greedy insertion algorithm with rules R1-R4.
    """
    def __init__(self, edges_df):
        """
        Initialize with a DataFrame of significant edges.
        
        Parameters
        ----------
        edges_df : pd.DataFrame
            Must contain columns ['source', 'target', 'sign'].
            'sign' should be 1 (positive) or -1 (negative).
        """
        self.root = GeneNode(node_id="root")
        self.unassigned_genes = []
        self.graph_dict = self._build_adjacency(edges_df)
        self.all_genes = list(self.graph_dict.keys())
        
    def _build_adjacency(self, df):
        """Convert edge list to efficient nested dictionary for O(1) lookups."""
        adj = {}
        for _, row in df.iterrows():
            u, v, s = row['source'], row['target'], row['sign']
            if u not in adj: adj[u] = {}
            if v not in adj: adj[v] = {}
            adj[u][v] = s
            adj[v][u] = s
        return adj

    def get_relation(self, gene, node_genes):
        """
        Determine the relationship between a query gene and a set of genes.
        Strict logic based on (C1)-(C3).
        
        Returns:
        - 'positive': All known edges are positive.
        - 'negative': All known edges are negative.
        - 'mixed': Contains both positive and negative edges.
        - 'none': No significant edges found (neutral).
        """
        if not node_genes:
            return 'none'
            
        signs = []
        neighbors = self.graph_dict.get(gene, {})
        
        for target in node_genes:
            if target in neighbors:
                signs.append(neighbors[target])
        
        if not signs:
            return 'none'
        
        if all(s > 0 for s in signs):
            return 'positive'
        elif all(s < 0 for s in signs):
            return 'negative'
        else:
            return 'mixed'

    def build(self):
        """
        Execute the Greedy Topological Construction.
        """
        # 1. Sort genes by degree (heuristic: hubs define structure first)
        sorted_genes = sorted(
            self.all_genes, 
            key=lambda g: len(self.graph_dict.get(g, {})), 
            reverse=True
        )
        
        if not sorted_genes:
            return

        # 2. Initialization: Find the strongest positive pair to start
        # Note: In a real run, we just pick the first pair from the sorted list that is connected
        start_gene = sorted_genes[0]
        neighbors = self.graph_dict.get(start_gene, {})
        
        # Find a positive neighbor to form the first module
        partner = None
        for n, sign in neighbors.items():
            if sign > 0:
                partner = n
                break
        
        if partner:
            # Create first child node
            c1 = GeneNode(genes=[start_gene, partner], node_id="Init_Module")
            self.root.add_child(c1)
            remaining_genes = [g for g in sorted_genes if g not in [start_gene, partner]]
        else:
            # If no positive partner, start alone (Case 3 logic)
            c1 = GeneNode(genes=[start_gene], node_id="Init_Gene")
            self.root.add_child(c1)
            remaining_genes = sorted_genes[1:]

        print(f"Initialized tree with root children: {[n.node_id for n in self.root.children]}")
        print(f"Inserting {len(remaining_genes)} remaining genes...")

        # 3. Iterative Insertion
        for gene in remaining_genes:
            self._insert_recursive(self.root, gene)

    def _insert_recursive(self, u, gene):
        """
        Recursive insertion function implementing R1-R4.
        u: current node (GeneNode)
        gene: gene name (str)
        """
        children = u.children
        
        # --- Step 1: Classify Children ---
        P = [] # Purely Positive
        N = [] # Purely Negative
        M = [] # Mixed
        
        for child in children:
            rel = self.get_relation(gene, child.genes)
            if rel == 'positive':
                P.append(child)
            elif rel == 'negative':
                N.append(child)
            elif rel == 'mixed':
                M.append(child)
            # 'none' implies neutral, effectively treated as noise or weak negative in this context
        
        # --- Step 2: Apply Rules ---

        # R1: Absorption (Specificity)
        # Condition: Exactly one positive child, no mixed confusion.
        if len(P) == 1 and len(M) == 0:
            target = P[0]
            # Check if we should just add to leaf or recurse
            if target.is_leaf:
                target.genes.add(gene)
                return
            else:
                self._insert_recursive(target, gene)
                return

        # R2: Parent Creation (Synergy)
        # Condition: Bridges multiple positive siblings.
        if len(P) >= 2 and len(M) == 0:
            # Create new intermediate parent v
            new_node = GeneNode(genes=[gene], node_id=f"Node_{gene}")
            u.add_child(new_node)
            
            # Re-attach P children to new_node
            for child in P:
                u.remove_child(child)
                new_node.add_child(child)
            return

        # R3: New Sibling (Antagonism / Novelty) -> REVISED LOGIC
        # Condition: Antagonistic (or neutral) to all existing lineages.
        if len(P) == 0 and len(M) == 0:
            # Gene represents a distinct lineage at this level
            new_leaf = GeneNode(genes=[gene], node_id=f"Leaf_{gene}")
            u.add_child(new_leaf)
            return

        # R4: Bifurcation (Resolution)
        # Condition: Conflicts exist (Mixed).
        if len(M) > 0:
            # Try to resolve first mixed conflict
            # We prioritize leaf splitting. If non-leaf is mixed, we recurse.
            
            target_mixed = M[0] 
            
            if target_mixed.is_leaf:
                self._split_leaf(u, target_mixed, gene)
                return
            else:
                # If it's not a leaf, the conflict might be resolved deeper down.
                # We force recursion into the mixed node.
                self._insert_recursive(target_mixed, gene)
                return

        # Case 5: Fallback (Should strictly be covered by Case 3 in revised logic, 
        # but kept for safety if strict N check is used)
        self.unassigned_genes.append(gene)

    def _split_leaf(self, parent, node, splitter_gene):
        """
        Helper for R4: Splits a leaf node based on relationship with splitter_gene.
        """
        pos_genes = []
        neg_genes = []
        neutral_genes = []

        neighbors = self.graph_dict.get(splitter_gene, {})

        for g in node.genes:
            sign = neighbors.get(g, 0)
            if sign > 0:
                pos_genes.append(g)
            elif sign < 0:
                neg_genes.append(g)
            else:
                neutral_genes.append(g)
        
        # Only split if we have valid separation
        if pos_genes and neg_genes:
            # Create new parent container for the split
            new_parent = GeneNode(genes=[splitter_gene], node_id=f"Node_{splitter_gene}_split")
            parent.add_child(new_parent)
            
            # Create children
            child_pos = GeneNode(genes=pos_genes + neutral_genes, node_id=f"{node.node_id}_pos") # Neutral follows pos? Or separate?
            child_neg = GeneNode(genes=neg_genes, node_id=f"{node.node_id}_neg")
            
            new_parent.add_child(child_pos)
            new_parent.add_child(child_neg)
            
            # Remove original
            parent.remove_child(node)
        else:
            # Split failed (e.g. only neutral vs pos), treat as absorption or noise
            # For simplicity, just add as noise if split fails
            self.unassigned_genes.append(splitter_gene)

    def to_networkx(self):
        """Convert internal tree to NetworkX DiGraph for plotting."""
        G = nx.DiGraph()
        
        def add_nodes(node):
            # Label is top 3 genes
            label = ",".join(list(node.genes)[:3])
            G.add_node(node.node_id, label=label, level=len(node.children))
            for child in node.children:
                G.add_edge(node.node_id, child.node_id)
                add_nodes(child)
                
        add_nodes(self.root)
        return G