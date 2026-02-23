from __future__ import annotations

import copy
import warnings

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable

import numpy as np

from .config import GeCCoConfig


@dataclass
class ModuleNode:
    node_id: str
    genes: set[str] = field(default_factory=set)
    children: list["ModuleNode"] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def total_genes(self) -> set[str]:
        genes = set(self.genes)
        for child in self.children:
            genes.update(child.total_genes())
        return genes

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "genes": sorted(self.genes),
            "children": [c.to_dict() for c in self.children],
        }


@dataclass(slots=True)
class TreeBuildResult:
    root: ModuleNode
    inserted_genes: list[str]
    noise_genes: list[str]
    insertion_trace: list[dict[str, object]]


class ModuleTreeBuilder:
    """
    Incremental insertion tree builder with R1/R2/R3/R4 rules.
    """

    def __init__(
        self,
        config: GeCCoConfig,
        gene_names: np.ndarray,
        phi: np.ndarray,
        fdr: np.ndarray,
        connectivity: np.ndarray,
        valid_mask: np.ndarray,
        n_expressed: np.ndarray,
    ):
        self.config = config
        self.gene_names = np.asarray(gene_names)
        self.phi = phi
        self.fdr = fdr
        self.connectivity = connectivity
        self.valid_mask = valid_mask
        # n_expressed: number of cells each gene is expressed in (binary_expr.sum(axis=0))
        self.n_expressed = np.asarray(n_expressed)
        self.gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}
        self._node_counter = 0
        self._current_step: int = 0   # updated each insertion in build()
        self._total_genes: int = 1    # updated at start of build()

    # ------------------------------------------------------------------ helpers

    def _next_node_id(self) -> str:
        self._node_counter += 1
        return f"M{self._node_counter}"

    def _idx(self, gene: str) -> int:
        return self.gene_to_idx[gene]

    def _phi(self, g1: str, g2: str) -> float:
        return float(self.phi[self._idx(g1), self._idx(g2)])

    def _fdr(self, g1: str, g2: str) -> float:
        return float(self.fdr[self._idx(g1), self._idx(g2)])

    def _connectivity(self, gene: str) -> int:
        return int(self.connectivity[self._idx(gene)])

    def _record_changed_nodes(self, trace: dict, node_ids: Iterable[str]) -> None:
        changed = trace.setdefault("changed_nodes", [])
        for node_id in node_ids:
            if node_id and node_id not in changed:
                changed.append(node_id)

    def _record_split_info(self, trace: dict, split_info: dict[str, object]) -> None:
        trace["split_parent_id"] = split_info.get("split_parent_id", "-")
        trace["split_child_ids"] = split_info.get("split_child_ids", [])
        trace["reassigned_existing_genes"] = split_info.get("reassigned_existing_genes", [])

    def _mean_phi(self, genes_a: Iterable[str], genes_b: Iterable[str]) -> float:
        a, b = list(genes_a), list(genes_b)
        if not a or not b:
            return 0.0
        ia = [self._idx(g) for g in a]
        ib = [self._idx(g) for g in b]
        return float(np.mean(self.phi[np.ix_(ia, ib)]))

    def _internal_phi(self, genes: Iterable[str]) -> float:
        g = list(genes)
        if len(g) < 2:
            return 1.0
        idx = [self._idx(x) for x in g]
        mat = self.phi[np.ix_(idx, idx)]
        triu = mat[np.triu_indices_from(mat, k=1)]
        return float(np.mean(triu)) if triu.size > 0 else 1.0

    @staticmethod
    def _node_ids(nodes: list[ModuleNode]) -> str:
        return ",".join(n.node_id for n in nodes) if nodes else "-"

    @staticmethod
    def _module_genes(node: ModuleNode) -> set[str]:
        """Direct genes if non-empty, else fall back to subtree genes."""
        return node.genes if node.genes else node.total_genes()

    # ---------------------------------------------------------- P/N/M classifier

    def _classify(self, gene: str, module_genes: Iterable[str]) -> str:
        """
        Classify relationship between a gene and a module as P / N / M.
        P = predominantly positive correlation
        N = predominantly negative correlation
        M = mixed / unclear
        """
        genes = list(module_genes)
        if not genes:
            return "M"

        vals = np.array([self._phi(gene, g) for g in genes], dtype=np.float64)
        mean_val = float(np.mean(vals))
        pos_frac = float(np.mean(vals > 0))
        neg_frac = float(np.mean(vals < 0))

        # Adaptive threshold: earlier insertions are stricter, later ones more lenient.
        # At step 0: threshold = phi_threshold. At last step: threshold → 0.
        progress = self._current_step / max(self._total_genes - 1, 1)  # 0.0 → 1.0
        adaptive_thr = self.config.phi_threshold * (1.0 - progress)

        # Clear positive
        if mean_val >= adaptive_thr and pos_frac >= 0.66 and neg_frac <= 0.20:
            return "P"

        # Clear negative
        if mean_val <= -adaptive_thr and neg_frac >= 0.66 and pos_frac <= 0.20:
            return "N"

        # Mixed: noticeable fractions on both sides — must check BEFORE W
        # (mean ≈ 0 with high pos_frac AND high neg_frac is M, not W)
        if pos_frac >= 0.06 and neg_frac >= 0.06:
            return "M"

        # Weak: mean near zero AND not strongly split → essentially unrelated
        if abs(mean_val) < self.config.phi_threshold:
            return "W"

        # Residual directional signal with mean above threshold
        return "P" if mean_val > 0 else "N"

    # Alias for backward compatibility with tests
    def _classify_gene_module_relation(self, gene: str, module_genes: Iterable[str]) -> str:
        return self._classify(gene, module_genes)

    def _classify_children(
        self, node: ModuleNode, gene: str
    ) -> tuple[list[ModuleNode], list[ModuleNode], list[ModuleNode], list[ModuleNode]]:
        p_cls, n_cls, m_cls, w_cls = [], [], [], []
        for child in node.children:
            rel = self._classify(gene, self._module_genes(child))
            if rel == "P":
                p_cls.append(child)
            elif rel == "N":
                n_cls.append(child)
            elif rel == "M":
                m_cls.append(child)
            else:  # W
                w_cls.append(child)
        return p_cls, n_cls, m_cls, w_cls

    # ---------------------------------------------------------- split helpers

    def _adaptive_thr(self) -> float:
        """Threshold that shrinks linearly from phi_threshold → 0 as insertion progresses."""
        progress = self._current_step / max(self._total_genes - 1, 1)
        return self.config.phi_threshold * (1.0 - progress)

    def _find_antagonistic_pair(self, genes: list[str]) -> tuple[str, str] | None:
        """Find the pair with the most negative phi (prefer FDR-significant pairs first)."""
        def search(require_fdr: bool) -> tuple[str, str] | None:
            best_pair, best_phi = None, 0.0
            for g1, g2 in combinations(genes, 2):
                p = self._phi(g1, g2)
                if require_fdr and self._fdr(g1, g2) > self.config.fdr_threshold:
                    continue
                if p < best_phi:
                    best_phi, best_pair = p, (g1, g2)
            return best_pair
        return search(require_fdr=True) or search(require_fdr=False)

    def _try_split_leaf(
        self, leaf: ModuleNode, trigger_gene: str | None = None
    ) -> tuple[bool, dict | None]:
        """
        Attempt to split a leaf into two children using its most antagonistic gene pair.

        Steps:
        1. Find the most antagonistic pair (seed_a, seed_b) in leaf.genes.
        2. Assign every other gene to whichever seed it is more correlated with;
           genes positively correlated with both seeds go to the backbone (leaf.genes).
        3. Accept the split if each side has >= min_module_size genes and their
           mean cross-phi is <= adaptive_thr (i.e. the two sides are distinct enough).

        Returns (True, split_info) on success, (False, None) on failure.
        Modifies leaf in place on success.
        """
        if not leaf.is_leaf() or len(leaf.genes) < 2 * self.config.min_module_size:
            return False, None

        pair = self._find_antagonistic_pair(sorted(leaf.genes))
        if pair is None:
            return False, None
        seed_a, seed_b = pair

        thr = self._adaptive_thr()
        side_a, side_b, backbone = {seed_a}, {seed_b}, set()
        for g in leaf.genes:
            if g in (seed_a, seed_b):
                continue
            s_a, s_b = self._phi(g, seed_a), self._phi(g, seed_b)
            if s_a >= thr and s_b >= thr:
                backbone.add(g)          # positive to both → shared backbone
            elif s_a >= s_b:
                side_a.add(g)
            else:
                side_b.add(g)

        # Accept only if both sides are big enough and cross-phi is non-positive
        if len(side_a) < self.config.min_module_size or len(side_b) < self.config.min_module_size:
            return False, None
        if self._mean_phi(side_a, side_b) > 0:
            return False, None

        # Commit the split
        leaf.genes = backbone
        child_a = ModuleNode(node_id=self._next_node_id(), genes=side_a)
        child_b = ModuleNode(node_id=self._next_node_id(), genes=side_b)
        leaf.children = [child_a, child_b]

        reassigned = sorted((side_a | side_b) - ({trigger_gene} if trigger_gene else set()))
        split_info = {
            "split_parent_id": leaf.node_id,
            "split_child_ids": [child_a.node_id, child_b.node_id],
            "reassigned_existing_genes": reassigned,
        }
        return True, split_info

    # ------------------------------------------------------------------ rules

    def _apply_r2(self, parent: ModuleNode, gene: str, p_children: list[ModuleNode]) -> ModuleNode:
        """R2: Group multiple positive children under a new intermediate parent."""
        p_ids = {id(c) for c in p_children}
        moved = [c for c in parent.children if id(c) in p_ids]
        new_parent = ModuleNode(node_id=self._next_node_id(), genes={gene}, children=moved)
        parent.children = [c for c in parent.children if id(c) not in p_ids] + [new_parent]
        return new_parent

    def _apply_r3(self, parent: ModuleNode, gene: str) -> ModuleNode:
        """R3: Add a new sibling node when the gene is antagonistic to all existing children."""
        new_node = ModuleNode(node_id=self._next_node_id(), genes={gene})
        parent.children.append(new_node)
        return new_node

    def _apply_r4(self, m_children: list[ModuleNode], gene: str, trace: dict | None = None) -> ModuleNode | None:
        """R4: Split a mixed leaf node triggered by a new gene that straddles the module."""
        for child in [c for c in m_children if c.is_leaf()]:
            child.genes.add(gene)
            ok, split_info = self._try_split_leaf(child, trigger_gene=gene)
            if ok:
                if trace is not None and split_info is not None:
                    self._record_changed_nodes(trace, [child.node_id, *split_info.get("split_child_ids", [])])
                    self._record_split_info(trace, split_info)
                target = next((sub for sub in child.children if gene in sub.genes), child)
                return target
            child.genes.discard(gene)
        return None

    # ---------------------------------------------------------- insertion

    def _insert_recursive(self, current: ModuleNode, gene: str, depth: int, trace: dict) -> bool:
        if depth >= self.config.max_depth:
            trace["rule"] = "MAX_DEPTH_REJECT"
            trace["target_node_id"] = current.node_id
            return False

        # Leaf node: absorb directly, then try splitting if large enough
        if not current.children:
            current.genes.add(gene)
            self._record_changed_nodes(trace, [current.node_id])
            if len(current.genes) >= 2 * self.config.min_module_size:
                did_split, split_info = self._try_split_leaf(current, trigger_gene=gene)
                if did_split:
                    target = next((sub for sub in current.children if gene in sub.genes), current)
                    trace["rule"] = "R1_LEAF_ABSORB_THEN_SPLIT"
                    trace["target_node_id"] = target.node_id
                    if split_info is not None:
                        self._record_changed_nodes(trace, split_info.get("split_child_ids", []))
                        self._record_split_info(trace, split_info)
                    return True
            trace["rule"] = "R1_LEAF_ABSORB"
            trace["target_node_id"] = current.node_id
            return True

        p_cls, n_cls, m_cls, w_cls = self._classify_children(current, gene)
        trace["path"].append(
            f"{current.node_id}|P:{self._node_ids(p_cls)}|N:{self._node_ids(n_cls)}"
            f"|M:{self._node_ids(m_cls)}|W:{self._node_ids(w_cls)}"
        )

        # R1: unique positive child → always recurse into it
        if len(p_cls) == 1:
            target = p_cls[0]
            trace["path"].append(f"R1->{target.node_id}")
            return self._insert_recursive(target, gene, depth + 1, trace)

        # R2: multiple positive children → group under intermediate parent
        # Also triggers when all children are weak (W), treating them as loosely positive
        r2_targets = p_cls if (len(p_cls) >= 2 and not m_cls) else (w_cls if (len(w_cls) >= 2 and not p_cls and not n_cls) else [])
        if r2_targets:
            if len(r2_targets) == len(current.children):
                current.genes.add(gene)
                trace["rule"] = "R2_ALL_P_ABSORB_PARENT"
                trace["target_node_id"] = current.node_id
                self._record_changed_nodes(trace, [current.node_id])
            else:
                new_parent = self._apply_r2(current, gene, r2_targets)
                trace["rule"] = "R2_CREATE_INTERMEDIATE_PARENT"
                trace["target_node_id"] = new_parent.node_id
                self._record_changed_nodes(trace, [current.node_id, new_parent.node_id, *[c.node_id for c in r2_targets]])
            return True

        # R3: antagonistic to all children (no P, no M, no W) → add new sibling
        if not p_cls and not m_cls and not w_cls:
            r3_node = self._apply_r3(current, gene)
            trace["rule"] = "R3_CREATE_NEW_SIBLING"
            trace["target_node_id"] = r3_node.node_id
            self._record_changed_nodes(trace, [current.node_id, r3_node.node_id])
            return True

        # R4: mixed leaf → try splitting
        r4_target = self._apply_r4([c for c in m_cls if c.is_leaf()], gene, trace=trace)
        if r4_target is not None:
            trace["rule"] = "R4_SPLIT_LEAF_AND_INSERT"
            trace["target_node_id"] = r4_target.node_id
            return True

        # Fallback: insert into the most similar child among P/M/W
        # W children are candidates but lower priority than P/M
        candidates = p_cls + m_cls + w_cls or n_cls
        best = max(candidates, key=lambda c: self._mean_phi([gene], self._module_genes(c)))
        best_score = self._mean_phi([gene], self._module_genes(best))
        if best_score >= self.config.phi_threshold / 3:
            best.genes.add(gene)
            trace["rule"] = "FALLBACK_BEST_CHILD"
            trace["target_node_id"] = best.node_id
            self._record_changed_nodes(trace, [best.node_id])
            return True

        trace["rule"] = "NO_RULE_MATCH_REJECT"
        trace["target_node_id"] = current.node_id
        return False

    # ---------------------------------------------------------- constraints

    def _check_c1(self, node: ModuleNode) -> bool:
        """C1: A node's direct genes must be internally coherent (no mutual antagonism)."""
        genes = list(node.genes)
        if len(genes) >= 2 and self._internal_phi(genes) < 0:
            return False
        return all(self._check_c1(c) for c in node.children)

    def _check_c2(self, node: ModuleNode) -> bool:
        """C2: Sibling nodes' direct genes must not be too similar to each other."""
        for a, b in combinations(node.children, 2):
            if a.genes and b.genes and self._mean_phi(a.genes, b.genes) > 0:
                return False
        return all(self._check_c2(c) for c in node.children)

    def _check_c3(self, node: ModuleNode) -> bool:
        """C3: A child's direct genes must not antagonize the parent's direct genes.
        Parent backbone genes are large-class markers; child direct genes (subtype-specific)
        should correlate positively with them, not oppose them.
        """
        if node.genes:
            for child in node.children:
                if child.genes and self._mean_phi(child.genes, node.genes) < 0:
                    return False
        return all(self._check_c3(c) for c in node.children)

    def _check_constraints(self, root: ModuleNode) -> tuple[bool, bool, bool]:
        return self._check_c1(root), self._check_c2(root), self._check_c3(root)

    # ---------------------------------------------------------- initialization

    def _pick_initial_triplet(self, valid_genes: list[str]) -> tuple[str, str, str]:
        """
        Pick three seed genes:
        anchor = highest connectivity gene
        pos    = best significantly positive partner to anchor
        neg    = best significantly negative partner to anchor
        """
        anchor = max(valid_genes, key=self._connectivity)
        others = [g for g in valid_genes if g != anchor]

        def best_positive(exclude: set[str]) -> str:
            cands = [g for g in others if g not in exclude]
            sig = [g for g in cands if self._fdr(anchor, g) <= self.config.fdr_threshold and self._phi(anchor, g) >= self.config.phi_threshold]
            return max(sig or cands, key=lambda g: self._phi(anchor, g))

        def best_negative(exclude: set[str]) -> str:
            cands = [g for g in others if g not in exclude]
            sig = [g for g in cands if self._fdr(anchor, g) <= self.config.fdr_threshold and self._phi(anchor, g) <= -self.config.phi_threshold]
            return min(sig or cands, key=lambda g: self._phi(anchor, g))

        pos = best_positive({anchor})
        neg = best_negative({anchor, pos})
        return anchor, pos, neg

    # ------------------------------------------------------------------ build

    def build(self) -> TreeBuildResult:
        valid_genes = self.gene_names[self.valid_mask].tolist()
        # Sort by number of cells expressing the gene (descending):
        # broadly expressed genes tend to represent major cell-type markers
        # and should be inserted first to form the coarse tree skeleton.
        valid_genes.sort(key=lambda g: int(self.n_expressed[self._idx(g)]), reverse=True)
        self._total_genes = len(valid_genes)
        self._current_step = 0
        traces: list[dict] = []
        root = ModuleNode(node_id="root")

        if len(valid_genes) == 0:
            return TreeBuildResult(root=root, inserted_genes=[], noise_genes=[], insertion_trace=traces)

        # Seed with anchor (backbone of root) + positive child + negative child
        # Each child starts with 3 genes: the seed + 2 most correlated partners
        anchor, pos, neg = self._pick_initial_triplet(valid_genes)
        used = {anchor, pos, neg}

        def pick_partners(seed: str, n: int) -> list[str]:
            """Pick n genes most positively correlated with seed from unused genes."""
            cands = [g for g in valid_genes if g not in used]
            ranked = sorted(cands, key=lambda g: self._phi(seed, g), reverse=True)
            chosen = ranked[:n]
            used.update(chosen)
            return chosen

        pos_partners = pick_partners(pos, 3)
        neg_partners = pick_partners(neg, 3)

        # anchor goes into whichever child it is most positively correlated with
        anchor_node_label = "INIT_ANCHOR_POS" if self._phi(anchor, pos) >= self._phi(anchor, neg) else "INIT_ANCHOR_NEG"
        pos_node = ModuleNode(node_id=self._next_node_id(), genes={pos, *pos_partners})
        neg_node = ModuleNode(node_id=self._next_node_id(), genes={neg, *neg_partners})
        anchor_node = pos_node if anchor_node_label == "INIT_ANCHOR_POS" else neg_node
        anchor_node.genes.add(anchor)
        root.children = [pos_node, neg_node]

        inserted = [anchor, pos, *pos_partners, neg, *neg_partners]
        init_steps = (
            [(anchor, anchor_node_label, anchor_node)]
            + [(pos, "INIT_POS", pos_node)]
            + [(g, "INIT_POS_PARTNER", pos_node) for g in pos_partners]
            + [(neg, "INIT_NEG", neg_node)]
            + [(g, "INIT_NEG_PARTNER", neg_node) for g in neg_partners]
        )
        for step, (g, rule, node) in enumerate(init_steps):
            traces.append({
                "step": step, "gene": g, "gene_index": self._idx(g),
                "connectivity": self._connectivity(g), "rule": rule,
                "target_node_id": node.node_id, "path": "root",
                "accepted": True, "is_noise": False,
                "changed_nodes": [node.node_id],
                "split_parent_id": "-",
                "split_child_ids": [],
                "reassigned_existing_genes": [],
            })

        noise: list[str] = []
        inserted_set = set(inserted)
        remaining = [g for g in valid_genes if g not in inserted_set]

        for i, gene in enumerate(remaining, start=len(inserted)):
            self._current_step = i
            trace: dict = {
                "step": i, "gene": gene, "gene_index": self._idx(gene),
                "connectivity": self._connectivity(gene), "rule": "",
                "target_node_id": "", "path": [], "accepted": False,
                "is_noise": False,
                "changed_nodes": [],
                "split_parent_id": "-",
                "split_child_ids": [],
                "reassigned_existing_genes": [],
            }
            snapshot = copy.deepcopy(root)
            ok = self._insert_recursive(root, gene, depth=1, trace=trace)

            # if ok:
            #     c1, c2, c3 = self._check_constraints(root)
            #     if c1 and c2 and c3:
            #         inserted.append(gene)
            #         trace["accepted"], trace["is_noise"] = True, False
            #     else:
            #         # Constraint violated: roll back and treat as noise
            #         root = snapshot
            #         noise.append(gene)
            #         trace["accepted"], trace["is_noise"] = False, True
            #         trace["rule"] = ("C1_VIOLATED" if not c1 else "C2_VIOLATED" if not c2 else "C3_VIOLATED")
            # else:
            #     noise.append(gene)
            #     trace["accepted"], trace["is_noise"] = False, True

            trace["path"] = " -> ".join(trace["path"]) if isinstance(trace["path"], list) else str(trace["path"])
            trace["rule"] = trace["rule"] or "UNSET"
            trace["target_node_id"] = trace["target_node_id"] or "-"
            traces.append(trace)

        return TreeBuildResult(root=root, inserted_genes=inserted, noise_genes=noise, insertion_trace=traces)


# ------------------------------------------------------------------ utilities

def nodes_at_depth(root: ModuleNode, depth: int) -> list[ModuleNode]:
    if depth < 1:
        return []
    current = [root]
    for _ in range(depth - 1):
        current = [child for node in current for child in node.children]
        if not current:
            break
    return current


def iter_nodes(root: ModuleNode) -> Iterable[ModuleNode]:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


# ══════════════════════════════════════════════════════════════════════════════
# Post-processing: reshape topology into a fixed-depth tree
# ══════════════════════════════════════════════════════════════════════════════

class _PostProcessor:

    def __init__(
        self,
        phi: np.ndarray,
        gene_names: list[str],
        phi_threshold: float,
        pos_pair_frac: float,
        id_offset: int = 0,
    ):
        self.phi = phi
        self.gene_names = gene_names
        self.gene_to_idx: dict[str, int] = {g: i for i, g in enumerate(gene_names)}
        self.phi_threshold = phi_threshold
        self.pos_pair_frac = pos_pair_frac
        self._counter = id_offset
        # backbone thresholds — may be overridden by post_process() before build()
        self._backbone_thr_d1: float = 0.0
        self._backbone_thr_d2: float = 0

    # ── id helpers ────────────────────────────────────────────────────────

    def _next_id(self) -> str:
        self._counter += 1
        return f"M{self._counter}"

    # ── phi helpers ───────────────────────────────────────────────────────

    def _idx(self, g: str) -> int:
        return self.gene_to_idx[g]

    def _phi_val(self, g1: str, g2: str) -> float:
        return float(self.phi[self._idx(g1), self._idx(g2)])

    def _mean_phi(self, a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        ia = [self._idx(g) for g in a]
        ib = [self._idx(g) for g in b]
        return float(np.mean(self.phi[np.ix_(ia, ib)]))

    def _pos_pair_frac(self, genes: set[str]) -> float:
        """Fraction of off-diagonal gene pairs with phi > 0."""
        g = list(genes)
        if len(g) < 2:
            return 1.0
        idx = [self._idx(x) for x in g]
        mat = self.phi[np.ix_(idx, idx)]
        triu = mat[np.triu_indices_from(mat, k=1)]
        return float(np.mean(triu > 0)) if triu.size > 0 else 1.0

    # ── seed selection ────────────────────────────────────────────────────

    def _pick_k_seeds(self, genes: list[str], k: int) -> list[str]:
        """
        Greedily pick k seeds that are maximally mutually antagonistic.
        seed_1/2 = most negatively correlated pair;
        seed_i   = gene with smallest mean_phi to all previously chosen seeds.
        """
        if k == 1 or not genes:
            return genes[:k]

        best_pair: tuple[str, str] | None = None
        best_v = np.inf
        for g1, g2 in combinations(genes, 2):
            v = self._phi_val(g1, g2)
            if v < best_v:
                best_v, best_pair = v, (g1, g2)

        seeds: list[str] = list(best_pair) if best_pair else genes[:2]
        remaining = [g for g in genes if g not in seeds]

        while len(seeds) < k and remaining:
            scores = {
                g: float(np.mean([self._phi_val(g, s) for s in seeds]))
                for g in remaining
            }
            nxt = min(scores, key=scores.__getitem__)
            seeds.append(nxt)
            remaining.remove(nxt)

        return seeds

    # ── k-way split ───────────────────────────────────────────────────────

    def _split_k_way(
        self, genes: set[str], k: int, max_iter: int = 30, backbone_thr: float = 0.0
    ) -> tuple[list[set[str]], set[str]]:
        """
        Partition genes into k subtype groups + a backbone set.

        A gene is moved to backbone if its mean_phi with every OTHER group
        exceeds backbone_thr.  Use backbone_thr=0.0 (default) for a loose
        "not antagonistic to any group" criterion; use a higher value for a
        stricter "positively correlated with every group" criterion.

        Returns
        -------
        groups   : list[set[str]]  — k subtype-specific gene sets
        backbone : set[str]        — shared marker genes (goes to parent node)
        """
        gene_list = sorted(genes)  # deterministic order
        if len(gene_list) < k:
            groups = [set(gene_list)] + [set() for _ in range(k - 1)]
            return groups, set()

        seeds = self._pick_k_seeds(gene_list, k)

        # Initial assignment: each gene → group of most correlated seed
        assignment: dict[str, int] = {}
        for g in gene_list:
            scores = [self._phi_val(g, s) for s in seeds]
            assignment[g] = int(np.argmax(scores))

        # Iterative refinement (k-means on phi rows)
        for _ in range(max_iter):
            centroids: list[np.ndarray | None] = []
            for ki in range(k):
                members = [g for g, a in assignment.items() if a == ki]
                if members:
                    idxs = [self._idx(g) for g in members]
                    centroids.append(self.phi[idxs, :].mean(axis=0))
                else:
                    centroids.append(None)

            new_assignment: dict[str, int] = {}
            for g in gene_list:
                gi = self._idx(g)
                best_k, best_score = 0, -np.inf
                for ki, centroid in enumerate(centroids):
                    if centroid is None:
                        continue
                    score = float(centroid[gi])
                    if score > best_score:
                        best_score, best_k = score, ki
                new_assignment[g] = best_k

            if new_assignment == assignment:
                break
            assignment = new_assignment

        # Build groups
        groups: list[set[str]] = [set() for _ in range(k)]
        for g, ki in assignment.items():
            groups[ki].add(g)

        # Extract backbone: genes whose mean_phi with every OTHER group > backbone_thr.
        # Guard: only move a gene to backbone if its group will still have >= 1 gene left.
        backbone: set[str] = set()
        for ki in range(k):
            other_groups = [groups[j] for j in range(k) if j != ki]
            to_remove: set[str] = set()
            for g in list(groups[ki]):
                if len(groups[ki]) - len(to_remove) <= 1:
                    break  # keep at least 1 gene in this group
                if all(
                    self._mean_phi({g}, og) > backbone_thr
                    for og in other_groups if og
                ):
                    backbone.add(g)
                    to_remove.add(g)
            groups[ki] -= to_remove

        return groups, backbone

    # ── constraint checks (warnings only) ────────────────────────────────

    # def _warn_siblings(self, groups: list[set[str]], label: str) -> None:
    #     for (i, a), (j, b) in combinations(enumerate(groups), 2):
    #         v = self._mean_phi(a, b)
    #         if v > self.phi_threshold:
    #             warnings.warn(
    #                 f"[post_process] {label}: sibling groups {i}&{j} "
    #                 f"mean_phi={v:.3f} > {self.phi_threshold} (C2)",
    #                 stacklevel=4,
    #             )

    # def _warn_backbone_children(self, backbone: set[str], groups: list[set[str]], label: str) -> None:
    #     if not backbone:
    #         return
    #     for i, child in enumerate(groups):
    #         if not child:
    #             continue
    #         v = self._mean_phi(backbone, child)
    #         if v < 0:
    #             warnings.warn(
    #                 f"[post_process] {label}: backbone vs child {i} "
    #                 f"mean_phi={v:.3f} < 0 (C3)",
    #                 stacklevel=4,
    #             )

    # def _warn_coherence(self, genes: set[str], label: str) -> None:
    #     if len(genes) < 2:
    #         return
    #     frac = self._pos_pair_frac(genes)
    #     if frac < self.pos_pair_frac:
    #         warnings.warn(
    #             f"[post_process] {label}: pos_pair_frac={frac:.3f} "
    #             f"< {self.pos_pair_frac} (C1/C4)",
    #             stacklevel=4,
    #         )

    # ── coherence pruning ──────────────────────────────────────────────────

    def _prune_to_coherence(self, genes: set[str]) -> tuple[set[str], set[str]]:
        """
        Iteratively remove the gene with the lowest mean internal phi
        until pos_pair_frac >= self.pos_pair_frac or only 1 gene remains.

        Returns
        -------
        kept   : genes that survive pruning
        pruned : genes that were removed (to be treated as noise)
        """
        kept = set(genes)
        pruned: set[str] = set()
        while len(kept) >= 2 and self._pos_pair_frac(kept) < self.pos_pair_frac:
            # find the gene with the lowest mean phi to all other kept genes
            worst = min(
                kept,
                key=lambda g: float(np.mean(
                    [self._phi_val(g, other) for other in kept if other != g]
                )) if len(kept) > 1 else 0.0,
            )
            kept.discard(worst)
            pruned.add(worst)
        return kept, pruned

    # ── main build ────────────────────────────────────────────────────────

    def build(self, n_depth1: int, n_depth2: int | list[int]) -> ModuleNode:
        all_genes: set[str] = set(self.gene_names)
        root = ModuleNode(node_id="root")

        # Step 1: split all genes into n_depth1 groups
        # backbone_thr=0: any gene not antagonistic to any group → root backbone
        depth1_groups, root_backbone = self._split_k_way(
            all_genes, k=n_depth1, backbone_thr=self._backbone_thr_d1
        )
        root.genes = root_backbone  # residual — no coherence constraint

        # Sibling check at depth-1 (use each group's total genes)
        # self._warn_siblings(depth1_groups, label="depth-1")

        # Normalise n_depth2 to a per-group list
        if isinstance(n_depth2, int):
            n_depth2_list = [n_depth2] * n_depth1
        else:
            if len(n_depth2) != n_depth1:
                raise ValueError(
                    f"n_depth2 list length ({len(n_depth2)}) must equal n_depth1 ({n_depth1})"
                )
            n_depth2_list = list(n_depth2)

        # Step 2: for each depth-1 group, split into n_depth2 sub-groups
        for d1_genes, k2 in zip(depth1_groups, n_depth2_list):
            d1_node = ModuleNode(node_id=self._next_id())

            if not d1_genes or k2 <= 1:
                # k2 <= 1 means caller will attach depth-2 children later (local phi)
                d1_node.genes = d1_genes
                root.children.append(d1_node)
                continue

            # backbone_thr = phi_threshold * 0.3: slightly stricter than step1
            # (must have a weak but positive correlation with every sibling group)
            depth2_groups, d1_backbone = self._split_k_way(
                d1_genes, k=k2, backbone_thr=self._backbone_thr_d2
            )
            d1_node.genes = d1_backbone

            label = d1_node.node_id
            # self._warn_backbone_children(d1_backbone, depth2_groups, label)
            # self._warn_siblings(depth2_groups, label=f"{label} depth-2")

            for d2_genes in depth2_groups:
                d2_node = ModuleNode(node_id=self._next_id(), genes=d2_genes)
                d1_node.children.append(d2_node)

            root.children.append(d1_node)

        # ── Step 3: prune incoherent genes out of every node ──────────────
        # Pass 1: depth-2 leaf nodes
        noise_genes: set[str] = set()
        for d1_node in root.children:
            for d2_node in d1_node.children:
                if len(d2_node.genes) >= 2:
                    d2_node.genes, pruned = self._prune_to_coherence(d2_node.genes)
                    noise_genes.update(pruned)

        # Pass 2: depth-1 backbone genes
        for d1_node in root.children:
            if len(d1_node.genes) >= 2:
                d1_node.genes, pruned = self._prune_to_coherence(d1_node.genes)
                noise_genes.update(pruned)

        # Attach noise as a flat list on root for callers that want it
        root._noise_from_pruning = sorted(noise_genes)  # type: ignore[attr-defined]

        return root


def post_process(
    phi: np.ndarray,
    gene_names: list[str] | np.ndarray,
    n_depth1: int = 3,
    n_depth2: int | list[int] = 3,
    phi_threshold: float = 0.1,
    pos_pair_frac: float = 0.88,
    backbone_thr_depth1: float = -6*1e-5,
    backbone_thr_depth2: float = 6*1e-3,
    id_offset: int = 0,
    enabled: bool = True,
) -> ModuleNode | None:
    """
    n_depth2 : int or list[int]
        Number of sub-groups for each depth-1 node.
        Pass a single int to use the same k for all depth-1 nodes, or a list
        of length n_depth1 to set each one individually.
        Example: n_depth1=3, n_depth2=[2, 3, 4]  →  first group splits into 2,
                 second into 3, third into 4.
    backbone_thr_depth1 : threshold for root backbone extraction (step 1).
        Default 0.0 — any gene not antagonistic to any depth-1 group goes to root.
    backbone_thr_depth2 : threshold for depth-1 backbone extraction (step 2).
    """
    if not enabled:
        return None
    names = list(gene_names)
    assert phi.shape == (len(names), len(names)), "phi shape must match gene_names length"
    pp = _PostProcessor(phi, names, phi_threshold, pos_pair_frac, id_offset=id_offset)
    # Override backbone thresholds on the instance so build() can read them
    pp._backbone_thr_d1 = backbone_thr_depth1
    pp._backbone_thr_d2 = backbone_thr_depth2
    return pp.build(n_depth1, n_depth2)
