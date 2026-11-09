from __future__ import annotations

import copy

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
    """Incremental R1-R4 insertion with explicit pairwise pattern constraints."""

    def __init__(
        self, config: GeCCoConfig, gene_names: np.ndarray, phi: np.ndarray,
        fdr: np.ndarray | None, connectivity: np.ndarray,
        valid_mask: np.ndarray, n_expressed: np.ndarray,
    ):
        self.config = config
        self.gene_names = np.asarray(gene_names)
        self.phi = phi
        self.fdr = fdr
        self.connectivity = connectivity
        self.valid_mask = valid_mask
        self.n_expressed = np.asarray(n_expressed)
        self.gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}
        self._node_counter = 0

    def _next_node_id(self) -> str:
        self._node_counter += 1
        return f"M{self._node_counter}"

    def _idx(self, gene: str) -> int:
        return self.gene_to_idx[gene]

    def _phi(self, g1: str, g2: str) -> float:
        return float(self.phi[self._idx(g1), self._idx(g2)])

    def _connectivity(self, gene: str) -> int:
        return int(self.connectivity[self._idx(gene)])

    def _matches(self, values: np.ndarray, positive: bool, qvalues=None) -> np.ndarray:
        directed = values if positive else -values
        matches = (directed > 0) & (directed >= self.config.phi_threshold)
        if self.config.test:
            matches &= qvalues <= self.config.fdr_threshold
        return matches

    def _enough(self, matches: np.ndarray) -> bool:
        return matches.size == 0 or (np.count_nonzero(~matches) / matches.size <= self.config.max_pair_violation_fraction)

    def _pairwise(self, genes_a, genes_b=None, *, positive=True) -> bool:
        a = [self._idx(g) for g in sorted(genes_a)]
        b = a if genes_b is None else [self._idx(g) for g in sorted(genes_b)]
        vals = self.phi[np.ix_(a, b)]
        qs = self.fdr[np.ix_(a, b)] if self.config.test else None
        matches = self._matches(vals, positive, qs)
        if genes_b is None:
            matches = matches[np.triu_indices(len(a), k=1)]
        return self._enough(matches)

    def _classify(self, gene: str, module_genes: Iterable[str]) -> str:
        genes = set(module_genes) - {gene}
        if not genes:
            return "W"
        if self._pairwise({gene}, genes):
            return "P"
        if self._pairwise({gene}, genes, positive=False):
            return "N"
        vals = np.array([self._phi(gene, g) for g in sorted(genes)])
        return "M" if np.any(vals > 0) and np.any(vals < 0) else "W"

    def _classify_gene_module_relation(self, gene: str, module_genes: Iterable[str]) -> str:
        return self._classify(gene, module_genes)

    @staticmethod
    def _module_genes(node: ModuleNode) -> set[str]:
        return node.genes if node.genes else node.total_genes()

    @staticmethod
    def _record(trace, rule, target, changed):
        trace["rule"] = rule
        trace["target_node_id"] = target.node_id
        trace["changed_nodes"] = list(dict.fromkeys(changed))

    def _try_split_leaf(self, leaf: ModuleNode, trigger_gene: str | None = None):
        """Separate antagonistic markers, retaining shared positive markers at parent."""
        if not leaf.is_leaf():
            return False, None
        genes = sorted(leaf.genes)
        pairs = sorted(combinations(genes, 2), key=lambda p: (self._phi(*p), p))
        for seed_a, seed_b in pairs:
            if self._classify(seed_a, {seed_b}) != "N":
                continue
            side_a, side_b, backbone = {seed_a}, {seed_b}, set()
            for gene in genes:
                if gene in (seed_a, seed_b):
                    continue
                ra = self._classify(gene, {seed_a})
                rb = self._classify(gene, {seed_b})
                if ra == rb == "P":
                    backbone.add(gene)
                elif ra == "P" and rb == "N":
                    side_a.add(gene)
                elif ra == "N" and rb == "P":
                    side_b.add(gene)
                else:
                    break
            else:
                if min(len(side_a), len(side_b)) < self.config.min_module_size:
                    continue
                if not all((self._pairwise(side_a), self._pairwise(side_b),
                            self._pairwise(backbone), self._pairwise(side_a, side_b, positive=False),
                            self._pairwise(backbone, side_a | side_b))):
                    continue
                leaf.genes = backbone
                leaf.children = [ModuleNode(self._next_node_id(), side_a),
                                 ModuleNode(self._next_node_id(), side_b)]
                return True, {
                    "split_parent_id": leaf.node_id,
                    "split_child_ids": [c.node_id for c in leaf.children],
                    "reassigned_existing_genes": sorted((side_a | side_b) - {trigger_gene}),
                }
        return False, None

    def _apply_r2(self, parent, gene, p_children):
        moved_ids = {c.node_id for c in p_children}
        new_parent = ModuleNode(self._next_node_id(), {gene}, p_children)
        parent.children = [c for c in parent.children if c.node_id not in moved_ids] + [new_parent]
        return new_parent

    def _apply_r3(self, parent, gene):
        node = ModuleNode(self._next_node_id(), {gene})
        parent.children.append(node)
        return node

    def _insert_recursive(self, current, gene, depth, trace):
        if not current.children:
            if current.node_id == "root":
                target = self._apply_r3(current, gene)
                self._record(trace, "INIT", target, [target.node_id])
                return True
            if self._classify(gene, current.genes) == "P":
                current.genes.add(gene)
                self._record(trace, "R1_LEAF_ABSORB", current, [current.node_id])
                return True
            trace["rule"] = "PAIRWISE_REJECT"
            return False

        relations = {c.node_id: self._classify(gene, self._module_genes(c)) for c in current.children}
        trace["path"].append(current.node_id + "|" + ",".join(f"{k}:{v}" for k, v in relations.items()))
        p = [c for c in current.children if relations[c.node_id] == "P"]
        n = [c for c in current.children if relations[c.node_id] == "N"]
        mixed = [c for c in current.children if relations[c.node_id] in {"M", "W"}]

        if not mixed:
            if len(p) == 1:
                return self._insert_recursive(p[0], gene, depth + 1, trace)
            if len(p) >= 2:
                if current.node_id != "root" and len(p) == len(current.children):
                    current.genes.add(gene)
                    target, rule = current, "R2_ALL_P_ABSORB_PARENT"
                else:
                    target = self._apply_r2(current, gene, p)
                    rule = "R2_CREATE_INTERMEDIATE_PARENT"
                self._record(trace, rule, target, [current.node_id, target.node_id])
                return True
            if len(n) == len(current.children):
                target = self._apply_r3(current, gene)
                self._record(trace, "R3_CREATE_NEW_SIBLING", target, [current.node_id, target.node_id])
                return True

        # A broad marker may share a leaf with one subtype until the opposing subtype arrives.
        if len(mixed) == 1 and not p and mixed[0].is_leaf():
            child = mixed[0]
            child.genes.add(gene)
            ok, info = self._try_split_leaf(child, trigger_gene=gene)
            if ok:
                target = next((c for c in child.children if gene in c.genes), child)
                self._record(trace, "R4_SPLIT_LEAF_AND_INSERT", target, [child.node_id, *info["split_child_ids"]])
                trace.update(info)
                return True
            child.genes.remove(gene)
        trace["rule"] = "NO_RULE_MATCH_REJECT"
        return False

    def _check_c1(self, node):
        return self._pairwise(node.genes) and all(self._check_c1(c) for c in node.children)

    def _check_c2(self, node):
        # Entire sibling branches must remain distinct, including their subtype markers.
        return all(self._pairwise(a.total_genes(), b.total_genes(), positive=False)
                   for a, b in combinations(node.children, 2)) and all(self._check_c2(c) for c in node.children)

    def _check_c3(self, node):
        descendants = set().union(*(c.total_genes() for c in node.children))
        return self._pairwise(node.genes, descendants) and all(self._check_c3(c) for c in node.children)

    def _check_constraints(self, root):
        return self._check_c1(root), self._check_c2(root), self._check_c3(root)

    @staticmethod
    def _depth(node):
        return 1 + max((ModuleTreeBuilder._depth(c) for c in node.children), default=0)

    def build(self) -> TreeBuildResult:
        # Stable ties make input column order irrelevant. Every seed obeys the same rules.
        genes = self.gene_names[self.valid_mask].tolist()
        genes.sort(key=lambda g: (-int(self.n_expressed[self._idx(g)]), -self._connectivity(g), g))
        self._node_counter = 0
        root = ModuleNode("root")
        inserted, noise, traces = [], [], []
        for step, gene in enumerate(genes):
            trace = dict(step=step, gene=gene, gene_index=self._idx(gene),
                         connectivity=self._connectivity(gene), rule="", target_node_id="-",
                         path=[], accepted=False, is_noise=False, changed_nodes=[],
                         split_parent_id="-", split_child_ids=[], reassigned_existing_genes=[])
            snapshot, counter = copy.deepcopy(root), self._node_counter
            if len(genes) > 1 and self._connectivity(gene) == 0:
                trace["rule"] = "NO_SUPPORTED_PAIRS"
                ok = False
            else:
                ok = self._insert_recursive(root, gene, 0, trace)
            if ok:
                constraints = self._check_constraints(root)
                if not all(constraints):
                    trace["rule"] = next(f"C{i}_VIOLATED" for i, passed in enumerate(constraints, 1) if not passed)
                    ok = False
                elif self._depth(root) - 1 > self.config.max_depth:
                    trace["rule"] = "MAX_DEPTH_REJECT"
                    ok = False
            if ok:
                inserted.append(gene)
                trace["accepted"] = True
            else:
                root, self._node_counter = snapshot, counter
                noise.append(gene)
                trace.update(is_noise=True, target_node_id="-", changed_nodes=[],
                             split_parent_id="-", split_child_ids=[], reassigned_existing_genes=[])
            trace["path"] = " -> ".join(trace["path"])
            traces.append(trace)
        return TreeBuildResult(root, inserted, noise, traces)


# ------------------------------------------------------------------ utilities

def nodes_at_depth(root: ModuleNode, depth: int) -> list[ModuleNode]:
    """Use the same root=0 depth convention as plots, tables and max_depth."""
    if depth < 0:
        return []
    current = [root]
    for _ in range(depth):
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
        ia = [self._idx(g) for g in sorted(a)]
        ib = [self._idx(g) for g in sorted(b)]
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
        if not gene_list:
            return [], set()
        k = min(k, len(gene_list))
        if k == 1:
            return [set(gene_list)], set()

        seeds = self._pick_k_seeds(gene_list, k)
        seed_groups = {gene: i for i, gene in enumerate(seeds)}

        # Initial assignment: each gene → group of most correlated seed
        assignment: dict[str, int] = {}
        for g in gene_list:
            scores = [self._phi_val(g, s) for s in seeds]
            assignment[g] = seed_groups[g] if g in seed_groups else int(np.argmax(scores))

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
                if g in seed_groups:
                    new_assignment[g] = seed_groups[g]
                    continue
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
            for g in sorted(groups[ki]):
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
                sorted(kept),
                key=lambda g: float(np.mean(
                    [self._phi_val(g, other) for other in sorted(kept) if other != g]
                )),
            )
            kept.discard(worst)
            pruned.add(worst)
        return kept, pruned

    # ── main build ────────────────────────────────────────────────────────

    def _grow(self, node: ModuleNode, genes: set[str], branching: list[int], depth: int) -> None:
        if not branching or len(genes) <= 1:
            node.genes = genes
            return
        threshold = self._backbone_thr_d1 if depth == 0 else self._backbone_thr_d2
        groups, node.genes = self._split_k_way(genes, branching[0], backbone_thr=threshold)
        for group in groups:
            child = ModuleNode(node_id=self._next_id())
            self._grow(child, group, branching[1:], depth + 1)
            node.children.append(child)

    def build(
        self, n_depth1: int, n_depth2: int | list[int], branching: list[int] | None = None
    ) -> ModuleNode:
        all_genes: set[str] = set(self.gene_names)
        root = ModuleNode(node_id="root")

        if branching is not None:
            if not branching or any(k < 1 for k in branching):
                raise ValueError("branching must contain positive numbers of children")
            # Keep a real depth-1 node even for a single selected gene.
            groups, root.genes = self._split_k_way(
                all_genes, branching[0], backbone_thr=self._backbone_thr_d1
            )
            for group in groups:
                child = ModuleNode(node_id=self._next_id())
                self._grow(child, group, branching[1:], depth=1)
                root.children.append(child)
        else:
            if n_depth1 < 1:
                raise ValueError("n_depth1 must be positive")
            n_depth2_list = [n_depth2] * n_depth1 if isinstance(n_depth2, int) else list(n_depth2)
            if len(n_depth2_list) != n_depth1 or any(k < 0 for k in n_depth2_list):
                raise ValueError("n_depth2 must be nonnegative, with one value per depth-1 group")
            groups, root.genes = self._split_k_way(
                all_genes, n_depth1, backbone_thr=self._backbone_thr_d1
            )
            for group, k2 in zip(groups, n_depth2_list):
                child = ModuleNode(node_id=self._next_id())
                self._grow(
                    child, group, [k2] if k2 > 1 else [], depth=1
                )
                root.children.append(child)

        noise_genes: set[str] = set()
        for node in iter_nodes(root):
            if node is not root:
                node.genes, pruned = self._prune_to_coherence(node.genes)
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
    branching: list[int] | tuple[int, ...] | None = None,
) -> ModuleNode | None:
    """
    n_depth2 : int or list[int]
        Number of sub-groups for each depth-1 node.
        Pass a single int to use the same k for all depth-1 nodes, or a list
        of length n_depth1 to set each one individually.
        Zero or one leaves the depth-1 group unsplit.
        Example: n_depth1=3, n_depth2=[2, 3, 4]  →  first group splits into 2,
                 second into 3, third into 4.
    branching : optional per-depth numbers of children, e.g. [3, 2, 2].
        Overrides n_depth1/n_depth2. Counts are capped by available genes;
        singleton groups stop splitting. Removed incoherent genes are listed
        in the returned root's _noise_from_pruning attribute.
    backbone_thr_depth1 : threshold for root backbone extraction (step 1).
        Default -0.00006, retained from the original post-processing workflow.
    backbone_thr_depth2 : threshold for depth-1 backbone extraction (step 2).
    """
    if not enabled:
        return None
    names = list(gene_names)
    if phi.shape != (len(names), len(names)):
        raise ValueError("phi shape must match gene_names length")
    if not 0 <= pos_pair_frac <= 1:
        raise ValueError("pos_pair_frac must be between 0 and 1")
    pp = _PostProcessor(phi, names, phi_threshold, pos_pair_frac, id_offset=id_offset)
    # Override backbone thresholds on the instance so build() can read them
    pp._backbone_thr_d1 = backbone_thr_depth1
    pp._backbone_thr_d2 = backbone_thr_depth2
    return pp.build(n_depth1, n_depth2, None if branching is None else list(branching))
