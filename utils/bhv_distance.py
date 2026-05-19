import math
from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple, Optional

Bitmask = int
Length = float

###############################################################################
# Basic utilities: norms, compatibility for bitmask splits, etc.
###############################################################################

def squared_norm(edge_set: Set[Bitmask], lengths: Dict[Bitmask, Length]) -> float:
    """Return sum of squared lengths over a set of splits (edges)."""
    return sum(lengths[e] ** 2 for e in edge_set)

def l2_norm(edge_set: Set[Bitmask], lengths: Dict[Bitmask, Length]) -> float:
    return math.sqrt(squared_norm(edge_set, lengths))

def splits_compatible(mask_a: Bitmask, mask_b: Bitmask, full_mask: Bitmask) -> bool:
    """
    Check split compatibility for unrooted splits encoded as bitmasks.
    mask represents a subset S of leaves; its complement is full_mask ^ mask.
    Two splits A|A^c and B|B^c are compatible iff one of the four intersections
    is empty: A∩B, A∩B^c, A^c∩B, A^c∩B^c.
    """
    a = mask_a
    b = mask_b
    ac = full_mask ^ a
    bc = full_mask ^ b

    # If any of these intersections is empty, the splits are compatible.
    if (a & b) == 0:       # A ∩ B = ∅
        return True
    if (a & bc) == 0:      # A ∩ B^c = ∅
        return True
    if (ac & b) == 0:      # A^c ∩ B = ∅
        return True
    if (ac & bc) == 0:     # A^c ∩ B^c = ∅
        return True
    return False


def _is_internal_split(mask: Bitmask, full_mask: Bitmask) -> bool:
    side = int(mask).bit_count()
    other = int(full_mask ^ mask).bit_count()
    return side > 1 and other > 1


def _is_nested_split(mask: Bitmask, parent_mask: Bitmask, full_mask: Bitmask) -> bool:
    mask_sides = (int(mask), int(full_mask ^ mask))
    parent_sides = (int(parent_mask), int(full_mask ^ parent_mask))
    for side in mask_sides:
        for parent_side in parent_sides:
            if side != parent_side and (side & parent_side) == side:
                return True
    return False


def _compatible_prefix_birth_groups(
    active_splits: Set[Bitmask],
    target_only_splits: Set[Bitmask],
    full_mask: Bitmask,
) -> List[Set[Bitmask]]:
    """
    Recover local boundary births from an exact boundary-point tree.

    When the start tree is a boundary tree, some target splits may already be
    compatible with every active positive split but still represent immediate
    local refinements inside an existing positive parent split. Those births
    should appear as a zero-length prefix boundary event before the remaining
    Owen–Provan support path continues.
    """
    if not active_splits or not target_only_splits:
        return []

    active_internal = {
        int(split)
        for split in active_splits
        if _is_internal_split(int(split), full_mask)
    }
    if not active_internal:
        return []

    promotable = set()
    for split in target_only_splits:
        split = int(split)
        if not all(
            splits_compatible(split, int(active), full_mask)
            for active in active_splits
        ):
            continue
        if any(
            _is_nested_split(split, int(parent), full_mask)
            for parent in active_internal
        ):
            promotable.add(split)

    if not promotable:
        return []
    return [promotable]


###############################################################################
# Max-flow / Min-cut (Dinic) for min-weight vertex cover on bipartite graphs
###############################################################################

class Dinic:
    """
    Simple Dinic max-flow implementation.
    Nodes are arbitrary hashable labels; internally we map to indices.
    """

    class Edge:
        __slots__ = ("to", "capacity", "rev")

        def __init__(self, to: int, capacity: float, rev: int):
            self.to = to
            self.capacity = capacity
            self.rev = rev

    def __init__(self):
        self.graph: List[List[Dinic.Edge]] = []
        self.node_index: Dict[object, int] = {}

    def _add_node(self, label) -> int:
        if label in self.node_index:
            return self.node_index[label]
        idx = len(self.graph)
        self.graph.append([])
        self.node_index[label] = idx
        return idx

    def add_edge(self, u_label, v_label, capacity: float):
        u = self._add_node(u_label)
        v = self._add_node(v_label)
        # forward edge
        self.graph[u].append(Dinic.Edge(v, capacity, len(self.graph[v])))
        # backward edge
        self.graph[v].append(Dinic.Edge(u, 0.0, len(self.graph[u]) - 1))

    def max_flow(self, s_label, t_label) -> float:
        if s_label not in self.node_index or t_label not in self.node_index:
            return 0.0
        s = self.node_index[s_label]
        t = self.node_index[t_label]
        flow = 0.0
        INF = float("inf")

        while True:
            # BFS level graph
            level = [-1] * len(self.graph)
            q = deque()
            q.append(s)
            level[s] = 0
            while q:
                v = q.popleft()
                for e in self.graph[v]:
                    if e.capacity > 1e-15 and level[e.to] < 0:
                        level[e.to] = level[v] + 1
                        q.append(e.to)
            if level[t] < 0:
                break  # no more augmenting paths

            # DFS blocking flow
            it = [0] * len(self.graph)

            def dfs(v, upTo) -> float:
                if v == t:
                    return upTo
                for i in range(it[v], len(self.graph[v])):
                    it[v] = i
                    e = self.graph[v][i]
                    if e.capacity > 1e-15 and level[v] < level[e.to]:
                        d = dfs(e.to, min(upTo, e.capacity))
                        if d > 1e-15:
                            e.capacity -= d
                            self.graph[e.to][e.rev].capacity += d
                            return d
                return 0.0

            while True:
                f = dfs(s, INF)
                if f <= 1e-15:
                    break
                flow += f

        return flow

    def min_cut_partition(self, s_label) -> Tuple[Set[object], Set[object]]:
        """
        After max_flow, compute reachable set from s in residual graph.
        Return (S_labels, T_labels).
        """
        s = self.node_index[s_label]
        visited = [False] * len(self.graph)
        stack = [s]
        visited[s] = True
        while stack:
            v = stack.pop()
            for e in self.graph[v]:
                if e.capacity > 1e-15 and not visited[e.to]:
                    visited[e.to] = True
                    stack.append(e.to)

        S_labels = set()
        T_labels = set()
        # invert node_index map
        inv = {idx: lbl for lbl, idx in self.node_index.items()}
        for idx, lbl in inv.items():
            if visited[idx]:
                S_labels.add(lbl)
            else:
                T_labels.add(lbl)
        return S_labels, T_labels


###############################################################################
# Extension Problem for a single support pair (A_i, B_i)
###############################################################################

def solve_extension_problem(
    Ai: Set[Bitmask],
    Bi: Set[Bitmask],
    lengths1: Dict[Bitmask, Length],
    lengths2: Dict[Bitmask, Length],
    full_mask: Bitmask,
) -> Optional[Tuple[Set[Bitmask], Set[Bitmask], Set[Bitmask], Set[Bitmask]]]:
    """
    Solve the Extension Problem for one support pair (Ai, Bi).

    Returns:
        (C1, C2, D1, D2) if there is a nontrivial refinement that shortens the path,
        or None if no such refinement exists (i.e., (Ai, Bi) satisfies P3).

    Ai is a subset of edges from tree1 (disjoint part), Bi from tree2 (disjoint part).
    lengths1 and lengths2 give the original edge lengths for each bitmask.
    """
    if not Ai or not Bi:
        # Nothing to refine if one side is empty.
        return None

    # Norms for this support pair
    normA2 = squared_norm(Ai, lengths1)
    normB2 = squared_norm(Bi, lengths2)
    if normA2 <= 0 or normB2 <= 0:
        return None  # degenerate

    # Build incompatibility graph G(Ai, Bi)
    # Vertex set: Ai ∪ Bi
    # Edges between incompatible splits.
    G = defaultdict(set)  # adjacency, but we just need structure for debugging; flow uses its own structure
    for e in Ai:
        for f in Bi:
            if not splits_compatible(e, f, full_mask):
                G[e].add(f)
                G[f].add(e)

    # If there are no incompatibilities, sets are compatible; no refinement needed.
    has_edges = any(G[v] for v in G)
    if not has_edges:
        return None

    # Build max-flow instance for min-weight vertex cover.
    # Capacity model: s -> e (for e∈Ai) with capacity = weight_e
    #                 f (for f∈Bi) -> t with capacity = weight_f
    #                 e -> f (for each incompatibility edge) with capacity = INF
    #
    # Standard result: from min s–t cut (S,T), a min vertex cover in bipartite graph is:
    #   C1 = { e ∈ Ai | e NOT in S }
    #   D2 = { f ∈ Bi | f IN S }
    #
    # Weighted capacity ensures cover weight is min.
    dinic = Dinic()
    s_label = "__source__"
    t_label = "__sink__"
    INF = 1e9  # sufficiently large

    # vertex weights (normalized squared lengths)
    for e in Ai:
        w = (lengths1[e] ** 2) / normA2
        if w < 0:
            w = 0.0
        dinic.add_edge(s_label, ("A", e), w)
    for f in Bi:
        w = (lengths2[f] ** 2) / normB2
        if w < 0:
            w = 0.0
        dinic.add_edge(("B", f), t_label, w)

    # incompatibility edges with infinite capacity
    for e in Ai:
        for f in G[e]:
            if f in Bi:  # only once; G is symmetric but we don't want duplicates
                dinic.add_edge(("A", e), ("B", f), INF)

    # Compute max-flow / min-cut
    _ = dinic.max_flow(s_label, t_label)
    S_labels, T_labels = dinic.min_cut_partition(s_label)

    # Build the vertex cover from min-cut
    C1: Set[Bitmask] = set()  # subset of Ai in cover
    D2: Set[Bitmask] = set()  # subset of Bi in cover

    for e in Ai:
        if ("A", e) not in S_labels:  # not reachable from s => in cover
            C1.add(e)
    for f in Bi:
        if ("B", f) in S_labels:      # reachable from s => in cover
            D2.add(f)

    C2 = Ai - C1
    D1 = Bi - D2

    # Trivial partition (no real split)? Then no useful refinement.
    if not C1 or not C2 or not D1 or not D2:
        return None

    # Check the cover weight < 1 condition (P3 reformulated).
    cover_weight = (
        squared_norm(C1, lengths1) / normA2
        + squared_norm(D2, lengths2) / normB2
    )
    if cover_weight >= 1.0 - 1e-12:
        return None

    return C1, C2, D1, D2


###############################################################################
# GTP algorithm over all support pairs (disjoint edges only)
###############################################################################

def gtp_geodesic_support(
    E1_disjoint: Set[Bitmask],
    E2_disjoint: Set[Bitmask],
    lengths1: Dict[Bitmask, Length],
    lengths2: Dict[Bitmask, Length],
    full_mask: Bitmask,
) -> Tuple[List[Set[Bitmask]], List[Set[Bitmask]]]:
    """
    Run the Owen–Provan GTP algorithm on trees with *disjoint* edge sets
    (i.e., common edges already removed).

    Returns:
        (A_support, B_support)
        where A_support = [A1, ..., Ak], B_support = [B1, ..., Bk]
    """
    # Start with the cone path support: A0 = (E1_disjoint), B0 = (E2_disjoint)
    A_support: List[Set[Bitmask]] = [set(E1_disjoint)]
    B_support: List[Set[Bitmask]] = [set(E2_disjoint)]

    # Iteratively refine until no Extension Problem has a solution
    while True:
        refined = False
        # Iterate over a *copy* of indices, since we may mutate lists
        for i in range(len(A_support)):
            Ai = A_support[i]
            Bi = B_support[i]
            if not Ai or not Bi:
                continue

            result = solve_extension_problem(Ai, Bi, lengths1, lengths2, full_mask)
            if result is None:
                continue

            C1, C2, D1, D2 = result

            # Replace (Ai, Bi) by (C1, C2) and (D1, D2) (preserving order)
            # A = [..., Ai-1, C1, C2, Ai+1, ...]
            # B = [..., Bi-1, D1, D2, Bi+1, ...]
            A_support = A_support[:i] + [C1, C2] + A_support[i+1:]
            B_support = B_support[:i] + [D1, D2] + B_support[i+1:]
            refined = True
            break

        if not refined:
            break

    return A_support, B_support


###############################################################################
# Top-level BHV distance interface for bitmask-encoded trees
###############################################################################

def bhv_geodesic_with_support(
    tree1: Dict[Bitmask, Length],
    tree2: Dict[Bitmask, Length],
    n_leaves: int,
    *,
    drop_zero_length_edges: bool = True,
    enable_prefix_birth_groups: bool = True,
):
    """
    Compute the BHV geodesic between two bitmask-encoded trees AND
    return the full support (A_i, B_i) and per-segment orthant info.

    Returns:
      {
        "distance": float,
        "common_sq": float,
        "disjoint_sq": float,
        "A_support": List[Set[Bitmask]],
        "B_support": List[Set[Bitmask]],
        "segments": List[{
            "Ai": Set[Bitmask],
            "Bi": Set[Bitmask],
            "start_splits": Set[Bitmask],
            "end_splits": Set[Bitmask],
            "normA": float,
            "normB": float,
            "ratio": float,
        }]
      }
    """
    if drop_zero_length_edges:
        # Zero-length representation edges are not part of the active BHV state.
        tree1 = {int(e): float(l) for e, l in tree1.items() if float(l) > 1e-8}
        tree2 = {int(e): float(l) for e, l in tree2.items() if float(l) > 1e-8}
    else:
        tree1 = {
            int(e): float(l)
            for e, l in tree1.items()
            if l is not None
        }
        tree2 = {
            int(e): float(l)
            for e, l in tree2.items()
            if l is not None
        }

    full_mask = (1 << n_leaves) - 1

    E1 = set(tree1.keys())
    E2 = set(tree2.keys())

    tree1_all = dict(tree1)
    tree2_all = dict(tree2)

    common = E1 & E2
    E1_only = E1 - common
    E2_only = E2 - common

    prefix_birth_groups = []
    if enable_prefix_birth_groups and len(E1_only) == 1:
        prefix_birth_groups = _compatible_prefix_birth_groups(E1, E2_only, full_mask)
    prefix_births = set().union(*prefix_birth_groups) if prefix_birth_groups else set()
    if prefix_births:
        common |= prefix_births
        E2_only -= prefix_births
        for split in prefix_births:
            tree1_all[int(split)] = 0.0

    # Common part: Euclidean on differences
    common_sq = 0.0
    for e in common:
        diff = tree1_all.get(int(e), 0.0) - tree2_all.get(int(e), 0.0)
        common_sq += diff * diff

    if not E1_only and not E2_only:
        # Trees have identical split sets; pure Euclidean
        dist = math.sqrt(common_sq)
        segments = compute_orthant_segments(
            common,
            [],
            [],
            {},
            {},
            tree1_all,
            tree2_all,
            prefix_birth_groups=prefix_birth_groups,
        )
        return {
            "distance": dist,
            "common_sq": common_sq,
            "disjoint_sq": 0.0,
            "A_support": [],
            "B_support": [],
            "segments": segments,
        }

    lengths1 = {e: tree1[e] for e in E1_only}
    lengths2 = {e: tree2[e] for e in E2_only}

    # GTP algorithm to get optimal support
    A_support, B_support = gtp_geodesic_support(
        E1_only, E2_only, lengths1, lengths2, full_mask
    )

    # Disjoint part of BHV distance
    disjoint_sq = 0.0
    for Ai, Bi in zip(A_support, B_support):
        normA = math.sqrt(sum(lengths1[e] ** 2 for e in Ai)) if Ai else 0.0
        normB = math.sqrt(sum(lengths2[f] ** 2 for f in Bi)) if Bi else 0.0
        disjoint_sq += (normA + normB) ** 2

    total_sq = common_sq + disjoint_sq
    distance = math.sqrt(total_sq)

    # Segment-wise orthant info for visualization
    segments = compute_orthant_segments(
        common,
        A_support,
        B_support,
        lengths1,
        lengths2,
        tree1_all,
        tree2_all,
        prefix_birth_groups=prefix_birth_groups,
    )

    return {
        "distance": distance,
        "common_sq": common_sq,
        "disjoint_sq": disjoint_sq,
        "A_support": A_support,
        "B_support": B_support,
        "segments": segments,
    }

def lengths_at_lambda(
    lam: float,
    common: Set[Bitmask],
    A_support: List[Set[Bitmask]],
    B_support: List[Set[Bitmask]],
    tree1_all: Dict[Bitmask, float],
    tree2_all: Dict[Bitmask, float],
    lengths1: Dict[Bitmask, float],
    lengths2: Dict[Bitmask, float],
    normsA: List[float],
    normsB: List[float],
) -> Dict[Bitmask, float]:
    """
    Compute edge lengths at parameter λ along the BHV geodesic,
    using Owen–Provan formulas.
    """

    # Precompute mapping: which support pair and which side does edge belong to?
    edge_to_pair = {}
    for j, A in enumerate(A_support):
        for e in A:
            edge_to_pair[e] = (j, "A")
    for j, B in enumerate(B_support):
        for e in B:
            edge_to_pair[e] = (j, "B")

    all_edges = set(lengths1.keys()) | set(lengths2.keys()) | common
    L = {}

    for e in all_edges:
        if e in common:
            # common edge: linear interpolation
            L[e] = (1.0 - lam) * tree1_all[e] + lam * tree2_all[e]
            continue

        # if it's in support pairs, use the piecewise formula
        if e in edge_to_pair:
            j, side = edge_to_pair[e]
            normA = normsA[j]
            normB = normsB[j]

            if side == "A":
                if normA == 0.0:
                    L[e] = 0.0
                else:
                    coeff = ((1.0 - lam) * normA - lam * normB) / normA
                    L[e] = max(0.0, coeff * lengths1[e])  # clip tiny negatives
            else:  # side == "B"
                if normB == 0.0:
                    L[e] = 0.0
                else:
                    coeff = (lam * normB - (1.0 - lam) * normA) / normB
                    L[e] = max(0.0, coeff * lengths2[e])
        else:
            # edge only in T1 or T2 but not in support (rare if you've already
            # done the standard preprocessing). Treat missing length as 0.
            if in1 and not in2:
                # shrinks from lenghts1 -> 0
                L[e] = max(0.0, (1.0 - lam) * lengths1[e])
            elif in2 and not in1:
                # grows from 0 -> lengths2
                L[e] = max(0.0, lam * lengths2[e])
            else:
                L[e] = 0.0

    return L

def compute_orthant_segments(
    common: Set[Bitmask],
    A_support: List[Set[Bitmask]],
    B_support: List[Set[Bitmask]],
    lengths1: Dict[Bitmask, float],
    lengths2: Dict[Bitmask, float],
    tree1_all: Dict[Bitmask, float],
    tree2_all: Dict[Bitmask, float],
    prefix_birth_groups: Optional[List[Set[Bitmask]]] = None,
):
    """
    Build a per-segment description of the BHV geodesic using Owen–Provan.

    Returns a list of segments, each with:
        - Ai, Bi
        - start_splits, end_splits
        - normA, normB, ratio
        - start_lengths, end_lengths
        - length   (true BHV length of that segment)
    """

    k = len(A_support)
    assert k == len(B_support)

    # Norms and ratios
    normsA = []
    normsB = []
    ratios = []

    for Ai, Bi in zip(A_support, B_support):
        normA = math.sqrt(sum(lengths1[e] ** 2 for e in Ai)) if Ai else 0.0
        normB = math.sqrt(sum(lengths2[f] ** 2 for f in Bi)) if Bi else 0.0
        normsA.append(normA)
        normsB.append(normB)

        if normB > 0:
            ratios.append(normA / normB)
        elif normA > 0:
            ratios.append(float("inf"))
        else:
            ratios.append(0.0)

    # Build λ-boundaries: λ0=0, λ_{i} = r_i/(1+r_i), λ_{k+1}=1
    lambdas = [0.0]
    for r in ratios:
        if math.isinf(r) and r > 0:
            lambdas.append(1.0)
        elif r <= -1e10:  # if you ever have negative ratios for common edges
            lambdas.append(0.0)
        else:
            lambdas.append(r / (1.0 + r))
    lambdas.append(1.0)

    # All edges we track for Euclidean norms
    all_edges = set(lengths1.keys()) | set(lengths2.keys()) | common

    segments = []

    # We’ll define k+1 legs as in Owen–Provan; typically the first/last may be short.
    for i in range(k + 1):
        lam_start = lambdas[i]
        lam_end = lambdas[i + 1]

        # Edge lengths at the two endpoints of this leg
        L_start = lengths_at_lambda(
            lam_start, common, A_support, B_support, tree1_all, tree2_all, lengths1, lengths2, normsA, normsB
        )
        L_end = lengths_at_lambda(
            lam_end, common, A_support, B_support, tree1_all, tree2_all, lengths1, lengths2, normsA, normsB
        )

        # Euclidean segment length
        seg_len_sq = 0.0
        for e in all_edges:
            d = L_end[e] - L_start[e]
            seg_len_sq += d * d
        seg_len = math.sqrt(seg_len_sq)

        # Which splits are nonzero at start/end
        eps = 1e-8
        start_splits = {e for e in all_edges if L_start[e] > eps}
        end_splits = {e for e in all_edges if L_end[e] > eps}

        # For bookkeeping, we can associate this leg with the pair (A_i, B_i),
        # where for i==0 or i==k we just use empty sets.
        Ai = A_support[i] if 0 <= i < k else set()
        Bi = B_support[i] if 0 <= i < k else set()
        normA = normsA[i] if 0 <= i < k else 0.0
        normB = normsB[i] if 0 <= i < k else 0.0
        ratio = ratios[i] if 0 <= i < k else None

        segments.append({
            "Ai": set(Ai),
            "Bi": set(Bi),
            "start_splits": start_splits,
            "end_splits": end_splits,
            "normA": normA,
            "normB": normB,
            "ratio": ratio,
            "lambda_start": lam_start,
            "lambda_end": lam_end,
            "start_lengths": L_start,
            "end_lengths": L_end,
            "length": seg_len,
            "velocity": {
                e: (L_end[e] - L_start[e]) / seg_len if seg_len > 0 else 0.0
                for e in all_edges
            },
        })

    prefix_birth_groups = prefix_birth_groups or []
    if not prefix_birth_groups:
        return segments

    prefix_lengths = lengths_at_lambda(
        0.0,
        common,
        A_support,
        B_support,
        tree1_all,
        tree2_all,
        lengths1,
        lengths2,
        normsA,
        normsB,
    )
    all_edges = set(prefix_lengths.keys())
    prefix_segments = []
    eps = 1e-8
    for births in prefix_birth_groups:
        prefix_segments.append(
            {
                "Ai": set(),
                "Bi": {int(split) for split in births},
                "start_splits": {
                    int(edge) for edge, length in prefix_lengths.items() if float(length) > eps
                },
                "end_splits": {
                    int(edge) for edge, length in prefix_lengths.items() if float(length) > eps
                },
                "normA": 0.0,
                "normB": 0.0,
                "ratio": 0.0,
                "lambda_start": 0.0,
                "lambda_end": 0.0,
                "start_lengths": dict(prefix_lengths),
                "end_lengths": dict(prefix_lengths),
                "length": 0.0,
                "velocity": {int(edge): 0.0 for edge in all_edges},
            }
        )

    return prefix_segments + segments
