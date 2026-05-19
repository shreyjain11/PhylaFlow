import os
import networkx as nx
from typing import List, Set, Tuple, Optional, Iterable, Dict, Union
from collections import defaultdict, deque
import torch
import re
from scipy.special import rel_entr
import numpy as np
from scipy.spatial.distance import jensenshannon
from ete3 import Tree as EteTree
import random
import hashlib
import torch.nn.functional as F

def get_possible_ids(nexus_root):
    ids = []
    for name in os.listdir(nexus_root):
        base, ext = os.path.splitext(name)
        ids.append(base)
    ids.sort()
    return ids

def remove_bit(mask: int, d: int) -> int:
    """
    Remove bit position d from 'mask' and compress higher bits down by 1.
    Example: remove_bit(0b101001, d=3) removes the 8's place.
    """
    low = mask & ((1 << d) - 1)     # bits [0..d-1]
    high = mask >> (d + 1)         # bits [d+1..] shifted down
    return low | (high << d)

def find_polytomy_nodes(G: nx.Graph, min_degree: int = 4) -> List[int]:
    """
    Internal nodes with degree >= 4 are polytomies (unrooted).
    """
    polys = []
    for u in G.nodes():
        if G.degree[u] >= min_degree:
            polys.append(u)
    return polys

def polytomy_components_at_node(G: nx.Graph, node: int, n_leaves: int, return_comps: bool = False) -> List[int]:
    """
    For a multifurcating node, return component masks of each incident branch.
    Assumes leaf nodes are labeled 0..n_leaves-1 (true in your build_tree_from_splits output).
    """
    leaf_nodes = [str(i) for i in range(n_leaves)]
    comps = []
    for nb in G.neighbors(node):
        m = leaves_in_component(G, nb, node, leaf_nodes)
        if m != 0:
            if return_comps:
                comps.append((nb, m))
            else:
                comps.append(m)
    return comps

def leaves_in_component_split(node: int, possible_splits: List[int]) -> List[int]:
    contained = [s for s in possible_splits if ((s & node) == s) and s != node]
    out = []
    for s in contained:
        # reject s if there exists a STRICT superset t that is ALSO contained in node
        if any((t != s) and ((t & node) == t) and ((s & t) == s) for t in contained):
            continue
        out.append(s)
    return out


def leaves_in_component(G: nx.Graph, start: int, forbidden: int, leaf_nodes: Set[int]) -> int:
    """Bitmask of leaves reachable from start without passing through forbidden."""
    seen = {forbidden}
    q = deque([start])
    mask = 0
    while q:
        u = q.popleft()
        if u in seen:
            continue
        seen.add(u)
        if u in leaf_nodes:
            mask |= (1 << int(u))
        for v in G.neighbors(u):
            if v not in seen:
                q.append(v)
    return mask

def bucket_by_overlap(splits: Iterable[int]) -> List[Set[int]]:
    """
    Buckets are connected components under the relation 'overlap' (a&b != 0).
    O(n^2) in #splits; usually fine for boundary-size sets.
    """
    splits = list(set(splits))
    n = len(splits)
    seen = set()
    buckets: List[Set[int]] = []

    for i in range(n):
        s = splits[i]
        if s in seen:
            continue

        comp = set([s])
        seen.add(s)
        q = deque([s])

        while q:
            cur = q.popleft()
            for t in splits:
                if t in seen:
                    continue
                if (cur & t) != 0:
                    seen.add(t)
                    comp.add(t)
                    q.append(t)

        buckets.append(comp)

    # sort buckets (largest "max split" first) just for nicer printing
    buckets.sort(key=lambda B: max(x.bit_count() for x in B), reverse=True)
    return buckets

def get_batch_polytomy_indices(
    edge_split_masks: List[torch.Tensor],  # [B, T_raw] int64 (bitmask per edge-token)
    edge_mask: torch.Tensor,         # [B, T_raw] bool or {0,1} (valid edge-token positions)
    min_children: int = 3,
    include_root: bool = True,
    num_leaves: Union[int, List[int], torch.Tensor, None] = None,
) -> List[List[torch.LongTensor]]:
    """
    Groups edge-token indices into overlap-buckets (polytomy "regions") per batch element.

    Returns:
      batch_polytomy_index:
        List over b, each is a List of 1D LongTensors of token indices (positions in [0..T_raw-1]).
        Each tensor corresponds to one "polytomy group" bucket.
    """

    if edge_mask.dim() != 2:
        raise ValueError(f"edge_mask must be [B,T], got {tuple(edge_mask.shape)}")

    B = len(edge_split_masks)
    device = edge_mask.device

    batch_polytomy_index: List[List[torch.LongTensor]] = []
    batch_polytomy_splits: List[List[List[int]]] = []

    for b in range(B):
        valid_pos = torch.nonzero(edge_mask[b], as_tuple=False).squeeze(1)  # positions in [0..T-1]

        # splits for valid edge tokens
        splits_b = edge_split_masks[b] #This only contains the valid splits it is not indexed for the full lenght of input 
        if len(splits_b) != edge_mask[b].sum().item():
            raise ValueError("Length mismatch between splits and valid edge mask. This SHOULD NOT HAPPEN.")

        # Map split_mask -> list of token positions that have that split.
        # (Important: keep duplicates! don't lose indices.)
        split_to_positions: Dict[int, List[int]] = defaultdict(list)
        for pos, sm in zip(valid_pos.tolist(), splits_b):
            # You can choose to ignore 0 masks if those mean "no split"
            # (often 0 is padding or placeholder)
            if sm == 0:
                continue
            split_to_positions[int(sm)].append(int(pos))

        unique_splits = list(split_to_positions.keys())

        # Resolve the leaf-universe mask for this batch element.
        if isinstance(num_leaves, torch.Tensor):
            if num_leaves.numel() == 0:
                n_b = 0
            elif num_leaves.numel() == 1:
                n_b = int(num_leaves.item())
            else:
                n_b = int(num_leaves[b].item())
        elif isinstance(num_leaves, (list, tuple)):
            n_b = int(num_leaves[b]) if len(num_leaves) > b else 0
        elif isinstance(num_leaves, int):
            n_b = int(num_leaves)
        else:
            n_b = 0

        if n_b > 0:
            full_mask = (1 << n_b) - 1
        else:
            full_mask = 0
            for s in unique_splits:
                full_mask |= int(s)

        #ADD in the root nodes
        # candidates = list(unique_splits)
        # if include_root:
        #     candidates.append(full_mask ^ root_bit)  # p_root
        if include_root and full_mask != 0:
            unique_splits.append(full_mask)

        polytomy_groups: List[torch.LongTensor] = []
        polytomy_splits: List[List[int]] = []

        def is_subset(sub: int, sup: int) -> bool:
            return (sub & ~sup) == 0

        n = len(unique_splits)
        for pi in range(n):
            p = unique_splits[pi]

            # Proper subsets of p (exclude p)
            subs = [s for s in unique_splits if s!= p and is_subset(s, p)]
            if len(subs) < min_children:
                continue

            # Maximal proper subsets within p:
            # s is maximal if there is NO t in subs such that s ⊂ t ⊂ p
            maximal_subs = []
            for s in subs:
                dominated = False
                for t in subs:
                    if s != t and is_subset(s, t):  # s ⊆ t
                        # if t strictly larger than s, s is not maximal
                        if t.bit_count() > s.bit_count():
                            dominated = True
                            break
                if not dominated:
                    maximal_subs.append(s)

            if len(maximal_subs) >= min_children:
                # Collect token positions for this polytomy region
                idxs: List[int] = []
                for s in maximal_subs:
                    idxs.extend(split_to_positions[int(s)])

                # Dedup + sort for stable indexing
                idxs = sorted(set(idxs))
                polytomy_groups.append(torch.tensor(idxs, dtype=torch.long, device=device))
                polytomy_splits.append(maximal_subs)

        batch_polytomy_index.append(polytomy_groups)
        batch_polytomy_splits.append(polytomy_splits)

    return batch_polytomy_index, batch_polytomy_splits

def pick_group(W, tau=0.5):
    # W: symmetric, diag=-inf
    G = W.size(0)
    i, j = divmod(torch.argmax(W).item(), G)
    if torch.sigmoid(W[i, j]) < tau:
        return None  # nothing confident

    S = {i, j}

    while True:
        best_k, best_score = None, None
        for k in range(G):
            if k in S: 
                continue
            # score to join group: conservative = min link, or average link
            score = torch.sigmoid(torch.stack([W[k, s] for s in S]).min())
            # alternatively: score = torch.sigmoid(torch.stack([W[k,s] for s in S]).mean())
            if best_score is None or score > best_score:
                best_k, best_score = k, score
        if best_score is None or best_score < tau:
            break
        S.add(best_k)

    return sorted(S)

def number_to_name_newick(newick: str, mapping: Dict[int, str], zero_indexed_tree: bool) -> str:
    # Replace digits that are immediately followed by ':' (branch length delimiter)
    pat = re.compile(r'\b(\d+)\b(?=:)')
    def repl(m):
        num = int(m.group(1))
        if not zero_indexed_tree:
            num = num-1
        if num not in mapping:
            raise Exception(f"Mapping missing for leaf number {num} in newick.")
        return mapping.get(num, m.group(1))  # NO colon here
    
    return pat.sub(repl, newick)

def jensenshannon_loglh_divergence(
    true_loglhs: List[float], 
    sampled_loglhs: List[float], 
    bins: int = 50
) -> float:
    """Compute Jensen-Shannon divergence between two log-likelihood distributions."""
    all_vals = true_loglhs + sampled_loglhs
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)
    p, _ = np.histogram(true_loglhs, bins=bin_edges, density=True)
    q, _ = np.histogram(sampled_loglhs, bins=bin_edges, density=True)
    # Add small epsilon to avoid zero probabilities
    p = p + 1e-10
    q = q + 1e-10
    return jensenshannon(p, q)


def kl_loglh_divergence(
    true_loglhs: List[float], 
    sampled_loglhs: List[float], 
    bins: int = 50
) -> float:
    """Compute KL divergence D(true || sampled) between two log-likelihood distributions."""
    all_vals = true_loglhs + sampled_loglhs
    bin_edges = np.histogram_bin_edges(all_vals, bins=bins)
    p, _ = np.histogram(true_loglhs, bins=bin_edges, density=True)
    q, _ = np.histogram(sampled_loglhs, bins=bin_edges, density=True)
    # Normalize to proper probability distributions and add epsilon
    p = (p + 1e-10) / (p + 1e-10).sum()
    q = (q + 1e-10) / (q + 1e-10).sum()
    return rel_entr(p, q).sum()

def return_total_tree_length(newick: str) -> float:
    """
    Computes the total tree length from a Newick string.
    Assumes branch lengths are provided in the Newick format.
    """
    length_pattern = re.compile(r':([\d\.eE+-]+)')
    lengths = length_pattern.findall(newick)
    total_length = sum(float(length) for length in lengths)
    return total_length

def _stable_seed(newick: str, salt: str = "") -> int:
    """
    Deterministic 64-bit seed from (salt | newick). Stable across runs/machines.
    """
    h = hashlib.blake2b((salt + "|" + newick).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False)

def resolve_polytomies_random_deterministic(
    newick: str,
    *,
    dummy_name: str = "ROOT_DUMMY",
    min_len: float = 1e-6,
    salt: str = "",
) -> str:
    """
    Make a Newick strictly bifurcating by deterministically-randomly resolving polytomies.
    - Drops a dummy leaf if present.
    - Contracts unary nodes (degree-2 after rooting representation).
    - Resolves any node with >2 children by repeatedly grouping 2 children under a new internal node.
    - Clamps all branch lengths to >= min_len.

    Determinism: seed = hash(salt | newick). Use salt to get K different but reproducible refinements.
    """
    rng = random.Random(_stable_seed(newick, salt))

    # Parse (format=1 expects branch lengths if present; still works if absent)
    t = EteTree(newick, format=1)

    # Drop dummy leaf if present
    dummies = t.search_nodes(name=dummy_name)
    if dummies:
        dummies[0].detach()

    def clamp_lengths(tree: EteTree):
        for n in tree.traverse():
            if n.is_root():
                continue
            if n.dist is None:
                n.dist = float(min_len)
            else:
                n.dist = max(float(n.dist), float(min_len))

    def contract_unary(tree: EteTree) -> EteTree:
        """
        Remove nodes with a single child by contracting them (add lengths).
        Returns (possibly new) root.
        """
        changed = True
        while changed:
            changed = False
            # Postorder so we contract from leaves upward
            for n in list(tree.traverse("postorder")):
                if n.is_leaf():
                    continue
                if len(n.children) == 1:
                    child = n.children[0]
                    # accumulate length into child
                    if not n.is_root():
                        child.dist = (child.dist or 0.0) + (n.dist or 0.0)
                        parent = n.up
                        child.detach()
                        n.detach()
                        parent.add_child(child)
                        changed = True
                    else:
                        # root with one child: promote child to root
                        child.dist = 0.0
                        child.detach()
                        tree = child
                        changed = True
        return tree

    # Clean up and clamp
    t = contract_unary(t)
    clamp_lengths(t)

    # Resolve polytomies
    # Postorder so deeper polytomies are resolved first
    for n in list(t.traverse("postorder")):
        while not n.is_leaf() and len(n.children) > 2:
            # Pick two children in a deterministic-random way
            c1 = rng.choice(n.children); c1.detach()
            c2 = rng.choice(n.children); c2.detach()

            mid = EteTree()
            mid.dist = float(min_len)
            mid.add_child(c1)
            mid.add_child(c2)

            n.add_child(mid)

    # Contract any unary nodes created by dummy removal / restructuring, clamp again
    t = contract_unary(t)
    clamp_lengths(t)

    # Write back
    return t.write(format=1)

def has_polytomy_fast(newick: str, unrooted_ok: bool = True) -> bool:
    comma_stack = []  # top-level comma count per open '(' group

    for ch in newick:
        if ch == '(':
            comma_stack.append(0)
        elif ch == ',':
            if comma_stack:
                comma_stack[-1] += 1
        elif ch == ')':
            if not comma_stack:
                continue
            commas = comma_stack.pop()

            # if stack is empty AFTER pop => this was the OUTERMOST group (the Newick "root")
            is_root_group = (len(comma_stack) == 0)

            children = commas + 1
            limit = 3 if (unrooted_ok and is_root_group) else 2
            if children > limit:
                return True

    return False

import torch

@torch.no_grad()
def binary_auc_roc(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    scores: [N] float (higher = more positive)
    labels: [N] bool/0-1
    returns: scalar tensor (nan if undefined)
    """
    labels = labels.bool()
    n_pos = labels.sum()
    n_neg = (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(float("nan"), device=scores.device)

    order = torch.argsort(scores, descending=True)
    y = labels[order]

    tps = torch.cumsum(y, dim=0).float()
    fps = torch.cumsum(~y, dim=0).float()

    tpr = tps / n_pos.float()
    fpr = fps / n_neg.float()

    # prepend (0,0)
    z = torch.zeros(1, device=scores.device)
    tpr = torch.cat([z, tpr])
    fpr = torch.cat([z, fpr])

    return torch.trapz(tpr, fpr)


@torch.no_grad()
def binary_auprc(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Average precision (area under precision-recall curve).
    """
    labels = labels.bool()
    n_pos = labels.sum()
    if n_pos == 0:
        return torch.tensor(float("nan"), device=scores.device)

    order = torch.argsort(scores, descending=True)
    y = labels[order]

    tps = torch.cumsum(y, dim=0).float()
    fps = torch.cumsum(~y, dim=0).float()

    precision = tps / (tps + fps).clamp_min(1.0)  # avoid 0/0 at start
    recall = tps / n_pos.float()

    # AP = sum over i where y_i=1 of precision_i * (recall_i - recall_{i-1})
    recall_prev = torch.cat([torch.zeros(1, device=scores.device), recall[:-1]])
    delta_recall = recall - recall_prev
    ap = (precision * delta_recall).sum()
    return ap


@torch.no_grad()
def _decode_merge_subsets_from_adjacency(adjacency: torch.Tensor) -> set[tuple[int, ...]]:
    adjacency = adjacency.detach().cpu().bool()
    n = int(adjacency.size(0))
    visited = [False] * n
    subsets = set()

    for start in range(n):
        if visited[start]:
            continue

        stack = [start]
        component = []
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            neighbors = torch.nonzero(adjacency[node], as_tuple=False).flatten().tolist()
            for nb in neighbors:
                if not visited[int(nb)]:
                    stack.append(int(nb))

        if len(component) >= 2:
            subsets.add(tuple(sorted(component)))

    return subsets


def compute_merge_metrics(
    logits: torch.Tensor,
    y: torch.Tensor,
    threshold_logit: float = 0.0,
) -> dict:
    """
    Computes fast subset-decoding metrics from a group-level pairwise logit matrix.

    logits: [G, G] pairwise logits
    y: [G, G] binary adjacency of true merge cliques
    """
    if logits.dim() != 2 or y.dim() != 2:
        raise ValueError(
            f"logits and y must both be [G, G]; got {tuple(logits.shape)} and {tuple(y.shape)}"
        )

    pred_adj = torch.isfinite(logits) & (logits > float(threshold_logit))
    true_adj = y > 0.5

    pred_adj.fill_diagonal_(False)
    true_adj.fill_diagonal_(False)

    pred_subsets = _decode_merge_subsets_from_adjacency(pred_adj)
    true_subsets = _decode_merge_subsets_from_adjacency(true_adj)

    tp = len(pred_subsets & true_subsets)
    fp = len(pred_subsets - true_subsets)
    fn = len(true_subsets - pred_subsets)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    jaccard = tp / max(len(pred_subsets | true_subsets), 1)

    true_subset_sizes = [len(subset) for subset in true_subsets]
    pred_subset_sizes = [len(subset) for subset in pred_subsets]

    return {
        "autoregressive_stats/subset_exact_match": float(pred_subsets == true_subsets),
        "autoregressive_stats/subset_precision": float(precision),
        "autoregressive_stats/subset_recall": float(recall),
        "autoregressive_stats/subset_f1": float(f1),
        "autoregressive_stats/subset_jaccard": float(jaccard),
        "autoregressive_stats/n_true_subsets": float(len(true_subsets)),
        "autoregressive_stats/n_pred_subsets": float(len(pred_subsets)),
        "autoregressive_stats/avg_true_subset_size": float(
            np.mean(true_subset_sizes) if true_subset_sizes else 0.0
        ),
        "autoregressive_stats/avg_pred_subset_size": float(
            np.mean(pred_subset_sizes) if pred_subset_sizes else 0.0
        ),
    }


class RunningAvg:
    """Simple running mean over step-level dict metrics."""
    def __init__(self):
        self.sum = {}
        self.count = 0

    def update(self, d: dict):
        self.count += 1
        for k, v in d.items():
            if v != v:  # NaN check
                continue
            self.sum[k] = self.sum.get(k, 0.0) + float(v)

    def compute(self) -> dict:
        if self.count == 0:
            return {}
        return {k: v / self.count for k, v in self.sum.items()}


def _bit_indices(mask: int, max_bits: int):
    """Return indices of set bits in mask, clipped to [0, max_bits-1]."""
    out = []
    m = int(mask)
    i = 0
    while m:
        if m & 1:
            if i < max_bits:
                out.append(i)
        m >>= 1
        i += 1
    return out

def _pick_knn_pair(component_embs: torch.Tensor, topM: int = 32, tau: float = 0.05, stochastic: bool = False):
    """
    component_embs: [k, D]
    returns (i, j) indices to merge
    """
    # distances [k,k]
    d = torch.cdist(component_embs, component_embs, p=2)
    d.fill_diagonal_(float("inf"))

    # flatten
    k = d.shape[0]
    flat = d.view(-1)

    # choose smallest pair (or stochastic among topM)
    if not stochastic:
        ij = torch.argmin(flat).item()
        return ij // k, ij % k

    topM = min(topM, flat.numel())
    vals, idxs = torch.topk(-flat, k=topM, largest=True)  # negative distances -> largest = smallest dist
    # vals are -dist, convert to logits
    logits = vals / max(tau, 1e-8)
    probs = torch.softmax(logits, dim=0)
    pick = torch.multinomial(probs, num_samples=1).item()
    ij = idxs[pick].item()
    return ij // k, ij % k


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation between two 1-D tensors."""
    xm = x - x.mean()
    ym = y - y.mean()
    denom = xm.norm() * ym.norm()
    if float(denom) <= 1e-12:
        return 1.0 if torch.allclose(x, y) else 0.0
    return float((xm * ym).sum() / denom)


def _spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """Spearman rank correlation between two 1-D tensors."""
    xr = torch.empty_like(x)
    yr = torch.empty_like(y)
    xr[torch.argsort(x)] = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
    yr[torch.argsort(y)] = torch.arange(y.numel(), dtype=y.dtype, device=y.device)
    return _pearson_corr(xr, yr)


def _velocity_diagnostics(
    p: torch.Tensor,
    y: torch.Tensor,
    topk: int = 3,
    sign_eps: float = 1e-3,
    lengths: Optional[torch.Tensor] = None,
    dt_first_hit_tol: float = 0.01,
    dt_eps: float = 1e-8,
) -> dict:
    """
    Compute diagnostic metrics for predicted (p) vs true (y) velocity vectors.
    Returns a dict with correlation/sign/top-k metrics and dt-hit overlap diagnostics.
    """
    metrics = {}
    metrics["n_edges"] = int(p.numel())
    metrics["mse"] = float(torch.mean((p - y) ** 2))
    metrics["zero_baseline_mse"] = float(torch.mean(y ** 2))
    metrics["mean_baseline_mse"] = float(torch.mean((y - y.mean()) ** 2))
    metrics["mse_vs_zero"] = metrics["mse"] / max(metrics["zero_baseline_mse"], 1e-12)
    metrics["mse_vs_mean"] = metrics["mse"] / max(metrics["mean_baseline_mse"], 1e-12)
    metrics["cosine"] = float(F.cosine_similarity(p, y, dim=0))
    metrics["pearson"] = _pearson_corr(p.detach(), y.detach())
    metrics["spearman"] = _spearman_corr(p.detach(), y.detach())

    # Sign accuracy on edges that are actually moving
    moving = y.abs() > float(sign_eps)
    if int(moving.sum()) > 0:
        metrics["sign_acc"] = float(
            (torch.sign(p[moving]) == torch.sign(y[moving])).float().mean()
        )
    else:
        metrics["sign_acc"] = 1.0

    # Top-k overlap: do the top-k largest-magnitude predicted edges match the true ones?
    k = min(int(topk), int(p.numel()))
    if k > 0:
        pred_topk = set(torch.topk(p.abs(), k=k).indices.tolist())
        true_topk = set(torch.topk(y.abs(), k=k).indices.tolist())
        metrics["topk_overlap"] = len(pred_topk & true_topk) / float(k)
    else:
        metrics["topk_overlap"] = 1.0

    metrics["dt_first_hit_tol"] = float(dt_first_hit_tol)
    metrics["dt_hit_pred"] = float("inf")
    metrics["dt_hit_true"] = float("inf")
    metrics["dt_hit_abs_err"] = 0.0
    metrics["dt_hit_rel_err"] = 0.0
    metrics["dt_neg_jaccard"] = 1.0
    metrics["dt_first_hit_match"] = 1.0
    metrics["dt_first_hit_recall"] = 1.0
    metrics["dt_first_hit_precision"] = 1.0
    metrics["dt_topk_overlap"] = 1.0
    metrics["n_pred_dt_candidates"] = 0
    metrics["n_true_dt_candidates"] = 0

    if lengths is not None and int(lengths.numel()) == int(p.numel()):
        L = lengths.detach()
        p_det = p.detach()
        y_det = y.detach()
        valid = L > float(dt_eps)
        pred_neg = (p_det < 0.0) & valid
        true_neg = (y_det < 0.0) & valid

        pred_neg_idx = torch.where(pred_neg)[0]
        true_neg_idx = torch.where(true_neg)[0]

        if int(pred_neg_idx.numel()) > 0:
            pred_dt_all = L[pred_neg_idx] / (-p_det[pred_neg_idx]).clamp_min(1e-8)
            metrics["dt_hit_pred"] = float(torch.min(pred_dt_all))
            metrics["n_pred_dt_candidates"] = int(pred_dt_all.numel())
        else:
            pred_dt_all = torch.empty(0, dtype=L.dtype, device=L.device)

        if int(true_neg_idx.numel()) > 0:
            true_dt_all = L[true_neg_idx] / (-y_det[true_neg_idx]).clamp_min(1e-8)
            metrics["dt_hit_true"] = float(torch.min(true_dt_all))
            metrics["n_true_dt_candidates"] = int(true_dt_all.numel())
        else:
            true_dt_all = torch.empty(0, dtype=L.dtype, device=L.device)

        both_neg = pred_neg & true_neg
        any_neg = pred_neg | true_neg
        if int(any_neg.sum()) > 0:
            metrics["dt_neg_jaccard"] = float(
                int(both_neg.sum()) / float(int(any_neg.sum()))
            )

        pred_dt_hit = metrics["dt_hit_pred"]
        true_dt_hit = metrics["dt_hit_true"]
        if np.isfinite(pred_dt_hit) and np.isfinite(true_dt_hit):
            metrics["dt_hit_abs_err"] = abs(pred_dt_hit - true_dt_hit)
            metrics["dt_hit_rel_err"] = metrics["dt_hit_abs_err"] / max(
                abs(true_dt_hit), 1e-8
            )
        elif (not np.isfinite(pred_dt_hit)) and (not np.isfinite(true_dt_hit)):
            metrics["dt_hit_abs_err"] = 0.0
            metrics["dt_hit_rel_err"] = 0.0
        else:
            metrics["dt_hit_abs_err"] = float("inf")
            metrics["dt_hit_rel_err"] = float("inf")

        if int(pred_neg_idx.numel()) == 0 and int(true_neg_idx.numel()) == 0:
            metrics["dt_first_hit_match"] = 1.0
            metrics["dt_topk_overlap"] = 1.0
            metrics["dt_first_hit_recall"] = 1.0
            metrics["dt_first_hit_precision"] = 1.0
        elif int(pred_neg_idx.numel()) == 0 or int(true_neg_idx.numel()) == 0:
            metrics["dt_first_hit_match"] = 0.0
            metrics["dt_topk_overlap"] = 0.0
            metrics["dt_first_hit_recall"] = 0.0
            metrics["dt_first_hit_precision"] = 0.0
        else:
            pred_order = pred_neg_idx[torch.argsort(pred_dt_all)]
            true_order = true_neg_idx[torch.argsort(true_dt_all)]

            pred_first_idx = torch.where(
                torch.abs(pred_dt_all - torch.min(pred_dt_all))
                <= float(dt_first_hit_tol)
            )[0]
            true_first_idx = torch.where(
                torch.abs(true_dt_all - torch.min(true_dt_all)) <= 0.0
            )[0]

            pred_first_masks = set(pred_neg_idx[pred_first_idx].tolist())
            true_first_masks = set(true_neg_idx[true_first_idx].tolist())
            first_hit_overlap = pred_first_masks & true_first_masks
            metrics["dt_first_hit_recall"] = len(first_hit_overlap) / float(
                max(len(true_first_masks), 1)
            )
            metrics["dt_first_hit_precision"] = len(first_hit_overlap) / float(
                max(len(pred_first_masks), 1)
            )
            metrics["dt_first_hit_match"] = (
                1.0 if true_first_masks.issubset(pred_first_masks) else 0.0
            )

            k_dt = min(int(topk), int(pred_order.numel()), int(true_order.numel()))
            if k_dt > 0:
                pred_top_masks = set(pred_order[:k_dt].tolist())
                true_top_masks = set(true_order[:k_dt].tolist())
                metrics["dt_topk_overlap"] = len(pred_top_masks & true_top_masks) / float(
                    k_dt
                )
            else:
                metrics["dt_topk_overlap"] = 1.0

    return metrics
