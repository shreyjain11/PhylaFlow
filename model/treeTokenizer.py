import math
import pdb
import os
import multiprocessing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from ete3 import Tree as EteTree


class BatchedStructuralCache:
    def __init__(self, caches, device, encoder_embed_dim):
        self.device = device
        self.batch_size = len(caches)
        self.encoder_embed_dim = encoder_embed_dim

        # Determine max dimensions
        self.max_tokens = 0
        self.max_edges = 0
        for c in caches:
            # Token count = node_num + edge_num
            total_tokens = c["static_tokens"].size(0)
            self.max_tokens = max(self.max_tokens, total_tokens)+100
            self.max_edges = max(self.max_edges, c["edge_num"])+100

        # Pre-allocate batched tensors
        self.static_tokens = torch.zeros(
            (self.batch_size, self.max_tokens, encoder_embed_dim), device=device
        )
        self.padding_mask = torch.ones(
            (self.batch_size, self.max_tokens), device=device, dtype=torch.bool
        ) 
        self.full_padded_indices = torch.zeros(
            (self.batch_size, self.max_tokens, 2), device=device, dtype=torch.long
        )
        self.leaf_mask = torch.zeros(
            (self.batch_size, self.max_tokens), device=device, dtype=torch.bool
        )
        self.edge_mask = torch.zeros(
            (self.batch_size, self.max_tokens), device=device, dtype=torch.bool
        )
        self.edge_scatter_indices = torch.zeros(
            (self.batch_size, self.max_edges), device=device, dtype=torch.long
        )
        self.edge_scatter_mask = torch.zeros(
            (self.batch_size, self.max_edges), device=device, dtype=torch.bool
        )

        self.leaf_indices_list = []
        self.edge_split_masks_list = []  # List of lists

        for i, c in enumerate(caches):
            self._set_cache_item(i, c)

    def _set_cache_item(self, i, c):
        length = c["static_tokens"].size(0)

        # Check size
        if length > self.static_tokens.size(1):
            # Resize logic could go here, but for now we warn/error -> simplistic re-alloc
            # Ideally we shouldn't hit this often if we init with 'max' of batch
            raise RuntimeError(
                f"Tree token size {length} grew larger than batch capacity {self.static_tokens.size(1)}"
            )

        self.static_tokens[i].zero_()  # clear old potentially
        self.static_tokens[i, :length] = c["static_tokens"]

        self.padding_mask[i] = True
        self.padding_mask[i, :length] = c[
            "padding_mask"
        ]  # c["padding_mask"] is usually 0s (False)

        self.full_padded_indices[i].zero_()
        self.full_padded_indices[i, :length] = c["full_padded_index"]

        self.leaf_mask[i].zero_()
        self.leaf_mask[i, :length] = c["leaf_mask"]

        self.edge_mask[i].zero_()
        self.edge_mask[i, :length] = c["edge_mask"]

        if i >= len(self.leaf_indices_list):
            self.leaf_indices_list.append(c["leaf_idx"])
            self.edge_split_masks_list.append(c["edge_split_masks"])
        else:
            self.leaf_indices_list[i] = c["leaf_idx"]
            self.edge_split_masks_list[i] = c["edge_split_masks"]

        n_node = c["node_num"]
        n_edge = c["edge_num"]

        # Check edge size
        if n_edge > self.edge_scatter_indices.size(1):
            raise RuntimeError(
                f"Edge num {n_edge} > capacity {self.edge_scatter_indices.size(1)}"
            )

        self.edge_scatter_indices[i].zero_()
        # Edges start at node_num in the token sequence
        indices = torch.arange(n_node, n_node + n_edge, device=self.device)
        self.edge_scatter_indices[i, :n_edge] = indices

        self.edge_scatter_mask[i].zero_()
        self.edge_scatter_mask[i, :n_edge] = True

    def update(self, idx, single_cache):
        self._set_cache_item(idx, single_cache)


def _worker_newick_parser(tree_str):
    if isinstance(tree_str, str):
        if "C(0)" in tree_str:
            tree_str = tree_str.replace("C(0)", '"C"')
        try:
            t = EteTree(tree_str, format=1, quoted_node_names=True)

        except Exception as e:
            # Fallback or re-raise
            raise ValueError(f"Failed to parse newick: {tree_str[:50]}...") from e
    else:
        # Assuming run in same process or serialized
        t = tree_str

    # Deterministic rooting for ambiguous roots only:
    # Equivalent unrooted trees can be serialized with a multifurcating root
    # (e.g., 3 children) vs a bifurcating root, which changes node/edge counts
    # and token lengths. For such ambiguous cases we re-root on a stable
    # outgroup. For already-bifurcating roots we preserve the serialized root
    # to keep directed split-mask orientation consistent with upstream TD2/BHV
    # split dictionaries.
    leaves = list(t.iter_leaves())
    if len(leaves) > 1 and len(t.children) > 2:
        leaves_by_name = {lf.name: lf for lf in leaves}
        outgroup = None
        for lbl in ("0", "1"):
            if lbl in leaves_by_name:
                outgroup = leaves_by_name[lbl]
                break
        if outgroup is None:
            try:
                outgroup = min(leaves, key=lambda x: int(x.name))
            except ValueError:
                outgroup = min(leaves, key=lambda x: x.name)
        t.set_outgroup(outgroup)

    # Canonicalize tree topology
    for node in t.traverse("postorder"):
        if node.is_leaf():
            try:
                sort_val = int(node.name)
            except ValueError:
                sort_val = float('inf')
        else:
            if node.children:
                sort_val = min(getattr(c, "sort_val", float('inf')) for c in node.children)
            else:
                sort_val = float('inf')
        node.add_feature("sort_val", sort_val)
    
    t.sort_descendants(attr="sort_val")

    # Postorder traversal and index assignment
    nodes = list(t.traverse("postorder"))

    leaf_masks = {}
    leaf_nodes = [node for node in nodes if node.is_leaf()]
    # Match BHV/Tree indexing: remap leaves to contiguous IDs by sorted name order.
    # Using raw labels directly (e.g., 1..N) shifts split bits and breaks alignment.
    try:
        leaf_nodes.sort(key=lambda x: int(x.name))
    except ValueError:
        leaf_nodes.sort(key=lambda x: x.name)

    for uid, node in enumerate(leaf_nodes):
        node.add_feature("uid", uid)
        leaf_masks[uid] = 1 << uid

    if not leaf_masks:
        # Single node tree or weird case
        max_uid = 0
    else:
        max_uid = max(leaf_masks.keys())

    next_internal_id = max_uid + 1
    for n in t.traverse("postorder"):
        if not n.is_leaf():
            if "uid" not in n.features:
                n.add_feature("uid", next_internal_id)
                next_internal_id += 1

    # Postorder accumulate subtree masks
    for node in t.traverse("postorder"):
        if not node.is_leaf():
            m = 0
            for ch in node.children:
                if ch.uid not in leaf_masks:
                    leaf_masks[ch.uid] = 0
                m |= int(leaf_masks[ch.uid])
            leaf_masks[node.uid] = m

    # Build split universe in contiguous BHV-compatible leaf index space.
    full = 0
    for leaf in t.iter_leaves():
        full |= (1 << int(leaf.uid))

    if full == 0:
        full = 1  # fallback for degenerate cases

    parent_list = []
    child_list = []
    branch_list = []
    edge_type_list = []

    split_mask_list = []

    # We iterate nodes again. ETE3 nodes are not pickle-safe effectively across boundaries if we rely on features?
    # Actually we just traverse 'nodes' list which we built.

    for parent in nodes:
        p_idx = parent.uid
        for child in parent.children:  # can be 0,1,2,... children
            c_idx = child.uid
            parent_list.append(p_idx)
            child_list.append(c_idx)

            bl = getattr(child, "dist", 1.0)
            branch_list.append(float(bl))

            et = getattr(child, "edge_type_id", 1)  # default 1
            edge_type_list.append(int(et))

            A = int(leaf_masks[child.uid])
            if A == 0 or A == full:
                split_mask_list.append(0)  # trivial / ignore
            else:
                # Keep the directed child-subtree mask instead of canonical min(A, full^A).
                # Autoregressive polytomy grouping relies on component-set structure
                # (maximal proper subsets), which is broken if leaf sides are flipped.
                split_mask_list.append(A)

    E = len(child_list)
    N = len(nodes)  # Rough node count based on traversal

    # Need to handle empty tree case
    if N == 0:
        return (
            np.zeros(1, dtype=np.int64),  # child_ptr
            np.zeros(0, dtype=np.int64),  # child_ids
            np.zeros(0, dtype=np.int64),  # parent_arr
            np.zeros(0, dtype=np.int64),  # child_arr
            0,  # root_idx
            np.zeros(1, dtype=np.float32),  # branch_lengths
            np.zeros(1, dtype=np.int64),  # edge_types
            [],  # edge_split_masks
        )

    child_arr = np.asarray(child_list, dtype=np.int64)
    order = np.argsort(child_arr)

    parent_arr = np.asarray(parent_list, dtype=np.int64)[order]
    child_arr = child_arr[order]
    branch_arr = np.asarray(branch_list, dtype=np.float32)[order]
    etype_arr = np.asarray(edge_type_list, dtype=np.int64)[order]

    ordered_split_mask = []
    for i in order:
        ordered_split_mask.append(split_mask_list[i])
    split_arr = ordered_split_mask

    # Root index
    root_idx = t.uid

    # CSR Construction
    counts = (
        np.bincount(parent_arr, minlength=next_internal_id)
        if E > 0
        else np.zeros((next_internal_id,), dtype=np.int64)
    )
    # Ensure size is enough (next_internal_id is usually N)

    child_ptr_arr = np.zeros((len(counts) + 1,), dtype=np.int64)
    np.cumsum(counts, out=child_ptr_arr[1:])

    child_ids_arr = np.empty((E,), dtype=np.int64)

    write_pos = child_ptr_arr[:-1].copy()  # current write offset per parent
    for p, c in zip(parent_arr, child_arr):
        if p < len(write_pos):
            j = write_pos[p]
            child_ids_arr[j] = c
            write_pos[p] += 1

    # Add root self-edge last
    parent_arr = np.concatenate([parent_arr, np.array([root_idx], dtype=np.int64)])
    child_arr = np.concatenate([child_arr, np.array([root_idx], dtype=np.int64)])
    branch_arr = np.concatenate([branch_arr, np.array([0.0], dtype=np.float32)])
    etype_arr = np.concatenate([etype_arr, np.array([0], dtype=np.int64)])
    split_arr.append(0)

    return (
        child_ptr_arr,
        child_ids_arr,
        parent_arr,
        child_arr,
        root_idx,
        branch_arr,
        etype_arr,
        split_arr,
    )


class TreeFeatureTokenizer(nn.Module):
    """
    TokenGT-style tokenizer for trees.
    Each token = sum of [node/edge attribute embedding, pairwise Laplacian PE, type embedding]
    """

    def __init__(
        self,
        num_node_types,
        num_edge_types,
        hidden_dim,
        n_layers=6,
        lap_dim=16,
        lap_dropout=0.2,
        orf_dim=16,
        max_nodes=100,
        identifier=["orf", "lap"],
        concat_features=False,
        branch_length_mode="linear",
        branch_length_num_buckets=64,
        branch_length_log_min=-8.0,
        branch_length_log_max=1.0,
        raw_graph_cache_vectorized=False,
    ):
        super().__init__()
        self.encoder_embed_dim = hidden_dim
        self.node_encoder = nn.Embedding(num_node_types, hidden_dim, padding_idx=0)
        self.edge_encoder = nn.Embedding(num_edge_types, hidden_dim, padding_idx=0)

        self.branch_length_mode = str(branch_length_mode)
        if self.branch_length_mode not in {"linear", "bucket"}:
            raise ValueError(
                "branch_length_mode must be one of {'linear', 'bucket'}, "
                f"got {branch_length_mode!r}."
            )
        self.branch_length_num_buckets = int(branch_length_num_buckets)
        if self.branch_length_num_buckets < 2:
            raise ValueError(
                "branch_length_num_buckets must be >= 2, "
                f"got {branch_length_num_buckets}."
            )
        self.branch_length_log_min = float(branch_length_log_min)
        self.branch_length_log_max = float(branch_length_log_max)
        if self.branch_length_log_max <= self.branch_length_log_min:
            raise ValueError(
                "branch_length_log_max must be > branch_length_log_min, "
                f"got {branch_length_log_min} and {branch_length_log_max}."
            )
        self.branch_length_encoder = nn.Linear(1, hidden_dim, bias=False)
        self.branch_length_bucket_embedding = nn.Embedding(
            self.branch_length_num_buckets,
            hidden_dim,
        )

        self.lap_dim = lap_dim
        self.lap_dropout = lap_dropout
        self.lap_encoder = nn.Linear(
            2 * lap_dim, hidden_dim, bias=False
        )  # Projects [PE_u, PE_v] to hidden_dim

        self.orf_dim = orf_dim
        self.orf_encoder = nn.Linear(2 * orf_dim, hidden_dim, bias=False)
        self.type_encoder = nn.Embedding(2, hidden_dim)  # 0=node, 1=edge
        self.identifier = identifier
        self.concat_features = concat_features
        self.raw_graph_cache_vectorized = bool(raw_graph_cache_vectorized)
        if self.concat_features:
            self.feature_combiner = nn.Linear(3 * hidden_dim, hidden_dim)
        else:
            self.feature_combiner = nn.Identity()

        m = max(max_nodes, orf_dim)
        random_matrix = torch.randn(m, m)
        q, _ = torch.linalg.qr(random_matrix)
        self.register_buffer("orf_matrix", q)

        self.apply(lambda module: self.init_params(module, n_layers=n_layers))

    def _branch_length_bucket_ids(self, branch_lengths: torch.Tensor) -> torch.Tensor:
        lengths = branch_lengths.float()
        bucket_ids = torch.zeros_like(lengths, dtype=torch.long)
        positive = lengths > 0.0
        if not bool(positive.any()):
            return bucket_ids

        log_lengths = torch.log(lengths[positive].clamp_min(1e-12))
        scaled = (log_lengths - self.branch_length_log_min) / (
            self.branch_length_log_max - self.branch_length_log_min
        )
        max_nonzero_bucket = self.branch_length_num_buckets - 2
        clipped = torch.clamp(scaled, 0.0, 1.0)
        bucket_vals = torch.floor(clipped * max_nonzero_bucket).to(torch.long) + 1
        bucket_ids[positive] = bucket_vals
        return bucket_ids

    def encode_branch_lengths(self, branch_lengths: torch.Tensor) -> torch.Tensor:
        if self.branch_length_mode == "bucket":
            bucket_ids = self._branch_length_bucket_ids(branch_lengths)
            return self.branch_length_bucket_embedding(bucket_ids.to(branch_lengths.device))
        return self.branch_length_encoder(branch_lengths.unsqueeze(-1).to(branch_lengths.device))

    @staticmethod
    def init_params(module, n_layers):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02 / n_layers**0.5)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def sinusoidal_pos_enc(self, n_positions: int, dim: int, device):
        position = torch.arange(n_positions, device=device).float()  # [n]
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device).float() * -(math.log(10000.0) / dim)
        )  # [dim/2]
        pe = torch.zeros(n_positions, dim, device=device)  # [n, d]
        pe[:, 0::2] = torch.sin(position.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(position.unsqueeze(1) * div_term)
        return pe

    def _ensure_lap_dim(self, lap: torch.Tensor) -> torch.Tensor:
        # Ensure lap has exactly self.lap_dim columns
        N, d = lap.size(0), lap.size(1)
        k = self.lap_dim
        if d == k:
            return lap
        if d > k:
            return lap[:, :k]
        # d < k
        pad = torch.zeros((N, k - d), dtype=lap.dtype, device=lap.device)
        return torch.cat([lap, pad], dim=1)

    def lap_pe_from_edge_index(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        k: int,
        device=None,
    ) -> torch.Tensor:
        """
        edge_index: Long[2,E] directed parent->child edges. Self-edges are ignored.
        Returns LapPE: Float[N,k].
        """
        dev = device or edge_index.device
        N = int(num_nodes)
        if N == 0 or k == 0:
            return torch.zeros((N, k), dtype=torch.float32, device=dev)

        if edge_index.numel() == 0:
            return torch.zeros((N, k), dtype=torch.float32, device=dev)

        edge_index = edge_index.to(dev)
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}"
            )

        non_self = edge_index[0] != edge_index[1]
        if not bool(non_self.any()):
            return torch.zeros((N, k), dtype=torch.float32, device=dev)

        src = edge_index[0, non_self].long()
        dst = edge_index[1, non_self].long()

        # Create dense adjacency matrix
        # Note: For very large N, sparse would be better, but trees are usually manageable
        A = torch.zeros((N, N), device=dev, dtype=torch.float32)
        A[src, dst] = 1.0
        A[dst, src] = 1.0  # Symmetric

        # Degree
        deg = A.sum(dim=1)
        D = torch.diag(deg)
        L = D - A

        # Eigendecomposition (Symmetric)
        # Returns eigenvalues in ascending order and corresponding eigenvectors
        # vals: (N,), vecs: (N, N)
        vals, vecs = torch.linalg.eigh(L)

        # Skip the first trivial eigenvector (corresponding to 0 eigenvalue)
        # We want up to k eigenvectors
        # Check available count logic from scipy version: target = min(k, max(N - 1, 0))
        target_k = min(k, max(N - 1, 0))

        if target_k == 0:
            return torch.zeros((N, k), dtype=torch.float32, device=dev)

        # Select columns 1 to target_k + 1
        output_vecs = vecs[:, 1 : target_k + 1]

        # Pad if needed to reach exactly k columns (handled by _ensure_lap_dim)
        return self._ensure_lap_dim(output_vecs)

    def lap_pe_torch(self, children: torch.Tensor, k: int, device=None) -> torch.Tensor:
        """
        children: Long[N,2], -1 for missing. Returns LapPE: Float[N,k].
        Backwards-compatible wrapper for binary-tree callers.
        """
        dev = device or children.device
        N = children.size(0)
        if N == 0 or k == 0:
            return torch.zeros((N, k), dtype=torch.float32, device=dev)

        mask = children >= 0
        if not mask.any():
            return torch.zeros((N, k), dtype=torch.float32, device=dev)

        row_idx = torch.arange(N, device=dev).unsqueeze(1).expand_as(children)
        src = row_idx[mask]
        dst = children[mask]
        edge_index = torch.stack([src, dst], dim=0)
        return self.lap_pe_from_edge_index(edge_index, num_nodes=N, k=k, device=dev)

    def lap_pe_scipy(self, children: torch.Tensor, k: int, device=None) -> torch.Tensor:
        """
        children: Long[N,2], -1 for missing. Returns LapPE: Float[N,k].
        Always returns exactly k columns by requesting k+1 smallest and dropping the trivial.
        """
        dev = device or children.device
        N = int(children.size(0))
        if N == 0 or k == 0:
            return torch.zeros((N, k), dtype=torch.float32, device=dev)

        c0 = children[:, 0].detach().cpu().numpy()
        c1 = children[:, 1].detach().cpu().numpy()
        rows, cols = [], []
        if np.any(c0 >= 0):
            u = np.nonzero(c0 >= 0)[0]
            v = c0[c0 >= 0].astype(np.int64)
            rows += [u, v]
            cols += [v, u]
        if np.any(c1 >= 0):
            u = np.nonzero(c1 >= 0)[0]
            v = c1[c1 >= 0].astype(np.int64)
            rows += [u, v]
            cols += [v, u]

        if not rows:
            return torch.zeros((N, k), dtype=torch.float32, device=dev)

        row = np.concatenate(rows)
        col = np.concatenate(cols)
        data = np.ones_like(row, dtype=np.float64)

        A = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()
        deg = np.asarray(A.sum(axis=1)).ravel()
        L = sp.diags(deg) - A

        # number of non-trivial eigenvectors the graph can actually provide
        target = min(k, max(N - 1, 0))
        if target == 0:
            return torch.zeros((N, k), dtype=torch.float32, device=dev)

        # Request one extra (k+1) to safely drop the trivial eigenvector,
        # but eigsh requires k < N; clamp accordingly.
        k_req = min(target + 1, N - 1)  # N-1 smallest (excluding dimension issues)
        # For very small N (e.g., N=2), k_req may equal target==1; that's fine.

        vals, vecs = spla.eigsh(L, k=k_req, which="SM")  # smallest magnitude

        # Sort, then drop the smallest (≈0), keep next 'target'
        order = np.argsort(vals)
        vecs = vecs[:, order]
        # It’s possible numerical ordering puts the zero not strictly first;
        # the sort ensures we drop the smallest.
        vecs = vecs[:, 1 : 1 + target]

        # Pad to exactly k columns
        if target < k:
            vecs = np.pad(vecs, ((0, 0), (0, k - target)), mode="constant")

        out = torch.from_numpy(vecs.astype(np.float32)).to(dev)
        # Optional TokenGT random sign flip + dropout (training only)
        if self.training and out.numel() > 0:
            sign = (torch.randint(0, 2, (1, out.size(1)), device=dev) * 2 - 1).float()
            out = out * sign
            if self.lap_dropout > 0:
                out = out * (torch.rand_like(out) > self.lap_dropout)
        return out

    def compute_laplacian_eigvecs(self, tree, k=None, device=None):
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        else:
            device = (
                torch.device(device) if not isinstance(device, torch.device) else device
            )
        k = k if k is not None else self.lap_dim
        node_list = list(tree.traverse("preorder"))

        # Check if all node names can be converted to integers
        try:
            node_list = sorted(node_list, key=lambda n: int(n.name))
        except ValueError as e:
            raise ValueError(
                f"Tree nodes must have numeric names, but found non-numeric name. "
                f"This usually means you need to call tree_numbering() function from treeVector.py first. "
                f"Original error: {e}"
            ) from e
        n = len(node_list)
        if n == 0:
            return torch.zeros((0, k), dtype=torch.float32, device=device)

        idx_map = {node: i for i, node in enumerate(node_list)}
        adj = np.zeros((n, n), dtype=np.float32)
        for node in node_list:
            for child in node.children:
                i, j = idx_map[node], idx_map[child]
                adj[i, j] = 1
                adj[j, i] = 1
        D_val = np.diag(adj.sum(axis=1))
        L = D_val - adj
        if n == 1:
            eigvecs = np.zeros((1, 0), dtype=np.float32)
        else:
            try:
                eigvals, eigvecs = np.linalg.eigh(L)
                max_eigs = min(k, n - 1 if n > 0 else 0)
                idx = np.argsort(eigvals)
                start_idx = 1 if n > 1 and max_eigs > 0 else 0
                idx_to_take = idx[start_idx : start_idx + max_eigs]
                eigvecs = (
                    eigvecs[:, idx_to_take]
                    if max_eigs > 0
                    else np.zeros((n, 0), dtype=np.float32)
                )
            except np.linalg.LinAlgError:
                eigvecs = np.zeros((n, 0), dtype=np.float32)
        if eigvecs.shape[1] < k:
            pad_width = k - eigvecs.shape[1]
            eigvecs = np.pad(eigvecs, ((0, 0), (0, pad_width)), mode="constant")
        if eigvecs.shape[1] > k:
            eigvecs = eigvecs[:, :k]
        eigvecs = torch.tensor(eigvecs, dtype=torch.float32, device=device)
        if eigvecs.size(0) > 0 and eigvecs.size(1) > 0 and self.training:
            # TokenGT: random sign flip and LapPE dropout, only during training
            sign_flip = torch.randint(0, 2, (1, eigvecs.size(1)), device=device) * 2 - 1
            eigvecs = eigvecs * sign_flip
            if self.lap_dropout > 0:
                dropout_mask = torch.rand_like(eigvecs) > self.lap_dropout
                eigvecs = eigvecs * dropout_mask
        return eigvecs

    def compute_laplacian_eigvecs_from_children(
        self, children: torch.Tensor, k: int, device=None
    ):
        """
        children: Long[N,2], -1 for missing child.
        Returns LapPE: Float[N, k] (skipping the trivial eigenvector; padded if N-1 < k).
        Uses torch (GPU-capable). For large N, uses LOBPCG; else eigh.
        """
        device = device or next(self.parameters()).device
        children = children.to(device)
        N = children.size(0)
        if N == 0 or k == 0:
            return torch.zeros((N, k), dtype=torch.float32, device=device)

        # Build symmetric adjacency from children
        rows = []
        cols = []
        c0 = children[:, 0]
        c1 = children[:, 1]
        u = torch.arange(N, device=device)

        mask0 = c0 >= 0
        mask1 = c1 >= 0
        if mask0.any():
            rows.append(u[mask0])
            cols.append(c0[mask0])
            rows.append(c0[mask0])
            cols.append(u[mask0])
        if mask1.any():
            rows.append(u[mask1])
            cols.append(c1[mask1])
            rows.append(c1[mask1])
            cols.append(u[mask1])

        if rows:
            row = torch.cat(rows)
            col = torch.cat(cols)
            A = torch.zeros((N, N), dtype=torch.float32, device=device)
            A.index_put_(
                (row, col), torch.ones_like(row, dtype=torch.float32), accumulate=True
            )
        else:
            A = torch.zeros((N, N), dtype=torch.float32, device=device)

        deg = A.sum(dim=1)
        L = torch.diag(deg) - A

        # We want the first k non-trivial eigenvectors (skip the constant one)
        need = min(k, max(N - 1, 0))
        if need == 0:
            return torch.zeros((N, k), dtype=torch.float32, device=device)

        # Choose solver
        use_lobpcg = N >= 800  # heuristic; tune if needed

        if use_lobpcg:
            # LOBPCG needs a symmetric positive semidefinite operator
            # Compute a few smallest eigenpairs; add tiny shift to improve stability
            X = torch.randn(N, need, device=device)
            vals, vecs = torch.lobpcg(
                L + 1e-6 * torch.eye(N, device=device),
                k=need,
                B=None,
                X=X,
                largest=False,
                tol=1e-3,
                maxiter=200,
            )
            # Remove (near-)zero eigenvector if included
            # We’ll sort by eigenvalue and skip the smallest one
            order = torch.argsort(vals)
            vals = vals[order]
            vecs = vecs[:, order]
            vecs_nontriv = vecs[:, 1 : 1 + need]
        else:
            vals, vecs = torch.linalg.eigh(L)  # sorted ascending
            vecs_nontriv = vecs[:, 1 : 1 + need]

        # Pad to k if need < k
        if need < k:
            pad = torch.zeros((N, k - need), device=device, dtype=torch.float32)
            vecs_k = torch.cat([vecs_nontriv, pad], dim=1)
        else:
            vecs_k = vecs_nontriv[:, :k]

        # TokenGT random sign flip + dropout during training
        if self.training and vecs_k.numel() > 0:
            sign = (torch.randint(0, 2, (1, vecs_k.size(1)), device=device) * 2 - 1).to(
                vecs_k.dtype
            )
            vecs_k = vecs_k * sign
            if self.lap_dropout > 0:
                vecs_k = vecs_k * (torch.rand_like(vecs_k) > self.lap_dropout)

        return vecs_k.to(torch.float32)

    def is_structural_pack(self, x):
        # Batched structural pack: (children[BK,Nmax,2], root_BK[BK][, ...])
        if not isinstance(x, (tuple, list)) or len(x) < 2:
            return False
        ch, rt = x[0], x[1]
        return (
            isinstance(ch, torch.Tensor)
            and ch.dim() == 3
            and ch.size(-1) == 2
            and isinstance(rt, torch.Tensor)
            and rt.dim() == 1
        )

    def tree_to_graph_from_children(
        self,
        child_ptr: torch.Tensor,
        child_ids: torch.Tensor,
        parent_arr: torch.Tensor,
        child_arr: torch.Tensor,
        root_idx: int,
        branch_lengths: torch.Tensor | None = None,
        edge_types: torch.Tensor | None = None,
    ):
        """
        Inputs:
        children      : Long[N,2] with -1 for missing
        parent_ids    : Long[N] parent indices for each node
        root_idx      : int
        branch_lengths: Optional Float[E] aligned with edge_index after creation; if None, set 1.0; root self-edge 0.0
        edge_types    : Optional Long[E]; default 1; root self-edge 0

        Returns (matches your tree_to_graph outputs):
        node_data, edge_index, edge_data, branch_lengths, node_num, edge_num,
        lap_eigvecs, leaf_mask, leaf_idx, sin_embed_node, sin_embed_edge
        """
        N = child_ptr.numel() - 1
        device = child_ptr.device

        deg_out = child_ptr[1:] - child_ptr[:-1]
        is_leaf = deg_out == 0
        node_data = torch.where(
            is_leaf,
            torch.ones(N, dtype=torch.long, device=device),
            torch.full((N,), 2, dtype=torch.long, device=device),
        )

        edge_index = torch.stack([parent_arr, child_arr], dim=0)
        E = edge_index.size(1)

        # Edge data / branch lengths
        if edge_types is None:
            edge_data = torch.ones(E, dtype=torch.long, device=device)
            edge_data[-1] = 0  # root self-edge
        else:
            edge_data = edge_types.to(device)

        if branch_lengths is None:
            branch_lengths = torch.ones(E, dtype=torch.float32, device=device)
            branch_lengths[-1] = 0.0
        else:
            branch_lengths = branch_lengths.to(device)

        # Positional encodings
        sin_embed_node = self.sinusoidal_pos_enc(N, self.encoder_embed_dim, device)
        sin_embed_edge = self.sinusoidal_pos_enc(E, self.encoder_embed_dim, device)

        # Laplacian PE (N x lap_dim) from the full tree adjacency.
        # This preserves non-binary structure instead of truncating polytomies
        # down to two children for positional encoding.
        lap_eigvecs = self.lap_pe_from_edge_index(
            edge_index=edge_index,
            num_nodes=N,
            k=self.lap_dim,
            device=device,
        )
        lap_eigvecs = self._ensure_lap_dim(lap_eigvecs)

        # Leaf mask for tokens: you had nodes first then edges.
        leaf_mask_nodes = is_leaf.clone()
        # force ids 0 and 1 not to be treated as leaves in your convention
        # REMOVING THIS, THIS IS A BUG
        # if N > 1:
        #     leaf_mask_nodes[0] = False
        #     leaf_mask_nodes[1] = False

        leaf_mask = torch.cat(
            [leaf_mask_nodes, torch.zeros(E, dtype=torch.bool, device=device)], dim=0
        )
        # leaf_idx = preorder indices for tips 2..num_leaves (here: node IDs themselves are already integers)
        # If your IDs are consistent (0..), then leaf indices are simply the node ids of leaves excluding 0,1.
        leaf_ids = torch.nonzero(leaf_mask_nodes, as_tuple=True)[0]
        leaf_ids = leaf_ids[leaf_ids >= 2]
        leaf_idx = leaf_ids.to(torch.long)

        return (
            node_data,
            edge_index,
            edge_data,
            branch_lengths,
            N,
            E,
            lap_eigvecs,
            leaf_mask,
            leaf_idx,
            sin_embed_node,
            sin_embed_edge,
        )

    def tree_to_graph(self, tree):
        """Node list is the tree in preorder"""
        node_list = list(tree.traverse("preorder"))
        node_list = sorted(node_list, key=lambda n: int(n.name))
        device = torch.device("cpu")

        name_to_type = {
            int(node.name): (1 if node.is_leaf() else 2) for node in node_list
        }
        num_leaves = sum(1 for node in node_list if node.is_leaf())
        name_to_preorder_idx = {int(node.name): i for i, node in enumerate(node_list)}
        node_data = torch.tensor(
            [name_to_type[int(i.name)] for i in node_list],
            dtype=torch.long,
            device=device,
        )

        # Build edge list: each edge is (parent, child), ordered by child node name
        edge_tuples = []
        edge_data_list = []
        branch_length_list = []
        for node in node_list:
            for child in node.children:
                edge_tuples.append((int(node.name), int(child.name)))
                edge_type = getattr(child, "edge_type_id", 1)
                edge_data_list.append(edge_type)
                branch_length = getattr(child, "dist", 1.0)
                branch_length_list.append(branch_length)
        # Add root node connection: root is node with no parent
        root_nodes = [n for n in node_list if n.up is None]

        for root in root_nodes:
            # Convention: connect root to itself, length 0
            edge_tuples.append((int(root.name), int(root.name)))
            edge_data_list.append(0)  # or a special edge type for root
            branch_length_list.append(0.0)

        # Order edges by child node name
        edge_order = sorted(range(len(edge_tuples)), key=lambda i: edge_tuples[i][1])
        edge_tuples = [edge_tuples[i] for i in edge_order]
        edge_data_list = [edge_data_list[i] for i in edge_order]
        branch_length_list = [branch_length_list[i] for i in edge_order]
        edge_index_list = [
            [name_to_preorder_idx[parent], name_to_preorder_idx[child]]
            for parent, child in edge_tuples
        ]
        edge_index = (
            torch.tensor(edge_index_list, dtype=torch.long, device=device).t()
            if edge_index_list
            else torch.zeros(2, 0, dtype=torch.long, device=device)
        )
        edge_data = (
            torch.tensor(edge_data_list, dtype=torch.long, device=device)
            if edge_data_list
            else torch.zeros(0, dtype=torch.long, device=device)
        )
        branch_lengths = (
            torch.tensor(branch_length_list, dtype=torch.float32, device=device)
            if branch_length_list
            else torch.zeros(0, dtype=torch.float32, device=device)
        )
        node_num = [len(node_list)]
        edge_num = [edge_index.size(1)]
        sin_embed_node = self.sinusoidal_pos_enc(
            node_num[0], self.encoder_embed_dim, node_data.device
        )
        sin_embed_edge = self.sinusoidal_pos_enc(
            edge_num[0], self.encoder_embed_dim, edge_data.device
        )
        lap_eigvecs = self.compute_laplacian_eigvecs(
            tree, k=self.lap_dim, device=node_data.device
        )
        leaf_mask_nodes = torch.tensor(
            [name_to_type[int(node.name)] == 1 for node in node_list],
            device=device,
        )
        leaf_mask_nodes[name_to_preorder_idx[0]] = 0
        leaf_mask_nodes[name_to_preorder_idx[1]] = 0
        leaf_mask = torch.cat(
            [
                leaf_mask_nodes,
                torch.zeros(edge_num[0], dtype=torch.bool, device=node_data.device),
            ]
        )
        leaf_idx = torch.tensor(
            [name_to_preorder_idx[i] for i in range(2, num_leaves)],
            dtype=torch.long,
            device=device,
        )
        return (
            node_data,
            edge_index,
            edge_data,
            branch_lengths,
            node_num,
            edge_num,
            lap_eigvecs,
            leaf_mask,
            leaf_idx,
            sin_embed_node,
            sin_embed_edge,
        )

    def forward(self, trees):
        if isinstance(trees, (str, EteTree)):
            trees = [trees]

        if (
            isinstance(trees, list)
            and len(trees) > 0
            and all(
                isinstance(t, dict)
                and bool(t.get("_tree_tokenizer_raw_graph_cache", False))
                for t in trees
            )
        ):
            return self.forward_raw_graph_cache(trees)

        # List of Newick strings / ETE3 Trees
        if isinstance(trees, list) and len(trees) > 0:
            if all(isinstance(t, (str, EteTree)) for t in trees):
                trees = [self._newick_to_structural(t) for t in trees]

        # Output is: (child_ptr, child_ids, parent_ids, root_idx, branch_lengths, edge_types)

        batch_size = len(trees)

        # if batch_size == 1:
        #     return self._forward_single_tree(trees[0])

        # Process each tree and collect results
        batch_features = []
        batch_padding_masks = []
        batch_padded_indices = []
        batch_leaf_masks = []
        batch_leaf_indices = []
        batch_edge_masks = []
        batch_edge_split_masks = []
        max_tokens = 0

        for tree in trees:
            (
                features,
                padding_mask,
                padded_index,
                leaf_mask_single,
                leaf_idx_single,
                edge_mask,
                edge_split_masks,
            ) = self._forward_single_tree(tree)
            # Remove batch dimension from single tree output
            features = features.squeeze(0)
            padding_mask = padding_mask.squeeze(0)
            padded_index = padded_index.squeeze(0)

            batch_features.append(features)
            batch_padding_masks.append(padding_mask)
            batch_padded_indices.append(padded_index)
            batch_leaf_masks.append(leaf_mask_single)
            batch_leaf_indices.append(leaf_idx_single)
            batch_edge_masks.append(edge_mask)
            batch_edge_split_masks.append(edge_split_masks)
            max_tokens = max(max_tokens, features.size(0))

        device = batch_features[0].device

        # Pad sequences to max length
        padded_features = torch.zeros(
            (batch_size, max_tokens, self.encoder_embed_dim), device=device
        )
        padded_masks = torch.ones(
            (batch_size, max_tokens), device=device, dtype=torch.bool
        )  # True = padded
        padded_indices = torch.zeros(
            (batch_size, max_tokens, 2), device=device, dtype=torch.long
        )
        padded_leaf_masks = torch.zeros(
            (batch_size, max_tokens), device=device, dtype=torch.bool
        )

        padded_edge_masks = torch.zeros(
            (batch_size, max_tokens), device=device, dtype=torch.bool
        )

        for i, (
            features,
            mask,
            indices,
            leaf_mask_single,
            leaf_idx_single,
            edge_mask,
        ) in enumerate(
            zip(
                batch_features,
                batch_padding_masks,
                batch_padded_indices,
                batch_leaf_masks,
                batch_leaf_indices,
                batch_edge_masks,
            )
        ):
            seq_len = features.size(0)
            if seq_len > 0:
                padded_features[i, :seq_len] = features
                padded_masks[i, :seq_len] = mask
                padded_indices[i, :seq_len] = indices
                padded_leaf_masks[i, :seq_len] = leaf_mask_single
                padded_edge_masks[i, :seq_len] = edge_mask

        return (
            padded_features,
            padded_masks,
            padded_indices,
            padded_leaf_masks,
            batch_leaf_indices,
            padded_edge_masks,
            batch_edge_split_masks,
        )

    def _forward_single_tree(self, tree_info):

        # Output from newick to structure is (child_ptr, child_ids, parent_arr, child_arr, root_idx, branch_lengths, edge_types, split_mask_list)
        child_ptr = tree_info[0]
        child_ids = tree_info[1]
        parent_arr = tree_info[2]
        child_arr = tree_info[3]
        root_idx = tree_info[4]
        branch_lengths = tree_info[5]
        edge_types = tree_info[6]
        edge_split_masks = tree_info[7]

        device = next(self.parameters()).device
        if isinstance(child_ptr, np.ndarray):
            child_ptr = torch.from_numpy(child_ptr).to(device)
            child_ids = torch.from_numpy(child_ids).to(device)
            parent_arr = torch.from_numpy(parent_arr).to(device)
            child_arr = torch.from_numpy(child_arr).to(device)
            branch_lengths = torch.from_numpy(branch_lengths).to(device)
            edge_types = torch.from_numpy(edge_types).to(device)
        else:
            device = child_ptr.device

        (
            node_data,
            edge_index,
            edge_data,
            branch_lengths,
            node_num,
            edge_num,
            lap_pe,
            leaf_mask,
            leaf_idx,
            sin_embed_node,
            sin_embed_edge,
        ) = self.tree_to_graph_from_children(
            child_ptr,
            child_ids,
            parent_arr,
            child_arr,
            root_idx,
            branch_lengths,
            edge_types,
        )

        return self._process_and_pack_single(
            node_data,
            edge_index,
            edge_data,
            branch_lengths,
            node_num,
            edge_num,
            lap_pe,
            leaf_mask,
            leaf_idx,
            sin_embed_node,
            sin_embed_edge,
            device,
            edge_split_masks,
        )

    def _process_and_pack_single(
        self,
        node_data,
        edge_index,
        edge_data,
        branch_lengths,
        node_num,
        edge_num,
        lap_pe,
        leaf_mask,
        leaf_idx,
        sin_embed_node,
        sin_embed_edge,
        device,
        edge_split_masks,
    ):

        node_indices = torch.arange(node_num, device=device)
        node_attr_embedding = self.node_encoder(node_data) + sin_embed_node
        node_pairs = torch.stack([node_indices, node_indices], dim=1)

        edge_attr_embedding = self.edge_encoder(edge_data)
        if edge_attr_embedding.size(0) > 0:
            branch_length_feat = self.encode_branch_lengths(branch_lengths.to(device))
            edge_attr_embedding = (
                edge_attr_embedding + branch_length_feat + sin_embed_edge
            )
        edge_pairs = edge_index.t()

        full_attr_embedding = torch.cat(
            [node_attr_embedding, edge_attr_embedding], dim=0
        )
        full_padded_index = torch.cat([node_pairs, edge_pairs], dim=0)

        type_ids = torch.cat(
            [
                torch.zeros(node_num, dtype=torch.long, device=device),
                torch.ones(edge_num, dtype=torch.long, device=device),
            ]
        )
        type_embedding = self.type_encoder(type_ids)

        u = full_padded_index[:, 0]
        v = full_padded_index[:, 1]
        pos_pe_concat = torch.cat([lap_pe[u], lap_pe[v]], dim=1)
        pos_embedding = self.lap_encoder(pos_pe_concat)

        if self.concat_features:
            final_token_features = self.feature_combiner(
                torch.cat([full_attr_embedding, type_embedding, pos_embedding], dim=1)
            )
            edge_mask = torch.cat(
                [
                    torch.zeros(node_num, dtype=torch.bool, device=device),
                    torch.ones(edge_num, dtype=torch.bool, device=device),
                    torch.zeros(
                        type_embedding.size(0), dtype=torch.bool, device=device
                    ),
                    torch.zeros(pos_embedding.size(0), dtype=torch.bool, device=device),
                ]
            )
        else:
            final_token_features = full_attr_embedding + type_embedding + pos_embedding
            edge_mask = torch.cat(
                [
                    torch.zeros(node_num, dtype=torch.bool, device=device),
                    torch.ones(edge_num, dtype=torch.bool, device=device),
                ]
            )

        padding_mask = torch.zeros(
            final_token_features.size(0), dtype=torch.bool, device=device
        )

        # Return unbatched tensors for collate
        return (
            final_token_features,
            padding_mask,
            full_padded_index,
            leaf_mask,
            leaf_idx,
            edge_mask,
            edge_split_masks,
        )

    def raw_graph_cache_from_structural(self, tree_info, device=None):
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        else:
            device = torch.device(device) if not isinstance(device, torch.device) else device

        child_ptr = tree_info[0]
        child_ids = tree_info[1]
        parent_arr = tree_info[2]
        child_arr = tree_info[3]
        root_idx = int(tree_info[4])
        branch_lengths = tree_info[5]
        edge_types = tree_info[6]
        edge_split_masks = [int(mask) for mask in tree_info[7]]

        if not torch.is_tensor(child_ptr):
            child_ptr = torch.as_tensor(child_ptr, dtype=torch.long, device=device)
            child_ids = torch.as_tensor(child_ids, dtype=torch.long, device=device)
            parent_arr = torch.as_tensor(parent_arr, dtype=torch.long, device=device)
            child_arr = torch.as_tensor(child_arr, dtype=torch.long, device=device)
            branch_lengths = torch.as_tensor(
                branch_lengths,
                dtype=torch.float32,
                device=device,
            )
            edge_types = torch.as_tensor(edge_types, dtype=torch.long, device=device)
        else:
            child_ptr = child_ptr.to(device=device, dtype=torch.long)
            child_ids = child_ids.to(device=device, dtype=torch.long)
            parent_arr = parent_arr.to(device=device, dtype=torch.long)
            child_arr = child_arr.to(device=device, dtype=torch.long)
            branch_lengths = branch_lengths.to(device=device, dtype=torch.float32)
            edge_types = edge_types.to(device=device, dtype=torch.long)

        (
            node_data,
            edge_index,
            edge_data,
            branch_lengths,
            node_num,
            edge_num,
            lap_pe,
            leaf_mask,
            leaf_idx,
            sin_embed_node,
            sin_embed_edge,
        ) = self.tree_to_graph_from_children(
            child_ptr,
            child_ids,
            parent_arr,
            child_arr,
            root_idx,
            branch_lengths,
            edge_types,
        )

        node_indices = torch.arange(int(node_num), device=device)
        node_pairs = torch.stack([node_indices, node_indices], dim=1)
        edge_pairs = edge_index.t()
        full_padded_index = torch.cat([node_pairs, edge_pairs], dim=0)

        return {
            "_tree_tokenizer_raw_graph_cache": True,
            "node_data": node_data.detach(),
            "edge_data": edge_data.detach(),
            "branch_lengths": branch_lengths.detach(),
            "node_num": int(node_num),
            "edge_num": int(edge_num),
            "lap_pe": lap_pe.detach(),
            "leaf_mask": leaf_mask.detach(),
            "leaf_idx": leaf_idx.detach(),
            "sin_embed_node": sin_embed_node.detach(),
            "sin_embed_edge": sin_embed_edge.detach(),
            "full_padded_index": full_padded_index.detach(),
            "edge_split_masks": edge_split_masks,
        }

    def compute_raw_graph_cache(self, tree_list):
        if isinstance(tree_list, (str, EteTree)):
            tree_list = [tree_list]
        caches = []
        for tree in tree_list:
            if (
                isinstance(tree, dict)
                and bool(tree.get("_tree_tokenizer_raw_graph_cache", False))
            ):
                caches.append(tree)
                continue
            structural = (
                tree
                if isinstance(tree, tuple)
                else _worker_newick_parser(tree)
            )
            caches.append(self.raw_graph_cache_from_structural(structural))
        return caches

    def _forward_single_raw_graph_cache(self, cache):
        device = next(self.parameters()).device

        def _tensor(key, dtype=None):
            value = cache[key]
            if torch.is_tensor(value):
                return value.to(device=device, dtype=dtype) if dtype is not None else value.to(device)
            return torch.as_tensor(value, device=device, dtype=dtype)

        node_data = _tensor("node_data", torch.long)
        edge_data = _tensor("edge_data", torch.long)
        branch_lengths = _tensor("branch_lengths", torch.float32)
        node_num = int(cache["node_num"])
        edge_num = int(cache["edge_num"])
        lap_pe = _tensor("lap_pe", torch.float32)
        leaf_mask = _tensor("leaf_mask", torch.bool)
        leaf_idx = _tensor("leaf_idx", torch.long)
        sin_embed_node = _tensor("sin_embed_node", torch.float32)
        sin_embed_edge = _tensor("sin_embed_edge", torch.float32)
        full_padded_index = _tensor("full_padded_index", torch.long)
        edge_split_masks = [int(mask) for mask in cache["edge_split_masks"]]

        node_attr_embedding = self.node_encoder(node_data) + sin_embed_node
        edge_attr_embedding = self.edge_encoder(edge_data)
        if edge_attr_embedding.size(0) > 0:
            branch_length_feat = self.encode_branch_lengths(branch_lengths)
            edge_attr_embedding = (
                edge_attr_embedding + branch_length_feat + sin_embed_edge
            )

        full_attr_embedding = torch.cat(
            [node_attr_embedding, edge_attr_embedding],
            dim=0,
        )
        type_ids = torch.cat(
            [
                torch.zeros(node_num, dtype=torch.long, device=device),
                torch.ones(edge_num, dtype=torch.long, device=device),
            ]
        )
        type_embedding = self.type_encoder(type_ids)

        u = full_padded_index[:, 0]
        v = full_padded_index[:, 1]
        pos_pe_concat = torch.cat([lap_pe[u], lap_pe[v]], dim=1)
        pos_embedding = self.lap_encoder(pos_pe_concat)

        if self.concat_features:
            final_token_features = self.feature_combiner(
                torch.cat(
                    [full_attr_embedding, type_embedding, pos_embedding],
                    dim=1,
                )
            )
            edge_mask = torch.cat(
                [
                    torch.zeros(node_num, dtype=torch.bool, device=device),
                    torch.ones(edge_num, dtype=torch.bool, device=device),
                    torch.zeros(
                        type_embedding.size(0),
                        dtype=torch.bool,
                        device=device,
                    ),
                    torch.zeros(
                        pos_embedding.size(0),
                        dtype=torch.bool,
                        device=device,
                    ),
                ]
            )
        else:
            final_token_features = full_attr_embedding + type_embedding + pos_embedding
            edge_mask = torch.cat(
                [
                    torch.zeros(node_num, dtype=torch.bool, device=device),
                    torch.ones(edge_num, dtype=torch.bool, device=device),
                ]
            )

        padding_mask = torch.zeros(
            final_token_features.size(0),
            dtype=torch.bool,
            device=device,
        )

        return (
            final_token_features,
            padding_mask,
            full_padded_index,
            leaf_mask,
            leaf_idx,
            edge_mask,
            edge_split_masks,
        )

    def _forward_raw_graph_batch_cache(self, cache):
        device = next(self.parameters()).device

        def _tensor(key, dtype=None):
            value = cache[key]
            if torch.is_tensor(value):
                return (
                    value.to(device=device, dtype=dtype)
                    if dtype is not None
                    else value.to(device)
                )
            return torch.as_tensor(value, device=device, dtype=dtype)

        batch_size = int(cache["batch_size"])
        max_tokens = int(cache["max_tokens"])
        node_data = _tensor("node_data", torch.long)
        edge_data = _tensor("edge_data", torch.long)
        branch_lengths = _tensor("branch_lengths", torch.float32)
        lap_pe = _tensor("lap_pe", torch.float32)
        sin_embed_node = _tensor("sin_embed_node", torch.float32)
        sin_embed_edge = _tensor("sin_embed_edge", torch.float32)
        flat_lap_indices = _tensor("flat_lap_indices", torch.long)
        flat_type_ids = _tensor("flat_type_ids", torch.long)
        flat_batch_indices = _tensor("flat_batch_indices", torch.long)
        flat_token_positions = _tensor("flat_token_positions", torch.long)
        node_token_positions = _tensor("node_token_positions", torch.long)
        edge_token_positions = _tensor("edge_token_positions", torch.long)

        total_tokens = int(flat_type_ids.numel())
        total_nodes = int(node_data.size(0))
        total_edges = int(edge_data.size(0))
        flat_attr = torch.zeros(
            (total_tokens, self.encoder_embed_dim),
            dtype=torch.float32,
            device=device,
        )
        if total_nodes > 0:
            flat_attr[node_token_positions] = (
                self.node_encoder(node_data) + sin_embed_node
            )
        if total_edges > 0:
            flat_attr[edge_token_positions] = (
                self.edge_encoder(edge_data)
                + self.encode_branch_lengths(branch_lengths)
                + sin_embed_edge
            )

        if total_tokens > 0:
            pos_pe_concat = torch.cat(
                [
                    lap_pe[flat_lap_indices[:, 0]],
                    lap_pe[flat_lap_indices[:, 1]],
                ],
                dim=1,
            )
            flat_features = (
                flat_attr
                + self.type_encoder(flat_type_ids)
                + self.lap_encoder(pos_pe_concat)
            )
        else:
            flat_features = flat_attr

        padded_features = torch.zeros(
            (batch_size, max_tokens, self.encoder_embed_dim),
            dtype=torch.float32,
            device=device,
        )
        if total_tokens > 0:
            padded_features[flat_batch_indices, flat_token_positions] = flat_features

        def _leaf_index(value):
            if torch.is_tensor(value):
                return value.to(device=device, dtype=torch.long)
            return torch.as_tensor(value, device=device, dtype=torch.long)

        return (
            padded_features,
            _tensor("padding_mask", torch.bool),
            _tensor("padded_indices", torch.long),
            _tensor("padded_leaf_masks", torch.bool),
            [_leaf_index(value) for value in cache["leaf_idx"]],
            _tensor("padded_edge_masks", torch.bool),
            [[int(mask) for mask in masks] for masks in cache["edge_split_masks"]],
        )

    def forward_raw_graph_cache(self, caches):
        if (
            isinstance(caches, dict)
            and bool(caches.get("_tree_tokenizer_raw_graph_batch_cache", False))
        ):
            return self._forward_raw_graph_batch_cache(caches)
        if isinstance(caches, dict):
            caches = [caches]
        batch_size = len(caches)
        if batch_size == 0:
            raise ValueError("forward_raw_graph_cache requires at least one cache")
        if self.concat_features or not self.raw_graph_cache_vectorized:
            return self._forward_raw_graph_cache_loop(caches)

        device = next(self.parameters()).device
        records = []
        max_tokens = 0
        for cache in caches:
            def _tensor(key, dtype=None):
                value = cache[key]
                if torch.is_tensor(value):
                    return (
                        value.to(device=device, dtype=dtype)
                        if dtype is not None
                        else value.to(device)
                    )
                return torch.as_tensor(value, device=device, dtype=dtype)

            node_num = int(cache["node_num"])
            edge_num = int(cache["edge_num"])
            token_num = node_num + edge_num
            max_tokens = max(max_tokens, token_num)
            records.append(
                {
                    "node_data": _tensor("node_data", torch.long),
                    "edge_data": _tensor("edge_data", torch.long),
                    "branch_lengths": _tensor("branch_lengths", torch.float32),
                    "lap_pe": _tensor("lap_pe", torch.float32),
                    "leaf_mask": _tensor("leaf_mask", torch.bool),
                    "leaf_idx": _tensor("leaf_idx", torch.long),
                    "sin_embed_node": _tensor("sin_embed_node", torch.float32),
                    "sin_embed_edge": _tensor("sin_embed_edge", torch.float32),
                    "full_padded_index": _tensor("full_padded_index", torch.long),
                    "edge_split_masks": [
                        int(mask) for mask in cache["edge_split_masks"]
                    ],
                    "node_num": node_num,
                    "edge_num": edge_num,
                    "token_num": token_num,
                }
            )

        total_nodes = sum(record["node_num"] for record in records)
        total_edges = sum(record["edge_num"] for record in records)

        if total_nodes > 0:
            all_node_data = torch.cat(
                [record["node_data"] for record in records if record["node_num"] > 0],
                dim=0,
            )
            all_sin_node = torch.cat(
                [
                    record["sin_embed_node"]
                    for record in records
                    if record["node_num"] > 0
                ],
                dim=0,
            )
            all_node_attr = self.node_encoder(all_node_data) + all_sin_node
        else:
            all_node_attr = torch.empty(
                (0, self.encoder_embed_dim),
                dtype=torch.float32,
                device=device,
            )

        if total_edges > 0:
            all_edge_data = torch.cat(
                [record["edge_data"] for record in records if record["edge_num"] > 0],
                dim=0,
            )
            all_branch_lengths = torch.cat(
                [
                    record["branch_lengths"]
                    for record in records
                    if record["edge_num"] > 0
                ],
                dim=0,
            )
            all_sin_edge = torch.cat(
                [
                    record["sin_embed_edge"]
                    for record in records
                    if record["edge_num"] > 0
                ],
                dim=0,
            )
            all_edge_attr = (
                self.edge_encoder(all_edge_data)
                + self.encode_branch_lengths(all_branch_lengths)
                + all_sin_edge
            )
        else:
            all_edge_attr = torch.empty(
                (0, self.encoder_embed_dim),
                dtype=torch.float32,
                device=device,
            )

        node_offset = 0
        edge_offset = 0
        per_tree_attr = []
        per_tree_type_ids = []
        per_tree_pos_inputs = []
        batch_leaf_indices = []
        batch_edge_split_masks = []
        for record in records:
            node_num = record["node_num"]
            edge_num = record["edge_num"]
            node_attr = all_node_attr[node_offset : node_offset + node_num]
            edge_attr = all_edge_attr[edge_offset : edge_offset + edge_num]
            node_offset += node_num
            edge_offset += edge_num
            per_tree_attr.append(torch.cat([node_attr, edge_attr], dim=0))
            per_tree_type_ids.append(
                torch.cat(
                    [
                        torch.zeros(node_num, dtype=torch.long, device=device),
                        torch.ones(edge_num, dtype=torch.long, device=device),
                    ],
                    dim=0,
                )
            )
            full_padded_index = record["full_padded_index"]
            u = full_padded_index[:, 0]
            v = full_padded_index[:, 1]
            lap_pe = record["lap_pe"]
            per_tree_pos_inputs.append(torch.cat([lap_pe[u], lap_pe[v]], dim=1))
            batch_leaf_indices.append(record["leaf_idx"])
            batch_edge_split_masks.append(record["edge_split_masks"])

        flat_attr = torch.cat(per_tree_attr, dim=0)
        flat_type_embedding = self.type_encoder(torch.cat(per_tree_type_ids, dim=0))
        flat_pos_embedding = self.lap_encoder(torch.cat(per_tree_pos_inputs, dim=0))
        flat_features = flat_attr + flat_type_embedding + flat_pos_embedding

        padded_features = torch.zeros(
            (batch_size, max_tokens, self.encoder_embed_dim),
            device=device,
        )
        padded_masks = torch.ones(
            (batch_size, max_tokens),
            device=device,
            dtype=torch.bool,
        )
        padded_indices = torch.zeros(
            (batch_size, max_tokens, 2),
            device=device,
            dtype=torch.long,
        )
        padded_leaf_masks = torch.zeros(
            (batch_size, max_tokens),
            device=device,
            dtype=torch.bool,
        )
        padded_edge_masks = torch.zeros(
            (batch_size, max_tokens),
            device=device,
            dtype=torch.bool,
        )

        token_offset = 0
        for i, record in enumerate(records):
            seq_len = int(record["token_num"])
            if seq_len > 0:
                features = flat_features[token_offset : token_offset + seq_len]
                token_offset += seq_len
                padded_features[i, :seq_len] = features
                padded_masks[i, :seq_len] = False
                padded_indices[i, :seq_len] = record["full_padded_index"]
                padded_leaf_masks[i, :seq_len] = record["leaf_mask"]
                padded_edge_masks[
                    i,
                    record["node_num"] : record["node_num"] + record["edge_num"],
                ] = True

        return (
            padded_features,
            padded_masks,
            padded_indices,
            padded_leaf_masks,
            batch_leaf_indices,
            padded_edge_masks,
            batch_edge_split_masks,
        )

    def _forward_raw_graph_cache_loop(self, caches):
        batch_size = len(caches)
        batch_features = []
        batch_padding_masks = []
        batch_padded_indices = []
        batch_leaf_masks = []
        batch_leaf_indices = []
        batch_edge_masks = []
        batch_edge_split_masks = []
        max_tokens = 0

        for cache in caches:
            (
                features,
                padding_mask,
                padded_index,
                leaf_mask_single,
                leaf_idx_single,
                edge_mask,
                edge_split_masks,
            ) = self._forward_single_raw_graph_cache(cache)
            batch_features.append(features)
            batch_padding_masks.append(padding_mask)
            batch_padded_indices.append(padded_index)
            batch_leaf_masks.append(leaf_mask_single)
            batch_leaf_indices.append(leaf_idx_single)
            batch_edge_masks.append(edge_mask)
            batch_edge_split_masks.append(edge_split_masks)
            max_tokens = max(max_tokens, features.size(0))

        device = batch_features[0].device
        padded_features = torch.zeros(
            (batch_size, max_tokens, self.encoder_embed_dim),
            device=device,
        )
        padded_masks = torch.ones(
            (batch_size, max_tokens),
            device=device,
            dtype=torch.bool,
        )
        padded_indices = torch.zeros(
            (batch_size, max_tokens, 2),
            device=device,
            dtype=torch.long,
        )
        padded_leaf_masks = torch.zeros(
            (batch_size, max_tokens),
            device=device,
            dtype=torch.bool,
        )
        padded_edge_masks = torch.zeros(
            (batch_size, max_tokens),
            device=device,
            dtype=torch.bool,
        )

        for i, (
            features,
            mask,
            indices,
            leaf_mask_single,
            edge_mask,
        ) in enumerate(
            zip(
                batch_features,
                batch_padding_masks,
                batch_padded_indices,
                batch_leaf_masks,
                batch_edge_masks,
            )
        ):
            seq_len = features.size(0)
            if seq_len > 0:
                padded_features[i, :seq_len] = features
                padded_masks[i, :seq_len] = mask
                padded_indices[i, :seq_len] = indices
                padded_leaf_masks[i, :seq_len] = leaf_mask_single
                padded_edge_masks[i, :seq_len] = edge_mask

        return (
            padded_features,
            padded_masks,
            padded_indices,
            padded_leaf_masks,
            batch_leaf_indices,
            padded_edge_masks,
            batch_edge_split_masks,
        )

    def compute_structural_cache(self, tree_list):
        """
        Precompute expensive structural data for a list of trees (Newick strings or ETE trees).
        Returns a list of 'cache' dictionaries, one per tree.
        """
        device = next(self.parameters()).device
        cache_list = []

        use_mp = len(tree_list) > 10
        parsed_results = []

        if use_mp:
            try:
                if all(isinstance(t, str) for t in tree_list):
                    workers = min(len(tree_list), os.cpu_count() or 1)
                    chunksize = max(1, len(tree_list) // (workers * 4))
                    with multiprocessing.Pool(processes=workers) as pool:
                        parsed_results = pool.map(
                            _worker_newick_parser, tree_list, chunksize=chunksize
                        )
                else:
                    parsed_results = [_worker_newick_parser(t) for t in tree_list]
            except Exception as e:
                print(f"Pool failed: {e}")
                parsed_results = [_worker_newick_parser(t) for t in tree_list]
        else:
            parsed_results = [_worker_newick_parser(t) for t in tree_list]

        for tree_info in parsed_results:
            # 1. Parse Structure
            child_ptr = torch.from_numpy(tree_info[0]).to(device)
            child_ids = torch.from_numpy(tree_info[1]).to(device)
            parent_arr = torch.from_numpy(tree_info[2]).to(device)
            child_arr = torch.from_numpy(tree_info[3]).to(device)
            root_idx = tree_info[4]
            branch_lengths = torch.from_numpy(tree_info[5]).to(device)
            edge_types = torch.from_numpy(tree_info[6]).to(device)
            edge_split_masks = tree_info[7]

            # 2. Build Graph (Laplacian, masks, etc.)
            (
                node_data,
                edge_index,
                edge_data,
                _,  # branch_lengths (returned from method, but we ignore it here)
                node_num,
                edge_num,
                lap_pe,
                leaf_mask,
                leaf_idx,
                sin_embed_node,
                sin_embed_edge,
            ) = self.tree_to_graph_from_children(
                child_ptr,
                child_ids,
                parent_arr,
                child_arr,
                root_idx,
                branch_lengths,
                edge_types,
            )

            # Store in cache
            # Note: edge_index corresponds to [sorted_real_edges..., root_self_edge]
            # edge_split_masks also corresponds to this order.

            # OPTIMIZATION: pre-compute static parts of the embedding
            # We bypass _process_and_pack_single to store the static 'base' tensors.
            # 1. Node tokens
            node_indices = torch.arange(node_num, device=device)
            node_attr_embedding = self.node_encoder(node_data) + sin_embed_node

            # 2. Edge tokens (static part: EdgeType + Sin + TypeId + Pos)
            # Dynamic part is branch_length_encoder(len).
            edge_base_embedding = self.edge_encoder(edge_data) + sin_embed_edge

            # 3. Type Embeddings
            type_ids = torch.cat(
                [
                    torch.zeros(node_num, dtype=torch.long, device=device),
                    torch.ones(edge_num, dtype=torch.long, device=device),
                ]
            )
            type_embedding = self.type_encoder(type_ids)

            # 4. Pos Embeddings (Laplacian)
            node_pairs = torch.stack([node_indices, node_indices], dim=1)
            edge_pairs = edge_index.t()
            full_padded_index = torch.cat([node_pairs, edge_pairs], dim=0)

            u = full_padded_index[:, 0]
            v = full_padded_index[:, 1]
            pos_pe_concat = torch.cat([lap_pe[u], lap_pe[v]], dim=1)
            pos_embedding = self.lap_encoder(pos_pe_concat)

            # 5. Combine static parts
            # Static Feature = (NodeAttr | EdgeBase) + Type + Pos
            # For Nodes: NodeAttr + Type[Node] + Pos[Node]
            # For Edges: EdgeBase + Type[Edge] + Pos[Edge]
            # We can concat them into [N+E, D]

            static_attr = torch.cat([node_attr_embedding, edge_base_embedding], dim=0)

            # Assuming additive combination (concat_features=False)
            if not self.concat_features:
                static_tokens = static_attr + type_embedding + pos_embedding
            else:
                static_tokens = static_attr + type_embedding + pos_embedding

            # Pre-compute masks
            if self.concat_features:
                # Just use the code logic matching _process_and_pack
                final_len = static_tokens.size(0)  # or output of linear
                pass

            # Masks
            edge_mask = torch.cat(
                [
                    torch.zeros(node_num, dtype=torch.bool, device=device),
                    torch.ones(edge_num, dtype=torch.bool, device=device),
                ]
            )
            padding_mask = torch.zeros(
                node_num + edge_num, dtype=torch.bool, device=device
            )

            cache = {
                # Static Precomputed
                "static_tokens": static_tokens,  # [N+E, D] (Base)
                "node_num": node_num,
                "edge_num": edge_num,
                "padding_mask": padding_mask,
                "full_padded_index": full_padded_index,
                "leaf_mask": leaf_mask,
                "leaf_idx": leaf_idx,
                "edge_mask": edge_mask,
                "edge_split_masks": edge_split_masks,
                # Keep raw data just in case needed for debugging or fallback
                "edge_split_masks_list": edge_split_masks,
            }
            cache_list.append(cache)

        return cache_list

    def create_batched_cache(self, tree_list):
        caches = self.compute_structural_cache(tree_list)
        return BatchedStructuralCache(
            caches, next(self.parameters()).device, self.encoder_embed_dim
        )

    def forward_batched(self, batched_cache, branch_lengths_list):
        B = batched_cache.batch_size
        MaxE = batched_cache.max_edges
        device = batched_cache.device

        lengths_tensor = torch.zeros((B, MaxE), device=device)

        # Convert inputs to tensor [B, MaxE]
        if isinstance(branch_lengths_list[0], dict):
            # Fast-ish path for dicts
            for i, length_dict in enumerate(branch_lengths_list):
                masks = batched_cache.edge_split_masks_list[i]
                # We simply iterate.
                # For 100 trees * 100 edges = 10000 lookups. Should be < 10ms.
                vals = [length_dict.get(m, 0.0) for m in masks]
                lengths_tensor[i, : len(vals)] = torch.tensor(vals, device=device)
        else:
            for i, l in enumerate(branch_lengths_list):
                if isinstance(l, (list, tuple, np.ndarray)):
                    t = torch.as_tensor(l, device=device)
                else:
                    t = l
                lengths_tensor[i, : len(t)] = t

        # Embed
        branch_emb = self.encode_branch_lengths(lengths_tensor)

        # Clone static tokens
        out_tokens = batched_cache.static_tokens.clone()

        # Prepare scatter
        # Mask out invalid embeddings (where lengths didn't exist)
        mask = batched_cache.edge_scatter_mask.unsqueeze(-1)
        branch_emb = branch_emb * mask

        # Indices: [B, MaxE, D]
        indices = batched_cache.edge_scatter_indices.unsqueeze(-1).expand(
            -1, -1, self.encoder_embed_dim
        )

        # Scatter add: Add branch embeddings to the correct positions in out_tokens
        # Note: indices where mask=False are 0. branch_emb is 0.
        # Adding 0 to token 0 (node) is harmless.
        out_tokens.scatter_add_(1, indices, branch_emb)

        return (
            out_tokens,
            batched_cache.padding_mask,
            batched_cache.full_padded_indices,
            batched_cache.leaf_mask,
            batched_cache.leaf_indices_list,
            batched_cache.edge_mask,
            batched_cache.edge_split_masks_list,
        )

    def _newick_to_structural(
        self,
        tree_or_str,
        default_edge_type: int = 1,
    ):
        return _worker_newick_parser(tree_or_str)
