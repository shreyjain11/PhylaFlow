import networkx as nx
import math
import io
import matplotlib.pyplot as plt
from matplotlib import animation
from Bio import Phylo
from ete3 import Tree as eteTree
import random

def popcount(x: int) -> int:
    return x.bit_count()  # Python 3.8+: or bin(x).count("1")

def tree_to_newick(G: nx.Graph, root=None, leaf_label_attr="label", dummy_node=None, mapping=None) -> str:
    """
    Convert a NetworkX tree into a Newick string.

    Assumptions:
      - G is an undirected tree.
      - Edge lengths (branch lengths) are stored in edge attribute 'length'.
      - Leaves have some label (default attr 'label'); if missing, node id is used.
      - 'root' is either a node in G or None.
        If None, we'll try to find a node with label 'root', else pick an arbitrary node.

    Returns:
      Newick string ending with a semicolon.
    """

    if dummy_node is not None and G.has_node(dummy_node):
        # 1. If the current root IS the dummy we are about to delete...
        if root == dummy_node:
            # Find the neighbor (the "Original Root" of the MrBayes tree)
            neighbors = list(G.neighbors(dummy_node))
            if neighbors:
                # Shift the root pointer to the valid internal node
                root = neighbors[0]
            else:
                # Edge case: Graph was just (Dummy)-(Node), now it's just (Node)
                # or empty. We'll handle empty later.
                root = None 

        # 2. NOW it is safe to remove the dummy
        G.remove_node(dummy_node)

    if mapping is not None:
        # Filter mapping to only nodes that exist (dummy is gone now)
        mapping = {node: name for node, name in mapping.items() if node in G}
        
        # Update attributes for Newick label usage
        for nid, name in mapping.items():
            if G.has_node(nid):
                G.nodes[nid]['label'] = name
        
        # Relabel the keys in the graph itself
        G = nx.relabel_nodes(G, mapping, copy=False)
        
        # CRITICAL: If we relabeled the nodes, we must update our 'root' variable 
        # because the old ID (e.g., 0) might now be a string (e.g., "73")
        # However, typically 'root' is an internal node (like "C_root") which 
        # isn't in the leaf mapping. But if root WAS a mapped leaf (edge case), 
        # we need to update it.
        if root in mapping:
            root = mapping[root]

    # Recursive DFS to build Newick
    def build_subtree(node, parent):
        # children are neighbors except the parent
        children = [nbr for nbr in G.neighbors(node) if nbr != parent]

        if not children:
            # Leaf: return its label only (branch length added by parent)
            label = G.nodes[node].get(leaf_label_attr, str(node))
            return label
        else:
            # Internal node: recursively build children
            parts = []
            for child in children:
                child_subtree = build_subtree(child, node)
                blen = G.edges[node, child].get("length", 1.0)
                parts.append(f"{child_subtree}:{blen}")
            return "(" + ",".join(parts) + ")"

    newick_body = build_subtree(root, parent=None)
    return newick_body + ";"

def build_tree_from_splits(split_set, length_map, n_leaves, root_leaf=1, mapping=None):
    """
    Build a NetworkX tree from a compatible set of splits (bitmasks).

    split_set: iterable of bitmasks (each is one side of an unrooted split)
    length_map: dict[bitmask -> branch_length] for those splits
    n_leaves: number of leaves (labels 1..n_leaves)
    root_leaf: leaf index used as "outgroup" to orient clusters

    Returns:
        G: networkx.Graph with nodes:
            - 1..n_leaves are leaves (attr is_leaf=True)
            - internal nodes (cluster nodes) (attr is_leaf=False)
          and edges have 'length' attribute.
        newick: string
    """
    dummy_node = n_leaves - 1
    full_mask = (1 << n_leaves) - 1
    root_bit = 1 << root_leaf

    G = nx.Graph()

    # Add leaf nodes
    for i in range(n_leaves):
        G.add_node(i, is_leaf=True, label=str(i))

    # Special case: no splits => star tree; use leaf-lengths if present
    if not split_set:
        root_node = "R"
        G.add_node(root_node, is_leaf=False, label="R")
        for i in range(n_leaves):
            m_leaf = 1 << i
            length = float(length_map.get(m_leaf, 0.1))
            G.add_edge(root_node, i, length=length)
        return G, tree_to_newick(G, root=root_leaf)

    # ------------------------------------------------------------------
    # STEP 0: separate internal splits from pendant (leaf) splits
    # ------------------------------------------------------------------
    leaf_lengths = {i: 0.0 for i in range(0, n_leaves)}
    internal_splits = []

    for m in split_set:
        A = m
        B = full_mask ^ m
        sizeA = popcount(A)
        sizeB = popcount(B)
        length = float(length_map.get(m, 0.5))

        if sizeA == 1 or sizeB == 1:
            # Pendant edge: one side is a single leaf
            if sizeA == 1:
                leaf_mask = A
            else:
                leaf_mask = B
            leaf_index = (leaf_mask.bit_length() - 1)  # 1-based index
            leaf_lengths[leaf_index] = length
            # Do NOT include this in internal split system
        else:
            internal_splits.append(m)

    # If there are no internal splits, just connect leaves to a root
    if not internal_splits:
        root_node = "R"
        G.add_node(root_node, is_leaf=False, label="R")
        for i in range(n_leaves):
            G.add_edge(root_node, i, length=leaf_lengths.get(i, 0.0))
        return G, tree_to_newick(G, root=root_leaf)

    # ------------------------------------------------------------------
    # STEP 1: convert internal splits -> oriented clusters (away from root)
    # ------------------------------------------------------------------
    cluster_masks = set()
    cluster_to_split = {}

    for m in internal_splits:
        if m & root_bit:
            # take side that does NOT contain root_leaf
            cluster = full_mask ^ m
        else:
            cluster = m
        cluster_masks.add(cluster)
        cluster_to_split[cluster] = m

    # Include full cluster (all leaves) as root cluster
    root_cluster = full_mask
    all_clusters = list(cluster_masks)
    all_clusters.append(root_cluster)

    # STEP 2: sort clusters (excluding root) by size (descending)
    cluster_list = sorted(cluster_masks, key=lambda c: popcount(c), reverse=True)

    # STEP 3: create nodes for clusters & connect to parent cluster
    cluster_nodes = {}

    # Root node
    root_node = "C_root"
    G.add_node(root_node, is_leaf=False, label="root")
    cluster_nodes[root_cluster] = root_node

    # For each cluster, find minimal parent cluster that strictly contains it
    for C in cluster_list:
        # identify parent cluster P: smallest cluster with C ⊂ P
        parent = root_cluster
        parent_size = popcount(root_cluster)
        sizeC = popcount(C)
        for P in all_clusters:
            if P == C:
                continue
            # C subset P?
            if (C & ~P) == 0:
                sizeP = popcount(P)
                if sizeC < sizeP < parent_size:
                    parent = P
                    parent_size = sizeP

        # make node for this cluster
        nodeC = f"C_{C}"
        G.add_node(nodeC, is_leaf=False, label=f"C({sizeC})")
        cluster_nodes[C] = nodeC

        # ensure parent node exists
        nodeP = cluster_nodes[parent]

        # internal edge length determined by split that induced this cluster
        split_mask = cluster_to_split[C]
        length = float(length_map.get(split_mask, 0.5))
        G.add_edge(nodeP, nodeC, length=length)

    # ------------------------------------------------------------------
    # STEP 4: connect leaves to smallest cluster that contains them
    #         using the *pendant* lengths
    # ------------------------------------------------------------------
    all_clusters_with_root = [root_cluster] + cluster_list
    for leaf in range(n_leaves):
        leaf_bit = 1 << leaf
        candidates = [C for C in all_clusters_with_root if (C & leaf_bit) != 0]
        if not candidates:
            parent_cluster = root_cluster
        else:
            parent_cluster = min(candidates, key=lambda c: popcount(c))
        parent_node = cluster_nodes[parent_cluster]
        length = float(leaf_lengths.get(leaf, 0.0))
        G.add_edge(parent_node, leaf, length=length)
    return G, tree_to_newick(G, root=root_leaf, dummy_node=dummy_node, mapping=mapping)

def make_bhv_topology_movie(
    geodesic_result,
    n_leaves,
    filename="bhv_topology.mp4",
    root = None,
    mapping =None,
    F=10,
    fps=1,
    dpi=150,
):
    """
    Make a simple movie where each frame is a NetworkX drawing of the tree
    topology at a different step along the BHV geodesic.

    One frame per segment boundary:
      - frame 0: start topology (tree1-like)
      - frame i>0: topology after segment i-1
    """
    snapshots = []
    for k in range(F):
        u = k / (F - 1)
        G, newick, info = sample_tree_along_geodesic(geodesic_result, n_leaves, u=u, root=root, mapping=mapping)
        snapshots.append(newick)

    # snapshots = build_geodesic_snapshots(tree1, tree2, geodesic_result, n_leaves)

    fig, ax = plt.subplots(figsize=(6, 6))

    def init():
        ax.clear()
        ax.axis("off")
        return []

    def draw_snapshot(idx):
        ax.clear()
        ax.axis("off")
        snap = snapshots[idx]
        newick = snap
        desc = f"Frame {idx} (u={idx/(len(snapshots)-1):.2f})"
        print(newick)

        # Render the Newick tree with Biopython
        tree = Phylo.read(io.StringIO(newick), "newick")
        # Label each edge with its branch length if present
        Phylo.draw(
            tree,
            axes=ax,
            do_show=False,
            branch_labels=lambda clade: (f"{clade.branch_length:.1f}" if clade.branch_length not in (None, 0.0) else None),
        )
        ax.set_title(desc)
        return []

    anim = animation.FuncAnimation(
        fig,
        draw_snapshot,
        init_func=init,
        frames=len(snapshots),
        interval=1000.0 / fps,
        blit=False,
    )

    # if filename.endswith(".gif"):
    anim.save(filename, writer="pillow", fps=fps, dpi=dpi)
    # else:
    #     Writer = animation.writers["ffmpeg"]
    #     writer = Writer(fps=fps, bitrate=1800)
    #     anim.save(filename, writer=writer, dpi=dpi)

    plt.close(fig)
    print(f"Saved BHV topology movie to {filename}")

def sample_tree_along_geodesic(geodesic_result, n_leaves, u=None, root=None, mapping=None):
    """
    Sample a tree at a *continuous* position along a BHV geodesic.

    geodesic_result["segments"] must be a list of dicts with keys:
        - "length": float BHV length of the segment
        - "splits": set of split bitmasks present in this orthant
        - "start_lengths": dict[split -> length at start of segment]
        - "end_lengths":   dict[split -> length at end of segment]

    n_leaves: number of leaves in the tree
    u: scalar in [0,1]; if None, sampled uniformly

    Returns:
        G: NetworkX tree at that point
        newick: Newick string for that tree
        info: dict with where we are along the path
    """
    segments = geodesic_result["segments"]
    motion_segments = [seg for seg in segments if float(seg.get("length", 0.0)) > 1e-12]
    if motion_segments:
        segments = motion_segments
    if u is None:
        u = random.random()

    # 1) total BHV length
    total_L = sum(seg["length"] for seg in segments)

    # 2) convert u -> arc length
    s = u * total_L

    # 3) find segment
    cum = 0.0
    seg_idx = None
    offset = 0.0
    for i, seg in enumerate(segments):
        if s <= cum + seg["length"] or i == len(segments) - 1:
            seg_idx = i
            offset = s - cum
            break
        cum += seg["length"]

    seg = segments[seg_idx]
    L_seg = seg["length"]
    alpha = 0.0 if L_seg == 0 else offset / L_seg

    # 4) interpolate lengths for splits in this orthant
    curr_lengths = {}
    keys = set(seg["start_lengths"].keys()) | set(seg["end_lengths"].keys())
    for m in keys:
        l0 = seg["start_lengths"][m]
        l1 = seg["end_lengths"][m]
        curr_lengths[m] = (1.0 - alpha) * l0 + alpha * l1

    # 5) drop ~zero edges
    eps = 1e-8
    split_set = {m for m, L in curr_lengths.items() if L > eps}
    length_map = {m: L for m, L in curr_lengths.items() if L > eps}
    active = {e for e, L in curr_lengths.items() if L > eps}

    # velocity restricted to active splits at sample point
    vel_active = {e: seg["velocity"][e] for e in active if e in seg["velocity"]}

    # 6) build tree
    G, newick = build_tree_from_splits(split_set, length_map, n_leaves, root_leaf=n_leaves-1, mapping=mapping)
    
    # print(f"Current length is {len(curr_lengths)} and velocity is {len(seg['velocity'])}, graph has {G.number_of_edges()} edges")

    info = {
        "u": u,
        "s": s,
        "segment_index": seg_idx,
        "alpha": alpha,
        "total_length": total_L,
        #Note an important bug fix here we need to multiply by total_L to get in terms of time, not BHV unit length!
        "velocity": {e: v * total_L for e, v in seg['velocity'].items()},
        "active_velocity": {e: v * total_L for e, v in vel_active.items()},
    }

    return G, newick, info


# ---------------------------------------------------------------------------
# Robinson-Foulds (RF) distance utilities
# ---------------------------------------------------------------------------
def normalized_rf(newick_a: str, newick_b: str, unrooted: bool = True):
    """Compute Robinson-Foulds distance between two Newick trees.

    Returns (rf, max_rf, normalized_rf) where normalized_rf = rf / max_rf.
    If max_rf is zero (degenerate), normalized_rf is 0.0.
    """
    t1 = eteTree(newick_a)
    t2 = eteTree(newick_b)
    rf, max_rf, *_ = t1.robinson_foulds(t2, unrooted_trees=unrooted)
    norm = 0.0 if max_rf == 0 else rf / max_rf
    return rf, max_rf, norm


if __name__ == "__main__":
    # Simple test with two random trees
    n_leaves = 5
    from utils.random_tree import Tree
    from utils.bhv_utils import BHVEncoder

    # rt1 = Tree(n_leaves)
    # newick1 = rt1.to_newick()
    newick1 = "((73:2.000000e-02,(17:2.000000e-02,(144:2.000000e-02,(47:2.000000e-02,16:2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02,((152:2.000000e-02,(60:2.000000e-02,(97:2.000000e-02,((141:2.000000e-02,88:2.000000e-02):2.000000e-02,((133:2.000000e-02,(92:2.000000e-02,(117:2.000000e-02,19:2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02,14:2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02,(((39:2.000000e-02,23:2.000000e-02):2.000000e-02,(72:2.000000e-02,13:2.000000e-02):2.000000e-02):2.000000e-02,((22:2.000000e-02,(((119:2.000000e-02,116:2.000000e-02):2.000000e-02,((140:2.000000e-02,56:2.000000e-02):2.000000e-02,51:2.000000e-02):2.000000e-02):2.000000e-02,(127:2.000000e-02,((91:2.000000e-02,((145:2.000000e-02,(135:2.000000e-02,(106:2.000000e-02,(101:2.000000e-02,(122:2.000000e-02,52:2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02,(((89:2.000000e-02,42:2.000000e-02):2.000000e-02,41:2.000000e-02):2.000000e-02,12:2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02,(95:2.000000e-02,((104:2.000000e-02,((137:2.000000e-02,(136:2.000000e-02,130:2.000000e-02):2.000000e-02):2.000000e-02,98:2.000000e-02):2.000000e-02):2.000000e-02,(115:2.000000e-02,((28:2.000000e-02,20:2.000000e-02):2.000000e-02,(81:2.000000e-02,(96:2.000000e-02,(111:2.000000e-02,11:2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02,(65:2.000000e-02,(149:2.000000e-02,(68:2.000000e-02,(63:2.000000e-02,(82:2.000000e-02,((146:2.000000e-02,124:2.000000e-02):2.000000e-02,(77:2.000000e-02,(85:2.000000e-02,((((50:2.000000e-02,32:2.000000e-02):2.000000e-02,((71:2.000000e-02,(109:2.000000e-02,(46:2.000000e-02,((143:2.000000e-02,(48:2.000000e-02,((79:2.000000e-02,(121:2.000000e-02,(114:2.000000e-02,58:2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02,(84:2.000000e-02,(67:2.000000e-02,((123:2.000000e-02,(142:2.000000e-02,44:2.000000e-02):2.000000e-02):2.000000e-02,26:2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02,(((54:2.000000e-02,34:2.000000e-02):2.000000e-02,(53:2.000000e-02,24:2.000000e-02):2.000000e-02):2.000000e-02,(18:2.000000e-02,(107:2.000000e-02,9:2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02,4:2.000000e-02):2.000000e-02):2.000000e-02,((150:2.000000e-02,120:2.000000e-02):2.000000e-02,(40:2.000000e-02,(5:2.000000e-02,((59:2.000000e-02,(55:2.000000e-02,(80:2.000000e-02,(132:2.000000e-02,(94:2.000000e-02,(110:2.000000e-02,8:2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02,(83:2.000000e-02,(86:2.000000e-02,((118:2.000000e-02,49:2.000000e-02):2.000000e-02,((61:2.000000e-02,((128:2.000000e-02,74:2.000000e-02):2.000000e-02,(113:2.000000e-02,(105:2.000000e-02,(108:2.000000e-02,(125:2.000000e-02,((151:2.000000e-02,33:2.000000e-02):2.000000e-02,((147:2.000000e-02,57:2.000000e-02):2.000000e-02,((36:2.000000e-02,(25:2.000000e-02,(102:2.000000e-02,7:2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02,(((153:2.000000e-02,78:2.000000e-02):2.000000e-02,43:2.000000e-02):2.000000e-02,(103:2.000000e-02,(134:2.000000e-02,(99:2.000000e-02,(38:2.000000e-02,(10:2.000000e-02,((64:2.000000e-02,(66:2.000000e-02,27:2.000000e-02):2.000000e-02):2.000000e-02,(((126:2.000000e-02,75:2.000000e-02):2.000000e-02,(139:2.000000e-02,45:2.000000e-02):2.000000e-02):2.000000e-02,((87:2.000000e-02,((148:2.000000e-02,(129:2.000000e-02,37:2.000000e-02):2.000000e-02):2.000000e-02,21:2.000000e-02):2.000000e-02):2.000000e-02,(35:2.000000e-02,6:2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02,(30:2.000000e-02,((90:2.000000e-02,(62:2.000000e-02,15:2.000000e-02):2.000000e-02):2.000000e-02,(131:2.000000e-02,(155:2.000000e-02,(76:2.000000e-02,((100:2.000000e-02,29:2.000000e-02):2.000000e-02,(93:2.000000e-02,(69:2.000000e-02,(112:2.000000e-02,2:2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02,(138:2.000000e-02,(154:2.000000e-02,(70:2.000000e-02,(31:2.000000e-02,3:2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02):2.000000e-02,1:2.000000e-02);"
    print("Generated random tree 1:", newick1)
    # rt2 = RandomTree(n_leaves)
    # newick2 = rt2.to_newick()

    T1 = Tree(newick1)
    print("Parsed tree 1 now see it as:", T1)
    print("Number of leaves:", T1.n_leaves)
    # T2 = Tree(newick2)

    # print("Tree 1 Newick:", newick1)
    # print("Tree 2 Newick:", newick2)

    enc = BHVEncoder()
    t1_edge_mask, t1_edge_length = enc.return_BHV_encoding(T1)
    # t2_edge_mask, t2_edge_length = enc.return_BHV_encoding(T2)

    tree1 = {m: l for m, l in zip(t1_edge_mask, t1_edge_length)}
    # tree2 = {m: l for m, l in zip(t2_edge_mask, t2_edge_length)}
    G_rec, recovered_newick_1 = build_tree_from_splits(t1_edge_mask, tree1, T1.n_leaves, root_leaf = T1.n_leaves-1, mapping=T1.id_to_name)
    # # recovered_newick_2 = build_tree_from_splits(t2_edge_mask, tree2, n_leaves)[1]
    # print("Recovered Tree 1 Newick from BHV:", recovered_newick_1)
    # dummy_id = T1.n_leaves - 1
    # if G_rec.has_node(dummy_id):
    #     G_rec.remove_node(dummy_id)
    #     mapping = {node: name for node, name in T1.id_to_name.items() if node in G_rec}

    #     # Step A: Update the internal attributes
    #     for nid, name in mapping.items():
    #         G_rec.nodes[nid]['label'] = name
    #     #Get name of one node in graph
    #     print(mapping)
    #     G_rec = nx.relabel_nodes(G_rec, mapping, copy=False)
    #     final_newick = tree_to_newick(G_rec, root=None)
    #     print("Final Match:", final_newick)

    print("RF distance between original and recovered tree 1:", normalized_rf(newick1, recovered_newick_1))
    
    # import pdb; pdb.set_trace()
