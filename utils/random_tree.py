import random
from collections import defaultdict
from ete3.coretype.tree import Tree as eteTree


class Tree:
    """Build a tree object from a Newick string.

    Leaves are assumed to be labeled with integers (1..n). Internal nodes are
    assigned integer IDs greater than the maximum leaf label. Edge lengths are
    read from the Newick branch lengths; if missing, a small default (0.1) is
    used.
    """

    def __init__(
        self, newick: str = None, num_leaves: int = None, random: bool = False,
        dummy: bool = True
    ):
        self.adj = defaultdict(list)
        self.lengths = {}  # symmetric edge lengths: key=(u,v) or (v,u)
        self.n_leaves = 0
        self.id_to_name = {}
        self.dummy = dummy

        if random and num_leaves is not None:
            self._build_random_tree(num_leaves)
        elif newick is not None:
            self._build_from_newick(newick)

    def _add_edge(self, u, v, length=None):
        self.adj[u].append(v)
        self.adj[v].append(u)
        if length is not None:
            self.lengths[(u, v)] = self.lengths[(v, u)] = length

    def _remove_edge(self, u, v):
        if v in self.adj[u]:
            self.adj[u].remove(v)
        if u in self.adj[v]:
            self.adj[v].remove(u)
        self.lengths.pop((u, v), None)
        self.lengths.pop((v, u), None)

    def _build_random_tree(self, num_leaves: int):
        """
        Generates a random tree with 'num_leaves' biological leaves (0..N-1),
        plus 1 Dummy leaf (N).
        """
        # 1. Setup IDs
        # Biological leaves: 0 to num_leaves-1
        # Dummy leaf: num_leaves
        if self.dummy:
            dummy_leaf_id = num_leaves

            self.n_leaves = num_leaves + 1

        # Populate Names
        for i in range(num_leaves):
            self.id_to_name[i] = str(i)  # Name is string of int
        
        if self.dummy:
            self.id_to_name[dummy_leaf_id] = "ROOT_DUMMY"

        # Internal IDs start after the leaves
        next_internal_id = self.n_leaves

        # 2. Build the Biological Tree (Unrooted Start)
        # We start with a central node connecting 3 leaves (0, 1, 2)
        # This guarantees we have a valid internal structure to grow from.
        if num_leaves < 3:
            raise ValueError(
                "Random generation requires at least 3 leaves to be interesting."
            )

        bio_root = next_internal_id
        next_internal_id += 1

        self._add_edge(0, bio_root)
        self._add_edge(1, bio_root)
        self._add_edge(2, bio_root)

        # Add remaining biological leaves (3..N-1)
        for leaf in range(3, num_leaves):
            # Pick a random existing edge to subdivide
            edges = []
            seen = set()
            for u in self.adj:
                for v in self.adj[u]:
                    # Only split biological edges (don't have dummy yet, so all are valid)
                    if (u, v) not in seen and (v, u) not in seen:
                        edges.append((u, v))
                        seen.add((u, v))
                        seen.add((v, u))

            u, v = random.choice(edges)

            # Remove old edge
            self._remove_edge(u, v)

            # Create new internal node
            w = next_internal_id
            next_internal_id += 1

            # Connect u-w, v-w, w-leaf
            self._add_edge(u, w)
            self._add_edge(v, w)
            self._add_edge(w, leaf)

        # 3. Inject the Super Root and Dummy
        # We attach the Super Root to the 'bio_root' we created earlier.
        super_root_id = next_internal_id
        next_internal_id += 1
        self.root = super_root_id  # Set the class root property!

        nbr0 = self.adj[0][0]  # leaf 0 has exactly one neighbor
        self._remove_edge(0, nbr0)

        self._add_edge(super_root_id, 0, length=0.0)
        self._add_edge(super_root_id, nbr0, length=0.0)

        # 4. Randomize lengths for all other edges
        for u in list(self.adj):
            for v in self.adj[u]:
                if (u, v) not in self.lengths:
                    # Biological branch length
                    L = random.uniform(0.1, 1.0)
                    self.lengths[(u, v)] = self.lengths[(v, u)] = L

    def _build_from_newick(self, newick: str):
        t = eteTree(newick)

        # Ensure we don't already have a dummy structure by checking for "ROOT_DUMMY" logic or similar.
        # But this method is for loading raw newick.
        # We will wrap it in the dummy structure HERE.

        # Create a new "Super Root"
        # We move the original tree to be a child of this new node,
        # and add the dummy leaf as the second child.
        new_root = eteTree()
        new_root.add_child(t, dist=0.0)  # Original tree attached here
        # Add dummy leaf. We'll give it a temporary name, but we'll assign UID later.
        dummy_leaf = new_root.add_child(name="ROOT_DUMMY", dist=0.0)

        # Point our tree reference to this new super structure
        t = new_root

        # 1) Collect leaves and assume their names are integers 1..n
        leaf_nodes = list(t.iter_leaves())
        # Sort leaves by name to ensure deterministic ID assignment
        # We need to handle mixed types if necessary, but typically names are strings.
        # Ensure ROOT_DUMMY is at the end or handled specially.
        real_leaves = [l for l in leaf_nodes if l.name != "ROOT_DUMMY"]
        try:
            real_leaves.sort(key=lambda x: int(x.name))
        except:
            real_leaves.sort(key=lambda x: x.name)

        self.n_leaves = len(real_leaves) + 1  # +1 for dummy

        # Assign UIDs to real leaves: 0 to N-2
        for i, n in enumerate(real_leaves):
            n.add_feature("uid", i)
            self.id_to_name[i] = n.name

        # Assign UID to dummy leaf: N-1
        dummy_uid = self.n_leaves - 1
        dummy_leaf.add_feature("uid", dummy_uid)
        self.id_to_name[dummy_uid] = "ROOT_DUMMY"

        next_internal_id = self.n_leaves

        # internal nodes get new IDs
        for n in t.traverse("postorder"):
            if not n.is_leaf():
                if "uid" not in n.features:
                    n.add_feature("uid", next_internal_id)
                    next_internal_id += 1

        self.root = t.uid
        # 3) Build adjacency and lengths
        for parent in t.traverse():
            u = parent.uid
            for child in parent.children:
                v = child.uid
                L = child.dist if child.dist is not None else 0.1
                self.adj[u].append(v)
                self.adj[v].append(u)
                self.lengths[(u, v)] = self.lengths[(v, u)] = L

    def length(self, u, v):
        return self.lengths.get((u, v), self.lengths.get((v, u)))

    def __str__(self):
        import networkx as nx
        from utils.bhv_movie import tree_to_newick

        # Build a NetworkX graph from adjacency/lengths and convert to Newick
        G = nx.Graph()
        # add nodes
        for u in self.adj:
            G.add_node(u)
        # add edges with length
        for u in self.adj:
            for v in self.adj[u]:
                if not G.has_edge(u, v):
                    G.add_edge(u, v, length=self.length(u, v))

        return tree_to_newick(
            G, root=self.root, dummy_node=self.n_leaves - 1, mapping=self.id_to_name
        )
