from utils.bhv_distance import bhv_geodesic_with_support
from utils.random_tree import Tree
import logging 
logger = logging.getLogger(__name__)

class BHVEncoder():

    def _choose_root(self, tree, root=None):
        """Choose a root for an (unrooted) tree.

        If `root` is provided, use it. Otherwise, pick a random leaf in 1..n.
        Falls back to the smallest node id if no leaves are found.
        """
        if root is not None:
            return root

        # Prefer a random leaf among 1..n_leaves
        leaves = [u for u in tree.adj if len(tree.adj[u]) == 2]
        if len(leaves) > 1:
            #THIS SHOULD NOT HAPPEN
            import pdb; pdb.set_trace()
        else:
            return leaves[0]

    def compute_edge_masks(self, tree, root=None):
        """
        Returns:
        edge_masks: dict[(u,v)] -> mask over leaves below v, for directed edges u->v
                    Only for edges that correspond to nontrivial splits.
        Assumes leaves are labeled 1..n_leaves, internal nodes >= n_leaves.
        """
        #root = self._choose_root(tree, root)
        root = tree.root
        n = tree.n_leaves
        full = (1 << n) - 1

        parent = {}
        order = []

        # iterative DFS to get postorder
        stack = [root]
        parent[root] = None
        while stack:
            u = stack.pop()
            order.append(u)
            for v in tree.adj[u]:
                if v not in parent:
                    parent[v] = u
                    stack.append(v)

        # postorder accumulation of leaf masks
        #So every leaf has a bit mask associated with just that identity from 0 to n
        #Every internal node then gets ORd with that bit mask to make a bitmask that represents all nodes it is a parent of

        node_mask = {u: 0 for u in tree.adj}
        for u in reversed(order):
            if 0 <= u < n:  # leaf
                node_mask[u] = (1 << u)
            else:
                m = 0
                for v in tree.adj[u]:
                    if parent.get(v) == u:  # child
                        m |= node_mask[v]
                node_mask[u] = m

        # build edge masks for internal edges
        # For each internal edge take the sort of anti-not somehow like if both 1 then 0, this represents the split of the tree
        edge_masks = []
        edge_lengths = []
        for v in tree.adj:
            p = parent.get(v)
            if p is not None:
                A = node_mask[v]
                if A != 0 and A != full:  # nontrivial split
                    # canonical side
                    Ac = full ^ A
                    canon = min(A, Ac)
                    edge_masks.append(canon)
                    edge_lengths.append(tree.length(p,v))

        return edge_masks, edge_lengths

    def return_BHV_encoding(self, tree):
        #Find root of the tree
        
        edge_masks, edge_lengths = self.compute_edge_masks(tree)
        return edge_masks, edge_lengths
    
    def BHV_length(self, one, two):
        edge_mask_1, edge_length_1 = one 
        edge_mask_2, edge_length_2 = two 

        t1 = {x:y for x,y in zip(edge_mask_1, edge_length_1)}
        t2 = {x:y for x,y in zip(edge_mask_2, edge_length_2)}
        result = bhv_geodesic_with_support(t1, t2, n_leaves=self.n_leaves)

        print("BHV distance:", result["distance"])
        for i, seg in enumerate(result["segments"]):
            print(f"Segment {i}:")
            print("  Ai (collapsed):", seg["Ai"])
            print("  Bi (grown):    ", seg["Bi"])
            print("  ratio:", seg["ratio"])
            # seg["start_splits"], seg["end_splits"] give you orthant topology at each step

def _bio_full_mask(n_leaves):
    return (1 << (n_leaves - 1)) - 1


def _sort_masks(masks):
    return sorted({int(mask) for mask in masks}, key=lambda mask: (mask.bit_count(), mask))


def _is_strict_subset(sub, sup):
    sub = int(sub)
    sup = int(sup)
    return sub != sup and (sub & ~sup) == 0


def _orient_split_away_from_dummy(split, n_leaves):
    split = int(split)
    root_leaf = n_leaves - 1
    root_bit = 1 << root_leaf
    full_mask = (1 << n_leaves) - 1
    cluster = full_mask ^ split if split & root_bit else split
    return cluster & _bio_full_mask(n_leaves)


def _resolve_num_leaves(num_leaves, batch_index, edge_split_masks_b):
    if isinstance(num_leaves, int):
        return int(num_leaves)
    if isinstance(num_leaves, (list, tuple)):
        if len(num_leaves) > batch_index:
            return int(num_leaves[batch_index])
    try:
        import torch

        if isinstance(num_leaves, torch.Tensor):
            if num_leaves.numel() == 0:
                return 0
            if num_leaves.numel() == 1:
                return int(num_leaves.item())
            return int(num_leaves[batch_index].item())
    except Exception:
        pass

    max_bit = 0
    for split in edge_split_masks_b:
        split_int = int(split)
        if split_int != 0:
            max_bit = max(max_bit, split_int.bit_length())
    return max_bit


def _positive_bhv_tree_dict(tree, eps=1e-8):
    masks, lengths = BHVEncoder().return_BHV_encoding(tree)
    return {
        int(mask): float(length)
        for mask, length in zip(masks, lengths)
        if length is not None and float(length) > float(eps)
    }


def _full_bhv_tree_dict(tree):
    masks, lengths = BHVEncoder().return_BHV_encoding(tree)
    return {
        int(mask): float(length)
        for mask, length in zip(masks, lengths)
        if length is not None
    }


def return_boundary_training_geodesic(newick_tree_one, newick_tree_two):
    t1 = Tree(newick_tree_one)
    t2 = Tree(newick_tree_two)
    tree1 = _full_bhv_tree_dict(t1)
    tree2 = _full_bhv_tree_dict(t2)
    return bhv_geodesic_with_support(
        tree1,
        tree2,
        n_leaves=t1.n_leaves,
        drop_zero_length_edges=False,
        enable_prefix_birth_groups=False,
    )


def _resolve_boundary_geodesic(
    newick_tree_one,
    newick_tree_two,
    *,
    legacy_training_semantics: bool = False,
):
    if legacy_training_semantics:
        return return_boundary_training_geodesic(newick_tree_one, newick_tree_two)

    t1 = Tree(newick_tree_one)
    t2 = Tree(newick_tree_two)
    tree1 = _positive_bhv_tree_dict(t1)
    tree2 = _positive_bhv_tree_dict(t2)
    return bhv_geodesic_with_support(tree1, tree2, n_leaves=t1.n_leaves)


def _lookup_component_positions(mask_to_positions, component, bio_full):
    component = int(component)
    positions = mask_to_positions.get(component)
    if positions:
        return positions

    complement = int(bio_full) ^ component
    if complement == 0 or complement == int(bio_full):
        return None
    return mask_to_positions.get(complement)


def _internal_bio_clusters_from_splits(splits, n_leaves):
    bio_full = _bio_full_mask(n_leaves)
    clusters = set()
    for split in splits:
        cluster = _orient_split_away_from_dummy(split, n_leaves)
        if 1 < cluster.bit_count() < bio_full.bit_count():
            clusters.add(cluster)
    return clusters


def get_structural_polytomy_groups_from_newick(newick_tree, min_children=3):
    tree = Tree(newick_tree)
    split_masks, _ = BHVEncoder().return_BHV_encoding(tree)
    current_clusters = _internal_bio_clusters_from_splits(split_masks, tree.n_leaves)
    current_regions = set(current_clusters)
    current_regions.add(_bio_full_mask(tree.n_leaves))

    groups = []
    seen = set()
    for parent in sorted(current_regions, key=lambda region: (region.bit_count(), region)):
        children = tuple(_direct_children(parent, current_clusters, tree.n_leaves))
        if len(children) < min_children:
            continue
        if children in seen:
            continue
        seen.add(children)
        groups.append(list(children))

    return groups


def _direct_children(parent, internal_clusters, n_leaves):
    parent = int(parent)
    leaf_masks = [1 << leaf for leaf in range(n_leaves - 1) if parent & (1 << leaf)]
    candidates = [cluster for cluster in internal_clusters if _is_strict_subset(cluster, parent)]
    candidates.extend(leaf_masks)

    children = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if _is_strict_subset(candidate, other) and _is_strict_subset(other, parent):
                dominated = True
                break
        if not dominated:
            children.append(int(candidate))

    children = _sort_masks(children)
    union = 0
    for child in children:
        union |= child

    if union != parent:
        raise ValueError(
            f"Children of parent mask {parent} do not partition the parent. "
            f"Observed union {union}."
        )

    return children


def get_batch_structural_polytomy_indices(
    edge_split_masks,
    edge_mask,
    min_children=3,
    num_leaves=None,
):
    import torch

    if edge_mask.dim() != 2:
        raise ValueError(f"edge_mask must be [B,T], got {tuple(edge_mask.shape)}")

    batch_polytomy_index = []
    batch_polytomy_splits = []

    for batch_index, splits_b in enumerate(edge_split_masks):
        valid_pos = torch.nonzero(edge_mask[batch_index], as_tuple=False).squeeze(1)
        if len(splits_b) != edge_mask[batch_index].sum().item():
            raise ValueError("Length mismatch between splits and valid edge mask. This SHOULD NOT HAPPEN.")

        n_b = _resolve_num_leaves(num_leaves, batch_index, splits_b)
        if n_b <= 1:
            batch_polytomy_index.append([])
            batch_polytomy_splits.append([])
            continue

        bio_full = _bio_full_mask(n_b)
        mask_to_positions = {}
        for pos, split in zip(valid_pos.tolist(), splits_b):
            split_int = int(split)
            if split_int == 0:
                continue
            bio_mask = _orient_split_away_from_dummy(split_int, n_b)
            if bio_mask == 0 or bio_mask == bio_full:
                continue
            mask_to_positions.setdefault(int(bio_mask), []).append(int(pos))

        current_clusters = _internal_bio_clusters_from_splits(splits_b, n_b)
        current_regions = set(current_clusters)
        current_regions.add(bio_full)

        groups = []
        group_splits = []
        seen = set()
        for parent in sorted(current_regions, key=lambda region: (region.bit_count(), region)):
            children = _direct_children(parent, current_clusters, n_b)
            if len(children) < min_children:
                continue

            key = tuple(children)
            if key in seen:
                continue
            seen.add(key)

            group_indices = []
            valid_group = True
            for child in children:
                positions = _lookup_component_positions(
                    mask_to_positions,
                    child,
                    bio_full,
                )
                if not positions:
                    valid_group = False
                    break
                group_indices.append(int(positions[0]))

            if valid_group:
                groups.append(torch.tensor(group_indices, dtype=torch.long, device=edge_mask.device))
                group_splits.append([int(child) for child in children])

        batch_polytomy_index.append(groups)
        batch_polytomy_splits.append(group_splits)

    return batch_polytomy_index, batch_polytomy_splits


def get_batch_explicit_structural_group_indices(
    edge_split_masks,
    edge_mask,
    structural_groups,
    num_leaves=None,
):
    import torch

    if edge_mask.dim() != 2:
        raise ValueError(f"edge_mask must be [B,T], got {tuple(edge_mask.shape)}")

    batch_group_index = []
    batch_group_splits = []

    for batch_index, splits_b in enumerate(edge_split_masks):
        valid_pos = torch.nonzero(edge_mask[batch_index], as_tuple=False).squeeze(1)
        if len(splits_b) != edge_mask[batch_index].sum().item():
            raise ValueError("Length mismatch between splits and valid edge mask. This SHOULD NOT HAPPEN.")

        n_b = _resolve_num_leaves(num_leaves, batch_index, splits_b)
        if n_b <= 1:
            batch_group_index.append([])
            batch_group_splits.append([])
            continue

        bio_full = _bio_full_mask(n_b)
        mask_to_positions = {}
        for pos, split in zip(valid_pos.tolist(), splits_b):
            split_int = int(split)
            if split_int == 0:
                continue
            bio_mask = _orient_split_away_from_dummy(split_int, n_b)
            if bio_mask == 0 or bio_mask == bio_full:
                continue
            mask_to_positions.setdefault(int(bio_mask), []).append(int(pos))

        groups = []
        group_splits = []
        for group in structural_groups[batch_index]:
            group = [int(component) for component in group]
            if len(group) < 2:
                continue

            group_indices = []
            valid_group = True
            for component in group:
                positions = _lookup_component_positions(
                    mask_to_positions,
                    component,
                    bio_full,
                )
                if not positions:
                    valid_group = False
                    break
                group_indices.append(int(positions[0]))

            if valid_group:
                groups.append(
                    torch.tensor(group_indices, dtype=torch.long, device=edge_mask.device)
                )
                group_splits.append(group)

        batch_group_index.append(groups)
        batch_group_splits.append(group_splits)

    return batch_group_index, batch_group_splits


def _merge_schedule_for_parent(parent, atoms, target_clusters):
    target_clusters = _sort_masks(target_clusters)
    if not target_clusters:
        return []

    child_map = {}
    for cluster in target_clusters:
        candidates = [
            atom for atom in atoms
            if (int(atom) & ~int(cluster)) == 0
        ]
        candidates.extend(
            other for other in target_clusters if _is_strict_subset(other, cluster)
        )

        maximal_children = []
        for candidate in candidates:
            dominated = False
            for other in candidates:
                if _is_strict_subset(candidate, other) and _is_strict_subset(other, cluster):
                    dominated = True
                    break
            if not dominated:
                maximal_children.append(int(candidate))

        maximal_children = _sort_masks(maximal_children)
        union = 0
        for child in maximal_children:
            union |= child

        if union != cluster or len(maximal_children) < 2:
            raise ValueError(
                f"Target cluster {cluster} inside parent {parent} is not a valid "
                f"merge from current components. Children={maximal_children}, union={union}."
            )

        child_map[cluster] = maximal_children

    pending = list(target_clusters)
    current_components = set(int(atom) for atom in atoms)
    events = []

    while pending:
        ready = []
        for cluster in pending:
            children = child_map[cluster]
            if all(child in current_components for child in children):
                ready.append((cluster, children))

        if not ready:
            raise ValueError(
                f"Could not find a ready merge inside parent {parent}. "
                f"Pending clusters={pending}, current_components={sorted(current_components)}."
            )

        used_children = set()
        for _, children in ready:
            for child in children:
                if child in used_children:
                    raise ValueError(
                        f"Parent {parent} produced overlapping ready merges in one step."
                    )
                used_children.add(child)

        ready = sorted(
            ready,
            key=lambda item: (len(item[1]), item[0].bit_count(), item[0]),
        )
        for cluster, children in ready:
            for child in children:
                current_components.remove(child)
            current_components.add(cluster)

        ready_clusters = {cluster for cluster, _ in ready}
        pending = [cluster for cluster in pending if cluster not in ready_clusters]
        events.append(ready)

    return events


def _build_boundary_merge_events(boundary_lengths, boundary_births, n_leaves):
    current_clusters = _internal_bio_clusters_from_splits(boundary_lengths.keys(), n_leaves)
    final_clusters = _internal_bio_clusters_from_splits(
        list(boundary_lengths.keys()) + [int(split) for split in boundary_births],
        n_leaves,
    )
    new_clusters = _sort_masks(final_clusters - current_clusters)

    if not new_clusters:
        return []

    current_regions = set(current_clusters)
    current_regions.add(_bio_full_mask(n_leaves))

    parent_to_clusters = {}
    for cluster in new_clusters:
        candidate_parents = [
            region for region in current_regions if _is_strict_subset(cluster, region)
        ]
        if not candidate_parents:
            raise ValueError(f"Could not locate a current parent region for cluster {cluster}.")

        parent = min(candidate_parents, key=lambda region: (region.bit_count(), region))
        parent_to_clusters.setdefault(parent, []).append(cluster)

    parent_events = []
    for parent in sorted(parent_to_clusters, key=lambda region: (region.bit_count(), region)):
        atoms = _direct_children(parent, current_clusters, n_leaves)
        if len(atoms) < 2:
            raise ValueError(
                f"Parent {parent} has only {len(atoms)} children, so it cannot be refined."
            )

        levels = _merge_schedule_for_parent(parent, atoms, parent_to_clusters[parent])
        parent_events.append(levels)

    events = []
    max_depth = max(len(levels) for levels in parent_events)
    for depth in range(max_depth):
        labels = []
        for levels in parent_events:
            if depth < len(levels):
                labels.extend(levels[depth])
        if labels:
            events.append(
                sorted(
                    labels,
                    key=lambda item: (len(item[1]), item[0].bit_count(), item[0]),
                )
            )

    returned_clusters = {int(cluster) for level in events for cluster, _ in level}
    if returned_clusters != set(new_clusters):
        raise ValueError(
            "Boundary merge schedule did not recover the full post-boundary refinement. "
            f"Returned={sorted(returned_clusters)}, expected={new_clusters}."
        )

    return events


def _current_parent_and_children_for_split(split, current_clusters, n_leaves):
    current_regions = set(int(cluster) for cluster in current_clusters)
    current_regions.add(_bio_full_mask(n_leaves))

    candidate_parents = [
        region for region in current_regions if _is_strict_subset(int(split), region)
    ]
    if not candidate_parents:
        raise ValueError(f"Could not locate a current parent region for split {split}.")

    parent = min(candidate_parents, key=lambda region: (region.bit_count(), region))
    children = _direct_children(parent, current_clusters, n_leaves)
    return int(parent), [int(child) for child in children]


def _filter_training_boundary_events(boundary_paths):
    training_events = []
    for boundary_path in boundary_paths:
        filtered_events = []
        for event in boundary_path["events"]:
            labels = [
                label
                for label in event["labels"]
                if len(label["components"]) >= 3
            ]
            if labels:
                filtered_events.append(
                    {
                        "newick": event["newick"],
                        "labels": labels,
                    }
                )
        for event_idx, event in enumerate(filtered_events):
            event["stop_after_merge"] = bool(
                event_idx == (len(filtered_events) - 1)
                and len(event["labels"]) == 1
            )
            training_events.append(event)
    return training_events


def _split_multi_label_training_events(filtered_events):
    from utils.bhv_movie import build_tree_from_splits

    split_events = []
    encoder = BHVEncoder()

    for event in filtered_events:
        labels = list(event["labels"])
        if len(labels) <= 1:
            split_events.append(
                {
                    "newick": event["newick"],
                    "labels": labels,
                }
            )
            continue

        current_newick = event["newick"]
        current_tree = Tree(current_newick)
        split_masks, split_lengths = encoder.return_BHV_encoding(current_tree)
        current_lengths = {
            int(mask): float(length)
            for mask, length in zip(split_masks, split_lengths)
            if length is not None and float(length) > 1e-8
        }
        n_leaves = current_tree.n_leaves
        mapping = current_tree.id_to_name

        for label in labels:
            merge_components = tuple(
                int(label["components"][idx]) for idx in label["merge_indices"]
            )
            result_split = int(label["result_split"])
            current_clusters = _internal_bio_clusters_from_splits(
                current_lengths.keys(),
                n_leaves,
            )
            parent_split, polytomy_components = _current_parent_and_children_for_split(
                result_split,
                current_clusters,
                n_leaves,
            )
            merge_indices = []
            for component in merge_components:
                if component not in polytomy_components:
                    raise ValueError(
                        f"Cannot split multi-label event: component {component} not "
                        f"present in current polytomy {polytomy_components}."
                    )
                merge_indices.append(polytomy_components.index(component))

            split_events.append(
                {
                    "newick": current_newick,
                    "labels": [
                        {
                            "result_split": result_split,
                            "parent_split": int(parent_split),
                            "components": polytomy_components,
                            "merge_indices": merge_indices,
                        }
                    ],
                }
            )

            current_lengths[result_split] = 0.1
            _, current_newick = build_tree_from_splits(
                list(current_lengths.keys()),
                current_lengths,
                n_leaves,
                root_leaf=n_leaves - 1,
                mapping=mapping,
            )

    for event_idx, event in enumerate(split_events):
        event["stop_after_merge"] = bool(event_idx == (len(split_events) - 1))

    return split_events


def return_tree_boundary_merge_paths(
    newick_tree_one,
    newick_tree_two,
    verbose=False,
    id_to_test=None,
    legacy_training_semantics: bool = False,
):
    from utils.bhv_movie import build_tree_from_splits

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    t1 = Tree(newick_tree_one)
    t2 = Tree(newick_tree_two)

    geodesic_result = _resolve_boundary_geodesic(
        newick_tree_one,
        newick_tree_two,
        legacy_training_semantics=legacy_training_semantics,
    )
    segments = geodesic_result["segments"]

    if id_to_test is not None:
        idxs = [id_to_test]
    else:
        idxs = list(range(len(segments) - 1))

    boundary_paths = []
    for bi in idxs:
        boundary_lengths = {
            int(mask): float(length)
            for mask, length in segments[bi]["end_lengths"].items()
            if length > 1e-8
        }
        boundary_births = [int(split) for split in segments[bi]["Bi"]]

        _, start_newick = build_tree_from_splits(
            list(boundary_lengths.keys()),
            boundary_lengths,
            t1.n_leaves,
            root_leaf=t1.n_leaves - 1,
            mapping=t1.id_to_name,
        )

        current_clusters = _internal_bio_clusters_from_splits(
            boundary_lengths.keys(),
            t1.n_leaves,
        )
        final_clusters = _internal_bio_clusters_from_splits(
            list(boundary_lengths.keys()) + boundary_births,
            t1.n_leaves,
        )
        new_clusters = _sort_masks(final_clusters - current_clusters)

        end_lengths = dict(boundary_lengths)
        for cluster in new_clusters:
            end_lengths[int(cluster)] = 0.1

        _, end_newick = build_tree_from_splits(
            list(end_lengths.keys()),
            end_lengths,
            t1.n_leaves,
            root_leaf=t1.n_leaves - 1,
            mapping=t1.id_to_name,
        )

        current_lengths = dict(boundary_lengths)
        events = []
        for labels in _build_boundary_merge_events(
            boundary_lengths,
            boundary_births,
            t1.n_leaves,
        ):
            event_current_clusters = _internal_bio_clusters_from_splits(
                current_lengths.keys(),
                t1.n_leaves,
            )
            _, newick = build_tree_from_splits(
                list(current_lengths.keys()),
                current_lengths,
                t1.n_leaves,
                root_leaf=t1.n_leaves - 1,
                mapping=t1.id_to_name,
            )

            emitted_labels = []
            for split_merge, comps in labels:
                check = 0
                for comp in comps:
                    check |= int(comp)
                if check != int(split_merge):
                    raise ValueError(
                        f"Merged components {comps} do not equal resulting split {split_merge}."
                    )

                parent_split, polytomy_components = _current_parent_and_children_for_split(
                    split_merge,
                    event_current_clusters,
                    t1.n_leaves,
                )
                merge_indices = []
                for comp in comps:
                    comp_int = int(comp)
                    if comp_int not in polytomy_components:
                        raise ValueError(
                            f"Merge component {comp_int} is not in current polytomy "
                            f"{polytomy_components} for split {split_merge}."
                        )
                    merge_indices.append(polytomy_components.index(comp_int))

                emitted_labels.append(
                    {
                        "result_split": int(split_merge),
                        "parent_split": int(parent_split),
                        "components": polytomy_components,
                        "merge_indices": merge_indices,
                    }
                )
                current_lengths[int(split_merge)] = 0.1

            events.append({"newick": newick, "labels": emitted_labels})

        boundary_paths.append(
            {
                "boundary_index": bi,
                "global_time": float(segments[bi]["lambda_end"]),
                "start_newick": start_newick,
                "end_newick": end_newick,
                "births": new_clusters,
                "events": events,
            }
        )

    return boundary_paths


def return_sampled_tree_boundary_decisions(
    newick_tree_one,
    newick_tree_two,
    verbose=False,
    id_to_test=None,
    require_complete_boundary=False,
    split_multi_label_events=False,
    legacy_training_semantics: bool = False,
):
    _ = require_complete_boundary
    boundary_paths = return_tree_boundary_merge_paths(
        newick_tree_one,
        newick_tree_two,
        verbose=verbose,
        id_to_test=id_to_test,
        legacy_training_semantics=legacy_training_semantics,
    )
    training_events = _filter_training_boundary_events(boundary_paths)
    if split_multi_label_events:
        training_events = _split_multi_label_training_events(training_events)
    return training_events

def return_sampled_tree_orthant_velocity(
    newick_tree_one,
    newick_tree_two,
    time_point,
    extra = False,
    legacy_training_semantics: bool = False,
):
    from utils.bhv_movie import sample_tree_along_geodesic
    t1 = Tree(newick_tree_one)
    t2 = Tree(newick_tree_two)

    geodesic_result = _resolve_boundary_geodesic(
        newick_tree_one,
        newick_tree_two,
        legacy_training_semantics=legacy_training_semantics,
    )
    G, newick, info = sample_tree_along_geodesic(
        geodesic_result,
        t1.n_leaves,
        u=time_point,
        mapping=t1.id_to_name,
    )

    #This was debugging for a particular tree
    # # newick = '(((((((((((((((((((((((26:0.0,27:0.0,28:0.0,29:0.0):0.365231256756337,30:0.0,31:0.0):0.46562144283500395,32:0.0):0.13098585901031182,(33:0.0,34:0.0):0.5291394654659506):0.11776353850130768,35:0.0):0.26157583756205705,(24:0.0,25:0.0):0.36258921686734963):0.5033803018034739,36:0.0):0.20237942744692658,37:0.0):0.3963250321841045,38:0.0):0.3885226443458947,39:0.0):0.5347065021413112,40:0.0):0.1889407459500561,((41:0.0,42:0.0):0.3123941602336278,43:0.0):0.4617405890461322):0.4096061346588971,44:0.0):0.09230307198090551,45:0.0):0.3418153243061678,46:0.0):0.3554843469643267,47:0.0):0.10002600125544758,48:0.0,49:0.0):0.4654256573541056,(((16:0.0,17:0.0,18:0.0):0.0738641626933164,19:0.0):0.31399054621088424,(20:0.0,21:0.0,22:0.0):0.07592098916706257,23:0.0):0.3001943741954219):0.24382687114231433,((((((((((50:0.0,51:0.0):0.22142174476410542,52:0.0):0.17074685221592145,53:0.0):0.22838704277064867,(54:0.0,55:0.0,56:0.0):0.37569841866156367):0.13616650211948078,57:0.0):0.0641547613661573,58:0.0):0.5359239417814602,59:0.0):0.25989987338680104,((60:0.0,61:0.0,62:0.0):0.43040397244018275,(63:0.0,64:0.0):0.39425131617022474,65:0.0,66:0.0):0.5344297477988371):0.22352185160016247,67:0.0,68:0.0):0.5166211118827367,69:0.0,70:0.0):0.43780235312355326):0.4562863329696488,71:0.0):0.44293449831375736,72:0.0):0.15731748844500548,((((((((0:0.0,1:0.0):0.16131819778555037,2:0.0):0.1598279400172913,3:0.0):0.07150417342001554,4:0.0):0.4143589617385931,5:0.0):0.18956447983105795,6:0.0):0.26457093652387426,7:0.0):0.3509120493508661,(((((8:0.0,9:0.0,10:0.0):0.23219967414249623,11:0.0):0.47890150538212484,(12:0.0,13:0.0):0.5300847583428087):0.19037701078440555,14:0.0):0.4826082684585838,15:0.0):0.21491917547287828):0.24452064127567635):0.3516082338078238,(((((((((((((((((((((100:0.0,101:0.0):0.4075151593913468,98:0.0,99:0.0):0.35246140233112444,(102:0.0,103:0.0):0.31257845845204657,104:0.0):0.13875849332367882,105:0.0):0.3956641634593137,(106:0.0,107:0.0):0.3082347174751331):0.4173298184039069,108:0.0):0.1890248480912928,(96:0.0,97:0.0):0.4099729156249853):0.26452779741123134,109:0.0):0.4070499942463294,110:0.0):0.11205970478777688,(111:0.0,112:0.0):0.2561745153331085):0.49159967584084857,(((91:0.0,92:0.0):0.08362457977610852,89:0.0,90:0.0,93:0.0,94:0.0):0.18330279434802157,95:0.0):0.3493371033744729):0.3663038054950591,113:0.0):0.23649380819270865,114:0.0):0.4464045694170972,115:0.0):0.5178991085420517,(((((116:0.0,117:0.0):0.1626560063932636,118:0.0):0.22208903780237932,119:0.0):0.3658530512957051,120:0.0):0.1166761718242802,(121:0.0,122:0.0):0.32511291828552646):0.5020210696622494):0.07262859964821185,(((((((((81:0.0,82:0.0):0.4701788003557065,83:0.0):0.2838378314914868,84:0.0):0.36065683669899545,85:0.0):0.12577566798746628,86:0.0):0.327628012142499,87:0.0):0.11414385983118146,88:0.0):0.10439351856711432,78:0.0,79:0.0,80:0.0):0.2883345348574025,((((73:0.0,74:0.0):0.16959541461333827,75:0.0):0.3470113937468134,76:0.0):0.15024189043096217,77:0.0):0.3342242085245652):0.47486736798186163):0.4584589852726523,123:0.0):0.12823236347459735,124:0.0):0.4647828243748471,125:0.0):0.23519184935769752,(126:0.0,127:0.0):0.20818437562919903):0.22921477851937236,((((131:0.0,132:0.0,133:0.0):0.49818678251290305,(128:0.0,129:0.0,130:0.0):0.1081456116685175,134:0.0):0.374861791247522,135:0.0):0.0625760728935306,136:0.0):0.176919171345831):0.4956524440616365,((((((((139:0.0,140:0.0,141:0.0):0.2044406361563448,(137:0.0,138:0.0):0.24757861246008647):0.36496011409578083,142:0.0):0.3394472667518442,143:0.0):0.056707772310875176,144:0.0):0.15478537361656458,((((145:0.0,146:0.0):0.32455724511050016,147:0.0):0.15788405331664848,148:0.0):0.11136587699554824,149:0.0):0.4129118538256446):0.4299676167090581,150:0.0):0.07132799911059619,(((151:0.0,152:0.0):0.35895846909210893,153:0.0):0.15181817937207243,154:0.0):0.4425292371341138):0.24829617861092312);'
    # test = Tree(newick)
    # edge_mask, edge_length = enc.return_BHV_encoding(test)
    # active_edge_mask = []
    # active_edge_length = []
    # for i,z in zip(edge_mask, edge_length):
    #     if z > 1e-6:
    #         active_edge_mask.append(i)
    #         active_edge_length.append(z)
    # print(f"Sampled tree has {len(edge_mask)} edges, velocity has {len(info['velocity'])} entries")
    # real_max_bit = max(m.bit_length() for m in active_edge_mask)
    # for i in info['active_velocity']:
    #     vel = i
    #     if vel.bit_length() == real_max_bit+1:
    #         vel = remove_bit(vel, t1.n_leaves+1)
    #         print("Adjusted velocity bitmask by removing dummy leaf bit.")
    #     if vel not in active_edge_mask:
    #         print(f"Velocity entry {vel} not found in sampled tree edge masks!")
    #         print(bin(vel))
    #         print([i for i in range(vel.bit_length()) if (vel >> i) & 1])
    #         print([[i for i in range(vel.bit_length()) if (vel >> i) & 1] for vel in active_edge_mask])
    #         import pdb; pdb.set_trace()
    #     else:
    #         print("FOUND WOOHOOOO")
    # import pdb; pdb.set_trace()
    if extra:
        return newick, info['active_velocity'], geodesic_result

    return newick, info['active_velocity']


def test_bhv_on_two_random_20_leaf_trees():
    from utils.bhv_movie import make_bhv_topology_movie
    n = 20
    print("Generating random trees...")
    T1 = Tree(num_leaves=n, random=True)
    T2 = Tree(num_leaves=n, random=True)

    print("Tree 1 Newick:", T1)
    print("Tree 2 Newick:", T2)

    enc = BHVEncoder()

    print("Encoding trees into bitmask form...")
    edge_masks_1, edge_lengths_1 = enc.return_BHV_encoding(T1)
    edge_masks_2, edge_lengths_2 = enc.return_BHV_encoding(T2)

    tree1 = {m: l for m, l in zip(edge_masks_1, edge_lengths_1)}
    tree2 = {m: l for m, l in zip(edge_masks_2, edge_lengths_2)}

    print("Computing BHV geodesic with support pairs...")
    result = bhv_geodesic_with_support(tree1, tree2, n_leaves=T1.n_leaves)

    print("\n======================")
    print("BHV DISTANCE =", result["distance"])
    print("======================\n")

    print("Common-edge contribution squared =", result["common_sq"])
    print("Disjoint-edge contribution squared =", result["disjoint_sq"])
    print("Number of support pairs =", len(result["A_support"]))
    print()

    for i, seg in enumerate(result["segments"]):
        print(f"--- Segment {i} ---")
        print("Ai (collapse):", seg["Ai"])
        print("Bi (grow):    ", seg["Bi"])
        print("||A||=", seg["normA"], "||B||=", seg["normB"], "ratio=", seg["ratio"])
        print("#start splits =", len(seg["start_splits"]))
        print("#end splits   =", len(seg["end_splits"]))
        print()

    print("Test completed.")

    make_bhv_topology_movie(
        result,
        n_leaves=T1.n_leaves,
        root = T1.n_leaves-1,
        filename="bhv_topology_20leaf.gif",
        mapping=T1.id_to_name,
        F=20,
        fps=1,   # 1 frame per second (one per step)
    )


##############################################################################
# Run the test
##############################################################################

if __name__ == "__main__":
    test_bhv_on_two_random_20_leaf_trees()

        
       
        
