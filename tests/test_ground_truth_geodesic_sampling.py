import os
import random
import unittest

from data.dataset import TreeDataset
from utils.bhv_distance import bhv_geodesic_with_support
from utils.bhv_movie import build_tree_from_splits
from utils.bhv_utils import BHVEncoder
from utils.random_tree import Tree
from utils.utils import number_to_name_newick
from utils.bhv_utils import (
    return_sampled_tree_orthant_velocity,
    return_sampled_tree_boundary_decisions,
)
import numpy as np
from utils.utils import has_polytomy_fast
from utils.metric_utils import calculate_norm_rf


def encode_newick(newick: str):
    tree = Tree(newick)
    enc = BHVEncoder()
    masks, lens = enc.return_BHV_encoding(tree)
    lengths = {m: float(l) for m, l in zip(masks, lens) if l is not None}
    return lengths, tree.n_leaves, tree.id_to_name


def sample(
        newick_starting_trees: list[str],
        real_tree_newick: str,
        phyla_embeddings,
        num_samples=1,
        T=1.0,
        dt_base=0.02,
        eps_len=1e-8,
        hit_tol=1e-10,
        max_events=1000,
        max_steps=20000
    ):

        max_logits = []

        # SPEED UP SAMPLING
        # 1) init: parse tree -> {mask: length}
        trees = []
        num_leaves = []
        mapping = []
        # Precompute cache for initial trees
        # Since topology changes in the loop, we will update this cache dynamically
        # Initialize tokenized structure cache
        current_newicks = list(newick_starting_trees)

        for nw in newick_starting_trees:
            t = Tree(nw)
            enc = BHVEncoder()
            masks, lens = enc.return_BHV_encoding(t)
            #Initial trees have no polytomies and all lengths should be greater than 0, so any 0 edges need to be removed
            trees.append({m: float(l) for m, l in zip(masks, lens) if l is not None})
            num_leaves.append(t.n_leaves)
            mapping.append(t.id_to_name)

        t = 0.0
        n_events = 0
        n_steps = 0
        n_topology_changes = 0

        while t < T and n_steps < max_steps and n_events < max_events:
            n_steps += 1

            # --- encode/tokenize current trees for the model ---

            # Use the real velocity here 
            _, velocity = return_sampled_tree_orthant_velocity(newick_starting_trees[0], real_tree_newick, t)

            dt_hit_list = []
            cache = []
            for b_idx, (td, v, n_leaves, mapp) in enumerate(zip(trees, velocity, num_leaves, mapping)):
                # model_masks = edge_splits[b_idx]
                # mask_idx = {mask: i for i, mask in enumerate(model_masks)}
                # V = v.squeeze(1).detach().cpu().numpy()
                V = velocity

                L = []
                V_val = []
                masks = []

                for m in td:
                    if m not in velocity:
                        # WE GOTTA FIX THIS LOL
                        print(f"Whoa there is a split missing in velocity masks! {m}")
                    else:
                        L.append(td[m])
                        V_val.append(V[m])
                        masks.append(m)

                V = np.array(V_val, dtype=np.float64)
                L = np.array(L, dtype=np.float64)
                
                if len(V) != len(L):
                    raise Exception("I assume these two things are equal length!")

                if (L < 0).any():
                    raise Exception("There are negative lengths that is not possible!")

                # --- compute dt_hit ---
                neg = (V < 0) & (L > eps_len)
                if np.any(neg):
                    dt_candidates = L[neg] / -V[neg]
                    dt_hit = float(np.min(dt_candidates))
                else:
                    dt_hit = float("inf")

                cache.append((td, L, V, n_leaves, mapp, dt_hit, masks))
                dt_hit_list.append(dt_hit)

            # ---- GLOBAL dt across the batch ----
            dt_hit_global = min(dt_hit_list) if len(dt_hit_list) else float("inf")
            # Experimenting here, dt_hit_global is not a good metric we just jump, jump, jump, so why not use dt_base
            # dt = min(dt_base, dt_hit_global, T - t)
            dt = min(dt_base, T-t)

            # defensive: prevent hard stall
            if dt <= 0:
                dt = min(dt_base, T - t)


            # ---- SECOND PASS: advance everyone with the SAME dt ----
            new_trees = []

            # Since update of token_cache happens per tree potentially, we need to defer it or track which ones changed.
            # However, batch indices align with zip(trees...), so we can update token_cache[i] if needed.

            for b_idx, (td, L, V, n_leaves, mapp, dt_hit, masks) in enumerate(
                cache
            ):
                # model_masks = edge_splits[b_idx]
                # --- advance ---
                L_new = L + dt * V
                # import pdb; pdb.set_trace()

                # Did we hit boundary this step?
                hit_boundary = (abs(dt - dt_hit) <= hit_tol) or (L_new <= eps_len).any()

                if hit_boundary:
                    hit = L_new <= eps_len
                    L_new[hit] = 0.0

                # update dict
                td2 = {m: float(l) for m, l in zip(masks, L_new) if l > eps_len}

                # We only need to rebuild Newick/Graph if we hit a boundary (topology changed)
                if hit_boundary:
                    import pdb; pdb.set_trace()
                    graph, td2_newick = build_tree_from_splits(
                        list(td2.keys()),
                        td2,
                        n_leaves,
                        root_leaf=n_leaves - 1,
                        mapping=mapp,
                    )

                    polytomy_nodes = has_polytomy_fast(td2_newick, unrooted_ok=False)
                    # td2 = {m: float(l) for m, l in zip(active_masks, L_new)}

                    if polytomy_nodes:
                        topology_changed = False
                        # For autoregressive step, we just use standard tokenizer for now as it's rare event
                        #tokenized_trees = self.model.tokenizer([td2_newick])
                        # import pdb; pdb.set_trace()

                        with torch.no_grad():
                            logit_outputs = self.forward(
                                tokenized_trees,
                                t,
                                phyla_embeddings,
                                autoregressive=True,
                            )

                        for output in logit_outputs:
                            x = output["logits"]
                            W = 0.5 * (x + x.T)  # [G,G]
                            W.fill_diagonal_(-float("inf"))
                            P = torch.sigmoid(
                                W
                            )  # mergeability prob, pick_group already does the sigmoid but I'm doing it here for logging later
                            # Can do something here to look at the prob of merging to see if the model is really learning anything or just learning junk for logging purposes
							# import pdb; pdb.set_trace()
                            max_logits.append(P.max().item())
                            res = pick_group(W, tau=0.55)
                            if res is None:
                                logger.debug("No merges found!")
                            else:
                                logger.debug(f"Merges found: {res}")
                                # import pdb; pdb.set_trace()
                                split_masks = [
                                    output["splits_represented"][idx] for idx in res
                                ]
                                new_split = 0
                                for sm in split_masks:
                                    new_split |= sm

                                if new_split in td2:
                                    logger.debug("Whoa already in there!")
                                else:
                                    # New length is average of merged splits
                                    td2[new_split] = 1e-3
                                topology_changed = True

                        n_events += 1
                        logger.debug("Finished processing merges")
                        if topology_changed:
                            n_topology_changes += 1

                    _, td2_newick_final = build_tree_from_splits(
                        list(td2.keys()),
                        td2,
                        n_leaves,
                        root_leaf=n_leaves - 1,
                        mapping=mapp,
                    )
                    # Update the cache for this batch index
                    new_item = self.model.tokenizer.compute_structural_cache(
                        [td2_newick_final]
                    )[0]

                    token_cache.update(b_idx, new_item)

                new_trees.append(td2)

            trees = new_trees
            t += dt

            if n_steps % 100 == 0:
                print(f"Step {n_steps}: dt={dt:.2e}, t={t:.2f}/{T}")

        # print(f"Sampling finished in {n_steps} steps. Total events: {n_events}")
        return [
            build_tree_from_splits(
                list(td.keys()),
                td,
                n_leaves=n_leaves,
                root_leaf=n_leaves - 1,
                mapping=mapp,
            )[1]
            for td, n_leaves, mapp in zip(trees, num_leaves, mapping)
        ], n_topology_changes, sum(max_logits) / len(max_logits) if len(max_logits) > 0 else 0.0


def geodesic_state_at_time(
    geodesic_result, tree1, tree2, t: float
):
    segments = geodesic_result["segments"]

    total_L = sum(seg["length"] for seg in segments)
    if total_L <= 0.0:
        return dict(tree1), {e: 0.0 for e in tree1}

    s = t * total_L
    cum = 0.0
    seg_idx = 0
    offset = 0.0
    for i, seg in enumerate(segments):
        if s <= cum + seg["length"] or i == len(segments) - 1:
            seg_idx = i
            offset = s - cum
            break
        cum += seg["length"]

    seg = segments[seg_idx]
    seg_len = seg["length"]
    alpha = 0.0 if seg_len == 0.0 else offset / seg_len

    keys = set(seg["start_lengths"].keys()) | set(seg["end_lengths"].keys())
    lengths = {
        e: (1.0 - alpha) * seg["start_lengths"].get(e, 0.0)
        + alpha * seg["end_lengths"].get(e, 0.0)
        for e in keys
    }
    velocity = {e: seg["velocity"].get(e, 0.0) * total_L for e in keys}
    return lengths, velocity


class TestGroundTruthGeodesicSampling(unittest.TestCase):
    def test_return_sampled_tree_orthant_velocity_preserves_leaf_mapping(self):
        random.seed(123)
        np.random.seed(123)

        data_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "example_data")
        )
        dataset = TreeDataset(
            nexus_root=os.path.join(data_root, "nexus"),
            mrbayes_root=os.path.join(data_root, "runs"),
            random_sanity_check=True,
            overfit_start_boundary_prefix_k=10,
            overfit_boundary_prefix_k=11,
        )

        real_tree = dataset.return_posterior_trees(0)[0]
        start_tree = dataset.sample_random_tree(real_tree)
        target_tree = dataset.resolve_training_target_tree(start_tree, real_tree)

        sampled_start, _ = return_sampled_tree_orthant_velocity(
            start_tree, target_tree, 0.0
        )
        sampled_end, _ = return_sampled_tree_orthant_velocity(
            start_tree, target_tree, 1.0
        )

        self.assertLess(calculate_norm_rf(sampled_start, start_tree), 1e-8)
        self.assertLess(calculate_norm_rf(sampled_end, target_tree), 1e-8)

    def test_direct_transition_target_uses_same_leaf_mapping_as_sampled_tree(self):
        random.seed(123)
        np.random.seed(123)

        data_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "example_data")
        )
        dataset = TreeDataset(
            nexus_root=os.path.join(data_root, "nexus"),
            mrbayes_root=os.path.join(data_root, "runs"),
            random_sanity_check=True,
            overfit_velocity_zero=True,
            overfit_start_boundary_prefix_k=10,
            overfit_boundary_prefix_k=11,
        )

        sample = dataset[0]

        sampled_tree = Tree(sample["newick_tree"])
        target_tree = Tree(sample["target_tree"])
        sampled_labels = sorted(
            [name for name in sampled_tree.id_to_name.values() if str(name).isdigit()],
            key=lambda x: int(x),
        )
        target_labels = sorted(
            [name for name in target_tree.id_to_name.values() if str(name).isdigit()],
            key=lambda x: int(x),
        )

        self.assertEqual(sampled_labels, target_labels)

    def test_ground_truth_geodesic_sampler(self):
        random.seed(0)
        data_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "example_data")
        )
        dataset = TreeDataset(
            nexus_root=os.path.join(data_root, "nexus"),
            mrbayes_root=os.path.join(data_root, "runs"),
        )

        posterior_trees = dataset.return_posterior_trees(0)
        self.assertTrue(posterior_trees)
        target_tree = posterior_trees[0]

        num_leaves = dataset.return_number_leaves(0)
        mapping = dataset.return_nexus_number_to_name(0)

        start_tree_raw = str(Tree(num_leaves=num_leaves, random=True))
        sample(
            newick_starting_trees=[start_tree_raw],
            real_tree_newick=str(target_tree),
            phyla_embeddings=None,
            num_samples=1,
            T=1.0,
            dt_base=0.02,
            eps_len=1e-8,
            hit_tol=1e-10,
            max_events=1000,
            max_steps=20000
        )
        import pdb; pdb.set_trace()
        # start_tree = number_to_name_newick(start_tree_raw, mapping, True)

        sampled_tree, geodesic_result = sample_with_ground_truth_geodesic(
            start_tree,
            target_tree,
            dt_base=0.02,
            T=1.0,
        )

        start_enc, n_leaves, _ = encode_newick(start_tree)
        target_enc, _, _ = encode_newick(target_tree)
        sampled_enc, _, _ = encode_newick(sampled_tree)

        self.assertGreater(geodesic_result["distance"], 0.0)

        final_geodesic = bhv_geodesic_with_support(
            sampled_enc, target_enc, n_leaves=n_leaves
        )
        self.assertLess(final_geodesic["distance"], 1e-3)


if __name__ == "__main__":
    unittest.main()
