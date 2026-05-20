import random
import unittest

from utils.bhv_distance import bhv_geodesic_with_support
from utils.bhv_utils import BHVEncoder
from utils.random_tree import Tree
from utils.bhv_utils import (
    return_sampled_tree_orthant_velocity,
)
import numpy as np
from utils.metric_utils import calculate_norm_rf


def encode_newick(newick: str):
    tree = Tree(newick)
    enc = BHVEncoder()
    masks, lens = enc.return_BHV_encoding(tree)
    lengths = {m: float(l) for m, l in zip(masks, lens) if l is not None}
    return lengths, tree.n_leaves, tree.id_to_name


class TestGroundTruthGeodesicSampling(unittest.TestCase):
    START_TREE = "((0:0.10,1:0.10):0.20,(2:0.10,(3:0.10,4:0.10):0.20):0.30);"
    TARGET_TREE = "((0:0.10,(2:0.10,3:0.10):0.20):0.30,(1:0.10,4:0.10):0.20);"

    def test_return_sampled_tree_orthant_velocity_preserves_leaf_mapping(self):
        random.seed(123)
        np.random.seed(123)

        start_tree = self.START_TREE
        target_tree = self.TARGET_TREE

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

        sampled_newick, _ = return_sampled_tree_orthant_velocity(
            self.START_TREE,
            self.TARGET_TREE,
            0.5,
        )

        sampled_tree = Tree(sampled_newick)
        target_tree = Tree(self.TARGET_TREE)
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
        np.random.seed(0)

        start_tree = self.START_TREE
        target_tree = self.TARGET_TREE
        sampled_tree, _ = return_sampled_tree_orthant_velocity(
            start_tree,
            target_tree,
            1.0,
        )

        start_enc, n_leaves, _ = encode_newick(start_tree)
        target_enc, _, _ = encode_newick(target_tree)
        sampled_enc, _, _ = encode_newick(sampled_tree)

        geodesic_result = bhv_geodesic_with_support(
            start_enc, target_enc, n_leaves=n_leaves
        )
        self.assertGreater(geodesic_result["distance"], 0.0)

        final_geodesic = bhv_geodesic_with_support(
            sampled_enc, target_enc, n_leaves=n_leaves
        )
        self.assertLess(final_geodesic["distance"], 1e-3)


if __name__ == "__main__":
    unittest.main()
