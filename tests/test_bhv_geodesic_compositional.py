import unittest
from unittest import mock

from ete3 import Tree as EteTree

from run.TrainingModule import _apply_merge_subset_to_newick
from utils.bhv_distance import bhv_geodesic_with_support
from utils.bhv_movie import build_tree_from_splits, sample_tree_along_geodesic
from utils.bhv_utils import (
    BHVEncoder,
    _internal_bio_clusters_from_splits,
    _sort_masks,
    return_tree_boundary_merge_paths,
)
from utils.random_tree import Tree


def _encode_positive(newick: str):
    tree = Tree(newick)
    masks, lengths = BHVEncoder().return_BHV_encoding(tree)
    encoded = {
        int(mask): float(length)
        for mask, length in zip(masks, lengths)
        if float(length) > 1e-8
    }
    return encoded, tree.n_leaves, tree.id_to_name


def _norm_rf(left_newick: str, right_newick: str) -> float:
    rf, max_rf, *_ = EteTree(left_newick).robinson_foulds(
        EteTree(right_newick),
        unrooted_trees=True,
    )
    return 0.0 if max_rf == 0 else rf / max_rf


def _boundary_end_newick(segment, n_leaves: int, mapping):
    boundary_lengths = {
        int(mask): float(length)
        for mask, length in segment["end_lengths"].items()
        if float(length) > 1e-8
    }
    boundary_births = [int(split) for split in segment["Bi"]]

    current_clusters = _internal_bio_clusters_from_splits(
        boundary_lengths.keys(),
        n_leaves,
    )
    final_clusters = _internal_bio_clusters_from_splits(
        list(boundary_lengths.keys()) + boundary_births,
        n_leaves,
    )
    new_clusters = _sort_masks(final_clusters - current_clusters)

    end_lengths = dict(boundary_lengths)
    for cluster in new_clusters:
        end_lengths[int(cluster)] = 0.1

    _, newick = build_tree_from_splits(
        list(end_lengths.keys()),
        end_lengths,
        n_leaves,
        root_leaf=n_leaves - 1,
        mapping=mapping,
    )
    return newick, new_clusters


def _first_nondegenerate_segment(segments):
    for segment in segments:
        if float(segment["length"]) > 1e-12:
            return segment
    return segments[0]


def _first_boundary_segment(segments):
    for segment in segments:
        if segment["Bi"]:
            return segment
    return _first_nondegenerate_segment(segments)


def _nonempty_boundary_paths(paths):
    return [
        path
        for path in paths
        if path["births"] or any(event["labels"] for event in path["events"])
    ]


class TestBhvGeodesicCompositional(unittest.TestCase):
    def test_apply_merge_subset_preserves_existing_lengths(self):
        start = "((0:0.0,1:0.0,2:0.0):0.37,(3:0.0,4:0.0):0.91);"

        updated = _apply_merge_subset_to_newick(
            None,
            start,
            subset=(1, 2),
        )
        self.assertIsNotNone(updated)

        masks, lengths = BHVEncoder().return_BHV_encoding(Tree(updated))
        encoded = {
            int(mask): float(length)
            for mask, length in zip(masks, lengths)
            if float(length) > 1e-8
        }

        self.assertAlmostEqual(encoded[7], 0.37)
        self.assertAlmostEqual(encoded[24], 0.91)
        self.assertAlmostEqual(encoded[3], 0.1)

    def test_boundary_paths_ignore_zero_length_encoder_edges(self):
        start = (
            "((((1:0.5373349269065314,4:0.9710199954281543):0.8468676132930923,"
            "3:0.9264108986066186):0.9689183977257254,2:0.1364359403626998):0.0,0:0.0);"
        )
        target = (
            "((((3:0.18534768712917032,4:0.8194623166729189):0.225771206470011,"
            "1:0.22534633467833737):0.29659845422370856,2:0.9024945938386142):0.0,0:0.0);"
        )

        orig_paths = return_tree_boundary_merge_paths(start, target)
        self.assertEqual(len(orig_paths), 1)
        self.assertEqual(orig_paths[0]["births"], [24])
        self.assertEqual(
            [label["result_split"] for label in orig_paths[0]["events"][0]["labels"]],
            [24],
        )

        entry = orig_paths[0]["start_newick"]
        local_paths = return_tree_boundary_merge_paths(entry, target)
        local_nonempty = _nonempty_boundary_paths(local_paths)

        self.assertEqual(len(local_nonempty), 1)
        self.assertEqual(
            local_nonempty[0]["births"],
            [24],
            "Recomputing from an exact boundary point should not invent extra zero-length splits.",
        )
        self.assertEqual(
            [label["result_split"] for label in local_nonempty[0]["events"][0]["labels"]],
            [24],
        )

    def test_geodesic_suffix_from_boundary_point_is_compositional(self):
        start = (
            "((((2:0.9482052553993453,3:0.7659087172659377):0.9300924969988753,"
            "4:0.12610470545525326):0.815674209009127,1:0.7676082903346565):0.0,0:0.0);"
        )
        target = (
            "(((1:0.20188536818782993,3:0.3219155493578473):0.6840770978232318,"
            "(2:0.5221621430039474,4:0.5893847733123374):0.9108104425755604):0.0,0:0.0);"
        )

        start_dict, n_leaves, mapping = _encode_positive(start)
        target_dict, _, _ = _encode_positive(target)
        original = bhv_geodesic_with_support(start_dict, target_dict, n_leaves)

        entry = {
            int(mask): float(length)
            for mask, length in original["segments"][0]["end_lengths"].items()
            if float(length) > 1e-8
        }
        local = bhv_geodesic_with_support(entry, target_dict, n_leaves)

        original_boundary_newick, original_births = _boundary_end_newick(
            original["segments"][0],
            n_leaves,
            mapping,
        )
        local_boundary_newick, local_births = _boundary_end_newick(
            _first_boundary_segment(local["segments"]),
            n_leaves,
            mapping,
        )

        self.assertEqual(original_births, [20])
        self.assertEqual(
            local_births,
            [20],
            "A geodesic suffix sampled from an exact boundary point should preserve the first local birth set.",
        )
        self.assertEqual(
            _norm_rf(original_boundary_newick, local_boundary_newick),
            0.0,
            "The first local boundary topology should match the original geodesic suffix.",
        )

        original_paths = return_tree_boundary_merge_paths(start, target)
        entry_newick = original_paths[0]["start_newick"]
        local_paths = _nonempty_boundary_paths(
            return_tree_boundary_merge_paths(entry_newick, target)
        )

        self.assertEqual(
            [path["births"] for path in local_paths],
            [path["births"] for path in original_paths],
            "Recomputing from the exact first boundary point should preserve the remaining birth schedule.",
        )
        self.assertEqual(
            [
                [label["result_split"] for label in event["labels"]]
                for path in local_paths
                for event in path["events"]
            ],
            [
                [label["result_split"] for label in event["labels"]]
                for path in original_paths
                for event in path["events"]
            ],
            "Recomputing from the exact first boundary point should preserve remaining merge labels.",
        )

    def test_oracle_helpers_drop_representation_only_zero_length_splits(self):
        start = (
            "((((2:0.9482052553993453,3:0.7659087172659377):0.9300924969988753,"
            "4:0.12610470545525326):0.815674209009127,1:0.7676082903346565):0.0,0:0.0);"
        )
        target = (
            "(((1:0.20188536818782993,3:0.3219155493578473):0.6840770978232318,"
            "(2:0.5221621430039474,4:0.5893847733123374):0.9108104425755604):0.0,0:0.0);"
        )

        expected_start, n_leaves, _ = _encode_positive(start)
        expected_target, _, _ = _encode_positive(target)

        captured_boundary = {}
        original = bhv_geodesic_with_support

        def _capture_boundary(tree1, tree2, n_leaves_arg=None, **kwargs):
            if n_leaves_arg is None:
                n_leaves_arg = kwargs["n_leaves"]
            captured_boundary["tree1"] = dict(tree1)
            captured_boundary["tree2"] = dict(tree2)
            captured_boundary["n_leaves"] = n_leaves_arg
            return original(tree1, tree2, n_leaves_arg)

        with mock.patch("utils.bhv_utils.bhv_geodesic_with_support", side_effect=_capture_boundary):
            return_tree_boundary_merge_paths(start, target)

        self.assertEqual(captured_boundary["n_leaves"], n_leaves)
        self.assertEqual(
            captured_boundary["tree1"],
            expected_start,
            "Boundary-path oracle construction should ignore dummy-root zero-length representation splits.",
        )
        self.assertEqual(captured_boundary["tree2"], expected_target)

        captured_velocity = {}

        def _capture_velocity(tree1, tree2, n_leaves_arg=None, **kwargs):
            if n_leaves_arg is None:
                n_leaves_arg = kwargs["n_leaves"]
            captured_velocity["tree1"] = dict(tree1)
            captured_velocity["tree2"] = dict(tree2)
            captured_velocity["n_leaves"] = n_leaves_arg
            return original(tree1, tree2, n_leaves_arg)

        with mock.patch("utils.bhv_utils.bhv_geodesic_with_support", side_effect=_capture_velocity):
            from utils.bhv_utils import return_sampled_tree_orthant_velocity

            return_sampled_tree_orthant_velocity(start, target, 0.0)

        self.assertEqual(captured_velocity["n_leaves"], n_leaves)
        self.assertEqual(
            captured_velocity["tree1"],
            expected_start,
            "Velocity oracle construction should ignore dummy-root zero-length representation splits.",
        )
        self.assertEqual(captured_velocity["tree2"], expected_target)

    def test_sample_tree_along_geodesic_skips_zero_length_prefix_segments(self):
        geodesic_result = {
            "segments": [
                {
                    "Ai": set(),
                    "Bi": {12},
                    "start_splits": {3},
                    "end_splits": {3},
                    "normA": 0.0,
                    "normB": 1.0,
                    "ratio": 0.0,
                    "lambda_start": 0.0,
                    "lambda_end": 0.0,
                    "start_lengths": {3: 1.0},
                    "end_lengths": {3: 1.0},
                    "length": 0.0,
                    "velocity": {3: 0.0},
                },
                {
                    "Ai": {3},
                    "Bi": {12},
                    "start_splits": {3},
                    "end_splits": set(),
                    "normA": 1.0,
                    "normB": 1.0,
                    "ratio": 1.0,
                    "lambda_start": 0.0,
                    "lambda_end": 1.0,
                    "start_lengths": {3: 1.0},
                    "end_lengths": {3: 0.0},
                    "length": 1.0,
                    "velocity": {3: -1.0},
                },
            ]
        }

        _, _, info = sample_tree_along_geodesic(geodesic_result, n_leaves=4, u=0.0)

        self.assertEqual(
            info["segment_index"],
            0,
            "Zero-length bookkeeping segments should be skipped before selecting the runtime segment.",
        )
        self.assertIn(3, info["active_velocity"])
        self.assertLess(
            info["active_velocity"][3],
            0.0,
            "The right-hand derivative at u=0 should come from the first positive-length segment.",
        )
