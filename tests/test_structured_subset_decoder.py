import unittest

import torch

from model.model import StructuredSubsetMergeHead
from run.TrainingModule import (
    _best_autoregressive_fallback_candidate,
    _plan_autoregressive_boundary_merges,
    _structured_subset_loss_and_prediction,
)


class StructuredSubsetDecoderTests(unittest.TestCase):
    def test_head_returns_expected_shapes(self):
        head = StructuredSubsetMergeHead(
            d_model=8,
            hidden=16,
            dropout=0.0,
            max_subset_size=8,
        )
        embeddings = torch.randn(4, 8)
        outputs = head(embeddings)

        self.assertEqual(outputs["starter_pair_logits"].shape, (6,))
        self.assertEqual(outputs["member_logits"].shape, (6, 4))
        self.assertEqual(len(outputs["starter_pair_indices"]), 6)
        self.assertEqual(outputs["logits"].shape, (4, 4))
        self.assertEqual(outputs["subset_size_logits"].shape, (9,))

    def test_planner_decodes_non_empty_structured_subset(self):
        pair_indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        pair_logits = torch.tensor([5.0, -2.0, -3.0, -1.0, -2.0, -4.0])
        member_logits = torch.full((6, 4), -5.0)
        member_logits[0, 0] = 5.0
        member_logits[0, 1] = 5.0
        member_logits[0, 2] = 3.0
        subset_size_logits = torch.tensor([-4.0, -4.0, -3.0, 4.0, -2.0])
        group_output = {
            "decoder_mode": "structured_subset",
            "polytomy_pred": torch.tensor(4.0),
            "splits_represented": [1, 2, 4, 8],
            "starter_pair_logits": pair_logits,
            "starter_pair_indices": pair_indices,
            "member_logits": member_logits,
            "subset_size_logits": subset_size_logits,
            "logits": torch.full((4, 4), float("-inf")),
        }

        planned = _plan_autoregressive_boundary_merges([group_output], existing_splits={1, 2, 4, 8})
        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0]["subsets"][0][0], (1, 2, 4))
        self.assertEqual(planned[0]["subsets"][0][1], 7)

    def test_structured_subset_loss_is_finite(self):
        pair_indices = [(0, 1), (0, 2), (1, 2)]
        group_output = {
            "starter_pair_logits": torch.tensor([3.0, -1.0, 0.5]),
            "starter_pair_indices": pair_indices,
            "member_logits": torch.tensor(
                [
                    [4.0, 4.0, -3.0],
                    [4.0, -3.0, 4.0],
                    [-3.0, 4.0, 4.0],
                ]
            ),
            "subset_size_logits": torch.tensor([-3.0, -4.0, 5.0, 1.0]),
        }

        result = _structured_subset_loss_and_prediction(
            group_output,
            group_splits=(1, 2, 4),
            subset=(1, 2),
        )
        self.assertIsNotNone(result)
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertEqual(result["predicted_subset"], (1, 2))
        self.assertEqual(result["predicted_size"], 2)

    def test_size_head_blocks_extra_members_when_predicting_pair(self):
        pair_indices = [(0, 1), (0, 2), (1, 2)]
        group_output = {
            "decoder_mode": "structured_subset",
            "polytomy_pred": torch.tensor(3.0),
            "splits_represented": [1, 2, 4],
            "starter_pair_logits": torch.tensor([4.0, -1.0, -2.0]),
            "starter_pair_indices": pair_indices,
            "member_logits": torch.tensor(
                [
                    [4.0, 4.0, 10.0],
                    [4.0, -3.0, 4.0],
                    [-3.0, 4.0, 4.0],
                ]
            ),
            "subset_size_logits": torch.tensor([-2.0, -5.0, 20.0, 1.0]),
            "logits": torch.full((3, 3), float("-inf")),
        }

        planned = _plan_autoregressive_boundary_merges(
            [group_output],
            existing_splits={1, 2, 4},
        )
        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0]["subsets"][0][0], (1, 2))

    def test_structured_fallback_uses_size_head_without_crashing(self):
        pair_indices = [(0, 1), (0, 2), (1, 2)]
        group_output = {
            "decoder_mode": "structured_subset",
            "polytomy_pred": torch.tensor(1.0),
            "splits_represented": [1, 2, 4],
            "starter_pair_logits": torch.tensor([4.0, -1.0, -2.0]),
            "starter_pair_indices": pair_indices,
            "member_logits": torch.tensor(
                [
                    [4.0, 4.0, 10.0],
                    [4.0, -3.0, 4.0],
                    [-3.0, 4.0, 4.0],
                ]
            ),
            "subset_size_logits": torch.tensor([-2.0, -5.0, 20.0, 1.0]),
            "logits": torch.full((3, 3), float("-inf")),
        }

        fallback = _best_autoregressive_fallback_candidate(
            [group_output],
            existing_splits={1, 2, 4},
            planned_new_splits=set(),
        )
        self.assertIsNotNone(fallback)
        self.assertEqual(fallback["subsets"][0][0], (1, 2))


if __name__ == "__main__":
    unittest.main()
