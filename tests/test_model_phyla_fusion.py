import unittest

import torch

from model.model import TreeDenoiserTokenGT, return_model


class TestModelPhylaFusion(unittest.TestCase):
    def test_return_model_wires_phyla_config(self):
        config = {
            "model": {
                "num_node_types": 3,
                "num_edge_types": 2,
                "embed_dim": 32,
                "output_dim": 1,
                "n_layers": 2,
                "n_heads": 4,
                "dropout": 0.0,
                "attention_dropout": 0.0,
                "activation_dropout": 0.0,
                "drop_path_rate": 0.0,
                "use_performer": False,
                "performer_nb_features": 64,
                "performer_generalized_attention": False,
                "layernorm_style": "prenorm",
                "tokenizer_lap_dim": 4,
                "tokenizer_lap_dropout": 0.0,
                "tokenizer_n_layers": 1,
                "phyla_dim": 6,
                "phyla_use_leaf_tokens": True,
                "phyla_use_split_tokens": False,
                "phyla_leaf_scale": 1.5,
                "phyla_split_scale": 0.25,
                "phyla_use_global_context": True,
                "phyla_global_context_scale": 0.5,
                "phyla_use_clade_context": True,
                "phyla_clade_context_scale": 0.75,
            }
        }
        model = return_model(config)
        self.assertTrue(model.phyla_use_leaf_tokens)
        self.assertFalse(model.phyla_use_split_tokens)
        self.assertTrue(model.phyla_use_global_context)
        self.assertTrue(model.phyla_use_clade_context)
        self.assertEqual(model.phyla_leaf_scale, 1.5)
        self.assertEqual(model.phyla_split_scale, 0.25)
        self.assertEqual(model.phyla_global_context_scale, 0.5)
        self.assertEqual(model.phyla_clade_context_scale, 0.75)
        self.assertEqual(model.phyla_proj.in_features, 6)
        self.assertIsNotNone(model.phyla_global_proj)
        self.assertIsNotNone(model.phyla_clade_proj)
        self.assertIsNotNone(model.first_hit_phyla_global_proj)
        self.assertIsNotNone(model.autoregressive_phyla_global_proj)

    def test_leaf_phyla_additions_only_touch_leaf_positions(self):
        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=8,
            n_layers=2,
            n_heads=2,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            phyla_dim=4,
        )
        phyla_proj_full = torch.tensor(
            [
                [
                    [1.0] * 8,
                    [2.0] * 8,
                    [3.0] * 8,
                ]
            ]
        )
        leaf_idx_list = [torch.tensor([0, 2, 5], dtype=torch.long)]
        additions = model._compute_leaf_phyla_token_additions(
            phyla_proj_full=phyla_proj_full,
            leaf_idx_list=leaf_idx_list,
            num_tokens=7,
            device=phyla_proj_full.device,
            dtype=phyla_proj_full.dtype,
        )

        self.assertEqual(tuple(additions.shape), (1, 7, 8))
        self.assertTrue(torch.allclose(additions[0, 0], torch.ones(8)))
        self.assertTrue(torch.allclose(additions[0, 2], torch.full((8,), 2.0)))
        self.assertTrue(torch.allclose(additions[0, 5], torch.full((8,), 3.0)))
        self.assertTrue(torch.allclose(additions[0, 1], torch.zeros(8)))
        self.assertTrue(torch.allclose(additions[0, 3], torch.zeros(8)))

    def test_global_phyla_context_pools_leaf_embeddings(self):
        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=8,
            n_layers=2,
            n_heads=2,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            phyla_dim=4,
            phyla_use_global_context=True,
        )
        phyla_embeddings = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [99.0, 99.0, 99.0, 99.0],
                ]
            ]
        )
        context = model._compute_global_phyla_context(
            phyla_embeddings=phyla_embeddings,
            leaf_idx_list=[torch.tensor([0, 2, 5], dtype=torch.long)],
            device=phyla_embeddings.device,
            dtype=torch.float32,
        )

        self.assertEqual(tuple(context.shape), (1, 8))
        self.assertTrue(torch.isfinite(context).all())

    def test_clade_phyla_context_touch_edge_positions(self):
        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=8,
            n_layers=2,
            n_heads=2,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            phyla_dim=4,
            phyla_use_clade_context=True,
        )
        phyla_embeddings = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ]
        )
        leaf_idx_list = [torch.tensor([2, 3, 4], dtype=torch.long)]
        edge_mask = torch.tensor([[False, True, False, False, True, False]])
        edge_split_masks = [[0b001, 0b011]]

        context = model._compute_clade_phyla_token_context(
            phyla_embeddings=phyla_embeddings,
            leaf_idx_list=leaf_idx_list,
            edge_mask=edge_mask,
            edge_split_masks=edge_split_masks,
            num_tokens=6,
            device=phyla_embeddings.device,
            dtype=torch.float32,
        )

        self.assertEqual(tuple(context.shape), (1, 6, 8))
        self.assertFalse(torch.allclose(context[0, 1], torch.zeros(8)))
        self.assertFalse(torch.allclose(context[0, 4], torch.zeros(8)))
        self.assertTrue(torch.allclose(context[0, 0], torch.zeros(8)))
        self.assertTrue(torch.allclose(context[0, 2], torch.zeros(8)))
        self.assertTrue(torch.isfinite(context).all())

    def test_split_phyla_additions_touch_edge_positions(self):
        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=8,
            n_layers=2,
            n_heads=2,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            phyla_dim=4,
        )
        phyla_embeddings = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ]
        )
        leaf_idx_list = [torch.tensor([2, 3, 4], dtype=torch.long)]
        edge_mask = torch.tensor([[False, True, False, False, True, False]])
        edge_split_masks = [[0b001, 0b011]]

        additions = model._compute_split_phyla_token_additions(
            phyla_embeddings=phyla_embeddings,
            leaf_idx_list=leaf_idx_list,
            edge_mask=edge_mask,
            edge_split_masks=edge_split_masks,
            num_tokens=6,
            device=phyla_embeddings.device,
            dtype=torch.float32,
        )

        self.assertEqual(tuple(additions.shape), (1, 6, 8))
        self.assertFalse(torch.allclose(additions[0, 1], torch.zeros(8)))
        self.assertFalse(torch.allclose(additions[0, 4], torch.zeros(8)))
        self.assertTrue(torch.allclose(additions[0, 0], torch.zeros(8)))
        self.assertTrue(torch.allclose(additions[0, 2], torch.zeros(8)))

    def test_split_identity_cache_survives_inference_then_training(self):
        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=8,
            n_layers=2,
            n_heads=2,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            phyla_dim=4,
        )

        with torch.inference_mode():
            inference_out = model.create_split_identity_embedding([1, 3, 7], "cpu")
        self.assertEqual(tuple(inference_out.shape), (3, 8))

        training_out = model.create_split_identity_embedding([1, 3, 7], "cpu")
        self.assertTrue(training_out.requires_grad)
        training_out.sum().backward()
        self.assertIsNotNone(model.split_mask_proj[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
