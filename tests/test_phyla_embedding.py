import unittest
import torch
import os
import sys
import shutil

# Ensure workspace root is in path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(workspace_root)

from unittest.mock import MagicMock, patch
from run.TrainingModule import TrainingModule
from utils.random_tree import Tree


class MockDataset:
    def __init__(self):
        self.name_to_seq = {"seq1": "ATCG", "seq2": "ATTG", "seq3": "ATCC"}


class TestPhylaEmbedding(unittest.TestCase):
    def setUp(self):
        # Create a dummy config if not exists
        if not os.path.exists("configs"):
            os.makedirs("configs")
        if not os.path.exists("configs/sample_eval_config.yaml"):
            with open("configs/sample_eval_config.yaml", "w") as f:
                f.write("model: {}\ntrainer: {}\ndataset: {}\neval: {}")

    def test_init_with_phyla(self):
        # Mock load_config and load_model to avoid actually loading large models/files
        with patch("run.TrainingModule.load_config") as mock_lc, patch(
            "run.TrainingModule.load_model"
        ) as mock_lm:

            mock_config = MagicMock()
            mock_config.trainer.checkpoint_path = ""
            mock_config.eval.device = "cpu"
            mock_lc.return_value = mock_config

            mock_model = MagicMock()
            mock_lm.return_value = {"model": mock_model}

            module = TrainingModule(
                model=MagicMock(),
                phyla_checkpoint_path="dummy_ckpt.ckpt",
                phyla_config_path="configs/sample_eval_config.yaml",
            )

            self.assertIsNotNone(module.phyla_model)
            self.assertEqual(module.phyla_model, mock_model)

    def test_compute_embeddings(self):
        # Setup module with mocked phyla model
        module = TrainingModule(model=MagicMock())
        module.phyla_model = MagicMock()
        module.phyla_model.return_value = torch.randn(
            2, 5, 256
        )  # (batch, seq_len, dim)

        # Test helper
        with patch("run.TrainingModule._encode_sequences_openfold_style") as mock_enc:
            mock_enc.return_value = (
                {
                    "encoded_sequences": torch.zeros(2, 5),
                    "sequence_mask": torch.ones(2, 5),
                    "cls_positions": torch.zeros(2, 5),
                },
                None,
            )

            seqs = ["ATCG", "ATCG"]
            names = ["1", "2"]
            emb = module.compute_phyla_embeddings(seqs, names, device="cpu")

            self.assertEqual(emb.shape, (2, 5, 256))

    def test_sample_integration(self):
        # Test that sample calls compute_phyla_embeddings when phyla_embeddings is None

        module = TrainingModule(model=MagicMock())
        module.phyla_model = MagicMock()
        # Mocking phyla model output: (Batch, SeqLen, Dim) -> (1, 3, 10) for 1 sequence?
        # Wait, sample sends sorted sequences.
        # Let's say we have 1 tree with 3 leaves.

        module.dataset = MockDataset()
        module.model = MagicMock()
        module.model.tokenizer.return_value = ["mock_tokenized"]
        module.model.return_value = (
            torch.zeros(1, 1, 1),
            [0],
            [0],
        )  # velocity, edge_splits, edge_split_mask
        # Mock forward return signatures match usage in sample

        # Mock compute_phyla_embeddings
        # Make mock return consistent with real behavior (1, N_leaves, Dim)
        module.compute_phyla_embeddings = MagicMock(return_value=torch.randn(1, 3, 16))
        # Returns (1, N_leaves, Dim)

        newick_tree = "((seq1:0.1,seq2:0.1):0.1,seq3:0.1);"
        with patch("run.TrainingModule.BHVEncoder") as MockEncoder:
            # Setup mock encoder instance
            mock_enc_instance = MockEncoder.return_value
            # Return dummy masks and lengths (list of masks, list of lengths)
            mock_enc_instance.return_BHV_encoding.return_value = (
                [1, 2, 4],
                [0.1, 0.1, 0.1],
            )

            with patch("run.TrainingModule.Tree", side_effect=lambda nw: Tree(nw)):
                # We need to mock build_tree_from_splits because it's called inside sample loop

                # Breaking out of while loop immediately by max_steps=0 or T=0?
                # sample calls:
                #   trees = []
                #   for nw: Tree(nw)...
                #
                #   if phyla_embeddings is None: ... compute ...

                # Let's check logic extraction
                res = module.sample(
                    [newick_tree], phyla_embeddings=None, max_steps=0, T=0
                )

                module.compute_phyla_embeddings.assert_called_once()
                args, _ = module.compute_phyla_embeddings.call_args
                # Filtered names should optionally exclude ROOT_DUMMY
                # Tree implementation adds ROOT_DUMMY.
                # MockDataset has seq1, seq2, seq3.
                # The tree has seq1, seq2, seq3 and ROOT_DUMMY
                # Valid names = 3.
                self.assertEqual(len(args[0]), 3)  # 3 seqs
                self.assertEqual(len(args[1]), 3)  # 3 names

                pass

    def test_real_checkpoint_loading(self):
        ckpt_path = "/home/leo/PhylaFlow/checkpoints/chkpts/epoch%3D00-step%3D004000-treefam_avg_normrf%3D0.6721.ckpt"
        config_path = "configs/sample_eval_config.yaml"

        if not os.path.exists(ckpt_path):
            self.skipTest(f"Checkpoint not found at {ckpt_path}")

        print(f"Testing real checkpoint loading from {ckpt_path}")

        # We need to ensure Config is importable by the module setup, effectively it is.
        # But we pass the path.

        # Note: TrainingModule init takes 'model'. We can pass a mock for the main model.
        # But phyla loaded model is a real model.

        module = TrainingModule(
            model=MagicMock(),
            phyla_checkpoint_path=ckpt_path,
            phyla_config_path=config_path,
            verbose=True,
        )

        self.assertIsNotNone(module.phyla_model)

        # Test embedding generation with real model
        seqs = [
            "ATGCGTACGTAGCTAGCTAGCTAGCTGATCGATCGATCGTAGC",
            "ATGCGTACGTAGCTAGCTAGCTAGCTGATCGATCGATCGTAGC",
        ]
        names = ["seq1", "seq2"]

        print("Computing embeddings with real model...")
        embeddings = module.compute_phyla_embeddings(
            seqs, names, device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Embeddings shape: {embeddings.shape}")

        # Based on observed output, shape is (1, N_seqs, Dim)
        # We passed 2 sequences.
        self.assertEqual(embeddings.shape[0], 1)
        self.assertEqual(embeddings.shape[1], 2)
        # Check hidden dim is 256 based on config d_model
        self.assertEqual(embeddings.shape[2], 256)


if __name__ == "__main__":
    unittest.main()
