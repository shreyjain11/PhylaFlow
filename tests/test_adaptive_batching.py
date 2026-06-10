import sys
import os
import torch
import unittest
from unittest.mock import MagicMock, patch
from run.TrainingModule import TrainingModule
from data.dataset import PhylaDataModule
from data.dataset import TreeDataset
from ete3 import Tree


class TestAdaptiveBatching(unittest.TestCase):
    def test_oom_handling_trigger(self):
        # Mock Config
        config = {
            "model": {
                "num_node_types": 3,
                "num_edge_types": 2,
                "hidden_dim": 16,
                "embed_dim": 16,
                "output_dim": 1,
                "n_layers": 2,
                "n_heads": 2,
                "dropout": 0.1,
                "attention_dropout": 0.1,
                "activation_dropout": 0.1,
                "drop_path_rate": 0.0,
                "use_performer": False,
                "performer_nb_features": None,
                "performer_generalized_attention": False,
                "layernorm_style": "prenorm",
                "tokenizer_lap_dim": 4,
                "tokenizer_lap_dropout": 0.1,
                "tokenizer_n_layers": 2,
                "phyla_dim": 16,
            },
            "trainer": {
                "lr": 1e-4,
                "record": False,
                "epochs": 1,
                "val_callback_frequency": 0,
                "checkpoint_dir": "./mock_ckpt",
            },
            "data": {
                "nexus_root": "mock/nexus",
                "mrbayes_root": "mock/runs",
                "batch_size": 1,
                "num_workers": 0,
                "pin_memory": False,
            },
        }

        # Mock Dataset
        with patch("data.dataset.TreeDataset") as MockTreeDataset:
            # Setup mock behavior
            mock_dataset_instance = MockTreeDataset.return_value

            # Create DataModule
            dm = PhylaDataModule(config, train_ids=[], test_ids=[])
            # Inject the mocked dataset_train
            dm.dataset_train = mock_dataset_instance

            # Setup return values for proxied attributes
            mock_dataset_instance.chosen_tree = (0, 100, 1)
            mock_dataset_instance.size_detector = MagicMock()
            mock_dataset_instance.return_max_length.return_value = 50
            mock_dataset_instance.name_to_seq = {"a": "seq"}

            # Mock __getitem__ and collate_fn
            mock_dataset_instance.__getitem__.return_value = {
                "id": "test",
                "newick_tree": "(A,B);",
                "sequences": {"A": "ATCG"},
                "velocity": torch.tensor([0]),
                "timepoint": 0.5,
                "velocity": torch.tensor([1]),
            }

            dm.tree_tokenizer = MagicMock()
            dm.tree_tokenizer.return_value = "tokenized"

            # Mock TrainingModule
            model = TrainingModule(
                model=MagicMock(),
                dataset=dm,
                lr=0.001,
                record=False,
                deepspeed=False,
                logger=MagicMock(),
            )

            # Mock optimizers
            opt_mock = MagicMock()
            # structure: opt.optimizer.param_groups[0]["lr"]
            optimizer_mock = MagicMock()
            optimizer_mock.param_groups = [{"lr": 0.001}]
            opt_mock.optimizer = optimizer_mock
            model.optimizers = MagicMock(return_value=opt_mock)
            model.trainer = MagicMock()
            model.trainer.gradient_clip_val = None
            model.trainer.gradient_clip_algorithm = "norm"

            # Mock step to raise OOM
            model.step = MagicMock(
                side_effect=[
                    RuntimeError("CUDA out of memory"),
                    RuntimeError("CUDA out of memory"),
                    {"loss": torch.tensor(0.5, requires_grad=True)},
                ]
            )

            # Mock manual_backward to not actually backward
            model.manual_backward = MagicMock()

            # We expect this to fail initially because of missing attributes on dm
            print("Running training_step with expected failure...")
            model.training_step({}, 0)

    @patch("data.dataset.TreeDataset.build_index")
    @patch("data.dataset.TreeDataset.parse_nexus")
    @patch("data.dataset.TreeDataset.load_posterior_trees_from_tfiles")
    def test_tree_pruning_visualization(
        self, mock_load_trees, mock_parse_nexus, mock_build_index
    ):
        """
        Verify that pruning works by visualizing trees before and after.
        """
        print("\n=== Testing Tree Pruning Logic ===")

        # Create a real TreeDataset (with mocked IO)
        # build_index is mocked so __init__ won't fail or scan files
        dataset = TreeDataset(nexus_root="mock", mrbayes_root="mock")

        # Setup mock data
        # A tree with 8 leaves: 1-8
        # Using integer-like strings because utils.random_tree.Tree expects leaf names to be castable to int
        original_newick = "((((1:0.1,2:0.1):0.2,(3:0.1,4:0.1):0.2):0.3,((5:0.1,6:0.1):0.2,(7:0.1,8:0.1):0.2):0.3):0.4,9:0.5);"

        mock_load_trees.return_value = [original_newick]
        mock_parse_nexus.return_value = (
            {
                "1": "seq",
                "2": "seq",
                "3": "seq",
                "4": "seq",
                "5": "seq",
                "6": "seq",
                "7": "seq",
                "8": "seq",
                "9": "seq",
            },
            ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
        )

        # Setup internal index (usually done by build_index)
        dataset._index = [
            {
                "id": "test_id",
                "nexus_path": "mock/test.nex",
                "tree_paths": ["mock/test.t"],
            }
        ]

        print("Original Tree:")
        t_orig = Tree(original_newick, format=1)
        print(t_orig.get_ascii(show_internal=True))

        # Request a pruned version (e.g., 4 leaves)
        preset_size = 4
        print(f"\nRequesting pruned tree with size {preset_size}...")

        # Call __getitem__. The pruning happens inside.
        sample = dataset.__getitem__(0, preset_subtree_size=preset_size)

        pruned_newick = sample["newick_tree"]

        print(f"Pruned Tree (Newick: {pruned_newick}):")
        t_pruned = Tree(pruned_newick, format=1)
        print(t_pruned.get_ascii(show_internal=True))

        # Verification
        self.assertEqual(len(t_pruned.get_leaves()), preset_size)
        print(f"\nSuccessfully pruned to {len(t_pruned.get_leaves())} leaves.")

    @patch("data.dataset.TreeDataset.build_index")
    @patch("data.dataset.TreeDataset.parse_nexus")
    @patch("data.dataset.TreeDataset.load_posterior_trees_from_tfiles")
    @patch("data.dataset.TreeDataset.parse_translate_block")
    def test_tree_pruning_indexing(
        self, mock_translate, mock_load_trees, mock_parse_nexus, mock_build_index
    ):
        print("\n=== Testing Tree Pruning and Renaming Logic ===")
        dataset = TreeDataset(nexus_root="mock", mrbayes_root="mock")

        # Original tree with leaves "1", "2", "3"
        original_newick = "((1:0.1,2:0.1):0.2,3:0.2);"
        mock_load_trees.return_value = [original_newick]

        # Nexus seqs
        mock_parse_nexus.return_value = (
            {"TaxonA": "AAAA", "TaxonB": "BBBB", "TaxonC": "CCCC"},
            ["TaxonA", "TaxonB", "TaxonC"],
        )

        # Translation map
        mock_translate.return_value = {"1": "TaxonA", "2": "TaxonB", "3": "TaxonC"}

        # Mock index
        dataset._index = [
            {"id": "test", "nexus_path": "path.nex", "tree_paths": ["path.t"]}
        ]

        # Get Item with Pruning of size 3 (min size for random gen)
        # Should pick 3 random leaves.
        # Should rename them to "0", "1", "2".
        # Should return sequences subset.

        sample = dataset.__getitem__(0, preset_subtree_size=3)

        print(f"Sample Sequences Keys: {sample['sequences'].keys()}")
        print(f"Sample Newick: {sample['newick_tree']}")

        # Check keys are "0", "1", "2"
        self.assertEqual(sorted(list(sample["sequences"].keys())), ["0", "1", "2"])

        # Check values correspond
        # We don't know which original leaf maps to "0", but we can check values are from {AAAA, BBBB, CCCC}
        valid_seqs = {"AAAA", "BBBB", "CCCC"}
        for k, v in sample["sequences"].items():
            self.assertIn(v, valid_seqs)

        print("Test Pruning Renaming Passed")
