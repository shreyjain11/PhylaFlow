import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from model.model import TreeDenoiserTokenGT


def _install_lightning_stub():
    pl_mod = types.ModuleType("pytorch_lightning")
    utilities_mod = types.ModuleType("pytorch_lightning.utilities")

    class LightningModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        @property
        def device(self):
            return torch.device("cpu")

    class LightningDataModule:
        def __init__(self, *args, **kwargs):
            pass

    def grad_norm(*args, **kwargs):
        return {}

    pl_mod.LightningModule = LightningModule
    pl_mod.LightningDataModule = LightningDataModule
    utilities_mod.grad_norm = grad_norm

    sys.modules["pytorch_lightning"] = pl_mod
    sys.modules["pytorch_lightning.utilities"] = utilities_mod


def _install_matplotlib_stub():
    matplotlib_mod = types.ModuleType("matplotlib")
    pyplot_mod = types.ModuleType("matplotlib.pyplot")
    animation_mod = types.ModuleType("matplotlib.animation")

    def _noop(*args, **kwargs):
        return None

    pyplot_mod.figure = _noop
    pyplot_mod.plot = _noop
    pyplot_mod.savefig = _noop
    pyplot_mod.close = _noop

    matplotlib_mod.pyplot = pyplot_mod
    matplotlib_mod.animation = animation_mod
    sys.modules["matplotlib"] = matplotlib_mod
    sys.modules["matplotlib.pyplot"] = pyplot_mod
    sys.modules["matplotlib.animation"] = animation_mod


_install_lightning_stub()
_install_matplotlib_stub()

from run.TrainingModule import TrainingModule


class TestTrainingModulePrecomputedPhyla(unittest.TestCase):
    def _build_model(self):
        return TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=16,
            output_dim=1,
            n_layers=2,
            n_heads=2,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            tokenizer_lap_dim=4,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=1,
            phyla_dim=4,
        )

    def _build_dataset(self):
        split = SimpleNamespace(overfit_split_multi_subset_events=False)
        return SimpleNamespace(
            dataset_train=split,
            dataset_val=split,
            name_to_seq={},
        )

    def test_precomputed_lookup_respects_requested_taxon_order(self):
        payload = {
            "sequence_names": ["tax_a", "tax_b", "tax_c"],
            "embeddings": torch.tensor(
                [
                    [[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0]]
                ],
                dtype=torch.float32,
            ),
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
            path = handle.name
        try:
            torch.save(payload, path)
            module = TrainingModule(
                model=self._build_model(),
                dataset=self._build_dataset(),
                phyla_precomputed_embeddings_path=path,
            )

            looked_up = module._lookup_precomputed_phyla_embeddings(
                ["tax_c", "tax_a"],
                device="cpu",
            )
            expected = torch.tensor(
                [
                    [0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
            )
            self.assertTrue(torch.allclose(looked_up, expected))
        finally:
            os.remove(path)

    def test_harness_sample_kwargs_use_precomputed_embeddings(self):
        payload = {
            "sequence_names": ["tax_a", "tax_b", "tax_c"],
            "embeddings": torch.tensor(
                [
                    [[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0]]
                ],
                dtype=torch.float32,
            ),
        }

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
            path = handle.name
        try:
            torch.save(payload, path)
            module = TrainingModule(
                model=self._build_model(),
                dataset=self._build_dataset(),
                phyla_precomputed_embeddings_path=path,
            )
            pair = {
                "start_tree": "((0:0.1,1:0.1):0.1,2:0.1);",
                "target_tree": "((0:0.1,1:0.1):0.1,2:0.1);",
                "n_leaves": 3,
                "max_events": 5,
                "name_mapping": {0: "tax_a", 1: "tax_b", 2: "tax_c"},
            }

            sample_kwargs = module._build_harness_sample_kwargs(pair, train=True)
            phyla_embeddings = sample_kwargs["phyla_embeddings"].cpu()

            self.assertEqual(tuple(phyla_embeddings.shape), (1, 3, 4))
            self.assertTrue(
                torch.allclose(
                    phyla_embeddings[0],
                    torch.tensor(
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                        ],
                        dtype=torch.float32,
                    ),
                )
            )
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
