"""
Test that TrainingModule.step(autoregressive=True) does not crash with
'Did not find merge for split' when boundary-decision labels use
exact structural component groups.

Exercises the real TrainingModule.step() code path end-to-end so the
structural-group alignment is validated against the actual training loop.
"""

import random
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch


# ── stubs for optional heavy deps ──
def _install_deepspeed_stub():
    if "deepspeed" in sys.modules:
        return
    ds_mod = types.ModuleType("deepspeed")
    ds_ops = types.ModuleType("deepspeed.ops")
    ds_adam = types.ModuleType("deepspeed.ops.adam")

    class FusedAdam(torch.optim.Adam):
        pass

    ds_adam.FusedAdam = FusedAdam
    ds_ops.adam = ds_adam
    ds_mod.ops = ds_ops
    sys.modules["deepspeed"] = ds_mod
    sys.modules["deepspeed.ops"] = ds_ops
    sys.modules["deepspeed.ops.adam"] = ds_adam


def _install_phyla_stub():
    phyla = types.ModuleType("phyla")
    pu = types.ModuleType("phyla.utils")
    puu = types.ModuleType("phyla.utils.utils")
    pe = types.ModuleType("phyla.eval")
    per = types.ModuleType("phyla.eval.evo_reasoning_eval")

    class Config:
        def __init__(self):
            self.trainer = types.SimpleNamespace(checkpoint_path="")
            self.eval = types.SimpleNamespace(device="cpu")

    puu.load_config = lambda c: c()
    per.Config = Config
    per.load_model = lambda config=None, random_model=False: {"model": MagicMock()}
    per._encode_sequences_openfold_style = lambda s, n: (
        {
            "encoded_sequences": torch.zeros((len(s), 1), dtype=torch.long),
            "sequence_mask": torch.ones((len(s), 1), dtype=torch.long),
            "cls_positions": torch.ones((len(s), 1), dtype=torch.bool),
        },
        None,
    )

    phyla.utils = pu
    phyla.eval = pe
    pu.utils = puu
    pe.evo_reasoning_eval = per
    for name, mod in [
        ("phyla", phyla),
        ("phyla.utils", pu),
        ("phyla.utils.utils", puu),
        ("phyla.eval", pe),
        ("phyla.eval.evo_reasoning_eval", per),
    ]:
        sys.modules[name] = mod


try:
    from deepspeed.ops.adam import FusedAdam as _F  # noqa: F401
except Exception:
    _install_deepspeed_stub()

try:
    from phyla.utils.utils import load_config as _LC  # noqa: F401
except Exception:
    _install_phyla_stub()

from data.dataset import TreeDataset
from model.model import TreeDenoiserTokenGT
from run.TrainingModule import TrainingModule
from utils.bhv_utils import return_sampled_tree_boundary_decisions


class TestAutoRegressiveSplitAlignment(unittest.TestCase):
    """Run TrainingModule.step(autoregressive=True) on the sanity-check
    tree pair and verify it completes without 'Did not find merge'."""

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_step_autoregressive_does_not_crash(self, _mock):
        """End-to-end: build the exact batch the sanity config produces
        and call step(autoregressive=True).  Before the structural-group fix
        this raised 'Did not find merge for split …'."""
        random.seed(42)
        torch.manual_seed(42)
        device = torch.device("cpu")

        # ── 1. Reproduce the dataset's sanity-check tree pair ──
        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_velocity_zero=True,
        )
        real_tree = ds.load_posterior_trees_from_tfiles([])[0]
        random_tree = ds.sample_random_tree(real_tree)

        for i in range(20):
            print(f"Testing autoregressive step with boundary decision sample {i+1}/20...")
            boundary_labels = return_sampled_tree_boundary_decisions(
                random_tree, real_tree
            )
            self.assertTrue(
                len(boundary_labels) > 0,
                "No boundary decisions produced; cannot test.",
            )

            # chosen = boundary_labels[0]
            for chosen in boundary_labels:
                autoregressive_newick = chosen["newick"]
                autoregressive_labels = chosen["labels"]

                # ── 2. Build model + TrainingModule ──
                model = TreeDenoiserTokenGT(
                    num_node_types=3,
                    num_edge_types=2,
                    embed_dim=32,
                    n_layers=1,
                    n_heads=4,
                    output_dim=1,
                    dropout=0.0,
                    attention_dropout=0.0,
                    activation_dropout=0.0,
                    drop_path_rate=0.0,
                    use_performer=False,
                    performer_nb_features=None,
                    performer_generalized_attention=False,
                    layernorm_style="prenorm",
                    tokenizer_lap_dim=4,
                    tokenizer_lap_dropout=0.0,
                    tokenizer_n_layers=1,
                    phyla_dim=16,
                ).to(device)

                module = TrainingModule(
                    model=model,
                    dataset=MagicMock(),
                    lr=1e-3,
                    record=False,
                    epochs=1,
                    deepspeed=False,
                    logger=None,
                ).to(device)

                # ── 3. Construct the batch exactly as collate_fn does ──
                with torch.no_grad():
                    tokenized_ar = model.tokenizer([autoregressive_newick])

                batch = {
                    "tokenized_autoregressive_trees": tokenized_ar,
                    "newick_autoregressive_trees": [autoregressive_newick],
                    "batched_autoregressive_time": torch.tensor([0.0]),
                    "batched_autoregressive_labels": [autoregressive_labels],
                    "phyla_embeddings": None,
                }

                # ── 4. The actual test: step must not raise ──
                try:
                    logs = module.step(batch, autoregressive=True)
                except Exception as e:
                    if "Did not find merge" in str(e):
                        self.fail(
                            f"step(autoregressive=True) raised split-mismatch error "
                            f"that the structural-group fix should prevent: {e}"
                        )
                    raise  # re-raise unexpected errors

                self.assertIn("loss", logs)
                self.assertTrue(
                    torch.isfinite(logs["loss"]),
                    "Autoregressive loss is not finite.",
                )


if __name__ == "__main__":
    unittest.main()
