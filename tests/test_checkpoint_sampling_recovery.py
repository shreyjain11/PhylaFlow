import copy
import glob
import os
import random
import re
import sys
import types
import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import yaml
from ete3 import Tree as EteTree


def _install_phyla_stub():
    phyla_mod = types.ModuleType("phyla")
    phyla_utils_mod = types.ModuleType("phyla.utils")
    phyla_utils_utils_mod = types.ModuleType("phyla.utils.utils")
    phyla_eval_mod = types.ModuleType("phyla.eval")
    phyla_eval_evo_mod = types.ModuleType("phyla.eval.evo_reasoning_eval")

    class Config:
        def __init__(self):
            self.trainer = types.SimpleNamespace(checkpoint_path="")
            self.eval = types.SimpleNamespace(device="cpu")

    def load_config(config_cls):
        return config_cls()

    def load_model(config=None, random_model=False):
        return {"model": MagicMock()}

    def _encode_sequences_openfold_style(sequences, names):
        batch = {
            "encoded_sequences": torch.zeros((len(sequences), 1), dtype=torch.long),
            "sequence_mask": torch.ones((len(sequences), 1), dtype=torch.long),
            "cls_positions": torch.ones((len(sequences), 1), dtype=torch.bool),
        }
        return batch, None

    phyla_utils_utils_mod.load_config = load_config
    phyla_eval_evo_mod.Config = Config
    phyla_eval_evo_mod.load_model = load_model
    phyla_eval_evo_mod._encode_sequences_openfold_style = (
        _encode_sequences_openfold_style
    )

    phyla_mod.utils = phyla_utils_mod
    phyla_mod.eval = phyla_eval_mod
    phyla_utils_mod.utils = phyla_utils_utils_mod
    phyla_eval_mod.evo_reasoning_eval = phyla_eval_evo_mod

    sys.modules["phyla"] = phyla_mod
    sys.modules["phyla.utils"] = phyla_utils_mod
    sys.modules["phyla.utils.utils"] = phyla_utils_utils_mod
    sys.modules["phyla.eval"] = phyla_eval_mod
    sys.modules["phyla.eval.evo_reasoning_eval"] = phyla_eval_evo_mod


def _install_deepspeed_stub():
    if "deepspeed" in sys.modules:
        return

    ds_mod = types.ModuleType("deepspeed")
    ds_ops_mod = types.ModuleType("deepspeed.ops")
    ds_adam_mod = types.ModuleType("deepspeed.ops.adam")

    class FusedAdam(torch.optim.Adam):
        pass

    ds_adam_mod.FusedAdam = FusedAdam
    ds_ops_mod.adam = ds_adam_mod
    ds_mod.ops = ds_ops_mod

    sys.modules["deepspeed"] = ds_mod
    sys.modules["deepspeed.ops"] = ds_ops_mod
    sys.modules["deepspeed.ops.adam"] = ds_adam_mod


try:
    from deepspeed.ops.adam import FusedAdam as _FusedAdamCheck  # noqa: F401
except Exception:
    _install_deepspeed_stub()

try:
    from phyla.utils.utils import load_config as _LoadConfigCheck  # noqa: F401
    from phyla.eval.evo_reasoning_eval import Config as _ConfigCheck  # noqa: F401
except Exception:
    _install_phyla_stub()

from data.dataset import PhylaDataModule
from model.model import return_model
from run.TrainingModule import TrainingModule
from utils.bhv_distance import bhv_geodesic_with_support
from utils.bhv_utils import BHVEncoder
from utils.random_tree import Tree
from utils.utils import get_batch_polytomy_indices, get_possible_ids


def _find_latest_checkpoint(checkpoint_dir):
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    def _epoch_num(path):
        m = re.search(r"epoch=(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(checkpoint_paths, key=lambda p: (os.path.getmtime(p), _epoch_num(p), p))


def _normalized_rf(newick_a, newick_b):
    t1 = EteTree(newick_a, format=1)
    t2 = EteTree(newick_b, format=1)
    rf_result = t1.robinson_foulds(t2, unrooted_trees=True)
    rf_distance = rf_result[0]
    max_rf = rf_result[1]
    return 0.0 if max_rf == 0 else rf_distance / max_rf


def _normalize_tree_like_dataset(tree_newick):
    t_obj = EteTree(tree_newick, format=1)
    leaves = t_obj.get_leaves()
    leaves.sort(key=lambda x: x.name)

    seq_ordering_map = {}
    for i, leaf in enumerate(leaves):
        original_name = leaf.name
        mapped_name = str(i)
        leaf.name = mapped_name
        seq_ordering_map[original_name] = mapped_name

    return t_obj.write(format=1), seq_ordering_map


def _remap_tree_with_sequence_ordering(
    tree_newick, seq_ordering_map, offset=0, tree_kind="tree"
):
    t_obj = EteTree(tree_newick, format=1)
    for leaf in t_obj.get_leaves():
        lookup_name = leaf.name
        if offset:
            try:
                lookup_name = str(int(lookup_name) + offset)
            except ValueError as exc:
                raise ValueError(
                    f"Non-integer leaf name '{leaf.name}' encountered in {tree_kind} while applying offset {offset}."
                ) from exc

        mapped_name = seq_ordering_map.get(lookup_name)
        if mapped_name is None:
            raise ValueError(
                f"Leaf name '{lookup_name}' in {tree_kind} not found in sequence ordering map."
            )
        leaf.name = mapped_name

    return t_obj.write(format=1)


def _encode_newick_lengths(newick):
    t = Tree(newick)
    enc = BHVEncoder()
    masks, lens = enc.return_BHV_encoding(t)
    return {int(m): float(l) for m, l in zip(masks, lens) if l is not None}, t.n_leaves


def _geodesic_state_at_time(geodesic_result, t):
    segments = geodesic_result["segments"]
    total_len = sum(seg["length"] for seg in segments)
    if total_len <= 0.0:
        return {}, {}

    s = float(t) * total_len
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
    velocity = {e: seg["velocity"].get(e, 0.0) * total_len for e in keys}
    return lengths, velocity


def _load_checkpoint_model(config, checkpoint_path, device):
    model = return_model(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    model_state = {
        k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")
    }
    if not model_state:
        model_state = state_dict

    model.load_state_dict(model_state, strict=True)
    model.eval()
    return model


def _build_sanity_tree_pair(config):
    ids = get_possible_ids(config["data"]["nexus_root"])
    if not ids:
        raise RuntimeError("No dataset IDs found for sanity sampling test.")

    ran = random.Random(42)
    ran.shuffle(ids)
    train_ids = ids[: int(0.8 * len(ids))]
    test_ids = ids[int(0.8 * len(ids)) :]
    if not train_ids:
        train_ids = ids
    if not test_ids:
        test_ids = ids

    data_module = PhylaDataModule(config, train_ids=train_ids, test_ids=test_ids)
    real_tree_raw = data_module.dataset_train.return_posterior_trees(0)[0]
    target_tree, seq_ordering_map = _normalize_tree_like_dataset(real_tree_raw)

    random_tree_raw = data_module.dataset_train.sample_random_tree(real_tree_raw)
    start_tree = _remap_tree_with_sequence_ordering(
        random_tree_raw,
        seq_ordering_map,
        offset=0,
        tree_kind="start tree",
    )

    n_leaves = len(EteTree(target_tree, format=1).get_leaves())
    return data_module, start_tree, target_tree, n_leaves


class GroundTruthVelocityModel(nn.Module):
    def __init__(self, base_model, start_newick, target_newick):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = base_model.tokenizer

        start_tree = Tree(start_newick)
        target_tree = Tree(target_newick)
        if start_tree.n_leaves != target_tree.n_leaves:
            raise ValueError("Start and target trees must have the same number of leaves")

        self.n_total_leaves = start_tree.n_leaves
        self.bio_bits = self.n_total_leaves - 1
        self.full_bio = (1 << self.bio_bits) - 1 if self.bio_bits > 0 else 0

        start_lengths, _ = _encode_newick_lengths(start_newick)
        target_lengths, _ = _encode_newick_lengths(target_newick)
        self.geodesic_result = bhv_geodesic_with_support(
            start_lengths, target_lengths, n_leaves=self.n_total_leaves
        )

    def _canonical_mask(self, mask):
        m = int(mask)
        if self.full_bio == 0:
            return m
        m = m & self.full_bio
        return min(m, self.full_bio ^ m)

    def forward(
        self,
        batched_tokenized_trees,
        t,
        phyla_embeddings=None,
        return_leafs_only=False,
        return_edges_only=True,
        autoregressive=False,
    ):
        if autoregressive:
            return self.base_model(
                batched_tokenized_trees,
                t,
                phyla_embeddings=phyla_embeddings,
                return_leafs_only=return_leafs_only,
                return_edges_only=return_edges_only,
                autoregressive=True,
            )

        edge_split_masks = batched_tokenized_trees[-1]
        edge_mask = batched_tokenized_trees[-2]
        device = batched_tokenized_trees[0].device

        t_float = float(t.item()) if torch.is_tensor(t) else float(t)
        _, vel_map_full = _geodesic_state_at_time(self.geodesic_result, t_float)

        vel_map_bio = {}
        for k, v in vel_map_full.items():
            k_bio = int(k) & self.full_bio
            if k_bio in (0, self.full_bio):
                continue
            vel_map_bio[self._canonical_mask(k_bio)] = float(v)

        batch_size = len(edge_split_masks)
        max_edges = max((len(masks) for masks in edge_split_masks), default=0)
        velocity = torch.zeros((batch_size, max_edges, 1), dtype=torch.float32, device=device)

        for b_idx, masks in enumerate(edge_split_masks):
            for e_idx, split in enumerate(masks):
                m = int(split)
                if m == 0:
                    continue
                velocity[b_idx, e_idx, 0] = vel_map_bio.get(self._canonical_mask(m), 0.0)

        return velocity, edge_mask


class GroundTruthAutoregressiveModel(nn.Module):
    def __init__(self, base_model, target_newick):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = base_model.tokenizer

        target_tree = Tree(target_newick)
        self.n_total_leaves = target_tree.n_leaves
        self.bio_bits = self.n_total_leaves - 1
        self.full_bio = (1 << self.bio_bits) - 1 if self.bio_bits > 0 else 0

        target_tokenized = self.tokenizer([target_newick])
        target_masks = [int(s) for s in target_tokenized[-1][0] if int(s) != 0]
        self.target_canonical = {
            self._canonical_mask(s)
            for s in target_masks
            if self._canonical_mask(s) not in (0, self.full_bio)
        }

    def _canonical_mask(self, mask):
        m = int(mask)
        if self.full_bio == 0:
            return m
        m = m & self.full_bio
        return min(m, self.full_bio ^ m)

    def forward(
        self,
        batched_tokenized_trees,
        t,
        phyla_embeddings=None,
        return_leafs_only=False,
        return_edges_only=True,
        autoregressive=False,
    ):
        if not autoregressive:
            return self.base_model(
                batched_tokenized_trees,
                t,
                phyla_embeddings=phyla_embeddings,
                return_leafs_only=return_leafs_only,
                return_edges_only=return_edges_only,
                autoregressive=False,
            )

        edge_split_masks = batched_tokenized_trees[-1]
        edge_mask = batched_tokenized_trees[-2]
        device = batched_tokenized_trees[0].device

        num_leaves = []
        for masks in edge_split_masks:
            max_bit = 0
            for s in masks:
                s_int = int(s)
                if s_int != 0:
                    max_bit = max(max_bit, s_int.bit_length())
            num_leaves.append(max_bit)

        batch_groups, batch_group_splits = get_batch_polytomy_indices(
            edge_split_masks=edge_split_masks,
            edge_mask=edge_mask,
            min_children=3,
            include_root=True,
            num_leaves=num_leaves,
        )

        outputs = []
        for b_idx, groups in enumerate(batch_groups):
            current_splits = [int(s) for s in edge_split_masks[b_idx] if int(s) != 0]
            current_canonical = {self._canonical_mask(s) for s in current_splits}

            for g_idx, _group in enumerate(groups):
                splits = [int(s) for s in batch_group_splits[b_idx][g_idx]]
                if len(splits) < 2:
                    continue

                logits = torch.full(
                    (len(splits), len(splits)),
                    float("-inf"),
                    dtype=torch.float32,
                    device=device,
                )

                chosen = None
                for i in range(len(splits)):
                    for j in range(i + 1, len(splits)):
                        candidate = int(splits[i]) | int(splits[j])
                        if self.full_bio:
                            candidate = candidate & self.full_bio
                        if candidate == 0:
                            continue
                        if self.full_bio and candidate == self.full_bio:
                            continue

                        candidate_can = self._canonical_mask(candidate)
                        if (
                            candidate_can in self.target_canonical
                            and candidate_can not in current_canonical
                        ):
                            chosen = (i, j)
                            break
                    if chosen is not None:
                        break

                if chosen is not None:
                    i, j = chosen
                    logits[i, j] = 20.0
                    logits[j, i] = 20.0
                    polytomy_score = 10.0
                else:
                    polytomy_score = -10.0

                outputs.append(
                    {
                        "batch_index": b_idx,
                        "group_indices": groups[g_idx],
                        "logits": logits,
                        "splits_represented": splits,
                        "polytomy_pred": torch.tensor(polytomy_score, device=device),
                    }
                )

        return outputs


class GroundTruthPolytomyPredModel(nn.Module):
    def __init__(self, base_model, target_newick):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = base_model.tokenizer

        target_tree = Tree(target_newick)
        self.n_total_leaves = target_tree.n_leaves
        self.bio_bits = self.n_total_leaves - 1
        self.full_bio = (1 << self.bio_bits) - 1 if self.bio_bits > 0 else 0

        target_tokenized = self.tokenizer([target_newick])
        target_masks = [int(s) for s in target_tokenized[-1][0] if int(s) != 0]
        self.target_canonical = {
            self._canonical_mask(s)
            for s in target_masks
            if self._canonical_mask(s) not in (0, self.full_bio)
        }

    def _canonical_mask(self, mask):
        m = int(mask)
        if self.full_bio == 0:
            return m
        m = m & self.full_bio
        return min(m, self.full_bio ^ m)

    def _group_has_true_merge(self, splits, current_canonical):
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                candidate = int(splits[i]) | int(splits[j])
                if self.full_bio:
                    candidate = candidate & self.full_bio
                if candidate == 0:
                    continue
                if self.full_bio and candidate == self.full_bio:
                    continue

                candidate_can = self._canonical_mask(candidate)
                if (
                    candidate_can in self.target_canonical
                    and candidate_can not in current_canonical
                ):
                    return True
        return False

    def forward(
        self,
        batched_tokenized_trees,
        t,
        phyla_embeddings=None,
        return_leafs_only=False,
        return_edges_only=True,
        autoregressive=False,
    ):
        if not autoregressive:
            return self.base_model(
                batched_tokenized_trees,
                t,
                phyla_embeddings=phyla_embeddings,
                return_leafs_only=return_leafs_only,
                return_edges_only=return_edges_only,
                autoregressive=False,
            )

        outputs = self.base_model(
            batched_tokenized_trees,
            t,
            phyla_embeddings=phyla_embeddings,
            return_leafs_only=return_leafs_only,
            return_edges_only=return_edges_only,
            autoregressive=True,
        )

        edge_split_masks = batched_tokenized_trees[-1]
        patched_outputs = []

        for output in outputs:
            b_idx = int(output["batch_index"])
            current_splits = [int(s) for s in edge_split_masks[b_idx] if int(s) != 0]
            current_canonical = {self._canonical_mask(s) for s in current_splits}

            splits = [int(s) for s in output["splits_represented"]]
            has_true_merge = self._group_has_true_merge(splits, current_canonical)

            patched = dict(output)
            patched["polytomy_pred"] = torch.tensor(
                10.0 if has_true_merge else -10.0,
                dtype=torch.float32,
                device=output["logits"].device,
            )
            patched_outputs.append(patched)

        return patched_outputs


class TestCheckpointSamplingRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        random.seed(13)
        torch.manual_seed(13)
        cls.device = torch.device("cpu")

        with open("configs/sanity_train.yaml", "r", encoding="utf-8") as f:
            cls.config = yaml.safe_load(f)

        checkpoint_dir = cls.config["trainer"]["checkpoint_dir"]
        cls.checkpoint_path = _find_latest_checkpoint(checkpoint_dir)

        (
            cls.data_module,
            cls.start_tree,
            cls.target_tree,
            cls.n_leaves,
        ) = _build_sanity_tree_pair(cls.config)

        base_model = _load_checkpoint_model(cls.config, cls.checkpoint_path, cls.device)

        cls.full_model_module = TrainingModule(
            model=copy.deepcopy(base_model),
            dataset=cls.data_module,
            lr=cls.config["trainer"]["lr"],
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            phyla_checkpoint_path=None,
        ).to(cls.device)
        cls.full_model_module.eval()

        cls.gt_velocity_module = TrainingModule(
            model=GroundTruthVelocityModel(
                copy.deepcopy(base_model),
                start_newick=cls.start_tree,
                target_newick=cls.target_tree,
            ),
            dataset=cls.data_module,
            lr=cls.config["trainer"]["lr"],
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            phyla_checkpoint_path=None,
        ).to(cls.device)
        cls.gt_velocity_module.eval()

        cls.gt_velocity_and_gt_polytomy_pred_module = TrainingModule(
            model=GroundTruthPolytomyPredModel(
                GroundTruthVelocityModel(
                    copy.deepcopy(base_model),
                    start_newick=cls.start_tree,
                    target_newick=cls.target_tree,
                ),
                target_newick=cls.target_tree,
            ),
            dataset=cls.data_module,
            lr=cls.config["trainer"]["lr"],
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            phyla_checkpoint_path=None,
        ).to(cls.device)
        cls.gt_velocity_and_gt_polytomy_pred_module.eval()

        cls.gt_autoregressive_module = TrainingModule(
            model=GroundTruthAutoregressiveModel(
                copy.deepcopy(base_model),
                target_newick=cls.target_tree,
            ),
            dataset=cls.data_module,
            lr=cls.config["trainer"]["lr"],
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            phyla_checkpoint_path=None,
        ).to(cls.device)
        cls.gt_autoregressive_module.eval()

        # Combined oracle: ground-truth velocity + ground-truth autoregressive.
        cls.gt_velocity_and_autoregressive_module = TrainingModule(
            model=GroundTruthAutoregressiveModel(
                GroundTruthVelocityModel(
                    copy.deepcopy(base_model),
                    start_newick=cls.start_tree,
                    target_newick=cls.target_tree,
                ),
                target_newick=cls.target_tree,
            ),
            dataset=cls.data_module,
            lr=cls.config["trainer"]["lr"],
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            phyla_checkpoint_path=None,
        ).to(cls.device)
        cls.gt_velocity_and_autoregressive_module.eval()

        phyla_dim = int(cls.config["model"]["phyla_dim"])
        cls.phyla_embeddings = torch.zeros(
            (1, cls.n_leaves, phyla_dim),
            dtype=torch.float32,
            device=cls.device,
        )

    def _sample_and_score(self, module):
        sampled_trees, _, _, _, _ = module.sample(
            [self.start_tree],
            phyla_embeddings=self.phyla_embeddings,
            num_samples=1,
            T=1.0,
            dt_base=0.02,
            max_steps=256,
            max_events=1024,
        )

        self.assertEqual(len(sampled_trees), 1)
        sampled_tree = sampled_trees[0]
        EteTree(sampled_tree, format=1)
        return _normalized_rf(sampled_tree, self.target_tree)

    def test_checkpoint_model_only_sampling_recovers_target_tree(self):
        norm_rf = self._sample_and_score(self.full_model_module)
        self.assertEqual(
            norm_rf,
            0.0,
            msg=(
                f"Checkpoint {self.checkpoint_path} failed pure-model recovery on sanity trees. "
                f"normalized RF={norm_rf}"
            ),
        )

    def test_checkpoint_gt_velocity_model_autoregressive_sampling_recovers_target_tree(self):
        norm_rf = self._sample_and_score(self.gt_velocity_module)
        self.assertEqual(
            norm_rf,
            0.0,
            msg=(
                f"Checkpoint {self.checkpoint_path} failed GT-velocity + model-AR recovery. "
                f"normalized RF={norm_rf}"
            ),
        )

    def test_checkpoint_gt_autoregressive_model_velocity_sampling_recovers_target_tree(self):
        norm_rf = self._sample_and_score(self.gt_autoregressive_module)
        self.assertEqual(
            norm_rf,
            0.0,
            msg=(
                f"Checkpoint {self.checkpoint_path} failed GT-AR + model-velocity recovery. "
                f"normalized RF={norm_rf}"
            ),
        )

    def test_checkpoint_gt_velocity_and_gt_polytomy_pred_recovers_ar_path(self):
        baseline_rf = self._sample_and_score(self.gt_velocity_module)
        gt_polytomy_rf = self._sample_and_score(
            self.gt_velocity_and_gt_polytomy_pred_module
        )
        self.assertEqual(
            gt_polytomy_rf,
            0.0,
            msg=(
                f"Ground-truth polytomy_pred did not recover AR path. "
                f"baseline_rf={baseline_rf}, gt_polytomy_rf={gt_polytomy_rf}"
            ),
        )

    def test_checkpoint_gt_velocity_and_gt_autoregressive_sampling_recovers_target_tree(self):
        norm_rf = self._sample_and_score(self.gt_velocity_and_autoregressive_module)
        self.assertEqual(
            norm_rf,
            0.0,
            msg=(
                f"Checkpoint {self.checkpoint_path} failed GT-velocity + GT-AR recovery. "
                f"If this fails, the issue is likely in sampler dynamics, not the model heads. "
                f"normalized RF={norm_rf}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
