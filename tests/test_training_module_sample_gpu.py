import random
import sys
import types
import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn
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

from model.treeTokenizer import TreeFeatureTokenizer
from run.TrainingModule import TrainingModule
from utils.bhv_distance import bhv_geodesic_with_support
from utils.bhv_utils import BHVEncoder
from utils.random_tree import Tree
from utils.utils import get_batch_polytomy_indices


def _encode_newick_lengths(newick):
    t = Tree(newick)
    enc = BHVEncoder()
    masks, lens = enc.return_BHV_encoding(t)
    return {int(m): float(l) for m, l in zip(masks, lens) if l is not None}, t.n_leaves


def _geodesic_state_at_time(geodesic_result, t):
    segments = geodesic_result["segments"]
    total_L = sum(seg["length"] for seg in segments)
    if total_L <= 0.0:
        return {}, {}

    s = float(t) * total_L
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


def _remap_tree_with_sequence_ordering(tree_newick, seq_ordering_map, offset=0, tree_kind="tree"):
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


class FakeSamplerModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.tokenizer = TreeFeatureTokenizer(
            num_node_types=3,
            num_edge_types=2,
            hidden_dim=64,
            n_layers=2,
            lap_dim=8,
            lap_dropout=0.0,
            orf_dim=8,
            max_nodes=256,
        ).to(device)
        self.non_ar_calls = 0
        self.ar_calls = 0

    @staticmethod
    def _pick_non_existing_merge_pair(splits):
        split_set = set(splits)
        max_bit = max((s.bit_length() for s in splits), default=0)
        full_mask = (1 << max_bit) - 1 if max_bit > 0 else 0

        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                new_split = int(splits[i]) | int(splits[j])
                if full_mask:
                    new_split = min(new_split, full_mask ^ new_split)

                if new_split == 0:
                    continue
                if full_mask and new_split == full_mask:
                    continue
                if new_split not in split_set:
                    return i, j
        return None

    def forward(
        self,
        batched_tokenized_trees,
        t,
        phyla_embeddings=None,
        return_leafs_only=False,
        return_edges_only=True,
        autoregressive=False,
    ):
        edge_split_masks = batched_tokenized_trees[-1]
        edge_mask = batched_tokenized_trees[-2]
        device = batched_tokenized_trees[0].device

        if not autoregressive:
            self.non_ar_calls += 1

            batch_size = len(edge_split_masks)
            max_edges = max((len(masks) for masks in edge_split_masks), default=0)
            velocity = torch.zeros(
                (batch_size, max_edges, 1), dtype=torch.float32, device=device
            )

            for b_idx, masks in enumerate(edge_split_masks):
                for e_idx, split in enumerate(masks):
                    split_int = int(split)
                    # Drive boundary-crossings by shrinking non-leaf edges.
                    velocity[b_idx, e_idx, 0] = -1.0 if split_int.bit_count() > 1 else 0.0

            return velocity, None

        self.ar_calls += 1
        outputs = []
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

        for b_idx, groups in enumerate(batch_groups):
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
                pair = self._pick_non_existing_merge_pair(splits)
                if pair is not None:
                    i, j = pair
                    logits[i, j] = 8.0
                    logits[j, i] = 8.0

                outputs.append(
                    {
                        "batch_index": b_idx,
                        "group_indices": groups[g_idx],
                        "logits": logits,
                        "splits_represented": splits,
                        "polytomy_pred": torch.tensor(1.0, device=device),
                    }
                )

        return outputs


class ComplementDuplicatePathologicalSamplerModel(FakeSamplerModel):
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
            return super().forward(
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

        self.ar_calls += 1
        outputs = []
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

        for b_idx, groups in enumerate(batch_groups):
            current_splits = {int(s) for s in edge_split_masks[b_idx] if int(s) != 0}
            full_mask = (1 << num_leaves[b_idx]) - 1 if num_leaves[b_idx] > 0 else 0
            current_canonical = (
                {min(s, full_mask ^ s) for s in current_splits}
                if full_mask
                else set(current_splits)
            )

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
                        if full_mask:
                            candidate &= full_mask
                        if candidate == 0:
                            continue
                        if full_mask and candidate == full_mask:
                            continue

                        candidate_canonical = (
                            min(candidate, full_mask ^ candidate)
                            if full_mask
                            else candidate
                        )

                        if (
                            candidate not in current_splits
                            and candidate_canonical in current_canonical
                        ):
                            chosen = (i, j)
                            break
                    if chosen is not None:
                        break

                if chosen is not None:
                    i, j = chosen
                    logits[i, j] = 20.0
                    logits[j, i] = 20.0

                outputs.append(
                    {
                        "batch_index": b_idx,
                        "group_indices": groups[g_idx],
                        "logits": logits,
                        "splits_represented": splits,
                        "polytomy_pred": torch.tensor(1.0, device=device),
                    }
                )

        return outputs


class OracleGeodesicSamplerModel(nn.Module):
    def __init__(self, device, start_newick, target_newick):
        super().__init__()
        self.tokenizer = TreeFeatureTokenizer(
            num_node_types=3,
            num_edge_types=2,
            hidden_dim=64,
            n_layers=2,
            lap_dim=8,
            lap_dropout=0.0,
            orf_dim=8,
            max_nodes=256,
        ).to(device)

        self.non_ar_calls = 0
        self.ar_calls = 0

        start_tree = Tree(start_newick)
        target_tree = Tree(target_newick)
        if start_tree.n_leaves != target_tree.n_leaves:
            raise ValueError("Start and target trees must have the same number of leaves.")

        self.n_total_leaves = start_tree.n_leaves
        self.bio_bits = self.n_total_leaves - 1
        self.full_bio = (1 << self.bio_bits) - 1 if self.bio_bits > 0 else 0

        start_lengths, _ = _encode_newick_lengths(start_newick)
        target_lengths, _ = _encode_newick_lengths(target_newick)
        self.geodesic_result = bhv_geodesic_with_support(
            start_lengths, target_lengths, n_leaves=self.n_total_leaves
        )

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
        edge_split_masks = batched_tokenized_trees[-1]
        edge_mask = batched_tokenized_trees[-2]
        device = batched_tokenized_trees[0].device

        if not autoregressive:
            self.non_ar_calls += 1
            t_float = float(t.item()) if torch.is_tensor(t) else float(t)
            _, vel_map_full = _geodesic_state_at_time(self.geodesic_result, t_float)

            vel_map_bio = {}
            for k, v in vel_map_full.items():
                k_bio = int(k) & self.full_bio
                if k_bio == 0 or k_bio == self.full_bio:
                    continue
                vel_map_bio[self._canonical_mask(k_bio)] = float(v)

            batch_size = len(edge_split_masks)
            max_edges = max((len(masks) for masks in edge_split_masks), default=0)
            velocity = torch.zeros(
                (batch_size, max_edges, 1), dtype=torch.float32, device=device
            )

            for b_idx, masks in enumerate(edge_split_masks):
                for e_idx, split in enumerate(masks):
                    m = int(split)
                    if m == 0:
                        velocity[b_idx, e_idx, 0] = 0.0
                        continue
                    m_can = self._canonical_mask(m)
                    velocity[b_idx, e_idx, 0] = vel_map_bio.get(m_can, 0.0)

            return velocity, None

        self.ar_calls += 1
        outputs = []
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
                        if candidate == 0:
                            continue
                        if self.full_bio > 0 and candidate == self.full_bio:
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


class TestTrainingModuleSample(unittest.TestCase):
    # def test_sample_n10_seed1_pathological_merge_regression(self):
    #     random.seed(1)
    #     torch.manual_seed(1)
    #     device = torch.device("cpu")

    #     pathological_model = ComplementDuplicatePathologicalSamplerModel(device=device)
    #     module = TrainingModule(
    #         model=pathological_model,
    #         dataset=MagicMock(),
    #         lr=1e-4,
    #         record=False,
    #         epochs=1,
    #         deepspeed=False,
    #         logger=None,
    #     ).to(device)
    #     module.eval()

    #     start_tree = str(Tree(num_leaves=10, random=True))
    #     phyla_embeddings = torch.zeros((1, 10, 16), dtype=torch.float32, device=device)

    #     sampled_trees, n_topology_changes, avg_max_logit, avg_polytomy_size, n_polytomies = module.sample(
    #         [start_tree],
    #         phyla_embeddings=phyla_embeddings,
    #         num_samples=1,
    #         T=0.5,
    #         dt_base=0.05,
    #         max_steps=120,
    #         max_events=1000,
    #     )

    #     self.assertEqual(len(sampled_trees), 1)
    #     self.assertIsInstance(sampled_trees[0], str)
    #     self.assertGreater(len(sampled_trees[0]), 0)
    #     EteTree(sampled_trees[0], format=1)

    #     self.assertIsInstance(n_topology_changes, int)
    #     self.assertIsInstance(n_polytomies, int)
    #     self.assertIsInstance(avg_max_logit, float)
    #     self.assertIsInstance(avg_polytomy_size, float)

    def test_sample_runs_on_cpu_with_real_tokenizer_path(self):
        random.seed(7)
        torch.manual_seed(7)
        device = torch.device("cpu")

        fake_model = FakeSamplerModel(device=device)
        module = TrainingModule(
            model=fake_model,
            dataset=MagicMock(),
            lr=1e-4,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
        ).to(device)
        module.eval()

        random_tree =  '((((((74:0.00158,67:0.00147):0.00219,(((80:0.00175,(69:0.00153,(81:0.00128,68:0.00047):0.00108):0.00021):0.00156,4:0.00497):0.00013,((133:0.00707,((115:0.00142,96:0.00162):0.00389,(127:0.00419,57:0.00234):0.00024):0.00043):0.00610,(((129:0.01483,128:0.01258):0.02349,10:0.02220):0.01067,(((110:0.01986,32:0.03018):0.00085,27:0.02439):0.00161,((84:0.02267,(90:0.05224,65:0.01999):0.06741):0.08712,((((98:0.01559,95:0.01357):0.00483,((((113:0.00996,(118:0.01509,112:0.01146):0.00201):0.00243,111:0.01759):0.00797,(126:0.01728,(((153:0.03625,135:0.03207):0.02023,((123:0.00285,(122:0.00146,121:0.00028):0.00194):0.00538,86:0.01007):0.00596):0.00314,44:0.01368):0.00247):0.00218):0.00103,(60:0.00611,(16:0.00049,15:0.00169):0.00607):0.01037):0.00035):0.00489,(((125:0.01786,124:0.01565):0.00892,(((130:0.03630,114:0.03724):0.00223,106:0.02954):0.00318,102:0.02412):0.00628):0.00240,((54:0.01483,19:0.02781):0.01776,((((45:0.00394,(136:0.00477,(51:0.00003,18:0.00041):0.00002):0.00346):0.01614,(20:0.01759,((((116:0.00391,(((105:0.00001,104:0.00042):0.00068,103:0.00237):0.00066,((63:0.00081,29:0.00050):0.00051,(30:0.00104,28:0.00070):0.00036):0.00043):0.00088):0.00037,26:0.00309):0.00137,(120:0.00521,25:0.00175):0.00206):0.00460,(17:0.00870,((62:0.00415,((131:0.00377,(100:0.00199,85:0.00976):0.00058):0.00016,53:0.00071):0.00238):0.00213,(55:0.00505,(97:0.00254,13:0.00443):0.00431):0.00283):0.00208):0.01527):0.00236):0.00180):0.00285,((99:0.01663,(23:0.00062,(22:0.00022,21:0.00022):0.00068):0.01808):0.00968,(101:0.00441,12:0.00560):0.01501):0.00020):0.00037,((((108:0.01911,48:0.01179):0.00251,(46:0.00155,((58:0.00043,37:0.00001):0.00143,((56:0.00191,38:0.00157):0.00046,(36:0.00140,24:0.00121):0.00020):0.00042):0.00063):0.00811):0.00699,((59:0.00365,(50:0.00067,39:0.00107):0.00288):0.00897,(((((109:0.00043,49:0.00000):0.00067,47:0.00107):0.00202,34:0.00102):0.00266,((107:0.00304,43:0.00088):0.00136,31:0.00212):0.00103):0.00509,(40:0.00414,(35:0.00000,14:0.00000):0.00326):0.00796):0.00133):0.00212):0.00123,((41:0.00009,33:0.00165):0.01313,(148:0.01747,(42:0.00000,6:0.00000):0.02431):0.02581):0.00435):0.00457):0.00092):0.00074):0.00062):0.00244,((52:0.00623,(94:0.00106,1:0.03289):0.00421):0.02268,(((((143:0.01269,142:0.00907):0.06123,(140:0.02122,((141:0.01612,(139:0.00865,138:0.00789):0.00673):0.00139,137:0.02363):0.00457):0.02164):0.02569,(134:0.10397,(144:0.07853,(151:0.09246,(154:0.27905,117:0.06125):0.07269):0.04451):0.01766):0.00510):0.00403,((147:0.01720,146:0.01718):0.01042,(119:0.03662,83:0.02953):0.00351):0.01413):0.00066,(82:0.04391,((9:0.03393,((((145:0.01759,91:0.00547):0.02071,((132:0.01806,93:0.01632):0.00670,(92:0.01241,(89:0.00723,88:0.00408):0.01522):0.00272):0.00104):0.00989,87:0.02134):0.00196,((11:0.00845,(61:0.00226,8:0.00427):0.00417):0.01284,(150:0.03854,(7:0.00957,5:0.00871):0.00715):0.00266):0.00414):0.01177):0.00574,(((155:0.32836,152:0.04457):0.02833,149:0.02867):0.07311,2:0.09508):0.02500):0.00275):0.00127):0.00272):0.00096):0.00172):0.00170):0.00016):0.01078):0.00373):0.00032):0.00194,((79:0.00087,78:0.00044):0.00036,3:0.00225):0.00104):0.00028,(77:0.00227,(76:0.00200,64:0.00148):0.00034):0.00008):0.00025,(70:0.00002,66:0.00085):0.00013):0.00009,(73:0.00075,(72:0.00087,71:0.00044):0.00012):0.00022,75:0.00196):0.00000;'
        real_tree_newick = '((52:6.821929e-03,((((2:4.398080e-02,(((((145:8.657433e-03,91:4.622826e-03):2.222114e-02,((93:1.284674e-02,132:1.985680e-02):8.439914e-03,(((89:1.020633e-02,88:4.611548e-03):8.036501e-03,90:1.429933e-02):1.439583e-02,92:9.908956e-03):1.750425e-03):5.724766e-03):7.037225e-03,87:1.626403e-02):4.747258e-03,(((7:9.739291e-03,5:5.587849e-03):6.613494e-03,(150:1.664500e-02,152:1.724125e-02):1.823357e-02):1.534167e-03,(11:9.469409e-03,(61:5.838223e-04,8:8.467112e-03):4.839481e-03):1.478233e-02):5.742910e-03):1.842902e-02,9:3.628046e-02):4.929418e-03):4.712541e-03,82:4.185746e-02):3.380475e-03,(((((124:1.558528e-02,125:1.705412e-02):5.917463e-03,((102:2.129084e-02,155:9.638513e-03):4.271744e-03,(149:2.436750e-02,(106:2.722806e-02,(130:3.687226e-02,114:4.149299e-02):1.000061e-02):4.023099e-03):2.147868e-03):1.233641e-02):3.719823e-03,((((54:1.214695e-02,19:2.168458e-02):1.996561e-02,((((38:1.057892e-03,((36:7.753757e-04,(46:2.622027e-03,24:1.787443e-03):2.029430e-04):1.481560e-03,(58:1.714262e-04,37:2.458943e-04):2.601513e-03):1.274226e-04):2.389942e-03,56:1.949454e-03):1.489446e-02,(48:1.137571e-02,108:1.964277e-02):1.324782e-03):7.644887e-03,(((39:1.804400e-03,50:3.879771e-04):3.067958e-03,59:3.259752e-03):6.995533e-03,(((6:1.574761e-04,42:2.710545e-05):4.343268e-04,(33:2.616433e-03,41:6.108494e-06):1.184825e-03):1.058122e-02,(((14:6.407019e-05,35:1.283827e-03):3.152093e-03,40:3.500127e-03):1.097830e-02,((((107:1.333270e-03,84:3.209979e-04):2.963766e-03,43:1.910537e-03):1.115617e-03,31:2.194031e-03):1.856883e-03,((47:2.408071e-03,(109:3.641932e-04,49:1.155934e-04):3.415181e-03):2.683443e-03,34:2.281842e-03):4.028540e-03):3.179844e-03):2.139192e-03):1.070869e-03):2.343303e-03):8.589032e-03):3.753159e-03,(((((((136:1.054858e-03,45:1.777442e-04):9.470442e-04,18:8.069737e-04):2.680686e-03,51:1.513089e-03):1.472587e-02,(20:1.791231e-02,((17:6.050183e-03,(((53:2.324964e-03,((100:9.971849e-04,85:1.361781e-03):1.047096e-03,131:4.906427e-03):7.243432e-04):1.869450e-03,62:6.183967e-03):2.898555e-03,(55:6.269062e-03,(13:5.770393e-03,97:4.514650e-03):3.446166e-03):3.282607e-03):1.379633e-03):2.240766e-02,((120:1.168439e-03,25:4.536840e-03):2.917149e-03,(((((63:3.576971e-04,((104:6.782281e-04,105:8.341392e-05):2.542285e-03,(103:2.267958e-03,148:4.839063e-05):2.790594e-04):1.147993e-03):3.031773e-04,29:1.385090e-03):2.901980e-04,(28:1.255159e-03,30:8.757002e-04):1.490579e-03):1.139399e-03,26:2.656695e-03):5.694807e-04,116:5.345046e-03):3.441066e-03):2.961141e-03):5.676048e-03):6.805834e-03):5.342581e-03,(((22:5.519097e-04,21:5.260546e-04):1.063093e-03,23:1.409785e-03):2.054762e-02,99:1.438413e-02):1.228856e-02):1.107615e-03,(((95:9.283287e-03,98:1.739668e-02):6.505733e-03,((16:1.866647e-03,(153:9.563353e-04,15:6.850920e-04):5.016234e-04):6.647760e-03,60:6.967831e-03):1.297224e-02):6.103056e-04,((111:1.652701e-02,(113:1.019696e-02,(118:1.794353e-02,112:9.963961e-03):5.803295e-03):3.456559e-03):1.124700e-02,(126:1.994845e-02,((86:1.119997e-02,(((135:4.125296e-03,123:2.609975e-03):1.338629e-03,122:5.599224e-04):1.247347e-04,121:1.956632e-03):4.255348e-03):6.514329e-03,44:8.702270e-03):1.406738e-03):3.081689e-03):4.390131e-03):6.593693e-03):2.821015e-04,(101:4.858902e-03,12:4.198135e-03):1.335322e-02):4.585962e-04):2.756843e-03,((144:4.292494e-02,((143:8.977315e-03,142:8.391560e-03):7.271121e-02,(140:1.547496e-02,(137:2.375343e-02,(141:1.801854e-02,(139:4.845625e-03,138:7.509111e-03):5.295656e-03):2.358940e-03):1.229887e-02):2.599447e-02):4.443259e-02):7.088933e-03,134:5.769189e-02):6.826836e-03):5.876646e-04):4.031933e-03,(32:3.250200e-02,((117:3.522599e-02,(151:7.473737e-03,110:9.434156e-03):5.396982e-03):7.039427e-03,(27:2.809174e-02,154:2.327384e-02):4.710086e-03):2.332033e-03):3.477086e-03):1.630971e-03,((((127:5.961962e-03,57:2.631306e-03):6.478147e-04,((96:1.409200e-03,115:2.740476e-03):7.137445e-03,133:7.424993e-03):1.159072e-03):6.523926e-03,(4:5.405740e-03,(((((81:6.815847e-04,68:1.155674e-03):2.735598e-03,69:7.025004e-04):8.998414e-04,80:2.236265e-03):1.839654e-03,(((3:4.236887e-03,79:1.677530e-03):3.770879e-04,78:1.062032e-03):2.312128e-03,((77:1.334890e-03,(66:2.301659e-04,((70:8.259407e-05,(75:2.060793e-03,65:3.982049e-03):3.647037e-04):4.418750e-04,(71:1.999872e-03,(72:7.517143e-04,73:6.105537e-04):8.489613e-05):5.596686e-04):9.888103e-04):1.665829e-03):7.810491e-04,(64:1.400547e-03,76:2.738529e-03):7.529917e-04):1.655366e-03):3.548366e-03):4.212272e-04,(67:1.216874e-03,74:1.827134e-03):2.430697e-03):1.433684e-03):2.358579e-03):1.337100e-02,((128:1.130754e-02,129:1.857543e-02):2.664069e-02,10:2.449606e-02):1.431688e-02):1.932062e-03):3.746715e-03):1.785518e-03,((83:4.168728e-02,119:4.097966e-02):7.403229e-03,(146:1.888170e-02,147:1.810523e-02):1.032872e-02):2.647861e-02):2.939351e-02):4.262485e-03,94:2.756089e-04,1:6.820178e-04);'
        real_tree_preprocessed, seq_ordering_map = _normalize_tree_like_dataset(real_tree_newick)
        random_tree_preprocessed = _remap_tree_with_sequence_ordering(
            random_tree, seq_ordering_map, offset=0, tree_kind="random tree"
        )

        n_leaves = len(EteTree(real_tree_preprocessed, format=1).get_leaves())
        phyla_embeddings = torch.zeros((1, n_leaves, 16), dtype=torch.float32, device=device)

        sampled_trees, n_topology_changes, avg_max_logit, avg_polytomy_size, n_polytomies = module.sample(
            [random_tree_preprocessed],
            phyla_embeddings=phyla_embeddings,
            num_samples=1,
            T=0.2,
            dt_base=0.05,
            max_steps=16,
            max_events=128,
        )

        self.assertGreater(fake_model.non_ar_calls, 0)
        self.assertGreater(fake_model.ar_calls, 0)
        self.assertEqual(len(sampled_trees), 1)
        self.assertIsInstance(sampled_trees[0], str)
        self.assertGreater(len(sampled_trees[0]), 0)
        EteTree(sampled_trees[0], format=1)

        self.assertIsInstance(n_topology_changes, int)
        self.assertIsInstance(n_polytomies, int)
        self.assertIsInstance(avg_max_logit, float)
        self.assertIsInstance(avg_polytomy_size, float)

    def test_sample_matches_target_with_geodesic_oracle_outputs(self):
        random.seed(11)
        torch.manual_seed(11)
        device = torch.device("cpu")

        # start_tree = str(Tree(num_leaves=8, random=True))
        # target_tree = str(Tree(num_leaves=8, random=True))

        start_tree =  '((((((74:0.00158,67:0.00147):0.00219,(((80:0.00175,(69:0.00153,(81:0.00128,68:0.00047):0.00108):0.00021):0.00156,4:0.00497):0.00013,((133:0.00707,((115:0.00142,96:0.00162):0.00389,(127:0.00419,57:0.00234):0.00024):0.00043):0.00610,(((129:0.01483,128:0.01258):0.02349,10:0.02220):0.01067,(((110:0.01986,32:0.03018):0.00085,27:0.02439):0.00161,((84:0.02267,(90:0.05224,65:0.01999):0.06741):0.08712,((((98:0.01559,95:0.01357):0.00483,((((113:0.00996,(118:0.01509,112:0.01146):0.00201):0.00243,111:0.01759):0.00797,(126:0.01728,(((153:0.03625,135:0.03207):0.02023,((123:0.00285,(122:0.00146,121:0.00028):0.00194):0.00538,86:0.01007):0.00596):0.00314,44:0.01368):0.00247):0.00218):0.00103,(60:0.00611,(16:0.00049,15:0.00169):0.00607):0.01037):0.00035):0.00489,(((125:0.01786,124:0.01565):0.00892,(((130:0.03630,114:0.03724):0.00223,106:0.02954):0.00318,102:0.02412):0.00628):0.00240,((54:0.01483,19:0.02781):0.01776,((((45:0.00394,(136:0.00477,(51:0.00003,18:0.00041):0.00002):0.00346):0.01614,(20:0.01759,((((116:0.00391,(((105:0.00001,104:0.00042):0.00068,103:0.00237):0.00066,((63:0.00081,29:0.00050):0.00051,(30:0.00104,28:0.00070):0.00036):0.00043):0.00088):0.00037,26:0.00309):0.00137,(120:0.00521,25:0.00175):0.00206):0.00460,(17:0.00870,((62:0.00415,((131:0.00377,(100:0.00199,85:0.00976):0.00058):0.00016,53:0.00071):0.00238):0.00213,(55:0.00505,(97:0.00254,13:0.00443):0.00431):0.00283):0.00208):0.01527):0.00236):0.00180):0.00285,((99:0.01663,(23:0.00062,(22:0.00022,21:0.00022):0.00068):0.01808):0.00968,(101:0.00441,12:0.00560):0.01501):0.00020):0.00037,((((108:0.01911,48:0.01179):0.00251,(46:0.00155,((58:0.00043,37:0.00001):0.00143,((56:0.00191,38:0.00157):0.00046,(36:0.00140,24:0.00121):0.00020):0.00042):0.00063):0.00811):0.00699,((59:0.00365,(50:0.00067,39:0.00107):0.00288):0.00897,(((((109:0.00043,49:0.00000):0.00067,47:0.00107):0.00202,34:0.00102):0.00266,((107:0.00304,43:0.00088):0.00136,31:0.00212):0.00103):0.00509,(40:0.00414,(35:0.00000,14:0.00000):0.00326):0.00796):0.00133):0.00212):0.00123,((41:0.00009,33:0.00165):0.01313,(148:0.01747,(42:0.00000,6:0.00000):0.02431):0.02581):0.00435):0.00457):0.00092):0.00074):0.00062):0.00244,((52:0.00623,(94:0.00106,1:0.03289):0.00421):0.02268,(((((143:0.01269,142:0.00907):0.06123,(140:0.02122,((141:0.01612,(139:0.00865,138:0.00789):0.00673):0.00139,137:0.02363):0.00457):0.02164):0.02569,(134:0.10397,(144:0.07853,(151:0.09246,(154:0.27905,117:0.06125):0.07269):0.04451):0.01766):0.00510):0.00403,((147:0.01720,146:0.01718):0.01042,(119:0.03662,83:0.02953):0.00351):0.01413):0.00066,(82:0.04391,((9:0.03393,((((145:0.01759,91:0.00547):0.02071,((132:0.01806,93:0.01632):0.00670,(92:0.01241,(89:0.00723,88:0.00408):0.01522):0.00272):0.00104):0.00989,87:0.02134):0.00196,((11:0.00845,(61:0.00226,8:0.00427):0.00417):0.01284,(150:0.03854,(7:0.00957,5:0.00871):0.00715):0.00266):0.00414):0.01177):0.00574,(((155:0.32836,152:0.04457):0.02833,149:0.02867):0.07311,2:0.09508):0.02500):0.00275):0.00127):0.00272):0.00096):0.00172):0.00170):0.00016):0.01078):0.00373):0.00032):0.00194,((79:0.00087,78:0.00044):0.00036,3:0.00225):0.00104):0.00028,(77:0.00227,(76:0.00200,64:0.00148):0.00034):0.00008):0.00025,(70:0.00002,66:0.00085):0.00013):0.00009,(73:0.00075,(72:0.00087,71:0.00044):0.00012):0.00022,75:0.00196):0.00000;'
        target_tree = '((52:6.821929e-03,((((2:4.398080e-02,(((((145:8.657433e-03,91:4.622826e-03):2.222114e-02,((93:1.284674e-02,132:1.985680e-02):8.439914e-03,(((89:1.020633e-02,88:4.611548e-03):8.036501e-03,90:1.429933e-02):1.439583e-02,92:9.908956e-03):1.750425e-03):5.724766e-03):7.037225e-03,87:1.626403e-02):4.747258e-03,(((7:9.739291e-03,5:5.587849e-03):6.613494e-03,(150:1.664500e-02,152:1.724125e-02):1.823357e-02):1.534167e-03,(11:9.469409e-03,(61:5.838223e-04,8:8.467112e-03):4.839481e-03):1.478233e-02):5.742910e-03):1.842902e-02,9:3.628046e-02):4.929418e-03):4.712541e-03,82:4.185746e-02):3.380475e-03,(((((124:1.558528e-02,125:1.705412e-02):5.917463e-03,((102:2.129084e-02,155:9.638513e-03):4.271744e-03,(149:2.436750e-02,(106:2.722806e-02,(130:3.687226e-02,114:4.149299e-02):1.000061e-02):4.023099e-03):2.147868e-03):1.233641e-02):3.719823e-03,((((54:1.214695e-02,19:2.168458e-02):1.996561e-02,((((38:1.057892e-03,((36:7.753757e-04,(46:2.622027e-03,24:1.787443e-03):2.029430e-04):1.481560e-03,(58:1.714262e-04,37:2.458943e-04):2.601513e-03):1.274226e-04):2.389942e-03,56:1.949454e-03):1.489446e-02,(48:1.137571e-02,108:1.964277e-02):1.324782e-03):7.644887e-03,(((39:1.804400e-03,50:3.879771e-04):3.067958e-03,59:3.259752e-03):6.995533e-03,(((6:1.574761e-04,42:2.710545e-05):4.343268e-04,(33:2.616433e-03,41:6.108494e-06):1.184825e-03):1.058122e-02,(((14:6.407019e-05,35:1.283827e-03):3.152093e-03,40:3.500127e-03):1.097830e-02,((((107:1.333270e-03,84:3.209979e-04):2.963766e-03,43:1.910537e-03):1.115617e-03,31:2.194031e-03):1.856883e-03,((47:2.408071e-03,(109:3.641932e-04,49:1.155934e-04):3.415181e-03):2.683443e-03,34:2.281842e-03):4.028540e-03):3.179844e-03):2.139192e-03):1.070869e-03):2.343303e-03):8.589032e-03):3.753159e-03,(((((((136:1.054858e-03,45:1.777442e-04):9.470442e-04,18:8.069737e-04):2.680686e-03,51:1.513089e-03):1.472587e-02,(20:1.791231e-02,((17:6.050183e-03,(((53:2.324964e-03,((100:9.971849e-04,85:1.361781e-03):1.047096e-03,131:4.906427e-03):7.243432e-04):1.869450e-03,62:6.183967e-03):2.898555e-03,(55:6.269062e-03,(13:5.770393e-03,97:4.514650e-03):3.446166e-03):3.282607e-03):1.379633e-03):2.240766e-02,((120:1.168439e-03,25:4.536840e-03):2.917149e-03,(((((63:3.576971e-04,((104:6.782281e-04,105:8.341392e-05):2.542285e-03,(103:2.267958e-03,148:4.839063e-05):2.790594e-04):1.147993e-03):3.031773e-04,29:1.385090e-03):2.901980e-04,(28:1.255159e-03,30:8.757002e-04):1.490579e-03):1.139399e-03,26:2.656695e-03):5.694807e-04,116:5.345046e-03):3.441066e-03):2.961141e-03):5.676048e-03):6.805834e-03):5.342581e-03,(((22:5.519097e-04,21:5.260546e-04):1.063093e-03,23:1.409785e-03):2.054762e-02,99:1.438413e-02):1.228856e-02):1.107615e-03,(((95:9.283287e-03,98:1.739668e-02):6.505733e-03,((16:1.866647e-03,(153:9.563353e-04,15:6.850920e-04):5.016234e-04):6.647760e-03,60:6.967831e-03):1.297224e-02):6.103056e-04,((111:1.652701e-02,(113:1.019696e-02,(118:1.794353e-02,112:9.963961e-03):5.803295e-03):3.456559e-03):1.124700e-02,(126:1.994845e-02,((86:1.119997e-02,(((135:4.125296e-03,123:2.609975e-03):1.338629e-03,122:5.599224e-04):1.247347e-04,121:1.956632e-03):4.255348e-03):6.514329e-03,44:8.702270e-03):1.406738e-03):3.081689e-03):4.390131e-03):6.593693e-03):2.821015e-04,(101:4.858902e-03,12:4.198135e-03):1.335322e-02):4.585962e-04):2.756843e-03,((144:4.292494e-02,((143:8.977315e-03,142:8.391560e-03):7.271121e-02,(140:1.547496e-02,(137:2.375343e-02,(141:1.801854e-02,(139:4.845625e-03,138:7.509111e-03):5.295656e-03):2.358940e-03):1.229887e-02):2.599447e-02):4.443259e-02):7.088933e-03,134:5.769189e-02):6.826836e-03):5.876646e-04):4.031933e-03,(32:3.250200e-02,((117:3.522599e-02,(151:7.473737e-03,110:9.434156e-03):5.396982e-03):7.039427e-03,(27:2.809174e-02,154:2.327384e-02):4.710086e-03):2.332033e-03):3.477086e-03):1.630971e-03,((((127:5.961962e-03,57:2.631306e-03):6.478147e-04,((96:1.409200e-03,115:2.740476e-03):7.137445e-03,133:7.424993e-03):1.159072e-03):6.523926e-03,(4:5.405740e-03,(((((81:6.815847e-04,68:1.155674e-03):2.735598e-03,69:7.025004e-04):8.998414e-04,80:2.236265e-03):1.839654e-03,(((3:4.236887e-03,79:1.677530e-03):3.770879e-04,78:1.062032e-03):2.312128e-03,((77:1.334890e-03,(66:2.301659e-04,((70:8.259407e-05,(75:2.060793e-03,65:3.982049e-03):3.647037e-04):4.418750e-04,(71:1.999872e-03,(72:7.517143e-04,73:6.105537e-04):8.489613e-05):5.596686e-04):9.888103e-04):1.665829e-03):7.810491e-04,(64:1.400547e-03,76:2.738529e-03):7.529917e-04):1.655366e-03):3.548366e-03):4.212272e-04,(67:1.216874e-03,74:1.827134e-03):2.430697e-03):1.433684e-03):2.358579e-03):1.337100e-02,((128:1.130754e-02,129:1.857543e-02):2.664069e-02,10:2.449606e-02):1.431688e-02):1.932062e-03):3.746715e-03):1.785518e-03,((83:4.168728e-02,119:4.097966e-02):7.403229e-03,(146:1.888170e-02,147:1.810523e-02):1.032872e-02):2.647861e-02):2.939351e-02):4.262485e-03,94:2.756089e-04,1:6.820178e-04);'

        # Mirror dataset preprocessing: real tree is normalized to 0..N-1 and
        # random/start tree is remapped with the same sequence ordering map.
        target_tree, seq_ordering_map = _normalize_tree_like_dataset(target_tree)
        start_tree = _remap_tree_with_sequence_ordering(
            start_tree, seq_ordering_map, offset=0, tree_kind="start tree"
        )

        oracle_model = OracleGeodesicSamplerModel(
            device=device, start_newick=start_tree, target_newick=target_tree
        )
        module = TrainingModule(
            model=oracle_model,
            dataset=MagicMock(),
            lr=1e-4,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
        ).to(device)
        module.eval()

        n_leaves = len(EteTree(target_tree, format=1).get_leaves())
        phyla_embeddings = torch.zeros((1, n_leaves, 16), dtype=torch.float32, device=device)

        sampled_trees, _, _, _, _ = module.sample(
            [start_tree],
            phyla_embeddings=phyla_embeddings,
            num_samples=1,
            T=1.0,
            dt_base=0.02,
            max_steps=256,
            max_events=1024,
        )

        self.assertGreater(oracle_model.non_ar_calls, 0)
        self.assertGreater(oracle_model.ar_calls, 0)
        self.assertEqual(len(sampled_trees), 1)

        sampled_tree = sampled_trees[0]
        norm_rf = _normalized_rf(sampled_tree, target_tree)
        self.assertEqual(
            norm_rf,
            0.0,
            f"Expected perfect topology recovery (norm-RF=0), got {norm_rf}",
        )


if __name__ == "__main__":
    unittest.main()
