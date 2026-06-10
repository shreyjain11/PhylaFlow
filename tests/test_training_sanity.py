import random
import sys
import types
import unittest
import itertools
from unittest.mock import MagicMock
from unittest.mock import patch
import copy
import math
import numpy as np

import torch
from ete3 import Tree as EteTree

_DT_FIRST_HIT_TOL = 0.01


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

from model.model import TreeDenoiserTokenGT
from data.dataset import TreeDataset
from run.TrainingModule import (
    TrainingModule,
    _apply_boundary_vanish_one_step,
    _best_pairwise_merge_label_for_current_tree,
    _build_autoregressive_replay_batch,
    _build_velocity_replay_batch,
    _collect_oracle_replay_samples_from_anchors,
    _boundary_event_distribution_loss,
    _boundary_event_precision_margin_loss,
    _tree_to_model_split_lengths,
    _combine_autoregressive_losses,
    _edge_set_bce_loss,
    _oracle_training_topology_keys,
    _plan_autoregressive_boundary_merges,
    _predict_boundary_vanish_mask_from_logits,
    _record_repeated_topology_visit,
    _select_replay_samples_across_rollout,
    _select_rollout_replay_anchors,
    _summarize_trace_topology_repeats,
    _summarize_fixed_pair_eval_rows,
    _topology_key,
)
from utils.bhv_distance import bhv_geodesic_with_support
from utils.bhv_utils import (
    BHVEncoder,
    _filter_training_boundary_events,
    _split_multi_label_training_events,
    get_structural_polytomy_groups_from_newick,
    return_boundary_training_geodesic,
    return_sampled_tree_boundary_decisions,
    return_sampled_tree_orthant_velocity,
    return_tree_boundary_merge_paths,
)
from utils.bhv_movie import build_tree_from_splits
from utils.metric_utils import calculate_norm_rf
from utils.random_tree import Tree
from utils.utils import remove_bit


def _detach_tokenized_batch(tokenized):
    out = []
    for item in tokenized:
        if torch.is_tensor(item):
            out.append(item.detach())
        else:
            out.append(item)
    return tuple(out)


def _make_single_velocity_batch(tokenizer, n_leaves, seed):
    random.seed(seed)
    torch.manual_seed(seed)

    start_tree = str(Tree(num_leaves=n_leaves, random=True))
    target_tree = str(Tree(num_leaves=n_leaves, random=True))
    sampled_newick, velocity = return_sampled_tree_orthant_velocity(
        start_tree,
        target_tree,
        0.0,
        legacy_training_semantics=True,
    )
    with torch.no_grad():
        tokenized = _detach_tokenized_batch(tokenizer([sampled_newick]))

    return {
        "tokenized_trees": tokenized,
        "batched_time": torch.tensor([0.0], dtype=torch.float32),
        "phyla_embeddings": None,
        "original_trees": [sampled_newick],
        "batched_velocity": [velocity],
        "num_leaves": [Tree(sampled_newick).n_leaves],
    }


def _leaf_sort_key(name):
    try:
        return (0, int(name))
    except ValueError:
        return (1, str(name))


def _prune_and_renumber_tree_pair(start_newick, target_newick, keep_leaves=12):
    t_start = EteTree(start_newick, format=1)
    t_target = EteTree(target_newick, format=1)

    start_names = {leaf.name for leaf in t_start.get_leaves()}
    target_names = {leaf.name for leaf in t_target.get_leaves()}
    common = sorted(start_names & target_names, key=_leaf_sort_key)
    if len(common) < keep_leaves:
        raise AssertionError(
            f"Not enough shared leaves to prune: have {len(common)}, need {keep_leaves}"
        )

    kept = common[:keep_leaves]
    t_start.prune(kept, preserve_branch_length=True)
    t_target.prune(kept, preserve_branch_length=True)

    renumber_map = {
        old_name: str(i)
        for i, old_name in enumerate(sorted(kept, key=_leaf_sort_key))
    }

    for tree in (t_start, t_target):
        for leaf in tree.get_leaves():
            leaf.name = renumber_map[leaf.name]

    return t_start.write(format=1), t_target.write(format=1)


def _make_batch_from_tree_pair(tokenizer, start_tree, target_tree, time_point=0.0):
    sampled_newick, velocity = return_sampled_tree_orthant_velocity(
        start_tree,
        target_tree,
        time_point,
        legacy_training_semantics=True,
    )
    with torch.no_grad():
        tokenized = _detach_tokenized_batch(tokenizer([sampled_newick]))

    return {
        "tokenized_trees": tokenized,
        "batched_time": torch.tensor([float(time_point)], dtype=torch.float32),
        "phyla_embeddings": None,
        "original_trees": [sampled_newick],
        "batched_velocity": [velocity],
        "num_leaves": [Tree(sampled_newick).n_leaves],
    }


def _make_batch_from_tree_pair_with_autoregressive(
    tokenizer,
    start_tree,
    target_tree,
    time_point=0.0,
    max_boundary_attempts=20,
):
    sampled_newick, velocity = return_sampled_tree_orthant_velocity(
        start_tree,
        target_tree,
        time_point,
        legacy_training_semantics=True,
    )

    boundary_labels = []
    for _ in range(max_boundary_attempts):
        boundary_labels = return_sampled_tree_boundary_decisions(
            start_tree,
            target_tree,
            legacy_training_semantics=True,
        )
        if boundary_labels:
            break

    if not boundary_labels:
        raise AssertionError(
            "Could not sample autoregressive boundary decisions for the sanity tree pair."
        )

    chosen_boundary = boundary_labels[0]
    with torch.no_grad():
        tokenized = _detach_tokenized_batch(tokenizer([sampled_newick]))
        tokenized_ar = _detach_tokenized_batch(tokenizer([chosen_boundary["newick"]]))

    return {
        "tokenized_trees": tokenized,
        "batched_time": torch.tensor([float(time_point)], dtype=torch.float32),
        "phyla_embeddings": None,
        "original_trees": [sampled_newick],
        "batched_velocity": [velocity],
        "num_leaves": [Tree(sampled_newick).n_leaves],
        "tokenized_autoregressive_trees": tokenized_ar,
        "newick_autoregressive_trees": [chosen_boundary["newick"]],
        "batched_autoregressive_time": torch.tensor([0.0], dtype=torch.float32),
        "batched_autoregressive_labels": [chosen_boundary["labels"]],
    }


def _make_autoregressive_event_batch(tokenizer, newick, labels, event_time):
    with torch.no_grad():
        tokenized_ar = _detach_tokenized_batch(tokenizer([newick]))

    return {
        "tokenized_autoregressive_trees": tokenized_ar,
        "newick_autoregressive_trees": [newick],
        "batched_autoregressive_time": torch.tensor(
            [float(event_time)], dtype=torch.float32
        ),
        "batched_autoregressive_labels": [labels],
        "phyla_embeddings": None,
    }


def _normalized_event_time(event_index, num_events):
    return 0.0 if num_events <= 1 else float(event_index) / float(num_events - 1)


def _select_random_nonbinary_boundary_path(start_tree, target_tree, seed=777):
    boundary_paths = return_tree_boundary_merge_paths(
        start_tree,
        target_tree,
        legacy_training_semantics=True,
    )
    candidates = [
        path
        for path in boundary_paths
        if len(path["events"]) > 1
        and any(
            len(label["merge_indices"]) > 2
            for event in path["events"]
            for label in event["labels"]
        )
    ]
    if not candidates:
        raise AssertionError(
            "Did not find any multi-step non-binary boundary path on the sanity tree pair."
        )

    rng = random.Random(seed)
    return rng.choice(candidates)


def _target_merge_subsets_for_event(labels):
    target = {}
    for label in labels:
        components = tuple(int(component) for component in label["components"])
        merge_subset = frozenset(int(components[idx]) for idx in label["merge_indices"])
        target.setdefault(components, set()).add(merge_subset)
    return target


def _decode_positive_merge_subsets(group, threshold_logit=0.0):
    splits = [int(split) for split in group["splits_represented"]]
    logits = group["logits"].detach().cpu()
    num_splits = len(splits)
    adjacency = {idx: set() for idx in range(num_splits)}

    for i in range(num_splits):
        for j in range(i + 1, num_splits):
            score = float(logits[i, j].item())
            if not math.isfinite(score) or score <= float(threshold_logit):
                continue
            adjacency[i].add(j)
            adjacency[j].add(i)

    visited = set()
    decoded_subsets = set()
    for idx in range(num_splits):
        if idx in visited:
            continue
        stack = [idx]
        component = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            stack.extend(adjacency[node] - visited)

        if len(component) >= 2:
            decoded_subsets.add(
                frozenset(splits[node_idx] for node_idx in sorted(component))
            )

    return decoded_subsets


def _predict_autoregressive_event(module, tokenizer, newick, event_time):
    groups = get_structural_polytomy_groups_from_newick(newick)
    with torch.no_grad():
        tokenized_ar = _detach_tokenized_batch(tokenizer([newick]))
        outputs = module.forward(
            tokenized_ar,
            torch.tensor([float(event_time)], dtype=torch.float32),
            None,
            autoregressive=True,
            autoregressive_component_groups=[groups],
        )

    predicted = {}
    if any(
        str(group.get("decoder_mode", "pairwise_threshold")) == "structured_subset"
        for group in outputs
    ):
        current_tree = Tree(newick)
        encoder = BHVEncoder()
        existing_masks, existing_lengths = encoder.return_BHV_encoding(current_tree)
        existing_splits = {
            int(mask)
            for mask, length in zip(existing_masks, existing_lengths)
            if length is not None and float(length) > 1e-8
        }
        planned = _plan_autoregressive_boundary_merges(
            outputs,
            existing_splits=existing_splits,
            threshold_logit=0.0,
        )
        for item in planned:
            group_key = tuple(int(split) for split in item["splits_represented"])
            predicted[group_key] = {
                frozenset(int(component) for component in subset)
                for subset, _ in item["subsets"]
            }
    else:
        allow_all_groups = len(outputs) == 1
        for group in outputs:
            group_key = tuple(int(split) for split in group["splits_represented"])
            if (
                (not allow_all_groups)
                and float(group["polytomy_pred"].detach().cpu().item()) <= 0.0
            ):
                continue
            predicted[group_key] = _decode_positive_merge_subsets(group)

    return predicted, outputs


def _boundary_start_length_map(start_tree, target_tree, boundary_index):
    start_obj = Tree(start_tree)
    geodesic = return_boundary_training_geodesic(start_tree, target_tree)
    boundary_lengths = {
        int(mask): float(length)
        for mask, length in geodesic["segments"][boundary_index]["end_lengths"].items()
        if float(length) > 1e-8
    }
    return boundary_lengths, start_obj.n_leaves, start_obj.id_to_name


def _apply_predicted_merge_subsets_to_length_map(length_map, predicted_group_subsets):
    next_lengths = dict(length_map)
    for merge_subsets in predicted_group_subsets.values():
        for subset in merge_subsets:
            new_split = 0
            for component in subset:
                new_split |= int(component)
            if new_split in next_lengths:
                raise AssertionError(
                    f"Predicted merge created an existing split {new_split}."
                )
            next_lengths[int(new_split)] = 0.1
    return next_lengths


def _assert_same_topology(testcase, left_newick, right_newick, message):
    left_tree = EteTree(left_newick)
    right_tree = EteTree(right_newick)
    rf_distance, max_rf, *_ = left_tree.robinson_foulds(
        right_tree,
        unrooted_trees=True,
    )
    testcase.assertEqual(
        0.0 if max_rf == 0 else rf_distance / max_rf,
        0.0,
        f"{message} RF distance={rf_distance}, max RF={max_rf}",
    )


def _normalized_rf(left_newick, right_newick):
    left_tree = EteTree(left_newick)
    right_tree = EteTree(right_newick)
    rf_distance, max_rf, *_ = left_tree.robinson_foulds(
        right_tree,
        unrooted_trees=True,
    )
    return 0.0 if max_rf == 0 else rf_distance / max_rf


def _rollout_boundary_with_autoregressive_head(
    testcase,
    module,
    tokenizer,
    boundary_path,
    start_tree,
    target_tree,
):
    current_lengths, n_leaves, mapping = _boundary_start_length_map(
        start_tree,
        target_tree,
        boundary_path["boundary_index"],
    )
    _, current_newick = build_tree_from_splits(
        list(current_lengths.keys()),
        current_lengths,
        n_leaves,
        root_leaf=n_leaves - 1,
        mapping=mapping,
    )

    for event_idx, event in enumerate(boundary_path["events"]):
        _assert_same_topology(
            testcase,
            current_newick,
            event["newick"],
            f"Rollout diverged before boundary event {event_idx}.",
        )
        predicted, _ = _predict_autoregressive_event(
            module,
            tokenizer,
            current_newick,
            _normalized_event_time(event_idx, len(boundary_path["events"])),
        )
        current_lengths = _apply_predicted_merge_subsets_to_length_map(
            current_lengths,
            predicted,
        )
        _, current_newick = build_tree_from_splits(
            list(current_lengths.keys()),
            current_lengths,
            n_leaves,
            root_leaf=n_leaves - 1,
            mapping=mapping,
        )
        expected_next_newick = (
            boundary_path["events"][event_idx + 1]["newick"]
            if event_idx + 1 < len(boundary_path["events"])
            else boundary_path["end_newick"]
        )
        _assert_same_topology(
            testcase,
            current_newick,
            expected_next_newick,
            f"Rollout diverged after boundary event {event_idx}.",
        )

    return current_newick


def _sample_to_first_boundary(module, start_tree, dt_hit_true, boundary_path):
    max_events = max(8, 4 * len(boundary_path["events"]))
    sampled_trees, *_ = module.sample(
        [start_tree],
        None,
        num_samples=1,
        T=float(dt_hit_true),
        dt_base=max(float(dt_hit_true), 1e-6),
        max_events=max_events,
        max_steps=2048,
    )
    sampled_tree = sampled_trees[0]
    return {
        "sampled_tree": sampled_tree,
        "target_tree": boundary_path["end_newick"],
        "rf_norm": _normalized_rf(sampled_tree, boundary_path["end_newick"]),
        "boundary_path": boundary_path,
    }


def _boundary_prefix_time(start_tree, target_tree, boundary_index):
    geodesic = return_boundary_training_geodesic(start_tree, target_tree)
    segment_lengths = [float(segment["length"]) for segment in geodesic["segments"]]
    total_length = sum(segment_lengths)
    if total_length <= 0.0:
        raise AssertionError("Degenerate geodesic length for sanity tree pair.")
    return sum(segment_lengths[: boundary_index + 1]) / total_length


def _gather_supervised_velocity(module, batch, eps_len=1e-8):
    with torch.no_grad():
        outputs = module.forward(
            batch["tokenized_trees"],
            batch["batched_time"],
            batch["phyla_embeddings"],
        )
        v_pred = outputs[0]
        edge_split_masks = outputs[1]

    velocity_labels = batch["batched_velocity"]
    num_leaves = batch["num_leaves"]
    original_trees = batch["original_trees"]

    preds = []
    labels = []
    matched_masks = []
    lengths = []
    encoder = BHVEncoder()

    for b_idx, vel_dict in enumerate(velocity_labels):
        model_masks = [int(m) for m in edge_split_masks[b_idx] if int(m) != 0]
        if not model_masks:
            continue
        real_max_bit = max(int(m).bit_length() for m in model_masks)
        full_mask = (1 << real_max_bit) - 1 if real_max_bit > 0 else 0

        tree_obj = Tree(original_trees[b_idx])
        bhv_masks, bhv_lengths = encoder.return_BHV_encoding(tree_obj)
        bhv_len_map = {
            int(m): float(l)
            for m, l in zip(bhv_masks, bhv_lengths)
            if l is not None
        }
        model_length_map = {}
        for m_model in model_masks:
            length = bhv_len_map.get(m_model)
            if length is None and full_mask:
                length = bhv_len_map.get(full_mask ^ m_model)
            if length is None:
                continue
            model_length_map[m_model] = float(length)

        for original_vel, true_vel in vel_dict.items():
            vel = int(original_vel)
            if vel.bit_length() == real_max_bit + 1:
                vel = remove_bit(vel, int(num_leaves[b_idx]) - 1)
            elif vel.bit_length() > real_max_bit + 1:
                continue

            matched_vel = vel
            if matched_vel not in edge_split_masks[b_idx]:
                full_mask = (1 << real_max_bit) - 1
                complement_vel = full_mask ^ matched_vel
                if complement_vel in edge_split_masks[b_idx]:
                    matched_vel = complement_vel
                else:
                    continue

            n_bits = real_max_bit
            k_bits = int(matched_vel).bit_count()
            is_pendant = min(k_bits, n_bits - k_bits) == 1
            if is_pendant:
                continue

            split_list = [int(m) for m in edge_split_masks[b_idx]]
            edge_idx = split_list.index(matched_vel)
            edge_length = model_length_map.get(int(matched_vel), None)
            if edge_length is None or edge_length <= eps_len:
                continue
            preds.append(v_pred[b_idx, edge_idx, 0].detach().cpu())
            labels.append(torch.tensor(float(true_vel), dtype=torch.float32))
            matched_masks.append(int(matched_vel))
            lengths.append(torch.tensor(float(edge_length), dtype=torch.float32))

    if not preds:
        raise AssertionError("No supervised non-pendant velocity edges were matched.")

    return (
        torch.stack(preds).float(),
        torch.stack(labels).float(),
        matched_masks,
        torch.stack(lengths).float(),
    )


def _pearson_corr(x, y):
    xm = x - x.mean()
    ym = y - y.mean()
    denom = xm.norm() * ym.norm()
    if float(denom) <= 1e-12:
        return 1.0 if torch.allclose(x, y) else 0.0
    return float((xm * ym).sum() / denom)


def _spearman_corr(x, y):
    xr = torch.empty_like(x)
    yr = torch.empty_like(y)
    xr[torch.argsort(x)] = torch.arange(x.numel(), dtype=x.dtype)
    yr[torch.argsort(y)] = torch.arange(y.numel(), dtype=y.dtype)
    return _pearson_corr(xr, yr)


def _topk_mask_overlap(pred, true, masks, k):
    k = min(int(k), int(pred.numel()))
    if k <= 0:
        return 1.0
    pred_idx = torch.topk(pred.abs(), k=k).indices.tolist()
    true_idx = torch.topk(true.abs(), k=k).indices.tolist()
    pred_masks = {masks[i] for i in pred_idx}
    true_masks = {masks[i] for i in true_idx}
    return len(pred_masks & true_masks) / float(k)


def _dt_hit_and_candidates(lengths, velocity, eps_len=1e-8):
    valid = lengths > float(eps_len)
    neg = (velocity < 0.0) & valid
    if int(neg.sum()) == 0:
        return float("inf"), torch.empty(0, dtype=torch.float32), neg

    dt_candidates = lengths[neg] / (-velocity[neg])
    dt_hit = float(torch.min(dt_candidates))
    return dt_hit, dt_candidates, neg


def _first_hit_mask_set(dt_all, neg_idx, masks, tol=0.0): # Default to exact
    if int(neg_idx.numel()) == 0:
        return set()

    min_dt = float(torch.min(dt_all))
    # If tol is 0.0, this finds only the exact mathematical minimum
    mask_indices = torch.where(torch.abs(dt_all - min_dt) <= float(tol))[0]
    
    global_indices = neg_idx[mask_indices]
    return {int(masks[int(i)]) for i in global_indices.tolist()}


def _dt_by_mask(lengths, velocity, masks, eps_len=1e-8):
    dt_map = {}
    valid = lengths > float(eps_len)
    for idx, mask in enumerate(masks):
        if not bool(valid[idx]):
            dt_map[int(mask)] = float("inf")
            continue

        vel = float(velocity[idx])
        if vel < 0.0:
            dt_map[int(mask)] = float(lengths[idx] / (-velocity[idx]).clamp_min(1e-8))
        else:
            dt_map[int(mask)] = float("inf")
    return dt_map


def _format_dt(value):
    if math.isfinite(value):
        return f"{value:.6e}"
    return "inf"


def _format_first_hit_miss_details(metrics):
    missed_masks = sorted(metrics["true_first_masks"] - metrics["pred_first_masks"])
    if not missed_masks:
        return "none"

    details = []
    pred_dt_by_mask = metrics["pred_dt_by_mask"]
    true_dt_by_mask = metrics["true_dt_by_mask"]
    for mask in missed_masks:
        pred_dt = pred_dt_by_mask.get(mask, float("inf"))
        true_dt = true_dt_by_mask.get(mask, float("inf"))
        details.append(
            f"{mask}: pred_dt={_format_dt(pred_dt)}, true_dt={_format_dt(true_dt)}"
        )
    return "; ".join(details)


def _velocity_metrics(module, batch, topk=3):
    pred, true, masks, lengths = _gather_supervised_velocity(module, batch)
    mse = float(torch.mean((pred - true) ** 2))
    cosine = float(torch.nn.functional.cosine_similarity(pred, true, dim=0))
    pearson = _pearson_corr(pred, true)
    spearman = _spearman_corr(pred, true)
    pred_dt_by_mask = _dt_by_mask(lengths, pred, masks)
    true_dt_by_mask = _dt_by_mask(lengths, true, masks)

    # Tiny near-zero velocities are numerically unstable for sign comparisons.
    moving = true.abs() > 1e-3
    if int(moving.sum()) > 0:
        sign_acc = float((torch.sign(pred[moving]) == torch.sign(true[moving])).float().mean())
    else:
        sign_acc = 1.0

    pred_dt_hit, pred_dt_candidates, pred_neg = _dt_hit_and_candidates(lengths, pred)
    true_dt_hit, true_dt_candidates, true_neg = _dt_hit_and_candidates(lengths, true)
    both_neg = pred_neg & true_neg
    any_neg = pred_neg | true_neg

    if int(any_neg.sum()) > 0:
        dt_neg_jaccard = float(int(both_neg.sum()) / int(any_neg.sum()))
    else:
        dt_neg_jaccard = 1.0

    if int(both_neg.sum()) > 0:
        pred_dt_overlap = lengths[both_neg] / (-pred[both_neg])
        true_dt_overlap = lengths[both_neg] / (-true[both_neg])
        dt_candidates_mae = float(torch.mean(torch.abs(pred_dt_overlap - true_dt_overlap)))
        dt_candidates_rel_mae = float(
            torch.mean(
                torch.abs(pred_dt_overlap - true_dt_overlap)
                / torch.clamp(torch.abs(true_dt_overlap), min=1e-8)
            )
        )
    else:
        dt_candidates_mae = 0.0 if int(any_neg.sum()) == 0 else float("inf")
        dt_candidates_rel_mae = 0.0 if int(any_neg.sum()) == 0 else float("inf")

    pred_top_masks = set()
    true_top_masks = set()
    pred_first_masks = set()
    true_first_masks = set()
    dt_first_hit_recall = 1.0
    dt_first_hit_precision = 1.0
    pred_neg_idx = torch.where(pred_neg)[0]
    true_neg_idx = torch.where(true_neg)[0]
    if int(pred_neg_idx.numel()) == 0 and int(true_neg_idx.numel()) == 0:
        dt_first_hit_match = 1.0
        dt_topk_overlap = 1.0
    elif int(pred_neg_idx.numel()) == 0 or int(true_neg_idx.numel()) == 0:
        dt_first_hit_match = 0.0
        dt_topk_overlap = 0.0
        dt_first_hit_recall = 0.0
        dt_first_hit_precision = 0.0
    else:
        pred_dt_all = lengths[pred_neg_idx] / (-pred[pred_neg_idx]).clamp_min(1e-8)
        true_dt_all = lengths[true_neg_idx] / (-true[true_neg_idx]).clamp_min(1e-8)

        pred_order = pred_neg_idx[torch.argsort(pred_dt_all)]
        true_order = true_neg_idx[torch.argsort(true_dt_all)]
        pred_first_masks = _first_hit_mask_set(pred_dt_all, pred_neg_idx, masks, tol=_DT_FIRST_HIT_TOL)
        true_first_masks = _first_hit_mask_set(true_dt_all, true_neg_idx, masks, tol = _DT_FIRST_HIT_TOL)
        first_hit_overlap = pred_first_masks & true_first_masks
        dt_first_hit_recall = len(first_hit_overlap) / float(len(true_first_masks))
        dt_first_hit_precision = len(first_hit_overlap) / float(len(pred_first_masks))
        dt_first_hit_match = 1.0 if true_first_masks.issubset(pred_first_masks) else 0.0

        k = min(3, int(pred_order.numel()), int(true_order.numel()))
        pred_top_masks = {int(masks[int(i)]) for i in pred_order[:k].tolist()}
        true_top_masks = {int(masks[int(i)]) for i in true_order[:k].tolist()}
        dt_topk_overlap = len(pred_top_masks & true_top_masks) / float(k)

    if math.isfinite(pred_dt_hit) and math.isfinite(true_dt_hit):
        dt_hit_abs_err = abs(pred_dt_hit - true_dt_hit)
        dt_hit_rel_err = dt_hit_abs_err / max(abs(true_dt_hit), 1e-8)
    elif (not math.isfinite(pred_dt_hit)) and (not math.isfinite(true_dt_hit)):
        dt_hit_abs_err = 0.0
        dt_hit_rel_err = 0.0
    else:
        dt_hit_abs_err = float("inf")
        dt_hit_rel_err = float("inf")

    topk_overlap = _topk_mask_overlap(pred, true, masks, k=topk)
    return {
        "mse": mse,
        "cosine": cosine,
        "pearson": pearson,
        "spearman": spearman,
        "sign_acc": sign_acc,
        "topk_overlap": topk_overlap,
        "dt_hit_pred": pred_dt_hit,
        "dt_hit_true": true_dt_hit,
        "dt_hit_abs_err": dt_hit_abs_err,
        "dt_hit_rel_err": dt_hit_rel_err,
        "dt_neg_jaccard": dt_neg_jaccard,
        "dt_first_hit_match": dt_first_hit_match,
        "dt_first_hit_recall": dt_first_hit_recall,
        "dt_first_hit_precision": dt_first_hit_precision,
        "dt_first_hit_tol": _DT_FIRST_HIT_TOL,
        "dt_topk_overlap": dt_topk_overlap,
        "dt_candidates_mae": dt_candidates_mae,
        "dt_candidates_rel_mae": dt_candidates_rel_mae,
        "n_pred_dt_candidates": int(pred_dt_candidates.numel()),
        "n_true_dt_candidates": int(true_dt_candidates.numel()),
        "pred_first_masks": pred_first_masks,
        "true_first_masks": true_first_masks,
        "pred_dt_by_mask": pred_dt_by_mask,
        "true_dt_by_mask": true_dt_by_mask,
        'pred_top_masks': pred_top_masks if dt_topk_overlap > 0 else set(),
        'true_top_masks': true_top_masks if dt_topk_overlap > 0 else set(),
        "n_supervised_edges": int(pred.numel()),
    }


class _OptimizerProxy:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        self.optimizer.step()


_ORIGINAL_TENSOR_TO = torch.Tensor.to


def _tensor_to_cpu_for_cuda(self, *args, **kwargs):
    if args:
        device = args[0]
        if isinstance(device, str) and device.startswith("cuda"):
            args = ("cpu",) + args[1:]
        elif isinstance(device, torch.device) and device.type == "cuda":
            args = (torch.device("cpu"),) + args[1:]

    if "device" in kwargs:
        device = kwargs["device"]
        if isinstance(device, str) and device.startswith("cuda"):
            kwargs["device"] = "cpu"
        elif isinstance(device, torch.device) and device.type == "cuda":
            kwargs["device"] = torch.device("cpu")

    return _ORIGINAL_TENSOR_TO(self, *args, **kwargs)


class TestTrainingSanity(unittest.TestCase):
    def test_combine_autoregressive_losses_respects_polytomy_weight(self):
        merge_loss = torch.tensor(2.0)
        polytomy_loss = torch.tensor(3.0)

        self.assertEqual(
            float(
                _combine_autoregressive_losses(
                    merge_loss,
                    polytomy_loss,
                    polytomy_choosing_weight=0.0,
                ).item()
            ),
            2.0,
        )
        self.assertEqual(
            float(
                _combine_autoregressive_losses(
                    merge_loss,
                    polytomy_loss,
                    polytomy_choosing_weight=0.5,
                ).item()
            ),
            3.5,
        )

    def test_filter_training_boundary_events_marks_last_structural_event_stop(self):
        boundary_paths = [
            {
                "events": [
                    {
                        "newick": "event0",
                        "labels": [{"components": [1, 2, 4]}],
                    },
                    {
                        "newick": "event1",
                        "labels": [{"components": [1, 2]}],
                    },
                    {
                        "newick": "event2",
                        "labels": [{"components": [8, 16, 32]}],
                    },
                ]
            }
        ]

        events = _filter_training_boundary_events(boundary_paths)

        self.assertEqual(len(events), 2)
        self.assertFalse(events[0]["stop_after_merge"])
        self.assertTrue(events[1]["stop_after_merge"])

    def test_filter_training_boundary_events_keeps_stop_false_for_multi_merge_event(self):
        boundary_paths = [
            {
                "events": [
                    {
                        "newick": "event0",
                        "labels": [
                            {"components": [1, 2, 4]},
                            {"components": [8, 16, 32]},
                        ],
                    }
                ]
            }
        ]

        events = _filter_training_boundary_events(boundary_paths)

        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]["stop_after_merge"])

    def test_planner_falls_back_to_best_structured_subset_when_threshold_blocks_group(self):
        output = {
            "polytomy_pred": torch.tensor([-0.2], dtype=torch.float32),
            "splits_represented": [1, 2, 4],
            "decoder_mode": "structured_subset",
            "starter_pair_logits": torch.tensor([3.0, -1.0, -2.0], dtype=torch.float32),
            "starter_pair_indices": [(0, 1), (0, 2), (1, 2)],
            "member_logits": torch.tensor(
                [
                    [-10.0, -10.0, -10.0],
                    [-10.0, -10.0, -10.0],
                    [-10.0, -10.0, -10.0],
                ],
                dtype=torch.float32,
            ),
        }
        blocked_output = {
            "polytomy_pred": torch.tensor([-0.5], dtype=torch.float32),
            "splits_represented": [8, 16, 32],
            "decoder_mode": "structured_subset",
            "starter_pair_logits": torch.tensor([0.1, -1.0, -2.0], dtype=torch.float32),
            "starter_pair_indices": [(0, 1), (0, 2), (1, 2)],
            "member_logits": torch.full((3, 3), -10.0, dtype=torch.float32),
        }

        planned = _plan_autoregressive_boundary_merges(
            [blocked_output, output],
            existing_splits=set(),
            threshold_logit=0.0,
        )

        self.assertEqual(len(planned), 1)
        self.assertEqual(tuple(planned[0]["subsets"][0][0]), (1, 2))
        self.assertEqual(int(planned[0]["subsets"][0][1]), 3)
        self.assertEqual(planned[0].get("decoder_mode"), "structured_subset")
        self.assertTrue(planned[0].get("fallback"))

    def test_structured_planner_propagates_stop_after_merge_logit(self):
        output = {
            "polytomy_pred": torch.tensor([0.5], dtype=torch.float32),
            "splits_represented": [1, 2, 4],
            "decoder_mode": "structured_subset",
            "starter_pair_logits": torch.tensor([3.0, -1.0, -2.0], dtype=torch.float32),
            "starter_pair_indices": [(0, 1), (0, 2), (1, 2)],
            "member_logits": torch.full((3, 3), -10.0, dtype=torch.float32),
            "stop_after_merge_logit": torch.tensor(2.0, dtype=torch.float32),
        }

        planned = _plan_autoregressive_boundary_merges(
            [output],
            existing_splits=set(),
            threshold_logit=0.0,
        )

        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0].get("decoder_mode"), "structured_subset")
        self.assertAlmostEqual(planned[0]["stop_after_merge_logit"], 2.0)

    def test_planner_falls_back_to_best_pair_when_pairwise_has_no_positive_edges(self):
        logits = torch.tensor(
            [
                [float("-inf"), -0.1, -0.5],
                [-0.1, float("-inf"), -0.3],
                [-0.5, -0.3, float("-inf")],
            ],
            dtype=torch.float32,
        )
        output = {
            "polytomy_pred": torch.tensor([-0.2], dtype=torch.float32),
            "splits_represented": [1, 2, 4],
            "decoder_mode": "pairwise_threshold",
            "logits": logits,
        }
        blocked_output = {
            "polytomy_pred": torch.tensor([-0.5], dtype=torch.float32),
            "splits_represented": [8, 16, 32],
            "decoder_mode": "pairwise_threshold",
            "logits": torch.tensor(
                [
                    [float("-inf"), -2.0, -3.0],
                    [-2.0, float("-inf"), -4.0],
                    [-3.0, -4.0, float("-inf")],
                ],
                dtype=torch.float32,
            ),
        }

        planned = _plan_autoregressive_boundary_merges(
            [blocked_output, output],
            existing_splits=set(),
            threshold_logit=0.0,
        )

        self.assertEqual(len(planned), 1)
        self.assertEqual(tuple(planned[0]["subsets"][0][0]), (1, 2))
        self.assertEqual(int(planned[0]["subsets"][0][1]), 3)
        self.assertEqual(planned[0].get("decoder_mode"), "pairwise_threshold")
        self.assertTrue(planned[0].get("fallback"))

    def test_sample_compare_harness_respects_overfit_event_prefix_cap(self):
        module = object.__new__(TrainingModule)
        module.model = types.SimpleNamespace(
            phyla_proj=types.SimpleNamespace(in_features=8)
        )
        module.training_sampling_dt_base = 0.02

        captured = {}

        def fake_pair(train=True):
            return {
                "start_tree": "((0:0.1,1:0.1):0.1,(2:0.1,3:0.1):0.1);",
                "target_tree": "((0:0.1,2:0.1):0.1,(1:0.1,3:0.1):0.1);",
                "n_leaves": 4,
                "max_events": 1,
            }

        def fake_sample(*args, **kwargs):
            captured["max_events"] = kwargs.get("max_events")
            return (
                [fake_pair()["target_tree"]],
                None,
                None,
                None,
                None,
                {"velocity": [], "autoregressive": []},
            )

        module._get_harness_sampling_pair = fake_pair
        module.sample = fake_sample

        with patch.object(
            TrainingModule,
            "device",
            new=property(lambda self: torch.device("cpu")),
        ):
            metrics = TrainingModule.sample_compare_harness(module, train=True)

        self.assertEqual(captured["max_events"], 1)
        self.assertEqual(metrics["rf_norm"], 0.0)
        self.assertGreater(metrics["start_rf_norm"], 0.0)

    def test_sample_compare_harness_includes_fixed_pair_path_metrics(self):
        module = object.__new__(TrainingModule)
        module.model = types.SimpleNamespace(
            phyla_proj=types.SimpleNamespace(in_features=8)
        )
        module.training_sampling_dt_base = 0.02

        def fake_pair(train=True):
            return {
                "start_tree": "((0:0.1,1:0.1):0.1,(2:0.1,3:0.1):0.1);",
                "target_tree": "((0:0.1,2:0.1):0.1,(1:0.1,3:0.1):0.1);",
                "n_leaves": 4,
                "max_events": 1,
            }

        def fake_sample(*args, **kwargs):
            return (
                [fake_pair()["target_tree"]],
                None,
                None,
                None,
                None,
                {"velocity": [], "autoregressive": []},
            )

        module._get_harness_sampling_pair = fake_pair
        module.sample = fake_sample
        module._evaluate_fixed_pair_path_metrics = lambda train=True: {
            "fixed_path_velocity_joint_exact_frac": 0.5,
            "fixed_path_autoregressive_exact_frac": 0.75,
        }

        with patch.object(
            TrainingModule,
            "device",
            new=property(lambda self: torch.device("cpu")),
        ):
            metrics = TrainingModule.sample_compare_harness(module, train=True)

        self.assertEqual(metrics["fixed_path_velocity_joint_exact_frac"], 0.5)
        self.assertEqual(metrics["fixed_path_autoregressive_exact_frac"], 0.75)

    def test_sample_compare_harness_reports_skipped_no_valid_boundary_revisits(self):
        module = object.__new__(TrainingModule)
        module.model = types.SimpleNamespace(
            phyla_proj=types.SimpleNamespace(in_features=8)
        )
        module.training_sampling_dt_base = 0.02
        module._evaluate_fixed_pair_path_metrics = lambda train=True: {}

        def fake_pair(train=True):
            return {
                "start_tree": "((0:0.1,1:0.1):0.1,(2:0.1,3:0.1):0.1);",
                "target_tree": "((0:0.1,2:0.1):0.1,(1:0.1,3:0.1):0.1);",
                "n_leaves": 4,
                "max_events": 1,
            }

        def fake_sample(*args, **kwargs):
            return (
                [fake_pair()["target_tree"]],
                None,
                None,
                None,
                None,
                {
                    "velocity": [],
                    "autoregressive": [],
                    "skipped_no_valid_boundary_revisits": 7.0,
                },
            )

        module._get_harness_sampling_pair = fake_pair
        module.sample = fake_sample

        with patch.object(
            TrainingModule,
            "device",
            new=property(lambda self: torch.device("cpu")),
        ):
            metrics = TrainingModule.sample_compare_harness(module, train=True)

        self.assertEqual(metrics["skipped_no_valid_boundary_revisits"], 7.0)

    def test_summarize_fixed_pair_eval_rows_reports_exact_fractions(self):
        velocity_rows = [
            {
                "index": 0,
                "first_hit_precision": 1.0,
                "first_hit_recall": 1.0,
                "vanish_precision": 1.0,
                "vanish_recall": 1.0,
            },
            {
                "index": 1,
                "first_hit_precision": 1.0,
                "first_hit_recall": 1.0,
                "vanish_precision": 0.5,
                "vanish_recall": 1.0,
            },
        ]
        ar_rows = [
            {"event_index": 0, "exact_match": True},
            {"event_index": 1, "exact_match": False},
            {"event_index": 2, "exact_match": True},
        ]

        metrics = _summarize_fixed_pair_eval_rows(velocity_rows, ar_rows)

        self.assertEqual(metrics["fixed_path_num_velocity_states"], 2.0)
        self.assertEqual(metrics["fixed_path_num_autoregressive_events"], 3.0)
        self.assertEqual(metrics["fixed_path_velocity_first_hit_exact_frac"], 1.0)
        self.assertEqual(metrics["fixed_path_velocity_vanish_exact_frac"], 0.5)
        self.assertEqual(metrics["fixed_path_velocity_joint_exact_frac"], 0.5)
        self.assertAlmostEqual(
            metrics["fixed_path_autoregressive_exact_frac"], 2.0 / 3.0
        )
        self.assertEqual(metrics["fixed_path_first_wrong_velocity_index"], 1.0)
        self.assertEqual(metrics["fixed_path_first_wrong_autoregressive_index"], 1.0)

    def test_get_harness_sampling_pair_infers_full_transition_event_cap(self):
        module = object.__new__(TrainingModule)
        module._cached_harness_sampling_pairs = {}

        start_tree = "((0:0.1,1:0.1):0.1,(2:0.1,3:0.1):0.1);"
        target_tree = "(((0:0.1,1:0.1):0.1,2:0.1):0.1,3:0.1);"

        dataset_split = types.SimpleNamespace(
            overfit_event_prefix_count=-1,
            return_posterior_trees=lambda idx: [target_tree],
            sample_random_tree_with_base=lambda real_tree: (start_tree, start_tree),
            resolve_training_target_tree=lambda random_tree, real_tree, base_start_tree_newick=None: target_tree,
        )
        module.dataset = types.SimpleNamespace(
            dataset_train=dataset_split,
            dataset_val=dataset_split,
        )

        fake_boundary_paths = [{"events": [{} for _ in range(11)]}]
        with patch(
            "run.TrainingModule.return_tree_boundary_merge_paths",
            return_value=fake_boundary_paths,
        ):
            pair = TrainingModule._get_harness_sampling_pair(module, train=True)

        self.assertEqual(pair["max_events"], 11)
        self.assertEqual(pair["start_tree"], start_tree)
        self.assertEqual(pair["target_tree"], target_tree)

    def test_get_harness_sampling_pair_uses_cached_overfit_pair_when_enabled(self):
        module = object.__new__(TrainingModule)
        module._cached_harness_sampling_pairs = {}

        fixed_pair = {
            "random_tree": "((0:0.1,1:0.1):0.1,(2:0.1,3:0.1):0.1);",
            "effective_target_tree": "(((0:0.1,1:0.1):0.1,2:0.1):0.1,3:0.1);",
            "final_labels": [{"labels": []}, {"labels": []}],
        }
        dataset_split = types.SimpleNamespace(
            overfit_fixed_pair=True,
            overfit_event_prefix_count=-1,
            get_overfit_fixed_pair=lambda idx: fixed_pair,
        )
        module.dataset = types.SimpleNamespace(
            dataset_train=dataset_split,
            dataset_val=dataset_split,
        )

        pair = TrainingModule._get_harness_sampling_pair(module, train=True)

        self.assertEqual(pair["start_tree"], fixed_pair["random_tree"])
        self.assertEqual(pair["target_tree"], fixed_pair["effective_target_tree"])
        self.assertEqual(pair["max_events"], 2)

    def test_autoregressive_group_refinement_block_runs(self):
        model = TreeDenoiserTokenGT(
            num_node_types=10,
            num_edge_types=10,
            embed_dim=32,
            n_layers=2,
            n_heads=4,
            output_dim=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            tokenizer_n_layers=2,
            phyla_dim=8,
            autoregressive_head_mode="pairwise_threshold",
            autoregressive_group_refinement_layers=1,
        )
        model.eval()
        newick = "((0:0.1,1:0.1,2:0.1):0.1,3:0.1);"
        groups = [get_structural_polytomy_groups_from_newick(newick)]
        self.assertTrue(groups[0], "Expected a structural polytomy group.")
        tokenized = model.tokenizer([newick])
        phyla_embeddings = torch.zeros((1, 4, 8), dtype=torch.float32)

        with torch.no_grad():
            outputs = model(
                tokenized,
                torch.zeros(1, dtype=torch.float32),
                phyla_embeddings=phyla_embeddings,
                autoregressive=True,
                autoregressive_component_groups=groups,
            )

        self.assertEqual(len(outputs), 1)
        logits = outputs[0]["logits"]
        self.assertEqual(tuple(logits.shape), (3, 3))
        finite_mask = ~torch.eye(3, dtype=torch.bool)
        self.assertTrue(torch.isfinite(logits[finite_mask]).all())

    def test_edge_head_can_return_first_hit_logits(self):
        model = TreeDenoiserTokenGT(
            num_node_types=10,
            num_edge_types=10,
            embed_dim=32,
            n_layers=2,
            n_heads=4,
            output_dim=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            tokenizer_n_layers=2,
            phyla_dim=8,
        )
        model.eval()
        newick = "((0:0.1,1:0.1,2:0.1):0.1,3:0.1);"
        tokenized = model.tokenizer([newick])
        phyla_embeddings = torch.zeros((1, 4, 8), dtype=torch.float32)

        with torch.no_grad():
            velocity, edge_pad_mask, first_hit_logits = model(
                tokenized,
                torch.zeros(1, dtype=torch.float32),
                phyla_embeddings=phyla_embeddings,
                return_leafs_only=False,
                return_edges_only=True,
                return_first_hit_logits=True,
            )

        self.assertEqual(tuple(velocity.shape), tuple(first_hit_logits.shape))
        self.assertEqual(tuple(edge_pad_mask.shape[:2]), tuple(velocity.shape[:2]))

    def test_boundary_event_distribution_loss_prefers_correct_event_order(self):
        lengths = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        y_true = torch.tensor([-1.0, -0.5, -0.25], dtype=torch.float32)
        y_pred_good = torch.tensor([-1.0, -0.45, -0.2], dtype=torch.float32)
        y_pred_bad = torch.tensor([-0.2, -1.0, -0.25], dtype=torch.float32)

        good_loss, good_stats = _boundary_event_distribution_loss(
            lengths=lengths,
            y_true=y_true,
            y_pred=y_pred_good,
            velocity_sign_eps=1e-3,
            dt_eps=1e-6,
            temp=0.5,
            rate_beta=5.0,
            normalize_by_log_candidates=False,
        )
        bad_loss, bad_stats = _boundary_event_distribution_loss(
            lengths=lengths,
            y_true=y_true,
            y_pred=y_pred_bad,
            velocity_sign_eps=1e-3,
            dt_eps=1e-6,
            temp=0.5,
            rate_beta=5.0,
            normalize_by_log_candidates=False,
        )

        self.assertLess(
            float(good_loss.item()),
            float(bad_loss.item()),
            "Boundary-event loss should prefer the velocity prediction with the correct first-hit ordering.",
        )
        self.assertEqual(good_stats["n_candidates"], 3)
        self.assertEqual(good_stats["target_first_size"], 1)
        self.assertGreater(good_stats["pred_first_mass"], bad_stats["pred_first_mass"])
        self.assertEqual(good_stats["top1_hits_first_set"], 1.0)
        self.assertEqual(bad_stats["top1_hits_first_set"], 0.0)

    def test_boundary_event_distribution_loss_penalizes_spurious_wrong_edge(self):
        lengths = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        y_true = torch.tensor([-1.0, -0.5, 0.2], dtype=torch.float32)
        y_pred_good = torch.tensor([-1.0, -0.4, 0.1], dtype=torch.float32)
        y_pred_spurious = torch.tensor([-0.3, -0.2, -2.0], dtype=torch.float32)

        good_loss, good_stats = _boundary_event_distribution_loss(
            lengths=lengths,
            y_true=y_true,
            y_pred=y_pred_good,
            velocity_sign_eps=1e-3,
            dt_eps=1e-6,
            temp=0.5,
            rate_beta=5.0,
            normalize_by_log_candidates=False,
        )
        bad_loss, bad_stats = _boundary_event_distribution_loss(
            lengths=lengths,
            y_true=y_true,
            y_pred=y_pred_spurious,
            velocity_sign_eps=1e-3,
            dt_eps=1e-6,
            temp=0.5,
            rate_beta=5.0,
            normalize_by_log_candidates=False,
        )

        self.assertLess(
            float(good_loss.item()),
            float(bad_loss.item()),
            "Boundary-event loss should penalize mass placed on a truly non-contracting edge.",
        )
        self.assertEqual(good_stats["target_first_size"], 1)
        self.assertEqual(good_stats["top1_hits_first_set"], 1.0)
        self.assertEqual(bad_stats["top1_hits_first_set"], 0.0)

    def test_boundary_event_precision_margin_loss_penalizes_fast_wrong_edge(self):
        lengths = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        y_true = torch.tensor([-1.0, -0.8, 0.2], dtype=torch.float32)
        y_pred_good = torch.tensor([-1.0, -0.7, 0.1], dtype=torch.float32)
        y_pred_bad = torch.tensor([-1.0, -0.7, -2.0], dtype=torch.float32)

        good_loss, good_stats = _boundary_event_precision_margin_loss(
            lengths=lengths,
            y_true=y_true,
            y_pred=y_pred_good,
            velocity_sign_eps=1e-3,
            dt_eps=1e-6,
            temp=0.5,
            rate_beta=5.0,
            margin=0.25,
        )
        bad_loss, bad_stats = _boundary_event_precision_margin_loss(
            lengths=lengths,
            y_true=y_true,
            y_pred=y_pred_bad,
            velocity_sign_eps=1e-3,
            dt_eps=1e-6,
            temp=0.5,
            rate_beta=5.0,
            margin=0.25,
        )

        self.assertLess(
            float(good_loss.item()),
            float(bad_loss.item()),
            "Precision margin loss should penalize a wrong edge outranking the true first-hit set.",
        )
        self.assertGreater(good_stats["margin_gap"], bad_stats["margin_gap"])
        self.assertEqual(good_stats["violated"], 0.0)
        self.assertEqual(bad_stats["violated"], 1.0)

    # def test_overfit_single_velocity_vector(self):
    #     random.seed(123)
    #     torch.manual_seed(123)
    #     device = torch.device("cpu")

    #     model = TreeDenoiserTokenGT(
    #         num_node_types=3,
    #         num_edge_types=2,
    #         embed_dim=64,
    #         n_layers=2,
    #         n_heads=4,
    #         output_dim=1,
    #         dropout=0.0,
    #         attention_dropout=0.0,
    #         activation_dropout=0.0,
    #         drop_path_rate=0.0,
    #         use_performer=False,
    #         performer_nb_features=None,
    #         performer_generalized_attention=False,
    #         layernorm_style="prenorm",
    #         tokenizer_lap_dim=8,
    #         tokenizer_lap_dropout=0.0,
    #         tokenizer_n_layers=2,
    #         phyla_dim=16,
    #     ).to(device)

    #     module = TrainingModule(
    #         model=model,
    #         dataset=MagicMock(),
    #         lr=1e-3,
    #         record=False,
    #         epochs=1,
    #         deepspeed=False,
    #         logger=None,
    #         velocity_loss_mode="plain",
    #         velocity_sign_eps=1e-3,
    #         velocity_event_weight = 0.0,
    #         verbose = True
    #     ).to(device)

    #     batch = _make_single_velocity_batch(
    #         tokenizer=model.tokenizer,
    #         n_leaves=10,
    #         seed=2024,
    #     )
    #     bootstrap_metrics = _velocity_metrics(module, batch, topk=3)
    #     self.assertGreaterEqual(
    #         bootstrap_metrics["n_supervised_edges"],
    #         6,
    #         "Not enough supervised internal edges to run a robust velocity-overfit sanity check.",
    #     )

    #     initial = _velocity_metrics(module, batch, topk=3)

    #     optimizer = torch.optim.Adam(module.model.parameters(), lr=5e-3)
    #     best_metrics = dict(initial)
    #     best_state = copy.deepcopy(module.model.state_dict())
    #     max_steps = 1000

    #     for step in range(max_steps):
    #         module.train()
    #         optimizer.zero_grad(set_to_none=True)
    #         logs = module.step(batch, autoregressive=False)
    #         loss = logs["loss"]
    #         self.assertTrue(torch.isfinite(loss).item(), "Training loss became non-finite.")
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(module.model.parameters(), max_norm=1.0)
    #         optimizer.step()

    #         if (step + 1) % 10 == 0:
    #             probe = _velocity_metrics(module, batch, topk=3)
    #             has_strong_corr_probe = (
    #                 probe["cosine"] > 0.95
    #                 and probe["pearson"] > 0.95
    #                 and probe["spearman"] > 0.95
    #             )
    #             has_strong_corr_best = (
    #                 best_metrics["cosine"] > 0.95
    #                 and best_metrics["pearson"] > 0.95
    #                 and best_metrics["spearman"] > 0.95
    #             )
    #             if has_strong_corr_probe and has_strong_corr_best:
    #                 if (
    #                     probe["dt_first_hit_recall"] > best_metrics["dt_first_hit_recall"]
    #                     or (
    #                         probe["dt_first_hit_recall"] == best_metrics["dt_first_hit_recall"]
    #                         and (
    #                             probe["dt_hit_rel_err"] < best_metrics["dt_hit_rel_err"]
    #                             or (
    #                                 probe["dt_hit_rel_err"] == best_metrics["dt_hit_rel_err"]
    #                                 and (
    #                                     probe["dt_topk_overlap"] > best_metrics["dt_topk_overlap"]
    #                                     or (
    #                                         probe["dt_topk_overlap"] == best_metrics["dt_topk_overlap"]
    #                                         and probe["cosine"] >= best_metrics["cosine"]
    #                                     )
    #                                 )
    #                             )
    #                         )
    #                     )
    #                 ):
    #                     best_metrics = dict(probe)
    #                     best_state = copy.deepcopy(module.model.state_dict())
    #             elif (
    #                 probe["spearman"] > best_metrics["spearman"]
    #                 or (
    #                     probe["spearman"] == best_metrics["spearman"]
    #                     and (
    #                         probe["cosine"] > best_metrics["cosine"]
    #                         or (
    #                             probe["cosine"] == best_metrics["cosine"]
    #                             and probe["dt_first_hit_recall"] >= best_metrics["dt_first_hit_recall"]
    #                         )
    #                     )
    #                 )
    #             ):
    #                 best_metrics = dict(probe)
    #                 best_state = copy.deepcopy(module.model.state_dict())
    #             if (
    #                 probe["cosine"] > 0.99
    #                 and probe["pearson"] > 0.99
    #                 and probe["topk_overlap"] == 1.0
    #                 and probe["sign_acc"] > 0.90
    #                 and probe["dt_hit_rel_err"] < 0.15
    #                 and probe["dt_neg_jaccard"] >= 0.90
    #                 and probe["dt_first_hit_recall"] == 1.0
    #                 and probe["dt_topk_overlap"] >= 0.67
    #             ):
    #                 break

    #     module.model.load_state_dict(best_state)
    #     final = _velocity_metrics(module, batch, topk=3)
    #     print(final)

    #     self.assertLess(
    #         final["mse"],
    #         initial["mse"],
    #         "Overfit sanity check did not reduce velocity MSE.",
    #     )
    #     self.assertLess(
    #         final["mse"],
    #         max(3e-3, initial["mse"] * 0.1),
    #         f"MSE did not improve enough (initial={initial['mse']:.6f}, final={final['mse']:.6f})",
    #     )
    #     self.assertGreater(
    #         final["cosine"], 0.95, f"Cosine similarity too low: {final['cosine']:.6f}"
    #     )
    #     self.assertGreater(
    #         final["pearson"], 0.95, f"Pearson correlation too low: {final['pearson']:.6f}"
    #     )
    #     self.assertGreater(
    #         final["spearman"],
    #         0.95,
    #         f"Spearman correlation too low: {final['spearman']:.6f}",
    #     )
    #     self.assertGreater(
    #         final["sign_acc"], 0.95, f"Sign accuracy too low: {final['sign_acc']:.6f}"
    #     )
    #     self.assertEqual(
    #         final["topk_overlap"],
    #         1.0,
    #         f"Top-k velocity mask overlap not perfect: {final['topk_overlap']:.3f}",
    #     )
    #     self.assertLess(
    #         final["dt_hit_rel_err"],
    #         0.15,
    #         (
    #             f"dt_hit mismatch too large "
    #             f"(pred={final['dt_hit_pred']:.6e}, true={final['dt_hit_true']:.6e}, rel_err={final['dt_hit_rel_err']:.6f})"
    #         ),
    #     )
    #     self.assertGreaterEqual(
    #         final["dt_neg_jaccard"],
    #         0.90,
    #         f"Negative-velocity edge mismatch is too high (Jaccard={final['dt_neg_jaccard']:.3f})",
    #     )
    #     self.assertEqual(
    #         final["dt_first_hit_recall"],
    #         1.0,
    #         (
    #             "Did not recapitulate all true first-hit edge masks within the dt tolerance "
    #             f"{final['dt_first_hit_tol']:.2f} "
    #             f"(pred={sorted(final['pred_first_masks'])}, true={sorted(final['true_first_masks'])}, "
    #             f"missed={_format_first_hit_miss_details(final)})."
    #         ),
    #     )
    #     self.assertGreaterEqual(
    #         final["dt_topk_overlap"],
    #         0.66,
    #         f"dt candidate top-k overlap too low: {final['dt_topk_overlap']:.3f}",
    #     )

    # @patch.object(TreeDataset, "build_index", return_value=None)
    # def test_random_sanity_check_is_deterministic_for_tree_and_velocity(
    #     self, _mock_build_index
    # ):
    #     ds = TreeDataset(
    #         nexus_root="mock",
    #         mrbayes_root="mock",
    #         random_sanity_check=True,
    #         overfit_velocity_zero=True,
    #     )

    #     # random_sanity_check enforces fixed tree sources regardless tfiles input.
    #     real_one = ds.load_posterior_trees_from_tfiles([])[0]
    #     real_two = ds.load_posterior_trees_from_tfiles([])[0]
    #     self.assertEqual(real_one, real_two)

    #     rand_one = ds.sample_random_tree(real_one)
    #     rand_two = ds.sample_random_tree(real_one)
    #     self.assertEqual(rand_one, rand_two)

    #     # overfit_velocity_zero uses t=0.0, so sampled tree/velocity should be stable.
    #     sample_newick_1, velocity_1 = return_sampled_tree_orthant_velocity(
    #         rand_one, real_one, 0.0
    #     )
    #     sample_newick_2, velocity_2 = return_sampled_tree_orthant_velocity(
    #         rand_two, real_two, 0.0
    #     )

    #     self.assertEqual(sample_newick_1, sample_newick_2)
    #     self.assertEqual(set(velocity_1.keys()), set(velocity_2.keys()))
    #     for k in velocity_1:
    #         self.assertAlmostEqual(
    #             float(velocity_1[k]),
    #             float(velocity_2[k]),
    #             places=8,
    #             msg=f"Velocity mismatch for split {k}",
    #         )

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_overfit_velocity_on_random_sanity_tree_pair(self, _mock_build_index):
        random.seed(321)
        torch.manual_seed(321)
        device = torch.device("cpu")

        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_velocity_zero=True,
        )

        real_tree = ds.load_posterior_trees_from_tfiles([])[0]
        random_tree = ds.sample_random_tree(real_tree)
        # random_tree_small, real_tree_small = _prune_and_renumber_tree_pair(
        #     random_tree, real_tree, keep_leaves=8
        # )

        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=64,
            n_layers=2,
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
            tokenizer_lap_dim=8,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=2,
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
            velocity_loss_mode="plain",
            velocity_sign_eps=1e-3,
            velocity_event_weight = 0.0,
            verbose=True,
        ).to(device)

        batch = _make_batch_from_tree_pair(
            tokenizer=model.tokenizer,
            start_tree=random_tree,
            target_tree=real_tree,
            time_point=0.0,
        )

        bootstrap_metrics = _velocity_metrics(module, batch, topk=3)
        self.assertGreaterEqual(
            bootstrap_metrics["n_supervised_edges"],
            3,
            "Not enough supervised internal edges in random_sanity_check pair.",
        )

        initial = _velocity_metrics(module, batch, topk=3)
        optimizer = torch.optim.Adam(module.model.parameters(), lr=5e-3)
        best_metrics = dict(initial)
        best_state = copy.deepcopy(module.model.state_dict())

        for step in range(1000):
            module.train()
            optimizer.zero_grad(set_to_none=True)
            logs = module.step(batch, autoregressive=False)
            loss = logs["loss"]
            self.assertTrue(torch.isfinite(loss).item(), "Training loss became non-finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.model.parameters(), max_norm=1.0)
            optimizer.step()

            if (step + 1) % 10 == 0:
                probe = _velocity_metrics(module, batch, topk=3)
                has_strong_corr_probe = (
                    probe["cosine"] > 0.95
                    and probe["pearson"] > 0.95
                    and probe["spearman"] > 0.95
                )
                has_strong_corr_best = (
                    best_metrics["cosine"] > 0.95
                    and best_metrics["pearson"] > 0.95
                    and best_metrics["spearman"] > 0.95
                )
                if has_strong_corr_probe and has_strong_corr_best:
                    if (
                        probe["dt_first_hit_recall"] > best_metrics["dt_first_hit_recall"]
                        or (
                            probe["dt_first_hit_recall"] == best_metrics["dt_first_hit_recall"]
                            and (
                                probe["dt_hit_rel_err"] < best_metrics["dt_hit_rel_err"]
                                or (
                                    probe["dt_hit_rel_err"] == best_metrics["dt_hit_rel_err"]
                                    and (
                                        probe["dt_topk_overlap"] > best_metrics["dt_topk_overlap"]
                                        or (
                                            probe["dt_topk_overlap"] == best_metrics["dt_topk_overlap"]
                                            and probe["cosine"] >= best_metrics["cosine"]
                                        )
                                    )
                                )
                            )
                        )
                    ):
                        best_metrics = dict(probe)
                        best_state = copy.deepcopy(module.model.state_dict())
                elif (
                    probe["spearman"] > best_metrics["spearman"]
                    or (
                        probe["spearman"] == best_metrics["spearman"]
                        and (
                            probe["cosine"] > best_metrics["cosine"]
                            or (
                                probe["cosine"] == best_metrics["cosine"]
                                and probe["dt_first_hit_recall"] >= best_metrics["dt_first_hit_recall"]
                            )
                        )
                    )
                ):
                    best_metrics = dict(probe)
                    best_state = copy.deepcopy(module.model.state_dict())
                if (
                    probe["cosine"] > 0.99
                    and probe["pearson"] > 0.99
                    and probe["spearman"] > 0.95
                    and probe["topk_overlap"] == 1.0
                    and probe["sign_acc"] > 0.90
                    and probe["dt_hit_rel_err"] < 0.20
                    and probe["dt_neg_jaccard"] >= 0.85
                    and probe["dt_first_hit_recall"] == 1.0
                    and probe["dt_topk_overlap"] >= 0.67
                ):
                    break

        module.model.load_state_dict(best_state)
        final = _velocity_metrics(module, batch, topk=3)
        print(final)

        self.assertLess(final["mse"], initial["mse"])
        self.assertLess(
            final["mse"],
            max(3e-3, initial["mse"] * 0.1),
            f"MSE did not improve enough (initial={initial['mse']:.6f}, final={final['mse']:.6f})",
        )
        self.assertGreater(final["cosine"], 0.90)
        self.assertGreater(final["pearson"], 0.90)
        self.assertGreater(final["spearman"], 0.90)
        self.assertGreater(final["sign_acc"], 0.85)
        self.assertEqual(final["topk_overlap"], 1.0)
        self.assertLess(
            final["dt_hit_rel_err"],
            0.20,
            (
                f"dt_hit mismatch too large "
                f"(pred={final['dt_hit_pred']:.6e}, true={final['dt_hit_true']:.6e}, rel_err={final['dt_hit_rel_err']:.6f})"
            ),
        )
        self.assertGreaterEqual(
            final["dt_neg_jaccard"],
            0.85,
            f"Negative-velocity edge mismatch is too high (Jaccard={final['dt_neg_jaccard']:.3f})",
        )

        self.assertEqual(
            final["dt_first_hit_recall"],
            1.0,
            (
                "Did not recapitulate all true first-hit edge masks within the dt tolerance "
                f"{final['dt_first_hit_tol']:.2f} "
                f"(pred={sorted(final['pred_first_masks'])}, true={sorted(final['true_first_masks'])}, "
                f"missed={_format_first_hit_miss_details(final)})."
            ),
        )
        self.assertGreaterEqual(
            final["dt_topk_overlap"],
            0.67,
            f"dt candidate top-k overlap too low: {final['dt_topk_overlap']:.3f}",
        )

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_training_step_with_autoregressive_still_converges_velocity(
        self, _mock_build_index
    ):
        random.seed(777)
        torch.manual_seed(777)
        device = torch.device("cpu")

        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_velocity_zero=True,
        )
        real_tree = ds.load_posterior_trees_from_tfiles([])[0]
        random_tree = ds.sample_random_tree(real_tree)

        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=64,
            n_layers=2,
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
            tokenizer_lap_dim=8,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=2,
            phyla_dim=16,
        ).to(device)

        dataset_stub = MagicMock()
        dataset_stub.msa_distance = True
        dataset_stub.chosen_tree = (0, 0, 1)

        module = TrainingModule(
            model=model,
            dataset=dataset_stub,
            lr=1e-3,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            velocity_loss_mode="plain",
            velocity_sign_eps=1e-3,
            #I already weighed it by 100 in the loss, so this would make it 1000 which is a bit much
            training_step_velocity_weight=1,
            training_step_autoregressive_weight=0.1,
            velocity_dt_candidate_weight = 1,
            velocity_dt_hit_weight = 1,
            velocity_event_weight = 0.0,
        ).to(device)

        batch = _make_batch_from_tree_pair_with_autoregressive(
            tokenizer=model.tokenizer,
            start_tree=random_tree,
            target_tree=real_tree,
            time_point=0.0,
        )
        dataset_stub.chosen_tree = (0, int(batch["num_leaves"][0]), 1)

        bootstrap_metrics = _velocity_metrics(module, batch, topk=3)
        self.assertGreaterEqual(
            bootstrap_metrics["n_supervised_edges"],
            3,
            "Not enough supervised internal edges in random_sanity_check pair.",
        )

        initial = _velocity_metrics(module, batch, topk=3)
        best_metrics = dict(initial)
        best_state = copy.deepcopy(module.model.state_dict())

        optimizer = torch.optim.Adam(module.model.parameters(), lr=5e-3)
        module.optimizers = MagicMock(return_value=_OptimizerProxy(optimizer))
        module.manual_backward = lambda loss: loss.backward()
        module.clip_gradients = lambda _opt, gradient_clip_val, gradient_clip_algorithm: torch.nn.utils.clip_grad_norm_(
            module.model.parameters(), max_norm=float(gradient_clip_val)
        )
        module.log = MagicMock()

        with patch.object(torch.Tensor, "to", _tensor_to_cpu_for_cuda):
            for step in range(1000):
                module.train()
                loss = module.training_step(batch, step)
                self.assertTrue(
                    torch.isfinite(loss).item(),
                    "training_step loss became non-finite.",
                )

                if (step + 1) % 5 == 0:
                    probe = _velocity_metrics(module, batch, topk=3)
                    has_strong_corr_probe = (
                        probe["cosine"] > 0.90
                        and probe["pearson"] > 0.90
                        and probe["spearman"] > 0.90
                    )
                    has_strong_corr_best = (
                        best_metrics["cosine"] > 0.90
                        and best_metrics["pearson"] > 0.90
                        and best_metrics["spearman"] > 0.90
                    )
                    if has_strong_corr_probe and has_strong_corr_best:
                        if (
                            probe["dt_first_hit_recall"] > best_metrics["dt_first_hit_recall"]
                            or (
                                probe["dt_first_hit_recall"] == best_metrics["dt_first_hit_recall"]
                                and (
                                    probe["dt_hit_rel_err"] < best_metrics["dt_hit_rel_err"]
                                    or (
                                        probe["dt_hit_rel_err"] == best_metrics["dt_hit_rel_err"]
                                        and (
                                            probe["dt_topk_overlap"] > best_metrics["dt_topk_overlap"]
                                            or (
                                                probe["dt_topk_overlap"] == best_metrics["dt_topk_overlap"]
                                                and probe["mse"] <= best_metrics["mse"]
                                            )
                                        )
                                    )
                                )
                            )
                        ):
                            best_metrics = dict(probe)
                            best_state = copy.deepcopy(module.model.state_dict())
                    elif (
                        probe["mse"] < best_metrics["mse"]
                        or (
                            probe["mse"] == best_metrics["mse"]
                            and (
                                probe["cosine"] > best_metrics["cosine"]
                                or (
                                    probe["cosine"] == best_metrics["cosine"]
                                    and probe["dt_first_hit_recall"] >= best_metrics["dt_first_hit_recall"]
                                )
                            )
                        )
                    ):
                        best_metrics = dict(probe)
                        best_state = copy.deepcopy(module.model.state_dict())

                    if (
                        probe["cosine"] > 0.95
                        and probe["pearson"] > 0.95
                        and probe["spearman"] > 0.90
                        and probe["topk_overlap"] == 1.0
                        and probe["sign_acc"] > 0.85
                        and probe["dt_hit_rel_err"] < 0.30
                        and probe["dt_neg_jaccard"] >= 0.75
                        and probe["dt_first_hit_recall"] == 1.0
                        and probe["dt_topk_overlap"] >= 0.50
                    ):
                        break

        module.model.load_state_dict(best_state)
        final = _velocity_metrics(module, batch, topk=3)
        print(final)

        self.assertLess(final["mse"], initial["mse"])
        self.assertLess(
            final["mse"],
            max(2e-2, initial["mse"] * 0.25),
            f"MSE did not improve enough (initial={initial['mse']:.6f}, final={final['mse']:.6f})",
        )
        self.assertGreater(final["cosine"], 0.90)
        self.assertGreater(final["pearson"], 0.90)
        self.assertGreater(final["spearman"], 0.90)
        self.assertGreater(final["sign_acc"], 0.80)
        self.assertEqual(final["topk_overlap"], 1.0)
        self.assertLess(
            final["dt_hit_rel_err"],
            0.30,
            (
                f"dt_hit mismatch too large "
                f"(pred={final['dt_hit_pred']:.6e}, true={final['dt_hit_true']:.6e}, rel_err={final['dt_hit_rel_err']:.6f})"
            ),
        )
        self.assertGreaterEqual(
            final["dt_neg_jaccard"],
            0.75,
            f"Negative-velocity edge mismatch is too high (Jaccard={final['dt_neg_jaccard']:.3f})",
        )
        self.assertGreaterEqual(
            final["dt_topk_overlap"],
            0.50,
            f"dt candidate top-k overlap too low: {final['dt_topk_overlap']:.3f}",
        )

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_autoregressive_can_overfit_one_nonbinary_boundary_sequence(
        self, _mock_build_index
    ):
        random.seed(777)
        torch.manual_seed(777)
        device = torch.device("cpu")

        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_velocity_zero=True,
        )
        real_tree = ds.load_posterior_trees_from_tfiles([])[0]
        random_tree = ds.sample_random_tree(real_tree)
        boundary_path = _select_random_nonbinary_boundary_path(
            random_tree,
            real_tree,
            seed=777,
        )

        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=64,
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
            autoregressive_head_mode="structured_subset",
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
        module.autoregressive_allow_multi_subset_targets = True

        event_batches = [
            _make_autoregressive_event_batch(
                model.tokenizer,
                event["newick"],
                event["labels"],
                _normalized_event_time(event_idx, len(boundary_path["events"])),
            )
            for event_idx, event in enumerate(boundary_path["events"])
        ]

        optimizer = torch.optim.Adam(module.model.parameters(), lr=5e-3)

        def evaluate_state(include_sample=False):
            module.eval()
            event_details = []
            exact_events_ok = True
            for event_idx, event in enumerate(boundary_path["events"]):
                predicted, _ = _predict_autoregressive_event(
                    module,
                    model.tokenizer,
                    event["newick"],
                    _normalized_event_time(event_idx, len(boundary_path["events"])),
                )
                target = _target_merge_subsets_for_event(event["labels"])
                if predicted != target:
                    exact_events_ok = False
                    event_details.append(
                        f"event {event_idx}: predicted={predicted} target={target}"
                    )

            rollout_ok = False
            rollout_error = None
            try:
                _rollout_boundary_with_autoregressive_head(
                    self,
                    module,
                    model.tokenizer,
                    boundary_path,
                    random_tree,
                    real_tree,
                )
                rollout_ok = True
            except AssertionError as exc:
                rollout_error = str(exc)

            return {
                "exact_events_ok": exact_events_ok,
                "event_details": event_details,
                "rollout_ok": rollout_ok,
                "rollout_error": rollout_error,
            }

        best_state = copy.deepcopy(module.model.state_dict())
        best_eval = {
            "exact_events_ok": False,
            "event_details": ["evaluation not run"],
            "rollout_ok": False,
            "rollout_error": "evaluation not run",
        }
        best_score = (-1, -1)

        max_steps = 120
        for step in range(max_steps):
            module.train()
            optimizer.zero_grad()
            total_loss = None
            for batch in event_batches:
                logs = module.step(batch, autoregressive=True)
                total_loss = (
                    logs["loss"]
                    if total_loss is None
                    else total_loss + logs["loss"]
                )

            total_loss.backward()
            optimizer.step()

            if (step + 1) % 10 == 0 or step == 0:
                current_eval = evaluate_state()
                score = (
                    sum(
                        1
                        for event_idx, event in enumerate(boundary_path["events"])
                        if _target_merge_subsets_for_event(event["labels"])
                        == _predict_autoregressive_event(
                            module,
                            model.tokenizer,
                            event["newick"],
                            _normalized_event_time(event_idx, len(boundary_path["events"])),
                        )[0]
                    ),
                    int(current_eval["rollout_ok"]),
                )
                if score > best_score:
                    best_score = score
                    best_eval = current_eval
                    best_state = copy.deepcopy(module.model.state_dict())
                if current_eval["exact_events_ok"] and current_eval["rollout_ok"]:
                    break

        module.model.load_state_dict(best_state)
        final_eval = evaluate_state()

        self.assertTrue(
            final_eval["exact_events_ok"],
            "Autoregressive head did not overfit the selected non-binary boundary events. "
            + " | ".join(final_eval["event_details"]),
        )
        self.assertTrue(
            final_eval["rollout_ok"],
            "Autoregressive rollout did not reproduce the selected boundary sequence. "
            + str(final_eval["rollout_error"]),
        )

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_velocity_and_autoregressive_can_overfit_first_boundary_transition(
        self, _mock_build_index
    ):
        random.seed(777)
        torch.manual_seed(777)
        device = torch.device("cpu")

        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_velocity_zero=True,
        )
        real_tree = ds.load_posterior_trees_from_tfiles([])[0]
        random_tree = ds.sample_random_tree(real_tree)
        random_tree, real_tree = _prune_and_renumber_tree_pair(
            random_tree,
            real_tree,
            keep_leaves=12,
        )
        boundary_path = return_tree_boundary_merge_paths(
            random_tree,
            real_tree,
            legacy_training_semantics=True,
        )[0]
        self.assertTrue(
            boundary_path["events"],
            "The first boundary on the sanity tree pair had no merge events.",
        )

        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=64,
            n_layers=2,
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
            tokenizer_lap_dim=8,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=2,
            phyla_dim=16,
            autoregressive_head_mode="structured_subset",
            autoregressive_group_refinement_layers=1,
        ).to(device)

        dataset_stub = MagicMock()
        dataset_stub.msa_distance = True
        dataset_stub.chosen_tree = (0, 0, 1)

        module = TrainingModule(
            model=model,
            dataset=dataset_stub,
            lr=1e-3,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            velocity_loss_mode="plain",
            velocity_sign_eps=1e-3,
            training_step_velocity_weight=1,
            training_step_autoregressive_weight=0.1,
            velocity_dt_candidate_weight=1,
            velocity_dt_hit_weight=1,
            velocity_event_weight=0.0,
            velocity_first_hit_head_weight=1.0,
            velocity_first_hit_head_use_at_sampling=True,
            velocity_first_hit_predictor_mode="edge_token_attention",
        ).to(device)
        module.legacy_first_hit_gather_only = True
        module.autoregressive_allow_multi_subset_targets = True

        velocity_batch = _make_batch_from_tree_pair(
            tokenizer=model.tokenizer,
            start_tree=random_tree,
            target_tree=real_tree,
            time_point=0.0,
        )
        dataset_stub.chosen_tree = (0, int(velocity_batch["num_leaves"][0]), 1)

        initial_velocity = _velocity_metrics(module, velocity_batch, topk=3)
        self.assertTrue(
            math.isfinite(initial_velocity["dt_hit_true"]),
            "The sanity tree pair did not have a finite first boundary hit time.",
        )

        event_batches = [
            _make_autoregressive_event_batch(
                model.tokenizer,
                event["newick"],
                event["labels"],
                _normalized_event_time(event_idx, len(boundary_path["events"])),
            )
            for event_idx, event in enumerate(boundary_path["events"])
        ]

        optimizer = torch.optim.Adam(module.model.parameters(), lr=5e-3)

        def evaluate_state(include_sample=False):
            module.eval()
            velocity_eval = _velocity_metrics(module, velocity_batch, topk=3)

            exact_events_ok = True
            event_details = []
            for event_idx, event in enumerate(boundary_path["events"]):
                predicted, _ = _predict_autoregressive_event(
                    module,
                    model.tokenizer,
                    event["newick"],
                    _normalized_event_time(event_idx, len(boundary_path["events"])),
                )
                target = _target_merge_subsets_for_event(event["labels"])
                if predicted != target:
                    exact_events_ok = False
                    event_details.append(
                        f"event {event_idx}: predicted={predicted} target={target}"
                    )

            sample_eval = _sample_to_first_boundary(
                module,
                random_tree,
                initial_velocity["dt_hit_true"],
                boundary_path,
            )
            return {
                "velocity": velocity_eval,
                "exact_events_ok": exact_events_ok,
                "event_details": event_details,
                "sample": sample_eval,
            }

        def score_state(eval_state):
            velocity_eval = eval_state["velocity"]
            rf_zero = int(eval_state["sample"]["rf_norm"] == 0.0)
            exact_events = int(eval_state["exact_events_ok"])
            dt_recall = float(velocity_eval["dt_first_hit_recall"])
            dt_rel_err = velocity_eval["dt_hit_rel_err"]
            mse = velocity_eval["mse"]
            dt_rel_err_score = -dt_rel_err if math.isfinite(dt_rel_err) else float("-inf")
            return (rf_zero, exact_events, dt_recall, dt_rel_err_score, -mse)

        best_state = copy.deepcopy(module.model.state_dict())
        best_eval = None
        best_score = (-1, -1, -1.0, float("-inf"), float("-inf"))
        max_steps = 400

        for step in range(max_steps):
            module.train()
            optimizer.zero_grad()

            velocity_logs = module.step(velocity_batch, autoregressive=False)
            autoregressive_losses = [
                module.step(batch, autoregressive=True)["loss"]
                for batch in event_batches
            ]
            autoregressive_loss = torch.stack(autoregressive_losses).mean()
            total_loss = velocity_logs["loss"] + 0.1 * autoregressive_loss
            total_loss.backward()
            optimizer.step()

            if (step + 1) % 10 == 0 or step == 0:
                current_eval = evaluate_state()
                current_score = score_state(current_eval)
                if current_score > best_score:
                    best_score = current_score
                    best_eval = current_eval
                    best_state = copy.deepcopy(module.model.state_dict())

                if (
                    current_eval["sample"]["rf_norm"] == 0.0
                    and current_eval["exact_events_ok"]
                    and current_eval["velocity"]["dt_first_hit_recall"] == 1.0
                    and current_eval["velocity"]["dt_hit_rel_err"] < 0.30
                ):
                    break

        module.model.load_state_dict(best_state)
        if best_eval is None:
            best_eval = evaluate_state()
        final_eval = evaluate_state()

        self.assertLess(
            final_eval["velocity"]["mse"],
            initial_velocity["mse"],
            "Velocity supervision did not improve on the first-boundary sanity pair.",
        )
        self.assertEqual(
            final_eval["velocity"]["dt_first_hit_recall"],
            1.0,
            (
                "The trained velocity head did not recover the full first-hit set at t=0. "
                f"pred={sorted(final_eval['velocity']['pred_first_masks'])}, "
                f"true={sorted(final_eval['velocity']['true_first_masks'])}"
            ),
        )
        self.assertLess(
            final_eval["velocity"]["dt_hit_rel_err"],
            0.30,
            (
                "The trained velocity head did not localize the first boundary accurately enough. "
                f"pred={final_eval['velocity']['dt_hit_pred']:.6e}, "
                f"true={final_eval['velocity']['dt_hit_true']:.6e}, "
                f"rel_err={final_eval['velocity']['dt_hit_rel_err']:.6f}"
            ),
        )
        self.assertTrue(
            final_eval["exact_events_ok"],
            "Autoregressive supervision did not overfit the first boundary events. "
            + " | ".join(final_eval["event_details"]),
        )
        self.assertEqual(
            final_eval["sample"]["rf_norm"],
            0.0,
            (
                "Sampling to the true first-hit time did not recover the oracle post-boundary tree. "
                f"rf_norm={final_eval['sample']['rf_norm']:.6f}, "
                f"sampled={final_eval['sample']['sampled_tree']}, "
                f"target={final_eval['sample']['target_tree']}"
            ),
        )

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_oracle_training_topology_keys_cover_full_boundary_path(
        self, _mock_build_index
    ):
        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_velocity_zero=True,
        )
        target_tree = ds.load_posterior_trees_from_tfiles([])[0]
        start_tree = ds.sample_random_tree(target_tree)

        oracle_keys = set(_oracle_training_topology_keys(start_tree, target_tree))
        boundary_paths = return_tree_boundary_merge_paths(
            start_tree,
            target_tree,
            legacy_training_semantics=True,
        )

        self.assertGreater(
            len(boundary_paths),
            0,
            "Sanity tree pair did not expose any boundary paths.",
        )

        missing = []
        for boundary_path in boundary_paths:
            candidate_newicks = [boundary_path["start_newick"], boundary_path["end_newick"]]
            candidate_newicks.extend(
                event["newick"] for event in boundary_path["events"]
            )
            for newick in candidate_newicks:
                topo_key = _topology_key(newick)
                if topo_key not in oracle_keys:
                    missing.append(newick)

        self.assertEqual(
            missing,
            [],
            "Oracle topology whitelist missed valid boundary-path states.",
        )

    def test_separate_optimizer_steps_reject_grad_ratio(self):
        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=32,
            n_layers=1,
            n_heads=2,
            output_dim=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            performer_nb_features=None,
            performer_generalized_attention=False,
            layernorm_style="prenorm",
            tokenizer_lap_dim=8,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=1,
            phyla_dim=16,
        )

        with self.assertRaisesRegex(
            ValueError,
            "training_step_separate_optimizer_steps cannot be combined",
        ):
            TrainingModule(
                model=model,
                dataset=MagicMock(),
                lr=1e-3,
                record=False,
                epochs=1,
                deepspeed=False,
                logger=None,
                training_step_autoregressive_grad_ratio=0.05,
                training_step_separate_optimizer_steps=True,
            )

    def test_rollout_replay_rejects_negative_weights(self):
        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=32,
            n_layers=1,
            n_heads=2,
            output_dim=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            performer_nb_features=None,
            performer_generalized_attention=False,
            layernorm_style="prenorm",
            tokenizer_lap_dim=8,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=1,
            phyla_dim=16,
        )

        with self.assertRaisesRegex(
            ValueError,
            "rollout_replay_velocity_weight must be non-negative",
        ):
            TrainingModule(
                model=model,
                dataset=MagicMock(),
                lr=1e-3,
                record=False,
                epochs=1,
                deepspeed=False,
                logger=None,
                rollout_replay_velocity_weight=-0.1,
            )

    def test_velocity_replay_batch_uses_module_device_and_marks_skip(self):
        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=32,
            n_layers=1,
            n_heads=2,
            output_dim=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            performer_nb_features=None,
            performer_generalized_attention=False,
            layernorm_style="prenorm",
            tokenizer_lap_dim=8,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=1,
            phyla_dim=16,
        )
        module = TrainingModule(
            model=model,
            dataset=MagicMock(),
            lr=1e-3,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
        )

        batch = _build_velocity_replay_batch(
            module,
            [
                {
                    "newick_tree": "((1:0.1,2:0.1):0.1,(3:0.1,4:0.1):0.1);",
                    "target_tree": "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);",
                    "velocity": {3: -0.1},
                    "velocity_next_boundary_tree": "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);",
                    "timepoint": 0.25,
                    "num_leaves": 4,
                }
            ],
        )

        self.assertTrue(batch["_is_replay_batch"])
        self.assertTrue(batch["_skip_training_augmentations"])
        self.assertEqual(batch["batched_time"].device.type, module.device.type)

    def test_autoregressive_replay_batch_uses_module_device_and_marks_skip(self):
        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=32,
            n_layers=1,
            n_heads=2,
            output_dim=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            performer_nb_features=None,
            performer_generalized_attention=False,
            layernorm_style="prenorm",
            tokenizer_lap_dim=8,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=1,
            phyla_dim=16,
        )
        module = TrainingModule(
            model=model,
            dataset=MagicMock(),
            lr=1e-3,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
        )

        batch = _build_autoregressive_replay_batch(
            module,
            [
                {
                    "newick": "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);",
                    "target_tree": "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);",
                    "labels": [{"components": [1, 2, 4], "merge_indices": [0, 1]}],
                    "stop_after_merge": True,
                    "time": 0.5,
                }
            ],
        )

        self.assertTrue(batch["_is_replay_batch"])
        self.assertTrue(batch["_skip_training_augmentations"])
        self.assertEqual(
            batch["batched_autoregressive_time"].device.type,
            module.device.type,
        )
        self.assertEqual(
            batch["batched_autoregressive_stop_after_merge"].device.type,
            module.device.type,
        )

    def test_prepare_training_batches_respect_skip_flag(self):
        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=32,
            n_layers=1,
            n_heads=2,
            output_dim=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            performer_nb_features=None,
            performer_generalized_attention=False,
            layernorm_style="prenorm",
            tokenizer_lap_dim=8,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=1,
            phyla_dim=16,
        )
        module = TrainingModule(
            model=model,
            dataset=MagicMock(),
            lr=1e-3,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            velocity_length_jitter_prob=1.0,
            velocity_length_jitter_scale=0.5,
            autoregressive_rollin_prob=1.0,
        )
        replay_batch = {"_skip_training_augmentations": True}

        updated_velocity_batch, velocity_stats = module._prepare_velocity_training_batch(
            replay_batch
        )
        updated_ar_batch, ar_stats = module._prepare_autoregressive_training_batch(
            replay_batch
        )

        self.assertIs(updated_velocity_batch, replay_batch)
        self.assertEqual(velocity_stats["attempted"], 0.0)
        self.assertEqual(velocity_stats["applied"], 0.0)
        self.assertIs(updated_ar_batch, replay_batch)
        self.assertEqual(ar_stats["rollin_attempted"], 0.0)
        self.assertEqual(ar_stats["dagger_attempted"], 0.0)
        self.assertEqual(ar_stats["structure_perturb_attempted"], 0.0)

    def test_collect_rollout_replay_batches_dedupes_sampled_states(self):
        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=32,
            n_layers=1,
            n_heads=2,
            output_dim=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            performer_nb_features=None,
            performer_generalized_attention=False,
            layernorm_style="prenorm",
            tokenizer_lap_dim=8,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=1,
            phyla_dim=16,
        )
        dataset = MagicMock()
        dataset.dataset_train = types.SimpleNamespace(
            overfit_split_multi_subset_events=True
        )
        module = TrainingModule(
            model=model,
            dataset=dataset,
            lr=1e-3,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            rollout_replay_velocity_weight=0.5,
            rollout_replay_autoregressive_weight=0.5,
            rollout_replay_anchor_states=2,
            rollout_replay_oracle_horizon=1,
            rollout_replay_max_velocity_states=2,
            rollout_replay_max_autoregressive_states=1,
        )

        module._get_harness_sampling_pair = MagicMock(
            return_value={
                "start_tree": "((1:0.1,2:0.1):0.1,(3:0.1,4:0.1):0.1);",
                "target_tree": "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);",
                "n_leaves": 4,
                "max_events": 4,
            }
        )
        module.sample = MagicMock(
            return_value=(
                ["(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);"],
                0,
                0.0,
                0.0,
                0,
                {
                    "velocity": [
                        {
                            "newick_tree": "((1:0.1,2:0.1):0.1,(3:0.1,4:0.1):0.1);",
                            "target_tree": "T",
                            "velocity": {3: -0.1},
                            "velocity_next_boundary_tree": "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);",
                            "timepoint": 0.0,
                            "num_leaves": 4,
                        },
                        {
                            "newick_tree": "((1:0.1,2:0.1):0.1,(3:0.1,4:0.1):0.1);",
                            "target_tree": "T",
                            "velocity": {3: -0.1},
                            "velocity_next_boundary_tree": "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);",
                            "timepoint": 0.0,
                            "num_leaves": 4,
                        },
                        {
                            "newick_tree": "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);",
                            "target_tree": "T",
                            "velocity": {5: -0.2},
                            "velocity_next_boundary_tree": "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);",
                            "timepoint": 0.5,
                            "num_leaves": 4,
                        },
                        {
                            "newick_tree": "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);",
                            "target_tree": "T",
                            "velocity": {6: -0.3},
                            "velocity_next_boundary_tree": "(1:0.1,2:0.1,3:0.1,4:0.1);",
                            "timepoint": 0.9,
                            "num_leaves": 4,
                        },
                    ],
                    "autoregressive": [
                        {
                            "newick": "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);",
                            "target_tree": "T",
                            "labels": [{"components": [1, 2, 4], "merge_indices": [0, 1]}],
                            "stop_after_merge": False,
                            "time": 0.1,
                        },
                        {
                            "newick": "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);",
                            "target_tree": "T",
                            "labels": [{"components": [1, 2, 4], "merge_indices": [0, 1]}],
                            "stop_after_merge": False,
                            "time": 0.1,
                        },
                        {
                            "newick": "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);",
                            "target_tree": "T",
                            "labels": [{"components": [1, 4], "merge_indices": [0, 1]}],
                            "stop_after_merge": False,
                            "time": 0.2,
                        },
                        {
                            "newick": "(1:0.1,2:0.1,3:0.1,4:0.1);",
                            "target_tree": "T",
                            "labels": [
                                {
                                    "components": [1, 2, 3, 4],
                                    "merge_indices": [1, 2],
                                }
                            ],
                            "stop_after_merge": False,
                            "time": 0.9,
                        },
                    ],
                },
            )
        )

        oracle_velocity_samples = [
            {
                "newick_tree": "((1:0.1,2:0.1):0.1,(3:0.1,4:0.1):0.1);",
                "target_tree": module._get_harness_sampling_pair.return_value[
                    "target_tree"
                ],
                "velocity": {3: -0.1},
                "velocity_next_boundary_tree": "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);",
                "timepoint": 0.0,
                "num_leaves": 4,
            },
            {
                "newick_tree": "((1:0.1,2:0.1):0.1,(3:0.1,4:0.1):0.1);",
                "target_tree": module._get_harness_sampling_pair.return_value[
                    "target_tree"
                ],
                "velocity": {3: -0.1},
                "velocity_next_boundary_tree": "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);",
                "timepoint": 0.0,
                "num_leaves": 4,
            },
            {
                "newick_tree": "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);",
                "target_tree": module._get_harness_sampling_pair.return_value[
                    "target_tree"
                ],
                "velocity": {5: -0.2},
                "velocity_next_boundary_tree": "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);",
                "timepoint": 0.5,
                "num_leaves": 4,
            },
            {
                "newick_tree": "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);",
                "target_tree": module._get_harness_sampling_pair.return_value[
                    "target_tree"
                ],
                "velocity": {6: -0.3},
                "velocity_next_boundary_tree": "(1:0.1,2:0.1,3:0.1,4:0.1);",
                "timepoint": 0.9,
                "num_leaves": 4,
            },
        ]
        oracle_autoregressive_samples = [
            {
                "newick": "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);",
                "target_tree": module._get_harness_sampling_pair.return_value[
                    "target_tree"
                ],
                "labels": [{"components": [1, 2, 4], "merge_indices": [0, 1]}],
                "stop_after_merge": False,
                "time": 0.1,
            },
            {
                "newick": "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);",
                "target_tree": module._get_harness_sampling_pair.return_value[
                    "target_tree"
                ],
                "labels": [{"components": [1, 2, 4], "merge_indices": [0, 1]}],
                "stop_after_merge": False,
                "time": 0.1,
            },
            {
                "newick": "(1:0.1,2:0.1,3:0.1,4:0.1);",
                "target_tree": module._get_harness_sampling_pair.return_value[
                    "target_tree"
                ],
                "labels": [
                    {
                        "components": [1, 2, 3, 4],
                        "merge_indices": [1, 2],
                    }
                ],
                "stop_after_merge": False,
                "time": 0.9,
            },
        ]

        with patch(
            "run.TrainingModule._collect_oracle_replay_samples_from_anchors",
            return_value=(oracle_velocity_samples, oracle_autoregressive_samples),
        ) as mocked_collect:
            velocity_batch, autoregressive_batch, replay_logs = (
                module._collect_rollout_replay_batches(train=True)
            )

        mocked_collect.assert_called_once()

        self.assertEqual(len(velocity_batch["original_trees"]), 2)
        self.assertEqual(
            velocity_batch["original_trees"],
            [
                "((1:0.1,2:0.1):0.1,(3:0.1,4:0.1):0.1);",
                "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);",
            ],
        )
        self.assertEqual(
            float(replay_logs["replay/num_velocity_states"].item()),
            2.0,
        )
        self.assertEqual(len(autoregressive_batch["newick_autoregressive_trees"]), 1)
        self.assertEqual(
            autoregressive_batch["newick_autoregressive_trees"],
            ["((1:0.1,2:0.1,3:0.1):0.1,4:0.1);"],
        )
        self.assertEqual(
            float(replay_logs["replay/num_autoregressive_states"].item()),
            1.0,
        )
        self.assertEqual(
            float(replay_logs["replay/num_invalid_autoregressive_states"].item()),
            1.0,
        )
        self.assertEqual(
            float(replay_logs["replay/num_anchor_states"].item()),
            2.0,
        )
        self.assertEqual(
            float(replay_logs["replay/cache_refreshed"].item()),
            1.0,
        )
        self.assertEqual(
            float(replay_logs["replay/cache_reused"].item()),
            0.0,
        )

    def test_collect_rollout_replay_batches_reuses_cached_batches_between_refreshes(self):
        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=32,
            n_layers=1,
            n_heads=2,
            output_dim=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            performer_nb_features=None,
            performer_generalized_attention=False,
            layernorm_style="prenorm",
            tokenizer_lap_dim=8,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=1,
            phyla_dim=16,
        )
        dataset = MagicMock()
        dataset.dataset_train = types.SimpleNamespace(
            overfit_split_multi_subset_events=True
        )
        module = TrainingModule(
            model=model,
            dataset=dataset,
            lr=1e-3,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            rollout_replay_velocity_weight=0.5,
            rollout_replay_autoregressive_weight=0.5,
            rollout_replay_start_step=0,
            rollout_replay_frequency=50,
            rollout_replay_anchor_states=1,
            rollout_replay_oracle_horizon=1,
            rollout_replay_max_velocity_states=1,
            rollout_replay_max_autoregressive_states=1,
        )

        module._get_harness_sampling_pair = MagicMock(
            return_value={
                "start_tree": "((1:0.1,2:0.1):0.1,(3:0.1,4:0.1):0.1);",
                "target_tree": "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);",
                "n_leaves": 4,
                "max_events": 4,
            }
        )
        module.sample = MagicMock(
            return_value=(
                ["(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);"],
                0,
                0.0,
                0.0,
                0,
                {
                    "velocity": [
                        {
                            "newick_tree": "((1:0.1,2:0.1):0.1,(3:0.1,4:0.1):0.1);",
                            "target_tree": "T",
                            "velocity": {3: -0.1},
                            "velocity_next_boundary_tree": "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);",
                            "timepoint": 0.0,
                            "num_leaves": 4,
                        },
                    ],
                    "autoregressive": [
                        {
                            "newick": "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);",
                            "target_tree": "T",
                            "labels": [{"components": [1, 2, 4], "merge_indices": [0, 1]}],
                            "stop_after_merge": False,
                            "time": 0.1,
                        },
                    ],
                },
            )
        )
        oracle_velocity_samples = [
            {
                "newick_tree": "((1:0.1,2:0.1):0.1,(3:0.1,4:0.1):0.1);",
                "target_tree": module._get_harness_sampling_pair.return_value[
                    "target_tree"
                ],
                "velocity": {3: -0.1},
                "velocity_next_boundary_tree": "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);",
                "timepoint": 0.0,
                "num_leaves": 4,
            },
        ]
        oracle_autoregressive_samples = [
            {
                "newick": "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);",
                "target_tree": module._get_harness_sampling_pair.return_value[
                    "target_tree"
                ],
                "labels": [
                    {
                        "components": [3, 12],
                        "merge_indices": [0, 1],
                    }
                ],
                "stop_after_merge": False,
                "time": 0.0,
            },
        ]

        with patch(
            "run.TrainingModule._collect_oracle_replay_samples_from_anchors",
            return_value=(oracle_velocity_samples, oracle_autoregressive_samples),
        ):
            module.stepper = 50
            velocity_batch_1, autoregressive_batch_1, replay_logs_1 = (
                module._collect_rollout_replay_batches(train=True)
            )
            self.assertEqual(module.sample.call_count, 1)
            self.assertEqual(float(replay_logs_1["replay/cache_refreshed"].item()), 1.0)
            self.assertEqual(float(replay_logs_1["replay/cache_reused"].item()), 0.0)

            module.stepper = 51
            velocity_batch_2, autoregressive_batch_2, replay_logs_2 = (
                module._collect_rollout_replay_batches(train=True)
            )
            self.assertEqual(module.sample.call_count, 1)
            self.assertIs(velocity_batch_2, velocity_batch_1)
            self.assertIs(autoregressive_batch_2, autoregressive_batch_1)
            self.assertEqual(float(replay_logs_2["replay/cache_refreshed"].item()), 0.0)
            self.assertEqual(float(replay_logs_2["replay/cache_reused"].item()), 1.0)

            module.stepper = 100
            velocity_batch_3, autoregressive_batch_3, replay_logs_3 = (
                module._collect_rollout_replay_batches(train=True)
            )
            self.assertEqual(module.sample.call_count, 2)
            self.assertEqual(float(replay_logs_3["replay/cache_refreshed"].item()), 1.0)
            self.assertEqual(float(replay_logs_3["replay/cache_reused"].item()), 0.0)

    def test_select_rollout_replay_anchors_includes_final_sampled_tree(self):
        trace = {
            "velocity": [
                {
                    "newick_tree": "((1:0.1,2:0.1):0.1,(3:0.1,4:0.1):0.1);",
                    "target_tree": "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);",
                    "timepoint": 0.0,
                    "num_leaves": 4,
                },
                {
                    "newick_tree": "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);",
                    "target_tree": "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);",
                    "timepoint": 0.5,
                    "num_leaves": 4,
                },
            ]
        }

        anchors = _select_rollout_replay_anchors(
            trace,
            "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);",
            "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);",
            3,
        )

        self.assertEqual(len(anchors), 3)
        self.assertEqual(
            anchors[-1]["newick_tree"],
            "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);",
        )

    def test_select_replay_samples_across_rollout_spans_trajectory(self):
        samples = ["s0", "s1", "s2", "s3", "s4"]

        self.assertEqual(
            _select_replay_samples_across_rollout(samples, 2),
            ["s0", "s4"],
        )
        self.assertEqual(
            _select_replay_samples_across_rollout(samples, 3),
            ["s0", "s2", "s4"],
        )
        self.assertEqual(
            _select_replay_samples_across_rollout(samples, 1),
            ["s4"],
        )

    def test_record_repeated_topology_visit_trips_after_cap(self):
        counts = {}
        topo = (1, 3, 7)

        self.assertFalse(_record_repeated_topology_visit(counts, topo, 2))
        self.assertFalse(_record_repeated_topology_visit(counts, topo, 2))
        self.assertTrue(_record_repeated_topology_visit(counts, topo, 2))

    def test_summarize_trace_topology_repeats_reports_repeat_counts(self):
        tree_a = "((0:0.1,1:0.1):0.1,(2:0.1,3:0.1):0.1);"
        tree_b = "((0:0.1,2:0.1):0.1,(1:0.1,3:0.1):0.1);"
        trace = {
            "velocity": [
                {"newick_tree": tree_a},
                {"newick_tree": tree_b},
                {"newick_tree": tree_a},
            ],
            "autoregressive": [
                {"newick": tree_b},
                {"newick": tree_b},
                {"newick": tree_a},
                {"newick": tree_b},
            ],
        }

        summary = _summarize_trace_topology_repeats(trace)

        self.assertEqual(summary["velocity_num_states"], 3.0)
        self.assertEqual(summary["velocity_num_unique_topologies"], 2.0)
        self.assertEqual(summary["velocity_num_repeated_topologies"], 1.0)
        self.assertEqual(summary["velocity_max_topology_repeat"], 2.0)
        self.assertEqual(summary["autoregressive_num_states"], 4.0)
        self.assertEqual(summary["autoregressive_num_unique_topologies"], 2.0)
        self.assertEqual(summary["autoregressive_num_repeated_topologies"], 1.0)
        self.assertEqual(summary["autoregressive_max_topology_repeat"], 3.0)

    def test_best_pairwise_merge_label_for_current_tree_uses_current_group(self):
        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=32,
            n_layers=1,
            n_heads=2,
            output_dim=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            performer_nb_features=None,
            performer_generalized_attention=False,
            layernorm_style="prenorm",
            tokenizer_lap_dim=8,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=1,
            phyla_dim=16,
        )
        module = TrainingModule(
            model=model,
            dataset=MagicMock(),
            lr=1e-3,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
        )

        current_newick = "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);"
        target_newick = "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);"

        replay_event = _best_pairwise_merge_label_for_current_tree(
            module,
            current_newick,
            target_newick,
        )

        self.assertIsNotNone(replay_event)
        self.assertEqual(replay_event["newick"], current_newick)
        self.assertEqual(len(replay_event["labels"]), 1)

        label = replay_event["labels"][0]
        structural_groups = {
            tuple(int(component) for component in group)
            for group in get_structural_polytomy_groups_from_newick(current_newick)
        }
        self.assertIn(tuple(int(component) for component in label["components"]), structural_groups)

        merged_components = [
            int(label["components"][idx]) for idx in label["merge_indices"]
        ]
        self.assertEqual(
            int(label["result_split"]),
            int(merged_components[0]) | int(merged_components[1]),
        )

        td, n_leaves, mapping = _tree_to_model_split_lengths(module, current_newick)
        td[int(label["result_split"])] = 1e-3
        _, candidate_newick = build_tree_from_splits(
            list(td.keys()),
            td,
            n_leaves=n_leaves,
            root_leaf=n_leaves - 1,
            mapping=mapping,
        )
        chosen_rf = float(calculate_norm_rf(candidate_newick, target_newick))
        self.assertLess(
            chosen_rf,
            float(calculate_norm_rf(current_newick, target_newick)),
        )

        exhaustive_best_rf = None
        td_base, n_leaves, mapping = _tree_to_model_split_lengths(module, current_newick)
        for group in get_structural_polytomy_groups_from_newick(current_newick):
            group = tuple(int(component) for component in group)
            for left_idx, right_idx in itertools.combinations(range(len(group)), 2):
                result_split = int(group[left_idx]) | int(group[right_idx])
                if result_split in td_base:
                    continue
                candidate_td = dict(td_base)
                candidate_td[result_split] = 1e-3
                _, exhaustive_newick = build_tree_from_splits(
                    list(candidate_td.keys()),
                    candidate_td,
                    n_leaves=n_leaves,
                    root_leaf=n_leaves - 1,
                    mapping=mapping,
                )
                candidate_rf = float(calculate_norm_rf(exhaustive_newick, target_newick))
                if exhaustive_best_rf is None or candidate_rf < exhaustive_best_rf:
                    exhaustive_best_rf = candidate_rf

        self.assertIsNotNone(exhaustive_best_rf)
        self.assertAlmostEqual(chosen_rf, exhaustive_best_rf, places=7)

    def test_best_pairwise_merge_label_for_current_tree_skips_large_groups(self):
        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=32,
            n_layers=1,
            n_heads=2,
            output_dim=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            performer_nb_features=None,
            performer_generalized_attention=False,
            layernorm_style="prenorm",
            tokenizer_lap_dim=8,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=1,
            phyla_dim=16,
        )
        module = TrainingModule(
            model=model,
            dataset=MagicMock(),
            lr=1e-3,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            rollout_replay_pairwise_max_group_size=2,
        )

        current_newick = "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);"
        target_newick = "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);"

        replay_event = _best_pairwise_merge_label_for_current_tree(
            module,
            current_newick,
            target_newick,
        )

        self.assertIsNone(replay_event)

    def test_sampling_autoregressive_time_uses_event_index_when_enabled(self):
        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=32,
            n_layers=1,
            n_heads=2,
            output_dim=1,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            drop_path_rate=0.0,
            use_performer=False,
            performer_nb_features=None,
            performer_generalized_attention=False,
            layernorm_style="prenorm",
            tokenizer_lap_dim=8,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=1,
            phyla_dim=16,
        )
        module = TrainingModule(
            model=model,
            dataset=MagicMock(),
            lr=1e-3,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            autoregressive_use_time=True,
        )

        self.assertAlmostEqual(
            module._sampling_autoregressive_time_value(
                current_time=0.0,
                event_index=1,
                max_events=16,
            ),
            1.0 / 15.0,
        )
        self.assertAlmostEqual(
            module._sampling_autoregressive_time_value(
                current_time=0.42,
                event_index=None,
                max_events=None,
            ),
            0.42,
        )

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_overfit_boundary_prefix_resolves_to_boundary_endpoint(self, _mock_build_index):
        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_boundary_prefix_k=0,
        )
        target_tree = ds.load_posterior_trees_from_tfiles([])[0]
        start_tree = ds.sample_random_tree(target_tree)
        boundary_paths = return_tree_boundary_merge_paths(
            start_tree,
            target_tree,
            legacy_training_semantics=True,
        )
        self.assertGreater(
            len(boundary_paths),
            0,
            "Sanity tree pair did not expose any boundaries for prefix truncation.",
        )

        resolved_target = ds.resolve_training_target_tree(start_tree, target_tree)
        expected_target = boundary_paths[0]["end_newick"]
        self.assertEqual(
            _normalized_rf(resolved_target, expected_target),
            0.0,
            "Prefix truncation did not resolve to the first boundary endpoint.",
        )

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_overfit_start_and_target_boundary_prefix_resolve_to_oracle_endpoints(
        self, _mock_build_index
    ):
        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_boundary_prefix_k=11,
            overfit_start_boundary_prefix_k=10,
        )
        target_tree = ds.load_posterior_trees_from_tfiles([])[0]
        base_start_tree, start_tree = ds.sample_random_tree_with_base(target_tree)
        resolved_target = ds.resolve_training_target_tree(
            start_tree,
            target_tree,
            base_start_tree_newick=base_start_tree,
        )

        base_ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
        )
        base_start_tree = base_ds.sample_random_tree(target_tree)
        boundary_paths = return_tree_boundary_merge_paths(
            base_start_tree,
            target_tree,
            legacy_training_semantics=True,
        )
        self.assertGreater(
            len(boundary_paths),
            11,
            "Sanity tree pair did not expose enough boundaries for the k10->k11 transition test.",
        )

        expected_start = boundary_paths[10]["end_newick"]
        expected_target = boundary_paths[11]["end_newick"]
        self.assertEqual(
            _normalized_rf(start_tree, expected_start),
            0.0,
            "Direct transition start tree did not resolve to the oracle k10 endpoint.",
        )
        self.assertEqual(
            _normalized_rf(resolved_target, expected_target),
            0.0,
            "Direct transition target tree did not resolve to the absolute oracle k11 endpoint.",
        )

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_overfit_event_prefix_resolves_inside_direct_transition(
        self, _mock_build_index
    ):
        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_boundary_prefix_k=11,
            overfit_start_boundary_prefix_k=10,
            overfit_event_prefix_count=1,
        )
        target_tree = ds.load_posterior_trees_from_tfiles([])[0]
        base_start_tree, start_tree = ds.sample_random_tree_with_base(target_tree)
        resolved_target = ds.resolve_training_target_tree(
            start_tree,
            target_tree,
            base_start_tree_newick=base_start_tree,
        )

        base_ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
        )
        base_start_tree = base_ds.sample_random_tree(target_tree)
        boundary_paths = return_tree_boundary_merge_paths(
            base_start_tree,
            target_tree,
            legacy_training_semantics=True,
        )
        transition = boundary_paths[11]
        self.assertGreater(
            len(transition["events"]),
            1,
            "Expected the k10->k11 transition to expose multiple merge events.",
        )

        expected_target = transition["events"][1]["newick"]
        self.assertEqual(
            _normalized_rf(resolved_target, expected_target),
            0.0,
            "Event-prefix truncation did not resolve to the tree after the first merge event inside k10->k11.",
        )

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_velocity_event_state_sampling_uses_oracle_event_newick(
        self, _mock_build_index
    ):
        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_boundary_prefix_k=11,
            overfit_velocity_zero=True,
            overfit_velocity_event_states=True,
        )
        ds._index = [{"id": "mock", "nexus_path": "mock", "tree_paths": []}]
        ds._id_to_idx = {"mock": 0}
        ds.parse_nexus = lambda path: (
            {str(i + 1): "A" for i in range(155)},
            [str(i + 1) for i in range(155)],
        )

        sample = ds[0]

        self.assertEqual(sample["timepoint"], 0.0)
        self.assertEqual(
            _normalized_rf(sample["newick_tree"], sample["autoregressive_newick"]),
            0.0,
            "Velocity state should be sampled from the same oracle event tree used for the autoregressive label.",
        )

    @patch.object(TreeDataset, "build_index", return_value=None)
    @patch("data.dataset.random.choice")
    @patch("data.dataset.return_sampled_tree_orthant_velocity")
    @patch("data.dataset.return_tree_boundary_merge_paths")
    @patch("data.dataset.return_sampled_tree_boundary_decisions")
    def test_velocity_orthant_start_sampling_uses_resolved_boundary_starts(
        self,
        mock_boundary_decisions,
        mock_boundary_paths,
        mock_velocity,
        mock_choice,
        _mock_build_index,
    ):
        start_tree = "((1:0.1,2:0.1):0.1,(3:0.1,4:0.1):0.1);"
        after_first_boundary = "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);"
        target_tree = "((1:0.1,3:0.1):0.1,(2:0.1,4:0.1):0.1);"
        ar_event_tree = "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);"

        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_velocity_orthant_start_states=True,
        )
        ds._index = [{"id": "mock", "nexus_path": "mock", "tree_paths": []}]
        ds._id_to_idx = {"mock": 0}
        ds.parse_nexus = lambda path: (
            {str(i + 1): "A" for i in range(4)},
            [str(i + 1) for i in range(4)],
        )
        ds.load_posterior_trees_from_tfiles = lambda paths: [target_tree]
        ds.sample_random_tree_with_base = lambda target: (start_tree, start_tree)
        ds.resolve_training_target_tree = (
            lambda random_tree, real_tree_newick, base_start_tree_newick=None: target_tree
        )

        mock_boundary_decisions.return_value = [
            {
                "newick": ar_event_tree,
                "labels": [{"components": [1, 2, 3], "merge_indices": [0, 1]}],
            }
        ]
        mock_boundary_paths.return_value = [
            {"end_newick": after_first_boundary, "events": [{}]},
            {"end_newick": target_tree, "events": [{}]},
        ]
        mock_choice.return_value = after_first_boundary
        mock_velocity.side_effect = lambda source_tree, _target_tree, timepoint: (
            source_tree,
            {"timepoint": timepoint},
        )

        sample = ds[0]

        self.assertEqual(sample["timepoint"], 0.0)
        self.assertEqual(sample["newick_tree"], after_first_boundary)
        self.assertNotEqual(
            sample["newick_tree"],
            target_tree,
            "Velocity orthant-start sampling should not supervise on the final target tree.",
        )

    @patch.object(TreeDataset, "build_index", return_value=None)
    @patch("data.dataset.return_sampled_tree_orthant_velocity")
    def test_velocity_fixed_timepoints_use_global_time_choices(
        self,
        mock_velocity,
        _mock_build_index,
    ):
        start_tree = "((1:0.1,2:0.1):0.1,(3:0.1,4:0.1):0.1);"
        target_tree = "((1:0.1,3:0.1):0.1,(2:0.1,4:0.1):0.1);"
        ar_event_tree = "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);"

        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_velocity_fixed_timepoints=[0.0, 1.0],
        )
        ds._index = [{"id": "mock", "nexus_path": "mock", "tree_paths": []}]
        ds._id_to_idx = {"mock": 0}
        ds.parse_nexus = lambda path: (
            {str(i + 1): "A" for i in range(4)},
            [str(i + 1) for i in range(4)],
        )
        ds.load_posterior_trees_from_tfiles = lambda paths: [target_tree]
        ds.sample_random_tree_with_base = lambda target: (start_tree, start_tree)
        ds.resolve_training_target_tree = (
            lambda random_tree, real_tree_newick, base_start_tree_newick=None: target_tree
        )

        with patch("data.dataset.random.choice", return_value=1.0), patch(
            "data.dataset.return_sampled_tree_boundary_decisions",
            return_value=[
                {
                    "newick": ar_event_tree,
                    "labels": [{"components": [1, 2, 3], "merge_indices": [0, 1]}],
                }
            ],
        ), patch(
            "data.dataset.return_tree_boundary_merge_paths",
            return_value=[{"end_newick": target_tree, "events": [{}]}],
        ):
            mock_velocity.side_effect = lambda source_tree, _target_tree, timepoint: (
                f"time={timepoint}",
                {"timepoint": timepoint},
            )

            sample = ds[0]
        self.assertEqual(sample["timepoint"], 1.0)
        self.assertEqual(sample["newick_tree"], "time=1.0")

    @patch.object(TreeDataset, "build_index", return_value=None)
    @patch("data.dataset.random.choice")
    @patch("data.dataset.return_sampled_tree_orthant_velocity")
    @patch("data.dataset.return_tree_boundary_merge_paths")
    @patch("data.dataset.return_sampled_tree_boundary_decisions")
    def test_velocity_explicit_boundary_end_states_use_end_newick_with_global_time(
        self,
        mock_boundary_decisions,
        mock_boundary_paths,
        mock_velocity,
        mock_choice,
        _mock_build_index,
        ):
        start_tree = "((1:0.1,2:0.1):0.1,(3:0.1,4:0.1):0.1);"
        boundary0_end = "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);"
        boundary1_start = "((1:0.1,2:0.1):0.1,(3:0.1,4:0.1):0.2);"
        target_tree = "((1:0.1,3:0.1):0.1,(2:0.1,4:0.1):0.1);"
        ar_event_tree = "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);"

        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_velocity_explicit_boundary_end_states=True,
            overfit_velocity_fixed_timepoints=[0.0, 0.25],
        )
        ds._index = [{"id": "mock", "nexus_path": "mock", "tree_paths": []}]
        ds._id_to_idx = {"mock": 0}
        ds.parse_nexus = lambda path: (
            {str(i + 1): "A" for i in range(4)},
            [str(i + 1) for i in range(4)],
        )
        ds.load_posterior_trees_from_tfiles = lambda paths: [target_tree]
        ds.sample_random_tree_with_base = lambda target: (start_tree, start_tree)
        ds.resolve_training_target_tree = (
            lambda random_tree, real_tree_newick, base_start_tree_newick=None: target_tree
        )

        mock_boundary_decisions.return_value = [
            {
                "newick": ar_event_tree,
                "labels": [{"components": [1, 2, 3], "merge_indices": [0, 1]}],
            }
        ]
        mock_boundary_paths.return_value = [
            {"start_newick": start_tree, "end_newick": boundary0_end, "events": [{}]},
            {
                "start_newick": boundary1_start,
                "end_newick": target_tree,
                "events": [{}],
            },
        ]
        mock_choice.return_value = (boundary0_end, boundary1_start, 0.25)
        mock_velocity.side_effect = lambda source_tree, _target_tree, timepoint: (
            f"{source_tree}|local_t={timepoint}",
            {"timepoint": timepoint},
        )

        sample = ds[0]

        self.assertEqual(sample["timepoint"], 0.25)
        self.assertEqual(sample["newick_tree"], f"{boundary0_end}|local_t=0.0")
        self.assertEqual(sample["velocity_next_boundary_tree"], boundary1_start)
        mock_velocity.assert_called_once_with(boundary0_end, target_tree, 0.0)

    @patch.object(TreeDataset, "build_index", return_value=None)
    @patch("data.dataset.random.choice")
    @patch("data.dataset.return_sampled_tree_orthant_velocity")
    @patch("data.dataset.return_tree_boundary_merge_paths")
    @patch("data.dataset.return_sampled_tree_boundary_decisions")
    def test_velocity_explicit_boundary_end_states_derive_global_timepoints_from_paths(
        self,
        mock_boundary_decisions,
        mock_boundary_paths,
        mock_velocity,
        mock_choice,
        _mock_build_index,
    ):
        start_tree = "((1:0.1,2:0.1):0.1,(3:0.1,4:0.1):0.1);"
        boundary0_end = "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);"
        boundary1_start = "((1:0.1,2:0.1):0.1,(3:0.1,4:0.1):0.2);"
        target_tree = "((1:0.1,3:0.1):0.1,(2:0.1,4:0.1):0.1);"
        ar_event_tree = "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);"

        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_velocity_explicit_boundary_end_states=True,
        )
        ds._index = [{"id": "mock", "nexus_path": "mock", "tree_paths": []}]
        ds._id_to_idx = {"mock": 0}
        ds.parse_nexus = lambda path: (
            {str(i + 1): "A" for i in range(4)},
            [str(i + 1) for i in range(4)],
        )
        ds.load_posterior_trees_from_tfiles = lambda paths: [target_tree]
        ds.sample_random_tree_with_base = lambda target: (start_tree, start_tree)
        ds.resolve_training_target_tree = (
            lambda random_tree, real_tree_newick, base_start_tree_newick=None: target_tree
        )

        mock_boundary_decisions.return_value = [
            {
                "newick": ar_event_tree,
                "labels": [{"components": [1, 2, 3], "merge_indices": [0, 1]}],
            }
        ]
        mock_boundary_paths.return_value = [
            {
                "global_time": 0.25,
                "start_newick": start_tree,
                "end_newick": boundary0_end,
                "events": [{}],
            },
            {
                "global_time": 0.75,
                "start_newick": boundary1_start,
                "end_newick": target_tree,
                "events": [{}],
            },
        ]
        mock_choice.return_value = (boundary0_end, boundary1_start, 0.25)
        mock_velocity.side_effect = lambda source_tree, _target_tree, timepoint: (
            f"{source_tree}|local_t={timepoint}",
            {"timepoint": timepoint},
        )

        sample = ds[0]

        self.assertEqual(sample["timepoint"], 0.25)
        self.assertEqual(sample["newick_tree"], f"{boundary0_end}|local_t=0.0")
        self.assertEqual(sample["velocity_next_boundary_tree"], boundary1_start)
        mock_velocity.assert_called_once_with(boundary0_end, target_tree, 0.0)

    def test_boundary_vanish_mask_decoder_falls_back_to_argmax(self):
        logits = torch.tensor([-2.0, -1.0, -3.0], dtype=torch.float32)
        candidate_mask = np.array([True, True, True], dtype=bool)
        pred_mask = _predict_boundary_vanish_mask_from_logits(logits.numpy(), candidate_mask)
        self.assertTrue(pred_mask[1])
        self.assertEqual(int(pred_mask.sum()), 1)

    def test_boundary_vanish_one_step_zeros_targets_and_clips_blocked_collapses(self):
        lengths = np.array([0.1, 0.6, 0.2], dtype=np.float64)
        velocities = np.array([-0.1, -0.2, -0.5], dtype=np.float64)
        predicted_vanish_mask = np.array([True, True, False], dtype=bool)
        supervised_mask = np.array([True, True, True], dtype=bool)

        lengths_new, dt_boundary, used = _apply_boundary_vanish_one_step(
            lengths=lengths,
            velocities=velocities,
            predicted_vanish_mask=predicted_vanish_mask,
            supervised_mask=supervised_mask,
            dt_cap=float("inf"),
            eps_len=1e-8,
        )

        self.assertTrue(used)
        self.assertAlmostEqual(dt_boundary, 3.0)
        self.assertEqual(float(lengths_new[0]), 0.0)
        self.assertEqual(float(lengths_new[1]), 0.0)
        self.assertGreater(float(lengths_new[2]), 0.0)
        self.assertAlmostEqual(float(lengths_new[2]), 1e-7)

    def test_edge_set_bce_loss_reports_exact_match(self):
        logits = torch.tensor([5.0, -5.0, 5.0], dtype=torch.float32)
        target = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
        loss, stats = _edge_set_bce_loss(logits, target)
        self.assertLess(float(loss.item()), 0.1)
        self.assertEqual(stats["target_size"], 2)
        self.assertEqual(stats["pred_size"], 2)
        self.assertEqual(stats["recall"], 1.0)
        self.assertEqual(stats["precision"], 1.0)
        self.assertEqual(stats["jaccard"], 1.0)

    @patch.object(TreeDataset, "build_index", return_value=None)
    @patch("data.dataset.return_sampled_tree_orthant_velocity")
    @patch("data.dataset.return_tree_boundary_merge_paths")
    @patch("data.dataset.return_sampled_tree_boundary_decisions")
    def test_overfit_fixed_pair_reuses_same_start_target_pair(
        self,
        mock_boundary_decisions,
        mock_boundary_paths,
        mock_velocity,
        _mock_build_index,
    ):
        real_tree = "((1:0.1,2:0.1):0.1,(3:0.1,4:0.1):0.1);"
        start_tree_a = "((1:0.1,2:0.1):0.1,(3:0.1,4:0.1):0.1);"
        start_tree_b = "((1:0.1,3:0.1):0.1,(2:0.1,4:0.1):0.1);"
        target_tree_a = "(((1:0.1,2:0.1):0.1,3:0.1):0.1,4:0.1);"
        target_tree_b = "(((1:0.1,3:0.1):0.1,2:0.1):0.1,4:0.1);"
        ar_event_tree = "((1:0.1,2:0.1,3:0.1):0.1,4:0.1);"

        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_start_boundary_prefix_k=10,
            overfit_boundary_prefix_k=12,
            overfit_velocity_fixed_timepoints=[0.0],
            overfit_fixed_pair=True,
        )
        ds._index = [{"id": "mock", "nexus_path": "mock", "tree_paths": []}]
        ds._id_to_idx = {"mock": 0}
        ds.parse_nexus = lambda path: (
            {str(i + 1): "A" for i in range(4)},
            [str(i + 1) for i in range(4)],
        )
        ds.load_posterior_trees_from_tfiles = lambda paths: [real_tree]

        call_counter = {"count": 0}

        def sample_random_tree_with_base(_target):
            call_counter["count"] += 1
            if call_counter["count"] == 1:
                return start_tree_a, start_tree_a
            return start_tree_b, start_tree_b

        ds.sample_random_tree_with_base = sample_random_tree_with_base
        ds.resolve_training_target_tree = lambda random_tree, _real_tree, base_start_tree_newick=None: (
            target_tree_a if random_tree == start_tree_a else target_tree_b
        )

        mock_boundary_decisions.return_value = [
            {
                "newick": ar_event_tree,
                "labels": [{"components": [1, 2, 3], "merge_indices": [0, 1]}],
            }
        ]
        mock_boundary_paths.return_value = [{"end_newick": target_tree_a, "events": [{}]}]
        mock_velocity.side_effect = lambda source_tree, _target_tree, timepoint: (
            source_tree,
            {"timepoint": timepoint},
        )

        sample_one = ds[0]
        sample_two = ds[0]

        self.assertEqual(call_counter["count"], 1)
        self.assertEqual(sample_one["newick_tree"], sample_two["newick_tree"])
        self.assertEqual(sample_one["target_tree"], sample_two["target_tree"])

    def test_split_multi_label_training_events_creates_sequential_singletons(self):
        current_newick = "((0:0.1,1:0.1,2:0.1,3:0.1):0.1,4:0.1);"
        filtered_events = [
            {
                "newick": current_newick,
                "labels": [
                    {
                        "result_split": 3,
                        "parent_split": 15,
                        "components": [1, 2, 4, 8],
                        "merge_indices": [0, 1],
                    },
                    {
                        "result_split": 12,
                        "parent_split": 15,
                        "components": [1, 2, 4, 8],
                        "merge_indices": [2, 3],
                    },
                ],
            }
        ]

        split_events = _split_multi_label_training_events(filtered_events)

        self.assertEqual(len(split_events), 2)
        self.assertEqual(split_events[0]["newick"], current_newick)
        self.assertNotEqual(split_events[1]["newick"], current_newick)
        self.assertFalse(split_events[0]["stop_after_merge"])
        self.assertTrue(split_events[1]["stop_after_merge"])
        self.assertEqual(len(split_events[0]["labels"]), 1)
        self.assertEqual(len(split_events[1]["labels"]), 1)
        self.assertEqual(split_events[0]["labels"][0]["result_split"], 3)
        self.assertEqual(split_events[1]["labels"][0]["result_split"], 12)
        second_label = split_events[1]["labels"][0]
        merged_components = {
            int(second_label["components"][idx]) for idx in second_label["merge_indices"]
        }
        self.assertEqual(merged_components, {4, 8})

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_overfit_event_horizon_returns_consecutive_event_samples(
        self, _mock_build_index
    ):
        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_boundary_prefix_k=11,
            overfit_velocity_zero=True,
            overfit_velocity_event_states=True,
            overfit_event_horizon=2,
        )
        ds._index = [{"id": "mock", "nexus_path": "mock", "tree_paths": []}]
        ds._id_to_idx = {"mock": 0}
        ds.parse_nexus = lambda path: (
            {str(i + 1): "A" for i in range(155)},
            [str(i + 1) for i in range(155)],
        )
        ds.tree_tokenizer = lambda trees: trees

        sample = ds[0]
        self.assertIn("multi_step_samples", sample)
        self.assertEqual(len(sample["multi_step_samples"]), 2)

        first_index = sample["multi_step_samples"][0]["autoregressive_event_index"]
        second_index = sample["multi_step_samples"][1]["autoregressive_event_index"]
        self.assertEqual(
            second_index,
            first_index + 1,
            "Multi-step overfit horizon did not return consecutive oracle event indices.",
        )

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_harness_sampling_pair_for_direct_transition_uses_raw_prefix_resolution(
        self, _mock_build_index
    ):
        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_boundary_prefix_k=11,
            overfit_start_boundary_prefix_k=10,
        )
        real_tree_raw = ds.load_posterior_trees_from_tfiles([])[0]
        target_obj = EteTree(real_tree_raw, format=1)
        leaves = target_obj.get_leaves()
        leaves.sort(key=lambda leaf: leaf.name)
        seq_ordering_map = {}
        for i, leaf in enumerate(leaves):
            original_name = leaf.name
            mapped_name = str(i)
            leaf.name = mapped_name
            seq_ordering_map[original_name] = mapped_name
        base_start_tree_raw, start_tree_raw = ds.sample_random_tree_with_base(real_tree_raw)
        target_tree_raw = ds.resolve_training_target_tree(
            start_tree_raw,
            real_tree_raw,
            base_start_tree_newick=base_start_tree_raw,
        )
        start_tree_obj = EteTree(start_tree_raw, format=1)
        for leaf in start_tree_obj.get_leaves():
            leaf.name = seq_ordering_map[leaf.name]
        start_tree = start_tree_obj.write(format=1)

        target_tree_obj = EteTree(target_tree_raw, format=1)
        for leaf in target_tree_obj.get_leaves():
            leaf.name = seq_ordering_map[leaf.name]
        target_tree = target_tree_obj.write(format=1)

        self.assertAlmostEqual(
            _normalized_rf(start_tree, target_tree),
            0.09352517985611511,
            places=9,
            msg="Harness direct-transition pair did not preserve the oracle k10->k11 normalized RF.",
        )

    @patch.object(TreeDataset, "build_index", return_value=None)
    def test_velocity_and_autoregressive_multi_boundary_prefix_overfit_sanity_pair(
        self, _mock_build_index
    ):
        random.seed(777)
        torch.manual_seed(777)
        device = torch.device("cpu")

        ds = TreeDataset(
            nexus_root="mock",
            mrbayes_root="mock",
            random_sanity_check=True,
            overfit_velocity_zero=True,
        )
        real_tree = ds.load_posterior_trees_from_tfiles([])[0]
        random_tree = ds.sample_random_tree(real_tree)

        boundary_paths = return_tree_boundary_merge_paths(
            random_tree,
            real_tree,
            legacy_training_semantics=True,
        )
        prefix_last_boundary = 11
        self.assertGreater(
            len(boundary_paths),
            prefix_last_boundary,
            "Sanity tree pair did not expose enough boundaries for the multi-boundary overfit test.",
        )
        prefix_paths = boundary_paths[: prefix_last_boundary + 1]
        prefix_time = _boundary_prefix_time(
            random_tree,
            real_tree,
            prefix_last_boundary,
        )

        model = TreeDenoiserTokenGT(
            num_node_types=3,
            num_edge_types=2,
            embed_dim=64,
            n_layers=2,
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
            tokenizer_lap_dim=8,
            tokenizer_lap_dropout=0.0,
            tokenizer_n_layers=2,
            phyla_dim=16,
        ).to(device)

        dataset_stub = MagicMock()
        dataset_stub.msa_distance = True
        dataset_stub.chosen_tree = (0, 0, 1)

        module = TrainingModule(
            model=model,
            dataset=dataset_stub,
            lr=1e-3,
            record=False,
            epochs=1,
            deepspeed=False,
            logger=None,
            velocity_loss_mode="plain",
            velocity_sign_eps=1e-3,
            training_step_velocity_weight=1,
            training_step_autoregressive_weight=0.1,
            velocity_dt_candidate_weight=1,
            velocity_dt_hit_weight=1,
            velocity_event_weight=0.0,
            velocity_first_hit_head_weight=1.0,
            velocity_first_hit_head_use_at_sampling=True,
            velocity_first_hit_predictor_mode="edge_token_attention",
        ).to(device)
        module.legacy_first_hit_gather_only = True
        module.autoregressive_allow_multi_subset_targets = True

        velocity_times = [prefix_time * i / 8.0 for i in range(8)]
        velocity_batches = [
            _make_batch_from_tree_pair(
                tokenizer=model.tokenizer,
                start_tree=random_tree,
                target_tree=real_tree,
                time_point=time_point,
            )
            for time_point in velocity_times
        ]
        dataset_stub.chosen_tree = (0, int(velocity_batches[0]["num_leaves"][0]), 1)

        event_records = []
        global_event_idx = 0
        for path in prefix_paths:
            for event in path["events"]:
                event_records.append((path["boundary_index"], global_event_idx, event))
                global_event_idx += 1

        event_batches = [
            _make_autoregressive_event_batch(
                model.tokenizer,
                event["newick"],
                event["labels"],
                _normalized_event_time(global_event_idx, len(event_records)),
            )
            for _, global_event_idx, event in event_records
        ]

        optimizer = torch.optim.Adam(module.model.parameters(), lr=5e-3)

        def evaluate_state(include_sample=False):
            module.eval()

            exact_events = 0
            first_failure = None
            for boundary_index, global_event_index, event in event_records:
                predicted, _ = _predict_autoregressive_event(
                    module,
                    model.tokenizer,
                    event["newick"],
                    _normalized_event_time(global_event_index, len(event_records)),
                )
                target = _target_merge_subsets_for_event(event["labels"])
                if predicted == target:
                    exact_events += 1
                elif first_failure is None:
                    first_failure = (
                        boundary_index,
                        global_event_index,
                        predicted,
                        target,
                    )

            velocity_metrics = [
                _velocity_metrics(module, batch, topk=3) for batch in velocity_batches
            ]
            mean_velocity_mse = sum(m["mse"] for m in velocity_metrics) / len(
                velocity_metrics
            )
            min_dt_recall = min(m["dt_first_hit_recall"] for m in velocity_metrics)
            max_dt_rel_err = max(m["dt_hit_rel_err"] for m in velocity_metrics)

            sample_eval = {
                "rf_norm": float("inf"),
                "sampled_tree": None,
                "target_tree": prefix_paths[-1]["end_newick"],
                "exception": None,
            }
            if include_sample:
                try:
                    sampled_trees, *_ = module.sample(
                        [random_tree],
                        None,
                        num_samples=1,
                        T=float(prefix_time),
                        dt_base=min(float(prefix_time), 0.02),
                        max_events=1024,
                        max_steps=4096,
                    )
                    sample_eval["sampled_tree"] = sampled_trees[0]
                    sample_eval["rf_norm"] = _normalized_rf(
                        sampled_trees[0],
                        prefix_paths[-1]["end_newick"],
                    )
                except Exception as exc:
                    sample_eval["exception"] = f"{type(exc).__name__}: {exc}"

            return {
                "exact_events": exact_events,
                "first_failure": first_failure,
                "mean_velocity_mse": mean_velocity_mse,
                "min_dt_recall": min_dt_recall,
                "max_dt_rel_err": max_dt_rel_err,
                "sample": sample_eval,
            }

        def score_state(eval_state):
            sample = eval_state["sample"]
            rf = sample["rf_norm"]
            rf_score = -rf if math.isfinite(rf) else float("-inf")
            return (
                eval_state["exact_events"],
                eval_state["min_dt_recall"],
                -eval_state["mean_velocity_mse"],
                rf_score,
            )

        best_state = copy.deepcopy(module.model.state_dict())
        best_eval = None
        best_score = (
            -1,
            -1.0,
            float("-inf"),
            float("-inf"),
        )

        max_steps = 40
        for step in range(max_steps):
            module.train()
            optimizer.zero_grad(set_to_none=True)

            velocity_losses = [
                module.step(batch, autoregressive=False)["loss"]
                for batch in velocity_batches
            ]
            autoregressive_losses = [
                module.step(batch, autoregressive=True)["loss"]
                for batch in event_batches
            ]
            total_loss = torch.stack(velocity_losses).mean() + 0.1 * torch.stack(
                autoregressive_losses
            ).mean()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(module.model.parameters(), max_norm=1.0)
            optimizer.step()

            if (step + 1) % 10 == 0 or step == 0:
                current_eval = evaluate_state(include_sample=False)
                current_score = score_state(current_eval)
                if current_score > best_score:
                    best_score = current_score
                    best_eval = current_eval
                    best_state = copy.deepcopy(module.model.state_dict())

                if (
                    current_eval["exact_events"] == len(event_records)
                    and current_eval["min_dt_recall"] == 1.0
                    and current_eval["max_dt_rel_err"] < 0.30
                ):
                    break

        module.model.load_state_dict(best_state)
        if best_eval is None:
            best_eval = evaluate_state(include_sample=False)
        final_eval = evaluate_state(include_sample=True)

        self.assertEqual(
            final_eval["exact_events"],
            len(event_records),
            (
                "Joint overfit did not memorize the flattened multi-boundary oracle events "
                f"through boundary {prefix_last_boundary}. "
                f"first_failure={final_eval['first_failure']}"
            ),
        )
        self.assertEqual(
            final_eval["min_dt_recall"],
            1.0,
            (
                "Velocity supervision did not recover the full first-hit set across the sampled prefix times. "
                f"best_eval={best_eval}"
            ),
        )
        self.assertLess(
            final_eval["max_dt_rel_err"],
            0.30,
            (
                "Velocity supervision did not localize the prefix boundary hits accurately enough. "
                f"best_eval={best_eval}"
            ),
        )
        self.assertIsNone(
            final_eval["sample"]["exception"],
            "Sampling through the multi-boundary prefix raised an exception. "
            + str(final_eval["sample"]["exception"]),
        )
        self.assertEqual(
            final_eval["sample"]["rf_norm"],
            0.0,
            (
                "Sampling to the end of the multi-boundary prefix did not recover the oracle topology. "
                f"rf_norm={final_eval['sample']['rf_norm']:.6f}, "
                f"sampled={final_eval['sample']['sampled_tree']}, "
                f"target={final_eval['sample']['target_tree']}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
