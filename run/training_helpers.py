import functools
import itertools
import json
import math
import operator
import os
from collections import Counter, OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from ete3 import Tree as EteTree

from utils.utils import remove_bit, has_polytomy_fast
from utils.random_tree import Tree
from utils.bhv_utils import (
    BHVEncoder,
    get_structural_polytomy_groups_from_newick,
    return_sampled_tree_orthant_velocity,
    return_tree_boundary_merge_paths,
)
from utils.bhv_movie import build_tree_from_splits
from utils.metric_utils import calculate_norm_rf


def _decode_positive_merge_subsets(group_output, threshold_logit=0.0):
    splits = group_output.get("_splits_represented_ints")
    if splits is None:
        splits = [int(split) for split in group_output["splits_represented"]]
        group_output["_splits_represented_ints"] = splits
    logits = group_output["logits"].detach()
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
    decoded_subsets = []

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
            decoded_subsets.append(
                tuple(sorted(int(splits[node_idx]) for node_idx in component))
            )

    return decoded_subsets


def _score_merge_subset(group_output, subset):
    splits = group_output.get("_splits_represented_ints")
    if splits is None:
        splits = [int(split) for split in group_output["splits_represented"]]
        group_output["_splits_represented_ints"] = splits
    split_to_index = {split: idx for idx, split in enumerate(splits)}
    logits = group_output["logits"].detach()
    subset_indices = [split_to_index[int(split)] for split in subset if int(split) in split_to_index]

    if len(subset_indices) < 2:
        return float("-inf")

    scores = []
    for i, left_idx in enumerate(subset_indices):
        for right_idx in subset_indices[i + 1 :]:
            score = float(logits[left_idx, right_idx].item())
            if math.isfinite(score):
                scores.append(score)

    return float(sum(scores) / len(scores)) if scores else float("-inf")


def _subset_prediction_logits(group_splits, subset, device, positive_logit=1.0, negative_logit=-1.0):
    size = len(group_splits)
    logits = torch.full(
        (size, size),
        float(negative_logit),
        dtype=torch.float32,
        device=device,
    )
    logits.fill_diagonal_(float("-inf"))
    subset = {int(split) for split in subset}
    subset_indices = [
        idx for idx, split in enumerate(group_splits) if int(split) in subset
    ]
    for left_idx in subset_indices:
        for right_idx in subset_indices:
            if left_idx != right_idx:
                logits[left_idx, right_idx] = float(positive_logit)
    return logits


def _boundary_event_distribution_loss(
    lengths,
    y_true,
    y_pred,
    velocity_sign_eps=0.0,
    dt_eps=1e-6,
    temp=0.5,
    rate_beta=5.0,
    normalize_by_log_candidates=True,
    first_hit_tol=0.01,
):
    zero = y_pred.new_tensor(0.0)
    stats = {
        "n_candidates": 0,
        "target_first_size": 0,
        "pred_first_mass": 0.0,
        "top1_hits_first_set": 0.0,
    }

    candidate_mask = lengths > 1e-8
    if int(candidate_mask.sum().item()) == 0:
        return zero, stats

    Lc_all = lengths[candidate_mask].clamp_min(float(dt_eps))
    yc_all = y_true[candidate_mask]
    pc_all = y_pred[candidate_mask]

    contract_mask = yc_all < -float(velocity_sign_eps)
    if int(contract_mask.sum().item()) == 0:
        return zero, stats

    eps = float(dt_eps)
    n_candidates = int(candidate_mask.sum().item())
    rate_pred = (
        F.softplus(-pc_all, beta=float(rate_beta)).clamp_min(eps) / Lc_all
    )
    pred_logits = torch.log(rate_pred.clamp_min(eps)) / float(temp)

    tau_true = Lc_all[contract_mask] / (-yc_all[contract_mask]).clamp_min(eps)
    tau_true_min = tau_true.min()
    first_contract_mask = torch.abs(tau_true - tau_true_min) <= float(first_hit_tol)
    first_mask = torch.zeros_like(contract_mask)
    contract_indices = torch.where(contract_mask)[0]
    first_mask[contract_indices[first_contract_mask]] = True

    target_probs = torch.zeros_like(pred_logits)
    target_probs[first_mask] = 1.0 / first_mask.sum().clamp_min(1)
    pred_log_probs = F.log_softmax(pred_logits, dim=0)

    loss = -(target_probs * pred_log_probs).sum()
    if normalize_by_log_candidates and n_candidates > 1:
        normalizer = math.log(float(n_candidates))
        if normalizer > 0.0:
            loss = loss / normalizer

    pred_probs = pred_log_probs.exp()
    pred_top_idx = int(torch.argmax(pred_logits).item())

    stats = {
        "n_candidates": n_candidates,
        "target_first_size": int(first_mask.sum().item()),
        "pred_first_mass": float(pred_probs[first_mask].sum().detach().item()),
        "top1_hits_first_set": float(first_mask[pred_top_idx].float().detach().item()),
    }
    return loss, stats


def _boundary_event_precision_margin_loss(
    lengths,
    y_true,
    y_pred,
    velocity_sign_eps=0.0,
    dt_eps=1e-6,
    temp=0.5,
    rate_beta=5.0,
    first_hit_tol=0.01,
    margin=0.0,
):
    zero = y_pred.new_tensor(0.0)
    stats = {
        "margin_gap": 0.0,
        "n_pos": 0,
        "n_neg": 0,
        "violated": 0.0,
    }

    candidate_mask = lengths > 1e-8
    if int(candidate_mask.sum().item()) == 0:
        return zero, stats

    Lc_all = lengths[candidate_mask].clamp_min(float(dt_eps))
    yc_all = y_true[candidate_mask]
    pc_all = y_pred[candidate_mask]

    contract_mask = yc_all < -float(velocity_sign_eps)
    if int(contract_mask.sum().item()) == 0:
        return zero, stats

    eps = float(dt_eps)
    rate_pred = (
        F.softplus(-pc_all, beta=float(rate_beta)).clamp_min(eps) / Lc_all
    )
    pred_logits = torch.log(rate_pred.clamp_min(eps)) / float(temp)

    tau_true = Lc_all[contract_mask] / (-yc_all[contract_mask]).clamp_min(eps)
    tau_true_min = tau_true.min()
    first_contract_mask = torch.abs(tau_true - tau_true_min) <= float(first_hit_tol)
    first_mask = torch.zeros_like(contract_mask)
    contract_indices = torch.where(contract_mask)[0]
    first_mask[contract_indices[first_contract_mask]] = True

    pos_logits = pred_logits[first_mask]
    neg_logits = pred_logits[~first_mask]
    if pos_logits.numel() == 0 or neg_logits.numel() == 0:
        stats["n_pos"] = int(pos_logits.numel())
        stats["n_neg"] = int(neg_logits.numel())
        return zero, stats

    min_pos_logit = pos_logits.min()
    max_neg_logit = neg_logits.max()
    gap = min_pos_logit - max_neg_logit
    loss = F.relu(float(margin) - gap)

    stats = {
        "margin_gap": float(gap.detach().item()),
        "n_pos": int(pos_logits.numel()),
        "n_neg": int(neg_logits.numel()),
        "violated": float((gap < float(margin)).detach().item()),
    }
    return loss, stats


def _group_slices_from_sizes(group_sizes, total):
    sizes = [int(x) for x in group_sizes if int(x) > 0]
    if not sizes or sum(sizes) != int(total):
        return [slice(0, int(total))]
    out = []
    start = 0
    for size in sizes:
        out.append(slice(start, start + size))
        start += size
    return out


def _first_hit_target_for_group_control(
    lengths,
    y_true,
    *,
    velocity_sign_eps=0.0,
    dt_eps=1e-6,
    first_hit_tol=0.01,
):
    target = torch.zeros_like(y_true, dtype=torch.bool)
    tau_true = torch.full_like(y_true, float("inf"))
    contract = (y_true < -float(velocity_sign_eps)) & (lengths > 1e-8)
    if bool(contract.any().item()):
        tau_vals = lengths[contract].clamp_min(float(dt_eps)) / (
            -y_true[contract]
        ).clamp_min(float(dt_eps))
        tau_min = tau_vals.min()
        contract_idx = torch.nonzero(contract, as_tuple=False).reshape(-1)
        target[contract_idx[torch.abs(tau_vals - tau_min) <= float(first_hit_tol)]] = (
            True
        )
        tau_true[contract] = tau_vals
    return target, tau_true


def _full_path_control_velocity_loss(
    *,
    p,
    y,
    lengths,
    first_hit_logits,
    group_sizes,
    velocity_sign_eps=0.0,
    dt_eps=1e-6,
    first_hit_tol=0.01,
    first_hit_head_weight=0.0,
    logtau_all_weight=0.0,
    logtau_first_over_weight=0.0,
    logtau_first_tie_weight=0.0,
    logtau_predset_over_weight=0.0,
):
    zero = p.new_tensor(0.0)
    parts = {
        "mse": (p - y).pow(2).mean(),
        "firsthit_bce_raw": zero,
        "firsthit_bce": zero,
        "logtau_all_raw": zero,
        "logtau_all": zero,
        "logtau_first_over_raw": zero,
        "logtau_first_over": zero,
        "logtau_first_tie_raw": zero,
        "logtau_first_tie": zero,
        "logtau_predset_over_raw": zero,
        "logtau_predset_over": zero,
    }

    if first_hit_logits is not None and float(first_hit_head_weight) > 0.0:
        first_loss_raw, _stats = _first_hit_grouped_set_bce_loss(
            lengths=lengths,
            y_true=y,
            first_hit_logits=first_hit_logits,
            group_sizes=group_sizes,
            velocity_sign_eps=velocity_sign_eps,
            dt_eps=dt_eps,
            first_hit_tol=first_hit_tol,
        )
        parts["firsthit_bce_raw"] = first_loss_raw
        parts["firsthit_bce"] = float(first_hit_head_weight) * first_loss_raw

    logtau_all_losses = []
    logtau_first_over_losses = []
    logtau_first_tie_losses = []
    logtau_predset_over_losses = []
    for sl in _group_slices_from_sizes(group_sizes, int(p.numel())):
        pg = p[sl]
        yg = y[sl]
        lg = lengths[sl]
        logits_g = None if first_hit_logits is None else first_hit_logits[sl]
        target_first, tau_true_all = _first_hit_target_for_group_control(
            lg,
            yg,
            velocity_sign_eps=velocity_sign_eps,
            dt_eps=dt_eps,
            first_hit_tol=first_hit_tol,
        )
        contract = (yg < -float(velocity_sign_eps)) & (lg > 1e-8)
        if not bool(contract.any().item()):
            continue

        tau_pred = lg[contract].clamp_min(float(dt_eps)) / (
            -pg[contract]
        ).clamp_min(float(dt_eps))
        log_tau_pred = torch.log(tau_pred.clamp_min(float(dt_eps)))
        log_tau_true = torch.log(tau_true_all[contract].clamp_min(float(dt_eps)))
        logtau_all_losses.append(F.smooth_l1_loss(log_tau_pred, log_tau_true))

        if bool(target_first.any().item()):
            tau_pred_first = lg[target_first].clamp_min(float(dt_eps)) / (
                -pg[target_first]
            ).clamp_min(float(dt_eps))
            log_pred_first = torch.log(tau_pred_first.clamp_min(float(dt_eps)))
            log_true_first = torch.log(
                tau_true_all[target_first].clamp_min(float(dt_eps))
            )
            pred_max = log_pred_first.max()
            true_max = log_true_first.max()
            logtau_first_over_losses.append(F.relu(pred_max - true_max).pow(2))
            if int(log_pred_first.numel()) > 1:
                logtau_first_tie_losses.append(
                    (log_pred_first - log_pred_first.mean()).pow(2).mean()
                )
            if logits_g is not None:
                pred_contract = (pg < -float(velocity_sign_eps)) & (lg > 1e-8)
                if bool(pred_contract.any().item()):
                    tau_pred_contract = lg[pred_contract].clamp_min(float(dt_eps)) / (
                        -pg[pred_contract]
                    ).clamp_min(float(dt_eps))
                    log_pred_contract = torch.log(
                        tau_pred_contract.clamp_min(float(dt_eps))
                    )
                    pred_probs = torch.sigmoid(logits_g[pred_contract]).clamp_min(1e-6)
                    predset_over = F.relu(log_pred_contract - true_max).pow(2)
                    logtau_predset_over_losses.append(
                        (pred_probs * predset_over).sum()
                        / pred_probs.sum().clamp_min(1e-6)
                    )

    if logtau_all_losses and float(logtau_all_weight) > 0.0:
        parts["logtau_all_raw"] = torch.stack(logtau_all_losses).mean()
        parts["logtau_all"] = float(logtau_all_weight) * parts["logtau_all_raw"]
    if logtau_first_over_losses and float(logtau_first_over_weight) > 0.0:
        parts["logtau_first_over_raw"] = torch.stack(logtau_first_over_losses).mean()
        parts["logtau_first_over"] = (
            float(logtau_first_over_weight) * parts["logtau_first_over_raw"]
        )
    if logtau_first_tie_losses and float(logtau_first_tie_weight) > 0.0:
        parts["logtau_first_tie_raw"] = torch.stack(logtau_first_tie_losses).mean()
        parts["logtau_first_tie"] = (
            float(logtau_first_tie_weight) * parts["logtau_first_tie_raw"]
        )
    if logtau_predset_over_losses and float(logtau_predset_over_weight) > 0.0:
        parts["logtau_predset_over_raw"] = torch.stack(
            logtau_predset_over_losses
        ).mean()
        parts["logtau_predset_over"] = (
            float(logtau_predset_over_weight) * parts["logtau_predset_over_raw"]
        )

    total = (
        parts["mse"]
        + parts["firsthit_bce"]
        + parts["logtau_all"]
        + parts["logtau_first_over"]
        + parts["logtau_first_tie"]
        + parts["logtau_predset_over"]
    )
    return total, parts


def _first_hit_set_bce_loss(
    lengths,
    y_true,
    first_hit_logits,
    velocity_sign_eps=0.0,
    dt_eps=1e-6,
    first_hit_tol=0.01,
):
    zero = first_hit_logits.new_tensor(0.0)
    stats = {
        "n_candidates": int(first_hit_logits.numel()),
        "target_first_size": 0,
        "pred_first_size": 0,
        "top1_hits_first_set": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "jaccard": 0.0,
    }

    if first_hit_logits.numel() == 0:
        return zero, stats

    target = _first_hit_target(
        lengths,
        y_true,
        velocity_sign_eps=velocity_sign_eps,
        dt_eps=dt_eps,
        first_hit_tol=first_hit_tol,
    )

    pos = target.sum()
    if float(pos.item()) <= 0.0:
        return zero, stats

    neg = target.numel() - pos
    pos_weight = None
    if float(neg.item()) > 0.0:
        pos_weight = torch.clamp(neg / pos, min=1.0).detach()
    loss = F.binary_cross_entropy_with_logits(
        first_hit_logits,
        target,
        pos_weight=pos_weight,
    )

    pred_probs = torch.sigmoid(first_hit_logits)
    pred_mask = pred_probs > 0.5
    if int(pred_mask.sum().item()) == 0:
        pred_mask[torch.argmax(pred_probs)] = True

    top1_idx = int(torch.argmax(pred_probs).item())

    tp = (pred_mask & target.bool()).sum().float()
    pred_n = pred_mask.sum().float()
    true_n = target.sum().float()
    union = (pred_mask | target.bool()).sum().float()

    stats = {
        "n_candidates": int(first_hit_logits.numel()),
        "target_first_size": int(true_n.item()),
        "pred_first_size": int(pred_n.item()),
        "top1_hits_first_set": float(target[top1_idx].item()),
        "recall": float((tp / true_n.clamp_min(1.0)).item()),
        "precision": float((tp / pred_n.clamp_min(1.0)).item()),
        "jaccard": float((tp / union.clamp_min(1.0)).item()),
    }
    return loss, stats


def _slice_by_group_sizes(tensor, group_sizes):
    if tensor is None:
        return []
    total = int(tensor.numel())
    if not group_sizes:
        return [tensor]
    sizes = [int(size) for size in group_sizes if int(size) > 0]
    if sum(sizes) != total:
        return [tensor]

    groups = []
    start = 0
    for size in sizes:
        end = start + size
        groups.append(tensor[start:end])
        start = end
    return groups


def _first_hit_grouped_target(
    lengths,
    y_true,
    group_sizes=None,
    velocity_sign_eps=0.0,
    dt_eps=1e-6,
    first_hit_tol=0.01,
):
    length_groups = _slice_by_group_sizes(lengths, group_sizes)
    y_groups = _slice_by_group_sizes(y_true, group_sizes)
    if not length_groups or len(length_groups) != len(y_groups):
        return _first_hit_target(
            lengths,
            y_true,
            velocity_sign_eps=velocity_sign_eps,
            dt_eps=dt_eps,
            first_hit_tol=first_hit_tol,
        )

    targets = [
        _first_hit_target(
            group_lengths,
            group_y,
            velocity_sign_eps=velocity_sign_eps,
            dt_eps=dt_eps,
            first_hit_tol=first_hit_tol,
        )
        for group_lengths, group_y in zip(length_groups, y_groups)
    ]
    if not targets:
        return torch.zeros_like(lengths)
    return torch.cat(targets)


def _first_hit_grouped_set_bce_loss(
    lengths,
    y_true,
    first_hit_logits,
    group_sizes=None,
    velocity_sign_eps=0.0,
    dt_eps=1e-6,
    first_hit_tol=0.01,
):
    zero = first_hit_logits.new_tensor(0.0)
    stats = {
        "n_candidates": int(first_hit_logits.numel()),
        "target_first_size": 0,
        "pred_first_size": 0,
        "top1_hits_first_set": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "jaccard": 0.0,
    }
    if first_hit_logits.numel() == 0:
        return zero, stats

    length_groups = _slice_by_group_sizes(lengths, group_sizes)
    y_groups = _slice_by_group_sizes(y_true, group_sizes)
    logit_groups = _slice_by_group_sizes(first_hit_logits, group_sizes)
    if (
        not length_groups
        or len(length_groups) != len(y_groups)
        or len(length_groups) != len(logit_groups)
    ):
        return _first_hit_set_bce_loss(
            lengths=lengths,
            y_true=y_true,
            first_hit_logits=first_hit_logits,
            velocity_sign_eps=velocity_sign_eps,
            dt_eps=dt_eps,
            first_hit_tol=first_hit_tol,
        )

    losses = []
    total_candidates = 0
    total_target = 0.0
    total_pred = 0.0
    total_tp = 0.0
    total_union = 0.0
    top1_hits = 0.0
    valid_groups = 0
    for group_lengths, group_y, group_logits in zip(
        length_groups, y_groups, logit_groups
    ):
        total_candidates += int(group_logits.numel())
        target = _first_hit_target(
            group_lengths,
            group_y,
            velocity_sign_eps=velocity_sign_eps,
            dt_eps=dt_eps,
            first_hit_tol=first_hit_tol,
        )
        pos = target.sum()
        if float(pos.item()) <= 0.0:
            continue

        neg = target.numel() - pos
        pos_weight = None
        if float(neg.item()) > 0.0:
            pos_weight = torch.clamp(neg / pos, min=1.0).detach()
        losses.append(
            F.binary_cross_entropy_with_logits(
                group_logits,
                target,
                pos_weight=pos_weight,
            )
        )

        pred_probs = torch.sigmoid(group_logits)
        pred_mask = pred_probs > 0.5
        if int(pred_mask.sum().item()) == 0:
            pred_mask[torch.argmax(pred_probs)] = True

        top1_idx = int(torch.argmax(pred_probs).item())
        target_bool = target.bool()
        tp = (pred_mask & target_bool).sum().float()
        pred_n = pred_mask.sum().float()
        true_n = target.sum().float()
        union = (pred_mask | target_bool).sum().float()

        total_target += float(true_n.detach().item())
        total_pred += float(pred_n.detach().item())
        total_tp += float(tp.detach().item())
        total_union += float(union.detach().item())
        top1_hits += float(target[top1_idx].detach().item())
        valid_groups += 1

    if not losses:
        stats["n_candidates"] = total_candidates
        return zero, stats

    loss = torch.stack(losses).mean()
    stats = {
        "n_candidates": total_candidates,
        "target_first_size": int(total_target),
        "pred_first_size": int(total_pred),
        "top1_hits_first_set": top1_hits / max(valid_groups, 1),
        "recall": total_tp / max(total_target, 1.0),
        "precision": total_tp / max(total_pred, 1.0),
        "jaccard": total_tp / max(total_union, 1.0),
    }
    return loss, stats


def _first_hit_grouped_soft_mass_penalty(
    lengths,
    y_true,
    first_hit_logits,
    group_sizes=None,
    velocity_sign_eps=0.0,
    dt_eps=1e-6,
    first_hit_tol=0.01,
):
    target = _first_hit_grouped_target(
        lengths,
        y_true,
        group_sizes=group_sizes,
        velocity_sign_eps=velocity_sign_eps,
        dt_eps=dt_eps,
        first_hit_tol=first_hit_tol,
    )
    return _first_hit_soft_mass_penalty(first_hit_logits, target)


def _first_hit_target(
    lengths,
    y_true,
    velocity_sign_eps=0.0,
    dt_eps=1e-6,
    first_hit_tol=0.01,
):
    target = torch.zeros_like(lengths)
    if lengths.numel() == 0:
        return target

    candidate_mask = lengths > 1e-8
    contract_mask = (y_true < -float(velocity_sign_eps)) & candidate_mask
    if int(contract_mask.sum().item()) == 0:
        return target

    eps = float(dt_eps)
    tau_true = lengths[contract_mask].clamp_min(eps) / (
        -y_true[contract_mask]
    ).clamp_min(eps)
    tau_true_min = tau_true.min()
    first_contract_mask = torch.abs(tau_true - tau_true_min) <= float(first_hit_tol)
    contract_indices = torch.where(contract_mask)[0]
    target[contract_indices[first_contract_mask]] = 1.0
    return target


def _first_hit_soft_mass_penalty(
    first_hit_logits,
    target,
):
    zero = first_hit_logits.new_tensor(0.0)
    stats = {
        "fp_mass": 0.0,
        "fn_mass": 0.0,
        "cardinality_gap": 0.0,
    }
    if first_hit_logits.numel() == 0:
        return zero, stats
    pos = target.sum()
    if float(pos.item()) <= 0.0:
        return zero, stats

    probs = torch.sigmoid(first_hit_logits)
    denom = pos.clamp_min(1.0)
    fp_mass = (probs * (1.0 - target)).sum() / denom
    fn_mass = ((1.0 - probs) * target).sum() / denom
    cardinality_gap = torch.abs(probs.sum() - target.sum()) / denom
    penalty = fp_mass + fn_mass
    stats = {
        "fp_mass": float(fp_mass.detach().item()),
        "fn_mass": float(fn_mass.detach().item()),
        "cardinality_gap": float(cardinality_gap.detach().item()),
        "fp_mass_tensor": fp_mass,
        "fn_mass_tensor": fn_mass,
    }
    return penalty, stats


def _predict_first_hit_mask_from_logits(
    logits,
    candidate_mask,
    threshold=0.0,
    max_edges=-1,
):
    pred_mask = np.zeros_like(candidate_mask, dtype=bool)
    candidate_indices = np.where(candidate_mask)[0]
    if candidate_indices.size == 0:
        return pred_mask

    positive_indices = candidate_indices[logits[candidate_indices] > float(threshold)]
    if positive_indices.size > 0:
        chosen = positive_indices
        if int(max_edges) > 0 and positive_indices.size > int(max_edges):
            order = np.argsort(logits[positive_indices])[::-1]
            chosen = positive_indices[order[: int(max_edges)]]
        pred_mask[chosen] = True
        return pred_mask

    best_local = candidate_indices[int(np.argmax(logits[candidate_indices]))]
    pred_mask[best_local] = True
    return pred_mask


def _predict_first_hit_mask_with_fallback(
    logits,
    candidate_mask,
    threshold=0.0,
    max_edges=-1,
    fallback_threshold=-1,
    fallback_top_k=-1,
):
    # Never cap the sampled first-hit set. Training can predict a large
    # simultaneous event, and truncating it at sampling time invalidates the
    # learned movement program.
    raw_mask = _predict_first_hit_mask_from_logits(
        logits,
        candidate_mask,
        threshold=threshold,
        max_edges=-1,
    )
    raw_count = int(raw_mask.sum())
    return raw_mask, raw_count, False


def _sampling_supervised_candidate_mask(masks, lengths, n_leaves):
    candidate_mask = np.zeros(len(masks), dtype=bool)
    biological_bits = max(int(n_leaves) - 1, 0)
    for idx, (mask, length) in enumerate(zip(masks, lengths)):
        if float(length) <= 1e-8:
            continue
        k_bits = int(mask).bit_count()
        is_pendant = biological_bits > 0 and min(
            k_bits, biological_bits - k_bits
        ) == 1
        if not is_pendant:
            candidate_mask[idx] = True
    return candidate_mask


def _mask_precision_recall(pred_mask, true_mask):
    pred_mask = np.asarray(pred_mask, dtype=bool)
    true_mask = np.asarray(true_mask, dtype=bool)
    tp = float(np.logical_and(pred_mask, true_mask).sum())
    pred_n = float(pred_mask.sum())
    true_n = float(true_mask.sum())
    return {
        "precision": tp / max(pred_n, 1.0),
        "recall": tp / max(true_n, 1.0),
    }


def _oracle_first_hit_mask_for_sampling(
    current_newick,
    target_tree,
    *,
    masks,
    lengths,
    n_leaves,
    supervised_mask,
    velocity_sign_eps=0.0,
    dt_eps=1e-6,
    first_hit_tol=0.01,
):
    pred_mask = np.zeros_like(supervised_mask, dtype=bool)
    if not current_newick or not target_tree:
        return pred_mask

    try:
        _, true_velocity = return_sampled_tree_orthant_velocity(
            current_newick,
            target_tree,
            0.0,
        )
    except Exception:
        return pred_mask

    if not true_velocity:
        return pred_mask

    split_masks_num = [int(m) for m in masks]
    split_masks_nonzero = [m for m in split_masks_num if m != 0]
    if not split_masks_nonzero:
        return pred_mask

    real_max_bit = max(m.bit_length() for m in split_masks_nonzero)
    full_mask = (1 << real_max_bit) - 1 if real_max_bit > 0 else 0
    mask_to_idx = {m: i for i, m in enumerate(split_masks_num)}
    y_true = np.zeros(len(split_masks_num), dtype=np.float64)

    for vel_mask, true_vel in true_velocity.items():
        vel = int(vel_mask)
        if vel.bit_length() == real_max_bit + 1:
            vel = remove_bit(vel, n_leaves - 1)
        elif vel.bit_length() > real_max_bit + 1:
            continue

        matched_vel = vel
        if matched_vel not in mask_to_idx and full_mask:
            complement_vel = full_mask ^ matched_vel
            if complement_vel in mask_to_idx:
                matched_vel = complement_vel
            else:
                continue
        elif matched_vel not in mask_to_idx:
            continue

        k = int(matched_vel).bit_count()
        is_pendant = min(k, real_max_bit - k) == 1
        if is_pendant:
            continue

        idx = mask_to_idx[int(matched_vel)]
        y_true[idx] = float(true_vel)

    candidate_mask = supervised_mask & (lengths > 1e-8)
    contract_mask = candidate_mask & (y_true < -float(velocity_sign_eps))
    if not np.any(contract_mask):
        return pred_mask

    tau_true = lengths[contract_mask].clip(min=float(dt_eps)) / np.maximum(
        -y_true[contract_mask],
        float(dt_eps),
    )
    tau_true_min = float(np.min(tau_true))
    first_contract_mask = np.abs(tau_true - tau_true_min) <= float(first_hit_tol)
    contract_indices = np.where(contract_mask)[0]
    pred_mask[contract_indices[first_contract_mask]] = True
    return pred_mask


def _edge_set_bce_loss(logits, target):
    zero = logits.new_tensor(0.0)
    stats = {
        "n_candidates": int(logits.numel()),
        "target_size": 0,
        "pred_size": 0,
        "top1_hits_target_set": 0.0,
        "recall": 0.0,
        "precision": 0.0,
        "jaccard": 0.0,
    }

    if logits.numel() == 0 or target.numel() == 0 or logits.numel() != target.numel():
        return zero, stats

    target = target.float()
    pos = target.sum()
    if float(pos.item()) <= 0.0:
        return zero, stats

    neg = target.numel() - pos
    pos_weight = None
    if float(neg.item()) > 0.0:
        pos_weight = (neg / pos).detach()
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

    pred_probs = torch.sigmoid(logits)
    pred_mask = pred_probs > 0.5
    if int(pred_mask.sum().item()) == 0:
        pred_mask[torch.argmax(pred_probs)] = True

    top1_idx = int(torch.argmax(pred_probs).item())
    tp = (pred_mask & target.bool()).sum().float()
    pred_n = pred_mask.sum().float()
    true_n = target.sum().float()
    union = (pred_mask | target.bool()).sum().float()

    stats = {
        "n_candidates": int(logits.numel()),
        "target_size": int(true_n.item()),
        "pred_size": int(pred_n.item()),
        "top1_hits_target_set": float(target[top1_idx].item()),
        "recall": float((tp / true_n.clamp_min(1.0)).item()),
        "precision": float((tp / pred_n.clamp_min(1.0)).item()),
        "jaccard": float((tp / union.clamp_min(1.0)).item()),
    }
    return loss, stats


def _recompute_next_boundary_active_masks_from_newick(
    next_boundary_tree,
    *,
    masks,
    n_leaves,
):
    active_masks = set()
    if not next_boundary_tree:
        return active_masks

    enc = BHVEncoder()
    try:
        boundary_tree_obj = Tree(next_boundary_tree)
        boundary_masks, boundary_lengths = enc.return_BHV_encoding(boundary_tree_obj)
    except Exception:
        return active_masks

    split_masks_num = [int(m) for m in masks]
    split_masks_nonzero = [m for m in split_masks_num if m != 0]
    if not split_masks_nonzero:
        return active_masks

    real_max_bit = max(m.bit_length() for m in split_masks_nonzero)
    full_mask = (1 << real_max_bit) - 1 if real_max_bit > 0 else 0
    mask_to_idx = {m: i for i, m in enumerate(split_masks_num)}

    for boundary_mask, boundary_length in zip(boundary_masks, boundary_lengths):
        if boundary_length is None or float(boundary_length) <= 1e-8:
            continue

        boundary_mask = int(boundary_mask)
        if boundary_mask.bit_length() == real_max_bit + 1:
            boundary_mask = remove_bit(boundary_mask, n_leaves - 1)
        elif boundary_mask.bit_length() > real_max_bit + 1:
            continue

        matched_boundary_mask = boundary_mask
        if matched_boundary_mask not in mask_to_idx and full_mask:
            complement_boundary_mask = full_mask ^ matched_boundary_mask
            if complement_boundary_mask in mask_to_idx:
                matched_boundary_mask = complement_boundary_mask
            else:
                continue
        elif matched_boundary_mask not in mask_to_idx:
            continue

        active_masks.add(int(matched_boundary_mask))

    return active_masks


def _summarize_fixed_pair_eval_rows(velocity_rows, ar_rows):
    n_velocity = int(len(velocity_rows))
    n_ar = int(len(ar_rows))

    first_hit_exact = sum(
        1
        for row in velocity_rows
        if float(row.get("first_hit_precision", 0.0)) == 1.0
        and float(row.get("first_hit_recall", 0.0)) == 1.0
    )
    vanish_exact = sum(
        1
        for row in velocity_rows
        if float(row.get("vanish_precision", 0.0)) == 1.0
        and float(row.get("vanish_recall", 0.0)) == 1.0
    )
    velocity_joint_exact = sum(
        1
        for row in velocity_rows
        if float(row.get("first_hit_precision", 0.0)) == 1.0
        and float(row.get("first_hit_recall", 0.0)) == 1.0
        and float(row.get("vanish_precision", 0.0)) == 1.0
        and float(row.get("vanish_recall", 0.0)) == 1.0
    )
    ar_exact = sum(1 for row in ar_rows if bool(row.get("exact_match", False)))

    first_wrong_velocity_index = next(
        (
            int(row.get("index", idx))
            for idx, row in enumerate(velocity_rows)
            if not (
                float(row.get("first_hit_precision", 0.0)) == 1.0
                and float(row.get("first_hit_recall", 0.0)) == 1.0
                and float(row.get("vanish_precision", 0.0)) == 1.0
                and float(row.get("vanish_recall", 0.0)) == 1.0
            )
        ),
        -1,
    )
    first_wrong_first_hit_index = next(
        (
            int(row.get("index", idx))
            for idx, row in enumerate(velocity_rows)
            if not (
                float(row.get("first_hit_precision", 0.0)) == 1.0
                and float(row.get("first_hit_recall", 0.0)) == 1.0
            )
        ),
        -1,
    )
    first_wrong_vanish_index = next(
        (
            int(row.get("index", idx))
            for idx, row in enumerate(velocity_rows)
            if not (
                float(row.get("vanish_precision", 0.0)) == 1.0
                and float(row.get("vanish_recall", 0.0)) == 1.0
            )
        ),
        -1,
    )
    first_wrong_ar_index = next(
        (
            int(row.get("event_index", idx))
            for idx, row in enumerate(ar_rows)
            if not bool(row.get("exact_match", False))
        ),
        -1,
    )

    def _frac(numer, denom):
        if int(denom) <= 0:
            return 0.0
        return float(numer) / float(denom)

    return {
        "fixed_path_num_velocity_states": float(n_velocity),
        "fixed_path_num_autoregressive_events": float(n_ar),
        "fixed_path_velocity_first_hit_exact_frac": _frac(first_hit_exact, n_velocity),
        "fixed_path_velocity_vanish_exact_frac": _frac(vanish_exact, n_velocity),
        "fixed_path_velocity_joint_exact_frac": _frac(
            velocity_joint_exact, n_velocity
        ),
        "fixed_path_autoregressive_exact_frac": _frac(ar_exact, n_ar),
        "fixed_path_first_wrong_velocity_index": float(first_wrong_velocity_index),
        "fixed_path_first_wrong_first_hit_index": float(first_wrong_first_hit_index),
        "fixed_path_first_wrong_vanish_index": float(first_wrong_vanish_index),
        "fixed_path_first_wrong_autoregressive_index": float(first_wrong_ar_index),
    }


def _select_structured_subset_size(size_logits, max_group_size, allow_zero=True):
    if size_logits is None or size_logits.numel() == 0:
        return None

    masked_logits = size_logits.detach().clone()
    max_group_size = max(int(max_group_size), 0)
    max_valid_size = min(max_group_size, int(masked_logits.numel()) - 1)
    if max_valid_size + 1 < masked_logits.numel():
        masked_logits[max_valid_size + 1 :] = float("-inf")

    # Cardinality 1 is never a valid merge target.
    if masked_logits.numel() > 1:
        masked_logits[1] = float("-inf")
    if not allow_zero and masked_logits.numel() > 0:
        masked_logits[0] = float("-inf")

    if not torch.isfinite(masked_logits).any():
        if allow_zero:
            return 0
        return 2 if max_valid_size >= 2 else 0

    return int(torch.argmax(masked_logits).item())


def _ensure_structured_subset_full_member_logits(group_output):
    if group_output.get("_structured_subset_top_member_pairs") is None:
        return group_output
    head = group_output.get("_structured_subset_head")
    group_embeddings = group_output.get("group_embeddings")
    if head is None or group_embeddings is None:
        return group_output

    full_outputs = head(
        group_embeddings,
        context=group_output.get("_structured_subset_context"),
        top_member_pairs=None,
    )
    group_output.update(full_outputs)
    group_output["_structured_subset_top_member_pairs"] = None
    return group_output


def _structured_subset_from_pair_and_size(group_output, pair_id, subset_size, splits=None):
    pair_indices = group_output.get("starter_pair_indices")
    member_logits = group_output.get("member_logits")
    if splits is None:
        splits = [int(split) for split in group_output["splits_represented"]]
    else:
        splits = [int(split) for split in splits]
    if (
        pair_indices is None
        or member_logits is None
        or not pair_indices
        or int(pair_id) >= len(pair_indices)
    ):
        return tuple()

    subset_size = int(subset_size)
    if subset_size <= 0:
        return tuple()

    left_idx, right_idx = pair_indices[int(pair_id)]
    selected_indices = [int(left_idx), int(right_idx)]
    selected_set = set(selected_indices)

    if subset_size == 1:
        subset_size = 2
    subset_size = min(max(subset_size, 2), len(splits))

    extra_needed = subset_size - 2
    if extra_needed > 0:
        member_row = member_logits[int(pair_id)].detach()
        remaining_indices = [
            idx for idx in range(len(splits)) if idx not in selected_set
        ]
        remaining_indices.sort(
            key=lambda idx: float(member_row[idx].item()),
            reverse=True,
        )
        selected_indices.extend(remaining_indices[:extra_needed])

    return tuple(sorted(int(splits[node_idx]) for node_idx in selected_indices))


def _structured_size_loss_and_prediction(size_logits, target_sizes, max_group_size):
    if size_logits is None or size_logits.numel() == 0:
        return None

    target_sizes = [int(size) for size in target_sizes]
    max_class = int(size_logits.numel()) - 1
    clipped_targets = sorted(
        {
            min(max(size, 0), max_class)
            for size in target_sizes
            if int(size) != 1
        }
    )
    if not clipped_targets:
        return None

    target_tensor = torch.tensor(
        clipped_targets,
        dtype=torch.long,
        device=size_logits.device,
    )
    size_log_probs = F.log_softmax(size_logits, dim=0)
    size_loss = -torch.logsumexp(size_log_probs[target_tensor], dim=0)
    predicted_size = _select_structured_subset_size(
        size_logits,
        max_group_size=max_group_size,
        allow_zero=True,
    )

    return {
        "loss": size_loss,
        "predicted_size": int(predicted_size),
        "target_sizes": clipped_targets,
    }


def _decode_structured_merge_subset(group_output, member_threshold_logit=0.0):
    pair_logits = group_output.get("starter_pair_logits")
    pair_indices = group_output.get("starter_pair_indices")
    member_logits = group_output.get("member_logits")
    size_logits = group_output.get("subset_size_logits")
    splits = group_output.get("_splits_represented_ints")
    if splits is None:
        splits = [int(split) for split in group_output["splits_represented"]]
        group_output["_splits_represented_ints"] = splits
    if (
        pair_logits is None
        or member_logits is None
        or not pair_indices
        or pair_logits.numel() == 0
    ):
        return None

    best_pair_idx = int(torch.argmax(pair_logits).item())
    left_idx, right_idx = pair_indices[best_pair_idx]
    predicted_size = _select_structured_subset_size(
        size_logits,
        max_group_size=len(splits),
        allow_zero=True,
    )
    if predicted_size is None:
        left_idx, right_idx = pair_indices[best_pair_idx]
        selected_indices = {int(left_idx), int(right_idx)}
        member_row = member_logits[best_pair_idx].detach()
        for node_idx in range(len(splits)):
            if node_idx in selected_indices:
                continue
            score = float(member_row[node_idx].item())
            if math.isfinite(score) and score > float(member_threshold_logit):
                selected_indices.add(int(node_idx))
        subset = tuple(sorted(int(splits[node_idx]) for node_idx in selected_indices))
    else:
        subset = _structured_subset_from_pair_and_size(
            group_output,
            best_pair_idx,
            predicted_size,
            splits=splits,
        )
    if len(subset) < 2:
        return None

    new_split = 0
    for component in subset:
        new_split |= int(component)

    return {
        "subset": subset,
        "new_split": int(new_split),
        "best_pair_index": best_pair_idx,
        "best_pair": (int(left_idx), int(right_idx)),
        "pair_logit": float(pair_logits[best_pair_idx].detach().item()),
        "stop_after_merge_logit": float(
            group_output.get("stop_after_merge_logit", torch.tensor(0.0))
            .detach()
            .cpu()
            .item()
        ),
        "prediction_logits": _subset_prediction_logits(
            splits,
            subset,
            device=member_logits.device,
        ),
    }


def _ranked_structured_merge_subset_candidates(group_output, member_threshold_logit=0.0):
    group_output = _ensure_structured_subset_full_member_logits(group_output)
    pair_logits = group_output.get("starter_pair_logits")
    pair_indices = group_output.get("starter_pair_indices")
    member_logits = group_output.get("member_logits")
    size_logits = group_output.get("subset_size_logits")
    splits = group_output.get("_splits_represented_ints")
    if splits is None:
        splits = [int(split) for split in group_output["splits_represented"]]
        group_output["_splits_represented_ints"] = splits
    if (
        pair_logits is None
        or member_logits is None
        or not pair_indices
        or pair_logits.numel() == 0
    ):
        return []

    ranked_candidates = []
    seen_subsets = set()
    predicted_size = _select_structured_subset_size(
        size_logits,
        max_group_size=len(splits),
        allow_zero=True,
    )
    sorted_pair_ids = torch.argsort(pair_logits.detach(), descending=True)
    for pair_id in sorted_pair_ids.tolist():
        left_idx, right_idx = pair_indices[int(pair_id)]
        if predicted_size is None:
            selected_indices = {int(left_idx), int(right_idx)}
            member_row = member_logits[int(pair_id)].detach()
            for node_idx in range(len(splits)):
                if node_idx in selected_indices:
                    continue
                score = float(member_row[node_idx].item())
                if math.isfinite(score) and score > float(member_threshold_logit):
                    selected_indices.add(int(node_idx))
            subset = tuple(sorted(int(splits[node_idx]) for node_idx in selected_indices))
        else:
            subset = _structured_subset_from_pair_and_size(
                group_output,
                pair_id,
                predicted_size,
            )
        if len(subset) < 2 or subset in seen_subsets:
            continue
        seen_subsets.add(subset)

        new_split = 0
        for component in subset:
            new_split |= int(component)

        ranked_candidates.append(
            {
                "subset": subset,
                "new_split": int(new_split),
                "best_pair_index": int(pair_id),
                "best_pair": (int(left_idx), int(right_idx)),
                "pair_logit": float(pair_logits[int(pair_id)].detach().item()),
                "stop_after_merge_logit": float(
                    group_output.get("stop_after_merge_logit", torch.tensor(0.0))
                    .detach()
                    .cpu()
                    .item()
                ),
                "prediction_logits": _subset_prediction_logits(
                    splits,
                    subset,
                    device=member_logits.device,
                ),
            }
        )

    return ranked_candidates


def _best_autoregressive_fallback_candidate(
    logit_outputs,
    existing_splits,
    planned_new_splits,
    forbidden_new_splits=None,
    threshold_logit=0.0,
):
    forbidden_new_splits = (
        {int(split) for split in forbidden_new_splits}
        if forbidden_new_splits is not None
        else set()
    )
    best_candidate = None
    best_rank = None

    for output in logit_outputs:
        polytomy_score = float(output["polytomy_pred"].detach().cpu().item())
        decoder_mode = str(output.get("decoder_mode", "pairwise_threshold"))
        splits_represented = output.get("_splits_represented_ints")
        if splits_represented is None:
            splits_represented = [int(split) for split in output["splits_represented"]]
            output["_splits_represented_ints"] = splits_represented

        if decoder_mode == "structured_subset":
            ranked_candidates = _ranked_structured_merge_subset_candidates(
                output,
                member_threshold_logit=threshold_logit,
            )
            for candidate in ranked_candidates:
                new_split = int(candidate["new_split"])
                if (
                    new_split in existing_splits
                    or new_split in planned_new_splits
                ):
                    continue

                rank = (polytomy_score, float(candidate["pair_logit"]))
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_candidate = {
                        "polytomy_score": polytomy_score,
                        "splits_represented": splits_represented,
                        "subsets": [(candidate["subset"], candidate["new_split"])],
                        "logits": candidate["prediction_logits"],
                        "stop_after_merge_logit": float(
                            candidate.get("stop_after_merge_logit", 0.0)
                        ),
                        "decoder_mode": "structured_subset",
                        "fallback": True,
                    }
                break
            continue

        logits = output["logits"].detach()
        pair_candidates = []
        for left_idx in range(len(splits_represented)):
            for right_idx in range(left_idx + 1, len(splits_represented)):
                score = float(logits[left_idx, right_idx].item())
                if not math.isfinite(score):
                    continue
                subset = tuple(
                    sorted(
                        (
                            int(splits_represented[left_idx]),
                            int(splits_represented[right_idx]),
                        )
                    )
                )
                new_split = int(subset[0]) | int(subset[1])
                if (
                    new_split in existing_splits
                    or new_split in planned_new_splits
                    or new_split in forbidden_new_splits
                ):
                    continue
                pair_candidates.append((score, subset, int(new_split)))

        if not pair_candidates:
            continue

        score, subset, new_split = max(pair_candidates, key=lambda item: item[0])
        rank = (polytomy_score, float(score))
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_candidate = {
                "polytomy_score": polytomy_score,
                "splits_represented": splits_represented,
                "subsets": [(subset, new_split)],
                "logits": _subset_prediction_logits(
                    splits_represented,
                    subset,
                    device=logits.device,
                ),
                "decoder_mode": "pairwise_threshold",
                "fallback": True,
            }

    return best_candidate


def _combine_autoregressive_losses(
    merge_loss: torch.Tensor,
    polytomy_choosing_loss: torch.Tensor | None,
    polytomy_choosing_weight: float,
) -> torch.Tensor:
    total_loss = merge_loss
    if (
        polytomy_choosing_loss is not None
        and float(polytomy_choosing_weight) != 0.0
    ):
        total_loss = total_loss + (
            float(polytomy_choosing_weight) * polytomy_choosing_loss
        )
    return total_loss


def _structured_subset_loss_and_prediction(
    group_output,
    group_splits,
    subset,
    include_metric_logits=True,
):
    pair_logits = group_output["starter_pair_logits"]
    pair_indices = group_output["starter_pair_indices"]
    member_logits = group_output["member_logits"]
    size_logits = group_output.get("subset_size_logits")
    subset = {int(split) for split in subset}
    if pair_logits.numel() == 0 or not pair_indices:
        return None

    target_members = torch.tensor(
        [1.0 if int(split) in subset else 0.0 for split in group_splits],
        dtype=torch.float32,
        device=pair_logits.device,
    )
    valid_pair_ids = []
    for pair_id, (left_idx, right_idx) in enumerate(pair_indices):
        left_split = int(group_splits[int(left_idx)])
        right_split = int(group_splits[int(right_idx)])
        if left_split in subset and right_split in subset:
            valid_pair_ids.append(int(pair_id))

    if not valid_pair_ids:
        return None

    valid_pair_tensor = torch.tensor(
        valid_pair_ids,
        dtype=torch.long,
        device=pair_logits.device,
    )
    pair_log_probs = F.log_softmax(pair_logits, dim=0)
    pair_loss = -torch.logsumexp(pair_log_probs[valid_pair_tensor], dim=0)

    target_member_matrix = target_members.unsqueeze(0).expand(len(valid_pair_ids), -1)
    member_loss_matrix = F.binary_cross_entropy_with_logits(
        member_logits[valid_pair_tensor],
        target_member_matrix,
        reduction="none",
    )
    member_losses = member_loss_matrix.mean(dim=1)
    if member_losses.numel() == 1:
        member_loss = member_losses[0]
    else:
        member_loss = -torch.logsumexp(-member_losses, dim=0) + math.log(
            float(member_losses.numel())
        )

    best_pair_id = valid_pair_ids[int(torch.argmin(member_losses).item())]
    size_info = _structured_size_loss_and_prediction(
        size_logits,
        target_sizes=[len(subset)],
        max_group_size=len(group_splits),
    )
    size_loss = (
        size_info["loss"]
        if size_info is not None
        else pair_logits.new_tensor(0.0)
    )
    pred_pair_idx = int(torch.argmax(pair_logits).item())
    if size_info is None:
        pred_left_idx, pred_right_idx = pair_indices[pred_pair_idx]
        pred_selected = {int(pred_left_idx), int(pred_right_idx)}
        pred_member_row = member_logits[pred_pair_idx].detach()
        for node_idx in range(len(group_splits)):
            if node_idx in pred_selected:
                continue
            score = float(pred_member_row[node_idx].item())
            if math.isfinite(score) and score > 0.0:
                pred_selected.add(int(node_idx))
        predicted_subset = tuple(
            sorted(int(group_splits[node_idx]) for node_idx in pred_selected)
        )
        predicted_size = len(predicted_subset)
    else:
        predicted_size = int(size_info["predicted_size"])
        predicted_subset = _structured_subset_from_pair_and_size(
            {
                **group_output,
                "splits_represented": tuple(int(split) for split in group_splits),
            },
            pred_pair_idx,
            predicted_size,
        )

    result = {
        "loss": pair_loss + member_loss + size_loss,
        "best_pair_id": int(best_pair_id),
        "predicted_subset": predicted_subset,
        "predicted_size": int(predicted_size),
        "target_size": int(len(subset)),
        "stop_after_merge_logit": group_output.get("stop_after_merge_logit"),
    }
    if include_metric_logits:
        result["target_logits"] = _subset_target_matrix(
            group_splits,
            tuple(sorted(subset)),
            pair_logits.device,
        )
        result["predicted_logits"] = _subset_prediction_logits(
            group_splits,
            predicted_subset,
            device=pair_logits.device,
        )
    else:
        result["target_logits"] = None
        result["predicted_logits"] = None
    return result


def _plan_autoregressive_boundary_merges(
    logit_outputs,
    existing_splits,
    forbidden_new_splits=None,
    threshold_logit=0.0,
    top_only=False,
):
    existing_splits = {int(split) for split in existing_splits}
    forbidden_new_splits = (
        {int(split) for split in forbidden_new_splits}
        if forbidden_new_splits is not None
        else set()
    )
    outputs_sorted = sorted(
        logit_outputs,
        key=lambda output: float(output["polytomy_pred"].detach().cpu().item()),
        reverse=True,
    )

    planned = []
    planned_new_splits = set()
    for output in outputs_sorted:
        polytomy_score = float(output["polytomy_pred"].detach().cpu().item())
        if (len(logit_outputs) != 1) and polytomy_score <= float(threshold_logit):
            continue

        if output.get("decoder_mode") == "structured_subset":
            decoded = _decode_structured_merge_subset(
                output,
                member_threshold_logit=threshold_logit,
            )
            if decoded is None:
                continue
            if (
                decoded["new_split"] in existing_splits
                or decoded["new_split"] in planned_new_splits
                or decoded["new_split"] in forbidden_new_splits
            ):
                continue

            planned_new_splits.add(int(decoded["new_split"]))
            planned_item = {
                "polytomy_score": polytomy_score,
                "splits_represented": [
                    int(split) for split in output["splits_represented"]
                ],
                "subsets": [(decoded["subset"], decoded["new_split"])],
                "logits": decoded["prediction_logits"],
                "stop_after_merge_logit": float(
                    decoded.get("stop_after_merge_logit", 0.0)
                ),
                "decoder_mode": "structured_subset",
            }
            if top_only:
                return [planned_item]
            planned.append(planned_item)
            continue

        valid_subsets = []
        for subset in _decode_positive_merge_subsets(output, threshold_logit=threshold_logit):
            new_split = 0
            for component in subset:
                new_split |= int(component)

            if (
                new_split in existing_splits
                or new_split in planned_new_splits
                or new_split in forbidden_new_splits
            ):
                continue

            valid_subsets.append((subset, int(new_split)))

        if valid_subsets:
            best_subset = max(
                valid_subsets,
                key=lambda item: _score_merge_subset(output, item[0]),
            )
            planned_new_splits.add(int(best_subset[1]))
            planned_item = {
                "polytomy_score": polytomy_score,
                "splits_represented": [
                    int(split) for split in output["splits_represented"]
                ],
                "subsets": [best_subset],
                "logits": output["logits"],
            }
            if top_only:
                return [planned_item]
            planned.append(planned_item)

    if not planned:
        fallback = _best_autoregressive_fallback_candidate(
            outputs_sorted,
            existing_splits,
            planned_new_splits,
            forbidden_new_splits=forbidden_new_splits,
            threshold_logit=threshold_logit,
        )
        if fallback is not None:
            planned.append(fallback)

    return planned


def _is_strict_subset_mask(mask, region):
    mask = int(mask)
    region = int(region)
    return mask != region and (mask & ~region) == 0


def _move_tokenized_batch_to_device(tokenized, device):
    moved = []
    for item in tokenized:
        if torch.is_tensor(item):
            moved.append(item.to(device))
        else:
            moved.append(item)
    return tuple(moved)


def _tokenizer_structural_cache(module):
    cache = getattr(module, "_tree_tokenizer_structural_cache", None)
    if cache is None:
        cache = {}
        module._tree_tokenizer_structural_cache = cache
    return cache


def _tokenizer_tokenized_cache(module):
    cache = getattr(module, "_tree_tokenizer_tokenized_cache", None)
    if cache is None:
        cache = OrderedDict()
        module._tree_tokenizer_tokenized_cache = cache
    return cache


def _structuralize_trees_with_cache(module, trees):
    tokenizer = module.model.tokenizer
    cache = _tokenizer_structural_cache(module)
    structural_trees = []
    for tree in trees:
        key = str(tree)
        structural = cache.get(key)
        if structural is None:
            structural = tokenizer._newick_to_structural(key)
            cache[key] = structural
        structural_trees.append(structural)
    return structural_trees


def _tokenize_trees_with_structural_cache(module, trees):
    if (
        len(trees) == 1
        and bool(getattr(module, "sampling_tokenized_tree_cache_enabled", True))
    ):
        key = str(trees[0])
        cache = _tokenizer_tokenized_cache(module)
        cached = cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
            module._tree_tokenizer_tokenized_cache_hits = (
                int(getattr(module, "_tree_tokenizer_tokenized_cache_hits", 0)) + 1
            )
            return cached
        module._tree_tokenizer_tokenized_cache_misses = (
            int(getattr(module, "_tree_tokenizer_tokenized_cache_misses", 0)) + 1
        )
        tokenized = module.model.tokenizer(_structuralize_trees_with_cache(module, trees))
        cache[key] = tokenized
        max_entries = int(
            getattr(module, "sampling_tokenized_tree_cache_max_entries", 512)
        )
        while max_entries > 0 and len(cache) > max_entries:
            cache.popitem(last=False)
        return tokenized
    return module.model.tokenizer(_structuralize_trees_with_cache(module, trees))


def _build_start_topology_feature_tensor(module, start_trees, *, device=None, dtype=None):
    if not start_trees:
        return None
    if device is None:
        device = module.device
    if dtype is None:
        dtype = next(module.model.parameters()).dtype
    embed_dim = int(module.model.embed_dim)
    zero = torch.zeros(3 * embed_dim, device=device, dtype=dtype)
    features = []
    for start_tree in start_trees:
        if start_tree is None:
            features.append(zero.clone())
            continue
        try:
            tokenized = module.model.tokenizer([str(start_tree)])
            raw_masks = tokenized[-1][0]
            masks = [int(mask) for mask in raw_masks if int(mask) != 0]
            if not masks:
                features.append(zero.clone())
                continue
            split_identity = module.model.create_split_identity_embedding(
                masks,
                device=device,
            ).to(device=device, dtype=dtype)
            if split_identity.numel() == 0:
                features.append(zero.clone())
                continue
            start_sum = split_identity.sum(dim=0)
            start_mean = split_identity.mean(dim=0)
            start_max = split_identity.max(dim=0).values
            features.append(torch.cat([start_sum, start_mean, start_max], dim=0))
        except Exception:
            features.append(zero.clone())
    return torch.stack(features, dim=0)


def _build_start_topology_identity_batch(module, start_trees, *, device=None, dtype=None):
    if not start_trees:
        return None, None
    if device is None:
        device = module.device
    if dtype is None:
        dtype = next(module.model.parameters()).dtype
    embed_dim = int(module.model.embed_dim)
    identity_rows = []
    max_len = 0
    for start_tree in start_trees:
        identity = None
        if start_tree is not None:
            try:
                tokenized = module.model.tokenizer([str(start_tree)])
                raw_masks = tokenized[-1][0]
                masks = [int(mask) for mask in raw_masks if int(mask) != 0]
                if masks:
                    identity = module.model.create_split_identity_embedding(
                        masks,
                        device=device,
                    ).to(device=device, dtype=dtype)
            except Exception:
                identity = None
        if identity is None or identity.numel() == 0:
            identity = torch.zeros(0, embed_dim, device=device, dtype=dtype)
        identity_rows.append(identity)
        max_len = max(max_len, int(identity.shape[0]))
    max_len = max(max_len, 1)
    padded = torch.zeros(
        len(identity_rows),
        max_len,
        embed_dim,
        device=device,
        dtype=dtype,
    )
    pad_mask = torch.ones(
        len(identity_rows),
        max_len,
        device=device,
        dtype=torch.bool,
    )
    for row_idx, identity in enumerate(identity_rows):
        n = int(identity.shape[0])
        if n <= 0:
            continue
        padded[row_idx, :n] = identity
        pad_mask[row_idx, :n] = False
    return padded, pad_mask


def _build_start_tree_graph_context(
    module,
    start_trees,
    phyla_embeddings,
    *,
    device=None,
    detach=None,
):
    if not start_trees:
        return None
    if device is None:
        device = module.device
    if detach is None:
        detach = bool(
            getattr(module.model, "first_hit_start_tree_graph_detach", False)
        )
    tokenized = module.model.tokenizer([str(tree) for tree in start_trees])
    tokenized = _move_tokenized_batch_to_device(tokenized, device)
    if detach:
        with torch.no_grad():
            start_tokens = module.model(
                tokenized,
                t=None,
                phyla_embeddings=phyla_embeddings,
                return_all_tokens=True,
            )
        return start_tokens[:, 0, :].detach()
    start_tokens = module.model(
        tokenized,
        t=None,
        phyla_embeddings=phyla_embeddings,
        return_all_tokens=True,
    )
    return start_tokens[:, 0, :]


def _resolve_replay_phyla_embeddings(module, samples, tree_key):
    if not samples:
        return None
    has_precomputed = (
        getattr(module, "phyla_precomputed_name_to_embedding", None)
        or getattr(module, "phyla_precomputed_by_dataset_id", None)
    )
    has_live_model = getattr(module, "phyla_model", None) is not None
    if not has_precomputed and not has_live_model:
        return None

    embeddings = []
    for sample in samples:
        tree_newick = sample.get(tree_key)
        if tree_newick is None:
            return None
        mapping = sample.get("num_to_name") or sample.get("mapping")
        num_leaf = sample.get("num_leaves")
        resolved = None
        if has_precomputed:
            resolved = module._resolve_precomputed_phyla_embeddings_for_tree(
                str(tree_newick),
                mapping=mapping,
                num_leaf=num_leaf,
                device=module.device,
                dataset_id=sample.get("dataset_id"),
            )
        if resolved is None and has_live_model:
            names = module._ordered_leaf_names_from_mapping(
                mapping,
                num_leaf=num_leaf,
            )
            sequences = sample.get("sequences") or {}
            if names and sequences:
                seqs = []
                for idx, _name in enumerate(names):
                    seqs.append(
                        sequences.get(str(idx), sequences.get(idx, ""))
                    )
                resolved = module.compute_phyla_embeddings(
                    seqs,
                    names,
                    device=str(module.device),
                )
        if resolved is None:
            return None
        if resolved.dim() == 3 and int(resolved.size(0)) == 1:
            resolved = resolved.squeeze(0)
        embeddings.append(resolved)

    if not embeddings:
        return None
    first_shape = tuple(embeddings[0].shape)
    if all(tuple(embedding.shape) == first_shape for embedding in embeddings):
        return torch.stack(embeddings, dim=0)
    return embeddings


def _target_splits_from_velocity_sample(sample):
    tree_obj = Tree(sample["newick_tree"])
    encoder = BHVEncoder()
    masks, lengths = encoder.return_BHV_encoding(tree_obj)
    len_map = {int(m): float(l) for m, l in zip(masks, lengths) if l is not None}
    model_masks = [
        int(m)
        for m, l in zip(masks, lengths)
        if l is not None and float(l) > 1e-8 and int(m) != 0
    ]
    if not model_masks:
        return []
    real_max_bit = max(int(m).bit_length() for m in model_masks)
    full_mask = (1 << real_max_bit) - 1 if real_max_bit > 0 else 0
    taus = []
    matched = []
    for orig_mask, vel in sample.get("velocity", {}).items():
        vel = float(vel)
        mask = int(orig_mask)
        if mask.bit_length() == real_max_bit + 1:
            mask = remove_bit(mask, int(sample["num_leaves"]) - 1)
        elif mask.bit_length() > real_max_bit + 1:
            continue
        matched_mask = mask
        if matched_mask not in model_masks:
            comp = full_mask ^ matched_mask
            if comp in model_masks:
                matched_mask = comp
            else:
                continue
        k_bits = int(matched_mask).bit_count()
        if min(k_bits, real_max_bit - k_bits) == 1:
            continue
        length = len_map.get(int(matched_mask))
        if length is None and full_mask:
            length = len_map.get(full_mask ^ int(matched_mask))
        if length is None or float(length) <= 1e-8 or vel >= -1e-3:
            continue
        taus.append(float(length) / max(-vel, 1e-3))
        matched.append(int(matched_mask))
    if not taus:
        return []
    tau_min = min(taus)
    return sorted(m for m, t in zip(matched, taus) if abs(t - tau_min) <= 1e-3)


def _extract_case_index_from_group_key(group_key):
    if group_key is None:
        return None
    group_key = str(group_key)
    marker_idx = group_key.rfind("case")
    if marker_idx < 0:
        return None
    suffix = group_key[marker_idx + len("case") :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _load_frozen_case_metadata(path):
    if not path:
        return {}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _frozen_case_index_lookup_for_module(module):
    cached = getattr(module, "_frozen_start_case_index_lookup", None)
    if cached is not None:
        return cached

    model = getattr(module, "model", None)
    paths = []
    for attr in (
        "first_hit_frozen_start_case_embedding_path",
        "autoregressive_frozen_start_case_embedding_path",
    ):
        path = getattr(model, attr, None)
        if path and path not in paths:
            paths.append(path)

    lookup = {}
    offsets = {}
    for path in paths:
        metadata = _load_frozen_case_metadata(path)
        source_group_keys = metadata.get("source_group_keys")
        if isinstance(source_group_keys, list):
            for idx, group_key in enumerate(source_group_keys):
                key = str(group_key)
                lookup.setdefault(key, int(idx))
                lookup.setdefault(key.lower(), int(idx))
        raw_offsets = metadata.get("offsets")
        if isinstance(raw_offsets, dict):
            for dataset_id, info in raw_offsets.items():
                if isinstance(info, dict) and info.get("offset") is not None:
                    offsets[str(dataset_id).upper()] = int(info["offset"])

    cached = {
        "lookup": lookup,
        "offsets": offsets,
    }
    setattr(module, "_frozen_start_case_index_lookup", cached)
    return cached


def _extract_dataset_id_from_group_key(group_key):
    if group_key is None:
        return None
    lowered = str(group_key).lower()
    marker_idx = lowered.find("ds")
    if marker_idx < 0:
        return None
    digit_start = marker_idx + 2
    digit_end = digit_start
    while digit_end < len(lowered) and lowered[digit_end].isdigit():
        digit_end += 1
    if digit_end == digit_start:
        return None
    return f"DS{lowered[digit_start:digit_end]}".upper()


def _build_case_index_tensor_from_group_keys(
    group_keys,
    *,
    device,
    module=None,
    require_all_or_none=True,
):
    if group_keys is None:
        return None
    frozen_lookup = (
        _frozen_case_index_lookup_for_module(module)
        if module is not None
        else {"lookup": {}, "offsets": {}}
    )
    lookup = frozen_lookup.get("lookup", {})
    offsets = frozen_lookup.get("offsets", {})
    indices = []
    any_case_index = False
    for group_key in group_keys:
        key = None if group_key is None else str(group_key)
        case_index = None
        if key is not None:
            case_index = lookup.get(key)
            if case_index is None:
                case_index = lookup.get(key.lower())
        if case_index is None:
            case_index = _extract_case_index_from_group_key(group_key)
            dataset_id = _extract_dataset_id_from_group_key(group_key)
            if case_index is not None and dataset_id in offsets:
                case_index = int(offsets[dataset_id]) + int(case_index)
        indices.append(-1 if case_index is None else int(case_index))
        any_case_index = any_case_index or (case_index is not None)
    if not any_case_index:
        return None
    if require_all_or_none and any(int(idx) < 0 for idx in indices):
        raise ValueError(
            "Mixed missing/valid bank_group_key values are not supported for case indexing."
        )
    return torch.tensor(indices, dtype=torch.long, device=device)


def _cached_case_group_key_lookup(dataset_split):
    if dataset_split is None:
        return {}
    cached = getattr(dataset_split, "_cached_case_group_key_by_start_tree", None)
    if cached is not None:
        return cached

    lookup = {}
    for item in getattr(dataset_split, "overfit_fixed_pair_start_tree_bank_items", []) or []:
        tree = item.get("tree")
        group_key = item.get("group_key")
        if tree is None or group_key is None:
            continue
        tree_key = str(tree).strip()
        if tree_key:
            lookup.setdefault(tree_key, str(group_key))

    setattr(dataset_split, "_cached_case_group_key_by_start_tree", lookup)
    return lookup


def _infer_case_group_keys_from_batch(module, batch):
    if batch is None:
        return None

    start_trees = batch.get("start_trees")
    if start_trees is None:
        return None
    if isinstance(start_trees, str):
        start_trees = [start_trees]
    else:
        start_trees = list(start_trees)
    if not start_trees:
        return None

    dataset_obj = getattr(module, "dataset", None)
    dataset_splits = []
    for split_name in ("dataset_train", "dataset_val"):
        split = getattr(dataset_obj, split_name, None)
        if split is not None and split not in dataset_splits:
            dataset_splits.append(split)

    if not dataset_splits:
        return None

    resolved = []
    for start_tree in start_trees:
        tree_key = None if start_tree is None else str(start_tree).strip()
        group_key = None
        if tree_key:
            for dataset_split in dataset_splits:
                lookup = _cached_case_group_key_lookup(dataset_split)
                group_key = lookup.get(tree_key)
                if group_key is not None:
                    break
        if group_key is None:
            return None
        resolved.append(str(group_key))

    return resolved


def _build_velocity_replay_batch(module, samples):
    if not samples:
        return None

    all_samples = list(samples)
    filtered_samples = list(all_samples)
    if bool(getattr(module, "velocity_probe_direct_set_anchor_only", False)):
        anchor_samples = [sample for sample in filtered_samples if sample.get("anchor_family")]
        if anchor_samples:
            filtered_samples = anchor_samples
    samples = filtered_samples

    newicks = [sample["newick_tree"] for sample in samples]
    structural_trees = _structuralize_trees_with_cache(module, newicks)
    with torch.no_grad():
        tokenized = _move_tokenized_batch_to_device(
            module.model.tokenizer(structural_trees),
            module.device,
        )
    canonical_targets_by_topology = {}
    canonical_targets_by_state = {}
    if bool(getattr(module, "velocity_probe_direct_set_loss", False)):
        try:
            start_sample = None
            for candidate in all_samples:
                if int(candidate.get("path_index", -1)) == 0 and not candidate.get("anchor_family"):
                    start_sample = candidate
                    break
            if start_sample is None:
                for candidate in all_samples:
                    if not candidate.get("anchor_family"):
                        start_sample = candidate
                        break
            if start_sample is None and all_samples:
                start_sample = all_samples[0]
            target_tree = None if start_sample is None else start_sample.get("target_tree")
            if start_sample is not None and target_tree is not None:
                canonical_map, _ = _build_pair_oracle_orthant_velocity_label_map(
                    str(start_sample["newick_tree"]),
                    str(target_tree),
                )
                for sample in samples:
                    topo_key, _ = _velocity_replay_state_key(
                        sample.get("newick_tree"),
                        sample.get("velocity_next_boundary_tree"),
                    )
                    if topo_key is None:
                        continue
                    canonical = canonical_map.get(topo_key)
                    if canonical is None:
                        continue
                    canonical_sample = {
                        **dict(canonical),
                        "num_leaves": int(sample["num_leaves"]),
                    }
                    target_splits = _target_splits_from_velocity_sample(canonical_sample)
                    canonical_targets_by_topology[topo_key] = target_splits
                    canonical_state_key = _velocity_replay_state_key(
                        canonical_sample.get("newick_tree"),
                        canonical_sample.get("velocity_next_boundary_tree"),
                    )
                    canonical_targets_by_state[canonical_state_key] = target_splits
        except Exception:
            canonical_targets_by_topology = {}
            canonical_targets_by_state = {}
    probe_direct_set_targets = []
    probe_direct_set_mask = []
    include_base_samples = bool(
        getattr(module, "velocity_probe_direct_set_include_base_samples", False)
    )
    for sample in samples:
        use_direct = bool(
            getattr(module, "velocity_probe_direct_set_loss", False)
            and (sample.get("anchor_family") or include_base_samples)
        )
        probe_direct_set_mask.append(bool(use_direct))
        if not use_direct:
            probe_direct_set_targets.append([])
            continue
        topo_key, next_boundary_key = _velocity_replay_state_key(
            sample.get("newick_tree"),
            sample.get("velocity_next_boundary_tree"),
        )
        canonical_target = None
        if topo_key is not None:
            if sample.get("velocity_next_boundary_tree"):
                canonical_target = canonical_targets_by_state.get(
                    (topo_key, next_boundary_key)
                )
            else:
                canonical_target = canonical_targets_by_topology.get(topo_key)
        probe_direct_set_targets.append(
            canonical_target
            if canonical_target is not None
            else _target_splits_from_velocity_sample(sample)
        )
    first_hit_case_index_tensor = None
    if getattr(module.model, "first_hit_head_mode", "base") == "case_adapted_mlp":
        first_hit_case_index_tensor = _build_case_index_tensor_from_group_keys(
            [sample.get("bank_group_key") for sample in samples],
            device=module.device,
            module=module,
        )
    phyla_embeddings = _resolve_replay_phyla_embeddings(
        module,
        samples,
        "newick_tree",
    )
    return {
        "_is_replay_batch": True,
        "_skip_training_augmentations": True,
        "_use_full_path_control_velocity_loss": True,
        "_use_probe_parity_direct_set_loss": bool(any(probe_direct_set_mask)),
        "tokenized_trees": tokenized,
        "batched_time": torch.tensor(
            [float(sample["timepoint"]) for sample in samples],
            dtype=torch.float32,
            device=module.device,
        ),
        "phyla_embeddings": phyla_embeddings,
        "original_trees": newicks,
        "start_trees": [
            sample.get("start_tree", sample.get("newick_tree")) for sample in samples
        ],
        "target_trees": [sample["target_tree"] for sample in samples],
        "bank_group_key": [sample.get("bank_group_key") for sample in samples],
        "batched_velocity": [sample["velocity"] for sample in samples],
        "velocity_next_boundary_trees": [
            sample.get("velocity_next_boundary_tree") for sample in samples
        ],
        "num_leaves": [int(sample["num_leaves"]) for sample in samples],
        "_probe_direct_set_targets": probe_direct_set_targets,
        "_probe_direct_set_sample_mask": probe_direct_set_mask,
        "_first_hit_case_indices": first_hit_case_index_tensor,
    }


def _build_autoregressive_replay_batch(module, samples):
    if not samples:
        return None

    newicks = [sample["newick"] for sample in samples]
    structural_trees = _structuralize_trees_with_cache(module, newicks)
    with torch.no_grad():
        tokenized = _move_tokenized_batch_to_device(
            module.model.tokenizer(structural_trees),
            module.device,
        )
    autoregressive_case_index_tensor = None
    if getattr(module.model, "autoregressive_use_case_conditioning", False):
        autoregressive_case_index_tensor = _build_case_index_tensor_from_group_keys(
            [sample.get("bank_group_key") for sample in samples],
            device=module.device,
            module=module,
        )
    phyla_embeddings = _resolve_replay_phyla_embeddings(
        module,
        samples,
        "newick",
    )
    return {
        "_is_replay_batch": True,
        "_skip_training_augmentations": True,
        "tokenized_autoregressive_trees": tokenized,
        "newick_autoregressive_trees": newicks,
        "start_trees": [
            sample.get("start_tree", sample.get("newick")) for sample in samples
        ],
        "target_trees": [sample["target_tree"] for sample in samples],
        "bank_group_key": [sample.get("bank_group_key") for sample in samples],
        "batched_autoregressive_time": torch.tensor(
            [float(sample["time"]) for sample in samples],
            dtype=torch.float32,
            device=module.device,
        ),
        "batched_autoregressive_labels": [sample["labels"] for sample in samples],
        "batched_autoregressive_stop_after_merge": torch.tensor(
            [
                1.0 if sample.get("stop_after_merge", False) else 0.0
                for sample in samples
            ],
            dtype=torch.float32,
            device=module.device,
        ),
        "phyla_embeddings": phyla_embeddings,
        "_autoregressive_case_indices": autoregressive_case_index_tensor,
    }


def _tree_to_model_split_lengths(module, newick, tokenized=None):
    tree_obj = Tree(newick)
    encoder = BHVEncoder()
    split_masks, split_lengths = encoder.return_BHV_encoding(tree_obj)
    length_map = {
        int(mask): float(length)
        for mask, length in zip(split_masks, split_lengths)
        if length is not None and float(length) > 1e-8
    }
    if tokenized is None:
        tokenized = _tokenize_trees_with_structural_cache(module, [newick])
    model_masks = [int(mask) for mask in tokenized[-1][0] if int(mask) != 0]
    biological_bits = max(tree_obj.n_leaves - 1, 0)
    full_model_mask = (1 << biological_bits) - 1 if biological_bits > 0 else 0

    td = {}
    for model_mask in model_masks:
        edge_length = length_map.get(model_mask)
        if edge_length is None and full_model_mask:
            edge_length = length_map.get(full_model_mask ^ model_mask)
        if edge_length is not None and float(edge_length) > 1e-8:
            td[int(model_mask)] = float(edge_length)

    return td, int(tree_obj.n_leaves), tree_obj.id_to_name


def _tree_to_model_split_lengths_from_model_masks(newick, model_masks):
    tree_obj = Tree(newick)
    encoder = BHVEncoder()
    split_masks, split_lengths = encoder.return_BHV_encoding(tree_obj)
    length_map = {
        int(mask): float(length)
        for mask, length in zip(split_masks, split_lengths)
        if length is not None and float(length) > 1e-8
    }
    biological_bits = max(tree_obj.n_leaves - 1, 0)
    full_model_mask = (1 << biological_bits) - 1 if biological_bits > 0 else 0

    td = {}
    for model_mask in model_masks:
        model_mask = int(model_mask)
        if model_mask == 0:
            continue
        edge_length = length_map.get(model_mask)
        if edge_length is None and full_model_mask:
            edge_length = length_map.get(full_model_mask ^ model_mask)
        if edge_length is not None and float(edge_length) > 1e-8:
            td[int(model_mask)] = float(edge_length)
    return td, int(tree_obj.n_leaves), tree_obj.id_to_name


def _tree_to_model_split_lengths_from_tokenizer_edges(
    model_masks,
    edge_branch_lengths,
    n_leaves,
    mapping,
    eps_len=1e-8,
):
    if torch.is_tensor(edge_branch_lengths):
        edge_lengths = edge_branch_lengths.detach().cpu().tolist()
    else:
        edge_lengths = list(edge_branch_lengths)

    td = {}
    for model_mask, edge_length in zip(model_masks, edge_lengths):
        model_mask = int(model_mask)
        if model_mask == 0:
            continue
        edge_length = float(edge_length)
        if edge_length > float(eps_len):
            td[model_mask] = edge_length
    return td, int(n_leaves), mapping


def _ensure_newick_semicolon(newick):
    stripped = str(newick).strip()
    return stripped if stripped.endswith(";") else stripped + ";"


def _read_branch_relax_tree_list(path):
    trees = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            maybe_path = os.path.expanduser(line)
            if os.path.exists(maybe_path):
                with open(maybe_path, "r", encoding="utf-8") as tree_handle:
                    trees.append(_ensure_newick_semicolon(tree_handle.read().strip()))
            else:
                trees.append(_ensure_newick_semicolon(line))
    return trees


def _branch_relax_raw_length_map(newick):
    tree_obj = Tree(newick)
    encoder = BHVEncoder()
    split_masks, split_lengths = encoder.return_BHV_encoding(tree_obj)
    return {
        int(mask): float(length)
        for mask, length in zip(split_masks, split_lengths)
        if length is not None
    }, int(tree_obj.n_leaves), tree_obj.id_to_name


def _branch_relax_split_fraction(mask, n_leaves):
    biological_bits = max(int(n_leaves) - 1, 0)
    if biological_bits <= 0:
        return 0.0, 0.0
    k = int(mask).bit_count()
    small = min(k, biological_bits - k)
    return float(small) / float(max(biological_bits, 1)), 1.0 if small == 1 else 0.0


def _build_branch_relax_samples_for_module(module, start_tree_list_path, target_tree_list_path):
    start_trees = _read_branch_relax_tree_list(start_tree_list_path)
    target_trees = _read_branch_relax_tree_list(target_tree_list_path)
    if len(start_trees) != len(target_trees):
        raise ValueError(
            "branch relax start/target tree counts differ: "
            f"{len(start_trees)} vs {len(target_trees)}"
        )
    samples = []
    for case_index, (start_tree, target_tree) in enumerate(zip(start_trees, target_trees)):
        start_lengths, n_leaves, _mapping = _tree_to_model_split_lengths(module, start_tree)
        target_lengths, _target_n_leaves, _target_mapping = _branch_relax_raw_length_map(
            target_tree
        )
        biological_bits = max(int(n_leaves) - 1, 0)
        full_mask = (1 << biological_bits) - 1 if biological_bits > 0 else 0
        labels = {}
        for mask, start_length in sorted(start_lengths.items()):
            if full_mask and int(mask) == int(full_mask):
                continue
            target_length = target_lengths.get(int(mask))
            if target_length is None and full_mask:
                target_length = target_lengths.get(full_mask ^ int(mask))
            if target_length is None:
                continue
            labels[int(mask)] = float(target_length) - float(start_length)
        if labels:
            samples.append(
                {
                    "case_index": int(case_index),
                    "newick_tree": start_tree,
                    "target_tree": target_tree,
                    "num_leaves": int(n_leaves),
                    "labels": labels,
                }
            )
    if not samples:
        raise ValueError("branch relax label bank is empty")
    return samples


def _branch_relax_entries_for_tree(module, newick, edge_split_masks, labels=None):
    lengths, n_leaves, mapping = _tree_to_model_split_lengths(module, newick)
    model_masks = [int(mask) for mask in edge_split_masks if int(mask) != 0]
    mask_to_idx = {int(mask): idx for idx, mask in enumerate(model_masks)}
    biological_bits = max(int(n_leaves) - 1, 0)
    full_mask = (1 << biological_bits) - 1 if biological_bits > 0 else 0
    entries = []
    for mask, length in sorted(lengths.items()):
        if full_mask and int(mask) == int(full_mask):
            continue
        matched_mask = int(mask)
        idx = mask_to_idx.get(matched_mask)
        if idx is None and full_mask:
            complement = full_mask ^ matched_mask
            idx = mask_to_idx.get(complement)
            if idx is not None:
                matched_mask = complement
        if idx is None:
            continue
        split_fraction, is_pendant = _branch_relax_split_fraction(matched_mask, n_leaves)
        entry = {
            "edge_index": int(idx),
            "mask": int(matched_mask),
            "source_mask": int(mask),
            "length": float(length),
            "numeric": [float(length), float(split_fraction), float(is_pendant)],
        }
        if labels is not None:
            value = labels.get(int(mask))
            if value is None and full_mask:
                value = labels.get(full_mask ^ int(mask))
            if value is None:
                continue
            entry["label"] = float(value)
        entries.append(entry)
    return entries, lengths, int(n_leaves), mapping


def _align_model_outputs_to_tree_context(
    module,
    newick,
    num_leaves,
    split_masks_num,
    velocity_pred_tree,
    first_hit_logits_tree=None,
    boundary_vanish_logits_tree=None,
    edge_features_tree=None,
    eps_len=1e-8,
    tree_split_lengths=None,
):
    td = (
        tree_split_lengths
        if tree_split_lengths is not None
        else _tree_to_model_split_lengths(module, newick)[0]
    )
    mask_idx = {int(mask): i for i, mask in enumerate(split_masks_num)}
    biological_bits = max(int(num_leaves) - 1, 0)
    full_model_mask = (1 << biological_bits) - 1 if biological_bits > 0 else 0
    dummy_artifact_mask = full_model_mask

    lengths = []
    velocities = []
    first_hit_logits = []
    boundary_vanish_logits = []
    edge_features = []
    supervised_edge_flags = []
    aligned_model_masks = []
    original_model_masks = []

    for m in td:
        if int(m) == dummy_artifact_mask:
            continue

        matched_m = int(m)
        dummy_bit_idx = int(num_leaves) - 1
        if biological_bits > 0 and ((int(m) >> dummy_bit_idx) & 1):
            matched_m = remove_bit(int(m), dummy_bit_idx)

        idx = mask_idx.get(int(matched_m))
        if idx is None and full_model_mask:
            complement_m = full_model_mask ^ int(matched_m)
            idx = mask_idx.get(int(complement_m))
            if idx is not None:
                matched_m = int(complement_m)

        if idx is None:
            if getattr(module, "legacy_alignment_skip_missing_splits", False):
                continue
            raise Exception("Missing split in model outputs while aligning tree context")

        curr_len = float(td[m])
        if curr_len <= float(eps_len):
            continue

        lengths.append(curr_len)
        aligned_model_masks.append(int(matched_m))
        original_model_masks.append(int(m))

        k_bits = int(matched_m).bit_count()
        is_pendant = biological_bits > 0 and min(k_bits, biological_bits - k_bits) == 1
        if is_pendant:
            velocities.append(0.0)
            if first_hit_logits_tree is not None:
                first_hit_logits.append(float(first_hit_logits_tree[idx]))
            if boundary_vanish_logits_tree is not None:
                boundary_vanish_logits.append(float(boundary_vanish_logits_tree[idx]))
            supervised_edge_flags.append(False)
        else:
            velocities.append(float(velocity_pred_tree[idx]))
            if first_hit_logits_tree is not None:
                first_hit_logits.append(float(first_hit_logits_tree[idx]))
            if boundary_vanish_logits_tree is not None:
                boundary_vanish_logits.append(float(boundary_vanish_logits_tree[idx]))
            supervised_edge_flags.append(True)

        if edge_features_tree is not None:
            edge_features.append(edge_features_tree[idx])

    device = velocity_pred_tree.device
    velocity_dtype = velocity_pred_tree.dtype
    lengths_tensor = torch.tensor(lengths, dtype=torch.float32, device=device)
    velocities_tensor = torch.tensor(velocities, dtype=velocity_dtype, device=device)
    first_hit_tensor = None
    if first_hit_logits_tree is not None:
        first_hit_tensor = torch.tensor(
            first_hit_logits,
            dtype=first_hit_logits_tree.dtype,
            device=first_hit_logits_tree.device,
        )
    boundary_vanish_tensor = None
    if boundary_vanish_logits_tree is not None:
        boundary_vanish_tensor = torch.tensor(
            boundary_vanish_logits,
            dtype=boundary_vanish_logits_tree.dtype,
            device=boundary_vanish_logits_tree.device,
        )
    edge_features_tensor = None
    if edge_features_tree is not None and edge_features:
        edge_features_tensor = torch.stack(edge_features, dim=0).to(
            edge_features_tree.device, dtype=edge_features_tree.dtype
        )

    return {
        "lengths": lengths_tensor,
        "velocities": velocities_tensor,
        "first_hit_logits": first_hit_tensor,
        "boundary_vanish_logits": boundary_vanish_tensor,
        "edge_features": edge_features_tensor,
        "supervised_mask": torch.tensor(
            supervised_edge_flags, dtype=torch.bool, device=device
        ),
        "aligned_model_masks": aligned_model_masks,
        "position_by_model_mask": {
            int(mask): idx
            for idx, masks in enumerate(zip(aligned_model_masks, original_model_masks))
            for mask in masks
        },
    }


def _align_model_outputs_to_batch_tree_context(
    num_leaves,
    split_masks_num,
    edge_lengths_num,
    velocity_pred_tree,
    first_hit_logits_tree=None,
    boundary_vanish_logits_tree=None,
    edge_features_tree=None,
    eps_len=1e-8,
):
    biological_bits = max(int(num_leaves) - 1, 0)
    dummy_bit_idx = int(num_leaves) - 1
    full_model_mask = (1 << biological_bits) - 1 if biological_bits > 0 else 0
    mask_idx = {int(mask): i for i, mask in enumerate(split_masks_num)}

    lengths = []
    velocities = []
    first_hit_logits = []
    boundary_vanish_logits = []
    edge_features = []
    supervised_edge_flags = []
    aligned_model_masks = []
    original_model_masks = []

    for idx, (mask, edge_length) in enumerate(zip(split_masks_num, edge_lengths_num)):
        mask = int(mask)
        curr_len = float(edge_length)
        if mask == 0 or curr_len <= float(eps_len):
            continue

        matched_m = int(mask)
        if biological_bits > 0 and ((int(mask) >> dummy_bit_idx) & 1):
            matched_m = remove_bit(int(mask), dummy_bit_idx)

        aligned_lookup = mask_idx.get(int(matched_m))
        if aligned_lookup is None and full_model_mask:
            complement_m = full_model_mask ^ int(matched_m)
            aligned_lookup = mask_idx.get(int(complement_m))
            if aligned_lookup is not None:
                matched_m = int(complement_m)

        if aligned_lookup is None:
            continue

        lengths.append(curr_len)
        aligned_model_masks.append(int(matched_m))
        original_model_masks.append(int(mask))

        k_bits = int(matched_m).bit_count()
        is_pendant = biological_bits > 0 and min(k_bits, biological_bits - k_bits) == 1
        if is_pendant:
            velocities.append(0.0)
            if first_hit_logits_tree is not None:
                first_hit_logits.append(float(first_hit_logits_tree[idx]))
            if boundary_vanish_logits_tree is not None:
                boundary_vanish_logits.append(float(boundary_vanish_logits_tree[idx]))
            supervised_edge_flags.append(False)
        else:
            velocities.append(float(velocity_pred_tree[idx]))
            if first_hit_logits_tree is not None:
                first_hit_logits.append(float(first_hit_logits_tree[idx]))
            if boundary_vanish_logits_tree is not None:
                boundary_vanish_logits.append(float(boundary_vanish_logits_tree[idx]))
            supervised_edge_flags.append(True)

        if edge_features_tree is not None:
            edge_features.append(edge_features_tree[idx])

    device = velocity_pred_tree.device
    velocity_dtype = velocity_pred_tree.dtype
    lengths_tensor = torch.tensor(lengths, dtype=torch.float32, device=device)
    velocities_tensor = torch.tensor(velocities, dtype=velocity_dtype, device=device)
    first_hit_tensor = None
    if first_hit_logits_tree is not None:
        first_hit_tensor = torch.tensor(
            first_hit_logits,
            dtype=first_hit_logits_tree.dtype,
            device=first_hit_logits_tree.device,
        )
    boundary_vanish_tensor = None
    if boundary_vanish_logits_tree is not None:
        boundary_vanish_tensor = torch.tensor(
            boundary_vanish_logits,
            dtype=boundary_vanish_logits_tree.dtype,
            device=boundary_vanish_logits_tree.device,
        )
    edge_features_tensor = None
    if edge_features_tree is not None and edge_features:
        edge_features_tensor = torch.stack(edge_features, dim=0).to(
            edge_features_tree.device, dtype=edge_features_tree.dtype
        )

    return {
        "lengths": lengths_tensor,
        "velocities": velocities_tensor,
        "first_hit_logits": first_hit_tensor,
        "boundary_vanish_logits": boundary_vanish_tensor,
        "edge_features": edge_features_tensor,
        "supervised_mask": torch.tensor(
            supervised_edge_flags, dtype=torch.bool, device=device
        ),
        "aligned_model_masks": aligned_model_masks,
        "position_by_model_mask": {
            int(mask): idx
            for idx, masks in enumerate(zip(aligned_model_masks, original_model_masks))
            for mask in masks
        },
    }


def _extract_edge_splits_from_tokenized(tokenized, batch_index=0):
    edge_masks = tokenized[-1][batch_index]
    return [int(split) for split in edge_masks if int(split) != 0]


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _to_jsonable(value.item())
        return _to_jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _subset_target_matrix(group_splits, subset, device):
    subset = {int(split) for split in subset}
    size = len(group_splits)
    target = torch.zeros(size, size, dtype=torch.float32, device=device)
    subset_indices = [
        idx for idx, split in enumerate(group_splits) if int(split) in subset
    ]
    for i in subset_indices:
        for j in subset_indices:
            if i != j:
                target[i, j] = 1.0
    return target


def _ready_target_merge_subsets_for_group(group_splits, target_newick, n_leaves):
    group_splits = tuple(sorted({int(split) for split in group_splits}))
    if len(group_splits) < 2:
        return []

    biological_bits = max(int(n_leaves) - 1, 0)
    if biological_bits <= 0:
        return []
    full_mask = (1 << biological_bits) - 1

    parent_mask = 0
    for split in group_splits:
        parent_mask |= int(split)

    target_tree = Tree(target_newick)
    enc = BHVEncoder()
    target_masks, target_lengths = enc.return_BHV_encoding(target_tree)
    dummy_bit_idx = target_tree.n_leaves - 1

    target_clusters = set()
    for raw_mask, raw_length in zip(target_masks, target_lengths):
        if raw_length is None or float(raw_length) <= 0.0:
            continue

        mask = int(raw_mask)
        if biological_bits > 0 and ((mask >> dummy_bit_idx) & 1):
            mask = remove_bit(mask, dummy_bit_idx)
        elif biological_bits > 0 and mask.bit_length() > biological_bits:
            continue

        oriented = None
        candidates = [mask]
        if full_mask:
            candidates.append(full_mask ^ mask)
        for candidate in candidates:
            candidate = int(candidate)
            if candidate in (0, full_mask, parent_mask):
                continue
            if (candidate & ~parent_mask) == 0:
                oriented = candidate
                break
        if oriented is not None:
            target_clusters.add(oriented)

    atoms = tuple(sorted(group_splits))
    atom_set = set(atoms)
    relevant_clusters = sorted(
        [cluster for cluster in target_clusters if cluster not in atom_set],
        key=lambda cluster: (int(cluster).bit_count(), int(cluster)),
    )

    child_map = {}
    for cluster in relevant_clusters:
        candidates = [atom for atom in atoms if (int(atom) & ~int(cluster)) == 0]
        candidates.extend(
            other
            for other in relevant_clusters
            if _is_strict_subset_mask(other, cluster)
        )

        maximal_children = []
        for candidate in candidates:
            dominated = False
            for other in candidates:
                if _is_strict_subset_mask(candidate, other) and _is_strict_subset_mask(
                    other, cluster
                ):
                    dominated = True
                    break
            if not dominated:
                maximal_children.append(int(candidate))

        maximal_children = tuple(sorted(set(maximal_children)))
        union = 0
        for child in maximal_children:
            union |= int(child)

        if union == int(cluster) and len(maximal_children) >= 2:
            child_map[int(cluster)] = maximal_children

    ready_subsets = sorted(
        {
            children
            for children in child_map.values()
            if all(int(child) in atom_set for child in children)
        },
        key=lambda subset: (len(subset), tuple(int(split) for split in subset)),
    )
    return ready_subsets


def _apply_merge_subset_to_newick(
    tokenizer,
    current_newick,
    subset,
    new_split=None,
    birth_length=0.1,
):
    tree = Tree(current_newick)
    encoder = BHVEncoder()
    split_masks, split_lengths = encoder.return_BHV_encoding(tree)
    length_map = {
        int(mask): float(length)
        for mask, length in zip(split_masks, split_lengths)
        if int(mask) != 0 and length is not None and float(length) > 1e-8
    }

    if tokenizer is None:
        split_lengths = dict(length_map)
        existing_splits = list(split_lengths.keys())
    else:
        tokenized = tokenizer([current_newick])
        existing_splits = _extract_edge_splits_from_tokenized(tokenized, batch_index=0)
        biological_bits = max(tree.n_leaves - 1, 0)
        full_model_mask = (1 << biological_bits) - 1 if biological_bits > 0 else 0
        split_lengths = {}
        for split in existing_splits:
            split = int(split)
            edge_length = length_map.get(split)
            if edge_length is None and full_model_mask:
                edge_length = length_map.get(full_model_mask ^ split)
            if edge_length is not None and float(edge_length) > 1e-8:
                split_lengths[split] = float(edge_length)

    if new_split is None:
        new_split = 0
        for component in subset:
            new_split |= int(component)

    new_split = int(new_split)
    if new_split in existing_splits:
        return None

    split_lengths[new_split] = float(birth_length)

    _, newick = build_tree_from_splits(
        list(split_lengths.keys()),
        split_lengths,
        tree.n_leaves,
        root_leaf=tree.n_leaves - 1,
        mapping=tree.id_to_name,
    )
    return newick


def _jitter_internal_lengths_newick(current_newick, jitter_scale, min_length=1e-4):
    tree = EteTree(current_newick, format=1)
    internal_nodes = [
        node
        for node in tree.traverse("postorder")
        if not node.is_leaf() and not node.is_root()
    ]
    if not internal_nodes:
        return None

    changed = False
    lower = max(0.0, 1.0 - float(jitter_scale))
    upper = 1.0 + float(jitter_scale)
    for node in internal_nodes:
        dist = float(node.dist)
        if not math.isfinite(dist) or dist <= 0.0:
            continue
        factor = random.uniform(lower, upper)
        new_dist = max(float(min_length), dist * factor)
        if abs(new_dist - dist) > 1e-12:
            node.dist = new_dist
            changed = True

    if not changed:
        return None
    return tree.write(format=1)


def _normalize_tree_like_dataset(tree_newick):
    t_obj = EteTree(tree_newick, format=1)
    leaves = t_obj.get_leaves()
    leaves.sort(key=lambda leaf: leaf.name)

    seq_ordering_map = {}
    for i, leaf in enumerate(leaves):
        original_name = leaf.name
        mapped_name = str(i)
        leaf.name = mapped_name
        seq_ordering_map[original_name] = mapped_name

    return t_obj.write(format=1), seq_ordering_map


def _topology_key(newick_tree, eps_len=1e-8):
    masks, lengths = BHVEncoder().return_BHV_encoding(Tree(newick_tree))
    active_masks = [
        int(mask)
        for mask, length in zip(masks, lengths)
        if length is not None and float(length) > eps_len
    ]
    return tuple(sorted(active_masks))


def _record_repeated_topology_visit(topology_counts, topology_key, repeat_cap):
    topology_counts[topology_key] = int(topology_counts.get(topology_key, 0)) + 1
    return bool(int(repeat_cap) > 0 and topology_counts[topology_key] > int(repeat_cap))


def _summarize_trace_topology_repeats(trace):
    def _summarize_keys(keys):
        counts = {}
        for key in keys:
            counts[key] = int(counts.get(key, 0)) + 1
        repeated = [count for count in counts.values() if count > 1]
        return {
            "num_states": float(len(keys)),
            "num_unique_topologies": float(len(counts)),
            "num_repeated_topologies": float(len(repeated)),
            "max_repeat_count": float(max(counts.values()) if counts else 0),
        }

    velocity_keys = [
        _topology_key(sample["newick_tree"])
        for sample in trace.get("velocity", [])
        if sample.get("newick_tree")
    ]
    autoregressive_keys = [
        _topology_key(sample["newick"])
        for sample in trace.get("autoregressive", [])
        if sample.get("newick")
    ]

    velocity_summary = _summarize_keys(velocity_keys)
    autoregressive_summary = _summarize_keys(autoregressive_keys)
    return {
        "velocity_num_states": velocity_summary["num_states"],
        "velocity_num_unique_topologies": velocity_summary["num_unique_topologies"],
        "velocity_num_repeated_topologies": velocity_summary[
            "num_repeated_topologies"
        ],
        "velocity_max_topology_repeat": velocity_summary["max_repeat_count"],
        "autoregressive_num_states": autoregressive_summary["num_states"],
        "autoregressive_num_unique_topologies": autoregressive_summary[
            "num_unique_topologies"
        ],
        "autoregressive_num_repeated_topologies": autoregressive_summary[
            "num_repeated_topologies"
        ],
        "autoregressive_max_topology_repeat": autoregressive_summary[
            "max_repeat_count"
        ],
    }


@functools.lru_cache(maxsize=None)
def _oracle_training_topology_keys(current_newick, target_newick):
    keys = {_topology_key(current_newick)}
    boundary_paths = return_tree_boundary_merge_paths(current_newick, target_newick)
    for boundary_path in boundary_paths:
        keys.add(_topology_key(boundary_path["start_newick"]))
        keys.add(_topology_key(boundary_path["end_newick"]))
        for event in boundary_path["events"]:
            keys.add(_topology_key(event["newick"]))
    return tuple(sorted(keys))


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
                    f"Non-integer leaf name '{leaf.name}' encountered in "
                    f"{tree_kind} while applying offset {offset}."
                ) from exc

        mapped_name = seq_ordering_map.get(lookup_name)
        if mapped_name is None:
            raise ValueError(
                f"Leaf name '{lookup_name}' in {tree_kind} not found in "
                "sequence ordering map."
            )
        leaf.name = mapped_name

    return t_obj.write(format=1)


def _choose_wrong_pair_merge_subset(current_newick, target_newick, tokenizer):
    current_tree = Tree(current_newick)
    current_groups = get_structural_polytomy_groups_from_newick(current_newick)
    if not current_groups:
        return None

    n_leaves = current_tree.n_leaves
    shuffled_groups = [tuple(int(component) for component in group) for group in current_groups]
    random.shuffle(shuffled_groups)

    for group_splits in shuffled_groups:
        if len(group_splits) < 3:
            continue

        ready_subsets = {
            tuple(sorted(int(split) for split in subset))
            for subset in _ready_target_merge_subsets_for_group(
                group_splits,
                target_newick,
                n_leaves,
            )
        }

        candidates = []
        for left_idx in range(len(group_splits)):
            for right_idx in range(left_idx + 1, len(group_splits)):
                subset = tuple(
                    sorted(
                        (
                            int(group_splits[left_idx]),
                            int(group_splits[right_idx]),
                        )
                    )
                )
                if subset in ready_subsets:
                    continue
                candidates.append(subset)

        random.shuffle(candidates)
        for subset in candidates:
            perturbed_newick = _apply_merge_subset_to_newick(
                tokenizer,
                current_newick,
                subset,
            )
            if perturbed_newick is None or perturbed_newick == current_newick:
                continue
            return subset

    return None


def _choose_model_wrong_pair_merge_subset(
    module,
    current_newick,
    target_newick,
    current_time,
    phyla_embedding=None,
):
    current_tree = Tree(current_newick)
    current_groups = get_structural_polytomy_groups_from_newick(current_newick)
    if not current_groups:
        return None

    tokenized = _move_tokenized_batch_to_device(
        module.model.tokenizer([current_newick]),
        module.device,
    )
    existing_splits = set(_extract_edge_splits_from_tokenized(tokenized, batch_index=0))
    time_tensor = torch.tensor(
        [float(current_time)],
        dtype=torch.float32,
        device=module.device,
    )

    with torch.no_grad():
        logit_outputs = module.forward(
            tokenized,
            time_tensor,
            phyla_embedding,
            autoregressive=True,
            autoregressive_component_groups=[current_groups],
        )

    if not logit_outputs:
        return None

    n_leaves = current_tree.n_leaves
    ready_subsets_by_group = {}
    for group_splits in current_groups:
        normalized_group = tuple(sorted(int(component) for component in group_splits))
        ready_subsets_by_group[normalized_group] = {
            tuple(sorted(int(split) for split in subset))
            for subset in _ready_target_merge_subsets_for_group(
                normalized_group,
                target_newick,
                n_leaves,
            )
        }

    planned_merges = _plan_autoregressive_boundary_merges(
        logit_outputs,
        existing_splits,
    )
    for planned in planned_merges:
        group_splits = tuple(sorted(int(split) for split in planned["splits_represented"]))
        ready_subsets = ready_subsets_by_group.get(group_splits, set())
        subset, new_split = planned["subsets"][0]
        normalized_subset = tuple(sorted(int(split) for split in subset))
        if normalized_subset in ready_subsets or int(new_split) in existing_splits:
            continue
        return {
            "subset": normalized_subset,
            "new_split": int(new_split),
            "source": "planned_wrong",
        }

    best_candidate = None
    best_rank = None
    for output in logit_outputs:
        group_splits = tuple(sorted(int(split) for split in output["splits_represented"]))
        if len(group_splits) < 3:
            continue

        ready_subsets = ready_subsets_by_group.get(group_splits, set())
        polytomy_score = float(output["polytomy_pred"].detach().cpu().item())
        logits = output["logits"].detach()
        splits = [int(split) for split in output["splits_represented"]]

        for left_idx in range(len(splits)):
            for right_idx in range(left_idx + 1, len(splits)):
                subset = tuple(sorted((int(splits[left_idx]), int(splits[right_idx]))))
                if subset in ready_subsets:
                    continue

                new_split = int(subset[0]) | int(subset[1])
                if new_split in existing_splits:
                    continue

                score = float(logits[left_idx, right_idx].item())
                if not math.isfinite(score):
                    continue

                rank = (score, polytomy_score)
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_candidate = {
                        "subset": subset,
                        "new_split": new_split,
                        "source": "top_wrong_pair",
                    }

    return best_candidate


def _plan_first_autoregressive_model_merge(
    module,
    current_newick,
    current_time,
    phyla_embedding=None,
):
    tokenized = _move_tokenized_batch_to_device(
        module.model.tokenizer([current_newick]),
        module.device,
    )
    groups = [get_structural_polytomy_groups_from_newick(current_newick)]
    if not groups[0]:
        return None

    time_tensor = module._effective_autoregressive_time_tensor(current_time)

    with torch.no_grad():
        logit_outputs = module.forward(
            tokenized,
            time_tensor,
            phyla_embedding,
            autoregressive=True,
            autoregressive_component_groups=groups,
        )

    planned_merges = _plan_autoregressive_boundary_merges(
        logit_outputs,
        _extract_edge_splits_from_tokenized(tokenized, batch_index=0),
    )
    if not planned_merges:
        return None

    subset, new_split = planned_merges[0]["subsets"][0]
    return {
        "subset": tuple(int(split) for split in subset),
        "new_split": int(new_split),
    }


def _discrete_phase_rollout(
    module,
    start_tree,
    target_tree,
    phyla_embeddings,
    case_index=None,
    start_topology_features=None,
    start_topology_embeddings=None,
    start_topology_pad_mask=None,
    *,
    dt_base: float = 0.02,
    eps_len: float = 1e-8,
    autoregressive_birth_length: float = 1e-3,
    max_events: int = 1000,
    max_steps: int = 1000,
    max_phases: int = 8,
    return_trace: bool = False,
    trace_state_rf: bool = True,
    explicit_autoregressive_component_groups: bool = True,
):
    current_newick = str(start_tree)
    rollout_case_indices = (
        None
        if case_index is None
        else torch.tensor([int(case_index)], dtype=torch.long, device=module.device)
    )
    rollout_start_topology_features = start_topology_features
    if (
        rollout_start_topology_features is None
        and (
            getattr(module.model, "first_hit_head_mode", "base")
            in {
                "start_topology_adapter_mlp",
                "start_topology_raw_pool_concat_mlp",
            }
            or getattr(
                module.model,
                "autoregressive_use_start_topology_conditioning",
                False,
            )
        )
    ):
        rollout_start_topology_features = _build_start_topology_feature_tensor(
            module,
            [start_tree],
            device=module.device,
        )
    rollout_start_topology_embeddings = start_topology_embeddings
    rollout_start_topology_pad_mask = start_topology_pad_mask
    if (
        (
            rollout_start_topology_embeddings is None
            or rollout_start_topology_pad_mask is None
        )
        and getattr(module.model, "first_hit_head_mode", "base")
        == "start_topology_cross_attn_mlp"
    ):
        (
            rollout_start_topology_embeddings,
            rollout_start_topology_pad_mask,
        ) = _build_start_topology_identity_batch(
            module,
            [start_tree],
            device=module.device,
        )
    rollout_start_tree_graph_context = None
    if getattr(module.model, "first_hit_head_mode", "base") == "start_tree_graph_token_mlp":
        rollout_start_tree_graph_context = _build_start_tree_graph_context(
            module,
            [start_tree],
            phyla_embeddings,
            device=module.device,
            detach=getattr(module.model, "first_hit_start_tree_graph_detach", False),
        )
    phase = 0
    n_events = 0
    n_steps = 0
    trace = {
        "velocity": [],
        "autoregressive": [],
        "stopped_for_no_valid_merge": False,
        "stopped_for_repeated_topology": False,
        "skipped_no_valid_boundary_revisits": 0.0,
        "stopped_for_prefix_replay_quota": False,
        "silent_boundary_recoveries": 0.0,
        "autoregressive_boundary_stop_count": 0.0,
    }

    def _trace_rf_to_target(tree_newick):
        if not trace_state_rf:
            return None
        return float(calculate_norm_rf(tree_newick, target_tree))

    effective_max_events = (
        1000000 if max_events is None or int(max_events) < 0 else int(max_events)
    )
    effective_max_steps = (
        1000000 if max_steps is None or int(max_steps) < 0 else int(max_steps)
    )
    effective_max_phases = max(1, int(max_phases))

    while (
        phase < effective_max_phases
        and n_events < effective_max_events
        and n_steps < effective_max_steps
    ):
        n_steps += 1
        tokenized = _tokenize_trees_with_structural_cache(module, [current_newick])
        td, n_leaves, mapping = _tree_to_model_split_lengths(
            module,
            current_newick,
            tokenized=tokenized,
        )
        with torch.inference_mode():
            (
                velocity,
                edge_splits,
                _edge_split_mask,
                first_hit_logits,
                boundary_vanish_logits,
                edge_features,
            ) = module.forward(
                tokenized,
                float(phase),
                phyla_embeddings,
                first_hit_case_indices=rollout_case_indices,
                first_hit_start_topology_features=rollout_start_topology_features,
                first_hit_start_topology_embeddings=rollout_start_topology_embeddings,
                first_hit_start_topology_pad_mask=rollout_start_topology_pad_mask,
                first_hit_start_tree_graph_context=rollout_start_tree_graph_context,
            )

        aligned = _align_model_outputs_to_tree_context(
            module,
            current_newick,
            n_leaves,
            edge_splits[0],
            velocity[0, :, 0],
            first_hit_logits_tree=None
            if first_hit_logits is None
            else first_hit_logits[0, :, 0],
            boundary_vanish_logits_tree=None
            if boundary_vanish_logits is None
            else boundary_vanish_logits[0, :, 0],
            edge_features_tree=None if edge_features is None else edge_features[0],
            eps_len=eps_len,
            tree_split_lengths=td,
        )

        aligned_edge_features = aligned["edge_features"]
        aligned_lengths = aligned["lengths"]
        aligned_velocities = aligned["velocities"]
        aligned_first_hit_logits = module._compute_first_hit_logits(
            aligned["first_hit_logits"],
            lengths=aligned_lengths,
            velocities=aligned_velocities,
            edge_features=aligned_edge_features,
            group_sizes=[int(aligned_lengths.numel())],
        )

        lengths = aligned_lengths.detach().cpu().numpy().astype(np.float64)
        velocities = aligned_velocities.detach().cpu().numpy().astype(np.float64)
        supervised_mask = aligned["supervised_mask"].detach().cpu().numpy().astype(bool)
        masks = [int(x) for x in aligned["aligned_model_masks"]]
        first_logits = (
            aligned_first_hit_logits.detach().cpu().numpy().astype(np.float64)
            if aligned_first_hit_logits is not None
            else np.full_like(lengths, float("-inf"), dtype=np.float64)
        )
        candidate_mask = supervised_mask & (lengths > eps_len)
        predicted_first_mask, _raw_first_count, _used_first_fallback = (
            _predict_first_hit_mask_with_fallback(
                first_logits,
                candidate_mask,
                max_edges=getattr(module, "velocity_first_hit_sampling_max_edges", -1),
                fallback_threshold=getattr(
                    module,
                    "velocity_first_hit_sampling_fallback_threshold",
                    -1,
                ),
                fallback_top_k=getattr(
                    module,
                    "velocity_first_hit_sampling_fallback_top_k",
                    -1,
                ),
            )
        )
        pred_neg = predicted_first_mask & (velocities < 0.0) & (lengths > eps_len)
        if not np.any(pred_neg):
            trace["velocity"].append(
                {
                    "newick_tree": current_newick,
                    "target_tree": target_tree,
                    "timepoint": float(phase),
                    "phase_idx": int(phase),
                    "num_leaves": int(n_leaves),
                    "event": "no_predicted_negative_edges",
                }
            )
            break

        dt_target = float(
            np.max(lengths[pred_neg] / np.maximum(-velocities[pred_neg], eps_len))
        )
        if bool(
            getattr(
                module,
                "sampling_discrete_phase_exact_boundary_step_use_at_sampling",
                False,
            )
        ):
            dt = float(dt_target)
        else:
            dt = min(float(dt_base), dt_target)
        L_new = lengths + dt * velocities
        collapse_mask = predicted_first_mask.copy()
        if np.any(collapse_mask):
            L_new[collapse_mask] = 0.0
        blocked = supervised_mask & (~collapse_mask)
        if np.any(blocked):
            L_new[blocked] = np.maximum(L_new[blocked], eps_len * 10.0)

        td2 = {
            int(mask): float(length)
            for mask, length in zip(masks, L_new)
            if float(length) > eps_len
        }
        current_newick = build_tree_from_splits(
            list(td2.keys()),
            td2,
            n_leaves,
            root_leaf=n_leaves - 1,
            mapping=mapping,
        )[1]
        trace["velocity"].append(
            {
                "newick_tree": current_newick,
                "target_tree": target_tree,
                "timepoint": float(phase),
                "phase_idx": int(phase),
                "num_leaves": int(n_leaves),
                "rf_to_target": _trace_rf_to_target(current_newick),
                "predicted_masks": [
                    masks[i]
                    for i, on in enumerate(predicted_first_mask.tolist())
                    if on
                ],
                "dt_target": float(dt_target),
            }
        )

        phase_exhausted = False
        ar_boundary_complete = False
        while (
            has_polytomy_fast(current_newick, unrooted_ok=False)
            and n_events < effective_max_events
        ):
            tokenized_trees = _tokenize_trees_with_structural_cache(
                module,
                [current_newick],
            )
            component_groups = None
            if explicit_autoregressive_component_groups:
                component_groups = [
                    get_structural_polytomy_groups_from_newick(current_newick)
                ]
            with torch.inference_mode():
                logit_outputs = module.forward(
                    tokenized_trees,
                    torch.tensor(
                        [float(phase)],
                        dtype=torch.float32,
                        device=module.device,
                    ),
                    phyla_embeddings,
                    autoregressive=True,
                    autoregressive_component_groups=component_groups,
                    autoregressive_case_indices=rollout_case_indices,
                    autoregressive_start_topology_features=rollout_start_topology_features,
                )
            td_ar, n_ar, m_ar = _tree_to_model_split_lengths(
                module,
                current_newick,
                tokenized=tokenized_trees,
            )
            planned_merges = _plan_autoregressive_boundary_merges(
                logit_outputs,
                td_ar.keys(),
                top_only=False,
            )
            if planned_merges and not bool(
                getattr(
                    module,
                    "sampling_apply_all_planned_ar_merges_per_forward_enabled",
                    False,
                )
            ):
                planned_merges = planned_merges[:1]
            if not planned_merges:
                trace["autoregressive"].append(
                    {
                        "newick": current_newick,
                        "target_tree": target_tree,
                        "time": float(phase),
                        "phase_idx": int(phase),
                        "rf_to_target": _trace_rf_to_target(current_newick),
                        "planned_merge_count": 0,
                        "selected_result_split": None,
                    }
                )
                trace["stopped_for_no_valid_merge"] = True
                phase_exhausted = True
                break

            applied_any_merge = False
            for planned in planned_merges:
                if n_events >= effective_max_events:
                    break
                if not planned.get("subsets"):
                    continue
                _, new_split = planned["subsets"][0]
                if int(new_split) in td_ar:
                    continue
                stop_after_merge_logit = planned.get("stop_after_merge_logit")
                stop_after_merge_logit_value = (
                    None
                    if stop_after_merge_logit is None
                    else float(stop_after_merge_logit)
                )
                stop_after_merge_prob = (
                    None
                    if stop_after_merge_logit_value is None
                    else float(
                        1.0
                        / (
                            1.0
                            + math.exp(
                                -max(
                                    min(float(stop_after_merge_logit_value), 60.0),
                                    -60.0,
                                )
                            )
                        )
                    )
                )
                source_newick = current_newick
                td_ar[int(new_split)] = float(autoregressive_birth_length)
                n_events += 1
                applied_any_merge = True
                current_newick = build_tree_from_splits(
                    list(td_ar.keys()),
                    td_ar,
                    n_ar,
                    root_leaf=n_ar - 1,
                    mapping=m_ar,
                )[1]
                trace["autoregressive"].append(
                    {
                        "source_newick": source_newick,
                        "newick": current_newick,
                        "target_tree": target_tree,
                        "time": float(phase),
                        "phase_idx": int(phase),
                        "rf_to_target": _trace_rf_to_target(current_newick),
                        "planned_merge_count": int(len(planned_merges)),
                        "selected_result_split": int(new_split),
                        "stop_after_merge_logit": stop_after_merge_logit_value,
                        "stop_after_merge_prob": stop_after_merge_prob,
                        "stop_after_merge_requested": False,
                    }
                )
            if not applied_any_merge:
                trace["stopped_for_no_valid_merge"] = True
                phase_exhausted = True
                break

        if (
            ar_boundary_complete
            or phase_exhausted
            or not has_polytomy_fast(current_newick, unrooted_ok=False)
        ):
            phase += 1
            continue
        break


    out = {
        "final_tree": current_newick,
        "final_rf": float(calculate_norm_rf(current_newick, target_tree)),
        "num_velocity_states": int(len(trace["velocity"])),
        "num_ar_states": int(len(trace["autoregressive"])),
        "trace": trace,
    }
    if return_trace:
        return out
    return out


__all__ = [
    name
    for name in globals()
    if not name.startswith("__")
]
