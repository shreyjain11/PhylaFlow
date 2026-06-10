import random
import time
import inspect
import math
import json
import hashlib
import numbers
import subprocess
import re
import contextlib
import types
from collections import Counter, OrderedDict
import importlib.util
import functools
import operator
import itertools
import torch, torch.optim as optim
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import wandb
import logging
import gc
import torch.distributed
import gc
import torch
import sys
import os
import torch.nn as nn
import torch.nn.functional as F
from ete3 import Tree as EteTree

# Ensure the current directory is in sys.path to import 'phyla'
sys.path.append(os.getcwd())
# Import utilities from the provided codebase
from utils.utils import remove_bit, has_polytomy_fast

from utils.random_tree import Tree
from utils.bhv_utils import (
    BHVEncoder,
    _split_multi_label_training_events,
    get_structural_polytomy_groups_from_newick,
    return_sampled_tree_boundary_decisions,
    return_sampled_tree_orthant_velocity,
    return_tree_boundary_merge_paths,
)
from utils.bhv_movie import build_tree_from_splits
from utils.utils import (
pick_group,
find_polytomy_nodes,
number_to_name_newick,
has_polytomy_fast,
resolve_polytomies_random_deterministic,
_pick_knn_pair,
)
from utils.metric_utils import (
kl_divergence_topological_distributions,
kl_divergence_tree_topology_distributions,
topk_posterior_tree_recall,
split_bipartition_frequency_correlation,
compare_likelihood_distributions,
compare_branch_length_distributions,
calculate_norm_rf,
canonicalize_topology_newick,
align_numeric_leaf_labels_to_reference,
)
from data.dataset import PhylaDataModule, TreeDataset
from model.model import BirthSetTopologyHead, TreeDenoiserTokenGT
import numpy as np
import logging
from tqdm import tqdm
from utils.utils import compute_merge_metrics
from utils.utils import _velocity_diagnostics

logger = logging.getLogger(__name__)


class BranchRelaxHead(nn.Module):
    def __init__(
        self,
        edge_dim: int,
        num_cases: int,
        *,
        case_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.case_embedding = nn.Embedding(max(1, int(num_cases)), int(case_dim))
        self.net = nn.Sequential(
            nn.LayerNorm(int(edge_dim) + int(case_dim) + 3),
            nn.Linear(int(edge_dim) + int(case_dim) + 3, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, edge_features, numeric_features, case_indices):
        case_indices = torch.clamp(
            case_indices.to(device=edge_features.device, dtype=torch.long),
            min=0,
            max=self.case_embedding.num_embeddings - 1,
        )
        case_features = self.case_embedding(case_indices)
        x = torch.cat([edge_features, numeric_features, case_features], dim=-1)
        return self.net(x).squeeze(-1)


def _birthset_full_mask(num_leaves):
    biological_bits = max(int(num_leaves) - 1, 0)
    return (1 << biological_bits) - 1 if biological_bits > 0 else 0


def _birthset_local_subset_size(local_subset):
    return int(local_subset).bit_count()


def _birthset_local_subset_to_split(local_subset, component_masks):
    split = 0
    for idx, component in enumerate(component_masks):
        if (int(local_subset) >> idx) & 1:
            split |= int(component)
    return int(split)


def _birthset_valid_local_subset(local_subset, num_components):
    size = _birthset_local_subset_size(local_subset)
    return 2 <= size <= max(int(num_components) - 1, 0)


def _birthset_valid_rooted_split(split_mask, full_mask):
    split_mask = int(split_mask) & int(full_mask)
    return 1 < split_mask.bit_count() < int(full_mask).bit_count()


def _birthset_canonical_unrooted_split(split_mask, full_mask):
    full_mask = int(full_mask)
    split_mask = int(split_mask) & full_mask
    complement = full_mask ^ split_mask
    return min(int(split_mask), int(complement))


def _birthset_rooted_splits_compatible(mask_a, mask_b, full_mask):
    full_mask = int(full_mask)
    a = int(mask_a) & full_mask
    b = int(mask_b) & full_mask
    return (
        (a & b) == 0
        or (a & ~b & full_mask) == 0
        or (b & ~a & full_mask) == 0
    )


def _birthset_map_split_to_local_subset(split_mask, component_masks):
    split_mask = int(split_mask)
    parent_mask = 0
    for component in component_masks:
        parent_mask |= int(component)
    if split_mask & ~parent_mask:
        return None
    local_subset = 0
    for idx, component in enumerate(component_masks):
        component = int(component)
        overlap = component & split_mask
        if overlap and overlap != component:
            return None
        if overlap:
            local_subset |= 1 << idx
    if not _birthset_valid_local_subset(local_subset, len(component_masks)):
        return None
    return int(local_subset)


def _birthset_candidate_record(local_subset, component_masks, source):
    local_subset = int(local_subset)
    return {
        "local_subset": local_subset,
        "split_mask": _birthset_local_subset_to_split(
            local_subset,
            component_masks,
        ),
        "source": str(source),
        "size": _birthset_local_subset_size(local_subset),
    }


try:
    from deepspeed.ops.adam import FusedAdam
except Exception:
    FusedAdam = optim.Adam


def _load_phyla_runtime():
    from phyla.utils.utils import load_config
    from phyla.eval.evo_reasoning_eval import (
        Config,
        load_model,
        _encode_sequences_openfold_style,
    )

    return load_config, Config, load_model, _encode_sequences_openfold_style


def _install_skbio_stub_for_live_phyla() -> None:
    if "skbio" in sys.modules:
        return
    skbio = types.ModuleType("skbio")
    skbio_tree = types.ModuleType("skbio.tree")

    class _UnusedDistanceMatrix:
        def __init__(self, *args, **kwargs):
            raise ImportError("scikit-bio is not required for live Phyla embeddings")

    def _unused_nj(*args, **kwargs):
        raise ImportError("scikit-bio is not required for live Phyla embeddings")

    skbio.DistanceMatrix = _UnusedDistanceMatrix
    skbio_tree.nj = _unused_nj
    sys.modules["skbio"] = skbio
    sys.modules["skbio.tree"] = skbio_tree


def _load_live_phyla_beta_state_dict(checkpoint_path, map_location="cpu"):
    payload = torch.load(checkpoint_path, map_location=map_location)
    state_dict = (
        payload["state_dict"]
        if isinstance(payload, dict) and "state_dict" in payload
        else payload
    )
    normalized = {}
    for key, value in state_dict.items():
        key = str(key)
        for prefix in ("model_name.", "_forward_module.model.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        normalized[key] = value
    return normalized


def _load_live_phyla_beta_model(checkpoint_path, device="cpu"):
    _install_skbio_stub_for_live_phyla()
    from phyla.model.model import Config, Phyla

    cfg = Config()
    cfg.model.model_name = "phyla-beta"
    if str(device).startswith("cpu"):
        cfg.model.fused_add_norm = False
    model = Phyla(cfg, name="phyla-beta", device=device)
    model.load_state_dict(
        _load_live_phyla_beta_state_dict(checkpoint_path, map_location="cpu"),
        strict=True,
    )
    model.to(device)
    model.device = torch.device(device)
    return model


_HISTORICAL_STEP_MODULE = None
_HISTORICAL_METHOD_SIGNATURES = {}


def _load_historical_training_module_for_step():
    global _HISTORICAL_STEP_MODULE
    if _HISTORICAL_STEP_MODULE is not None:
        return _HISTORICAL_STEP_MODULE

    hist_path = "/tmp/phylaflow_hist_exact_ttWP/run/TrainingModule.py"
    spec = importlib.util.spec_from_file_location(
        "historical_trainingmodule_for_step", hist_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load historical TrainingModule from {hist_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _HISTORICAL_STEP_MODULE = module
    return module


def _call_historical_trainingmodule_method(method_name, self_obj, /, *args, **kwargs):
    historical_module = _load_historical_training_module_for_step()
    method = getattr(historical_module.TrainingModule, method_name)
    signature = _HISTORICAL_METHOD_SIGNATURES.get(method_name)
    if signature is None:
        signature = inspect.signature(method)
        _HISTORICAL_METHOD_SIGNATURES[method_name] = signature

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        filtered_kwargs = dict(kwargs)
    else:
        filtered_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }
    return method(self_obj, *args, **filtered_kwargs)


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


def _oracle_boundary_vanish_mask_for_sampling(
    current_newick,
    target_tree,
    *,
    masks,
    n_leaves,
    candidate_mask,
):
    pred_mask = np.zeros_like(candidate_mask, dtype=bool)
    if not current_newick or not target_tree:
        return pred_mask

    try:
        boundary_paths = return_tree_boundary_merge_paths(current_newick, target_tree)
    except Exception:
        return pred_mask
    if not boundary_paths:
        return pred_mask

    next_boundary_tree = boundary_paths[0].get("start_newick")
    if not next_boundary_tree:
        return pred_mask

    enc = BHVEncoder()
    try:
        boundary_tree_obj = Tree(next_boundary_tree)
        boundary_masks, boundary_lengths = enc.return_BHV_encoding(boundary_tree_obj)
    except Exception:
        return pred_mask

    split_masks_num = [int(m) for m in masks]
    split_masks_nonzero = [m for m in split_masks_num if m != 0]
    if not split_masks_nonzero:
        return pred_mask

    real_max_bit = max(m.bit_length() for m in split_masks_nonzero)
    full_mask = (1 << real_max_bit) - 1 if real_max_bit > 0 else 0
    mask_to_idx = {m: i for i, m in enumerate(split_masks_num)}
    next_boundary_active_masks = set()

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

        next_boundary_active_masks.add(int(matched_boundary_mask))

    for idx, matched_mask in enumerate(split_masks_num):
        if not candidate_mask[idx]:
            continue
        if int(matched_mask) not in next_boundary_active_masks:
            pred_mask[idx] = True
    return pred_mask


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


def _predict_boundary_vanish_mask_from_logits(logits, candidate_mask, threshold=0.0):
    pred_mask = np.zeros_like(candidate_mask, dtype=bool)
    candidate_indices = np.where(candidate_mask)[0]
    if candidate_indices.size == 0:
        return pred_mask

    positive_indices = candidate_indices[logits[candidate_indices] > float(threshold)]
    if positive_indices.size > 0:
        pred_mask[positive_indices] = True
        return pred_mask

    best_local = candidate_indices[int(np.argmax(logits[candidate_indices]))]
    pred_mask[best_local] = True
    return pred_mask


def _select_replay_samples_across_rollout(samples, max_count):
    samples = list(samples)
    max_count = int(max_count)
    if max_count <= 0:
        return []
    if len(samples) <= max_count:
        return samples
    if max_count == 1:
        return [samples[-1]]

    indices = []
    last_idx = len(samples) - 1
    for i in range(max_count):
        idx = int(round((i * last_idx) / float(max_count - 1)))
        if not indices or idx != indices[-1]:
            indices.append(idx)
    if indices[-1] != last_idx:
        indices[-1] = last_idx
    return [samples[idx] for idx in indices]


def _select_legacy_prefix_suffix_replay_samples(samples, max_count, tree_key):
    samples = list(samples)
    max_count = int(max_count)
    if max_count <= 0:
        return []
    if len(samples) <= max_count:
        return samples

    prefix_count = max_count // 2
    suffix_count = max_count - prefix_count
    prefix_indices = list(range(min(prefix_count, len(samples))))
    selected = set(prefix_indices)

    topology_keys = []
    counts = {}
    for sample in samples:
        tree_newick = sample.get(tree_key)
        topo_key = _topology_key(tree_newick) if tree_newick else None
        topology_keys.append(topo_key)
        if topo_key is not None:
            counts[topo_key] = int(counts.get(topo_key, 0)) + 1

    suffix_indices = []
    candidate_indices = []
    for idx in range(len(samples) - 1, -1, -1):
        topo_key = topology_keys[idx]
        if topo_key is not None and counts.get(topo_key, 0) > 1:
            candidate_indices.append(idx)
    for idx in range(len(samples) - 1, -1, -1):
        candidate_indices.append(idx)

    for idx in candidate_indices:
        if idx in selected:
            continue
        selected.add(idx)
        suffix_indices.append(idx)
        if len(suffix_indices) >= suffix_count:
            break

    ordered_indices = prefix_indices + sorted(suffix_indices)
    return [samples[idx] for idx in ordered_indices]


def _max_polytomy_size_from_newick(newick):
    if not newick:
        return 0
    groups = get_structural_polytomy_groups_from_newick(newick)
    if not groups:
        return 0
    return max(len(group) for group in groups)


def _filter_replay_samples_by_max_polytomy(samples, tree_key, max_polytomy_size):
    max_polytomy_size = int(max_polytomy_size)
    if max_polytomy_size < 0:
        return list(samples)
    filtered = []
    for sample in samples:
        tree_newick = sample.get(tree_key)
        if _max_polytomy_size_from_newick(tree_newick) <= max_polytomy_size:
            filtered.append(sample)
    return filtered


def _sample_replay_bank_samples(samples, max_count):
    samples = list(samples)
    max_count = int(max_count)
    if max_count <= 0 or not samples:
        return []
    if len(samples) <= max_count:
        return samples
    sampled_indices = sorted(random.sample(range(len(samples)), max_count))
    return [samples[idx] for idx in sampled_indices]


def _apply_boundary_vanish_one_step(
    lengths,
    velocities,
    predicted_vanish_mask,
    supervised_mask,
    dt_cap,
    eps_len,
):
    predicted_vanish_mask = np.asarray(predicted_vanish_mask, dtype=bool)
    supervised_mask = np.asarray(supervised_mask, dtype=bool)
    lengths = np.asarray(lengths, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)

    contract_mask = (
        predicted_vanish_mask & supervised_mask & (lengths > float(eps_len)) & (velocities < 0.0)
    )
    if not np.any(contract_mask):
        return lengths.copy(), float("inf"), False

    dt_candidates = lengths[contract_mask] / np.maximum(-velocities[contract_mask], float(eps_len))
    dt_boundary = float(np.max(dt_candidates))
    dt = min(float(dt_cap), dt_boundary)

    lengths_new = lengths + dt * velocities
    # This mode treats the vanish-set prediction as the target orthant endpoint.
    lengths_new[predicted_vanish_mask & supervised_mask] = 0.0

    blocked_collapse_mask = supervised_mask & (~predicted_vanish_mask)
    if np.any(blocked_collapse_mask):
        lengths_new[blocked_collapse_mask] = np.maximum(
            lengths_new[blocked_collapse_mask],
            float(eps_len) * 10.0,
        )

    return lengths_new, dt_boundary, True


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


def _structural_trees_from_samples(module, samples, newick_key, structural_key):
    structural_trees = []
    fallback_positions = []
    fallback_newicks = []
    for idx, sample in enumerate(samples):
        structural = sample.get(structural_key)
        if structural is None:
            fallback_positions.append(idx)
            fallback_newicks.append(sample[newick_key])
            structural_trees.append(None)
        else:
            structural_trees.append(structural)
    if fallback_newicks:
        parsed = _structuralize_trees_with_cache(module, fallback_newicks)
        for idx, structural in zip(fallback_positions, parsed):
            structural_trees[idx] = structural
    return structural_trees


def _tokenize_samples_with_optional_raw_graphs(
    module,
    samples,
    newick_key,
    structural_key,
    raw_graph_key,
):
    raw_graphs = [sample.get(raw_graph_key) for sample in samples]
    if raw_graphs and all(raw_graph is not None for raw_graph in raw_graphs):
        tokenizer = module.model.tokenizer
        if hasattr(tokenizer, "forward_raw_graph_cache"):
            return tokenizer.forward_raw_graph_cache(raw_graphs)
    return module.model.tokenizer(
        _structural_trees_from_samples(
            module,
            samples,
            newick_key,
            structural_key,
        )
    )


def _tokenized_batch_size_from_tokenizer_output(tokenized_trees):
    if tokenized_trees is None or not tokenized_trees:
        return 0
    return int(tokenized_trees[0].shape[0])


def _slice_tokenized_tree_batch(tokenized_trees, start, end):
    start = int(start)
    end = int(end)
    if tokenized_trees is None:
        return None
    return (
        tokenized_trees[0][start:end],
        tokenized_trees[1][start:end],
        tokenized_trees[2][start:end],
        tokenized_trees[3][start:end],
        list(tokenized_trees[4][start:end]),
        tokenized_trees[5][start:end],
        list(tokenized_trees[6][start:end]),
    )


def _tokenize_mixed_samples_with_optional_raw_graphs(module, specs):
    specs = list(specs or [])
    if not specs:
        return None
    raw_graphs = [sample.get(raw_key) for sample, _newick, _struct, raw_key in specs]
    if raw_graphs and all(raw_graph is not None for raw_graph in raw_graphs):
        tokenizer = module.model.tokenizer
        if hasattr(tokenizer, "forward_raw_graph_cache"):
            return tokenizer.forward_raw_graph_cache(raw_graphs)

    structural_trees = []
    fallback_positions = []
    fallback_newicks = []
    for idx, (sample, newick_key, structural_key, _raw_key) in enumerate(specs):
        structural = sample.get(structural_key)
        if structural is None:
            fallback_positions.append(idx)
            fallback_newicks.append(sample[newick_key])
            structural_trees.append(None)
        else:
            structural_trees.append(structural)
    if fallback_newicks:
        parsed = _structuralize_trees_with_cache(module, fallback_newicks)
        for idx, structural in zip(fallback_positions, parsed):
            structural_trees[idx] = structural
    return module.model.tokenizer(structural_trees)


def _tokenize_trees_with_structural_cache(module, trees):
    return module.model.tokenizer(_structuralize_trees_with_cache(module, trees))


def _pair_oracle_velocity_label_map_cache(module):
    cache = getattr(module, "_pair_oracle_velocity_label_map_cache", None)
    if cache is None:
        cache = OrderedDict()
        module._pair_oracle_velocity_label_map_cache = cache
    return cache


def _pair_oracle_velocity_label_map_with_cache(module, start_tree, target_tree):
    if not start_tree or not target_tree:
        return {}, 0

    key = (str(start_tree), str(target_tree))
    cache = _pair_oracle_velocity_label_map_cache(module)
    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached

    value = _build_pair_oracle_orthant_velocity_label_map(
        str(start_tree),
        str(target_tree),
    )
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > 4096:
        cache.popitem(last=False)
    return value


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


def _pooled_probe_terminal_stats(x, *, device, dtype):
    if x is None or x.numel() == 0:
        return torch.zeros(4, device=device, dtype=dtype)
    flat = x.reshape(-1).to(device=device, dtype=dtype)
    return torch.stack(
        [
            flat.mean(),
            flat.min(),
            flat.max(),
            torch.linalg.vector_norm(flat),
        ],
        dim=0,
    )


def _build_probe_terminal_feature(
    lengths,
    velocities,
    first_hit_logits,
    boundary_vanish_logits,
):
    device = lengths.device
    dtype = lengths.dtype
    count_feat = torch.tensor(
        [
            float(lengths.numel()),
            float(torch.count_nonzero(lengths > 1e-8).item()),
        ],
        device=device,
        dtype=dtype,
    )
    return torch.cat(
        [
            _pooled_probe_terminal_stats(lengths, device=device, dtype=dtype),
            _pooled_probe_terminal_stats(velocities, device=device, dtype=dtype),
            _pooled_probe_terminal_stats(
                first_hit_logits, device=device, dtype=dtype
            ),
            _pooled_probe_terminal_stats(
                boundary_vanish_logits, device=device, dtype=dtype
            ),
            count_feat,
        ],
        dim=0,
    )


def _build_topology_terminal_feature(
    module,
    aligned_model_masks,
    *,
    supervised_mask=None,
    device,
    dtype,
):
    if aligned_model_masks is None:
        return torch.zeros(int(module.model.embed_dim), device=device, dtype=dtype)

    masks = [int(mask) for mask in aligned_model_masks]
    if supervised_mask is not None:
        keep = (
            supervised_mask.reshape(-1)
            .detach()
            .to(device="cpu", dtype=torch.bool)
            .tolist()
        )
        masks = [mask for mask, include in zip(masks, keep) if include]

    if not masks:
        return torch.zeros(int(module.model.embed_dim), device=device, dtype=dtype)

    split_identity = module.model.create_split_identity_embedding(masks, device=device)
    if split_identity.numel() == 0:
        return torch.zeros(int(module.model.embed_dim), device=device, dtype=dtype)
    split_identity = split_identity.to(device=device, dtype=dtype)
    topology_pool = str(
        getattr(module, "velocity_terminal_head_topology_pool", "mean")
    ).lower()
    if topology_pool == "sum":
        return split_identity.sum(dim=0)
    return split_identity.mean(dim=0)


def _build_edge_terminal_feature(
    edge_features,
    *,
    supervised_mask=None,
    device,
    dtype,
):
    if edge_features is None or edge_features.numel() == 0:
        return None

    edge_features_flat = edge_features.reshape(-1, edge_features.shape[-1]).to(
        device=device, dtype=dtype
    )
    if supervised_mask is not None:
        keep = supervised_mask.reshape(-1).to(device=device, dtype=torch.bool)
        if keep.numel() == edge_features_flat.shape[0]:
            filtered = edge_features_flat[keep]
            if filtered.numel() > 0:
                edge_features_flat = filtered

    if edge_features_flat.numel() == 0:
        return None
    return edge_features_flat.mean(dim=0)


def _build_edge_topology_terminal_feature(
    module,
    edge_features,
    aligned_model_masks,
    *,
    supervised_mask=None,
    device,
    dtype,
):
    if edge_features is None or edge_features.numel() == 0 or aligned_model_masks is None:
        return None

    edge_features_flat = edge_features.reshape(-1, edge_features.shape[-1]).to(
        device=device, dtype=dtype
    )
    masks = [int(mask) for mask in aligned_model_masks]
    if supervised_mask is not None:
        keep = supervised_mask.reshape(-1).to(device=device, dtype=torch.bool)
        if keep.numel() == edge_features_flat.shape[0]:
            filtered = edge_features_flat[keep]
            if filtered.numel() > 0:
                edge_features_flat = filtered
            masks = [mask for mask, include in zip(masks, keep.tolist()) if include]

    if edge_features_flat.numel() == 0 or not masks:
        return None

    split_identity = module.model.create_split_identity_embedding(masks, device=device)
    if split_identity.numel() == 0:
        return None
    split_identity = split_identity.to(device=device, dtype=dtype)

    if split_identity.shape[0] != edge_features_flat.shape[0]:
        n = min(split_identity.shape[0], edge_features_flat.shape[0])
        if n <= 0:
            return None
        split_identity = split_identity[:n]
        edge_features_flat = edge_features_flat[:n]

    fused = torch.cat([edge_features_flat, split_identity], dim=-1)
    fused = module.velocity_terminal_head_edge_topology_fusion(fused)
    if fused.numel() == 0:
        return None
    return fused.mean(dim=0)


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


def _build_case_index_tensor_from_group_keys(
    group_keys,
    *,
    device,
    require_all_or_none=True,
):
    if group_keys is None:
        return None
    indices = []
    any_case_index = False
    for group_key in group_keys:
        case_index = _extract_case_index_from_group_key(group_key)
        indices.append(-1 if case_index is None else int(case_index))
        any_case_index = any_case_index or (case_index is not None)
    if not any_case_index:
        return None
    if require_all_or_none and any(int(idx) < 0 for idx in indices):
        raise ValueError(
            "Mixed missing/valid bank_group_key values are not supported for case indexing."
        )
    return torch.tensor(indices, dtype=torch.long, device=device)


def _case_index_group_keys_for_samples(module, samples):
    keys = [
        sample.get("bank_group_key") or sample.get("group_key")
        for sample in (samples or [])
    ]
    if any(_extract_case_index_from_group_key(key) is not None for key in keys):
        return keys

    dataset_obj = getattr(module, "dataset", None)
    dataset_splits = []
    if dataset_obj is not None:
        dataset_splits.extend(
            split
            for split in (
                getattr(dataset_obj, "dataset_train", None),
                getattr(dataset_obj, "dataset_val", None),
                getattr(dataset_obj, "dataset_test", None),
            )
            if split is not None
        )
    for dataset_split in dataset_splits:
        for attr in (
            "overfit_fixed_pair_start_tree_bank_items",
            "overfit_fixed_pair_target_tree_bank_items",
        ):
            items = list(getattr(dataset_split, attr, []) or [])
            if len(items) != 1:
                continue
            key = items[0].get("bank_group_key") or items[0].get("group_key")
            if _extract_case_index_from_group_key(key) is not None:
                return [key for _ in keys]
    return keys


def _case_index_group_key_for_pair(module, pair):
    key = pair.get("bank_group_key") or pair.get("group_key")
    if _extract_case_index_from_group_key(key) is not None:
        return key
    keys = _case_index_group_keys_for_samples(module, [pair])
    if keys:
        return keys[0]
    return key


def _selected_sequence_fields_from_samples(samples):
    selected_sequences = [sample.get("selected_sequences") for sample in samples]
    selected_sequence_names = [
        sample.get("selected_sequence_names") for sample in samples
    ]
    if not any(value is not None for value in selected_sequences):
        selected_sequences = None
    if not any(value is not None for value in selected_sequence_names):
        selected_sequence_names = None
    return selected_sequences, selected_sequence_names


def _normalize_selected_sequence_cases(selected_sequences, selected_sequence_names=None):
    if selected_sequences is None:
        return [], []
    if isinstance(selected_sequences, tuple):
        selected_sequences = list(selected_sequences)
    if isinstance(selected_sequences, str):
        selected_sequences = [[selected_sequences]]
    elif selected_sequences and all(
        isinstance(item, str) for item in selected_sequences
    ):
        selected_sequences = [selected_sequences]
    else:
        selected_sequences = list(selected_sequences)

    if selected_sequence_names is None:
        selected_sequence_names = [None] * len(selected_sequences)
    else:
        if isinstance(selected_sequence_names, tuple):
            selected_sequence_names = list(selected_sequence_names)
        if isinstance(selected_sequence_names, str):
            selected_sequence_names = [[selected_sequence_names]]
        elif selected_sequence_names and all(
            isinstance(item, str) for item in selected_sequence_names
        ):
            selected_sequence_names = [selected_sequence_names]
        else:
            selected_sequence_names = list(selected_sequence_names)
        if len(selected_sequence_names) != len(selected_sequences):
            selected_sequence_names = [None] * len(selected_sequences)

    return selected_sequences, selected_sequence_names


def _canonical_selected_sequence_key(sequences, names=None):
    if sequences is None:
        return None
    sequence_tuple = tuple(str(sequence) for sequence in list(sequences))
    if not sequence_tuple:
        return None
    if names is None:
        name_tuple = ()
    else:
        name_tuple = tuple(str(name) for name in list(names))
        if len(name_tuple) != len(sequence_tuple):
            name_tuple = ()
    return name_tuple, sequence_tuple


def _common_selected_sequence_fields(selected_sequences, selected_sequence_names=None):
    selected_sequences, selected_sequence_names = _normalize_selected_sequence_cases(
        selected_sequences,
        selected_sequence_names,
    )
    if not selected_sequences:
        return None, None, None

    common_key = None
    common_sequences = None
    common_names = None
    for sequences, names in zip(selected_sequences, selected_sequence_names):
        if sequences is None:
            return None, None, None
        sequence_list = list(sequences)
        name_list = None if names is None else list(names)
        key = _canonical_selected_sequence_key(sequence_list, name_list)
        if key is None:
            return None, None, None
        if common_key is None:
            common_key = key
            common_sequences = sequence_list
            common_names = name_list
        elif key != common_key:
            return None, None, None

    return common_sequences, common_names, common_key


def _selected_sequence_case_specs(selected_sequences, selected_sequence_names=None):
    selected_sequences, selected_sequence_names = _normalize_selected_sequence_cases(
        selected_sequences,
        selected_sequence_names,
    )
    if not selected_sequences:
        return None

    specs = []
    for sequences, names in zip(selected_sequences, selected_sequence_names):
        if sequences is None:
            return None
        sequence_list = list(sequences)
        name_list = None if names is None else list(names)
        key = _canonical_selected_sequence_key(sequence_list, name_list)
        if key is None:
            return None
        specs.append((key, sequence_list, name_list))
    return specs


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


def _select_velocity_replay_samples(module, samples):
    all_samples = list(samples)
    filtered_samples = list(all_samples)
    if bool(getattr(module, "velocity_probe_direct_set_anchor_only", False)):
        anchor_samples = [sample for sample in filtered_samples if sample.get("anchor_family")]
        if anchor_samples:
            filtered_samples = anchor_samples
    return all_samples, filtered_samples


def _build_velocity_replay_batch(module, samples, tokenized_override=None):
    if not samples:
        return None

    all_samples, samples = _select_velocity_replay_samples(module, samples)
    if not samples:
        return None

    newicks = [sample["newick_tree"] for sample in samples]
    if tokenized_override is None:
        tokenized = _tokenize_samples_with_optional_raw_graphs(
            module,
            samples,
            "newick_tree",
            "newick_tree_structural",
            "newick_tree_tokenizer_raw_graph",
        )
    else:
        tokenized = tokenized_override
    tokenized = _move_tokenized_batch_to_device(tokenized, module.device)
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
                canonical_map, _ = _pair_oracle_velocity_label_map_with_cache(
                    module,
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
    if getattr(module.model, "first_hit_head_mode", "base") in {
        "case_adapted_mlp",
        "frozen_start_case_mlp",
    }:
        first_hit_case_index_tensor = _build_case_index_tensor_from_group_keys(
            _case_index_group_keys_for_samples(module, samples),
            device=module.device,
        )
    dataset_ids = [
        str(sample.get("dataset_id")).upper()
        if sample.get("dataset_id") is not None
        else None
        for sample in samples
    ]
    ids = [
        sample.get("id") or dataset_id or str(idx)
        for idx, (sample, dataset_id) in enumerate(zip(samples, dataset_ids))
    ]
    mappings = [sample.get("num_to_name") for sample in samples]
    num_leaves = [int(sample["num_leaves"]) for sample in samples]
    selected_sequences, selected_sequence_names = _selected_sequence_fields_from_samples(
        samples
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
        "phyla_embeddings": None,
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
        "num_leaves": num_leaves,
        "ids": ids,
        "dataset_ids": dataset_ids,
        "mappings": mappings,
        "selected_sequences": selected_sequences,
        "selected_sequence_names": selected_sequence_names,
        "_probe_direct_set_targets": probe_direct_set_targets,
        "_probe_direct_set_sample_mask": probe_direct_set_mask,
        "_first_hit_case_indices": first_hit_case_index_tensor,
    }


def _build_terminal_replay_batch(module, samples):
    if not samples:
        return None

    newicks = [sample["newick_tree"] for sample in samples]
    tokenized = _move_tokenized_batch_to_device(
        _tokenize_samples_with_optional_raw_graphs(
            module,
            samples,
            "newick_tree",
            "newick_tree_structural",
            "newick_tree_tokenizer_raw_graph",
        ),
        module.device,
    )
    first_hit_case_index_tensor = None
    if getattr(module.model, "first_hit_head_mode", "base") == "case_adapted_mlp":
        first_hit_case_index_tensor = _build_case_index_tensor_from_group_keys(
            [sample.get("bank_group_key") for sample in samples],
            device=module.device,
        )
    dataset_ids = [
        str(sample.get("dataset_id")).upper()
        if sample.get("dataset_id") is not None
        else None
        for sample in samples
    ]
    ids = [
        sample.get("id") or dataset_id or str(idx)
        for idx, (sample, dataset_id) in enumerate(zip(samples, dataset_ids))
    ]
    mappings = [sample.get("num_to_name") for sample in samples]
    num_leaves = [
        int(sample.get("num_leaves", Tree(sample["newick_tree"]).n_leaves))
        for sample in samples
    ]
    selected_sequences, selected_sequence_names = _selected_sequence_fields_from_samples(
        samples
    )
    return {
        "_is_replay_batch": True,
        "_skip_training_augmentations": True,
        "tokenized_trees": tokenized,
        "batched_time": torch.tensor(
            [float(sample["timepoint"]) for sample in samples],
            dtype=torch.float32,
            device=module.device,
        ),
        "batched_terminal_stop": torch.tensor(
            [
                1.0 if sample.get("terminal_stop", False) else 0.0
                for sample in samples
            ],
            dtype=torch.float32,
            device=module.device,
        ),
        "phyla_embeddings": None,
        "original_trees": newicks,
        "start_trees": [
            sample.get("start_tree", sample.get("newick_tree")) for sample in samples
        ],
        "target_trees": [sample.get("target_tree") for sample in samples],
        "bank_group_key": [sample.get("bank_group_key") for sample in samples],
        "num_leaves": num_leaves,
        "ids": ids,
        "dataset_ids": dataset_ids,
        "mappings": mappings,
        "selected_sequences": selected_sequences,
        "selected_sequence_names": selected_sequence_names,
        "_first_hit_case_indices": first_hit_case_index_tensor,
    }


def _build_autoregressive_replay_batch(module, samples, tokenized_override=None):
    if not samples:
        return None

    newicks = [sample["newick"] for sample in samples]
    if tokenized_override is None:
        tokenized = _tokenize_samples_with_optional_raw_graphs(
            module,
            samples,
            "newick",
            "newick_structural",
            "newick_tokenizer_raw_graph",
        )
    else:
        tokenized = tokenized_override
    tokenized = _move_tokenized_batch_to_device(tokenized, module.device)
    autoregressive_case_index_tensor = None
    autoregressive_needs_case_indices = bool(
        getattr(module.model, "autoregressive_use_case_conditioning", False)
    ) or (
        bool(
            getattr(
                module.model,
                "autoregressive_use_start_topology_conditioning",
                False,
            )
        )
        and str(
            getattr(
                module.model,
                "autoregressive_start_topology_conditioning_mode",
                "",
            )
        )
        in {"frozen_case_probe", "frozen_case_probe_additive"}
    )
    if autoregressive_needs_case_indices:
        autoregressive_case_index_tensor = _build_case_index_tensor_from_group_keys(
            _case_index_group_keys_for_samples(module, samples),
            device=module.device,
        )
    dataset_ids = [
        str(sample.get("dataset_id")).upper()
        if sample.get("dataset_id") is not None
        else None
        for sample in samples
    ]
    ids = [
        sample.get("id") or dataset_id or str(idx)
        for idx, (sample, dataset_id) in enumerate(zip(samples, dataset_ids))
    ]
    mappings = [sample.get("num_to_name") for sample in samples]
    num_leaves = [
        int(sample.get("num_leaves", Tree(sample["newick"]).n_leaves))
        for sample in samples
    ]
    selected_sequences, selected_sequence_names = _selected_sequence_fields_from_samples(
        samples
    )
    return {
        "_is_replay_batch": True,
        "_skip_training_augmentations": True,
        "tokenized_autoregressive_trees": tokenized,
        "newick_autoregressive_trees": newicks,
        "start_trees": [sample.get("start_tree", sample.get("newick")) for sample in samples],
        "target_trees": [sample["target_tree"] for sample in samples],
        "bank_group_key": [sample.get("bank_group_key") for sample in samples],
        "batched_autoregressive_time": torch.tensor(
            [float(sample["time"]) for sample in samples],
            dtype=torch.float32,
            device=module.device,
        ),
        "batched_autoregressive_labels": [sample["labels"] for sample in samples],
        "batched_birthset_precomputed": [
            sample.get("birthset_precomputed") for sample in samples
        ],
        "_birthset_precomputed_candidate_info_enabled": bool(
            any(
                bool(sample.get("_birthset_precomputed_candidate_info_enabled", False))
                for sample in samples
            )
        ),
        "batched_autoregressive_stop_after_merge": torch.tensor(
            [
                1.0 if sample.get("stop_after_merge", False) else 0.0
                for sample in samples
            ],
            dtype=torch.float32,
            device=module.device,
        ),
        "phyla_embeddings": None,
        "num_leaves": num_leaves,
        "ids": ids,
        "dataset_ids": dataset_ids,
        "mappings": mappings,
        "selected_sequences": selected_sequences,
        "selected_sequence_names": selected_sequence_names,
        "_autoregressive_case_indices": autoregressive_case_index_tensor,
    }


def _build_velocity_autoregressive_replay_batches(
    module,
    velocity_samples,
    autoregressive_samples,
    joint_raw_graph_batch=None,
):
    velocity_samples = list(velocity_samples or [])
    autoregressive_samples = list(autoregressive_samples or [])
    _velocity_all_samples, selected_velocity_samples = _select_velocity_replay_samples(
        module,
        velocity_samples,
    )
    combined_tokenized = None
    if (
        isinstance(joint_raw_graph_batch, dict)
        and bool(
            joint_raw_graph_batch.get(
                "_tree_tokenizer_raw_graph_batch_cache",
                False,
            )
        )
        and int(joint_raw_graph_batch.get("velocity_count", -1))
        == len(selected_velocity_samples)
        and int(joint_raw_graph_batch.get("autoregressive_count", -1))
        == len(autoregressive_samples)
    ):
        tokenizer = module.model.tokenizer
        if hasattr(tokenizer, "forward_raw_graph_cache"):
            combined_tokenized = tokenizer.forward_raw_graph_cache(
                joint_raw_graph_batch
            )
    if combined_tokenized is None:
        specs = []
        if selected_velocity_samples:
            specs.extend(
                (
                    sample,
                    "newick_tree",
                    "newick_tree_structural",
                    "newick_tree_tokenizer_raw_graph",
                )
                for sample in selected_velocity_samples
            )
        if autoregressive_samples:
            specs.extend(
                (
                    sample,
                    "newick",
                    "newick_structural",
                    "newick_tokenizer_raw_graph",
                )
                for sample in autoregressive_samples
            )
        combined_tokenized = _tokenize_mixed_samples_with_optional_raw_graphs(
            module,
            specs,
        )
    if combined_tokenized is None:
        return (
            _build_velocity_replay_batch(module, velocity_samples),
            _build_autoregressive_replay_batch(module, autoregressive_samples),
        )
    combined_tokenized = _move_tokenized_batch_to_device(
        combined_tokenized,
        module.device,
    )
    velocity_count = len(selected_velocity_samples)
    autoregressive_count = len(autoregressive_samples)
    if _tokenized_batch_size_from_tokenizer_output(combined_tokenized) != (
        velocity_count + autoregressive_count
    ):
        return (
            _build_velocity_replay_batch(module, velocity_samples),
            _build_autoregressive_replay_batch(module, autoregressive_samples),
        )

    velocity_tokenized = (
        _slice_tokenized_tree_batch(combined_tokenized, 0, velocity_count)
        if velocity_count > 0
        else None
    )
    autoregressive_tokenized = (
        _slice_tokenized_tree_batch(
            combined_tokenized,
            velocity_count,
            velocity_count + autoregressive_count,
        )
        if autoregressive_count > 0
        else None
    )
    velocity_batch = _build_velocity_replay_batch(
        module,
        velocity_samples,
        tokenized_override=velocity_tokenized,
    )
    autoregressive_batch = _build_autoregressive_replay_batch(
        module,
        autoregressive_samples,
        tokenized_override=autoregressive_tokenized,
    )
    if velocity_batch is not None and autoregressive_batch is not None:
        for batch in (velocity_batch, autoregressive_batch):
            batch["_joint_tokenized_trees"] = combined_tokenized
            batch["_joint_velocity_batch_size"] = int(velocity_count)
            batch["_joint_autoregressive_batch_size"] = int(autoregressive_count)
    return velocity_batch, autoregressive_batch


_FULL_PATH_REPLAY_SAMPLE_KEYS = (
    "full_path_velocity_samples",
    "full_path_autoregressive_samples",
    "full_path_terminal_samples",
)


def _clone_full_path_replay_retry_base(batch):
    if not isinstance(batch, dict):
        return batch
    cloned = dict(batch)
    for key in _FULL_PATH_REPLAY_SAMPLE_KEYS:
        if key in cloned:
            cloned[key] = list(cloned.get(key) or [])
    return cloned


def _stable_replay_retry_rng(stepper, retry_attempt, key):
    payload = f"{int(stepper)}:{int(retry_attempt)}:{str(key)}"
    seed = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
    return random.Random(seed)


def _sample_replay_items_for_retry(items, keep_count, rng):
    items = list(items or [])
    if keep_count >= len(items):
        return items
    if keep_count <= 0:
        return []
    kept_indices = sorted(rng.sample(range(len(items)), int(keep_count)))
    return [items[idx] for idx in kept_indices]


def _subsample_full_path_replay_for_oom_retry(batch, *, stepper, retry_attempt):
    if not isinstance(batch, dict) or not batch.get("_full_path_control_mode", False):
        return batch, False, {}

    retry_attempt = max(1, int(retry_attempt))
    denominator = 2 ** retry_attempt
    reduced = dict(batch)
    counts = {}
    changed = False
    for key in _FULL_PATH_REPLAY_SAMPLE_KEYS:
        samples = list(batch.get(key) or [])
        original_count = len(samples)
        if original_count <= 1:
            keep_count = original_count
            reduced_samples = samples
        else:
            keep_count = max(1, original_count // denominator)
            rng = _stable_replay_retry_rng(stepper, retry_attempt, key)
            reduced_samples = _sample_replay_items_for_retry(samples, keep_count, rng)
        reduced[key] = reduced_samples
        counts[key] = (original_count, len(reduced_samples))
        changed = changed or len(reduced_samples) < original_count

    reduced["_full_path_replay_subsample_retry_attempt"] = retry_attempt
    reduced["_full_path_replay_subsample_counts"] = counts
    return reduced, changed, counts
def _make_replay_anchor_state(newick_tree, timepoint, target_tree, num_leaves):
    return {
        "newick_tree": str(newick_tree),
        "timepoint": float(timepoint),
        "target_tree": target_tree,
        "num_leaves": int(num_leaves),
    }


def _select_rollout_replay_anchors(
    trace,
    sampled_tree,
    target_tree,
    max_anchor_states,
    include_autoregressive=False,
):
    if int(max_anchor_states) <= 0:
        return []

    velocity_trace = trace.get("velocity", []) if trace is not None else []
    anchor_candidates = []
    for sample in velocity_trace:
        newick_tree = sample.get("newick_tree")
        if not newick_tree:
            continue
        anchor_candidates.append(
            _make_replay_anchor_state(
                newick_tree,
                sample.get("timepoint", 0.0),
                sample.get("target_tree", target_tree),
                sample.get("num_leaves", Tree(newick_tree).n_leaves),
            )
        )

    if bool(include_autoregressive):
        autoregressive_trace = trace.get("autoregressive", []) if trace is not None else []
        for sample in autoregressive_trace:
            newick_tree = sample.get("newick")
            if not newick_tree:
                continue
            anchor_candidates.append(
                _make_replay_anchor_state(
                    newick_tree,
                    sample.get("time", 0.0),
                    sample.get("target_tree", target_tree),
                    Tree(newick_tree).n_leaves,
                )
            )

    if sampled_tree:
        final_timepoint = (
            float(anchor_candidates[-1]["timepoint"]) if anchor_candidates else 1.0
        )
        final_num_leaves = (
            int(anchor_candidates[-1]["num_leaves"])
            if anchor_candidates
            else int(Tree(sampled_tree).n_leaves)
        )
        if not anchor_candidates or str(anchor_candidates[-1]["newick_tree"]) != str(
            sampled_tree
        ):
            anchor_candidates.append(
                _make_replay_anchor_state(
                    sampled_tree,
                    final_timepoint,
                    target_tree,
                    final_num_leaves,
                )
            )

    return _select_replay_samples_across_rollout(
        anchor_candidates,
        int(max_anchor_states),
    )


def _rescale_replay_anchor_time(anchor_time, local_progress):
    anchor_time = max(0.0, min(float(anchor_time), 1.0))
    local_progress = max(0.0, min(float(local_progress), 1.0))
    return anchor_time + (1.0 - anchor_time) * local_progress


def _collect_oracle_replay_samples_from_anchors(
    module,
    anchors,
    oracle_horizon,
    split_multi_label_events=False,
):
    velocity_samples = []
    autoregressive_samples = []
    horizon = max(int(oracle_horizon), 0)
    if horizon <= 0:
        return velocity_samples, autoregressive_samples

    for anchor in anchors:
        anchor_tree = anchor.get("newick_tree")
        target_tree = anchor.get("target_tree")
        if not anchor_tree or not target_tree:
            continue

        anchor_time = float(anchor.get("timepoint", 0.0))
        try:
            boundary_paths = return_tree_boundary_merge_paths(anchor_tree, target_tree)
        except Exception:
            continue

        max_paths = min(len(boundary_paths), horizon)
        if max_paths <= 0:
            continue

        for path_idx in range(max_paths):
            if path_idx == 0:
                source_tree = anchor_tree
                local_progress = 0.0
            else:
                source_tree = boundary_paths[path_idx - 1]["end_newick"]
                local_progress = float(boundary_paths[path_idx - 1]["global_time"])

            try:
                velocity_newick, oracle_velocity = return_sampled_tree_orthant_velocity(
                    source_tree,
                    target_tree,
                    0.0,
                )
            except Exception:
                continue

            velocity_samples.append(
                {
                    "newick_tree": velocity_newick,
                    "target_tree": target_tree,
                    "velocity": oracle_velocity,
                    "velocity_next_boundary_tree": boundary_paths[path_idx][
                        "start_newick"
                    ],
                    "timepoint": _rescale_replay_anchor_time(
                        anchor_time,
                        local_progress,
                    ),
                    "num_leaves": int(anchor.get("num_leaves", Tree(source_tree).n_leaves)),
                }
            )

            boundary_time = _rescale_replay_anchor_time(
                anchor_time,
                float(boundary_paths[path_idx]["global_time"]),
            )
            boundary_events = list(boundary_paths[path_idx].get("events", []))
            if split_multi_label_events:
                boundary_events = _split_multi_label_training_events(boundary_events)

            for event in boundary_events:
                if not event.get("labels"):
                    continue
                autoregressive_samples.append(
                    {
                        "newick": str(event["newick"]),
                        "target_tree": target_tree,
                        "labels": event["labels"],
                        "stop_after_merge": bool(
                            event.get("stop_after_merge", False)
                        ),
                        "time": float(boundary_time),
                    }
                )

    return velocity_samples, autoregressive_samples


def _build_legacy_velocity_oracle_sample(
    source_tree,
    target_tree,
    *,
    timepoint=0.0,
    num_leaves=None,
):
    if not source_tree or not target_tree:
        return None
    try:
        velocity_newick, oracle_velocity = return_sampled_tree_orthant_velocity(
            source_tree,
            target_tree,
            0.0,
        )
        boundary_paths = return_tree_boundary_merge_paths(source_tree, target_tree)
    except Exception:
        return None
    next_boundary_tree = boundary_paths[0]["start_newick"] if boundary_paths else None
    if num_leaves is None:
        try:
            num_leaves = int(Tree(source_tree).n_leaves)
        except Exception:
            num_leaves = 0
    return {
        "newick_tree": velocity_newick,
        "target_tree": target_tree,
        "velocity": oracle_velocity,
        "velocity_next_boundary_tree": next_boundary_tree,
        "timepoint": float(timepoint),
        "num_leaves": int(num_leaves),
    }


def _velocity_replay_label_signature(sample):
    velocity_items = tuple(
        sorted(
            (
                int(split_mask),
                round(float(value), 12),
            )
            for split_mask, value in sample.get("velocity", {}).items()
        )
    )
    next_boundary_key = None
    if sample.get("velocity_next_boundary_tree"):
        try:
            next_boundary_key = _topology_key(sample["velocity_next_boundary_tree"])
        except Exception:
            next_boundary_key = str(sample.get("velocity_next_boundary_tree"))
    return velocity_items, next_boundary_key


def _velocity_replay_state_key(newick_tree, next_boundary_tree):
    topology_key = None
    next_boundary_key = None
    try:
        topology_key = tuple(_topology_key(str(newick_tree)))
    except Exception:
        topology_key = None
    if next_boundary_tree:
        try:
            next_boundary_key = tuple(_topology_key(str(next_boundary_tree)))
        except Exception:
            next_boundary_key = str(next_boundary_tree)
    return topology_key, next_boundary_key


def _build_pair_oracle_orthant_velocity_label_map(start_tree, target_tree):
    if not start_tree or not target_tree:
        return {}, 0

    try:
        boundary_paths = return_tree_boundary_merge_paths(start_tree, target_tree)
    except Exception:
        return {}, 0
    if not boundary_paths:
        return {}, 0

    velocity_trees = [str(start_tree)]
    velocity_trees.extend(str(path["end_newick"]) for path in boundary_paths[:-1])
    timepoints = [0.0]
    timepoints.extend(float(path["global_time"]) for path in boundary_paths[:-1])
    next_boundary_trees = [str(path["start_newick"]) for path in boundary_paths]

    canonical_by_topology = {}
    ambiguous_topologies = set()
    for source_tree, next_boundary_tree, model_time in zip(
        velocity_trees,
        next_boundary_trees,
        timepoints,
    ):
        velocity_sample = _build_legacy_velocity_oracle_sample(
            source_tree,
            target_tree,
            timepoint=float(model_time),
        )
        if velocity_sample is None:
            continue
        velocity_sample = dict(velocity_sample)
        velocity_sample["velocity_next_boundary_tree"] = str(next_boundary_tree)
        try:
            topology_key = _topology_key(velocity_sample["newick_tree"])
        except Exception:
            continue
        existing = canonical_by_topology.get(topology_key)
        if existing is None:
            canonical_by_topology[topology_key] = velocity_sample
            continue
        if _velocity_replay_label_signature(existing) != _velocity_replay_label_signature(
            velocity_sample
        ):
            ambiguous_topologies.add(topology_key)

    for topology_key in ambiguous_topologies:
        canonical_by_topology.pop(topology_key, None)
    return canonical_by_topology, int(len(ambiguous_topologies))


def _apply_pair_oracle_orthant_velocity_labels(velocity_samples, pair):
    if not velocity_samples:
        return list(velocity_samples), {
            "matched": 0,
            "unmatched": 0,
            "canonical_topologies": 0,
            "ambiguous_topologies": 0,
        }

    canonical_map, ambiguous_topologies = _build_pair_oracle_orthant_velocity_label_map(
        pair.get("start_tree"),
        pair.get("target_tree"),
    )
    if not canonical_map:
        return list(velocity_samples), {
            "matched": 0,
            "unmatched": int(len(velocity_samples)),
            "canonical_topologies": 0,
            "ambiguous_topologies": int(ambiguous_topologies),
        }

    relabeled_samples = []
    matched = 0
    unmatched = 0
    for sample in velocity_samples:
        relabeled = dict(sample)
        topology_key = None
        if sample.get("newick_tree"):
            try:
                topology_key = _topology_key(sample["newick_tree"])
            except Exception:
                topology_key = None
        canonical = canonical_map.get(topology_key)
        if canonical is None:
            unmatched += 1
            relabeled_samples.append(relabeled)
            continue
        relabeled["velocity"] = dict(canonical["velocity"])
        relabeled["velocity_next_boundary_tree"] = canonical.get(
            "velocity_next_boundary_tree"
        )
        matched += 1
        relabeled_samples.append(relabeled)

    return relabeled_samples, {
        "matched": int(matched),
        "unmatched": int(unmatched),
        "canonical_topologies": int(len(canonical_map)),
        "ambiguous_topologies": int(ambiguous_topologies),
    }


def _build_legacy_autoregressive_oracle_sample(
    source_tree,
    target_tree,
    module=None,
    *,
    time=0.0,
    split_multi_label_events=False,
):
    if not source_tree or not target_tree:
        return None
    try:
        current_groups = [
            tuple(int(component) for component in group)
            for group in get_structural_polytomy_groups_from_newick(source_tree)
        ]
        n_leaves = int(Tree(source_tree).n_leaves)
    except Exception:
        return None
    if not current_groups:
        return None

    labels = []
    seen = set()
    for group in current_groups:
        ready_subsets = _ready_target_merge_subsets_for_group(
            group,
            target_tree,
            n_leaves,
        )
        for subset in ready_subsets:
            normalized_subset = tuple(sorted(int(component) for component in subset))
            if len(normalized_subset) < 2:
                continue
            try:
                merge_indices = [int(group.index(component)) for component in normalized_subset]
            except ValueError:
                continue
            result_split = 0
            for component in normalized_subset:
                result_split |= int(component)
            label_key = (
                int(result_split),
                tuple(int(component) for component in group),
                tuple(int(idx) for idx in merge_indices),
            )
            if label_key in seen:
                continue
            seen.add(label_key)
            labels.append(
                {
                    "result_split": int(result_split),
                    "parent_split": int(functools.reduce(operator.or_, group, 0)),
                    "components": [int(component) for component in group],
                    "merge_indices": [int(idx) for idx in merge_indices],
                }
            )
    if not labels and module is not None:
        fallback = _best_pairwise_merge_label_for_current_tree(
            module,
            source_tree,
            target_tree,
        )
        if fallback is not None and fallback.get("labels"):
            labels = list(fallback["labels"])
    if not labels:
        return None
    if split_multi_label_events:
        labels = labels[:1]
    return {
        "newick": str(source_tree),
        "target_tree": target_tree,
        "labels": labels,
        "stop_after_merge": False,
        "time": float(time),
    }


def _build_boundary_local_autoregressive_suffix_samples(
    source_tree,
    target_tree,
    *,
    time=0.0,
):
    if not source_tree or not target_tree:
        return []
    try:
        boundary_paths = return_tree_boundary_merge_paths(source_tree, target_tree)
    except Exception:
        return []
    if not boundary_paths:
        return []

    first_path = boundary_paths[0]
    filtered_events = []
    for event in first_path.get("events", []):
        labels = [
            label
            for label in event.get("labels", [])
            if len(label.get("components", [])) >= 3
        ]
        if labels:
            filtered_events.append(
                {
                    "newick": str(event["newick"]),
                    "labels": labels,
                }
            )
    if not filtered_events:
        return []

    split_events = _split_multi_label_training_events(filtered_events)
    suffix_samples = []
    for event in split_events:
        labels = list(event.get("labels", []))
        if not labels:
            continue
        suffix_samples.append(
            {
                "newick": str(event["newick"]),
                "target_tree": target_tree,
                "labels": labels,
                "stop_after_merge": bool(event.get("stop_after_merge", False)),
                "time": float(time),
            }
        )
    return suffix_samples


def _build_full_continuation_replay_samples(
    source_tree,
    target_tree,
    *,
    anchor_time=0.0,
    num_leaves=None,
):
    if not source_tree or not target_tree:
        return [], []
    try:
        boundary_paths = return_tree_boundary_merge_paths(source_tree, target_tree)
    except Exception:
        return [], []

    velocity_samples = []
    autoregressive_samples = []

    source_has_polytomy = False
    try:
        source_has_polytomy = bool(has_polytomy_fast(source_tree, unrooted_ok=False))
    except Exception:
        source_has_polytomy = False

    if not source_has_polytomy:
        velocity_sample = _build_legacy_velocity_oracle_sample(
            source_tree,
            target_tree,
            timepoint=float(anchor_time),
            num_leaves=num_leaves,
        )
        if velocity_sample is not None:
            velocity_samples.append(velocity_sample)

    for path_idx, boundary_path in enumerate(boundary_paths):
        if path_idx > 0:
            path_start_tree = boundary_paths[path_idx - 1]["end_newick"]
            path_start_time = _rescale_replay_anchor_time(
                anchor_time,
                float(boundary_paths[path_idx - 1]["global_time"]),
            )
            velocity_sample = _build_legacy_velocity_oracle_sample(
                path_start_tree,
                target_tree,
                timepoint=float(path_start_time),
            )
            if velocity_sample is not None:
                velocity_samples.append(velocity_sample)

        filtered_events = []
        for event in boundary_path.get("events", []):
            labels = [
                label
                for label in event.get("labels", [])
                if len(label.get("components", [])) >= 3
            ]
            if labels:
                filtered_events.append(
                    {
                        "newick": str(event["newick"]),
                        "labels": labels,
                    }
                )

        if not filtered_events:
            continue

        split_events = _split_multi_label_training_events(filtered_events)
        boundary_time = _rescale_replay_anchor_time(
            anchor_time,
            float(boundary_path["global_time"]),
        )
        for event in split_events:
            labels = list(event.get("labels", []))
            if not labels:
                continue
            autoregressive_samples.append(
                {
                    "newick": str(event["newick"]),
                    "target_tree": target_tree,
                    "labels": labels,
                    "stop_after_merge": bool(event.get("stop_after_merge", False)),
                    "time": float(boundary_time),
                }
            )

    return velocity_samples, autoregressive_samples


def _trees_match_topology(tree_a, tree_b):
    if not tree_a or not tree_b:
        return False
    try:
        return _topology_key(str(tree_a)) == _topology_key(str(tree_b))
    except Exception:
        return str(tree_a) == str(tree_b)


def _find_first_wrong_velocity_suffix_replay(
    *,
    pair,
    trace,
):
    if pair is None or trace is None:
        return None
    start_tree = pair.get("start_tree")
    target_tree = pair.get("target_tree")
    if not start_tree or not target_tree:
        return None
    try:
        boundary_paths = return_tree_boundary_merge_paths(start_tree, target_tree)
    except Exception:
        return None
    if not boundary_paths:
        return None

    oracle_phase_start_trees = [str(start_tree)]
    oracle_phase_start_times = [0.0]
    for path_item in boundary_paths[:-1]:
        oracle_phase_start_trees.append(str(path_item["end_newick"]))
        oracle_phase_start_times.append(float(path_item["global_time"]))
    oracle_boundary_trees = [str(path_item["start_newick"]) for path_item in boundary_paths]

    sampled_velocity_by_phase = {}
    for idx, sample in enumerate(trace.get("velocity", []) or []):
        newick_tree = sample.get("newick_tree")
        if not newick_tree:
            continue
        try:
            phase_idx = int(sample.get("phase_idx", idx))
        except Exception:
            phase_idx = int(idx)
        sampled_velocity_by_phase[phase_idx] = sample

    if not sampled_velocity_by_phase:
        return None

    sampled_phase_start_trees = {0: str(start_tree)}
    autoregressive_trace = trace.get("autoregressive", []) or []
    for phase_idx in range(1, len(oracle_boundary_trees)):
        previous_phase = phase_idx - 1
        previous_phase_ar = []
        for sample in autoregressive_trace:
            raw_phase = sample.get("phase_idx")
            if raw_phase is None:
                continue
            try:
                if int(raw_phase) == previous_phase:
                    previous_phase_ar.append(sample)
            except Exception:
                continue
        if previous_phase_ar:
            last_ar = previous_phase_ar[-1]
            sampled_start_tree = last_ar.get("newick") or last_ar.get("source_newick")
            if sampled_start_tree:
                sampled_phase_start_trees[phase_idx] = str(sampled_start_tree)
                continue
        previous_velocity = sampled_velocity_by_phase.get(previous_phase)
        if previous_velocity and previous_velocity.get("newick_tree"):
            sampled_phase_start_trees[phase_idx] = str(previous_velocity["newick_tree"])

    for phase_idx, oracle_boundary_tree in enumerate(oracle_boundary_trees):
        sampled_velocity = sampled_velocity_by_phase.get(phase_idx)
        oracle_start_tree = oracle_phase_start_trees[min(phase_idx, len(oracle_phase_start_trees) - 1)]
        oracle_start_time = oracle_phase_start_times[min(phase_idx, len(oracle_phase_start_times) - 1)]
        sampled_start_tree = sampled_phase_start_trees.get(phase_idx, oracle_start_tree)
        if sampled_velocity is None:
            return {
                "first_wrong_phase_idx": int(phase_idx),
                "sampled_start_tree": str(sampled_start_tree),
                "oracle_start_tree": str(oracle_start_tree),
                "oracle_start_time": float(oracle_start_time),
                "sampled_boundary_tree": None,
                "oracle_boundary_tree": str(oracle_boundary_tree),
                "sampled_start_rf_to_oracle_start": float(
                    calculate_norm_rf(str(sampled_start_tree), str(oracle_start_tree))
                ),
            }
        sampled_boundary_tree = sampled_velocity.get("newick_tree")
        if _trees_match_topology(sampled_boundary_tree, oracle_boundary_tree):
            continue
        payload = {
            "first_wrong_phase_idx": int(phase_idx),
            "sampled_start_tree": str(sampled_start_tree),
            "oracle_start_tree": str(oracle_start_tree),
            "oracle_start_time": float(oracle_start_time),
            "sampled_boundary_tree": str(sampled_boundary_tree),
            "oracle_boundary_tree": str(oracle_boundary_tree),
            "sampled_start_rf_to_oracle_start": float(
                calculate_norm_rf(str(sampled_start_tree), str(oracle_start_tree))
            ),
        }
        if sampled_boundary_tree:
            payload["sampled_boundary_rf_to_oracle_boundary"] = float(
                calculate_norm_rf(str(sampled_boundary_tree), str(oracle_boundary_tree))
            )
        return payload

    return None


def _collect_first_wrong_velocity_suffix_replay_samples(
    module,
    *,
    pair,
    trace,
):
    replay_focus = _find_first_wrong_velocity_suffix_replay(
        pair=pair,
        trace=trace,
    )
    if replay_focus is None:
        return [], [], None
    velocity_samples, autoregressive_samples = _build_full_continuation_replay_samples(
        replay_focus["sampled_start_tree"],
        pair["target_tree"],
        anchor_time=float(replay_focus["oracle_start_time"]),
        num_leaves=pair.get("n_leaves"),
    )
    return velocity_samples, autoregressive_samples, replay_focus


def _collect_legacy_oracle_replay_samples_from_trace(
    trace,
    module=None,
    split_multi_label_events=False,
    terminal_tree=None,
    terminal_target_tree=None,
):
    velocity_samples = []
    autoregressive_samples = []
    if trace is None:
        return velocity_samples, autoregressive_samples

    use_full_continuation_chain = bool(
        getattr(module, "rollout_replay_full_continuation_chain", False)
    )

    def _append_full_chain_samples(
        *,
        current_tree,
        target_tree,
        timepoint,
        num_leaves=None,
    ):
        if not current_tree or not target_tree:
            return
        velocity_chain, autoregressive_chain = _build_full_continuation_replay_samples(
            current_tree,
            target_tree,
            anchor_time=timepoint,
            num_leaves=num_leaves,
        )
        velocity_samples.extend(velocity_chain)
        autoregressive_samples.extend(autoregressive_chain)

    def _append_velocity_sample(
        *,
        current_tree,
        target_tree,
        timepoint,
        num_leaves=None,
    ):
        if not current_tree or not target_tree:
            return
        velocity_sample = _build_legacy_velocity_oracle_sample(
            current_tree,
            target_tree,
            timepoint=timepoint,
            num_leaves=num_leaves,
        )
        if velocity_sample is not None:
            velocity_samples.append(velocity_sample)

    def _append_autoregressive_sample(
        *,
        current_tree,
        target_tree,
        timepoint,
    ):
        if not current_tree or not target_tree:
            return
        if bool(
            getattr(
                module,
                "rollout_replay_autoregressive_boundary_local_suffix",
                False,
            )
        ):
            autoregressive_samples.extend(
                _build_boundary_local_autoregressive_suffix_samples(
                    current_tree,
                    target_tree,
                    time=timepoint,
                )
            )
            return
        autoregressive_sample = _build_legacy_autoregressive_oracle_sample(
            current_tree,
            target_tree,
            module=module,
            time=timepoint,
            split_multi_label_events=split_multi_label_events,
        )
        if autoregressive_sample is not None:
            autoregressive_samples.append(autoregressive_sample)

    for sample in trace.get("velocity", []):
        current_tree = sample.get("newick_tree")
        target_tree = sample.get("target_tree")
        timepoint = float(sample.get("timepoint", 0.0))
        num_leaves = sample.get("num_leaves")
        if current_tree and target_tree:
            if use_full_continuation_chain:
                _append_full_chain_samples(
                    current_tree=current_tree,
                    target_tree=target_tree,
                    timepoint=timepoint,
                    num_leaves=num_leaves,
                )
                continue
            if has_polytomy_fast(current_tree, unrooted_ok=False):
                _append_autoregressive_sample(
                    current_tree=current_tree,
                    target_tree=target_tree,
                    timepoint=timepoint,
                )
            else:
                _append_velocity_sample(
                    current_tree=current_tree,
                    target_tree=target_tree,
                    timepoint=timepoint,
                    num_leaves=num_leaves,
                )

    autoregressive_trace = list(trace.get("autoregressive", []))
    for idx, sample in enumerate(autoregressive_trace):
        current_tree = sample.get("newick")
        target_tree = sample.get("target_tree")
        timepoint = float(sample.get("time", 0.0))
        if not current_tree or not target_tree:
            continue
        if use_full_continuation_chain:
            _append_full_chain_samples(
                current_tree=current_tree,
                target_tree=target_tree,
                timepoint=timepoint,
            )
            continue
        if has_polytomy_fast(current_tree, unrooted_ok=False):
            _append_autoregressive_sample(
                current_tree=current_tree,
                target_tree=target_tree,
                timepoint=timepoint,
            )
        next_time = None
        if idx + 1 < len(autoregressive_trace):
            next_time = float(autoregressive_trace[idx + 1].get("time", 0.0))
        is_last_state_for_boundary = (
            idx + 1 == len(autoregressive_trace)
            or abs(next_time - timepoint) > 1e-8
        )
        if is_last_state_for_boundary:
            _append_velocity_sample(
                current_tree=current_tree,
                target_tree=target_tree,
                timepoint=timepoint,
            )

    if terminal_tree and terminal_target_tree:
        if use_full_continuation_chain:
            _append_full_chain_samples(
                current_tree=terminal_tree,
                target_tree=terminal_target_tree,
                timepoint=1.0,
            )
            return velocity_samples, autoregressive_samples
        if has_polytomy_fast(terminal_tree, unrooted_ok=False):
            _append_autoregressive_sample(
                current_tree=terminal_tree,
                target_tree=terminal_target_tree,
                timepoint=1.0,
            )
        else:
            _append_velocity_sample(
                current_tree=terminal_tree,
                target_tree=terminal_target_tree,
                timepoint=1.0,
            )

    return velocity_samples, autoregressive_samples


def _sample_trace_states_uniform(trace_states, max_count, time_key):
    if trace_states is None:
        return []
    trace_states = list(trace_states)
    if max_count is None or int(max_count) < 0 or len(trace_states) <= int(max_count):
        return trace_states
    sampled_indices = sorted(random.sample(range(len(trace_states)), int(max_count)))
    sampled = [trace_states[idx] for idx in sampled_indices]
    if time_key is not None:
        sampled.sort(key=lambda sample: float(sample.get(time_key, 0.0)))
    return sampled


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
):
    td, _, _ = _tree_to_model_split_lengths(module, newick)
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


def _final_orthant_relax_model_time(time_mode, phase_value, local_time):
    mode = str(time_mode).lower()
    if mode == "phase":
        return float(phase_value)
    if mode == "phase_local":
        return float(phase_value) + float(local_time)
    return float(local_time)


def _relax_final_orthant_branch_lengths(
    module,
    current_newick,
    target_tree,
    phyla_embeddings,
    case_index=None,
    start_topology_features=None,
    start_topology_embeddings=None,
    start_topology_pad_mask=None,
    start_tree_graph_context=None,
    *,
    phase_value: float = 0.0,
    eps_len: float = 1e-8,
):
    steps = int(getattr(module, "sampling_final_orthant_relax_steps", 0))
    total_time = float(
        getattr(module, "sampling_final_orthant_relax_total_time", 1.0)
    )
    time_mode = str(
        getattr(module, "sampling_final_orthant_relax_time_mode", "local")
    ).lower()
    edge_floor_cfg = getattr(module, "sampling_final_orthant_relax_edge_floor", None)
    edge_floor = (
        float(edge_floor_cfg)
        if edge_floor_cfg is not None
        else max(float(eps_len) * 10.0, 1e-8)
    )
    edge_floor = max(float(edge_floor), float(eps_len) * 10.0)

    start_newick = current_newick
    summary = {
        "applied": False,
        "head": "velocity",
        "requested_steps": int(steps),
        "applied_steps": 0,
        "total_time": float(total_time),
        "time_mode": time_mode,
        "edge_floor": float(edge_floor),
        "phase_value": float(phase_value),
        "start_tree": start_newick,
        "final_tree": current_newick,
        "rf_to_target_before": None
        if target_tree is None
        else float(calculate_norm_rf(current_newick, target_tree)),
        "rf_to_target_after": None,
        "topology_rf_before_after": None,
        "min_length_before": None,
        "min_length_after": None,
        "max_abs_delta": 0.0,
        "stopped_reason": None,
    }
    states = []

    if steps <= 0:
        summary["stopped_reason"] = "no_steps_requested"
        return current_newick, summary, states
    if total_time <= 0.0:
        summary["stopped_reason"] = "nonpositive_total_time"
        return current_newick, summary, states

    relax_case_indices = (
        None
        if case_index is None
        else torch.tensor([int(case_index)], dtype=torch.long, device=module.device)
    )

    if (
        bool(getattr(module, "branch_relax_head_use_at_sampling", False))
        and getattr(module, "branch_relax_head", None) is not None
    ):
        td, n_leaves, mapping = _tree_to_model_split_lengths(module, current_newick)
        if not td:
            summary["stopped_reason"] = "no_active_splits"
            return current_newick, summary, states
        tokenized = module.model.tokenizer([current_newick])
        with torch.inference_mode():
            (
                _velocity,
                edge_splits,
                _edge_split_mask,
                _first_hit_logits,
                _boundary_vanish_logits,
                edge_features,
            ) = module.forward(
                tokenized,
                float(phase_value),
                phyla_embeddings,
                first_hit_case_indices=relax_case_indices,
                first_hit_start_topology_features=start_topology_features,
                first_hit_start_topology_embeddings=start_topology_embeddings,
                first_hit_start_topology_pad_mask=start_topology_pad_mask,
                first_hit_start_tree_graph_context=start_tree_graph_context,
            )
        if edge_features is None:
            summary["stopped_reason"] = "missing_edge_features"
            return current_newick, summary, states
        entries, lengths, n_leaves, mapping = _branch_relax_entries_for_tree(
            module,
            current_newick,
            edge_splits[0],
        )
        if not entries:
            summary["stopped_reason"] = "no_aligned_splits"
            return current_newick, summary, states
        features = torch.stack(
            [edge_features[0, entry["edge_index"]] for entry in entries],
            dim=0,
        )
        numeric = torch.tensor(
            [entry["numeric"] for entry in entries],
            dtype=torch.float32,
            device=module.device,
        )
        case_tensor = torch.full(
            (len(entries),),
            0 if case_index is None else int(case_index),
            dtype=torch.long,
            device=module.device,
        )
        with torch.inference_mode():
            deltas = (
                module.branch_relax_head(features, numeric, case_tensor)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        next_lengths_by_mask = {int(mask): float(length) for mask, length in lengths.items()}
        before_lengths = np.asarray([entry["length"] for entry in entries], dtype=np.float64)
        after_lengths = []
        for entry, delta in zip(entries, deltas):
            next_length = max(
                float(entry["length"]) + float(total_time) * float(delta),
                float(edge_floor),
            )
            next_lengths_by_mask[int(entry.get("source_mask", entry["mask"]))] = next_length
            after_lengths.append(next_length)
        td_next = {
            int(mask): float(length)
            for mask, length in next_lengths_by_mask.items()
            if float(length) > eps_len
        }
        current_newick = build_tree_from_splits(
            list(td_next.keys()),
            td_next,
            n_leaves,
            root_leaf=n_leaves - 1,
            mapping=mapping,
        )[1]
        after_arr = np.asarray(after_lengths, dtype=np.float64)
        summary["head"] = "branch_relax"
        summary["applied"] = True
        summary["applied_steps"] = 1
        summary["min_length_before"] = float(np.min(before_lengths))
        summary["min_length_after"] = float(np.min(after_arr))
        summary["max_abs_delta"] = float(np.max(np.abs(after_arr - before_lengths)))
        summary["final_tree"] = current_newick
        summary["rf_to_target_after"] = (
            None
            if target_tree is None
            else float(calculate_norm_rf(current_newick, target_tree))
        )
        summary["topology_rf_before_after"] = float(
            calculate_norm_rf(start_newick, current_newick)
        )
        summary["stopped_reason"] = "completed"
        states.append(
            {
                "step": 0,
                "local_time": 0.0,
                "timepoint": float(phase_value),
                "dt": float(total_time),
                "num_splits": int(len(entries)),
                "min_length": float(np.min(after_arr)),
                "max_abs_delta": float(summary["max_abs_delta"]),
                "rf_to_target": summary["rf_to_target_after"],
            }
        )
        return current_newick, summary, states

    step_dt = total_time / float(steps)

    for step_idx in range(steps):
        td, n_leaves, mapping = _tree_to_model_split_lengths(module, current_newick)
        if not td:
            summary["stopped_reason"] = "no_active_splits"
            break

        local_time = float(step_idx) * step_dt
        model_time = _final_orthant_relax_model_time(
            time_mode,
            phase_value,
            local_time,
        )
        tokenized = module.model.tokenizer([current_newick])
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
                float(model_time),
                phyla_embeddings,
                first_hit_case_indices=relax_case_indices,
                first_hit_start_topology_features=start_topology_features,
                first_hit_start_topology_embeddings=start_topology_embeddings,
                first_hit_start_topology_pad_mask=start_topology_pad_mask,
                first_hit_start_tree_graph_context=start_tree_graph_context,
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
        )

        lengths = aligned["lengths"].detach().cpu().numpy().astype(np.float64)
        velocities = aligned["velocities"].detach().cpu().numpy().astype(np.float64)
        masks = [int(x) for x in aligned["aligned_model_masks"]]
        if len(lengths) == 0:
            summary["stopped_reason"] = "no_aligned_splits"
            break
        if summary["min_length_before"] is None:
            summary["min_length_before"] = float(np.min(lengths))

        next_lengths = np.maximum(lengths + step_dt * velocities, edge_floor)
        max_abs_delta_step = float(np.max(np.abs(next_lengths - lengths)))
        summary["max_abs_delta"] = max(
            float(summary["max_abs_delta"]),
            max_abs_delta_step,
        )

        td_next = {
            int(mask): float(length)
            for mask, length in zip(masks, next_lengths)
            if float(length) > eps_len
        }
        current_newick = build_tree_from_splits(
            list(td_next.keys()),
            td_next,
            n_leaves,
            root_leaf=n_leaves - 1,
            mapping=mapping,
        )[1]
        summary["applied"] = True
        summary["applied_steps"] = int(step_idx + 1)
        summary["min_length_after"] = float(np.min(next_lengths))
        states.append(
            {
                "step": int(step_idx),
                "local_time": float(local_time),
                "timepoint": float(model_time),
                "dt": float(step_dt),
                "num_splits": int(len(next_lengths)),
                "min_length": float(np.min(next_lengths)),
                "max_abs_delta": max_abs_delta_step,
                "rf_to_target": None
                if target_tree is None
                else float(calculate_norm_rf(current_newick, target_tree)),
            }
        )

    summary["final_tree"] = current_newick
    summary["rf_to_target_after"] = (
        None
        if target_tree is None
        else float(calculate_norm_rf(current_newick, target_tree))
    )
    summary["topology_rf_before_after"] = float(
        calculate_norm_rf(start_newick, current_newick)
    )
    if summary["stopped_reason"] is None:
        summary["stopped_reason"] = "completed"
    return current_newick, summary, states


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


def _best_pairwise_merge_label_for_current_tree(
    module,
    current_newick,
    target_tree,
    new_split_length=1e-3,
):
    current_groups = [
        tuple(int(component) for component in group)
        for group in get_structural_polytomy_groups_from_newick(current_newick)
    ]
    if not current_groups:
        return None

    td, n_leaves, mapping = _tree_to_model_split_lengths(module, current_newick)
    best_candidate = None

    for group in current_groups:
        if len(group) < 2:
            continue
        if (
            int(getattr(module, "rollout_replay_pairwise_max_group_size", 0)) > 0
            and len(group) > int(module.rollout_replay_pairwise_max_group_size)
        ):
            continue
        parent_split = functools.reduce(operator.or_, group, 0)
        for merge_indices in itertools.combinations(range(len(group)), 2):
            merge_components = tuple(int(group[idx]) for idx in merge_indices)
            result_split = int(merge_components[0]) | int(merge_components[1])
            if result_split in td:
                continue

            candidate_td = dict(td)
            candidate_td[result_split] = float(new_split_length)
            try:
                _, candidate_newick = build_tree_from_splits(
                    list(candidate_td.keys()),
                    candidate_td,
                    n_leaves=n_leaves,
                    root_leaf=n_leaves - 1,
                    mapping=mapping,
                )
            except Exception:
                continue

            candidate_rf = float(calculate_norm_rf(candidate_newick, target_tree))
            candidate = {
                "rf": candidate_rf,
                "newick": current_newick,
                "target_tree": target_tree,
                "labels": [
                    {
                        "result_split": int(result_split),
                        "parent_split": int(parent_split),
                        "components": list(group),
                        "merge_indices": [int(idx) for idx in merge_indices],
                    }
                ],
                "stop_after_merge": False,
            }
            if best_candidate is None or candidate_rf < best_candidate["rf"]:
                best_candidate = candidate

    return best_candidate


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


def _predsim_overrun_trace_to_sampling_trace(out, target_tree):
    trace = {
        "velocity": [],
        "autoregressive": [],
        "stopped_for_no_valid_merge": False,
        "stopped_for_repeated_topology": False,
        "skipped_no_valid_boundary_revisits": 0.0,
        "stopped_for_prefix_replay_quota": False,
        "silent_boundary_recoveries": 0.0,
    }
    for item in out.get("trace", []):
        if item.get("phase") == "velocity":
            trace["velocity"].append(
                {
                    "newick_tree": item.get("newick_tree"),
                    "time": item.get("time"),
                    "timepoint": item.get("time"),
                    "target_tree": target_tree,
                    "rf": item.get("rf"),
                }
            )
        elif item.get("phase") == "autoregressive":
            trace["autoregressive"].append(
                {
                    # Replay AR samples should use the decision-state tree the model
                    # actually acted on, not the post-merge tree after applying the
                    # chosen split. The generic sampler trace already records AR
                    # states this way; keep predsim_overrun aligned with that.
                    "newick": item.get("source_newick", item.get("newick")),
                    "time": item.get("time"),
                    "target_tree": target_tree,
                    "rf": item.get("rf"),
                }
            )
    return trace


def _predsim_overrun_rollout(
    module,
    start_tree: str,
    target_tree: str,
    phyla_embeddings,
    case_index=None,
    start_topology_features=None,
    *,
    T: float = 1.0,
    eps_len: float = 1e-8,
    first_hit_tol: float = 1e-4,
    max_steps: int = 100,
    max_events: int = 1000,
    max_autoregressive_merges_per_boundary: int = -1,
    raw_no_fallback: bool = True,
    autoregressive_birth_length: float = 1e-3,
    boundary_mode: str = "pred_simultaneous",
    allow_time_overrun: bool = False,
    blocked_edge_floor: float | None = None,
):
    current_newick = start_tree
    case_index_tensor = (
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
    trace = []
    t = 0.0
    n_events = 0
    n_steps = 0

    while t < T and n_steps < int(max_steps) and n_events < int(max_events):
        n_steps += 1
        td, n_leaves, mapping = _tree_to_model_split_lengths(module, current_newick)
        tokenized = module.model.tokenizer([current_newick])
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
                float(t),
                phyla_embeddings,
                first_hit_start_topology_features=rollout_start_topology_features,
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
        )

        aligned_edge_features = aligned["edge_features"]
        aligned_lengths = aligned["lengths"]
        aligned_velocities = aligned["velocities"]
        if (
            getattr(module, "velocity_refiner_mode", "base")
            == "edge_token_attention_delta"
            and aligned_edge_features is not None
        ):
            refine_mask = aligned["supervised_mask"] & (aligned_lengths > eps_len)
            if bool(refine_mask.any().item()):
                refine_indices = torch.nonzero(refine_mask, as_tuple=False).reshape(-1)
                refined_velocities = module._refine_velocity_predictions(
                    aligned_velocities.index_select(0, refine_indices),
                    lengths=aligned_lengths.index_select(0, refine_indices),
                    edge_features=aligned_edge_features.index_select(0, refine_indices),
                    group_sizes=[int(refine_indices.numel())],
                )
                aligned_velocities = aligned_velocities.clone()
                aligned_velocities[refine_indices] = refined_velocities.to(
                    aligned_velocities.device, dtype=aligned_velocities.dtype
                )

        if aligned["first_hit_logits"] is not None or aligned_edge_features is not None:
            aligned_first_hit_logits = module._compute_first_hit_logits(
                aligned["first_hit_logits"],
                lengths=aligned_lengths,
                velocities=aligned_velocities,
                edge_features=aligned_edge_features,
                group_sizes=[int(aligned_lengths.numel())],
            )
        else:
            aligned_first_hit_logits = None

        lengths = aligned_lengths.detach().cpu().numpy().astype(np.float64)
        velocities = aligned_velocities.detach().cpu().numpy().astype(np.float64)
        supervised_mask = aligned["supervised_mask"].detach().cpu().numpy().astype(bool)
        masks = [int(x) for x in aligned["aligned_model_masks"]]
        first_logits = None
        if aligned_first_hit_logits is not None:
            first_logits = (
                aligned_first_hit_logits.detach().cpu().numpy().astype(np.float64)
            )

        candidate_mask = supervised_mask & (lengths > eps_len)
        if first_logits is not None:
            if raw_no_fallback:
                pred_mask, raw_count, used_fallback = _predict_first_hit_mask_with_fallback(
                    first_logits,
                    candidate_mask,
                    max_edges=-1,
                    fallback_threshold=-1,
                    fallback_top_k=-1,
                )
            else:
                pred_mask, raw_count, used_fallback = _predict_first_hit_mask_with_fallback(
                    first_logits,
                    candidate_mask,
                    max_edges=getattr(
                        module, "velocity_first_hit_sampling_max_edges", -1
                    ),
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
        else:
            pred_mask = np.zeros_like(candidate_mask, dtype=bool)
            raw_count = 0
            used_fallback = False

        pred_neg = pred_mask & (velocities < 0.0) & (lengths > eps_len)
        actual_moving_neg = supervised_mask & (velocities < 0.0) & (lengths > eps_len)
        actual_hit_mask = np.zeros_like(candidate_mask, dtype=bool)
        actual_dt_hit = float("inf")
        if np.any(actual_moving_neg):
            actual_dt_candidates = np.full_like(lengths, np.inf, dtype=np.float64)
            actual_dt_candidates[actual_moving_neg] = (
                lengths[actual_moving_neg]
                / np.maximum(-velocities[actual_moving_neg], eps_len)
            )
            actual_dt_hit = float(np.min(actual_dt_candidates[actual_moving_neg]))
            actual_hit_mask = actual_moving_neg & (
                np.abs(actual_dt_candidates - actual_dt_hit) <= float(first_hit_tol)
            )

        if boundary_mode == "pred_simultaneous":
            collapse_mask = pred_mask.copy()
            event_neg_mask = pred_neg.copy()
            if not np.any(event_neg_mask):
                trace.append(
                    {
                        "phase": "velocity",
                        "time": float(t),
                        "rf": float(calculate_norm_rf(current_newick, target_tree)),
                        "raw_pred_count": int(raw_count),
                        "used_fallback": bool(used_fallback),
                        "predicted_first_count": int(pred_mask.sum()),
                        "actual_hit_count": int(actual_hit_mask.sum()),
                        "event": "no_predicted_negative_edges",
                    }
                )
                break
            dt_target = float(
                np.max(
                    lengths[event_neg_mask]
                    / np.maximum(-velocities[event_neg_mask], eps_len)
                )
            )
        elif boundary_mode == "actual_hit_predset":
            collapse_mask = pred_mask.copy()
            event_neg_mask = actual_moving_neg.copy()
            if not np.any(actual_hit_mask):
                trace.append(
                    {
                        "phase": "velocity",
                        "time": float(t),
                        "rf": float(calculate_norm_rf(current_newick, target_tree)),
                        "raw_pred_count": int(raw_count),
                        "used_fallback": bool(used_fallback),
                        "predicted_first_count": int(pred_mask.sum()),
                        "actual_hit_count": int(actual_hit_mask.sum()),
                        "event": "no_actual_hit_edges",
                    }
                )
                break
            dt_target = float(actual_dt_hit)
        elif boundary_mode == "actual_hit_union":
            collapse_mask = pred_mask | actual_hit_mask
            event_neg_mask = actual_moving_neg.copy()
            if not np.any(actual_hit_mask):
                trace.append(
                    {
                        "phase": "velocity",
                        "time": float(t),
                        "rf": float(calculate_norm_rf(current_newick, target_tree)),
                        "raw_pred_count": int(raw_count),
                        "used_fallback": bool(used_fallback),
                        "predicted_first_count": int(pred_mask.sum()),
                        "actual_hit_count": int(actual_hit_mask.sum()),
                        "event": "no_actual_hit_edges",
                    }
                )
                break
            dt_target = float(actual_dt_hit)
        elif boundary_mode == "pred_simultaneous_time_head":
            collapse_mask = pred_mask.copy()
            event_neg_mask = pred_neg.copy()
            if not np.any(event_neg_mask):
                trace.append(
                    {
                        "phase": "velocity",
                        "time": float(t),
                        "rf": float(calculate_norm_rf(current_newick, target_tree)),
                        "raw_pred_count": int(raw_count),
                        "used_fallback": bool(used_fallback),
                        "predicted_first_count": int(pred_mask.sum()),
                        "actual_hit_count": int(actual_hit_mask.sum()),
                        "event": "no_predicted_negative_edges",
                    }
                )
                break
            dt_target = float(
                np.max(
                    lengths[event_neg_mask]
                    / np.maximum(-velocities[event_neg_mask], eps_len)
                )
            )
            if aligned_edge_features is not None and hasattr(
                module, "_predict_boundary_time_log"
            ):
                with torch.inference_mode():
                    pred_log_dt = module._predict_boundary_time_log(
                        lengths=aligned["lengths"],
                        velocities=aligned["velocities"],
                        edge_features=aligned_edge_features,
                        group_sizes=[int(aligned["lengths"].numel())],
                    )
                if pred_log_dt is not None and int(pred_log_dt.numel()) > 0:
                    dt_target = float(torch.exp(pred_log_dt[0]).detach().cpu().item())
        else:
            raise ValueError(f"Unknown boundary_mode={boundary_mode}")

        if not np.any(collapse_mask):
            trace.append(
                {
                    "phase": "velocity",
                    "time": float(t),
                    "rf": float(calculate_norm_rf(current_newick, target_tree)),
                    "raw_pred_count": int(raw_count),
                    "used_fallback": bool(used_fallback),
                    "predicted_first_count": int(pred_mask.sum()),
                    "actual_hit_count": int(actual_hit_mask.sum()),
                    "event": "empty_collapse_mask",
                }
            )
            break

        remaining = float("inf") if allow_time_overrun else max(float(T - t), 0.0)
        dt = min(dt_target, remaining)
        if dt <= 0.0:
            break

        L_new = lengths + dt * velocities
        reached_boundary = bool(dt + 1e-10 >= dt_target)
        clipped_mask = np.zeros_like(candidate_mask, dtype=bool)
        if reached_boundary:
            L_new[collapse_mask] = 0.0
            blocked = supervised_mask & (~collapse_mask)
            if np.any(blocked):
                clipped_mask = blocked & (L_new <= eps_len)
                floor = (
                    eps_len * 10.0
                    if blocked_edge_floor is None
                    else max(float(blocked_edge_floor), eps_len * 10.0)
                )
                L_new[blocked] = np.maximum(L_new[blocked], floor)

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
        t += dt
        trace.append(
            {
                "phase": "velocity",
                "time": float(t),
                "rf": float(calculate_norm_rf(current_newick, target_tree)),
                "newick_tree": current_newick,
                "raw_pred_count": int(raw_count),
                "used_fallback": bool(used_fallback),
                "predicted_first_count": int(pred_mask.sum()),
                "predicted_negative_count": int(pred_neg.sum()),
                "actual_hit_count": int(actual_hit_mask.sum()),
                "dt_target": float(dt_target),
                "dt_used": float(dt),
                "reached_boundary": bool(reached_boundary),
                "boundary_mode": boundary_mode,
                "predicted_masks": [
                    masks[i] for i, on in enumerate(pred_mask.tolist()) if on
                ],
                "actual_hit_masks": [
                    masks[i] for i, on in enumerate(actual_hit_mask.tolist()) if on
                ],
                "collapse_masks": [
                    masks[i] for i, on in enumerate(collapse_mask.tolist()) if on
                ],
                "num_clipped_extra_edges": int(clipped_mask.sum()),
                "clipped_extra_masks": [
                    masks[i] for i, on in enumerate(clipped_mask.tolist()) if on
                ],
            }
        )

        if not reached_boundary:
            continue

        merges_this_boundary = 0
        while (
            has_polytomy_fast(current_newick, unrooted_ok=False)
            and n_events < int(max_events)
            and (
                int(max_autoregressive_merges_per_boundary) < 0
                or merges_this_boundary < int(max_autoregressive_merges_per_boundary)
            )
        ):
            tokenized_trees = module.model.tokenizer([current_newick])
            component_groups = [
                get_structural_polytomy_groups_from_newick(current_newick)
            ]
            with torch.inference_mode():
                logit_outputs = module.forward(
                    tokenized_trees,
                    module._sampling_autoregressive_time_tensor(
                        t,
                        event_index=n_events,
                        max_events=max_events,
                    ),
                    phyla_embeddings,
                    autoregressive=True,
                    autoregressive_component_groups=component_groups,
                    autoregressive_case_indices=case_index_tensor,
                    autoregressive_start_topology_features=rollout_start_topology_features,
                )
            td_ar, n_ar, m_ar = _tree_to_model_split_lengths(module, current_newick)
            planned_merges = _plan_autoregressive_boundary_merges(
                logit_outputs,
                td_ar.keys(),
                top_only=bool(getattr(module, "sampling_use_top_merge_planner", False)),
            )
            if planned_merges:
                planned_merges = planned_merges[:1]
            if not planned_merges:
                break

            top_change = False
            for planned in planned_merges:
                for subset, new_split in planned["subsets"]:
                    source_newick = current_newick
                    td_ar[int(new_split)] = float(autoregressive_birth_length)
                    n_events += 1
                    top_change = True
                    current_newick = build_tree_from_splits(
                        list(td_ar.keys()),
                        td_ar,
                        n_ar,
                        root_leaf=n_ar - 1,
                        mapping=m_ar,
                    )[1]
                    trace.append(
                        {
                            "phase": "autoregressive",
                            "time": float(t),
                            "rf": float(calculate_norm_rf(current_newick, target_tree)),
                            "source_newick": source_newick,
                            "newick": current_newick,
                            "new_split": int(new_split),
                            "subset": [int(x) for x in subset],
                        }
                    )
            if not top_change:
                break
            merges_this_boundary += 1

    current_newick, final_tree_label_remapped = (
        align_numeric_leaf_labels_to_reference(
            current_newick,
            start_tree,
            target_tree=target_tree,
        )
    )
    best_rf = (
        min(float(item["rf"]) for item in trace)
        if trace
        else float(calculate_norm_rf(start_tree, target_tree))
    )
    return {
        "final_tree": current_newick,
        "final_rf": float(calculate_norm_rf(current_newick, target_tree)),
        "best_rf": float(best_rf),
        "num_trace_states": int(len(trace)),
        "num_velocity_states": int(sum(1 for x in trace if x["phase"] == "velocity")),
        "num_ar_states": int(sum(1 for x in trace if x["phase"] == "autoregressive")),
        "final_tree_label_remapped": bool(final_tree_label_remapped),
        "trace": trace,
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
        "terminal": [],
        "stopped_for_no_valid_merge": False,
        "stopped_for_repeated_topology": False,
        "skipped_no_valid_boundary_revisits": 0.0,
        "stopped_for_prefix_replay_quota": False,
        "silent_boundary_recoveries": 0.0,
        "stopped_for_terminal_head": False,
        "autoregressive_boundary_stop_count": 0.0,
        "final_orthant_relax": [],
        "final_orthant_relax_summary": None,
    }

    def _trace_rf_to_target(tree_newick):
        if not trace_state_rf:
            return None
        return float(calculate_norm_rf(tree_newick, target_tree))

    def _terminal_probability_for_state(state_newick, phase_value):
        if not bool(getattr(module, "velocity_terminal_head_use_at_sampling", False)):
            return None
        if module.velocity_terminal_head is None:
            return None
        tokenized_term = _tokenize_trees_with_structural_cache(module, [state_newick])
        _td_term, n_term, _ = _tree_to_model_split_lengths(
            module,
            state_newick,
            tokenized=tokenized_term,
        )
        with torch.inference_mode():
            (
                v_term,
                edge_splits_term,
                _edge_split_mask_term,
                first_hit_term,
                boundary_vanish_term,
                edge_features_term,
            ) = module.forward(
                tokenized_term,
                float(phase_value),
                phyla_embeddings,
                first_hit_case_indices=rollout_case_indices,
                first_hit_start_topology_features=rollout_start_topology_features,
                first_hit_start_topology_embeddings=rollout_start_topology_embeddings,
                first_hit_start_topology_pad_mask=rollout_start_topology_pad_mask,
                first_hit_start_tree_graph_context=rollout_start_tree_graph_context,
            )
        aligned_term = _align_model_outputs_to_tree_context(
            module,
            state_newick,
            n_term,
            edge_splits_term[0],
            v_term[0, :, 0],
            first_hit_logits_tree=None
            if first_hit_term is None
            else first_hit_term[0, :, 0],
            boundary_vanish_logits_tree=None
            if boundary_vanish_term is None
            else boundary_vanish_term[0, :, 0],
            edge_features_tree=None
            if edge_features_term is None
            else edge_features_term[0],
            eps_len=eps_len,
        )
        aligned_term_first_hit_logits = module._compute_first_hit_logits(
            aligned_term["first_hit_logits"],
            lengths=aligned_term["lengths"],
            velocities=aligned_term["velocities"],
            edge_features=aligned_term["edge_features"],
            group_sizes=[int(aligned_term["lengths"].numel())],
        )
        term_logit = module._predict_terminal_stop_logit(
            aligned_term["lengths"],
            aligned_term["velocities"],
            time_value=float(phase_value),
            first_hit_logits=aligned_term_first_hit_logits,
            boundary_vanish_logits=aligned_term["boundary_vanish_logits"],
            edge_features=aligned_term["edge_features"],
            aligned_model_masks=aligned_term["aligned_model_masks"],
            supervised_mask=aligned_term["supervised_mask"],
            case_index=case_index,
        )
        if term_logit is None:
            return None
        return float(torch.sigmoid(term_logit).detach().cpu().item())

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
        terminal_prob = _terminal_probability_for_state(current_newick, phase)
        terminal_requested = (
            terminal_prob is not None and float(terminal_prob) > 0.5
        )
        terminal_sampling_action = str(
            getattr(module, "velocity_terminal_head_sampling_action", "after_phase")
        ).lower()
        stop_immediately_for_terminal = (
            terminal_requested and terminal_sampling_action == "immediate"
        )
        stop_after_current_phase = (
            terminal_requested and not stop_immediately_for_terminal
        )
        trace["terminal"].append(
            {
                "newick": current_newick,
                "target_tree": target_tree,
                "timepoint": float(phase),
                "phase_idx": int(phase),
                "rf_to_target": _trace_rf_to_target(current_newick),
                "pred_terminal_prob": terminal_prob,
                "position": "phase_start",
                "stop_after_phase": bool(stop_after_current_phase),
                "stop_immediately": bool(stop_immediately_for_terminal),
                "sampling_action": terminal_sampling_action,
            }
        )
        if stop_immediately_for_terminal:
            trace["stopped_for_terminal_head"] = True
            break
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
        rollout_terminal_stop = False
        birthset_attempted_this_boundary = False
        topology_decoder_mode = getattr(module, "topology_decoder", "ar")
        birthset_allows_ar_fallback = (
            topology_decoder_mode == "birthset_with_ar_fallback"
            and getattr(module, "birthset_fallback", "ar") == "ar"
        )
        birthset_polytomy_unrooted_ok = topology_decoder_mode in {
            "birthset",
            "birthset_with_ar_fallback",
        }
        while (
            has_polytomy_fast(
                current_newick,
                unrooted_ok=birthset_polytomy_unrooted_ok,
            )
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
            use_birthset_decoder = (
                topology_decoder_mode in {"birthset", "birthset_with_ar_fallback"}
                and not birthset_attempted_this_boundary
            )
            if use_birthset_decoder:
                birthset_attempted_this_boundary = True
                birthset_plan = module._plan_birthset_boundary_splits(
                    logit_outputs,
                    td_ar.keys(),
                    n_ar,
                )
                selected_births = list(birthset_plan.get("selected", []))
                remaining_events = max(int(effective_max_events) - int(n_events), 0)
                if remaining_events > 0:
                    selected_births = selected_births[:remaining_events]
                else:
                    selected_births = []
                if selected_births:
                    source_newick = current_newick
                    selected_splits = []
                    for item in selected_births:
                        new_split = int(item["split_mask"])
                        td_ar[new_split] = float(
                            getattr(
                                module,
                                "birthset_birth_length",
                                autoregressive_birth_length,
                            )
                        )
                        selected_splits.append(new_split)
                    n_events += len(selected_splits)
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
                            "decoder_mode": "birthset",
                            "planned_merge_count": int(len(selected_splits)),
                            "selected_result_splits": selected_splits,
                            "birthset_metrics": birthset_plan.get("metrics", {}),
                        }
                    )
                    if (
                        has_polytomy_fast(
                            current_newick,
                            unrooted_ok=birthset_polytomy_unrooted_ok,
                        )
                        and not birthset_allows_ar_fallback
                    ):
                        trace["birthset_incomplete_without_fallback"] = True
                        trace["stopped_for_no_valid_merge"] = True
                        phase_exhausted = True
                        break
                    continue
                if not birthset_allows_ar_fallback:
                    trace["autoregressive"].append(
                        {
                            "newick": current_newick,
                            "target_tree": target_tree,
                            "time": float(phase),
                            "phase_idx": int(phase),
                            "rf_to_target": _trace_rf_to_target(current_newick),
                            "decoder_mode": "birthset",
                            "planned_merge_count": 0,
                            "selected_result_split": None,
                            "birthset_metrics": birthset_plan.get("metrics", {}),
                        }
                    )
                    trace["birthset_incomplete_without_fallback"] = True
                    trace["stopped_for_no_valid_merge"] = True
                    phase_exhausted = True
                    break
            if (
                topology_decoder_mode in {"birthset", "birthset_with_ar_fallback"}
                and birthset_attempted_this_boundary
                and not birthset_allows_ar_fallback
            ):
                trace["birthset_incomplete_without_fallback"] = True
                trace["stopped_for_no_valid_merge"] = True
                phase_exhausted = True
                break
            planned_merges = _plan_autoregressive_boundary_merges(
                logit_outputs,
                td_ar.keys(),
                top_only=bool(getattr(module, "sampling_use_top_merge_planner", False)),
            )
            if planned_merges:
                planned_merges = planned_merges[:1]
            if not planned_merges:
                trace["autoregressive"].append(
                    {
                        "newick": current_newick,
                        "target_tree": target_tree,
                        "time": float(phase),
                        "phase_idx": int(phase),
                        "rf_to_target": _trace_rf_to_target(current_newick),
                        "decoder_mode": "ar",
                        "planned_merge_count": 0,
                        "selected_result_split": None,
                    }
                )
                trace["stopped_for_no_valid_merge"] = True
                phase_exhausted = True
                break

            planned = planned_merges[0]
            _, new_split = planned["subsets"][0]
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
                            -max(min(float(stop_after_merge_logit_value), 60.0), -60.0)
                        )
                    )
                )
            )
            stop_after_merge_requested = (
                bool(
                    getattr(
                        module,
                        "autoregressive_stop_after_merge_use_at_sampling",
                        False,
                    )
                )
                and planned.get("decoder_mode") == "structured_subset"
                and stop_after_merge_logit_value is not None
                and stop_after_merge_logit_value > 0.0
            )
            source_newick = current_newick
            td_ar[int(new_split)] = float(autoregressive_birth_length)
            n_events += 1
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
                    "decoder_mode": "ar",
                    "planned_merge_count": int(len(planned_merges)),
                    "selected_result_split": int(new_split),
                    "stop_after_merge_logit": stop_after_merge_logit_value,
                    "stop_after_merge_prob": stop_after_merge_prob,
                    "stop_after_merge_requested": bool(stop_after_merge_requested),
                }
            )
            post_ar_terminal_prob = _terminal_probability_for_state(
                current_newick,
                phase,
            )
            post_ar_terminal_requested = (
                post_ar_terminal_prob is not None
                and float(post_ar_terminal_prob) > 0.5
            )
            trace["terminal"].append(
                {
                    "newick": current_newick,
                    "target_tree": target_tree,
                    "timepoint": float(phase),
                    "phase_idx": int(phase),
                    "rf_to_target": _trace_rf_to_target(current_newick),
                    "pred_terminal_prob": post_ar_terminal_prob,
                    "position": "post_ar_merge",
                    "selected_result_split": int(new_split),
                    "stop_after_phase": False,
                    "stop_immediately": bool(post_ar_terminal_requested),
                    "sampling_action": terminal_sampling_action,
                }
            )
            if post_ar_terminal_requested:
                trace["stopped_for_terminal_head"] = True
                rollout_terminal_stop = True
                ar_boundary_complete = True
                break
            if stop_after_merge_requested:
                trace["autoregressive_boundary_stop_count"] += 1.0
                ar_boundary_complete = True
                break

        if rollout_terminal_stop:
            break
        if stop_after_current_phase:
            trace["stopped_for_terminal_head"] = True
            break
        if (
            ar_boundary_complete
            or phase_exhausted
            or not has_polytomy_fast(
                current_newick,
                unrooted_ok=birthset_polytomy_unrooted_ok,
            )
        ):
            phase += 1
            continue
        break

    if bool(getattr(module, "sampling_final_orthant_relax_use_at_sampling", False)):
        current_newick, relax_summary, relax_states = (
            _relax_final_orthant_branch_lengths(
                module,
                current_newick,
                target_tree,
                phyla_embeddings,
                case_index=case_index,
                start_topology_features=rollout_start_topology_features,
                start_topology_embeddings=rollout_start_topology_embeddings,
                start_topology_pad_mask=rollout_start_topology_pad_mask,
                start_tree_graph_context=rollout_start_tree_graph_context,
                phase_value=float(phase),
                eps_len=eps_len,
            )
        )
        trace["final_orthant_relax"] = relax_states
        trace["final_orthant_relax_summary"] = relax_summary

    current_newick, final_tree_label_remapped = (
        align_numeric_leaf_labels_to_reference(
            current_newick,
            start_tree,
            target_tree=target_tree,
        )
    )
    out = {
        "final_tree": current_newick,
        "final_rf": float(calculate_norm_rf(current_newick, target_tree)),
        "num_velocity_states": int(len(trace["velocity"])),
        "num_ar_states": int(len(trace["autoregressive"])),
        "final_tree_label_remapped": bool(final_tree_label_remapped),
        "trace": trace,
    }
    if return_trace:
        return out
    return out


def _slice_rollout_batch_value(value, indices, total_size, device=None):
    if value is None:
        return None
    indices = [int(idx) for idx in indices]
    if torch.is_tensor(value):
        if value.ndim >= 1:
            if int(value.shape[0]) == int(total_size):
                index_tensor = torch.as_tensor(
                    indices,
                    dtype=torch.long,
                    device=value.device,
                )
                return value.index_select(0, index_tensor)
            if int(value.shape[0]) == 1 and len(indices) != 1:
                return value.expand(len(indices), *value.shape[1:]).contiguous()
        return value.to(device) if device is not None else value
    if isinstance(value, (list, tuple)) and len(value) == int(total_size):
        return [value[idx] for idx in indices]
    return value


def _tree_to_model_split_lengths_from_batched_tokenized(tokenized, batch_index, newick):
    tree_obj = Tree(newick)
    encoder = BHVEncoder()
    split_masks, split_lengths = encoder.return_BHV_encoding(tree_obj)
    length_map = {
        int(mask): float(length)
        for mask, length in zip(split_masks, split_lengths)
        if length is not None and float(length) > 1e-8
    }
    model_masks = [int(mask) for mask in tokenized[-1][batch_index] if int(mask) != 0]
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


def _discrete_phase_rollout_batched_birthset(
    module,
    start_trees,
    target_trees,
    phyla_embeddings,
    case_indices=None,
    start_topology_features=None,
    start_topology_embeddings=None,
    start_topology_pad_mask=None,
    *,
    dt_base: float = 0.02,
    eps_len: float = 1e-8,
    max_events: int = 1000,
    max_steps: int = 1000,
    max_phases: int = 8,
    return_trace: bool = False,
    trace_state_rf: bool = True,
    explicit_autoregressive_component_groups: bool = True,
):
    start_trees = [str(tree) for tree in start_trees]
    target_trees = [str(tree) if tree is not None else None for tree in target_trees]
    batch_size = len(start_trees)
    current_newicks = list(start_trees)
    phase_by_tree = [0 for _ in range(batch_size)]
    n_events_by_tree = [0 for _ in range(batch_size)]
    n_steps_by_tree = [0 for _ in range(batch_size)]
    done = [False for _ in range(batch_size)]

    rollout_case_indices = None
    if case_indices is not None:
        rollout_case_indices = torch.as_tensor(
            case_indices,
            dtype=torch.long,
            device=module.device,
        ).reshape(-1)
        if int(rollout_case_indices.shape[0]) == 1 and batch_size > 1:
            rollout_case_indices = rollout_case_indices.expand(batch_size).contiguous()
        if int(rollout_case_indices.shape[0]) != batch_size:
            raise ValueError(
                "case_indices must have one entry per starting tree for batched rollout."
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
            start_trees,
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
            start_trees,
            device=module.device,
        )

    rollout_start_tree_graph_context = None
    if getattr(module.model, "first_hit_head_mode", "base") == "start_tree_graph_token_mlp":
        rollout_start_tree_graph_context = _build_start_tree_graph_context(
            module,
            start_trees,
            phyla_embeddings,
            device=module.device,
            detach=getattr(module.model, "first_hit_start_tree_graph_detach", False),
        )

    trace = {
        "velocity": [],
        "autoregressive": [],
        "terminal": [],
        "stopped_for_no_valid_merge": False,
        "stopped_for_repeated_topology": False,
        "skipped_no_valid_boundary_revisits": 0.0,
        "stopped_for_prefix_replay_quota": False,
        "silent_boundary_recoveries": 0.0,
        "stopped_for_terminal_head": False,
        "autoregressive_boundary_stop_count": 0.0,
        "final_orthant_relax": [],
        "final_orthant_relax_summary": None,
    }

    def _trace_rf_to_target(tree_index, tree_newick):
        if not trace_state_rf or target_trees[tree_index] is None:
            return None
        return float(calculate_norm_rf(tree_newick, target_trees[tree_index]))

    effective_max_events = (
        1000000 if max_events is None or int(max_events) < 0 else int(max_events)
    )
    effective_max_steps = (
        1000000 if max_steps is None or int(max_steps) < 0 else int(max_steps)
    )
    effective_max_phases = max(1, int(max_phases))

    while True:
        active = [
            idx
            for idx in range(batch_size)
            if (
                not done[idx]
                and phase_by_tree[idx] < effective_max_phases
                and n_events_by_tree[idx] < effective_max_events
                and n_steps_by_tree[idx] < effective_max_steps
            )
        ]
        if not active:
            break

        for idx in active:
            n_steps_by_tree[idx] += 1
            trace["terminal"].append(
                {
                    "newick": current_newicks[idx],
                    "target_tree": target_trees[idx],
                    "timepoint": float(phase_by_tree[idx]),
                    "phase_idx": int(phase_by_tree[idx]),
                    "rf_to_target": _trace_rf_to_target(idx, current_newicks[idx]),
                    "pred_terminal_prob": None,
                    "position": "phase_start",
                    "stop_after_phase": False,
                    "stop_immediately": False,
                    "sampling_action": "after_phase",
                }
            )

        active_newicks = [current_newicks[idx] for idx in active]
        tokenized = _tokenize_trees_with_structural_cache(module, active_newicks)
        phase_tensor = torch.tensor(
            [float(phase_by_tree[idx]) for idx in active],
            dtype=torch.float32,
            device=module.device,
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
                phase_tensor,
                _slice_rollout_batch_value(
                    phyla_embeddings,
                    active,
                    batch_size,
                    device=module.device,
                ),
                first_hit_case_indices=_slice_rollout_batch_value(
                    rollout_case_indices,
                    active,
                    batch_size,
                    device=module.device,
                ),
                first_hit_start_topology_features=_slice_rollout_batch_value(
                    rollout_start_topology_features,
                    active,
                    batch_size,
                    device=module.device,
                ),
                first_hit_start_topology_embeddings=_slice_rollout_batch_value(
                    rollout_start_topology_embeddings,
                    active,
                    batch_size,
                    device=module.device,
                ),
                first_hit_start_topology_pad_mask=_slice_rollout_batch_value(
                    rollout_start_topology_pad_mask,
                    active,
                    batch_size,
                    device=module.device,
                ),
                first_hit_start_tree_graph_context=_slice_rollout_batch_value(
                    rollout_start_tree_graph_context,
                    active,
                    batch_size,
                    device=module.device,
                ),
            )

        boundary_indices = []
        boundary_newicks = []
        for local_idx, tree_idx in enumerate(active):
            td, n_leaves, mapping = _tree_to_model_split_lengths_from_batched_tokenized(
                tokenized,
                local_idx,
                current_newicks[tree_idx],
            )
            aligned = _align_model_outputs_to_tree_context(
                module,
                current_newicks[tree_idx],
                n_leaves,
                edge_splits[local_idx],
                velocity[local_idx, :, 0],
                first_hit_logits_tree=None
                if first_hit_logits is None
                else first_hit_logits[local_idx, :, 0],
                boundary_vanish_logits_tree=None
                if boundary_vanish_logits is None
                else boundary_vanish_logits[local_idx, :, 0],
                edge_features_tree=None
                if edge_features is None
                else edge_features[local_idx],
                eps_len=eps_len,
            )
            aligned_first_hit_logits = module._compute_first_hit_logits(
                aligned["first_hit_logits"],
                lengths=aligned["lengths"],
                velocities=aligned["velocities"],
                edge_features=aligned["edge_features"],
                group_sizes=[int(aligned["lengths"].numel())],
            )
            lengths = aligned["lengths"].detach().cpu().numpy().astype(np.float64)
            velocities = aligned["velocities"].detach().cpu().numpy().astype(
                np.float64
            )
            supervised_mask = (
                aligned["supervised_mask"].detach().cpu().numpy().astype(bool)
            )
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
                    max_edges=getattr(
                        module,
                        "velocity_first_hit_sampling_max_edges",
                        -1,
                    ),
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
                        "newick_tree": current_newicks[tree_idx],
                        "target_tree": target_trees[tree_idx],
                        "timepoint": float(phase_by_tree[tree_idx]),
                        "phase_idx": int(phase_by_tree[tree_idx]),
                        "num_leaves": int(n_leaves),
                        "event": "no_predicted_negative_edges",
                    }
                )
                done[tree_idx] = True
                continue

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
            lengths_new = lengths + dt * velocities
            collapse_mask = predicted_first_mask.copy()
            if np.any(collapse_mask):
                lengths_new[collapse_mask] = 0.0
            blocked = supervised_mask & (~collapse_mask)
            if np.any(blocked):
                lengths_new[blocked] = np.maximum(
                    lengths_new[blocked],
                    eps_len * 10.0,
                )

            td2 = {
                int(mask): float(length)
                for mask, length in zip(masks, lengths_new)
                if float(length) > eps_len
            }
            current_newicks[tree_idx] = build_tree_from_splits(
                list(td2.keys()),
                td2,
                n_leaves,
                root_leaf=n_leaves - 1,
                mapping=mapping,
            )[1]
            trace["velocity"].append(
                {
                    "newick_tree": current_newicks[tree_idx],
                    "target_tree": target_trees[tree_idx],
                    "timepoint": float(phase_by_tree[tree_idx]),
                    "phase_idx": int(phase_by_tree[tree_idx]),
                    "num_leaves": int(n_leaves),
                    "rf_to_target": _trace_rf_to_target(
                        tree_idx,
                        current_newicks[tree_idx],
                    ),
                    "predicted_masks": [
                        masks[i]
                        for i, on in enumerate(predicted_first_mask.tolist())
                        if on
                    ],
                    "dt_target": float(dt_target),
                }
            )
            if (
                has_polytomy_fast(current_newicks[tree_idx], unrooted_ok=True)
                and n_events_by_tree[tree_idx] < effective_max_events
            ):
                boundary_indices.append(tree_idx)
                boundary_newicks.append(current_newicks[tree_idx])

        if boundary_indices:
            unique_boundary_map = {}
            unique_boundary_indices = []
            unique_boundary_newicks = []
            boundary_unique_ids = []
            for tree_idx, boundary_newick in zip(boundary_indices, boundary_newicks):
                case_value = None
                if rollout_case_indices is not None:
                    case_value = int(rollout_case_indices[int(tree_idx)].item())
                key = (
                    str(boundary_newick),
                    int(phase_by_tree[int(tree_idx)]),
                    case_value,
                    str(start_trees[int(tree_idx)]),
                )
                unique_id = unique_boundary_map.get(key)
                if unique_id is None:
                    unique_id = len(unique_boundary_indices)
                    unique_boundary_map[key] = unique_id
                    unique_boundary_indices.append(int(tree_idx))
                    unique_boundary_newicks.append(str(boundary_newick))
                boundary_unique_ids.append(int(unique_id))

            tokenized_boundary = _tokenize_trees_with_structural_cache(
                module,
                unique_boundary_newicks,
            )
            component_groups = None
            if explicit_autoregressive_component_groups:
                component_groups = [
                    get_structural_polytomy_groups_from_newick(newick)
                    for newick in unique_boundary_newicks
                ]
            boundary_phase_tensor = torch.tensor(
                [float(phase_by_tree[idx]) for idx in unique_boundary_indices],
                dtype=torch.float32,
                device=module.device,
            )
            with torch.inference_mode():
                logit_outputs_batch = module.forward(
                    tokenized_boundary,
                    boundary_phase_tensor,
                    _slice_rollout_batch_value(
                        phyla_embeddings,
                        unique_boundary_indices,
                        batch_size,
                        device=module.device,
                    ),
                    autoregressive=True,
                    autoregressive_component_groups=component_groups,
                    autoregressive_case_indices=_slice_rollout_batch_value(
                        rollout_case_indices,
                        unique_boundary_indices,
                        batch_size,
                        device=module.device,
                    ),
                    autoregressive_start_topology_features=_slice_rollout_batch_value(
                        rollout_start_topology_features,
                        unique_boundary_indices,
                        batch_size,
                        device=module.device,
                    ),
                )

            if not isinstance(logit_outputs_batch, (list, tuple)):
                logit_outputs_batch = [logit_outputs_batch]
            logit_outputs_by_local_tree = [
                [] for _ in range(len(unique_boundary_indices))
            ]
            for output in logit_outputs_batch:
                if not isinstance(output, dict):
                    continue
                output_batch_index = int(output.get("batch_index", 0))
                if 0 <= output_batch_index < len(logit_outputs_by_local_tree):
                    logit_outputs_by_local_tree[output_batch_index].append(output)

            plans_by_unique = []
            for unique_id, _representative_tree_idx in enumerate(unique_boundary_indices):
                source_newick = unique_boundary_newicks[unique_id]
                tokenized_single = tuple(
                    item[unique_id : unique_id + 1]
                    if (
                        torch.is_tensor(item)
                        and item.ndim >= 1
                        and int(item.shape[0]) == len(unique_boundary_indices)
                    )
                    else (
                        item[unique_id : unique_id + 1]
                        if isinstance(item, (list, tuple))
                        and len(item) == len(unique_boundary_indices)
                        else item
                    )
                    for item in tokenized_boundary
                )
                td_ar, n_ar, mapping_ar = _tree_to_model_split_lengths(
                    module,
                    source_newick,
                    tokenized=tokenized_single,
                )
                birthset_plan = module._plan_birthset_boundary_splits(
                    logit_outputs_by_local_tree[unique_id],
                    td_ar.keys(),
                    n_ar,
                )
                plans_by_unique.append(
                    {
                        "source_newick": source_newick,
                        "td_ar": td_ar,
                        "n_ar": n_ar,
                        "mapping_ar": mapping_ar,
                        "birthset_plan": birthset_plan,
                        "selected_births": list(birthset_plan.get("selected", [])),
                    }
                )

            for local_idx, tree_idx in enumerate(boundary_indices):
                plan_record = plans_by_unique[boundary_unique_ids[local_idx]]
                source_newick = plan_record["source_newick"]
                td_ar = dict(plan_record["td_ar"])
                n_ar = int(plan_record["n_ar"])
                mapping_ar = plan_record["mapping_ar"]
                birthset_plan = plan_record["birthset_plan"]
                selected_births = list(plan_record["selected_births"])
                remaining_events = max(
                    int(effective_max_events) - int(n_events_by_tree[tree_idx]),
                    0,
                )
                selected_births = selected_births[:remaining_events]
                selected_splits = []
                for item in selected_births:
                    split = int(item["split_mask"])
                    td_ar[split] = float(getattr(module, "birthset_birth_length", 1e-3))
                    selected_splits.append(split)
                n_events_by_tree[tree_idx] += len(selected_splits)
                if selected_splits:
                    current_newicks[tree_idx] = build_tree_from_splits(
                        list(td_ar.keys()),
                        td_ar,
                        n_ar,
                        root_leaf=n_ar - 1,
                        mapping=mapping_ar,
                    )[1]
                unresolved = has_polytomy_fast(
                    current_newicks[tree_idx],
                    unrooted_ok=True,
                )
                trace["autoregressive"].append(
                    {
                        "source_newick": source_newick,
                        "newick": current_newicks[tree_idx],
                        "target_tree": target_trees[tree_idx],
                        "time": float(phase_by_tree[tree_idx]),
                        "phase_idx": int(phase_by_tree[tree_idx]),
                        "rf_to_target": _trace_rf_to_target(
                            tree_idx,
                            current_newicks[tree_idx],
                        ),
                        "decoder_mode": "birthset",
                        "planned_merge_count": int(len(selected_splits)),
                        "selected_result_splits": selected_splits,
                        "birthset_metrics": birthset_plan.get("metrics", {}),
                    }
                )
                if unresolved:
                    trace["birthset_incomplete_without_fallback"] = True
                    trace["stopped_for_no_valid_merge"] = True

        for idx in active:
            if not done[idx]:
                phase_by_tree[idx] += 1

    final_trees = []
    remapped_flags = []
    for idx, tree_newick in enumerate(current_newicks):
        aligned, remapped = align_numeric_leaf_labels_to_reference(
            tree_newick,
            start_trees[idx],
            target_tree=target_trees[idx],
        )
        final_trees.append(aligned)
        remapped_flags.append(bool(remapped))

    out = {
        "final_trees": final_trees,
        "final_rf": [
            float(calculate_norm_rf(tree, target_trees[idx]))
            if target_trees[idx] is not None
            else None
            for idx, tree in enumerate(final_trees)
        ],
        "num_velocity_states": int(len(trace["velocity"])),
        "num_ar_states": int(len(trace["autoregressive"])),
        "final_tree_label_remapped": remapped_flags,
        "trace": trace,
    }
    if return_trace:
        return out
    return out


class TrainingModule(LightningModule):
    def __init__(
        self,
        model: TreeDenoiserTokenGT,
        dataset: PhylaDataModule,
        lr: float = 1e-4,
        optimizer_name: str = "adamw",
        record=False,
        epochs: int = 5000,
        lr_scheduler: str = "default",
        num_annealing_steps: int = 10000,
        num_warmup_steps: int = 1000,
        deepspeed: bool = False,
        logger=None,
        max_num_timesteps: int = 20,
        training_sampling_frequency: int = 200,
        training_sampling_start: int = 500,
        training_sampling_mode: str = "batch_compare",
        training_sampling_dt_base: float = 0.02,
        sampling_fixed_dt_base: float | None = None,
        sampling_max_steps: int | None = 256,
        sampling_max_events: int | None = None,
        sampling_max_autoregressive_merges_per_boundary: int = -1,
        training_sampling_stop_on_zero_rf: bool = False,
        training_sampling_stop_rf_threshold: float | None = None,
        num_samples: int = 10,
        dt: float = 0.1,
        # Figure out how to do typing here
        global_splits=None,
        random_trees=None,
        verbose: bool = False,
        phyla_checkpoint_path=None,
        phyla_precomputed_embeddings_path: str | None = None,
        live_phyla_checkpoint_path: str | None = None,
        live_phyla_unfreeze: bool = True,
        live_phyla_lr: float | None = None,
        live_phyla_input_mode: str = "raw-full",
        live_phyla_max_input_tokens: int = 0,
        live_phyla_device: str | None = None,
        velocity_loss_mode: str = "weighted",
        velocity_loss_plain_weight: float = 0.5,
        velocity_sign_eps: float = 1e-3,
        training_step_velocity_weight: float = 1.0,
        training_step_autoregressive_weight: float = 1.0,
        training_step_gradient_clip_val: float = 1.0,
        grad_norm_log_frequency: int = 1,
        training_step_profile_frequency: int = 0,
        training_step_profile_warmup_steps: int = 0,
        training_step_profile_sync_cuda: bool = True,
        training_step_autoregressive_grad_ratio = None,
        training_step_separate_optimizer_steps: bool = False,
        training_step_verbose_logging_enabled: bool = False,
        autoregressive_use_time: bool = False,
        autoregressive_target_mode: str = "scheduled",
        autoregressive_polytomy_choosing_weight: float = 1.0,
        autoregressive_stop_after_merge_weight: float = 0.0,
        autoregressive_stop_after_merge_use_at_sampling: bool = False,
        autoregressive_rollin_prob: float = 0.0,
        autoregressive_dagger_prob: float = 0.0,
        autoregressive_dagger_max_steps: int = 4,
        autoregressive_structure_perturb_prob: float = 0.0,
        autoregressive_structure_perturb_mode: str = "random_wrong_pair",
        topology_decoder: str = "ar",
        birthset_birth_length: float = 1e-3,
        birthset_lambda_birth: float = 0.2,
        birthset_lambda_rank: float = 0.1,
        birthset_lambda_proposal: float = 0.0,
        birthset_rank_margin: float = 1.0,
        birthset_pos_weight="auto",
        birthset_use_train_birth_split_bank: bool = True,
        birthset_use_small_polytomy_enumeration: bool = True,
        birthset_use_pair_prefix_candidates: bool = False,
        birthset_use_component_phyla_conditioning: bool = False,
        birthset_pair_prefix_top_pairs: int = 64,
        birthset_proposal_pair_target_mode: str = "contained",
        birthset_proposal_max_expansion_examples: int = 4096,
        birthset_proposal_max_order_seed_pairs: int = 128,
        birthset_proposal_train_topk: bool = False,
        birthset_max_enum_components: int = 12,
        birthset_max_candidates_per_polytomy: int = 2048,
        birthset_negatives_per_positive: int = 64,
        birthset_decoder: str = "greedy",
        birthset_beam_width: int = 8,
        birthset_fallback: str = "ar",
        velocity_length_jitter_prob: float = 0.0,
        velocity_length_jitter_scale: float = 0.0,
        velocity_dt_candidate_weight: float = 0.0,
        velocity_dt_hit_weight: float = 0.0,
        velocity_logtau_all_weight: float = 0.0,
        velocity_logtau_first_over_weight: float = 0.0,
        velocity_logtau_first_tie_weight: float = 0.0,
        velocity_logtau_predset_over_weight: float = 0.0,
        velocity_dt_eps: float = 1e-6,
        velocity_event_weight: float = 0.5,
        velocity_event_temp: float = 0.5,
        velocity_event_rate_beta: float = 5.0,
        velocity_event_normalize_by_log_candidates: bool = True,
        velocity_event_precision_weight: float = 0.0,
        velocity_event_precision_margin: float = 0.0,
        velocity_first_hit_head_weight: float = 0.0,
        velocity_first_hit_loss_tol: float = 0.01,
        velocity_first_hit_false_positive_mass_weight: float = 0.0,
        velocity_first_hit_false_negative_mass_weight: float = 0.0,
        velocity_first_hit_head_use_at_sampling: bool = False,
        velocity_first_hit_predictor_mode: str = "base",
        velocity_first_hit_use_geometry_features: bool = False,
        velocity_first_hit_geometry_hidden_dim: int = 32,
        velocity_first_hit_edge_length_hidden_dim: int = 64,
        velocity_first_hit_attention_layers: int = 1,
        velocity_first_hit_attention_heads: int = 4,
        velocity_first_hit_bucket_count: int = 32,
        velocity_first_hit_bucket_log_min: float = -8.0,
        velocity_first_hit_bucket_log_max: float = 1.0,
        velocity_refiner_mode: str = "base",
        velocity_refiner_attention_layers: int = 1,
        velocity_refiner_attention_heads: int = 4,
        velocity_refiner_bucket_count: int = 32,
        velocity_refiner_bucket_log_min: float = -8.0,
        velocity_refiner_bucket_log_max: float = 1.0,
        velocity_boundary_vanish_head_weight: float = 0.0,
        velocity_boundary_vanish_head_use_at_sampling: bool = False,
        velocity_boundary_vanish_one_step_use_at_sampling: bool = False,
        velocity_boundary_time_head_weight: float = 0.0,
        velocity_boundary_time_head_use_at_sampling: bool = False,
        velocity_boundary_time_hidden_dim: int = 64,
        velocity_terminal_head_weight: float = 0.0,
        velocity_terminal_head_use_at_sampling: bool = False,
        velocity_terminal_head_sampling_action: str = "after_phase",
        velocity_terminal_head_hidden_dim: int = 64,
        velocity_terminal_head_probe_features: bool = False,
        velocity_terminal_head_input_mode: str | None = None,
        velocity_terminal_head_use_case_adapt: bool = False,
        velocity_terminal_head_balance_loss: bool = False,
        velocity_terminal_head_topology_pool: str = "mean",
        velocity_probe_direct_set_loss: bool = False,
        velocity_probe_direct_set_anchor_only: bool = False,
        velocity_probe_direct_set_target_negative_weight: float = 1.0,
        velocity_probe_direct_set_nontarget_nonnegative_weight: float = 0.0,
        velocity_probe_direct_set_positive_reweight: bool = False,
        velocity_probe_direct_set_include_base_samples: bool = False,
        velocity_probe_direct_set_positive_reweight_power: float = 1.0,
        velocity_probe_direct_set_positive_reweight_max: float | None = None,
        velocity_probe_direct_set_bce_weight: float = 1.0,
        velocity_probe_direct_set_loss_weight: float = 1.0,
        velocity_probe_direct_set_mse_weight: float = 0.0,
        training_step_probe_parity_joint_update: bool = False,
        training_step_joint_tokenize_velocity_ar: bool = False,
        training_step_full_path_replay_initial_retry_attempt: int = 0,
        skip_repeated_no_valid_boundary_use_at_sampling: bool = False,
        sampling_discrete_phase_rollout_use_at_sampling: bool = False,
        sampling_discrete_phase_exact_boundary_step_use_at_sampling: bool = False,
        sampling_discrete_phase_max_phases: int = 8,
        sampling_final_orthant_relax_use_at_sampling: bool = False,
        sampling_final_orthant_relax_steps: int = 0,
        sampling_final_orthant_relax_total_time: float = 1.0,
        sampling_final_orthant_relax_time_mode: str = "local",
        sampling_final_orthant_relax_edge_floor: float | None = None,
        sample_metrics_trace_path: str | None = None,
        sample_metrics_num_pairs: int = 1,
        sample_metrics_trace_topology_repeats_enabled: bool = False,
        sample_metrics_unseen_start_eval: bool = False,
        sample_metrics_zero_shot_random_start_eval: bool = False,
        sample_metrics_unseen_start_seed: int = 20260430,
        sample_metrics_unseen_start_metric_encoder_path: str | None = None,
        sample_metrics_unseen_pair_selection_mode: str = "random_bank",
        sample_metrics_unseen_start_max_duplicate_tries: int = 100,
        sample_metrics_relaxed_likelihood_enabled: bool = False,
        sample_metrics_branch_relaxer_checkpoint_path: str | None = None,
        sample_metrics_mrbayes20k_enabled: bool = False,
        sample_metrics_mrbayes20k_num_starts: int = 64,
        sample_metrics_mrbayes20k_ngen: int = 20000,
        sample_metrics_mrbayes20k_samplefreq: int = 200,
        sample_metrics_mrbayes20k_printfreq: int = 5000,
        sample_metrics_mrbayes20k_max_workers: int = 12,
        sample_metrics_mrbayes20k_timeout_sec: int = 1800,
        sample_metrics_mrbayes20k_dataset_pickle_path: str | None = None,
        sample_metrics_mrbayes20k_golden_root: str | None = None,
        sample_metrics_mrbayes20k_work_root: str = "/tmp/phylaflow_sample_metrics_mrbayes20k",
        sample_metrics_mrbayes20k_output_dir: str | None = None,
        sample_metrics_mrbayes20k_bin: str = "/opt/conda/envs/phylaflow-mrbayes/bin/mb",
        sample_metrics_tree_dump_enabled: bool = False,
        sample_metrics_tree_dump_dir: str | None = None,
        sample_metrics_checkpoint_enabled: bool = True,
        sample_metrics_checkpoint_dir: str | None = None,
        metric_log_exact_keys=None,
        metric_log_prefixes=None,
        branch_relax_head_weight: float = 0.0,
        branch_relax_head_use_at_sampling: bool = False,
        branch_relax_start_tree_list_path: str | None = None,
        branch_relax_target_tree_list_path: str | None = None,
        branch_relax_detach_trunk: bool = True,
        branch_relax_batch_size: int = 1,
        branch_relax_case_dim: int = 64,
        branch_relax_hidden_dim: int = 256,
        branch_relax_likelihood_dataset_id: str | None = None,
        branch_relax_likelihood_metric_enabled: bool = False,
        rollout_replay_velocity_weight: float = 0.0,
        rollout_replay_autoregressive_weight: float = 0.0,
        rollout_replay_start_step: int = 0,
        rollout_replay_frequency: int = 1,
        rollout_replay_max_velocity_states: int = 0,
        rollout_replay_max_autoregressive_states: int = 0,
        rollout_replay_max_steps: int | None = 256,
        rollout_replay_max_events: int | None = None,
        rollout_replay_anchor_states: int = 4,
        rollout_replay_oracle_horizon: int = 2,
        rollout_replay_mode: str = "anchor_oracle",
        rollout_replay_anchor_include_autoregressive: bool = False,
        rollout_replay_pairwise_max_group_size: int = 0,
        rollout_replay_bank_max_polytomy_size: int = -1,
        rollout_replay_topology_repeat_cap: int = 0,
        rollout_replay_dump_refreshes: bool = False,
        rollout_replay_dump_dir: str | None = None,
        rollout_replay_fixed_dt_base: float | None = None,
        rollout_replay_prefix_stop_early: bool = False,
        rollout_replay_cache_reuse_every_step: bool = True,
        rollout_replay_refresh_only_if_better_rf: bool = False,
        rollout_replay_legacy_loss_structure: bool = False,
        rollout_replay_autoregressive_boundary_local_suffix: bool = False,
        rollout_replay_full_continuation_chain: bool = False,
        rollout_replay_velocity_use_pair_oracle_orthant_labels: bool = False,
        dynamic_start_bank_enabled: bool = False,
        dynamic_start_bank_start_step: int = 0,
        dynamic_start_bank_max_entries: int = 2,
        dynamic_start_bank_min_rf_improvement: float = 0.0,
        dynamic_start_bank_max_polytomy_size: int = -1,
        dynamic_start_bank_mode: str = "best_start",
        dynamic_start_bank_min_velocity_states: int = 2,
        dynamic_start_bank_best_rf_repeat: int = 18,
        dynamic_start_bank_best_multivel_repeat: int = 9,
        dynamic_start_bank_trace_path: str | None = None,
        dynamic_start_bank_artifact_dir: str | None = None,
        dynamic_start_bank_save_improved_checkpoint: bool = False,
        sampling_disable_inner_logging: bool = True,
        sampling_only_first_hit_collapse: bool = False,
        sampling_actual_event_boundary_use_at_sampling: bool = False,
        sampling_actual_event_boundary_include_predicted_first_hit: bool = False,
        sampling_predsim_overrun_use_at_sampling: bool = False,
        sampling_predsim_boundary_mode: str = "pred_simultaneous",
        sampling_predsim_allow_time_overrun: bool = True,
        sampling_blocked_edge_floor: float | None = None,
        sampling_random_fixed_pair_bank_use_at_sampling: bool = False,
        velocity_first_hit_sampling_max_edges: int = -1,
        velocity_first_hit_sampling_fallback_threshold: int = -1,
        velocity_first_hit_sampling_fallback_top_k: int = -1,
        sampling_use_top_merge_planner: bool = False,
        sampling_use_inference_mode: bool = False,
        sampling_cache_tri_mask: bool = False,
        sampling_cache_polytomy_groups: bool = False,
        sampling_cache_autoregressive_state: bool = False,
        use_historical_step_impl: bool = False,
        use_historical_sampling_impl: bool = False,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.optimizer_name = str(optimizer_name).strip().lower()
        self.record = record
        self.epochs = epochs
        self.warmup_steps = 400
        self.current_step_value = 0
        self.lr_scheduler = lr_scheduler
        self.num_annealing_steps = num_annealing_steps
        self.num_warmup_steps = num_warmup_steps
        self.dataset = dataset
        self.use_historical_step_impl = bool(use_historical_step_impl)
        self.use_historical_sampling_impl = bool(use_historical_sampling_impl)
        self.max_num_timesteps = max_num_timesteps
        self.global_splits = global_splits
        self.random_trees = random_trees
        self.verbose = verbose
        self.training_sampling_frequency = training_sampling_frequency
        self.training_sampling_start = training_sampling_start
        self._next_training_sample_step = None
        self.training_sampling_mode = str(training_sampling_mode)
        self.training_sampling_dt_base = float(training_sampling_dt_base)
        self.sampling_fixed_dt_base = (
            None
            if sampling_fixed_dt_base is None
            else float(sampling_fixed_dt_base)
        )
        self.sampling_max_steps = (
            None
            if sampling_max_steps is None or int(sampling_max_steps) < 0
            else int(sampling_max_steps)
        )
        self.sampling_max_events_uncapped = bool(
            sampling_max_events is not None and int(sampling_max_events) < 0
        )
        self.sampling_max_events = (
            None
            if sampling_max_events is None or int(sampling_max_events) < 0
            else int(sampling_max_events)
        )
        self.sampling_max_autoregressive_merges_per_boundary = int(
            sampling_max_autoregressive_merges_per_boundary
        )
        self.training_sampling_stop_on_zero_rf = bool(training_sampling_stop_on_zero_rf)
        self.training_sampling_stop_rf_threshold = (
            None
            if training_sampling_stop_rf_threshold is None
            else float(training_sampling_stop_rf_threshold)
        )
        self.num_samples = num_samples
        self.dt = dt
        self.training_step_gradient_clip_val = float(training_step_gradient_clip_val)
        self.grad_norm_log_frequency = int(grad_norm_log_frequency or 0)
        self.training_step_profile_frequency = int(
            training_step_profile_frequency or 0
        )
        self.training_step_profile_warmup_steps = int(
            training_step_profile_warmup_steps or 0
        )
        self.training_step_profile_sync_cuda = bool(training_step_profile_sync_cuda)
        self.train_tokenized_trees = None
        self.train_batched_time = None
        self.train_tree = None
        self._cached_harness_sampling_pairs = {}
        self.sampling_random_fixed_pair_bank_use_at_sampling = bool(
            sampling_random_fixed_pair_bank_use_at_sampling
        )

        self.automatic_optimization = False
        self.deepspeed = deepspeed
        self.logger_ = logger
        if verbose:
            logging.getLogger("filelock").setLevel(logging.WARNING)
            logging.getLogger("fsspec").setLevel(logging.WARNING)
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        self.phyla_checkpoint_path = phyla_checkpoint_path
        self.phyla_precomputed_embeddings_path = phyla_precomputed_embeddings_path
        self.phyla_model = None
        self.live_phyla_checkpoint_path = live_phyla_checkpoint_path
        self.live_phyla_unfreeze = bool(live_phyla_unfreeze)
        self.live_phyla_lr = (
            None if live_phyla_lr is None else float(live_phyla_lr)
        )
        self.live_phyla_input_mode = str(live_phyla_input_mode or "raw-full")
        self.live_phyla_max_input_tokens = max(
            0,
            int(live_phyla_max_input_tokens or 0),
        )
        live_phyla_device = (
            None
            if live_phyla_device in (None, "", "auto")
            else str(live_phyla_device)
        )
        self.live_phyla_device_config = live_phyla_device
        self.live_phyla_model = None
        self._live_phyla_device = None
        self.phyla_precomputed_name_to_embedding = None
        self.phyla_precomputed_dataset_name_to_embedding = {}
        self.phyla_precomputed_dataset_id_to_tensor = {}
        self.phyla_precomputed_dataset_id_allowlist = (
            self._infer_precomputed_phyla_dataset_ids()
        )

        phyla_config_path = "configs/sample_eval_config.yaml"

        if self.phyla_checkpoint_path is not None:
            original_argv = sys.argv
            sys.argv = ["script", phyla_config_path]
            try:
                if not os.path.exists(phyla_config_path):
                    logging.warning(
                        f"Phyla configuration file not found at {phyla_config_path}"
                    )

                load_config, Config, load_model, _ = _load_phyla_runtime()
                config = load_config(Config)
                config.trainer.checkpoint_path = self.phyla_checkpoint_path
                config.eval.device = "cuda" if torch.cuda.is_available() else "cpu"
                loaded = load_model(config=config, random_model=False)
                self.phyla_model = loaded["model"]
                self.phyla_model.eval()
                if verbose:
                    logging.info("Phyla model loaded successfully.")
            except Exception as e:
                logging.warning(f"Failed to load Phyla model: {e}")
            finally:
                sys.argv = original_argv

        if self.live_phyla_checkpoint_path is not None:
            live_device = (
                self.live_phyla_device_config
                if self.live_phyla_device_config is not None
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            try:
                self.live_phyla_model = _load_live_phyla_beta_model(
                    self.live_phyla_checkpoint_path,
                    device=live_device,
                )
                self._live_phyla_device = str(torch.device(live_device))
                for param in self.live_phyla_model.parameters():
                    param.requires_grad_(self.live_phyla_unfreeze)
                if self.live_phyla_unfreeze:
                    self.live_phyla_model.train()
                else:
                    self.live_phyla_model.eval()
                logging.info(
                    "Loaded live Phyla-beta checkpoint from %s (unfreeze=%s, device=%s)",
                    self.live_phyla_checkpoint_path,
                    self.live_phyla_unfreeze,
                    self._live_phyla_device,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load live Phyla-beta checkpoint from "
                    f"{self.live_phyla_checkpoint_path}: {e}"
                ) from e

        if self.phyla_precomputed_embeddings_path is not None:
            try:
                self._load_precomputed_phyla_embeddings(
                    self.phyla_precomputed_embeddings_path
                )
                logging.info(
                    "Loaded precomputed Phyla embeddings from %s",
                    self.phyla_precomputed_embeddings_path,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load precomputed Phyla embeddings from "
                    f"{self.phyla_precomputed_embeddings_path}: {e}"
                ) from e
        self.stepper = 1

        if self.optimizer_name not in {"adam", "adamw"}:
            raise ValueError(
                "optimizer_name must be one of ['adam', 'adamw'], "
                f"got {optimizer_name!r}."
            )

        valid_velocity_loss_modes = {"plain", "weighted", "blended"}
        if velocity_loss_mode not in valid_velocity_loss_modes:
            raise ValueError(
                f"Invalid velocity_loss_mode={velocity_loss_mode!r}. "
                f"Expected one of {sorted(valid_velocity_loss_modes)}."
            )
        if not (0.0 <= float(velocity_loss_plain_weight) <= 1.0):
            raise ValueError(
                "velocity_loss_plain_weight must be in [0, 1], "
                f"got {velocity_loss_plain_weight}."
            )
        if float(velocity_sign_eps) < 0.0:
            raise ValueError(
                f"velocity_sign_eps must be non-negative, got {velocity_sign_eps}."
            )
        if float(training_step_velocity_weight) < 0.0:
            raise ValueError(
                "training_step_velocity_weight must be non-negative, "
                f"got {training_step_velocity_weight}."
            )
        if float(training_step_autoregressive_weight) < 0.0:
            raise ValueError(
                "training_step_autoregressive_weight must be non-negative, "
                f"got {training_step_autoregressive_weight}."
            )
        if (
            training_step_autoregressive_grad_ratio is not None
            and float(training_step_autoregressive_grad_ratio) < 0.0
        ):
            raise ValueError(
                "training_step_autoregressive_grad_ratio must be non-negative "
                f"or None, got {training_step_autoregressive_grad_ratio}."
            )
        if (
            bool(training_step_separate_optimizer_steps)
            and training_step_autoregressive_grad_ratio is not None
        ):
            raise ValueError(
                "training_step_separate_optimizer_steps cannot be combined with "
                "training_step_autoregressive_grad_ratio."
            )
        valid_autoregressive_target_modes = {"scheduled", "ready_alternatives"}
        if autoregressive_target_mode not in valid_autoregressive_target_modes:
            raise ValueError(
                f"Invalid autoregressive_target_mode={autoregressive_target_mode!r}. "
                f"Expected one of {sorted(valid_autoregressive_target_modes)}."
            )
        valid_topology_decoders = {"ar", "birthset", "birthset_with_ar_fallback"}
        topology_decoder = str(topology_decoder or "ar").lower()
        if topology_decoder not in valid_topology_decoders:
            raise ValueError(
                f"Invalid topology_decoder={topology_decoder!r}. "
                f"Expected one of {sorted(valid_topology_decoders)}."
            )
        if float(birthset_lambda_birth) < 0.0:
            raise ValueError(
                "birthset_lambda_birth must be non-negative, "
                f"got {birthset_lambda_birth}."
            )
        if float(birthset_lambda_rank) < 0.0:
            raise ValueError(
                "birthset_lambda_rank must be non-negative, "
                f"got {birthset_lambda_rank}."
            )
        if float(birthset_lambda_proposal) < 0.0:
            raise ValueError(
                "birthset_lambda_proposal must be non-negative, "
                f"got {birthset_lambda_proposal}."
            )
        if float(birthset_rank_margin) < 0.0:
            raise ValueError(
                "birthset_rank_margin must be non-negative, "
                f"got {birthset_rank_margin}."
            )
        if float(birthset_birth_length) <= 0.0:
            raise ValueError(
                "birthset_birth_length must be positive, "
                f"got {birthset_birth_length}."
            )
        if int(birthset_max_enum_components) < 0:
            raise ValueError(
                "birthset_max_enum_components must be >= 0, "
                f"got {birthset_max_enum_components}."
            )
        if int(birthset_pair_prefix_top_pairs) < 1:
            raise ValueError(
                "birthset_pair_prefix_top_pairs must be >= 1, "
                f"got {birthset_pair_prefix_top_pairs}."
            )
        valid_birthset_proposal_pair_target_modes = {
            "contained",
            "strict_minimal",
        }
        if (
            str(birthset_proposal_pair_target_mode)
            not in valid_birthset_proposal_pair_target_modes
        ):
            raise ValueError(
                "birthset_proposal_pair_target_mode must be one of "
                f"{sorted(valid_birthset_proposal_pair_target_modes)}, got "
                f"{birthset_proposal_pair_target_mode}."
            )
        if int(birthset_proposal_max_expansion_examples) < 1:
            raise ValueError(
                "birthset_proposal_max_expansion_examples must be >= 1, "
                f"got {birthset_proposal_max_expansion_examples}."
            )
        if int(birthset_proposal_max_order_seed_pairs) < 1:
            raise ValueError(
                "birthset_proposal_max_order_seed_pairs must be >= 1, "
                f"got {birthset_proposal_max_order_seed_pairs}."
            )
        if int(birthset_max_candidates_per_polytomy) < 1:
            raise ValueError(
                "birthset_max_candidates_per_polytomy must be >= 1, "
                f"got {birthset_max_candidates_per_polytomy}."
            )
        if int(birthset_negatives_per_positive) < 1:
            raise ValueError(
                "birthset_negatives_per_positive must be >= 1, "
                f"got {birthset_negatives_per_positive}."
            )
        birthset_decoder = str(birthset_decoder or "greedy").lower()
        if birthset_decoder not in {"greedy", "beam"}:
            raise ValueError(
                "birthset_decoder must be one of ['greedy', 'beam'], "
                f"got {birthset_decoder!r}."
            )
        birthset_fallback = str(birthset_fallback or "ar").lower()
        if birthset_fallback not in {"ar", "balanced_completion", "none"}:
            raise ValueError(
                "birthset_fallback must be one of ['ar', 'balanced_completion', 'none'], "
                f"got {birthset_fallback!r}."
            )
        valid_training_sampling_modes = {"batch_compare", "harness_sanity"}
        if self.training_sampling_mode not in valid_training_sampling_modes:
            raise ValueError(
                f"Invalid training_sampling_mode={self.training_sampling_mode!r}. "
                f"Expected one of {sorted(valid_training_sampling_modes)}."
            )
        if self.training_sampling_dt_base <= 0.0:
            raise ValueError(
                "training_sampling_dt_base must be > 0, "
                f"got {training_sampling_dt_base}."
            )
        if (
            self.sampling_fixed_dt_base is not None
            and self.sampling_fixed_dt_base <= 0.0
        ):
            raise ValueError(
                "sampling_fixed_dt_base must be > 0 when provided, "
                f"got {sampling_fixed_dt_base}."
            )
        if sampling_max_steps is not None and int(sampling_max_steps) == 0:
            raise ValueError(
                "sampling_max_steps must be >= 1 or < 0 for uncapped, "
                f"got {sampling_max_steps}."
            )
        if sampling_max_events is not None and int(sampling_max_events) == 0:
            raise ValueError(
                "sampling_max_events must be >= 1 or < 0 for uncapped, "
                f"got {sampling_max_events}."
            )
        if int(sampling_max_autoregressive_merges_per_boundary) == 0:
            raise ValueError(
                "sampling_max_autoregressive_merges_per_boundary must be >= 1 or < 0 "
                f"for uncapped, got {sampling_max_autoregressive_merges_per_boundary}."
            )
        if self.training_step_gradient_clip_val < 0.0:
            raise ValueError(
                "training_step_gradient_clip_val must be >= 0, "
                f"got {training_step_gradient_clip_val}."
            )
        valid_structure_perturb_modes = {"random_wrong_pair", "model_wrong_pair"}
        if (
            autoregressive_structure_perturb_mode
            not in valid_structure_perturb_modes
        ):
            raise ValueError(
                "Invalid autoregressive_structure_perturb_mode="
                f"{autoregressive_structure_perturb_mode!r}. Expected one of "
                f"{sorted(valid_structure_perturb_modes)}."
            )
        if not (0.0 <= float(autoregressive_rollin_prob) <= 1.0):
            raise ValueError(
                "autoregressive_rollin_prob must be in [0, 1], "
                f"got {autoregressive_rollin_prob}."
            )
        if not (0.0 <= float(autoregressive_dagger_prob) <= 1.0):
            raise ValueError(
                "autoregressive_dagger_prob must be in [0, 1], "
                f"got {autoregressive_dagger_prob}."
            )
        if int(autoregressive_dagger_max_steps) < 1:
            raise ValueError(
                "autoregressive_dagger_max_steps must be >= 1, "
                f"got {autoregressive_dagger_max_steps}."
            )
        if not (0.0 <= float(autoregressive_structure_perturb_prob) <= 1.0):
            raise ValueError(
                "autoregressive_structure_perturb_prob must be in [0, 1], "
                f"got {autoregressive_structure_perturb_prob}."
            )
        if not (0.0 <= float(velocity_length_jitter_prob) <= 1.0):
            raise ValueError(
                "velocity_length_jitter_prob must be in [0, 1], "
                f"got {velocity_length_jitter_prob}."
            )
        if float(velocity_length_jitter_scale) < 0.0:
            raise ValueError(
                "velocity_length_jitter_scale must be non-negative, "
                f"got {velocity_length_jitter_scale}."
            )
        if float(velocity_dt_candidate_weight) < 0.0:
            raise ValueError(
                "velocity_dt_candidate_weight must be non-negative, "
                f"got {velocity_dt_candidate_weight}."
            )
        if float(velocity_dt_hit_weight) < 0.0:
            raise ValueError(
                "velocity_dt_hit_weight must be non-negative, "
                f"got {velocity_dt_hit_weight}."
            )
        if float(velocity_logtau_all_weight) < 0.0:
            raise ValueError(
                "velocity_logtau_all_weight must be non-negative, "
                f"got {velocity_logtau_all_weight}."
            )
        if float(velocity_logtau_first_over_weight) < 0.0:
            raise ValueError(
                "velocity_logtau_first_over_weight must be non-negative, "
                f"got {velocity_logtau_first_over_weight}."
            )
        if float(velocity_logtau_first_tie_weight) < 0.0:
            raise ValueError(
                "velocity_logtau_first_tie_weight must be non-negative, "
                f"got {velocity_logtau_first_tie_weight}."
            )
        if float(velocity_logtau_predset_over_weight) < 0.0:
            raise ValueError(
                "velocity_logtau_predset_over_weight must be non-negative, "
                f"got {velocity_logtau_predset_over_weight}."
            )
        if float(velocity_dt_eps) <= 0.0:
            raise ValueError(
                f"velocity_dt_eps must be > 0, got {velocity_dt_eps}."
            )
        if float(velocity_event_weight) < 0.0:
            raise ValueError(
                "velocity_event_weight must be non-negative, "
                f"got {velocity_event_weight}."
            )
        if float(velocity_event_temp) <= 0.0:
            raise ValueError(
                f"velocity_event_temp must be > 0, got {velocity_event_temp}."
            )
        if float(velocity_event_rate_beta) <= 0.0:
            raise ValueError(
                f"velocity_event_rate_beta must be > 0, got {velocity_event_rate_beta}."
            )
        if float(velocity_event_precision_weight) < 0.0:
            raise ValueError(
                "velocity_event_precision_weight must be non-negative, "
                f"got {velocity_event_precision_weight}."
            )
        if float(velocity_event_precision_margin) < 0.0:
            raise ValueError(
                "velocity_event_precision_margin must be non-negative, "
                f"got {velocity_event_precision_margin}."
            )
        if float(velocity_first_hit_head_weight) < 0.0:
            raise ValueError(
                "velocity_first_hit_head_weight must be non-negative, "
                f"got {velocity_first_hit_head_weight}."
            )
        if float(velocity_first_hit_loss_tol) < 0.0:
            raise ValueError(
                "velocity_first_hit_loss_tol must be non-negative, "
                f"got {velocity_first_hit_loss_tol}."
            )
        if float(velocity_first_hit_false_positive_mass_weight) < 0.0:
            raise ValueError(
                "velocity_first_hit_false_positive_mass_weight must be non-negative, "
                f"got {velocity_first_hit_false_positive_mass_weight}."
            )
        if float(velocity_first_hit_false_negative_mass_weight) < 0.0:
            raise ValueError(
                "velocity_first_hit_false_negative_mass_weight must be non-negative, "
                f"got {velocity_first_hit_false_negative_mass_weight}."
            )
        if float(velocity_boundary_vanish_head_weight) < 0.0:
            raise ValueError(
                "velocity_boundary_vanish_head_weight must be non-negative, "
                f"got {velocity_boundary_vanish_head_weight}."
            )
        if float(velocity_boundary_time_head_weight) < 0.0:
            raise ValueError(
                "velocity_boundary_time_head_weight must be non-negative, "
                f"got {velocity_boundary_time_head_weight}."
            )
        if float(velocity_terminal_head_weight) < 0.0:
            raise ValueError(
                "velocity_terminal_head_weight must be non-negative, "
                f"got {velocity_terminal_head_weight}."
            )
        if int(velocity_first_hit_geometry_hidden_dim) < 1:
            raise ValueError(
                "velocity_first_hit_geometry_hidden_dim must be >= 1, "
                f"got {velocity_first_hit_geometry_hidden_dim}."
            )
        if int(velocity_first_hit_edge_length_hidden_dim) < 1:
            raise ValueError(
                "velocity_first_hit_edge_length_hidden_dim must be >= 1, "
                f"got {velocity_first_hit_edge_length_hidden_dim}."
            )
        if int(velocity_boundary_time_hidden_dim) < 1:
            raise ValueError(
                "velocity_boundary_time_hidden_dim must be >= 1, "
                f"got {velocity_boundary_time_hidden_dim}."
            )
        if (
            bool(velocity_boundary_vanish_one_step_use_at_sampling)
            and not bool(velocity_boundary_vanish_head_use_at_sampling)
        ):
            raise ValueError(
                "velocity_boundary_vanish_one_step_use_at_sampling requires "
                "velocity_boundary_vanish_head_use_at_sampling."
            )
        if int(sampling_discrete_phase_max_phases) < 1:
            raise ValueError(
                "sampling_discrete_phase_max_phases must be >= 1, "
                f"got {sampling_discrete_phase_max_phases}."
            )
        if float(rollout_replay_velocity_weight) < 0.0:
            raise ValueError(
                "rollout_replay_velocity_weight must be non-negative, "
                f"got {rollout_replay_velocity_weight}."
            )
        if float(rollout_replay_autoregressive_weight) < 0.0:
            raise ValueError(
                "rollout_replay_autoregressive_weight must be non-negative, "
                f"got {rollout_replay_autoregressive_weight}."
            )
        if int(rollout_replay_start_step) < 0:
            raise ValueError(
                "rollout_replay_start_step must be >= 0, "
                f"got {rollout_replay_start_step}."
            )
        if int(rollout_replay_frequency) < 1:
            raise ValueError(
                "rollout_replay_frequency must be >= 1, "
                f"got {rollout_replay_frequency}."
            )
        if int(rollout_replay_max_velocity_states) < 0:
            raise ValueError(
                "rollout_replay_max_velocity_states must be >= 0, "
                f"got {rollout_replay_max_velocity_states}."
            )
        if int(rollout_replay_max_autoregressive_states) < 0:
            raise ValueError(
                "rollout_replay_max_autoregressive_states must be >= 0, "
                f"got {rollout_replay_max_autoregressive_states}."
            )
        if (
            rollout_replay_max_steps is not None
            and int(rollout_replay_max_steps) == 0
        ):
            raise ValueError(
                "rollout_replay_max_steps must be >= 1 or < 0 for uncapped, "
                f"got {rollout_replay_max_steps}."
            )
        if (
            rollout_replay_max_events is not None
            and int(rollout_replay_max_events) == 0
        ):
            raise ValueError(
                "rollout_replay_max_events must be >= 1 or < 0 for uncapped, "
                f"got {rollout_replay_max_events}."
            )
        if int(rollout_replay_anchor_states) < 1:
            raise ValueError(
                "rollout_replay_anchor_states must be >= 1, "
                f"got {rollout_replay_anchor_states}."
            )
        if int(rollout_replay_oracle_horizon) < 1:
            raise ValueError(
                "rollout_replay_oracle_horizon must be >= 1, "
                f"got {rollout_replay_oracle_horizon}."
            )
        valid_rollout_replay_modes = {
            "anchor_oracle",
            "legacy_prefix_oracle",
            "legacy_start_end_oracle",
            "filtered_polytomy_bank_oracle",
            "first_wrong_velocity_suffix_oracle",
        }
        if str(rollout_replay_mode) not in valid_rollout_replay_modes:
            raise ValueError(
                "rollout_replay_mode must be one of "
                f"{sorted(valid_rollout_replay_modes)}, got {rollout_replay_mode!r}."
            )
        if int(rollout_replay_pairwise_max_group_size) < 0:
            raise ValueError(
                "rollout_replay_pairwise_max_group_size must be >= 0, "
                f"got {rollout_replay_pairwise_max_group_size}."
            )
        if int(rollout_replay_bank_max_polytomy_size) < -1:
            raise ValueError(
                "rollout_replay_bank_max_polytomy_size must be >= -1, "
                f"got {rollout_replay_bank_max_polytomy_size}."
            )
        if int(rollout_replay_topology_repeat_cap) < 0:
            raise ValueError(
                "rollout_replay_topology_repeat_cap must be >= 0, "
                f"got {rollout_replay_topology_repeat_cap}."
            )
        if rollout_replay_dump_dir is not None and not str(
            rollout_replay_dump_dir
        ).strip():
            raise ValueError(
                "rollout_replay_dump_dir must be a non-empty path when provided."
            )
        if self.deepspeed and (
            float(rollout_replay_velocity_weight) > 0.0
            or float(rollout_replay_autoregressive_weight) > 0.0
        ):
            raise ValueError(
                "Rollout replay losses are not supported with deepspeed training."
            )
        self.velocity_loss_mode = velocity_loss_mode
        self.velocity_loss_plain_weight = float(velocity_loss_plain_weight)
        self.velocity_sign_eps = float(velocity_sign_eps)
        self.training_step_velocity_weight = float(training_step_velocity_weight)
        self.training_step_autoregressive_weight = float(
            training_step_autoregressive_weight
        )
        self.training_step_separate_optimizer_steps = bool(
            training_step_separate_optimizer_steps
        )
        self.training_step_verbose_logging_enabled = bool(
            training_step_verbose_logging_enabled
        )
        if training_step_autoregressive_grad_ratio is None:
            self.training_step_autoregressive_grad_ratio = None
        else:
            self.training_step_autoregressive_grad_ratio = float(
                training_step_autoregressive_grad_ratio
            )
        self.autoregressive_use_time = bool(autoregressive_use_time)
        self.autoregressive_target_mode = str(autoregressive_target_mode)
        self.autoregressive_polytomy_choosing_weight = float(
            autoregressive_polytomy_choosing_weight
        )
        self.autoregressive_stop_after_merge_weight = float(
            autoregressive_stop_after_merge_weight
        )
        self.autoregressive_stop_after_merge_use_at_sampling = bool(
            autoregressive_stop_after_merge_use_at_sampling
        )
        self.autoregressive_rollin_prob = float(autoregressive_rollin_prob)
        self.autoregressive_dagger_prob = float(autoregressive_dagger_prob)
        self.autoregressive_dagger_max_steps = int(autoregressive_dagger_max_steps)
        self.autoregressive_structure_perturb_prob = float(
            autoregressive_structure_perturb_prob
        )
        self.autoregressive_structure_perturb_mode = str(
            autoregressive_structure_perturb_mode
        )
        self.topology_decoder = topology_decoder
        self.birthset_birth_length = float(birthset_birth_length)
        self.birthset_lambda_birth = float(birthset_lambda_birth)
        self.birthset_lambda_rank = float(birthset_lambda_rank)
        self.birthset_lambda_proposal = float(birthset_lambda_proposal)
        self.birthset_rank_margin = float(birthset_rank_margin)
        self.birthset_pos_weight = birthset_pos_weight
        self.birthset_use_train_birth_split_bank = bool(
            birthset_use_train_birth_split_bank
        )
        self.birthset_use_small_polytomy_enumeration = bool(
            birthset_use_small_polytomy_enumeration
        )
        self.birthset_use_pair_prefix_candidates = bool(
            birthset_use_pair_prefix_candidates
        )
        self.birthset_use_component_phyla_conditioning = bool(
            birthset_use_component_phyla_conditioning
        )
        self.birthset_pair_prefix_top_pairs = int(birthset_pair_prefix_top_pairs)
        self.birthset_proposal_pair_target_mode = str(
            birthset_proposal_pair_target_mode
        )
        self.birthset_proposal_max_expansion_examples = int(
            birthset_proposal_max_expansion_examples
        )
        self.birthset_proposal_max_order_seed_pairs = int(
            birthset_proposal_max_order_seed_pairs
        )
        self.birthset_proposal_train_topk = bool(birthset_proposal_train_topk)
        self.birthset_max_enum_components = int(birthset_max_enum_components)
        self.birthset_max_candidates_per_polytomy = int(
            birthset_max_candidates_per_polytomy
        )
        self.birthset_negatives_per_positive = int(
            birthset_negatives_per_positive
        )
        self.birthset_decoder = birthset_decoder
        self.birthset_beam_width = max(1, int(birthset_beam_width))
        self.birthset_fallback = birthset_fallback
        self.birthset_split_bank = set()
        self.birthset_topology_head = None
        self.birthset_proposal_head = None
        if self.topology_decoder in {"birthset", "birthset_with_ar_fallback"}:
            birthset_component_phyla_dim = (
                int(getattr(self.model, "phyla_dim", 0))
                if self.birthset_use_component_phyla_conditioning
                else None
            )
            self.birthset_topology_head = BirthSetTopologyHead(
                int(self.model.embed_dim),
                hidden=max(128, int(self.model.embed_dim)),
                dropout=0.0,
                context_dim=int(self.model.embed_dim),
                max_components_norm=max(16, int(self.model.embed_dim)),
                component_phyla_dim=birthset_component_phyla_dim,
            )
            if self.birthset_use_pair_prefix_candidates:
                self.birthset_proposal_head = BirthSetTopologyHead(
                    int(self.model.embed_dim),
                    hidden=max(128, int(self.model.embed_dim)),
                    dropout=0.0,
                    context_dim=int(self.model.embed_dim),
                    max_components_norm=max(16, int(self.model.embed_dim)),
                    component_phyla_dim=birthset_component_phyla_dim,
                )
        self.velocity_length_jitter_prob = float(velocity_length_jitter_prob)
        self.velocity_length_jitter_scale = float(velocity_length_jitter_scale)
        self.velocity_dt_candidate_weight = float(velocity_dt_candidate_weight)
        self.velocity_dt_hit_weight = float(velocity_dt_hit_weight)
        self.velocity_logtau_all_weight = float(velocity_logtau_all_weight)
        self.velocity_logtau_first_over_weight = float(
            velocity_logtau_first_over_weight
        )
        self.velocity_logtau_first_tie_weight = float(velocity_logtau_first_tie_weight)
        self.velocity_logtau_predset_over_weight = float(
            velocity_logtau_predset_over_weight
        )
        self.velocity_dt_eps = float(velocity_dt_eps)
        self.velocity_event_weight = float(velocity_event_weight)
        self.velocity_event_temp = float(velocity_event_temp)
        self.velocity_event_rate_beta = float(velocity_event_rate_beta)
        self.velocity_event_normalize_by_log_candidates = bool(
            velocity_event_normalize_by_log_candidates
        )
        self.velocity_event_precision_weight = float(velocity_event_precision_weight)
        self.velocity_event_precision_margin = float(velocity_event_precision_margin)
        self.velocity_first_hit_head_weight = float(velocity_first_hit_head_weight)
        self.velocity_first_hit_loss_tol = float(velocity_first_hit_loss_tol)
        self.velocity_first_hit_false_positive_mass_weight = float(
            velocity_first_hit_false_positive_mass_weight
        )
        self.velocity_first_hit_false_negative_mass_weight = float(
            velocity_first_hit_false_negative_mass_weight
        )
        self.velocity_first_hit_head_use_at_sampling = bool(
            velocity_first_hit_head_use_at_sampling
        )
        self.sampling_only_first_hit_collapse = bool(
            sampling_only_first_hit_collapse
        )
        self.sampling_actual_event_boundary_use_at_sampling = bool(
            sampling_actual_event_boundary_use_at_sampling
        )
        self.sampling_actual_event_boundary_include_predicted_first_hit = bool(
            sampling_actual_event_boundary_include_predicted_first_hit
        )
        self.sampling_predsim_overrun_use_at_sampling = bool(
            sampling_predsim_overrun_use_at_sampling
        )
        valid_predsim_boundary_modes = {
            "pred_simultaneous",
            "actual_hit_predset",
            "actual_hit_union",
            "pred_simultaneous_time_head",
        }
        if str(sampling_predsim_boundary_mode) not in valid_predsim_boundary_modes:
            raise ValueError(
                "sampling_predsim_boundary_mode must be one of "
                f"{sorted(valid_predsim_boundary_modes)}, got "
                f"{sampling_predsim_boundary_mode!r}."
            )
        self.sampling_predsim_boundary_mode = str(sampling_predsim_boundary_mode)
        self.sampling_predsim_allow_time_overrun = bool(
            sampling_predsim_allow_time_overrun
        )
        self.sampling_blocked_edge_floor = (
            None
            if sampling_blocked_edge_floor is None
            else float(sampling_blocked_edge_floor)
        )
        self.velocity_first_hit_sampling_max_edges = int(
            velocity_first_hit_sampling_max_edges
        )
        self.velocity_first_hit_sampling_fallback_threshold = int(
            velocity_first_hit_sampling_fallback_threshold
        )
        self.velocity_first_hit_sampling_fallback_top_k = int(
            velocity_first_hit_sampling_fallback_top_k
        )
        predictor_mode = str(velocity_first_hit_predictor_mode)
        if predictor_mode == "base" and bool(velocity_first_hit_use_geometry_features):
            predictor_mode = "residual_geometry"
        valid_first_hit_predictor_modes = {
            "base",
            "residual_geometry",
            "edge_length",
            "edge_token_attention",
            "edge_token_attention_replace",
            "edge_token_attention_logitinput_replace",
            "edge_token_attention_logitinput_replace_latelength",
        }
        if predictor_mode not in valid_first_hit_predictor_modes:
            raise ValueError(
                "velocity_first_hit_predictor_mode must be one of "
                f"{sorted(valid_first_hit_predictor_modes)}, got {predictor_mode!r}."
            )
        self.velocity_first_hit_predictor_mode = predictor_mode
        self.velocity_first_hit_use_geometry_features = bool(
            velocity_first_hit_use_geometry_features
        )
        self.velocity_first_hit_geometry_hidden_dim = int(
            velocity_first_hit_geometry_hidden_dim
        )
        self.velocity_first_hit_edge_length_hidden_dim = int(
            velocity_first_hit_edge_length_hidden_dim
        )
        self.velocity_first_hit_attention_layers = int(
            velocity_first_hit_attention_layers
        )
        self.velocity_first_hit_attention_heads = int(
            velocity_first_hit_attention_heads
        )
        self.velocity_first_hit_bucket_count = int(velocity_first_hit_bucket_count)
        self.velocity_first_hit_bucket_log_min = float(
            velocity_first_hit_bucket_log_min
        )
        self.velocity_first_hit_bucket_log_max = float(
            velocity_first_hit_bucket_log_max
        )
        velocity_refiner_mode = str(velocity_refiner_mode or "base")
        valid_velocity_refiner_modes = {"base", "edge_token_attention_delta"}
        if velocity_refiner_mode not in valid_velocity_refiner_modes:
            raise ValueError(
                "velocity_refiner_mode must be one of "
                f"{sorted(valid_velocity_refiner_modes)}, got {velocity_refiner_mode!r}."
            )
        self.velocity_refiner_mode = velocity_refiner_mode
        self.velocity_refiner_attention_layers = int(velocity_refiner_attention_layers)
        self.velocity_refiner_attention_heads = int(velocity_refiner_attention_heads)
        self.velocity_refiner_bucket_count = int(velocity_refiner_bucket_count)
        self.velocity_refiner_bucket_log_min = float(velocity_refiner_bucket_log_min)
        self.velocity_refiner_bucket_log_max = float(velocity_refiner_bucket_log_max)
        self.velocity_first_hit_geometry_head = None
        self.velocity_first_hit_edge_length_head = None
        self.velocity_first_hit_attention_logit_proj = None
        self.velocity_first_hit_attention_length_bucket = None
        self.velocity_first_hit_attention_tau_bucket = None
        self.velocity_first_hit_attention_contract_embed = None
        self.velocity_first_hit_attention_norm = None
        self.velocity_first_hit_attention_layers_mod = None
        self.velocity_first_hit_attention_out = None
        self.velocity_boundary_time_head = None
        self.velocity_terminal_head = None
        self.velocity_terminal_head_edge_topology_fusion = None
        if self.velocity_first_hit_predictor_mode == "residual_geometry":
            self.velocity_first_hit_geometry_head = nn.Sequential(
                nn.LayerNorm(5),
                nn.Linear(5, self.velocity_first_hit_geometry_hidden_dim),
                nn.GELU(),
                nn.Linear(self.velocity_first_hit_geometry_hidden_dim, 1),
            )
        elif self.velocity_first_hit_predictor_mode == "edge_length":
            self.velocity_first_hit_edge_length_head = nn.Sequential(
                nn.LayerNorm(int(self.model.embed_dim) + 1),
                nn.Linear(
                    int(self.model.embed_dim) + 1,
                    self.velocity_first_hit_edge_length_hidden_dim,
                ),
                nn.GELU(),
                nn.Linear(self.velocity_first_hit_edge_length_hidden_dim, 1),
            )
        elif self.velocity_first_hit_predictor_mode in {
            "edge_token_attention",
            "edge_token_attention_replace",
            "edge_token_attention_logitinput_replace",
            "edge_token_attention_logitinput_replace_latelength",
        }:
            if self.velocity_first_hit_bucket_count < 2:
                raise ValueError(
                    "velocity_first_hit_bucket_count must be >= 2 for edge_token_attention."
                )
            if (
                self.velocity_first_hit_bucket_log_max
                <= self.velocity_first_hit_bucket_log_min
            ):
                raise ValueError(
                    "velocity_first_hit_bucket_log_max must be > velocity_first_hit_bucket_log_min "
                    "for edge_token_attention."
                )
            if (
                int(self.model.embed_dim)
                % max(1, self.velocity_first_hit_attention_heads)
                != 0
            ):
                raise ValueError(
                    "model.embed_dim must be divisible by velocity_first_hit_attention_heads "
                    f"for edge_token_attention, got {self.model.embed_dim} and "
                    f"{self.velocity_first_hit_attention_heads}."
                )
            self.velocity_first_hit_attention_logit_proj = nn.Linear(
                1,
                int(self.model.embed_dim),
                bias=False,
            )
            self.velocity_first_hit_attention_length_bucket = nn.Embedding(
                self.velocity_first_hit_bucket_count,
                int(self.model.embed_dim),
            )
            self.velocity_first_hit_attention_tau_bucket = nn.Embedding(
                self.velocity_first_hit_bucket_count,
                int(self.model.embed_dim),
            )
            self.velocity_first_hit_attention_contract_embed = nn.Embedding(
                2,
                int(self.model.embed_dim),
            )
            self.velocity_first_hit_attention_norm = nn.LayerNorm(
                int(self.model.embed_dim)
            )
            self.velocity_first_hit_attention_layers_mod = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=int(self.model.embed_dim),
                        nhead=self.velocity_first_hit_attention_heads,
                        dim_feedforward=4 * int(self.model.embed_dim),
                        dropout=0.0,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(max(1, self.velocity_first_hit_attention_layers))
                ]
            )
            attention_out_dim = int(self.model.embed_dim)
            if (
                self.velocity_first_hit_predictor_mode
                == "edge_token_attention_logitinput_replace_latelength"
            ):
                attention_out_dim = 2 * int(self.model.embed_dim)
            self.velocity_first_hit_attention_out = nn.Sequential(
                nn.LayerNorm(attention_out_dim),
                nn.Linear(attention_out_dim, int(self.model.embed_dim)),
                nn.GELU(),
                nn.Linear(int(self.model.embed_dim), 1),
            )
        if (
            float(velocity_boundary_time_head_weight) > 0.0
            or bool(velocity_boundary_time_head_use_at_sampling)
        ):
            boundary_time_input_dim = int(self.model.embed_dim) + 4
            self.velocity_boundary_time_head = nn.Sequential(
                nn.LayerNorm(boundary_time_input_dim),
                nn.Linear(
                    boundary_time_input_dim,
                    int(velocity_boundary_time_hidden_dim),
                ),
                nn.GELU(),
                nn.Linear(int(velocity_boundary_time_hidden_dim), 1),
            )
        if (
            float(velocity_terminal_head_weight) > 0.0
            or bool(velocity_terminal_head_use_at_sampling)
        ):
            terminal_input_mode = (
                "probe"
                if velocity_terminal_head_input_mode is None
                and bool(velocity_terminal_head_probe_features)
                else (
                    "edge_summary"
                    if velocity_terminal_head_input_mode is None
                    else str(velocity_terminal_head_input_mode).lower()
                )
            )
            terminal_input_dim = (
                18
                if terminal_input_mode == "probe"
                else int(self.model.embed_dim)
                if terminal_input_mode in {"topology_only", "edge_only", "edge_topology"}
                else int(self.model.embed_dim) + 5
            )
            if bool(velocity_terminal_head_use_case_adapt):
                terminal_input_dim += int(self.model.first_hit_head_case_dim)
            if terminal_input_mode == "edge_topology":
                self.velocity_terminal_head_edge_topology_fusion = nn.Sequential(
                    nn.LayerNorm(2 * int(self.model.embed_dim)),
                    nn.Linear(2 * int(self.model.embed_dim), int(self.model.embed_dim)),
                    nn.GELU(),
                    nn.Linear(int(self.model.embed_dim), int(self.model.embed_dim)),
                )
            self.velocity_terminal_head = nn.Sequential(
                nn.LayerNorm(terminal_input_dim),
                nn.Linear(
                    terminal_input_dim,
                    int(velocity_terminal_head_hidden_dim),
                ),
                nn.GELU(),
                nn.Linear(int(velocity_terminal_head_hidden_dim), 1),
            )
        self.velocity_refiner_base_proj = None
        self.velocity_refiner_length_bucket = None
        self.velocity_refiner_norm = None
        self.velocity_refiner_layers_mod = None
        self.velocity_refiner_out = None
        if self.velocity_refiner_mode == "edge_token_attention_delta":
            if self.velocity_refiner_bucket_count < 2:
                raise ValueError(
                    "velocity_refiner_bucket_count must be >= 2 for edge_token_attention_delta."
                )
            if self.velocity_refiner_bucket_log_max <= self.velocity_refiner_bucket_log_min:
                raise ValueError(
                    "velocity_refiner_bucket_log_max must be > velocity_refiner_bucket_log_min "
                    "for edge_token_attention_delta."
                )
            if (
                int(self.model.embed_dim)
                % max(1, self.velocity_refiner_attention_heads)
                != 0
            ):
                raise ValueError(
                    "model.embed_dim must be divisible by velocity_refiner_attention_heads "
                    f"for edge_token_attention_delta, got {self.model.embed_dim} and "
                    f"{self.velocity_refiner_attention_heads}."
                )
            self.velocity_refiner_base_proj = nn.Linear(
                1,
                int(self.model.embed_dim),
                bias=False,
            )
            self.velocity_refiner_length_bucket = nn.Embedding(
                self.velocity_refiner_bucket_count,
                int(self.model.embed_dim),
            )
            self.velocity_refiner_norm = nn.LayerNorm(int(self.model.embed_dim))
            self.velocity_refiner_layers_mod = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=int(self.model.embed_dim),
                        nhead=self.velocity_refiner_attention_heads,
                        dim_feedforward=4 * int(self.model.embed_dim),
                        dropout=0.0,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(max(1, self.velocity_refiner_attention_layers))
                ]
            )
            self.velocity_refiner_out = nn.Sequential(
                nn.LayerNorm(int(self.model.embed_dim)),
                nn.Linear(int(self.model.embed_dim), int(self.model.embed_dim)),
                nn.GELU(),
                nn.Linear(int(self.model.embed_dim), 1),
            )
        self.velocity_boundary_vanish_head_weight = float(
            velocity_boundary_vanish_head_weight
        )
        self.velocity_boundary_vanish_head_use_at_sampling = bool(
            velocity_boundary_vanish_head_use_at_sampling
        )
        self.velocity_boundary_vanish_one_step_use_at_sampling = bool(
            velocity_boundary_vanish_one_step_use_at_sampling
        )
        self.velocity_boundary_time_head_weight = float(
            velocity_boundary_time_head_weight
        )
        self.velocity_boundary_time_head_use_at_sampling = bool(
            velocity_boundary_time_head_use_at_sampling
        )
        self.velocity_boundary_time_hidden_dim = int(
            velocity_boundary_time_hidden_dim
        )
        self.velocity_terminal_head_weight = float(
            velocity_terminal_head_weight
        )
        self.velocity_terminal_head_use_at_sampling = bool(
            velocity_terminal_head_use_at_sampling
        )
        self.velocity_terminal_head_balance_loss = bool(
            velocity_terminal_head_balance_loss
        )
        terminal_topology_pool = str(velocity_terminal_head_topology_pool).lower()
        if terminal_topology_pool not in {"mean", "sum"}:
            raise ValueError(
                "velocity_terminal_head_topology_pool must be "
                f"'mean' or 'sum', got {terminal_topology_pool!r}."
            )
        self.velocity_terminal_head_topology_pool = terminal_topology_pool
        terminal_sampling_action = str(velocity_terminal_head_sampling_action).lower()
        if terminal_sampling_action not in {"after_phase", "immediate"}:
            raise ValueError(
                "velocity_terminal_head_sampling_action must be "
                f"'after_phase' or 'immediate', got {terminal_sampling_action!r}."
            )
        self.velocity_terminal_head_sampling_action = terminal_sampling_action
        self.velocity_terminal_head_hidden_dim = int(
            velocity_terminal_head_hidden_dim
        )
        self.velocity_terminal_head_probe_features = bool(
            velocity_terminal_head_probe_features
        )
        self.velocity_terminal_head_use_case_adapt = bool(
            velocity_terminal_head_use_case_adapt
        )
        if self.velocity_terminal_head_use_case_adapt and not hasattr(
            self.model, "first_hit_case_embedding"
        ):
            raise ValueError(
                "velocity_terminal_head_use_case_adapt requires a model with first_hit_case_embedding."
            )
        self.velocity_terminal_head_input_mode = (
            "probe"
            if velocity_terminal_head_input_mode is None
            and self.velocity_terminal_head_probe_features
            else (
                "edge_summary"
                if velocity_terminal_head_input_mode is None
                else str(velocity_terminal_head_input_mode).lower()
            )
        )
        self.velocity_probe_direct_set_loss = bool(
            velocity_probe_direct_set_loss
        )
        self.velocity_probe_direct_set_anchor_only = bool(
            velocity_probe_direct_set_anchor_only
        )
        self.velocity_probe_direct_set_target_negative_weight = float(
            velocity_probe_direct_set_target_negative_weight
        )
        self.velocity_probe_direct_set_nontarget_nonnegative_weight = float(
            velocity_probe_direct_set_nontarget_nonnegative_weight
        )
        self.velocity_probe_direct_set_positive_reweight = bool(
            velocity_probe_direct_set_positive_reweight
        )
        self.velocity_probe_direct_set_include_base_samples = bool(
            velocity_probe_direct_set_include_base_samples
        )
        self.velocity_probe_direct_set_positive_reweight_power = float(
            velocity_probe_direct_set_positive_reweight_power
        )
        self.velocity_probe_direct_set_positive_reweight_max = (
            None
            if velocity_probe_direct_set_positive_reweight_max is None
            else float(velocity_probe_direct_set_positive_reweight_max)
        )
        self.velocity_probe_direct_set_bce_weight = float(
            velocity_probe_direct_set_bce_weight
        )
        self.velocity_probe_direct_set_loss_weight = float(
            velocity_probe_direct_set_loss_weight
        )
        self.velocity_probe_direct_set_mse_weight = float(
            velocity_probe_direct_set_mse_weight
        )
        self.training_step_probe_parity_joint_update = bool(
            training_step_probe_parity_joint_update
        )
        self.training_step_joint_tokenize_velocity_ar = bool(
            training_step_joint_tokenize_velocity_ar
        )
        self.training_step_full_path_replay_initial_retry_attempt = max(
            0, int(training_step_full_path_replay_initial_retry_attempt or 0)
        )
        self.skip_repeated_no_valid_boundary_use_at_sampling = bool(
            skip_repeated_no_valid_boundary_use_at_sampling
        )
        self.sampling_discrete_phase_rollout_use_at_sampling = bool(
            sampling_discrete_phase_rollout_use_at_sampling
        )
        self.sampling_discrete_phase_exact_boundary_step_use_at_sampling = bool(
            sampling_discrete_phase_exact_boundary_step_use_at_sampling
        )
        self.sampling_discrete_phase_max_phases = int(
            sampling_discrete_phase_max_phases
        )
        self.sampling_final_orthant_relax_use_at_sampling = bool(
            sampling_final_orthant_relax_use_at_sampling
        )
        self.sampling_final_orthant_relax_steps = int(
            sampling_final_orthant_relax_steps
        )
        self.sampling_final_orthant_relax_total_time = float(
            sampling_final_orthant_relax_total_time
        )
        final_relax_time_mode = str(sampling_final_orthant_relax_time_mode).lower()
        valid_final_relax_time_modes = {"local", "phase", "phase_local"}
        if final_relax_time_mode not in valid_final_relax_time_modes:
            raise ValueError(
                "sampling_final_orthant_relax_time_mode must be one of "
                f"{sorted(valid_final_relax_time_modes)}, got "
                f"{sampling_final_orthant_relax_time_mode!r}."
            )
        self.sampling_final_orthant_relax_time_mode = final_relax_time_mode
        self.sampling_final_orthant_relax_edge_floor = (
            None
            if sampling_final_orthant_relax_edge_floor is None
            else float(sampling_final_orthant_relax_edge_floor)
        )
        if self.sampling_final_orthant_relax_steps < 0:
            raise ValueError(
                "sampling_final_orthant_relax_steps must be >= 0, "
                f"got {sampling_final_orthant_relax_steps}."
            )
        if (
            self.sampling_final_orthant_relax_use_at_sampling
            and self.sampling_final_orthant_relax_steps > 0
            and self.sampling_final_orthant_relax_total_time <= 0.0
        ):
            raise ValueError(
                "sampling_final_orthant_relax_total_time must be > 0, "
                f"got {sampling_final_orthant_relax_total_time}."
            )
        if (
            self.sampling_final_orthant_relax_edge_floor is not None
            and self.sampling_final_orthant_relax_edge_floor <= 0.0
        ):
            raise ValueError(
                "sampling_final_orthant_relax_edge_floor must be > 0 when provided, "
                f"got {sampling_final_orthant_relax_edge_floor}."
            )
        self.sample_metrics_trace_path = (
            str(sample_metrics_trace_path).strip()
            if sample_metrics_trace_path
            else None
        )
        self.sample_metrics_num_pairs = max(1, int(sample_metrics_num_pairs))
        self.sample_metrics_trace_topology_repeats_enabled = bool(
            sample_metrics_trace_topology_repeats_enabled
        )
        self.sample_metrics_unseen_start_eval = bool(sample_metrics_unseen_start_eval)
        self.sample_metrics_zero_shot_random_start_eval = bool(
            sample_metrics_zero_shot_random_start_eval
        )
        self.sample_metrics_unseen_start_seed = int(sample_metrics_unseen_start_seed)
        self.sample_metrics_unseen_start_metric_encoder_path = (
            os.path.abspath(str(sample_metrics_unseen_start_metric_encoder_path))
            if sample_metrics_unseen_start_metric_encoder_path
            else None
        )
        self.sample_metrics_unseen_pair_selection_mode = str(
            sample_metrics_unseen_pair_selection_mode
        ).strip().lower()
        self.sample_metrics_unseen_start_max_duplicate_tries = max(
            1, int(sample_metrics_unseen_start_max_duplicate_tries)
        )
        self.sample_metrics_relaxed_likelihood_enabled = bool(
            sample_metrics_relaxed_likelihood_enabled
        )
        self.sample_metrics_branch_relaxer_checkpoint_path = (
            os.path.abspath(str(sample_metrics_branch_relaxer_checkpoint_path))
            if sample_metrics_branch_relaxer_checkpoint_path
            else None
        )
        self.sample_metrics_mrbayes20k_enabled = bool(
            sample_metrics_mrbayes20k_enabled
        )
        self.sample_metrics_mrbayes20k_num_starts = max(
            1, int(sample_metrics_mrbayes20k_num_starts)
        )
        self.sample_metrics_mrbayes20k_ngen = max(
            1, int(sample_metrics_mrbayes20k_ngen)
        )
        self.sample_metrics_mrbayes20k_samplefreq = max(
            1, int(sample_metrics_mrbayes20k_samplefreq)
        )
        self.sample_metrics_mrbayes20k_printfreq = max(
            1, int(sample_metrics_mrbayes20k_printfreq)
        )
        self.sample_metrics_mrbayes20k_max_workers = max(
            1, int(sample_metrics_mrbayes20k_max_workers)
        )
        self.sample_metrics_mrbayes20k_timeout_sec = max(
            1, int(sample_metrics_mrbayes20k_timeout_sec)
        )
        self.sample_metrics_mrbayes20k_dataset_pickle_path = (
            os.path.abspath(str(sample_metrics_mrbayes20k_dataset_pickle_path))
            if sample_metrics_mrbayes20k_dataset_pickle_path
            else None
        )
        self.sample_metrics_mrbayes20k_golden_root = (
            os.path.abspath(str(sample_metrics_mrbayes20k_golden_root))
            if sample_metrics_mrbayes20k_golden_root
            else None
        )
        self.sample_metrics_mrbayes20k_work_root = os.path.abspath(
            str(sample_metrics_mrbayes20k_work_root)
        )
        self.sample_metrics_mrbayes20k_output_dir = (
            os.path.abspath(str(sample_metrics_mrbayes20k_output_dir))
            if sample_metrics_mrbayes20k_output_dir
            else None
        )
        self.sample_metrics_mrbayes20k_bin = str(sample_metrics_mrbayes20k_bin)
        self.sample_metrics_tree_dump_enabled = bool(
            sample_metrics_tree_dump_enabled
        )
        self.sample_metrics_tree_dump_dir = (
            os.path.abspath(str(sample_metrics_tree_dump_dir))
            if sample_metrics_tree_dump_dir
            else None
        )
        self.sample_metrics_checkpoint_enabled = bool(
            sample_metrics_checkpoint_enabled
        )
        self.sample_metrics_checkpoint_dir = (
            os.path.abspath(str(sample_metrics_checkpoint_dir))
            if sample_metrics_checkpoint_dir
            else None
        )
        self._sample_metrics_checkpoint_steps = set()
        self._sample_metrics_standalone_relaxer_cache = {}
        self._sample_metrics_likelihood_scorer_cache = {}
        self._sample_metrics_metric_encoder_cache = {}
        self.metric_log_exact_keys = (
            {
                str(key).strip()
                for key in metric_log_exact_keys
                if str(key).strip()
            }
            if metric_log_exact_keys
            else None
        )
        self.metric_log_prefixes = tuple(
            str(prefix).strip()
            for prefix in (metric_log_prefixes or [])
            if str(prefix).strip()
        )
        self.branch_relax_head_weight = float(branch_relax_head_weight)
        self.branch_relax_head_use_at_sampling = bool(branch_relax_head_use_at_sampling)
        self.branch_relax_detach_trunk = bool(branch_relax_detach_trunk)
        self.branch_relax_batch_size = max(1, int(branch_relax_batch_size))
        self.branch_relax_likelihood_dataset_id = (
            str(branch_relax_likelihood_dataset_id).strip()
            if branch_relax_likelihood_dataset_id
            else None
        )
        self.branch_relax_likelihood_metric_enabled = bool(
            branch_relax_likelihood_metric_enabled
        )
        self._branch_relax_likelihood_scorer = None
        self.branch_relax_samples = []
        if branch_relax_start_tree_list_path and branch_relax_target_tree_list_path:
            self.branch_relax_samples = _build_branch_relax_samples_for_module(
                self,
                str(branch_relax_start_tree_list_path),
                str(branch_relax_target_tree_list_path),
            )
        branch_relax_num_cases = (
            int(getattr(self.model, "first_hit_head_num_cases", 0) or 0)
            or len(self.branch_relax_samples)
            or 1
        )
        self.branch_relax_head = None
        if (
            self.branch_relax_head_weight > 0.0
            or self.branch_relax_head_use_at_sampling
        ):
            self.branch_relax_head = BranchRelaxHead(
                int(self.model.embed_dim),
                int(branch_relax_num_cases),
                case_dim=int(branch_relax_case_dim),
                hidden_dim=int(branch_relax_hidden_dim),
            )
        self.rollout_replay_velocity_weight = float(rollout_replay_velocity_weight)
        self.rollout_replay_autoregressive_weight = float(
            rollout_replay_autoregressive_weight
        )
        self.rollout_replay_start_step = int(rollout_replay_start_step)
        self.rollout_replay_frequency = int(rollout_replay_frequency)
        self.rollout_replay_max_velocity_states = int(
            rollout_replay_max_velocity_states
        )
        self.rollout_replay_max_autoregressive_states = int(
            rollout_replay_max_autoregressive_states
        )
        self.rollout_replay_max_steps = (
            None
            if rollout_replay_max_steps is None or int(rollout_replay_max_steps) < 0
            else int(rollout_replay_max_steps)
        )
        self.rollout_replay_max_events_uncapped = bool(
            rollout_replay_max_events is not None
            and int(rollout_replay_max_events) < 0
        )
        self.rollout_replay_max_events = (
            None
            if rollout_replay_max_events is None or int(rollout_replay_max_events) < 0
            else int(rollout_replay_max_events)
        )
        self.rollout_replay_anchor_states = int(rollout_replay_anchor_states)
        self.rollout_replay_oracle_horizon = int(rollout_replay_oracle_horizon)
        self.rollout_replay_mode = str(rollout_replay_mode)
        self.rollout_replay_anchor_include_autoregressive = bool(
            rollout_replay_anchor_include_autoregressive
        )
        self.rollout_replay_pairwise_max_group_size = int(
            rollout_replay_pairwise_max_group_size
        )
        self.rollout_replay_bank_max_polytomy_size = int(
            rollout_replay_bank_max_polytomy_size
        )
        self.rollout_replay_topology_repeat_cap = int(
            rollout_replay_topology_repeat_cap
        )
        self.rollout_replay_dump_refreshes = bool(rollout_replay_dump_refreshes)
        self.rollout_replay_dump_dir = (
            os.path.abspath(str(rollout_replay_dump_dir))
            if rollout_replay_dump_dir is not None
            else None
        )
        self.rollout_replay_fixed_dt_base = (
            None
            if rollout_replay_fixed_dt_base is None
            else float(rollout_replay_fixed_dt_base)
        )
        if (
            self.rollout_replay_fixed_dt_base is not None
            and self.rollout_replay_fixed_dt_base <= 0.0
        ):
            raise ValueError(
                "rollout_replay_fixed_dt_base must be > 0 when provided, "
                f"got {rollout_replay_fixed_dt_base}."
            )
        self.rollout_replay_prefix_stop_early = bool(
            rollout_replay_prefix_stop_early
        )
        self.rollout_replay_cache_reuse_every_step = bool(
            rollout_replay_cache_reuse_every_step
        )
        self.rollout_replay_refresh_only_if_better_rf = bool(
            rollout_replay_refresh_only_if_better_rf
        )
        self.rollout_replay_legacy_loss_structure = bool(
            rollout_replay_legacy_loss_structure
        )
        self.rollout_replay_autoregressive_boundary_local_suffix = bool(
            rollout_replay_autoregressive_boundary_local_suffix
        )
        self.rollout_replay_full_continuation_chain = bool(
            rollout_replay_full_continuation_chain
        )
        self.rollout_replay_velocity_use_pair_oracle_orthant_labels = bool(
            rollout_replay_velocity_use_pair_oracle_orthant_labels
        )
        self.dynamic_start_bank_enabled = bool(dynamic_start_bank_enabled)
        self.dynamic_start_bank_start_step = int(dynamic_start_bank_start_step)
        self.dynamic_start_bank_max_entries = int(dynamic_start_bank_max_entries)
        self.dynamic_start_bank_min_rf_improvement = float(
            dynamic_start_bank_min_rf_improvement
        )
        self.dynamic_start_bank_max_polytomy_size = int(
            dynamic_start_bank_max_polytomy_size
        )
        valid_dynamic_start_bank_modes = {
            "best_start",
            "soft_hybrid",
            "multivel_only",
        }
        if str(dynamic_start_bank_mode) not in valid_dynamic_start_bank_modes:
            raise ValueError(
                "dynamic_start_bank_mode must be one of "
                f"{sorted(valid_dynamic_start_bank_modes)}, got "
                f"{dynamic_start_bank_mode!r}."
            )
        if int(dynamic_start_bank_min_velocity_states) < 1:
            raise ValueError(
                "dynamic_start_bank_min_velocity_states must be >= 1, "
                f"got {dynamic_start_bank_min_velocity_states}."
            )
        if int(dynamic_start_bank_best_rf_repeat) < 1:
            raise ValueError(
                "dynamic_start_bank_best_rf_repeat must be >= 1, "
                f"got {dynamic_start_bank_best_rf_repeat}."
            )
        if int(dynamic_start_bank_best_multivel_repeat) < 1:
            raise ValueError(
                "dynamic_start_bank_best_multivel_repeat must be >= 1, "
                f"got {dynamic_start_bank_best_multivel_repeat}."
            )
        self.dynamic_start_bank_mode = str(dynamic_start_bank_mode)
        self.dynamic_start_bank_min_velocity_states = int(
            dynamic_start_bank_min_velocity_states
        )
        self.dynamic_start_bank_best_rf_repeat = int(
            dynamic_start_bank_best_rf_repeat
        )
        self.dynamic_start_bank_best_multivel_repeat = int(
            dynamic_start_bank_best_multivel_repeat
        )
        self.dynamic_start_bank_trace_path = (
            os.path.abspath(str(dynamic_start_bank_trace_path))
            if dynamic_start_bank_trace_path
            else None
        )
        self.dynamic_start_bank_artifact_dir = (
            os.path.abspath(str(dynamic_start_bank_artifact_dir))
            if dynamic_start_bank_artifact_dir
            else None
        )
        self.dynamic_start_bank_save_improved_checkpoint = bool(
            dynamic_start_bank_save_improved_checkpoint
        )
        self._dynamic_start_bank_base = None
        self._dynamic_start_bank_best_rf_norm = None
        self._dynamic_start_bank_best_rf_tree = None
        self._dynamic_start_bank_best_rf_item = None
        self._dynamic_start_bank_best_multivel_rf_norm = None
        self._dynamic_start_bank_best_multivel_tree = None
        self._dynamic_start_bank_best_multivel_item = None
        self._harness_sampling_frozen_start_bank_base = None
        self.sampling_disable_inner_logging = bool(sampling_disable_inner_logging)
        self.sampling_use_top_merge_planner = bool(sampling_use_top_merge_planner)
        self.sampling_use_inference_mode = bool(sampling_use_inference_mode)
        self.sampling_cache_tri_mask = bool(sampling_cache_tri_mask)
        self.sampling_cache_polytomy_groups = bool(sampling_cache_polytomy_groups)
        self.sampling_cache_autoregressive_state = bool(
            sampling_cache_autoregressive_state
        )
        self._rollout_replay_dump_counter = 0
        self._cached_rollout_replay_batches = {
            "train": {
                "velocity": None,
                "autoregressive": None,
                "velocity_bank": None,
                "autoregressive_bank": None,
                "sampled_rf_norm": None,
                "logs": {},
            },
            "val": {
                "velocity": None,
                "autoregressive": None,
                "velocity_bank": None,
                "autoregressive_bank": None,
                "sampled_rf_norm": None,
                "logs": {},
            },
        }
        self._posterior_reference_bundle_cache = {}

    def on_train_start(self):
        super().on_train_start()
        self._reset_training_sampling_schedule()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["birthset_split_bank"] = sorted(
            int(split) for split in getattr(self, "birthset_split_bank", set())
        )

    def on_load_checkpoint(self, checkpoint):
        bank = checkpoint.get("birthset_split_bank")
        if bank is not None:
            self.birthset_split_bank = {int(split) for split in bank}

    def _reset_training_sampling_schedule(self):
        frequency = int(self.training_sampling_frequency)
        if frequency <= 0:
            self._next_training_sample_step = None
            return

        next_step = int(self.training_sampling_start)
        current_step = int(self.global_step)
        if current_step >= next_step:
            missed_intervals = ((current_step - next_step) // frequency) + 1
            next_step += missed_intervals * frequency
        self._next_training_sample_step = int(next_step)

    def _training_sample_due(self):
        frequency = int(self.training_sampling_frequency)
        if frequency <= 0:
            return False
        if self._next_training_sample_step is None:
            self._reset_training_sampling_schedule()
        return int(self.global_step) >= int(self._next_training_sample_step)

    def _advance_training_sampling_schedule(self):
        frequency = int(self.training_sampling_frequency)
        if frequency <= 0 or self._next_training_sample_step is None:
            return
        current_step = int(self.global_step)
        while int(self._next_training_sample_step) <= current_step:
            self._next_training_sample_step += frequency

    def _metric_key_allowed(self, key):
        key = str(key)
        if self.metric_log_exact_keys is None and not self.metric_log_prefixes:
            return True
        if self.metric_log_exact_keys is not None and key in self.metric_log_exact_keys:
            return True
        return any(key.startswith(prefix) for prefix in self.metric_log_prefixes)

    def _filter_metric_dict(self, metrics):
        return {
            key: value
            for key, value in metrics.items()
            if self._metric_key_allowed(key)
        }

    def _log_scalar_filtered(self, key, value, **kwargs):
        if not self._metric_key_allowed(key):
            return
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return
        elif not isinstance(value, numbers.Number):
            return
        self.log(key, value, **kwargs)

    def _wandb_log_filtered(self, metrics, step=None):
        if not self.record:
            return
        filtered = self._filter_metric_dict(metrics)
        if filtered:
            wandb.log(filtered, step=self.stepper if step is None else step)

    def _append_dynamic_start_bank_trace(self, payload):
        if not self.dynamic_start_bank_trace_path:
            return
        os.makedirs(
            os.path.dirname(self.dynamic_start_bank_trace_path),
            exist_ok=True,
        )
        with open(self.dynamic_start_bank_trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _dynamic_start_bank_artifact_root(self):
        if self.dynamic_start_bank_artifact_dir:
            return self.dynamic_start_bank_artifact_dir
        if self.dynamic_start_bank_trace_path:
            stem = os.path.splitext(os.path.basename(self.dynamic_start_bank_trace_path))[0]
            return os.path.join(os.path.dirname(self.dynamic_start_bank_trace_path), f"{stem}_artifacts")
        return None

    def _save_dynamic_start_bank_artifacts(
        self,
        *,
        payload,
        sampled_tree,
        bank,
    ):
        artifact_root = self._dynamic_start_bank_artifact_root()
        if not artifact_root:
            return payload

        os.makedirs(artifact_root, exist_ok=True)
        step = int(payload["step"])
        rf_val = float(payload["sampled_rf_norm"])
        base_name = f"step={step:06d}-rf={rf_val:.6f}"

        tree_payload = dict(payload)
        tree_payload["sampled_tree"] = str(sampled_tree)
        tree_payload["bank_trees"] = list(bank)
        tree_path = os.path.join(artifact_root, f"{base_name}-tree.json")
        with open(tree_path, "w", encoding="utf-8") as handle:
            json.dump(tree_payload, handle, indent=2, sort_keys=True)
        payload["tree_artifact_path"] = tree_path

        if (
            self.dynamic_start_bank_save_improved_checkpoint
            and getattr(self, "trainer", None) is not None
            and getattr(self.trainer, "is_global_zero", True)
        ):
            ckpt_path = os.path.join(artifact_root, f"{base_name}.ckpt")
            self.trainer.save_checkpoint(ckpt_path)
            payload["checkpoint_artifact_path"] = ckpt_path
        return payload

    def _ensure_dynamic_start_bank_state(self, dataset_split):
        if self._dynamic_start_bank_base is None:
            base_items = list(
                getattr(dataset_split, "overfit_fixed_pair_start_tree_bank_items", [])
                or []
            )
            if base_items:
                self._dynamic_start_bank_base = [
                    dict(item) if isinstance(item, dict) else item
                    for item in base_items
                ]
            else:
                self._dynamic_start_bank_base = list(
                    getattr(
                        dataset_split,
                        "overfit_fixed_pair_start_tree_newick_bank",
                        [],
                    )
                    or []
                )

    def _refresh_dynamic_start_bank_soft_hybrid(self, dataset_split):
        self._ensure_dynamic_start_bank_state(dataset_split)
        bank = list(self._dynamic_start_bank_base or [])
        if (
            self.dynamic_start_bank_mode == "soft_hybrid"
            and self._dynamic_start_bank_best_rf_item is not None
        ):
            bank.extend(
                [self._dynamic_start_bank_best_rf_item]
                * max(self.dynamic_start_bank_best_rf_repeat, 1)
            )
        if self._dynamic_start_bank_best_multivel_item is not None:
            bank.extend(
                [self._dynamic_start_bank_best_multivel_item]
                * max(self.dynamic_start_bank_best_multivel_repeat, 1)
            )
        return dataset_split.set_overfit_fixed_pair_start_tree_bank(bank)

    def _copy_bank_items(self, items):
        return [
            dict(item) if isinstance(item, dict) else item
            for item in (items or [])
        ]

    def _get_frozen_harness_start_bank_base(self, dataset_split):
        if self._harness_sampling_frozen_start_bank_base is not None:
            return self._copy_bank_items(self._harness_sampling_frozen_start_bank_base)

        if self._dynamic_start_bank_base is not None:
            base = self._copy_bank_items(self._dynamic_start_bank_base)
        else:
            base = self._copy_bank_items(
                getattr(dataset_split, "overfit_fixed_pair_start_tree_bank_items", [])
            )
        self._harness_sampling_frozen_start_bank_base = self._copy_bank_items(base)
        return base

    def _sample_overfit_fixed_pair_bank_pair_for_harness(
        self,
        dataset_split,
        *,
        frozen_start_bank=False,
    ):
        if not hasattr(dataset_split, "sample_overfit_fixed_pair_bank_pair"):
            return None
        if not frozen_start_bank:
            return dataset_split.sample_overfit_fixed_pair_bank_pair()

        frozen_bank = self._get_frozen_harness_start_bank_base(dataset_split)
        if not frozen_bank:
            return dataset_split.sample_overfit_fixed_pair_bank_pair()

        current_bank = self._copy_bank_items(
            getattr(dataset_split, "overfit_fixed_pair_start_tree_bank_items", [])
        )
        try:
            dataset_split.set_overfit_fixed_pair_start_tree_bank(frozen_bank)
            return dataset_split.sample_overfit_fixed_pair_bank_pair()
        finally:
            dataset_split.set_overfit_fixed_pair_start_tree_bank(current_bank)

    def _maybe_update_dynamic_start_bank(
        self,
        *,
        pair,
        sampled_tree,
        sampled_rf_norm,
        trace,
        train=True,
    ):
        logs = {}
        if not train or not self.dynamic_start_bank_enabled:
            return logs
        if int(self.stepper) < self.dynamic_start_bank_start_step:
            return logs
        dataset_obj = getattr(self, "dataset", None)
        dataset_split = (
            getattr(dataset_obj, "dataset_train", None)
            if dataset_obj is not None
            else None
        )
        if dataset_split is None or not getattr(dataset_split, "overfit_fixed_pair", False):
            return logs
        if sampled_tree is None:
            return logs

        candidate_rf_norm = float(sampled_rf_norm)
        start_rf_norm = float(calculate_norm_rf(pair["start_tree"], pair["target_tree"]))
        num_velocity_states = int(len(trace.get("velocity", [])))
        num_ar_states = int(len(trace.get("autoregressive", [])))
        max_polytomy_size = _max_polytomy_size_from_newick(sampled_tree)
        candidate_bank_item = str(sampled_tree)
        if pair.get("bank_group_key") is not None:
            candidate_bank_item = {
                "start_tree": str(sampled_tree),
                "bank_group_key": str(pair.get("bank_group_key")),
            }

        if (
            self.dynamic_start_bank_max_polytomy_size >= 0
            and max_polytomy_size > self.dynamic_start_bank_max_polytomy_size
        ):
            logs["dynamic_start_bank/updated"] = torch.tensor(
                0.0, dtype=torch.float32, device=self.device
            )
            logs["dynamic_start_bank/rejected_max_polytomy_size"] = torch.tensor(
                float(max_polytomy_size),
                dtype=torch.float32,
                device=self.device,
            )
            return logs

        if self.dynamic_start_bank_mode in {"soft_hybrid", "multivel_only"}:
            self._ensure_dynamic_start_bank_state(dataset_split)
            min_improvement = float(self.dynamic_start_bank_min_rf_improvement)
            update_tags = []

            if self.dynamic_start_bank_mode == "soft_hybrid":
                current_best_rf = self._dynamic_start_bank_best_rf_norm
                if current_best_rf is None:
                    current_best_rf = start_rf_norm
                if candidate_rf_norm < (float(current_best_rf) - min_improvement):
                    self._dynamic_start_bank_best_rf_norm = float(candidate_rf_norm)
                    self._dynamic_start_bank_best_rf_tree = str(sampled_tree)
                    self._dynamic_start_bank_best_rf_item = candidate_bank_item
                    update_tags.append("best_rf")

            if num_velocity_states >= self.dynamic_start_bank_min_velocity_states:
                current_best_multivel = self._dynamic_start_bank_best_multivel_rf_norm
                if current_best_multivel is None:
                    current_best_multivel = start_rf_norm
                if candidate_rf_norm < (
                    float(current_best_multivel) - min_improvement
                ):
                    self._dynamic_start_bank_best_multivel_rf_norm = float(
                        candidate_rf_norm
                    )
                    self._dynamic_start_bank_best_multivel_tree = str(sampled_tree)
                    self._dynamic_start_bank_best_multivel_item = (
                        candidate_bank_item
                    )
                    update_tags.append("best_multivel")

            if not update_tags:
                logs["dynamic_start_bank/updated"] = torch.tensor(
                    0.0, dtype=torch.float32, device=self.device
                )
                logs["dynamic_start_bank/bank_size"] = torch.tensor(
                    float(
                        len(
                            getattr(
                                dataset_split,
                                "overfit_fixed_pair_start_tree_newick_bank",
                                [],
                            )
                        )
                    ),
                    dtype=torch.float32,
                    device=self.device,
                )
                if num_velocity_states < self.dynamic_start_bank_min_velocity_states:
                    logs["dynamic_start_bank/rejected_low_velocity_states"] = (
                        torch.tensor(
                            float(num_velocity_states),
                            dtype=torch.float32,
                            device=self.device,
                        )
                    )
                return logs

            bank = self._refresh_dynamic_start_bank_soft_hybrid(dataset_split)
            payload = {
                "step": int(self.stepper),
                "sampled_rf_norm": float(candidate_rf_norm),
                "start_rf_norm": float(start_rf_norm),
                "num_velocity_states": int(num_velocity_states),
                "num_ar_states": int(num_ar_states),
                "max_polytomy_size": int(max_polytomy_size),
                "bank_size": int(len(bank)),
                "base_bank_size": int(len(self._dynamic_start_bank_base or [])),
                "update_kind": "+".join(update_tags),
                "best_rf_norm": (
                    None
                    if self._dynamic_start_bank_best_rf_norm is None
                    else float(self._dynamic_start_bank_best_rf_norm)
                ),
                "best_multivel_rf_norm": (
                    None
                    if self._dynamic_start_bank_best_multivel_rf_norm is None
                    else float(self._dynamic_start_bank_best_multivel_rf_norm)
                ),
                "best_rf_repeat": int(self.dynamic_start_bank_best_rf_repeat),
                "best_multivel_repeat": int(
                    self.dynamic_start_bank_best_multivel_repeat
                ),
            }
            payload = self._save_dynamic_start_bank_artifacts(
                payload=payload,
                sampled_tree=sampled_tree,
                bank=bank,
            )
            self._append_dynamic_start_bank_trace(payload)

            logs["dynamic_start_bank/updated"] = torch.tensor(
                1.0, dtype=torch.float32, device=self.device
            )
            logs["dynamic_start_bank/bank_size"] = torch.tensor(
                float(len(bank)),
                dtype=torch.float32,
                device=self.device,
            )
            if self._dynamic_start_bank_best_rf_norm is not None:
                logs["dynamic_start_bank/best_rf_norm"] = torch.tensor(
                    float(self._dynamic_start_bank_best_rf_norm),
                    dtype=torch.float32,
                    device=self.device,
                )
            if self._dynamic_start_bank_best_multivel_rf_norm is not None:
                logs["dynamic_start_bank/best_multivel_rf_norm"] = torch.tensor(
                    float(self._dynamic_start_bank_best_multivel_rf_norm),
                    dtype=torch.float32,
                    device=self.device,
                )
            logs["dynamic_start_bank/max_polytomy_size"] = torch.tensor(
                float(max_polytomy_size),
                dtype=torch.float32,
                device=self.device,
            )
            logs["dynamic_start_bank/updated_best_rf"] = torch.tensor(
                1.0 if "best_rf" in update_tags else 0.0,
                dtype=torch.float32,
                device=self.device,
            )
            logs["dynamic_start_bank/updated_best_multivel"] = torch.tensor(
                1.0 if "best_multivel" in update_tags else 0.0,
                dtype=torch.float32,
                device=self.device,
            )
            return logs

        current_best = self._dynamic_start_bank_best_rf_norm
        if current_best is None:
            current_best = start_rf_norm
        if candidate_rf_norm >= (
            float(current_best) - float(self.dynamic_start_bank_min_rf_improvement)
        ):
            logs["dynamic_start_bank/updated"] = torch.tensor(
                0.0, dtype=torch.float32, device=self.device
            )
            logs["dynamic_start_bank/bank_size"] = torch.tensor(
                float(
                    len(
                        getattr(
                            dataset_split,
                            "overfit_fixed_pair_start_tree_newick_bank",
                            [],
                        )
                    )
                ),
                dtype=torch.float32,
                device=self.device,
            )
            return logs

        bank = dataset_split.set_overfit_fixed_pair_best_start_tree(
            sampled_tree,
            max_bank_size=self.dynamic_start_bank_max_entries,
            keep_first=True,
        )
        self._dynamic_start_bank_best_rf_norm = float(candidate_rf_norm)
        payload = {
            "step": int(self.stepper),
            "sampled_rf_norm": float(candidate_rf_norm),
            "start_rf_norm": float(start_rf_norm),
            "num_velocity_states": int(num_velocity_states),
            "num_ar_states": int(num_ar_states),
            "max_polytomy_size": int(max_polytomy_size),
            "bank_size": int(len(bank)),
        }
        payload = self._save_dynamic_start_bank_artifacts(
            payload=payload,
            sampled_tree=sampled_tree,
            bank=bank,
        )
        self._append_dynamic_start_bank_trace(payload)
        logs["dynamic_start_bank/updated"] = torch.tensor(
            1.0, dtype=torch.float32, device=self.device
        )
        logs["dynamic_start_bank/bank_size"] = torch.tensor(
            float(len(bank)),
            dtype=torch.float32,
            device=self.device,
        )
        logs["dynamic_start_bank/best_rf_norm"] = torch.tensor(
            float(candidate_rf_norm),
            dtype=torch.float32,
            device=self.device,
        )
        logs["dynamic_start_bank/max_polytomy_size"] = torch.tensor(
            float(max_polytomy_size),
            dtype=torch.float32,
            device=self.device,
        )
        return logs

    def _first_hit_bucket_ids(self, values):
        values = values.float()
        bucket_ids = torch.zeros_like(values, dtype=torch.long)
        positive_mask = values > 0.0
        if not bool(positive_mask.any()):
            return bucket_ids
        log_values = torch.log(values[positive_mask].clamp_min(1e-12))
        scaled = (log_values - self.velocity_first_hit_bucket_log_min) / (
            self.velocity_first_hit_bucket_log_max
            - self.velocity_first_hit_bucket_log_min
        )
        max_nonzero_bucket = self.velocity_first_hit_bucket_count - 2
        clipped = torch.clamp(scaled, 0.0, 1.0)
        bucket_vals = torch.floor(clipped * max_nonzero_bucket).to(torch.long) + 1
        bucket_ids[positive_mask] = bucket_vals
        return bucket_ids

    def _velocity_refiner_bucket_ids(self, values):
        values = values.float()
        bucket_ids = torch.zeros_like(values, dtype=torch.long)
        positive_mask = values > 0.0
        if not bool(positive_mask.any()):
            return bucket_ids
        log_values = torch.log(values[positive_mask].clamp_min(1e-12))
        scaled = (log_values - self.velocity_refiner_bucket_log_min) / (
            self.velocity_refiner_bucket_log_max
            - self.velocity_refiner_bucket_log_min
        )
        max_nonzero_bucket = self.velocity_refiner_bucket_count - 2
        clipped = torch.clamp(scaled, 0.0, 1.0)
        bucket_vals = torch.floor(clipped * max_nonzero_bucket).to(torch.long) + 1
        bucket_ids[positive_mask] = bucket_vals
        return bucket_ids

    def _refine_velocity_predictions(
        self,
        velocity_pred,
        lengths,
        edge_features=None,
        group_sizes=None,
    ):
        if self.velocity_refiner_mode == "base":
            return velocity_pred
        if (
            velocity_pred is None
            or edge_features is None
            or self.velocity_refiner_base_proj is None
            or self.velocity_refiner_length_bucket is None
            or self.velocity_refiner_norm is None
            or self.velocity_refiner_layers_mod is None
            or self.velocity_refiner_out is None
        ):
            return velocity_pred

        original_shape = velocity_pred.shape
        pred_flat = velocity_pred.reshape(-1).float()
        lengths_flat = lengths.reshape(-1).to(
            edge_features.device, dtype=edge_features.dtype
        )
        edge_features_flat = edge_features.reshape(
            -1, edge_features.shape[-1]
        ).to(edge_features.device, dtype=edge_features.dtype)

        if edge_features_flat.numel() == 0:
            return velocity_pred

        if group_sizes is None:
            group_sizes = [int(edge_features_flat.shape[0])]
        group_sizes = [int(size) for size in group_sizes if int(size) > 0]
        if sum(group_sizes) != int(edge_features_flat.shape[0]):
            group_sizes = [int(edge_features_flat.shape[0])]

        x = edge_features_flat
        x = x + self.velocity_refiner_base_proj(pred_flat.unsqueeze(-1))
        x = x + self.velocity_refiner_length_bucket(
            self._velocity_refiner_bucket_ids(lengths_flat)
        )
        x = self.velocity_refiner_norm(x)

        deltas = []
        start = 0
        for size in group_sizes:
            end = start + int(size)
            x_group = x[start:end].unsqueeze(0)
            for layer in self.velocity_refiner_layers_mod:
                x_group = layer(x_group)
            delta_group = self.velocity_refiner_out(x_group).reshape(-1)
            deltas.append(delta_group)
            start = end

        refined = (pred_flat + torch.cat(deltas, dim=0)).reshape(original_shape)
        return refined.to(dtype=velocity_pred.dtype)

    def _compute_first_hit_logits(
        self,
        first_hit_logits,
        lengths,
        velocities,
        edge_features=None,
        group_sizes=None,
    ):
        if first_hit_logits is None and edge_features is None:
            return None
        if self.velocity_first_hit_predictor_mode == "base":
            return first_hit_logits

        if self.velocity_first_hit_predictor_mode == "edge_length":
            if (
                edge_features is None
                or self.velocity_first_hit_edge_length_head is None
            ):
                return first_hit_logits
            feature_shape = edge_features.shape[:-1]
            edge_features_flat = edge_features.reshape(
                -1, edge_features.shape[-1]
            ).float()
            lengths_flat = lengths.reshape(-1).to(
                edge_features_flat.device, dtype=edge_features_flat.dtype
            )
            eps = float(self.velocity_dt_eps)
            log_length = torch.log(lengths_flat.clamp_min(eps)).unsqueeze(-1)
            mlp_input = torch.cat([edge_features_flat, log_length], dim=-1)
            out = self.velocity_first_hit_edge_length_head(mlp_input).reshape(
                feature_shape
            )
            dtype = (
                first_hit_logits.dtype
                if first_hit_logits is not None
                else edge_features.dtype
            )
            return out.to(dtype=dtype)

        if self.velocity_first_hit_predictor_mode in {
            "edge_token_attention",
            "edge_token_attention_replace",
            "edge_token_attention_logitinput_replace",
            "edge_token_attention_logitinput_replace_latelength",
        }:
            if (
                edge_features is None
                or self.velocity_first_hit_attention_logit_proj is None
                or self.velocity_first_hit_attention_length_bucket is None
                or self.velocity_first_hit_attention_tau_bucket is None
                or self.velocity_first_hit_attention_contract_embed is None
                or self.velocity_first_hit_attention_norm is None
                or self.velocity_first_hit_attention_layers_mod is None
                or self.velocity_first_hit_attention_out is None
            ):
                return first_hit_logits

            original_shape = edge_features.shape[:-1]
            logits_flat = (
                first_hit_logits.reshape(-1).float()
                if first_hit_logits is not None
                else None
            )
            lengths_flat = lengths.reshape(-1).to(
                edge_features.device, dtype=edge_features.dtype
            )
            velocities_flat = velocities.reshape(-1).to(
                edge_features.device, dtype=edge_features.dtype
            )
            edge_features_flat = edge_features.reshape(
                -1, edge_features.shape[-1]
            ).to(edge_features.device, dtype=edge_features.dtype)

            if edge_features_flat.numel() == 0:
                return first_hit_logits

            if group_sizes is None:
                group_sizes = [int(edge_features_flat.shape[0])]
            group_sizes = [int(size) for size in group_sizes if int(size) > 0]
            if sum(group_sizes) != int(edge_features_flat.shape[0]):
                group_sizes = [int(edge_features_flat.shape[0])]

            eps = float(self.velocity_dt_eps)
            sign_eps = float(self.velocity_sign_eps)
            contract_mask = (velocities_flat < -sign_eps).long()
            safe_rate = (-velocities_flat).clamp_min(sign_eps)
            tau_pred = lengths_flat.clamp_min(eps) / safe_rate

            x = edge_features_flat
            if (
                self.velocity_first_hit_predictor_mode
                in {
                    "edge_token_attention",
                    "edge_token_attention_logitinput_replace",
                    "edge_token_attention_logitinput_replace_latelength",
                }
                and logits_flat is not None
            ):
                x = x + self.velocity_first_hit_attention_logit_proj(
                    logits_flat.unsqueeze(-1)
                )
            length_bucket_embed = self.velocity_first_hit_attention_length_bucket(
                self._first_hit_bucket_ids(lengths_flat)
            )
            x = x + length_bucket_embed
            x = x + self.velocity_first_hit_attention_tau_bucket(
                self._first_hit_bucket_ids(tau_pred)
            )
            x = x + self.velocity_first_hit_attention_contract_embed(contract_mask)
            x = self.velocity_first_hit_attention_norm(x)

            deltas = []
            start = 0
            for size in group_sizes:
                end = start + int(size)
                x_group = x[start:end].unsqueeze(0)
                length_group = length_bucket_embed[start:end].unsqueeze(0)
                for layer in self.velocity_first_hit_attention_layers_mod:
                    x_group = layer(x_group)
                classifier_input = x_group
                if (
                    self.velocity_first_hit_predictor_mode
                    == "edge_token_attention_logitinput_replace_latelength"
                ):
                    classifier_input = torch.cat([x_group, length_group], dim=-1)
                delta_group = self.velocity_first_hit_attention_out(
                    classifier_input
                ).reshape(-1)
                deltas.append(delta_group)
                start = end

            delta = torch.cat(deltas, dim=0).reshape(original_shape)
            if self.velocity_first_hit_predictor_mode in {
                "edge_token_attention_replace",
                "edge_token_attention_logitinput_replace",
                "edge_token_attention_logitinput_replace_latelength",
            }:
                dtype = (
                    first_hit_logits.dtype
                    if first_hit_logits is not None
                    else edge_features.dtype
                )
                return delta.to(dtype=dtype)
            refined = (logits_flat + delta.reshape(-1)).reshape(original_shape)
            return refined.to(dtype=first_hit_logits.dtype)

        if self.velocity_first_hit_geometry_head is None or first_hit_logits is None:
            return first_hit_logits

        original_shape = first_hit_logits.shape
        logits_flat = first_hit_logits.reshape(-1).float()
        lengths_flat = lengths.reshape(-1).to(logits_flat.device, dtype=logits_flat.dtype)
        velocities_flat = velocities.reshape(-1).to(
            logits_flat.device, dtype=logits_flat.dtype
        )

        if logits_flat.numel() == 0:
            return first_hit_logits

        eps = float(self.velocity_dt_eps)
        sign_eps = float(self.velocity_sign_eps)
        log_length = torch.log(lengths_flat.clamp_min(eps))
        detached_velocity = velocities_flat.detach()
        contract_mask = (detached_velocity < -sign_eps).float()
        safe_rate = (-detached_velocity).clamp_min(sign_eps)
        tau_pred = lengths_flat.clamp_min(eps) / safe_rate
        log_tau_pred = torch.log(tau_pred.clamp_min(eps)).clamp(min=-20.0, max=20.0)

        geometry_features = torch.stack(
            [
                logits_flat,
                log_length,
                detached_velocity,
                log_tau_pred,
                contract_mask,
            ],
            dim=-1,
        )
        delta = self.velocity_first_hit_geometry_head(geometry_features).reshape(-1)
        refined = (logits_flat + delta).reshape(original_shape)
        return refined.to(dtype=first_hit_logits.dtype)

    def _predict_boundary_time_log(
        self,
        lengths,
        velocities,
        edge_features=None,
        group_sizes=None,
    ):
        if self.velocity_boundary_time_head is None or edge_features is None:
            return None

        if edge_features.numel() == 0:
            return None

        lengths_flat = lengths.reshape(-1).to(
            edge_features.device, dtype=edge_features.dtype
        )
        velocities_flat = velocities.reshape(-1).to(
            edge_features.device, dtype=edge_features.dtype
        )
        edge_features_flat = edge_features.reshape(
            -1, edge_features.shape[-1]
        ).to(edge_features.device, dtype=edge_features.dtype)

        if group_sizes is None:
            group_sizes = [int(edge_features_flat.shape[0])]
        group_sizes = [int(size) for size in group_sizes if int(size) > 0]
        if sum(group_sizes) != int(edge_features_flat.shape[0]):
            group_sizes = [int(edge_features_flat.shape[0])]

        eps = float(self.velocity_dt_eps)
        sign_eps = float(self.velocity_sign_eps)
        outputs = []
        start = 0
        for size in group_sizes:
            end = start + int(size)
            feature_group = edge_features_flat[start:end]
            length_group = lengths_flat[start:end]
            velocity_group = velocities_flat[start:end]
            start = end

            if feature_group.numel() == 0:
                continue

            safe_rate = (-velocity_group).clamp_min(sign_eps)
            tau_pred = length_group.clamp_min(eps) / safe_rate
            pooled_features = feature_group.mean(dim=0)
            summary = torch.stack(
                [
                    torch.log(length_group.clamp_min(eps)).mean(),
                    torch.log(tau_pred.clamp_min(eps)).mean(),
                    torch.log(tau_pred.clamp_min(eps)).min(),
                    (velocity_group < -sign_eps).float().mean(),
                ],
                dim=0,
            )
            boundary_time_input = torch.cat([pooled_features, summary], dim=0)
            outputs.append(
                self.velocity_boundary_time_head(boundary_time_input).reshape(1)
            )

        if not outputs:
            return None
        return torch.cat(outputs, dim=0).reshape(-1)

    def _augment_terminal_head_input_with_case_feature(
        self,
        terminal_input,
        *,
        case_index=None,
    ):
        if terminal_input is None or not self.velocity_terminal_head_use_case_adapt:
            return terminal_input
        if case_index is None:
            raise ValueError(
                "case_index is required when velocity_terminal_head_use_case_adapt is enabled."
            )
        case_index_tensor = torch.as_tensor(
            case_index,
            dtype=torch.long,
            device=terminal_input.device,
        ).reshape(-1)
        case_features = self.model.first_hit_case_embedding(case_index_tensor).to(
            device=terminal_input.device,
            dtype=terminal_input.dtype,
        )
        if terminal_input.ndim == 1:
            if case_features.shape[0] != 1:
                raise ValueError(
                    "A single terminal input requires exactly one case index."
                )
            return torch.cat([terminal_input, case_features[0]], dim=0)
        if case_features.shape[0] == 1 and int(terminal_input.shape[0]) > 1:
            case_features = case_features.expand(int(terminal_input.shape[0]), -1)
        if case_features.shape[0] != int(terminal_input.shape[0]):
            raise ValueError(
                "Terminal case feature batch size must match terminal_input batch size."
            )
        return torch.cat([terminal_input, case_features], dim=-1)

    def _predict_terminal_stop_logit(
        self,
        lengths,
        velocities,
        *,
        time_value,
        first_hit_logits=None,
        boundary_vanish_logits=None,
        edge_features=None,
        aligned_model_masks=None,
        supervised_mask=None,
        case_index=None,
    ):
        if self.velocity_terminal_head is None:
            return None

        if self.velocity_terminal_head_input_mode == "probe":
            terminal_input = _build_probe_terminal_feature(
                lengths,
                velocities,
                first_hit_logits,
                boundary_vanish_logits,
            )
            terminal_input = self._augment_terminal_head_input_with_case_feature(
                terminal_input,
                case_index=case_index,
            )
            return self.velocity_terminal_head(terminal_input).reshape(())

        if self.velocity_terminal_head_input_mode == "topology_only":
            terminal_input = _build_topology_terminal_feature(
                self,
                aligned_model_masks,
                supervised_mask=supervised_mask,
                device=lengths.device,
                dtype=lengths.dtype,
            )
            terminal_input = self._augment_terminal_head_input_with_case_feature(
                terminal_input,
                case_index=case_index,
            )
            return self.velocity_terminal_head(terminal_input).reshape(())

        if self.velocity_terminal_head_input_mode == "edge_only":
            terminal_input = _build_edge_terminal_feature(
                edge_features,
                supervised_mask=supervised_mask,
                device=lengths.device,
                dtype=lengths.dtype,
            )
            if terminal_input is None:
                return None
            terminal_input = self._augment_terminal_head_input_with_case_feature(
                terminal_input,
                case_index=case_index,
            )
            return self.velocity_terminal_head(terminal_input).reshape(())

        if self.velocity_terminal_head_input_mode == "edge_topology":
            terminal_input = _build_edge_topology_terminal_feature(
                self,
                edge_features,
                aligned_model_masks,
                supervised_mask=supervised_mask,
                device=lengths.device,
                dtype=lengths.dtype,
            )
            if terminal_input is None:
                return None
            terminal_input = self._augment_terminal_head_input_with_case_feature(
                terminal_input,
                case_index=case_index,
            )
            return self.velocity_terminal_head(terminal_input).reshape(())

        if edge_features is None or edge_features.numel() == 0:
            return None

        lengths_flat = lengths.reshape(-1).to(
            edge_features.device, dtype=edge_features.dtype
        )
        velocities_flat = velocities.reshape(-1).to(
            edge_features.device, dtype=edge_features.dtype
        )
        edge_features_flat = edge_features.reshape(
            -1, edge_features.shape[-1]
        ).to(edge_features.device, dtype=edge_features.dtype)
        if edge_features_flat.numel() == 0:
            return None

        eps = float(self.velocity_dt_eps)
        sign_eps = float(self.velocity_sign_eps)
        safe_rate = (-velocities_flat).clamp_min(sign_eps)
        tau_pred = lengths_flat.clamp_min(eps) / safe_rate
        pooled_features = edge_features_flat.mean(dim=0)
        summary = torch.stack(
            [
                torch.log(lengths_flat.clamp_min(eps)).mean(),
                torch.log(tau_pred.clamp_min(eps)).mean(),
                torch.log(tau_pred.clamp_min(eps)).min(),
                (velocities_flat < -sign_eps).float().mean(),
                torch.as_tensor(
                    float(time_value),
                    device=edge_features.device,
                    dtype=edge_features.dtype,
                ),
            ],
            dim=0,
        )
        terminal_input = torch.cat([pooled_features, summary], dim=0)
        terminal_input = self._augment_terminal_head_input_with_case_feature(
            terminal_input,
            case_index=case_index,
        )
        return self.velocity_terminal_head(terminal_input).reshape(())

    def _effective_autoregressive_time_value(self, time_value):
        if not self.autoregressive_use_time:
            return 0.0
        return float(time_value)

    def _effective_autoregressive_time_tensor(self, time_value):
        if torch.is_tensor(time_value):
            if self.autoregressive_use_time:
                return time_value
            return torch.zeros_like(time_value, dtype=torch.float32, device=time_value.device)
        return torch.tensor(
            [self._effective_autoregressive_time_value(time_value)],
            dtype=torch.float32,
            device=self.device,
        )

    def _sampling_autoregressive_time_value(
        self,
        current_time,
        event_index=None,
        max_events=None,
    ):
        if not self.autoregressive_use_time:
            return 0.0
        if event_index is None or max_events is None:
            return float(current_time)
        max_events = int(max_events)
        if max_events <= 1:
            return 0.0
        clipped_index = min(max(int(event_index), 0), max_events - 1)
        return float(clipped_index / float(max_events - 1))

    def _sampling_autoregressive_time_tensor(
        self,
        current_time,
        event_index=None,
        max_events=None,
    ):
        return torch.tensor(
            [
                self._sampling_autoregressive_time_value(
                    current_time,
                    event_index=event_index,
                    max_events=max_events,
                )
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def _rollin_single_autoregressive_state(
        self,
        current_newick,
        target_newick,
        current_time,
        phyla_embedding=None,
    ):
        planned_merge = _plan_first_autoregressive_model_merge(
            self,
            current_newick=current_newick,
            current_time=current_time,
            phyla_embedding=phyla_embedding,
        )
        if planned_merge is None:
            return None

        rolled_newick = _apply_merge_subset_to_newick(
            self.model.tokenizer,
            current_newick,
            planned_merge["subset"],
            new_split=planned_merge["new_split"],
        )
        if rolled_newick is None:
            return None

        corrective_events = return_sampled_tree_boundary_decisions(
            rolled_newick,
            target_newick,
        )
        if not corrective_events:
            return None

        next_event = corrective_events[0]
        return {
            "newick": rolled_newick,
            "labels": next_event["labels"],
            "stop_after_merge": bool(next_event.get("stop_after_merge", False)),
            "time": self._effective_autoregressive_time_value(current_time),
        }

    def _dagger_rollin_single_autoregressive_state(
        self,
        current_newick,
        target_newick,
        current_time,
        phyla_embedding=None,
    ):
        oracle_training_topologies = set(
            _oracle_training_topology_keys(current_newick, target_newick)
        )
        state_newick = current_newick
        state_time = float(current_time)

        for rollout_step in range(self.autoregressive_dagger_max_steps):
            planned_merge = _plan_first_autoregressive_model_merge(
                self,
                current_newick=state_newick,
                current_time=state_time,
                phyla_embedding=phyla_embedding,
            )
            if planned_merge is None:
                return None

            next_newick = _apply_merge_subset_to_newick(
                self.model.tokenizer,
                state_newick,
                planned_merge["subset"],
                new_split=planned_merge["new_split"],
            )
            if next_newick is None or next_newick == state_newick:
                return None

            if _topology_key(next_newick) not in oracle_training_topologies:
                corrective_events = return_sampled_tree_boundary_decisions(
                    next_newick,
                    target_newick,
                )
                if not corrective_events:
                    return None

                next_event = corrective_events[0]
                return {
                    "newick": next_newick,
                    "labels": next_event["labels"],
                    "stop_after_merge": bool(next_event.get("stop_after_merge", False)),
                    "time": self._effective_autoregressive_time_value(state_time),
                    "rollout_steps": rollout_step + 1,
                }

            state_newick = next_newick

        return None

    def _perturb_autoregressive_single_state(
        self,
        current_newick,
        target_newick,
        current_time,
        phyla_embedding=None,
    ):
        if self.autoregressive_structure_perturb_mode == "model_wrong_pair":
            chosen_merge = _choose_model_wrong_pair_merge_subset(
                self,
                current_newick=current_newick,
                target_newick=target_newick,
                current_time=current_time,
                phyla_embedding=phyla_embedding,
            )
        else:
            subset = _choose_wrong_pair_merge_subset(
                current_newick,
                target_newick,
                self.model.tokenizer,
            )
            chosen_merge = None if subset is None else {"subset": subset, "new_split": None}

        if chosen_merge is None:
            return None

        perturbed_newick = _apply_merge_subset_to_newick(
            self.model.tokenizer,
            current_newick,
            chosen_merge["subset"],
            new_split=chosen_merge.get("new_split"),
        )
        if perturbed_newick is None:
            return None

        corrective_events = return_sampled_tree_boundary_decisions(
            perturbed_newick,
            target_newick,
        )
        if not corrective_events:
            return None

        next_event = corrective_events[0]
        return {
            "newick": perturbed_newick,
            "labels": next_event["labels"],
            "stop_after_merge": bool(next_event.get("stop_after_merge", False)),
            "time": self._effective_autoregressive_time_value(current_time),
        }

    def _attach_case_indices_to_batch(self, batch):
        if batch is None:
            return batch
        if getattr(self.model, "first_hit_head_mode", "base") not in {
            "case_adapted_mlp",
            "frozen_start_case_mlp",
        }:
            return batch
        if batch.get("_first_hit_case_indices") is not None:
            return batch
        group_keys = batch.get("bank_group_key")
        if group_keys is None:
            group_keys = _infer_case_group_keys_from_batch(self, batch)
        if group_keys is None:
            return batch
        if isinstance(group_keys, str):
            group_keys = [group_keys]
        case_index_tensor = _build_case_index_tensor_from_group_keys(
            group_keys,
            device=self.device,
        )
        if case_index_tensor is None:
            return batch
        updated_batch = dict(batch)
        updated_batch["bank_group_key"] = list(group_keys)
        updated_batch["_first_hit_case_indices"] = case_index_tensor
        return updated_batch

    def _attach_start_topology_features_to_batch(self, batch):
        if batch is None:
            return batch
        mode = getattr(self.model, "first_hit_head_mode", "base")
        first_hit_summary_modes = {
            "start_topology_adapter_mlp",
            "start_topology_raw_pool_concat_mlp",
        }
        needs_first_hit_summary = mode in first_hit_summary_modes
        needs_first_hit_cross_attn = mode == "start_topology_cross_attn_mlp"
        autoregressive_start_topology_mode = getattr(
            self.model,
            "autoregressive_start_topology_conditioning_mode",
            "additive",
        )
        needs_autoregressive_summary = bool(
            getattr(self.model, "autoregressive_use_start_topology_conditioning", False)
            and autoregressive_start_topology_mode
            not in {"frozen_case_probe", "frozen_case_probe_additive"}
        )
        if (
            not needs_first_hit_summary
            and not needs_first_hit_cross_attn
            and not needs_autoregressive_summary
        ):
            return batch
        summary_ready = (
            (
                not needs_first_hit_summary
                or batch.get("_first_hit_start_topology_features") is not None
            )
            and (
                not needs_autoregressive_summary
                or batch.get("_autoregressive_start_topology_features") is not None
            )
        )
        cross_ready = (
            not needs_first_hit_cross_attn
            or (
                batch.get("_first_hit_start_topology_embeddings") is not None
                and batch.get("_first_hit_start_topology_pad_mask") is not None
            )
        )
        if summary_ready and cross_ready:
            return batch
        start_trees = batch.get("start_trees")
        if start_trees is None:
            start_trees = batch.get("original_trees")
        if start_trees is None:
            return batch
        if isinstance(start_trees, str):
            start_trees = [start_trees]
        updated_batch = dict(batch)
        if needs_first_hit_summary or needs_autoregressive_summary:
            feature_tensor = batch.get("_first_hit_start_topology_features")
            if feature_tensor is None:
                feature_tensor = batch.get("_autoregressive_start_topology_features")
            if feature_tensor is None:
                feature_tensor = _build_start_topology_feature_tensor(
                    self,
                    list(start_trees),
                    device=self.device,
                )
            if feature_tensor is None:
                return batch
            if needs_first_hit_summary:
                updated_batch["_first_hit_start_topology_features"] = feature_tensor
            if needs_autoregressive_summary:
                updated_batch["_autoregressive_start_topology_features"] = feature_tensor
        if needs_first_hit_cross_attn:
            embeddings = batch.get("_first_hit_start_topology_embeddings")
            pad_mask = batch.get("_first_hit_start_topology_pad_mask")
            if embeddings is None or pad_mask is None:
                embeddings, pad_mask = _build_start_topology_identity_batch(
                    self,
                    list(start_trees),
                    device=self.device,
                )
            if embeddings is None or pad_mask is None:
                return batch
            updated_batch["_first_hit_start_topology_embeddings"] = embeddings
            updated_batch["_first_hit_start_topology_pad_mask"] = pad_mask
        return updated_batch

    def _attach_start_tree_graph_context_to_batch(self, batch):
        if batch is None:
            return batch
        if getattr(self.model, "first_hit_head_mode", "base") != "start_tree_graph_token_mlp":
            return batch
        if batch.get("_first_hit_start_tree_graph_context") is not None:
            return batch
        start_trees = batch.get("start_trees")
        if start_trees is None:
            start_trees = batch.get("original_trees")
        phyla_embeddings = batch.get("phyla_embeddings")
        if start_trees is None:
            return batch
        if isinstance(start_trees, str):
            start_trees = [start_trees]
        graph_context = _build_start_tree_graph_context(
            self,
            list(start_trees),
            phyla_embeddings,
            device=self.device,
            detach=getattr(self.model, "first_hit_start_tree_graph_detach", False),
        )
        if graph_context is None:
            return batch
        updated_batch = dict(batch)
        updated_batch["_first_hit_start_tree_graph_context"] = graph_context
        return updated_batch

    def _prepare_velocity_training_batch(self, batch):
        batch = self._attach_case_indices_to_batch(batch)
        batch = self._attach_start_topology_features_to_batch(batch)
        batch = self._attach_start_tree_graph_context_to_batch(batch)
        if batch.get("_skip_training_augmentations", False):
            return batch, {"attempted": 0.0, "applied": 0.0}
        if (
            self.velocity_length_jitter_prob <= 0.0
            or self.velocity_length_jitter_scale <= 0.0
            or "original_trees" not in batch
            or "target_trees" not in batch
        ):
            return batch, {"attempted": 0.0, "applied": 0.0}

        newicks = list(batch["original_trees"])
        velocity_labels = list(batch["batched_velocity"])
        batched_time = batch.get("batched_time")
        updated_times = batched_time.clone() if batched_time is not None else None
        attempted = 0
        applied = 0

        for batch_index, (current_newick, target_newick) in enumerate(
            zip(newicks, batch["target_trees"])
        ):
            if random.random() > self.velocity_length_jitter_prob:
                continue
            attempted += 1

            perturbed_newick = _jitter_internal_lengths_newick(
                current_newick,
                self.velocity_length_jitter_scale,
            )
            if perturbed_newick is None:
                continue

            try:
                sampled_newick, perturbed_velocity = return_sampled_tree_orthant_velocity(
                    perturbed_newick,
                    target_newick,
                    0.0,
                )
            except Exception:
                continue

            newicks[batch_index] = sampled_newick
            velocity_labels[batch_index] = perturbed_velocity
            if updated_times is not None:
                updated_times[batch_index] = 0.0
            applied += 1

        if applied == 0:
            return batch, {"attempted": float(attempted), "applied": 0.0}

        updated_batch = dict(batch)
        updated_batch["original_trees"] = newicks
        updated_batch["batched_velocity"] = velocity_labels
        if updated_times is not None:
            updated_batch["batched_time"] = updated_times
        updated_batch["tokenized_trees"] = _move_tokenized_batch_to_device(
            self.model.tokenizer(newicks),
            self.device,
        )
        return updated_batch, {"attempted": float(attempted), "applied": float(applied)}

    def _prepare_autoregressive_training_batch(self, batch):
        batch = self._attach_start_topology_features_to_batch(batch)
        needs_autoregressive_case_indices = bool(
            getattr(self.model, "autoregressive_use_case_conditioning", False)
        ) or (
            bool(
                getattr(
                    self.model,
                    "autoregressive_use_start_topology_conditioning",
                    False,
                )
            )
            and getattr(
                self.model,
                "autoregressive_start_topology_conditioning_mode",
                "additive",
            )
            in {"frozen_case_probe", "frozen_case_probe_additive"}
        )
        if needs_autoregressive_case_indices and batch.get("_autoregressive_case_indices") is None:
            group_keys = batch.get("bank_group_key")
            if group_keys is None:
                group_keys = _infer_case_group_keys_from_batch(self, batch)
            if group_keys is not None:
                if isinstance(group_keys, str):
                    group_keys = [group_keys]
                case_index_tensor = _build_case_index_tensor_from_group_keys(
                    group_keys,
                    device=self.device,
                )
                if case_index_tensor is not None:
                    batch = dict(batch)
                    batch["bank_group_key"] = list(group_keys)
                    batch["_autoregressive_case_indices"] = case_index_tensor
        if batch.get("_skip_training_augmentations", False):
            return batch, {
                "rollin_attempted": 0.0,
                "rollin_applied": 0.0,
                "dagger_attempted": 0.0,
                "dagger_applied": 0.0,
                "dagger_rollout_steps": 0.0,
                "structure_perturb_attempted": 0.0,
                "structure_perturb_applied": 0.0,
            }
        if (
            self.autoregressive_rollin_prob <= 0.0
            and self.autoregressive_dagger_prob <= 0.0
            and self.autoregressive_structure_perturb_prob <= 0.0
            or "target_trees" not in batch
            or "newick_autoregressive_trees" not in batch
        ):
            return batch, {
                "rollin_attempted": 0.0,
                "rollin_applied": 0.0,
                "dagger_attempted": 0.0,
                "dagger_applied": 0.0,
                "dagger_rollout_steps": 0.0,
                "structure_perturb_attempted": 0.0,
                "structure_perturb_applied": 0.0,
            }

        newicks = list(batch["newick_autoregressive_trees"])
        labels = list(batch["batched_autoregressive_labels"])
        times = batch["batched_autoregressive_time"].detach().clone()
        stop_after_merge = None
        if "batched_autoregressive_stop_after_merge" in batch:
            stop_after_merge = (
                batch["batched_autoregressive_stop_after_merge"].detach().clone()
            )

        rollin_attempted = 0
        rollin_applied = 0
        dagger_attempted = 0
        dagger_applied = 0
        dagger_rollout_steps = 0
        structure_attempted = 0
        structure_applied = 0
        for batch_index, (current_newick, target_newick) in enumerate(
            zip(newicks, batch["target_trees"])
        ):
            if self.autoregressive_rollin_prob > 0.0:
                if random.random() <= self.autoregressive_rollin_prob:
                    rollin_attempted += 1

                    phyla_embedding = None
                    if batch["phyla_embeddings"] is not None:
                        phyla_embedding = batch["phyla_embeddings"][batch_index : batch_index + 1]

                    rolled = self._rollin_single_autoregressive_state(
                        current_newick=current_newick,
                        target_newick=target_newick,
                        current_time=float(times[batch_index].item()),
                        phyla_embedding=phyla_embedding,
                    )
                    if rolled is not None:
                        current_newick = rolled["newick"]
                        newicks[batch_index] = rolled["newick"]
                        labels[batch_index] = rolled["labels"]
                        if stop_after_merge is not None:
                            stop_after_merge[batch_index] = (
                                1.0 if rolled.get("stop_after_merge", False) else 0.0
                            )
                        times[batch_index] = float(rolled["time"])
                        rollin_applied += 1

            if self.autoregressive_dagger_prob > 0.0:
                if random.random() <= self.autoregressive_dagger_prob:
                    dagger_attempted += 1

                    phyla_embedding = None
                    if batch["phyla_embeddings"] is not None:
                        phyla_embedding = batch["phyla_embeddings"][
                            batch_index : batch_index + 1
                        ]

                    dagger = self._dagger_rollin_single_autoregressive_state(
                        current_newick=current_newick,
                        target_newick=target_newick,
                        current_time=float(times[batch_index].item()),
                        phyla_embedding=phyla_embedding,
                    )
                    if dagger is not None:
                        current_newick = dagger["newick"]
                        newicks[batch_index] = dagger["newick"]
                        labels[batch_index] = dagger["labels"]
                        if stop_after_merge is not None:
                            stop_after_merge[batch_index] = (
                                1.0 if dagger.get("stop_after_merge", False) else 0.0
                            )
                        times[batch_index] = float(dagger["time"])
                        dagger_applied += 1
                        dagger_rollout_steps += int(dagger["rollout_steps"])

            if self.autoregressive_structure_perturb_prob > 0.0:
                if random.random() <= self.autoregressive_structure_perturb_prob:
                    structure_attempted += 1
                    phyla_embedding = None
                    if batch["phyla_embeddings"] is not None:
                        phyla_embedding = batch["phyla_embeddings"][batch_index : batch_index + 1]
                    perturbed = self._perturb_autoregressive_single_state(
                        current_newick=current_newick,
                        target_newick=target_newick,
                        current_time=float(times[batch_index].item()),
                        phyla_embedding=phyla_embedding,
                    )
                    if perturbed is not None:
                        newicks[batch_index] = perturbed["newick"]
                        labels[batch_index] = perturbed["labels"]
                        if stop_after_merge is not None:
                            stop_after_merge[batch_index] = (
                                1.0 if perturbed.get("stop_after_merge", False) else 0.0
                            )
                        times[batch_index] = float(perturbed["time"])
                        structure_applied += 1

        if rollin_applied == 0 and dagger_applied == 0 and structure_applied == 0:
            return batch, {
                "rollin_attempted": float(rollin_attempted),
                "rollin_applied": 0.0,
                "dagger_attempted": float(dagger_attempted),
                "dagger_applied": 0.0,
                "dagger_rollout_steps": float(dagger_rollout_steps),
                "structure_perturb_attempted": float(structure_attempted),
                "structure_perturb_applied": 0.0,
            }

        updated_batch = dict(batch)
        updated_batch["newick_autoregressive_trees"] = newicks
        updated_batch["batched_autoregressive_labels"] = labels
        updated_batch["batched_autoregressive_time"] = times
        if stop_after_merge is not None:
            updated_batch["batched_autoregressive_stop_after_merge"] = stop_after_merge
        updated_batch["tokenized_autoregressive_trees"] = _move_tokenized_batch_to_device(
            self.model.tokenizer(newicks),
            self.device,
        )
        return updated_batch, {
            "rollin_attempted": float(rollin_attempted),
            "rollin_applied": float(rollin_applied),
            "dagger_attempted": float(dagger_attempted),
            "dagger_applied": float(dagger_applied),
            "dagger_rollout_steps": float(dagger_rollout_steps),
            "structure_perturb_attempted": float(structure_attempted),
            "structure_perturb_applied": float(structure_applied),
        }

    def _append_sample_metrics_trace(self, metrics):
        if not self.sample_metrics_trace_path:
            return

        payload = {
            "global_step": int(self.global_step),
            "stepper": int(self.stepper),
            "timestamp": time.time(),
        }
        for key, value in metrics.items():
            if not (
                self._metric_key_allowed(key)
                or self._metric_key_allowed(f"sample_metrics/{key}")
            ):
                continue
            if torch.is_tensor(value):
                payload[key] = float(value.detach().cpu().item())
            elif isinstance(value, np.generic):
                payload[key] = float(value)
            elif isinstance(value, (int, float, bool)):
                payload[key] = float(value) if isinstance(value, bool) else value
            else:
                payload[key] = value

        os.makedirs(os.path.dirname(self.sample_metrics_trace_path), exist_ok=True)
        with open(self.sample_metrics_trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _sample_metrics_checkpoint_output_dir(self):
        if self.sample_metrics_checkpoint_dir:
            return self.sample_metrics_checkpoint_dir

        trainer = getattr(self, "trainer", None)
        if trainer is not None:
            checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
            checkpoint_dir = getattr(checkpoint_callback, "dirpath", None)
            if checkpoint_dir:
                return str(checkpoint_dir)
            for callback in getattr(trainer, "callbacks", []) or []:
                checkpoint_dir = getattr(callback, "dirpath", None)
                if checkpoint_dir:
                    return str(checkpoint_dir)

        if self.sample_metrics_trace_path:
            return os.path.join(
                os.path.dirname(os.path.abspath(self.sample_metrics_trace_path)),
                "sample_metrics_checkpoints",
            )
        return os.path.abspath("sample_metrics_checkpoints")

    def _save_sample_metrics_checkpoint(self):
        if not self.sample_metrics_checkpoint_enabled:
            return

        trainer = getattr(self, "trainer", None)
        if trainer is None or not hasattr(trainer, "save_checkpoint"):
            return
        if not getattr(trainer, "is_global_zero", True):
            return

        step = int(self.global_step)
        if step <= 0 or step in self._sample_metrics_checkpoint_steps:
            return

        out_dir = self._sample_metrics_checkpoint_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        step_fragment = f"step={step:06d}"
        for name in os.listdir(out_dir):
            if name.endswith(".ckpt") and step_fragment in name:
                self._sample_metrics_checkpoint_steps.add(step)
                return

        ckpt_path = os.path.join(
            out_dir,
            f"sample-metrics-epoch={int(self.current_epoch)}-step={step:06d}.ckpt",
        )
        trainer.save_checkpoint(ckpt_path)
        self._sample_metrics_checkpoint_steps.add(step)
        logging.info(
            "Saved sample-metrics checkpoint at global_step=%s to %s",
            step,
            ckpt_path,
        )

    def _sample_metrics_tree_dump_output_dir(self):
        if self.sample_metrics_tree_dump_dir:
            return self.sample_metrics_tree_dump_dir
        if self.sample_metrics_trace_path:
            return os.path.join(
                os.path.dirname(self.sample_metrics_trace_path),
                "generated_trees",
            )
        return os.path.abspath("sample_metrics_generated_trees")

    def _sample_metrics_can_write_artifacts(self):
        trainer = getattr(self, "trainer", None)
        if trainer is not None and not bool(getattr(trainer, "is_global_zero", True)):
            return False
        try:
            return int(os.environ.get("RANK", "0")) == 0
        except Exception:
            return True

    @staticmethod
    def _sample_metrics_newick_line(tree):
        tree = str(tree).strip()
        return tree if tree.endswith(";") else tree + ";"

    @staticmethod
    def _sample_metrics_json_value(value):
        if torch.is_tensor(value):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return value.detach().cpu().tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _write_sample_metrics_tree_dump(self, rows, relaxed_tree_rows=None, train=True):
        if not self.sample_metrics_tree_dump_enabled:
            return {}
        if not rows or not self._sample_metrics_can_write_artifacts():
            return {}

        relaxed_by_index = {
            int(row.get("sample_index", index)): dict(row)
            for index, row in enumerate(relaxed_tree_rows or [])
        }
        split = "train" if train else "val"
        step = int(self.global_step)
        stepper = int(self.stepper)
        stamp = f"step{step:08d}_stepper{stepper:08d}_{split}"
        label = getattr(self, "_sample_metrics_tree_dump_label", None)
        if label:
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("_")
            if safe_label:
                stamp = f"{stamp}_{safe_label}"
        out_dir = self._sample_metrics_tree_dump_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        jsonl_path = os.path.join(out_dir, f"{stamp}_trees.jsonl")
        sampled_path = os.path.join(out_dir, f"{stamp}_sampled_trees.txt")
        relaxed_path = os.path.join(out_dir, f"{stamp}_relaxed_trees.txt")

        payloads = []
        sampled_lines = []
        relaxed_lines = []
        base_payload = {
            "global_step": step,
            "stepper": stepper,
            "timestamp": time.time(),
            "split": split,
        }
        if label:
            base_payload["eval_label"] = str(label)
        for index, row in enumerate(rows):
            row = dict(row)
            payload = dict(base_payload)
            payload["index"] = int(index)
            field_map = {
                "_start_tree": "start_tree",
                "_original_start_tree": "original_start_tree",
                "_target_tree": "target_tree",
                "_sampled_tree": "sampled_tree",
                "_bank_group_key": "bank_group_key",
                "_source_bank_index": "source_bank_index",
                "_n_leaves": "n_leaves",
            }
            for private_key, public_key in field_map.items():
                if private_key in row and row[private_key] is not None:
                    payload[public_key] = self._sample_metrics_json_value(
                        row[private_key]
                    )
            for key, value in row.items():
                if str(key).startswith("_") or key in payload:
                    continue
                payload[key] = self._sample_metrics_json_value(value)
            relaxed_row = relaxed_by_index.get(index)
            if relaxed_row:
                for key, value in relaxed_row.items():
                    if key == "sample_index":
                        continue
                    payload[key] = self._sample_metrics_json_value(value)
            sampled_tree = payload.get("sampled_tree")
            if sampled_tree:
                sampled_lines.append(self._sample_metrics_newick_line(sampled_tree))
            relaxed_tree = payload.get("relaxed_tree")
            if relaxed_tree:
                relaxed_lines.append(self._sample_metrics_newick_line(relaxed_tree))
            payloads.append(payload)

        with open(jsonl_path, "w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        with open(sampled_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(sampled_lines) + ("\n" if sampled_lines else ""))
        if relaxed_lines:
            with open(relaxed_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(relaxed_lines) + "\n")

        metrics = {
            "tree_dump_written": 1.0,
            "tree_dump_count": float(len(payloads)),
        }
        if relaxed_lines:
            metrics["tree_dump_relaxed_count"] = float(len(relaxed_lines))
        return metrics

    def _get_harness_sampling_pair(self, train=True, frozen_start_bank=False):
        cache_key = "train" if train else "val"

        dataset_split = self._sample_metrics_dataset_split(train=train)
        use_random_fixed_pair_bank = (
            bool(train)
            and self.sampling_random_fixed_pair_bank_use_at_sampling
            and getattr(dataset_split, "overfit_fixed_pair", False)
        )
        cached = None if use_random_fixed_pair_bank else self._cached_harness_sampling_pairs.get(cache_key)
        if cached is not None:
            return cached

        if use_random_fixed_pair_bank:
            sampled_pair = None
            sampled_pair = self._sample_overfit_fixed_pair_bank_pair_for_harness(
                dataset_split,
                frozen_start_bank=frozen_start_bank,
            )
            if sampled_pair is not None:
                start_tree = sampled_pair["random_tree"]
                target_tree = sampled_pair["effective_target_tree"]
                return {
                    "start_tree": start_tree,
                    "target_tree": target_tree,
                    "bank_group_key": sampled_pair.get("bank_group_key")
                    or sampled_pair.get("group_key"),
                    "dataset_id": sampled_pair.get("dataset_id"),
                    "n_leaves": len(EteTree(start_tree, format=1).get_leaves()),
                    "max_events": int(len(sampled_pair.get("final_labels", []))),
                    "name_mapping": (
                        dataset_split.return_nexus_number_to_name(0)
                        if hasattr(dataset_split, "return_nexus_number_to_name")
                        else None
                    ),
                    "selected_sequences": sampled_pair.get("selected_sequences"),
                    "selected_sequence_names": sampled_pair.get(
                        "selected_sequence_names"
                    ),
                }

            sampled = dataset_split[0]
            start_tree = sampled.get("start_tree")
            target_tree = sampled.get("target_tree")
            if start_tree and target_tree:
                bank_group_key_value = sampled.get("bank_group_key") or sampled.get(
                    "group_key"
                )
                return {
                    "start_tree": start_tree,
                    "target_tree": target_tree,
                    "bank_group_key": bank_group_key_value,
                    "dataset_id": sampled.get("dataset_id"),
                    "n_leaves": len(EteTree(start_tree, format=1).get_leaves()),
                    "max_events": int(sampled.get("fixed_pair_num_events", 1024)),
                    "name_mapping": (
                        dataset_split.return_nexus_number_to_name(0)
                        if hasattr(dataset_split, "return_nexus_number_to_name")
                        else None
                    ),
                    "selected_sequences": sampled.get("selected_sequences"),
                    "selected_sequence_names": sampled.get(
                        "selected_sequence_names"
                    ),
                }

        fixed_pair = None
        if getattr(dataset_split, "overfit_fixed_pair", False):
            fixed_pair = dataset_split.get_overfit_fixed_pair(0)
        if fixed_pair is not None:
            start_tree = fixed_pair["random_tree"]
            target_tree = fixed_pair["effective_target_tree"]
            name_mapping = (
                dataset_split.return_nexus_number_to_name(0)
                if hasattr(dataset_split, "return_nexus_number_to_name")
                else fixed_pair.get("name_mapping")
            )
            explicit_max_events = max(
                int(getattr(dataset_split, "overfit_event_prefix_count", -1)),
                -1,
            )
            if explicit_max_events >= 0:
                max_events = explicit_max_events
            else:
                max_events = len(fixed_pair["final_labels"])
            pair = {
                "start_tree": start_tree,
                "target_tree": target_tree,
                "dataset_id": fixed_pair.get("dataset_id"),
                "n_leaves": len(EteTree(start_tree, format=1).get_leaves()),
                "max_events": int(max_events),
                "name_mapping": name_mapping,
                "selected_sequences": fixed_pair.get("selected_sequences"),
                "selected_sequence_names": fixed_pair.get(
                    "selected_sequence_names"
                ),
            }
            self._cached_harness_sampling_pairs[cache_key] = pair
            return pair

        random_state = random.getstate()
        try:
            random.seed(13)
            real_tree_raw = dataset_split.return_posterior_trees(0)[0]
            base_random_tree_raw, random_tree_raw = dataset_split.sample_random_tree_with_base(real_tree_raw)
            target_tree_raw = dataset_split.resolve_training_target_tree(
                random_tree_raw,
                real_tree_raw,
                base_start_tree_newick=base_random_tree_raw,
            )
            _, seq_ordering_map = _normalize_tree_like_dataset(real_tree_raw)
        finally:
            random.setstate(random_state)

        start_tree = _remap_tree_with_sequence_ordering(
            random_tree_raw,
            seq_ordering_map,
            offset=0,
            tree_kind="start tree",
        )
        target_tree = _remap_tree_with_sequence_ordering(
            target_tree_raw,
            seq_ordering_map,
            offset=0,
            tree_kind="target tree",
        )
        explicit_max_events = max(
            int(getattr(dataset_split, "overfit_event_prefix_count", -1)),
            -1,
        )
        if explicit_max_events >= 0:
            max_events = explicit_max_events
        else:
            boundary_paths = return_tree_boundary_merge_paths(start_tree, target_tree)
            max_events = int(
                sum(len(path.get("events", [])) for path in boundary_paths)
            )
        pair = {
            "start_tree": start_tree,
            "target_tree": target_tree,
            "n_leaves": len(EteTree(target_tree, format=1).get_leaves()),
            "max_events": max_events,
            "name_mapping": (
                dataset_split.return_nexus_number_to_name(0)
                if hasattr(dataset_split, "return_nexus_number_to_name")
                else None
            ),
        }
        self._cached_harness_sampling_pairs[cache_key] = pair
        return pair

    def _get_fixed_pair_sampling_details(self, train=True):
        if not hasattr(self, "dataset") or self.dataset is None:
            return None
        dataset_split = self._sample_metrics_dataset_split(train=train)
        if dataset_split is None or not getattr(dataset_split, "overfit_fixed_pair", False):
            return None
        pair = dataset_split.get_overfit_fixed_pair(0)
        if pair is None:
            return None
        if pair.get("name_mapping") is None and hasattr(dataset_split, "return_nexus_number_to_name"):
            try:
                pair = dict(pair)
                pair["name_mapping"] = dataset_split.return_nexus_number_to_name(0)
            except Exception:
                pass
        return pair

    def _evaluate_fixed_pair_velocity_rows(self, fixed_pair):
        boundary_paths = fixed_pair["boundary_paths"]
        effective_target_tree = fixed_pair["effective_target_tree"]
        velocity_trees = [fixed_pair["random_tree"]]
        velocity_trees.extend(path["end_newick"] for path in boundary_paths[:-1])
        timepoints = [0.0]
        timepoints.extend(float(path["global_time"]) for path in boundary_paths[:-1])
        next_boundary_trees = [path["start_newick"] for path in boundary_paths]
        pair = {
            "start_tree": fixed_pair["random_tree"],
            "target_tree": effective_target_tree,
            "bank_group_key": fixed_pair.get("bank_group_key")
            or fixed_pair.get("group_key"),
            "n_leaves": len(EteTree(fixed_pair["random_tree"], format=1).get_leaves()),
            "name_mapping": fixed_pair.get("name_mapping"),
            "selected_sequences": fixed_pair.get("selected_sequences"),
            "selected_sequence_names": fixed_pair.get("selected_sequence_names"),
            "max_events": len(fixed_pair.get("final_labels", []) or []),
        }
        sample_kwargs = self._build_harness_sample_kwargs(pair, train=True)
        phyla_embeddings = sample_kwargs.get("phyla_embeddings")
        case_indices = sample_kwargs.get("case_indices")
        if case_indices is not None:
            case_indices = torch.as_tensor(
                case_indices,
                dtype=torch.long,
                device=self.device,
            ).reshape(-1)
        start_topology_features = sample_kwargs.get(
            "first_hit_start_topology_features"
        )
        start_topology_embeddings = sample_kwargs.get(
            "first_hit_start_topology_embeddings"
        )
        start_topology_pad_mask = sample_kwargs.get(
            "first_hit_start_topology_pad_mask"
        )
        start_tree_graph_context = sample_kwargs.get(
            "first_hit_start_tree_graph_context"
        )

        rows = []
        for idx, (source_tree, next_boundary_tree, model_time) in enumerate(
            zip(velocity_trees, next_boundary_trees, timepoints)
        ):
            input_newick, true_velocity = return_sampled_tree_orthant_velocity(
                source_tree,
                effective_target_tree,
                0.0,
            )
            with torch.no_grad():
                tokenized = _move_tokenized_batch_to_device(
                    self.model.tokenizer([input_newick]),
                    self.device,
                )
                (
                    pred_velocity,
                    edge_splits,
                    _edge_mask,
                    first_hit_logits,
                    boundary_vanish_logits,
                    edge_features,
                ) = self.forward(
                    tokenized,
                    float(model_time),
                    phyla_embeddings,
                    first_hit_case_indices=case_indices,
                    first_hit_start_topology_features=start_topology_features,
                    first_hit_start_topology_embeddings=start_topology_embeddings,
                    first_hit_start_topology_pad_mask=start_topology_pad_mask,
                    first_hit_start_tree_graph_context=start_tree_graph_context,
                )

            model_masks = [int(mask) for mask in edge_splits[0]]
            mask_to_idx = {mask: i for i, mask in enumerate(model_masks)}
            target_vals = torch.zeros(len(model_masks), dtype=torch.float32)
            for split_mask, value in true_velocity.items():
                idx_match = mask_to_idx.get(int(split_mask))
                if idx_match is None:
                    continue
                target_vals[idx_match] = float(value)

            active_current = {
                int(mask)
                for mask in self.model.tokenizer([input_newick])[-1][0]
                if int(mask) != 0
            }
            active_next = {
                int(mask)
                for mask in self.model.tokenizer([next_boundary_tree])[-1][0]
                if int(mask) != 0
            }
            source_tree_obj = Tree(input_newick)
            source_masks, source_lengths = BHVEncoder().return_BHV_encoding(
                source_tree_obj
            )
            length_map = {
                int(mask): float(length)
                for mask, length in zip(source_masks, source_lengths)
                if length is not None
            }
            split_masks_nonzero = [mask for mask in model_masks if int(mask) != 0]
            real_max_bit = max(
                (int(mask).bit_length() for mask in split_masks_nonzero),
                default=0,
            )
            full_mask = (1 << real_max_bit) - 1 if real_max_bit > 0 else 0
            lengths = torch.zeros(len(model_masks), dtype=torch.float32)
            vanish_target = torch.zeros(len(model_masks), dtype=torch.float32)
            for idx_mask, mask in enumerate(model_masks):
                edge_len = length_map.get(int(mask))
                if edge_len is None and full_mask:
                    edge_len = length_map.get(full_mask ^ int(mask))
                if edge_len is not None and float(edge_len) > 0.0:
                    lengths[idx_mask] = float(edge_len)
                if int(mask) in active_current and int(mask) not in active_next:
                    vanish_target[idx_mask] = 1.0

            candidate_mask = _sampling_supervised_candidate_mask(
                model_masks,
                lengths.cpu().numpy(),
                pair["n_leaves"],
            )

            first_hit_stats = {
                "precision": 0.0,
                "recall": 0.0,
            }
            if first_hit_logits is not None:
                pred_first_mask, _raw_first_count, _used_first_fallback = (
                    _predict_first_hit_mask_with_fallback(
                        first_hit_logits[0].squeeze(1).detach().cpu().numpy(),
                        candidate_mask,
                        max_edges=getattr(
                            self, "velocity_first_hit_sampling_max_edges", -1
                        ),
                        fallback_threshold=getattr(
                            self,
                            "velocity_first_hit_sampling_fallback_threshold",
                            -1,
                        ),
                        fallback_top_k=getattr(
                            self,
                            "velocity_first_hit_sampling_fallback_top_k",
                            -1,
                        ),
                    )
                )
                true_first_mask = _oracle_first_hit_mask_for_sampling(
                    input_newick,
                    effective_target_tree,
                    masks=model_masks,
                    lengths=lengths.cpu().numpy(),
                    n_leaves=pair["n_leaves"],
                    supervised_mask=candidate_mask,
                    velocity_sign_eps=float(self.velocity_sign_eps),
                    dt_eps=float(self.velocity_dt_eps),
                    first_hit_tol=1e-4,
                )
                first_hit_stats = _mask_precision_recall(
                    pred_first_mask,
                    true_first_mask,
                )

            vanish_stats = {
                "precision": 0.0,
                "recall": 0.0,
            }
            if boundary_vanish_logits is not None:
                pred_vanish_mask = _predict_boundary_vanish_mask_from_logits(
                    boundary_vanish_logits[0].squeeze(1).detach().cpu().numpy(),
                    candidate_mask,
                )
                true_vanish_mask = _oracle_boundary_vanish_mask_for_sampling(
                    input_newick,
                    effective_target_tree,
                    masks=model_masks,
                    n_leaves=pair["n_leaves"],
                    candidate_mask=candidate_mask,
                )
                vanish_stats = _mask_precision_recall(
                    pred_vanish_mask,
                    true_vanish_mask,
                )

            rows.append(
                {
                    "index": int(idx),
                    "timepoint": float(model_time),
                    "first_hit_precision": float(first_hit_stats["precision"]),
                    "first_hit_recall": float(first_hit_stats["recall"]),
                    "vanish_precision": float(vanish_stats["precision"]),
                    "vanish_recall": float(vanish_stats["recall"]),
                }
            )
        return rows

    def _labels_to_subset_tuples(self, labels):
        subsets = set()
        for label in labels:
            components = [int(component) for component in label["components"]]
            subset = tuple(
                sorted(int(components[idx]) for idx in label["merge_indices"])
            )
            subsets.add(subset)
        return subsets

    def _evaluate_fixed_pair_autoregressive_rows(self, fixed_pair):
        rows = []
        final_labels = fixed_pair["final_labels"]
        max_events = len(final_labels)
        pair = {
            "start_tree": fixed_pair["random_tree"],
            "target_tree": fixed_pair["effective_target_tree"],
            "bank_group_key": fixed_pair.get("bank_group_key")
            or fixed_pair.get("group_key"),
            "n_leaves": len(EteTree(fixed_pair["random_tree"], format=1).get_leaves()),
            "name_mapping": fixed_pair.get("name_mapping"),
            "selected_sequences": fixed_pair.get("selected_sequences"),
            "selected_sequence_names": fixed_pair.get("selected_sequence_names"),
            "max_events": len(final_labels),
        }
        sample_kwargs = self._build_harness_sample_kwargs(pair, train=True)
        phyla_embeddings = sample_kwargs.get("phyla_embeddings")
        case_indices = sample_kwargs.get("case_indices")
        if case_indices is not None:
            case_indices = torch.as_tensor(
                case_indices,
                dtype=torch.long,
                device=self.device,
            ).reshape(-1)
        for event_idx, event in enumerate(final_labels):
            current_newick = event["newick"]
            component_groups = [get_structural_polytomy_groups_from_newick(current_newick)]
            with torch.no_grad():
                tokenized = _move_tokenized_batch_to_device(
                    self.model.tokenizer([current_newick]),
                    self.device,
                )
                outputs = self.forward(
                    tokenized,
                    self._sampling_autoregressive_time_tensor(
                        0.0,
                        event_index=event_idx,
                        max_events=max_events,
                    ),
                    phyla_embeddings,
                    autoregressive=True,
                    autoregressive_component_groups=component_groups,
                    autoregressive_case_indices=case_indices,
                    autoregressive_start_topology_features=sample_kwargs.get(
                        "autoregressive_start_topology_features"
                    ),
                )
            existing_splits = {
                int(mask)
                for mask in self.model.tokenizer([current_newick])[-1][0]
                if int(mask) != 0
            }
            planned = _plan_autoregressive_boundary_merges(outputs, existing_splits)
            pred_subsets = set()
            if planned:
                for subset, _new_split in planned[0]["subsets"]:
                    pred_subsets.add(
                        tuple(sorted(int(component) for component in subset))
                    )
            true_subsets = self._labels_to_subset_tuples(event["labels"])
            rows.append(
                {
                    "event_index": int(event_idx),
                    "exact_match": pred_subsets == true_subsets,
                }
            )
        return rows

    def _evaluate_fixed_pair_path_metrics(self, train=True):
        fixed_pair = self._get_fixed_pair_sampling_details(train=train)
        if fixed_pair is None:
            return {}

        velocity_rows = self._evaluate_fixed_pair_velocity_rows(fixed_pair)
        autoregressive_rows = self._evaluate_fixed_pair_autoregressive_rows(fixed_pair)
        metrics = _summarize_fixed_pair_eval_rows(velocity_rows, autoregressive_rows)

        pair = {
            "start_tree": fixed_pair["random_tree"],
            "target_tree": fixed_pair["effective_target_tree"],
            "bank_group_key": fixed_pair.get("bank_group_key")
            or fixed_pair.get("group_key"),
            "n_leaves": len(EteTree(fixed_pair["random_tree"], format=1).get_leaves()),
            "name_mapping": fixed_pair.get("name_mapping"),
            "selected_sequences": fixed_pair.get("selected_sequences"),
            "selected_sequence_names": fixed_pair.get("selected_sequence_names"),
            "max_events": len(fixed_pair.get("final_labels", []) or []),
        }
        sample_kwargs = self._build_harness_sample_kwargs(pair, train=train)
        sample_kwargs.pop("return_trace", None)
        with torch.no_grad():
            sampled_trees, _, _, _, _, trace = self.sample(
                [pair["start_tree"]],
                return_trace=True,
                **sample_kwargs,
            )
        sampled_tree = sampled_trees[0]
        sampled_velocity = len(trace.get("velocity", []))
        sampled_ar = len(trace.get("autoregressive", []))
        expected_velocity = len(fixed_pair.get("boundary_paths", []) or [])
        expected_ar = len(fixed_pair.get("final_labels", []) or [])
        sampled_decoder_metrics = self._trace_topology_decoder_metrics(
            trace,
            prefix="fixed_path_sampled_",
        )
        metrics.update(
            {
                "fixed_path_sample_rf_norm": float(
                    calculate_norm_rf(sampled_tree, pair["target_tree"])
                ),
                "fixed_path_sampled_num_velocity_states": float(sampled_velocity),
                "fixed_path_sampled_num_autoregressive_events": float(sampled_ar),
                "fixed_path_expected_num_velocity_states": float(expected_velocity),
                "fixed_path_expected_num_autoregressive_events": float(expected_ar),
                "fixed_path_extra_velocity_states": float(
                    max(0, sampled_velocity - expected_velocity)
                ),
                "fixed_path_extra_autoregressive_events": float(
                    max(0, sampled_ar - expected_ar)
                ),
                "fixed_path_missing_velocity_states": float(
                    max(0, expected_velocity - sampled_velocity)
                ),
                "fixed_path_missing_autoregressive_events": float(
                    max(0, expected_ar - sampled_ar)
                ),
                "fixed_path_stopped_for_no_valid_merge": float(
                    1.0 if trace.get("stopped_for_no_valid_merge", False) else 0.0
                ),
                "fixed_path_stopped_for_repeated_topology": float(
                    1.0
                    if trace.get("stopped_for_repeated_topology", False)
                    else 0.0
                ),
            }
        )
        metrics.update(sampled_decoder_metrics)
        return metrics

    def _build_harness_sample_kwargs(
        self,
        pair,
        train=True,
        rollout_kind: str = "probe",
        **overrides,
    ):
        if self.use_historical_sampling_impl:
            return _call_historical_trainingmodule_method(
                "_build_harness_sample_kwargs",
                self,
                pair,
                train=train,
                rollout_kind=rollout_kind,
                **overrides,
            )
        phyla_embeddings = None
        if self.live_phyla_model is not None:
            phyla_embeddings = self._compute_live_phyla_embeddings_for_pair(pair)
        if phyla_embeddings is None:
            phyla_embeddings = self._resolve_precomputed_phyla_embeddings_for_tree(
                pair["start_tree"],
                mapping=pair.get("name_mapping"),
                num_leaf=pair.get("n_leaves"),
                device=self.device,
                dataset_id=pair.get("dataset_id"),
            )
        dataset_obj = getattr(self, "dataset", None)
        dataset_split = None
        if dataset_obj is not None:
            dataset_split = (
                getattr(dataset_obj, "dataset_train", None)
                if train
                else getattr(dataset_obj, "dataset_val", None)
            )
        split_multi_label_events = bool(
            getattr(dataset_split, "overfit_split_multi_subset_events", False)
        )
        if rollout_kind == "replay":
            fixed_dt_base = self.rollout_replay_fixed_dt_base
            max_steps = self.rollout_replay_max_steps
            uncapped_events = self.rollout_replay_max_events_uncapped
            max_events = (
                None if uncapped_events else self.rollout_replay_max_events
            )
        else:
            fixed_dt_base = self.sampling_fixed_dt_base
            max_steps = self.sampling_max_steps
            uncapped_events = self.sampling_max_events_uncapped
            max_events = None if uncapped_events else self.sampling_max_events
        if max_events is None and not uncapped_events:
            if int(pair.get("max_events", -1)) >= 0:
                max_events = int(pair["max_events"])
            else:
                max_events = 1024
        sample_kwargs = {
            "phyla_embeddings": phyla_embeddings,
            "num_samples": 1,
            "T": 1.0,
            "dt_base": (
                float(fixed_dt_base)
                if fixed_dt_base is not None
                else self.training_sampling_dt_base
            ),
            "fixed_dt_sampling": bool(fixed_dt_base is not None),
            # Keep replay rollouts and probe sampling aligned so their RF metrics are comparable.
            "max_steps": max_steps,
            "max_events": max_events,
            "max_autoregressive_merges_per_boundary": int(
                self.sampling_max_autoregressive_merges_per_boundary
            ),
            "return_trace": True,
            "trace_state_rf": bool(
                getattr(self, "sample_metrics_trace_state_rf_enabled", False)
            ),
            "explicit_autoregressive_component_groups": bool(
                getattr(
                    self,
                    "sample_metrics_explicit_autoregressive_component_groups_enabled",
                    True,
                )
            ),
            "target_trees": [pair["target_tree"]],
            "split_multi_label_events": split_multi_label_events,
        }
        needs_frozen_ar_case_probe = (
            bool(
                getattr(
                    self.model,
                    "autoregressive_use_start_topology_conditioning",
                    False,
                )
            )
            and getattr(
                self.model,
                "autoregressive_start_topology_conditioning_mode",
                "additive",
            )
            in {"frozen_case_probe", "frozen_case_probe_additive"}
        )
        if (
            getattr(self.model, "first_hit_head_mode", "base")
            in {"case_adapted_mlp", "frozen_start_case_mlp"}
            or getattr(self.model, "autoregressive_use_case_conditioning", False)
            or needs_frozen_ar_case_probe
        ):
            case_index = _extract_case_index_from_group_key(
                _case_index_group_key_for_pair(self, pair)
            )
            if case_index is not None:
                sample_kwargs["case_indices"] = [int(case_index)]
        needs_start_topology_summary = (
            getattr(self.model, "first_hit_head_mode", "base")
            in {
                "start_topology_adapter_mlp",
                "start_topology_raw_pool_concat_mlp",
            }
            or getattr(
                self.model,
                "autoregressive_use_start_topology_conditioning",
                False,
            )
            and not needs_frozen_ar_case_probe
        )
        if needs_start_topology_summary:
            start_topology_features = _build_start_topology_feature_tensor(
                self,
                [pair["start_tree"]],
                device=self.device,
            )
            if (
                getattr(self.model, "first_hit_head_mode", "base")
                in {
                    "start_topology_adapter_mlp",
                    "start_topology_raw_pool_concat_mlp",
                }
            ):
                sample_kwargs["first_hit_start_topology_features"] = (
                    start_topology_features
                )
            if getattr(
                self.model,
                "autoregressive_use_start_topology_conditioning",
                False,
            ):
                sample_kwargs["autoregressive_start_topology_features"] = (
                    start_topology_features
                )
        if (
            getattr(self.model, "first_hit_head_mode", "base")
            == "start_topology_cross_attn_mlp"
        ):
            embeddings, pad_mask = _build_start_topology_identity_batch(
                self,
                [pair["start_tree"]],
                device=self.device,
            )
            sample_kwargs["first_hit_start_topology_embeddings"] = embeddings
            sample_kwargs["first_hit_start_topology_pad_mask"] = pad_mask
        if (
            getattr(self.model, "first_hit_head_mode", "base")
            == "start_tree_graph_token_mlp"
        ):
            sample_kwargs["first_hit_start_tree_graph_context"] = (
                _build_start_tree_graph_context(
                    self,
                    [pair["start_tree"]],
                    phyla_embeddings,
                    device=self.device,
                    detach=getattr(
                        self.model, "first_hit_start_tree_graph_detach", False
                    ),
                )
            )
        sample_kwargs.update(overrides)
        return sample_kwargs

    def _build_harness_lexicographic_ordering_map(self, reference_tree_newick):
        tree = EteTree(reference_tree_newick, format=1)
        leaves = list(tree.get_leaves())
        leaves.sort(key=lambda leaf: str(leaf.name))
        return {str(leaf.name): str(i) for i, leaf in enumerate(leaves)}

    def _build_numeric_to_harness_lexicographic_ordering_map(
        self,
        reference_tree_newick,
    ):
        tree = EteTree(reference_tree_newick, format=1)
        original_names = [str(leaf.name) for leaf in tree.get_leaves()]
        numeric_sorted = sorted(original_names, key=lambda name: int(str(name)))
        lex_sorted = sorted(original_names, key=lambda name: str(name))
        original_to_numeric = {str(name): str(i) for i, name in enumerate(numeric_sorted)}
        original_to_lex = {str(name): str(i) for i, name in enumerate(lex_sorted)}
        return {
            original_to_numeric[str(name)]: original_to_lex[str(name)]
            for name in original_names
        }

    def _remap_tree_with_ordering_map(self, tree_newick, ordering_map):
        tree = EteTree(tree_newick, format=1)
        for leaf in tree.get_leaves():
            original_name = str(leaf.name)
            mapped_name = ordering_map.get(original_name)
            if mapped_name is None:
                raise KeyError(
                    f"Leaf {original_name!r} is missing from the harness ordering map."
                )
            leaf.name = str(mapped_name)
        return tree.write(format=1)

    def _infer_golden_posterior_root(self, short_root):
        if not short_root:
            return None
        short_root = str(short_root)
        if "short_run_data_DS1-8" in short_root:
            return short_root.replace("short_run_data_DS1-8", "golden_run_data_DS1-8")
        base_name = os.path.basename(short_root.rstrip("/"))
        if base_name.startswith("short_run_data_"):
            return os.path.join(
                os.path.dirname(short_root.rstrip("/")),
                base_name.replace("short_run_data_", "golden_run_data_", 1),
            )
        return None

    def _canonical_tree_topology_counts(self, trees):
        counts = Counter()
        for tree in trees:
            try:
                counts[canonicalize_topology_newick(str(tree))] += 1
            except Exception:
                continue
        return counts

    def _split_topology_counts(self, trees):
        counts = Counter()
        encoder = BHVEncoder()
        for tree in trees:
            try:
                masks, _lengths = encoder.return_BHV_encoding(Tree(str(tree)))
                counts.update(int(mask) for mask in masks)
            except Exception:
                continue
        return counts

    @staticmethod
    def _json_counter(counter):
        return {str(key): int(value) for key, value in Counter(counter).items()}

    @staticmethod
    def _counter_from_json(value, *, integer_keys=False):
        counts = Counter()
        if not isinstance(value, dict):
            return counts
        for key, count in value.items():
            try:
                parsed_key = int(key) if integer_keys else str(key)
                counts[parsed_key] = int(count)
            except Exception:
                continue
        return counts

    def _posterior_reference_cache_dir(self):
        if self.sample_metrics_trace_path:
            base_dir = os.path.dirname(os.path.abspath(self.sample_metrics_trace_path))
        elif self.sample_metrics_tree_dump_dir:
            base_dir = os.path.dirname(os.path.abspath(self.sample_metrics_tree_dump_dir))
        elif self.sample_metrics_mrbayes20k_output_dir:
            base_dir = os.path.dirname(
                os.path.abspath(self.sample_metrics_mrbayes20k_output_dir)
            )
        else:
            base_dir = os.path.abspath("metrics")
        return os.path.join(base_dir, "posterior_reference_cache")

    def _posterior_reference_cache_path(
        self,
        posterior_root,
        golden_root,
        dataset_id,
        trprobs_sample_count,
    ):
        payload = json.dumps(
            {
                "version": 1,
                "posterior_root": str(posterior_root),
                "golden_root": str(golden_root),
                "dataset_id": str(dataset_id),
                "trprobs_sample_count": int(trprobs_sample_count),
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        safe_dataset_id = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_"
            for ch in str(dataset_id)
        )
        return os.path.join(
            self._posterior_reference_cache_dir(),
            f"{safe_dataset_id}_posterior_reference_v1_{digest}.json",
        )

    def _posterior_reference_bundle_from_payload(self, payload):
        if not isinstance(payload, dict) or int(payload.get("version", 0)) != 1:
            return None
        short_block = dict(payload.get("short") or {})
        golden_block = dict(payload.get("golden") or {})
        return {
            "dataset_id": str(payload.get("dataset_id")),
            "short_root": payload.get("short_root"),
            "golden_root": payload.get("golden_root"),
            "posterior_raw_to_lex": {
                str(key): str(value)
                for key, value in dict(
                    payload.get("posterior_raw_to_lex") or {}
                ).items()
            },
            "numeric_to_lex": {
                str(key): str(value)
                for key, value in dict(payload.get("numeric_to_lex") or {}).items()
            },
            "num_leaves": int(payload.get("num_leaves", 0)),
            "short_counts": self._counter_from_json(
                short_block.get("topology_counts")
            ),
            "golden_counts": self._counter_from_json(
                golden_block.get("topology_counts")
            ),
            "short_split_counts": self._counter_from_json(
                short_block.get("split_counts"),
                integer_keys=True,
            ),
            "golden_split_counts": self._counter_from_json(
                golden_block.get("split_counts"),
                integer_keys=True,
            ),
            "short_tree_total": int(short_block.get("tree_total", 0)),
            "golden_tree_total": int(golden_block.get("tree_total", 0)),
            "short_split_total": int(short_block.get("split_total", 0)),
            "golden_split_total": int(golden_block.get("split_total", 0)),
            "cache_path": payload.get("cache_path"),
        }

    def _posterior_reference_bundle_to_payload(
        self,
        bundle,
        *,
        posterior_root,
        golden_root,
        dataset_id,
        trprobs_sample_count,
        cache_path,
    ):
        short_split_counts = Counter(bundle.get("short_split_counts") or {})
        golden_split_counts = Counter(bundle.get("golden_split_counts") or {})
        return {
            "version": 1,
            "dataset_id": str(dataset_id),
            "short_root": str(posterior_root),
            "golden_root": str(golden_root) if golden_root else None,
            "trprobs_sample_count": int(trprobs_sample_count),
            "num_leaves": int(bundle.get("num_leaves", 0)),
            "posterior_raw_to_lex": dict(bundle.get("posterior_raw_to_lex") or {}),
            "numeric_to_lex": dict(bundle.get("numeric_to_lex") or {}),
            "short": {
                "tree_total": int(bundle.get("short_tree_total", 0)),
                "topology_counts": self._json_counter(
                    bundle.get("short_counts") or {}
                ),
                "split_total": int(sum(short_split_counts.values())),
                "split_counts": self._json_counter(short_split_counts),
            },
            "golden": {
                "tree_total": int(bundle.get("golden_tree_total", 0)),
                "topology_counts": self._json_counter(
                    bundle.get("golden_counts") or {}
                ),
                "split_total": int(sum(golden_split_counts.values())),
                "split_counts": self._json_counter(golden_split_counts),
            },
            "cache_path": str(cache_path),
        }

    @staticmethod
    def _kl_from_topology_counts(
        posterior_counts,
        sampled_counts,
        *,
        alpha=1e-6,
    ):
        posterior_counts = Counter(posterior_counts or {})
        sampled_counts = Counter(sampled_counts or {})
        support = set(posterior_counts.keys()).union(sampled_counts.keys())
        if not support:
            return {
                "kl_divergence_tree_topology": 0.0,
                "n_unique_posterior_topologies": 0.0,
                "n_unique_sampled_topologies": 0.0,
                "n_shared_topologies": 0.0,
                "posterior_topology_support_recall": 1.0,
            }

        posterior_total = float(sum(posterior_counts.values()))
        sampled_total = float(sum(sampled_counts.values()))
        zp = posterior_total + float(alpha) * len(support)
        zq = sampled_total + float(alpha) * len(support)
        kl = 0.0
        for key in support:
            p = (float(posterior_counts.get(key, 0.0)) + float(alpha)) / zp
            q = (float(sampled_counts.get(key, 0.0)) + float(alpha)) / zq
            kl += p * math.log(p / q)

        shared = len(set(posterior_counts.keys()).intersection(sampled_counts.keys()))
        unique_posterior = len(posterior_counts)
        return {
            "kl_divergence_tree_topology": float(kl),
            "n_unique_posterior_topologies": float(unique_posterior),
            "n_unique_sampled_topologies": float(len(sampled_counts)),
            "n_shared_topologies": float(shared),
            "posterior_topology_support_recall": (
                float(shared) / float(unique_posterior)
                if unique_posterior
                else 1.0
            ),
        }

    @staticmethod
    def _kl_from_split_counts(
        posterior_split_counts,
        sampled_split_counts,
        *,
        alpha=1e-6,
    ):
        posterior_split_counts = Counter(posterior_split_counts or {})
        sampled_split_counts = Counter(sampled_split_counts or {})
        posterior_total = float(sum(posterior_split_counts.values()))
        sampled_total = float(sum(sampled_split_counts.values()))
        if posterior_total <= 0.0 or sampled_total <= 0.0:
            return {"kl_divergence_topological": 0.0}
        posterior_distribution = {
            key: float(value) / posterior_total
            for key, value in posterior_split_counts.items()
        }
        sampled_distribution = {
            key: float(value) / sampled_total
            for key, value in sampled_split_counts.items()
        }
        support = set(posterior_distribution.keys()).union(
            sampled_distribution.keys()
        )
        if not support:
            return {"kl_divergence_topological": 0.0}
        zp = 1.0 + float(alpha) * len(support)
        zq = 1.0 + float(alpha) * len(support)
        kl = 0.0
        for key in support:
            p = (posterior_distribution.get(key, 0.0) + float(alpha)) / zp
            q = (sampled_distribution.get(key, 0.0) + float(alpha)) / zq
            kl += p * math.log(p / q)
        return {"kl_divergence_topological": float(kl)}

    @staticmethod
    def _topk_recall_from_topology_counts(
        posterior_counts,
        sampled_counts,
        top_ks=(1, 5, 10, 20, 50),
    ):
        posterior_counts = Counter(posterior_counts or {})
        sampled_support = set(Counter(sampled_counts or {}).keys())
        if not posterior_counts:
            return {}
        ranked = sorted(
            posterior_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        posterior_total = float(sum(posterior_counts.values()))
        metrics = {}
        for raw_k in top_ks:
            k = max(1, int(raw_k))
            top_items = ranked[: min(k, len(ranked))]
            if not top_items:
                metrics[f"posterior_topology_recall_at_{k}"] = 1.0
                metrics[f"posterior_topology_mass_recall_at_{k}"] = 1.0
                continue
            hits = [key for key, _ in top_items if key in sampled_support]
            top_mass = sum(count for _, count in top_items)
            hit_mass = sum(posterior_counts[key] for key in hits)
            metrics[f"posterior_topology_recall_at_{k}"] = float(len(hits)) / float(
                len(top_items)
            )
            metrics[f"posterior_topology_mass_recall_at_{k}"] = (
                float(hit_mass) / float(top_mass) if top_mass > 0 else 1.0
            )
        metrics["posterior_topology_sample_support_size"] = float(len(sampled_support))
        metrics["posterior_topology_posterior_support_size"] = float(
            len(posterior_counts)
        )
        metrics["posterior_topology_total_mass"] = float(posterior_total)
        return metrics

    def _support_rate_from_topology_counts(self, sampled_counts, support_counts):
        sampled_counts = Counter(sampled_counts or {})
        if not sampled_counts:
            return float("nan")
        if not support_counts:
            return 0.0
        total = float(sum(sampled_counts.values()))
        if total <= 0.0:
            return float("nan")
        support_keys = set(Counter(support_counts).keys())
        hits = sum(
            int(count)
            for key, count in sampled_counts.items()
            if key in support_keys
        )
        return float(hits) / total

    def _sampled_topology_distribution_metrics_from_counts(
        self,
        counts,
        reference_support_size=None,
    ):
        counts = Counter(counts or {})
        total = float(sum(counts.values()))
        if total <= 0:
            return {
                "sampled_topology_unique_count": 0.0,
                "sampled_topology_mode_mass": float("nan"),
                "sampled_topology_entropy": float("nan"),
                "sampled_topology_entropy_normalized": float("nan"),
            }
        probs = [float(count) / total for count in counts.values()]
        entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
        norm_base = int(reference_support_size or 0)
        if norm_base <= 1:
            entropy_normalized = 0.0
        else:
            entropy_normalized = float(entropy) / float(math.log(norm_base))
        return {
            "sampled_topology_unique_count": float(len(counts)),
            "sampled_topology_mode_mass": float(max(probs)),
            "sampled_topology_entropy": float(entropy),
            "sampled_topology_entropy_normalized": float(entropy_normalized),
        }

    def _support_rate(self, sampled_trees, support_counts):
        if not sampled_trees:
            return float("nan")
        if not support_counts:
            return 0.0
        support_keys = set(support_counts.keys())
        hits = 0
        total = 0
        for tree in sampled_trees:
            try:
                key = canonicalize_topology_newick(str(tree))
            except Exception:
                continue
            total += 1
            if key in support_keys:
                hits += 1
        if total <= 0:
            return float("nan")
        return float(hits) / float(total)

    def _sampled_topology_distribution_metrics(self, sampled_trees, reference_support_size=None):
        counts = self._canonical_tree_topology_counts(sampled_trees)
        total = float(sum(counts.values()))
        if total <= 0:
            return {
                "sampled_topology_unique_count": 0.0,
                "sampled_topology_mode_mass": float("nan"),
                "sampled_topology_entropy": float("nan"),
                "sampled_topology_entropy_normalized": float("nan"),
            }
        probs = [float(count) / total for count in counts.values()]
        entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
        norm_base = int(reference_support_size or 0)
        if norm_base <= 1:
            entropy_normalized = 0.0
        else:
            entropy_normalized = float(entropy) / float(math.log(norm_base))
        return {
            "sampled_topology_unique_count": float(len(counts)),
            "sampled_topology_mode_mass": float(max(probs)),
            "sampled_topology_entropy": float(entropy),
            "sampled_topology_entropy_normalized": float(entropy_normalized),
        }

    def _prefix_metric_block(self, metrics, prefix):
        return {f"{prefix}_{key}": value for key, value in metrics.items()}

    def _sample_metrics_dataset_split(self, train=True):
        if self.dataset is None:
            return None
        override_name = (
            "sample_metrics_dataset_train" if train else "sample_metrics_dataset_val"
        )
        override_split = getattr(self.dataset, override_name, None)
        if override_split is not None:
            return override_split
        return self.dataset.dataset_train if train else self.dataset.dataset_val

    def _load_posterior_reference_bundle(self, train=True):
        dataset_split = self._sample_metrics_dataset_split(train=train)
        if dataset_split is None:
            return None
        posterior_root = getattr(dataset_split, "posterior_trprobs_root", None)
        posterior_ids = list(getattr(dataset_split, "posterior_dataset_ids", []) or [])
        if not posterior_root or len(posterior_ids) != 1:
            return None
        dataset_id = str(posterior_ids[0])
        trprobs_sample_count = int(
            getattr(dataset_split, "trprobs_sample_count_per_file", 1000)
        )
        golden_root = self._infer_golden_posterior_root(posterior_root)
        cache_key = (
            str(posterior_root),
            str(golden_root),
            dataset_id,
            int(trprobs_sample_count),
        )
        cache = getattr(self, "_posterior_reference_bundle_cache", None)
        if cache is None:
            cache = {}
            self._posterior_reference_bundle_cache = cache
        if cache_key in cache:
            return cache[cache_key]

        cache_path = self._posterior_reference_cache_path(
            posterior_root,
            golden_root,
            dataset_id,
            trprobs_sample_count,
        )
        if os.path.exists(cache_path) and os.environ.get(
            "PHYLAFLOW_POSTERIOR_REFERENCE_CACHE_DISABLE", "0"
        ) != "1":
            try:
                with open(cache_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                bundle = self._posterior_reference_bundle_from_payload(payload)
                if bundle is not None:
                    bundle["cache_path"] = cache_path
                    cache[cache_key] = bundle
                    return bundle
            except Exception as exc:
                logger.warning(
                    "Failed to load posterior reference cache %s: %s",
                    cache_path,
                    exc,
                )

        try:
            short_dataset = TreeDataset(
                nexus_root="unused",
                mrbayes_root="unused",
                posterior_trprobs_root=str(posterior_root),
                posterior_dataset_id=str(dataset_id),
                trprobs_sample_count_per_file=int(trprobs_sample_count),
            )
            short_raw = list(short_dataset.return_posterior_trees(str(dataset_id)))
        except Exception as exc:
            logger.warning(
                "Skipping posterior reference metrics for %s: %s",
                dataset_id,
                exc,
            )
            cache[cache_key] = None
            return None
        golden_raw = []
        if golden_root and os.path.isdir(str(golden_root)):
            try:
                golden_dataset = TreeDataset(
                    nexus_root="unused",
                    mrbayes_root="unused",
                    posterior_trprobs_root=str(golden_root),
                    posterior_dataset_id=str(dataset_id),
                    trprobs_sample_count_per_file=int(trprobs_sample_count),
                )
                golden_raw = list(golden_dataset.return_posterior_trees(str(dataset_id)))
            except Exception as exc:
                logger.warning(
                    "Skipping golden posterior reference metrics for %s: %s",
                    dataset_id,
                    exc,
                )
                golden_raw = []

        reference_tree = short_raw[0] if short_raw else (golden_raw[0] if golden_raw else None)
        if reference_tree is None:
            cache[cache_key] = None
            return None

        posterior_raw_to_lex = self._build_harness_lexicographic_ordering_map(reference_tree)
        numeric_to_lex = self._build_numeric_to_harness_lexicographic_ordering_map(reference_tree)

        short_lex = [
            self._remap_tree_with_ordering_map(tree, posterior_raw_to_lex)
            for tree in short_raw
        ]
        golden_lex = [
            self._remap_tree_with_ordering_map(tree, posterior_raw_to_lex)
            for tree in golden_raw
        ]
        bundle = {
            "dataset_id": str(dataset_id),
            "short_root": str(posterior_root),
            "golden_root": str(golden_root) if golden_root else None,
            "short_counts": self._canonical_tree_topology_counts(short_lex),
            "golden_counts": self._canonical_tree_topology_counts(golden_lex),
            "short_split_counts": self._split_topology_counts(short_lex),
            "golden_split_counts": self._split_topology_counts(golden_lex),
            "short_tree_total": int(len(short_lex)),
            "golden_tree_total": int(len(golden_lex)),
            "posterior_raw_to_lex": posterior_raw_to_lex,
            "numeric_to_lex": numeric_to_lex,
            "num_leaves": len(EteTree(short_lex[0] if short_lex else golden_lex[0], format=1).get_leaves()),
            "cache_path": cache_path,
        }
        bundle["short_split_total"] = int(sum(bundle["short_split_counts"].values()))
        bundle["golden_split_total"] = int(sum(bundle["golden_split_counts"].values()))
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            payload = self._posterior_reference_bundle_to_payload(
                bundle,
                posterior_root=posterior_root,
                golden_root=golden_root,
                dataset_id=dataset_id,
                trprobs_sample_count=trprobs_sample_count,
                cache_path=cache_path,
            )
            tmp_path = f"{cache_path}.tmp.{os.getpid()}"
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_path, cache_path)
        except Exception as exc:
            logger.warning(
                "Failed to write posterior reference cache %s: %s",
                cache_path,
                exc,
            )
        cache[cache_key] = bundle
        return bundle

    def _remap_sampled_tree_to_reference_lex(self, tree_newick, bundle):
        tree_newick = str(tree_newick)
        leaf_names = [str(leaf.name) for leaf in EteTree(tree_newick, format=1).iter_leaves()]
        posterior_raw_to_lex = bundle["posterior_raw_to_lex"]
        numeric_to_lex = bundle["numeric_to_lex"]
        if all(name in posterior_raw_to_lex for name in leaf_names):
            return self._remap_tree_with_ordering_map(tree_newick, posterior_raw_to_lex)
        if all(name in numeric_to_lex for name in leaf_names):
            return self._remap_tree_with_ordering_map(tree_newick, numeric_to_lex)
        return tree_newick

    def _posterior_reference_metrics(self, sampled_trees, train=True):
        bundle = self._load_posterior_reference_bundle(train=train)
        if bundle is None or not sampled_trees:
            return {}

        remapped_sampled = [
            self._remap_sampled_tree_to_reference_lex(tree, bundle)
            for tree in sampled_trees
        ]

        metrics = {}
        sampled_counts = self._canonical_tree_topology_counts(remapped_sampled)
        sampled_split_counts = self._split_topology_counts(remapped_sampled)
        short_counts = Counter(bundle.get("short_counts") or {})
        golden_counts = Counter(bundle.get("golden_counts") or {})
        short_split_counts = Counter(bundle.get("short_split_counts") or {})
        golden_split_counts = Counter(bundle.get("golden_split_counts") or {})
        support_size = len(short_counts) or len(golden_counts)
        metrics.update(
            self._sampled_topology_distribution_metrics_from_counts(
                sampled_counts,
                reference_support_size=support_size,
            )
        )
        metrics["short_support_rate"] = self._support_rate_from_topology_counts(
            sampled_counts,
            short_counts,
        )
        if golden_counts:
            metrics["golden_support_rate"] = self._support_rate_from_topology_counts(
                sampled_counts,
                golden_counts,
            )

        if short_counts:
            short_block = {}
            short_block.update(self._kl_from_split_counts(short_split_counts, sampled_split_counts))
            short_block.update(self._kl_from_topology_counts(short_counts, sampled_counts))
            short_block.update(
                self._topk_recall_from_topology_counts(short_counts, sampled_counts)
            )
            metrics.update(self._prefix_metric_block(short_block, "short"))
        if golden_counts:
            golden_block = {}
            golden_block.update(
                self._kl_from_split_counts(golden_split_counts, sampled_split_counts)
            )
            golden_block.update(self._kl_from_topology_counts(golden_counts, sampled_counts))
            golden_block.update(
                self._topk_recall_from_topology_counts(golden_counts, sampled_counts)
            )
            metrics.update(self._prefix_metric_block(golden_block, "golden"))
        return metrics

    def _get_branch_relax_likelihood_scorer(self):
        if not self.branch_relax_likelihood_metric_enabled:
            return None
        dataset_id = self.branch_relax_likelihood_dataset_id
        if not dataset_id:
            bundle = self._load_posterior_reference_bundle(train=True)
            dataset_id = None if bundle is None else bundle.get("dataset_id")
        if not dataset_id:
            return None
        if self._branch_relax_likelihood_scorer is None:
            from analysis.full_sanity_fixedpair_20260401.multi_ds_branchwarm_cumulative_mh_experiment import (
                GenericJCLikelihood,
            )

            self._branch_relax_likelihood_scorer = GenericJCLikelihood(
                dataset_id=str(dataset_id)
            )
        return self._branch_relax_likelihood_scorer

    def _sample_metrics_dataset_id(self, train=True):
        dataset_id = self.branch_relax_likelihood_dataset_id
        if dataset_id:
            return str(dataset_id)
        bundle = self._load_posterior_reference_bundle(train=train)
        if bundle is not None and bundle.get("dataset_id"):
            return str(bundle["dataset_id"])
        dataset_split = self._sample_metrics_dataset_split(train=train)
        if dataset_split is None:
            return None
        posterior_ids = list(getattr(dataset_split, "posterior_dataset_ids", []) or [])
        if len(posterior_ids) == 1:
            return str(posterior_ids[0])
        return None

    def _get_sample_metrics_likelihood_scorer(self, train=True):
        dataset_id = self._sample_metrics_dataset_id(train=train)
        if not dataset_id:
            return None
        cache = getattr(self, "_sample_metrics_likelihood_scorer_cache", None)
        if cache is None:
            cache = {}
            self._sample_metrics_likelihood_scorer_cache = cache
        key = str(dataset_id).upper()
        scorer = cache.get(key)
        if scorer is None:
            from analysis.full_sanity_fixedpair_20260401.multi_ds_branchwarm_cumulative_mh_experiment import (
                GenericJCLikelihood,
            )

            scorer = GenericJCLikelihood(dataset_id=key)
            cache[key] = scorer
        return scorer

    def _sample_metrics_standalone_relaxer_args(self, checkpoint):
        from types import SimpleNamespace

        raw_args = dict(checkpoint.get("args") or {})
        repo_root = os.getcwd()

        def _localize_repo_path(value):
            if value is None:
                return value
            value = str(value)
            old_root = "/home/yektefai/PhylaFlow"
            if value == old_root:
                return repo_root
            if value.startswith(old_root + os.sep):
                return os.path.join(repo_root, value[len(old_root) + 1 :])
            return value

        def _localize_base_config(value):
            value = _localize_repo_path(value)
            if value and not os.path.exists(value):
                directory, filename = os.path.split(value)
                if filename.startswith("local_"):
                    alternate = os.path.join(directory, filename[len("local_") :])
                    if os.path.exists(alternate):
                        return alternate
            return value

        return SimpleNamespace(
            base_config=_localize_base_config(
                raw_args.get(
                    "base_config",
                    os.path.join(
                        repo_root,
                        "configs/local_ds1_frozenprobe64_fh16_aradd_scale128x4_lr2e3_20260428.yaml",
                    ),
                )
            ),
            embed_dim=int(raw_args.get("embed_dim", 64)),
            n_layers=int(raw_args.get("n_layers", 2)),
            n_heads=int(raw_args.get("n_heads", 4)),
            dropout=float(raw_args.get("dropout", 0.0)),
            head_hidden_dim=int(raw_args.get("head_hidden_dim", 128)),
            case_dim=int(raw_args.get("case_dim", 0)),
            phyla_dim=int(raw_args.get("phyla_dim", 256)),
            phyla_use_leaf_tokens=bool(raw_args.get("phyla_use_leaf_tokens", True)),
            phyla_use_split_tokens=bool(raw_args.get("phyla_use_split_tokens", False)),
            phyla_embedding_dir=_localize_repo_path(
                raw_args.get(
                    "phyla_embedding_dir",
                    os.path.join(
                        repo_root,
                        "analysis/full_sanity_fixedpair_20260401/ds_phyla_embeddings_20260428",
                    ),
                )
            ),
        )

    def _get_sample_metrics_standalone_relaxer(self, train=True):
        path = self.sample_metrics_branch_relaxer_checkpoint_path
        if not path:
            return None
        dataset_id = self._sample_metrics_dataset_id(train=train)
        if not dataset_id:
            return None
        device = self.device
        cache = getattr(self, "_sample_metrics_standalone_relaxer_cache", None)
        if cache is None:
            cache = {}
            self._sample_metrics_standalone_relaxer_cache = cache
        cache_key = (str(path), str(device), str(dataset_id).upper())
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        from analysis.full_sanity_fixedpair_20260401.train_standalone_branch_relaxer_20260429 import (
            BranchDeltaHead,
            StandaloneRelaxer,
            _load_phyla_embedding_bank,
            _small_model_config,
        )
        from model.model import return_model

        checkpoint = torch.load(str(path), map_location=device)
        ckpt_args = self._sample_metrics_standalone_relaxer_args(checkpoint)
        cfg = _small_model_config(ckpt_args.base_config, ckpt_args)
        model = return_model(cfg).to(device)
        head_state = checkpoint.get("head") or {}
        case_weight = head_state.get("case_embedding.weight")
        num_cases = (
            int(case_weight.shape[0])
            if torch.is_tensor(case_weight)
            else max(1, int(self.sample_metrics_mrbayes20k_num_starts))
        )
        head = BranchDeltaHead(
            int(model.embed_dim),
            hidden_dim=int(ckpt_args.head_hidden_dim),
            case_dim=int(ckpt_args.case_dim),
            num_cases=int(num_cases),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        head.load_state_dict(head_state)
        relaxer = StandaloneRelaxer(model, head).to(device)
        relaxer.eval()
        phyla_bank = _load_phyla_embedding_bank(
            [str(dataset_id).upper()],
            ckpt_args.phyla_embedding_dir,
            device,
        )
        cached = {
            "relaxer": relaxer,
            "phyla_bank": phyla_bank,
            "dataset_id": str(dataset_id).upper(),
        }
        cache[cache_key] = cached
        return cached

    def _sample_metrics_source_mask_for_node(self, node, n_leaves):
        root_leaf = int(n_leaves) - 1
        biological_bits = max(int(n_leaves) - 1, 0)
        full_mask = (1 << biological_bits) - 1 if biological_bits > 0 else 0
        raw_indices = []
        for leaf in node.iter_leaves():
            value = int(str(leaf.name))
            raw_indices.append(value - 1 if 1 <= value <= int(n_leaves) else value)
        indices = set(raw_indices)
        if root_leaf in indices:
            indices = set(range(int(n_leaves))) - indices
        mask = 0
        for index in indices:
            if 0 <= int(index) < biological_bits:
                mask |= 1 << int(index)
        return int(mask if mask else full_mask)

    def _sample_metrics_apply_standalone_relaxer(self, tree_newick, case_index, train=True):
        bundle = self._get_sample_metrics_standalone_relaxer(train=train)
        if bundle is None:
            return str(tree_newick), {"applied": False}
        relaxer = bundle["relaxer"]
        device = self.device
        dataset_id = str(bundle["dataset_id"]).upper()
        newick = str(tree_newick).strip()
        if not newick.endswith(";"):
            newick += ";"
        tokenized = _move_tokenized_batch_to_device(relaxer.model.tokenizer([newick]), device)
        phyla_embeddings = None
        phyla_bank = bundle.get("phyla_bank") or {}
        if phyla_bank:
            phyla_embeddings = phyla_bank[dataset_id]
        with torch.inference_mode():
            edge_outputs = relaxer.model(
                tokenized,
                torch.tensor([4.0], dtype=torch.float32, device=device),
                phyla_embeddings=phyla_embeddings,
                return_leafs_only=False,
                return_edges_only=True,
                return_edge_features=True,
            )
            _edge_values, _edge_pad_mask, edge_features = edge_outputs
            edge_split_masks = tokenized[-1]
            entries, _lengths, n_leaves, _mapping = _branch_relax_entries_for_tree(
                relaxer,
                newick,
                edge_split_masks[0],
                labels=None,
            )
            if not entries:
                return newick, {"applied": False}
            features = torch.stack(
                [edge_features[0, entry["edge_index"]] for entry in entries],
                dim=0,
            )
            numeric = torch.tensor(
                [entry["numeric"] for entry in entries],
                dtype=torch.float32,
                device=device,
            )
            case_indices = torch.full(
                (len(entries),),
                int(case_index),
                dtype=torch.long,
                device=device,
            )
            deltas = relaxer.head(features, numeric, case_indices).detach().cpu().numpy()

        delta_by_source_mask = {
            int(entry.get("source_mask", entry["mask"])): float(delta)
            for entry, delta in zip(entries, deltas)
        }
        tree = EteTree(newick, format=1, quoted_node_names=True)
        applied = 0
        max_abs_delta = 0.0
        for node in tree.traverse("postorder"):
            if node.is_root():
                continue
            try:
                source_mask = self._sample_metrics_source_mask_for_node(
                    node,
                    int(n_leaves),
                )
            except Exception:
                continue
            delta = delta_by_source_mask.get(int(source_mask))
            if delta is None:
                continue
            before = float(node.dist)
            after = max(before + float(delta), 1e-8)
            node.dist = after
            applied += 1
            max_abs_delta = max(max_abs_delta, abs(after - before))
        return tree.write(format=1), {
            "applied": bool(applied),
            "applied_edge_count": int(applied),
            "max_abs_delta": float(max_abs_delta),
        }

    def _sample_metrics_infer_mrbayes_paths(self, train=True):
        dataset_id = self._sample_metrics_dataset_id(train=train)
        if not dataset_id:
            return None, None, None
        dataset_id = str(dataset_id).upper()
        dataset_pickle = self.sample_metrics_mrbayes20k_dataset_pickle_path
        golden_root = self.sample_metrics_mrbayes20k_golden_root
        bundle = self._load_posterior_reference_bundle(train=train)
        if golden_root is None and bundle is not None and bundle.get("golden_root"):
            golden_root = str(bundle["golden_root"])
        if dataset_pickle is None and golden_root:
            root = os.path.dirname(os.path.dirname(str(golden_root).rstrip("/")))
            candidate = os.path.join(root, f"{dataset_id}.pickle")
            if os.path.exists(candidate):
                dataset_pickle = candidate
        return dataset_id, dataset_pickle, golden_root

    def _sample_metrics_mrbayes20k_output_dir(self):
        if self.sample_metrics_mrbayes20k_output_dir:
            return self.sample_metrics_mrbayes20k_output_dir
        if self.sample_metrics_trace_path:
            return os.path.join(
                os.path.dirname(self.sample_metrics_trace_path),
                "mrbayes20k_sample_metrics",
            )
        return "/tmp/phylaflow_sample_metrics_mrbayes20k_outputs"

    def _sample_metrics_run_mrbayes20k(self, relaxed_trees, train=True):
        if not self.sample_metrics_mrbayes20k_enabled:
            return {}
        dataset_id, dataset_pickle, golden_root = self._sample_metrics_infer_mrbayes_paths(
            train=train
        )
        if not dataset_id or not dataset_pickle or not golden_root:
            return {"mrbayes20k_failed": 1.0}
        selected_trees = list(relaxed_trees)[
            : min(len(relaxed_trees), int(self.sample_metrics_mrbayes20k_num_starts))
        ]
        if not selected_trees:
            return {"mrbayes20k_failed": 1.0}

        step = int(self.global_step)
        stamp = f"pid{os.getpid()}_step{step:08d}_{int(time.time())}"
        work_dir = os.path.join(self.sample_metrics_mrbayes20k_work_root, stamp)
        output_dir = self._sample_metrics_mrbayes20k_output_dir()
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        start_list_path = os.path.join(output_dir, f"{stamp}_relaxed_starts.txt")
        output_json = os.path.join(output_dir, f"{stamp}_mrbayes20k_curve.json")
        log_path = os.path.join(output_dir, f"{stamp}_mrbayes20k.log")
        with open(start_list_path, "w", encoding="utf-8") as handle:
            for tree in selected_trees:
                tree = str(tree).strip()
                handle.write(tree if tree.endswith(";") else tree + ";")
                handle.write("\n")

        repo_root = os.getcwd()
        benchmark = os.path.join(
            repo_root,
            "analysis/full_sanity_fixedpair_20260401/"
            "benchmark_mrbayes_fixed_start_generic.py",
        )
        cmd = [
            sys.executable,
            benchmark,
            "--dataset-id",
            str(dataset_id),
            "--dataset-pickle",
            str(dataset_pickle),
            "--golden-root",
            str(golden_root),
            "--label",
            f"sample_metrics_mrbayes20k_step{step}",
            "--num-runs",
            str(len(selected_trees)),
            "--ngen",
            str(int(self.sample_metrics_mrbayes20k_ngen)),
            "--samplefreq",
            str(int(self.sample_metrics_mrbayes20k_samplefreq)),
            "--printfreq",
            str(int(self.sample_metrics_mrbayes20k_printfreq)),
            "--max-workers",
            str(int(self.sample_metrics_mrbayes20k_max_workers)),
            "--curve-interval",
            str(int(self.sample_metrics_mrbayes20k_ngen)),
            "--mrbayes-bin",
            str(self.sample_metrics_mrbayes20k_bin),
            "--work-dir",
            str(work_dir),
            "--output",
            str(output_json),
            "--start-tree-list",
            str(start_list_path),
            "--threshold-check-selected-only",
        ]
        try:
            with open(log_path, "w", encoding="utf-8") as log_file:
                result = subprocess.run(
                    cmd,
                    cwd=repo_root,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=int(self.sample_metrics_mrbayes20k_timeout_sec),
                )
        except Exception as exc:
            return {
                "mrbayes20k_failed": 1.0,
                "mrbayes20k_error": str(exc),
            }
        if result.returncode != 0 or not os.path.exists(output_json):
            return {
                "mrbayes20k_failed": 1.0,
                "mrbayes20k_returncode": float(result.returncode),
            }
        with open(output_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        final_block = payload.get("final_cumulative_by_generation") or {}
        tail_block = payload.get("tail_half_samples") or {}
        failures = payload.get("failures") or []
        return {
            "mrbayes20k_failed": 0.0,
            "mrbayes20k_num_starts": float(len(selected_trees)),
            "mrbayes20k_completed_runs": float(payload.get("completed_runs", 0)),
            "mrbayes20k_failure_count": float(len(failures)),
            "mrbayes20k_tree_kl": float(
                final_block.get("kl_divergence_tree_topology", float("nan"))
            ),
            "mrbayes20k_tail_tree_kl": float(
                tail_block.get("kl_divergence_tree_topology", float("nan"))
            ),
        }

    def _sample_metrics_relaxed_downstream_metrics(
        self,
        sampled_trees,
        train=True,
        return_tree_rows=False,
        run_mrbayes=True,
    ):
        if not (
            self.sample_metrics_relaxed_likelihood_enabled
            or self.sample_metrics_mrbayes20k_enabled
        ):
            return ({}, []) if return_tree_rows else {}
        if not sampled_trees:
            return ({}, []) if return_tree_rows else {}
        relaxed_trees = []
        relaxed_tree_rows = []
        likelihoods = []
        applied = 0
        scorer = (
            self._get_sample_metrics_likelihood_scorer(train=train)
            if self.sample_metrics_relaxed_likelihood_enabled
            else None
        )
        for index, tree in enumerate(sampled_trees):
            relaxed_tree, info = self._sample_metrics_apply_standalone_relaxer(
                tree,
                case_index=index,
                train=train,
            )
            relaxed_trees.append(relaxed_tree)
            applied += int(bool(info.get("applied")))
            relaxed_row = {
                "sample_index": int(index),
                "relaxed_tree": str(relaxed_tree),
                "relax_applied": bool(info.get("applied")),
                "relax_applied_edge_count": int(info.get("applied_edge_count", 0)),
                "relax_max_abs_delta": float(info.get("max_abs_delta", 0.0)),
            }
            if scorer is not None:
                likelihood = float(scorer.log_likelihood(str(relaxed_tree)))
                likelihoods.append(likelihood)
                relaxed_row["relaxed_log_likelihood"] = likelihood
            relaxed_tree_rows.append(relaxed_row)

        metrics = {
            "relaxed_start_count": float(len(relaxed_trees)),
            "relaxed_applied_count": float(applied),
        }
        if likelihoods:
            metrics["relaxed_log_likelihood_mean"] = float(
                np.asarray(likelihoods, dtype=np.float64).mean()
            )
        if run_mrbayes:
            metrics.update(
                self._sample_metrics_run_mrbayes20k(relaxed_trees, train=train)
            )
        if return_tree_rows:
            return metrics, relaxed_tree_rows
        return metrics

    def _trace_topology_decoder_metrics(self, trace, prefix="trace_"):
        events = list((trace or {}).get("autoregressive", []) or [])
        birthset_events = [
            event for event in events if event.get("decoder_mode") == "birthset"
        ]
        legacy_events = [
            event
            for event in events
            if event.get("decoder_mode")
            in {"ar", "ar_fallback", "pairwise_threshold", "structured_subset"}
        ]

        def _selected_split_count(event):
            splits = event.get("selected_result_splits")
            if splits is not None:
                return len(splits)
            return 1 if event.get("selected_result_split") is not None else 0

        birthset_metrics = [
            event.get("birthset_metrics", {}) or {} for event in birthset_events
        ]
        result = {
            f"{prefix}topology_trace_entries": float(len(events)),
            f"{prefix}birthset_events": float(len(birthset_events)),
            f"{prefix}legacy_ar_events": float(len(legacy_events)),
            f"{prefix}birthset_inserted_splits": float(
                sum(_selected_split_count(event) for event in birthset_events)
            ),
            f"{prefix}birthset_incomplete_without_fallback": float(
                1.0
                if (trace or {}).get("birthset_incomplete_without_fallback", False)
                else 0.0
            ),
        }
        if birthset_metrics:
            result[f"{prefix}birthset_required_splits"] = float(
                sum(
                    float(metrics.get("num_required_birth_splits", 0.0))
                    for metrics in birthset_metrics
                )
            )
            result[f"{prefix}birthset_candidate_splits"] = float(
                sum(
                    float(metrics.get("num_candidate_splits", 0.0))
                    for metrics in birthset_metrics
                )
            )
            result[f"{prefix}birthset_ar_fallback_calls"] = float(
                sum(
                    float(metrics.get("num_ar_fallback_calls", 0.0))
                    for metrics in birthset_metrics
                )
            )
            resolved_values = [
                float(metrics.get("fraction_resolved_without_fallback", 0.0))
                for metrics in birthset_metrics
                if "fraction_resolved_without_fallback" in metrics
            ]
            if resolved_values:
                result[
                    f"{prefix}birthset_fraction_resolved_without_fallback_mean"
                ] = float(np.mean(resolved_values))
        return result

    def _sample_compare_harness_once(self, pair, train=True):
        sampled_trees, _, _, _, _, trace = self.sample(
            [pair["start_tree"]],
            **self._build_harness_sample_kwargs(pair, train=train),
        )
        sampled_tree, sampled_tree_label_remapped = (
            align_numeric_leaf_labels_to_reference(
                sampled_trees[0],
                pair["start_tree"],
                target_tree=pair["target_tree"],
            )
        )
        likelihood_scorer = self._get_branch_relax_likelihood_scorer()
        metrics = {
            "rf_norm": float(calculate_norm_rf(sampled_tree, pair["target_tree"])),
            "start_rf_norm": float(
                calculate_norm_rf(pair["start_tree"], pair["target_tree"])
            ),
            "sampled_tree_label_remapped": float(
                1.0 if sampled_tree_label_remapped else 0.0
            ),
            "_start_tree": str(pair["start_tree"]),
            "_original_start_tree": (
                str(pair["original_start_tree"])
                if pair.get("original_start_tree") is not None
                else None
            ),
            "_sampled_tree": str(sampled_tree),
            "_target_tree": str(pair["target_tree"]),
            "_n_leaves": int(pair["n_leaves"]),
            "_bank_group_key": pair.get("bank_group_key"),
            "_source_bank_index": pair.get("source_bank_index"),
        }
        if likelihood_scorer is not None:
            metrics["branch_relax_after_log_likelihood"] = float(
                likelihood_scorer.log_likelihood(str(sampled_tree))
            )
        if self.sample_metrics_trace_topology_repeats_enabled:
            metrics.update(_summarize_trace_topology_repeats(trace))
        metrics.update(self._trace_topology_decoder_metrics(trace, prefix="trace_"))
        metrics["stopped_for_repeated_topology"] = float(
            1.0 if trace.get("stopped_for_repeated_topology", False) else 0.0
        )
        metrics["stopped_for_no_valid_merge"] = float(
            1.0 if trace.get("stopped_for_no_valid_merge", False) else 0.0
        )
        metrics["skipped_no_valid_boundary_revisits"] = float(
            trace.get("skipped_no_valid_boundary_revisits", 0.0)
        )
        return metrics

    def _summarize_sample_compare_harness_rows(self, rows, train=True):
        if len(rows) == 1:
            row = dict(rows[0])
            metrics = {"num_pairs": 1}
            for key in ("rf_norm", "start_rf_norm"):
                value = row.get(key)
                if isinstance(value, (int, float, np.generic)):
                    scalar = float(value)
                    metrics[key] = scalar
                    metrics[f"{key}_mean"] = scalar
                    metrics[f"{key}_median"] = scalar
                    metrics[f"{key}_best"] = scalar
                    metrics[f"{key}_worst"] = scalar
                    metrics[f"{key}_min"] = scalar
                    metrics[f"{key}_max"] = scalar
                    metrics[f"{key}_p10"] = scalar
                    metrics[f"{key}_p90"] = scalar
            metrics.update(row)
            sampled_tree = row.get("_sampled_tree")
            relaxed_tree_rows = []
            if sampled_tree:
                downstream_metrics, relaxed_tree_rows = (
                    self._sample_metrics_relaxed_downstream_metrics(
                        [sampled_tree],
                        train=train,
                        return_tree_rows=True,
                        run_mrbayes=False,
                    )
                )
                metrics.update(downstream_metrics)
            metrics.update(
                self._write_sample_metrics_tree_dump(
                    [row],
                    relaxed_tree_rows=relaxed_tree_rows,
                    train=train,
                )
            )
            if relaxed_tree_rows:
                metrics.update(
                    self._sample_metrics_run_mrbayes20k(
                        [row["relaxed_tree"] for row in relaxed_tree_rows],
                        train=train,
                    )
                )
            return metrics

        def _numeric_values(key):
            values = []
            for row in rows:
                value = row.get(key)
                if isinstance(value, (int, float, np.generic)):
                    values.append(float(value))
            return values

        metrics = {"num_pairs": int(len(rows))}
        for key in ("rf_norm", "start_rf_norm"):
            values = _numeric_values(key)
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            metrics[key] = float(arr.mean())
            metrics[f"{key}_mean"] = float(arr.mean())
            metrics[f"{key}_median"] = float(np.median(arr))
            metrics[f"{key}_best"] = float(arr.min())
            metrics[f"{key}_worst"] = float(arr.max())
            metrics[f"{key}_min"] = float(arr.min())
            metrics[f"{key}_max"] = float(arr.max())
            metrics[f"{key}_p10"] = float(np.quantile(arr, 0.10))
            metrics[f"{key}_p90"] = float(np.quantile(arr, 0.90))

        sampled_trees = [row.get("_sampled_tree") for row in rows]
        target_trees = [row.get("_target_tree") for row in rows]
        relaxed_tree_rows = []
        n_leaves_values = [
            int(row.get("_n_leaves"))
            for row in rows
            if row.get("_n_leaves") is not None
        ]
        if (
            len(sampled_trees) == len(rows)
            and len(target_trees) == len(rows)
            and all(sampled_trees)
            and all(target_trees)
        ):
            if n_leaves_values and len(set(n_leaves_values)) == 1:
                metrics.update(
                    kl_divergence_topological_distributions(
                        target_trees,
                        sampled_trees,
                        num_leaves=int(n_leaves_values[0]),
                    )
                )
            metrics.update(
                kl_divergence_tree_topology_distributions(
                    target_trees,
                    sampled_trees,
                )
            )
            metrics.update(
                self._posterior_reference_metrics(sampled_trees, train=train)
            )
            downstream_metrics, relaxed_tree_rows = (
                self._sample_metrics_relaxed_downstream_metrics(
                    sampled_trees,
                    train=train,
                    return_tree_rows=True,
                    run_mrbayes=False,
                )
            )
            metrics.update(downstream_metrics)

        metrics.update(
            self._write_sample_metrics_tree_dump(
                rows,
                relaxed_tree_rows=relaxed_tree_rows,
                train=train,
            )
        )
        if relaxed_tree_rows:
            metrics.update(
                self._sample_metrics_run_mrbayes20k(
                    [row["relaxed_tree"] for row in relaxed_tree_rows],
                    train=train,
                )
            )

        aggregate_keys = sorted(
            {
                key
                for row in rows
                for key in row.keys()
                if key not in {"rf_norm", "start_rf_norm"} and not str(key).startswith("_")
            }
        )
        for key in aggregate_keys:
            values = _numeric_values(key)
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            metrics[f"{key}_mean"] = float(arr.mean())
            metrics[f"{key}_max"] = float(arr.max())
        return metrics

    def _sample_metrics_bank_size(self, dataset_split):
        selections = getattr(dataset_split, "_frozen_full_path_control_selections", None)
        if selections:
            return len(selections)
        start_bank = getattr(dataset_split, "overfit_fixed_pair_start_tree_newick_bank", [])
        target_bank = getattr(dataset_split, "overfit_fixed_pair_target_tree_newick_bank", [])
        if target_bank:
            return len(target_bank)
        if start_bank:
            return len(start_bank)
        try:
            return len(dataset_split)
        except TypeError:
            return 0

    def _sample_metrics_select_bank_indices(self, total_count, num_pairs, train=True):
        total_count = max(0, int(total_count))
        if total_count <= 0:
            return []
        num_pairs = min(max(1, int(num_pairs)), total_count)
        mode = str(
            getattr(self, "sample_metrics_unseen_pair_selection_mode", "random_bank")
        ).strip().lower()
        if mode in {"first", "sequential"}:
            return list(range(num_pairs))
        seed = (
            int(self.sample_metrics_unseen_start_seed)
            + int(self.global_step) * 1009
            + (0 if train else 17)
        )
        rng = random.Random(seed)
        return rng.sample(range(total_count), k=num_pairs)

    def _sample_metrics_build_bank_pair(self, dataset_split, pair_index):
        fixed_pair = None
        sampled = None
        if hasattr(dataset_split, "get_overfit_fixed_pair"):
            try:
                fixed_pair = dataset_split.get_overfit_fixed_pair(int(pair_index))
            except Exception:
                fixed_pair = None
        if fixed_pair is None:
            sampled = dataset_split[int(pair_index)]
            if hasattr(dataset_split, "get_overfit_fixed_pair"):
                try:
                    fixed_pair = dataset_split.get_overfit_fixed_pair(int(pair_index))
                except Exception:
                    fixed_pair = None

        if fixed_pair is not None:
            start_tree = fixed_pair.get("random_tree", fixed_pair.get("start_tree"))
            target_tree = fixed_pair.get(
                "effective_target_tree", fixed_pair.get("target_tree")
            )
            max_events_value = int(
                len(fixed_pair.get("final_labels", []) or [])
                or fixed_pair.get("fixed_pair_num_events", 1024)
            )
            name_mapping_value = fixed_pair.get("name_mapping")
            bank_group_key_value = fixed_pair.get("bank_group_key") or fixed_pair.get(
                "group_key"
            )
            dataset_id_value = fixed_pair.get("dataset_id")
            selected_sequences_value = fixed_pair.get("selected_sequences")
            selected_sequence_names_value = fixed_pair.get("selected_sequence_names")
        else:
            start_tree = sampled.get("start_tree")
            target_tree = sampled.get("target_tree")
            max_events_value = int(sampled.get("fixed_pair_num_events", 1024))
            name_mapping_value = sampled.get("name_mapping")
            bank_group_key_value = sampled.get("bank_group_key") or sampled.get(
                "group_key"
            )
            dataset_id_value = sampled.get("dataset_id")
            selected_sequences_value = sampled.get("selected_sequences")
            selected_sequence_names_value = sampled.get("selected_sequence_names")

        topology_stream_index_path = getattr(
            dataset_split,
            "topology_stream_index_jsonl_path",
            None,
        )
        mapping = name_mapping_value
        if mapping is None and not topology_stream_index_path:
            mapping = (
                dataset_split.return_nexus_number_to_name(0)
                if hasattr(dataset_split, "return_nexus_number_to_name")
                else None
            )
        return {
            "start_tree": str(start_tree),
            "target_tree": str(target_tree),
            "bank_group_key": bank_group_key_value,
            "dataset_id": dataset_id_value,
            "n_leaves": len(EteTree(str(start_tree), format=1).get_leaves()),
            "max_events": int(max_events_value),
            "name_mapping": mapping,
            "selected_sequences": selected_sequences_value,
            "selected_sequence_names": selected_sequence_names_value,
            "source_bank_index": int(pair_index),
        }

    def _sample_metrics_topology_key(self, tree):
        try:
            return canonicalize_topology_newick(str(tree))
        except Exception:
            return str(tree)

    def _sample_metrics_training_start_topology_keys(self, dataset_split):
        bank = getattr(dataset_split, "overfit_fixed_pair_start_tree_newick_bank", [])
        topology_stream_index_path = getattr(
            dataset_split,
            "topology_stream_index_jsonl_path",
            None,
        )
        cache = getattr(self, "_sample_metrics_training_start_topology_key_cache", None)
        if cache is None:
            cache = {}
            self._sample_metrics_training_start_topology_key_cache = cache
        try:
            dataset_len = len(dataset_split)
        except TypeError:
            dataset_len = 0
        cache_key = (
            id(dataset_split),
            str(topology_stream_index_path) if topology_stream_index_path else None,
            int(dataset_len),
            id(bank),
            len(bank),
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        if topology_stream_index_path:
            keys = set()
            random_state = random.getstate()
            try:
                for index in range(int(dataset_len)):
                    try:
                        sample = dataset_split[index]
                    except Exception:
                        continue
                    start_tree = sample.get("start_tree") if isinstance(sample, dict) else None
                    if start_tree:
                        keys.add(self._sample_metrics_topology_key(start_tree))
            finally:
                random.setstate(random_state)
            cache[cache_key] = keys
            return keys
        keys = {self._sample_metrics_topology_key(tree) for tree in bank}
        cache[cache_key] = keys
        return keys

    def _sample_metrics_generate_unseen_start(
        self,
        dataset_split,
        target_tree,
        seen,
        training_starts=None,
    ):
        if training_starts is None:
            training_starts = self._sample_metrics_training_start_topology_keys(
                dataset_split
            )
        candidate = None
        for _attempt in range(self.sample_metrics_unseen_start_max_duplicate_tries):
            candidate = str(dataset_split.sample_random_tree(str(target_tree)))
            candidate_key = self._sample_metrics_topology_key(candidate)
            if candidate_key not in seen and candidate_key not in training_starts:
                seen.add(candidate_key)
                return candidate
        if candidate is None:
            candidate = str(dataset_split.sample_random_tree(str(target_tree)))
        seen.add(self._sample_metrics_topology_key(candidate))
        return candidate

    def _sample_metrics_unseen_bank_pairs(self, dataset_split, train=True):
        num_pairs = max(1, int(getattr(self, "sample_metrics_num_pairs", 1)))
        bank_size = self._sample_metrics_bank_size(dataset_split)
        if bank_size <= 0:
            return []
        if num_pairs <= bank_size:
            indices = self._sample_metrics_select_bank_indices(
                bank_size,
                num_pairs,
                train=train,
            )
        else:
            mode = str(
                getattr(self, "sample_metrics_unseen_pair_selection_mode", "random_bank")
            ).strip().lower()
            seed_for_indices = (
                int(self.sample_metrics_unseen_start_seed)
                + int(self.global_step) * 1009
                + (0 if train else 17)
            )
            if mode in {"first", "sequential"}:
                indices = [idx % bank_size for idx in range(num_pairs)]
            else:
                rng = random.Random(seed_for_indices)
                indices = [rng.randrange(bank_size) for _ in range(num_pairs)]
        random_state = random.getstate()
        seed = (
            int(self.sample_metrics_unseen_start_seed)
            + int(self.global_step) * 9973
            + (0 if train else 31)
        )
        pairs = []
        seen_starts = set()
        training_start_keys = self._sample_metrics_training_start_topology_keys(
            dataset_split
        )
        try:
            random.seed(seed)
            for eval_index, source_index in enumerate(indices):
                pair = self._sample_metrics_build_bank_pair(dataset_split, source_index)
                original_start_tree = pair["start_tree"]
                unseen_start_tree = self._sample_metrics_generate_unseen_start(
                    dataset_split,
                    pair["target_tree"],
                    seen_starts,
                    training_start_keys,
                )
                pair["original_start_tree"] = original_start_tree
                pair["start_tree"] = unseen_start_tree
                pair["bank_group_key"] = f"sample_metrics_unseen_case{eval_index:05d}"
                pair["n_leaves"] = len(EteTree(unseen_start_tree, format=1).get_leaves())
                pair["source_bank_index"] = int(source_index)
                pairs.append(pair)
        finally:
            random.setstate(random_state)
        return pairs

    def _sample_metrics_model_uses_frozen_start_case_table(self):
        return any(
            hasattr(self.model, name)
            for name in (
                "first_hit_frozen_start_case_embedding",
                "autoregressive_frozen_start_case_embedding",
            )
        )

    def _sample_metrics_model_needs_case_indices(self):
        needs_frozen_ar_case_probe = (
            bool(
                getattr(
                    self.model,
                    "autoregressive_use_start_topology_conditioning",
                    False,
                )
            )
            and getattr(
                self.model,
                "autoregressive_start_topology_conditioning_mode",
                "additive",
            )
            in {"frozen_case_probe", "frozen_case_probe_additive"}
        )
        return (
            getattr(self.model, "first_hit_head_mode", "base")
            in {"case_adapted_mlp", "frozen_start_case_mlp"}
            or getattr(self.model, "autoregressive_use_case_conditioning", False)
            or needs_frozen_ar_case_probe
        )

    def _get_sample_metrics_metric_encoder(self):
        path = self.sample_metrics_unseen_start_metric_encoder_path
        if not path:
            return None
        cached = self._sample_metrics_metric_encoder_cache.get(path)
        if cached is not None:
            return cached

        from scripts.pretrain_start_tree_metric_encoder import SplitSetEncoder

        checkpoint = torch.load(path, map_location="cpu")
        metadata = dict(checkpoint.get("metadata", {}))
        max_bits = int(metadata.get("max_bits", 64))
        hidden_dim = int(metadata.get("hidden_dim", 128))
        embedding_dim = int(metadata.get("embedding_dim", 64))
        encoder = SplitSetEncoder(
            max_bits=max_bits,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=0.0,
        )
        state_dict = checkpoint.get("encoder_state_dict")
        if state_dict is None and checkpoint.get("model_state_dict") is not None:
            state_dict = {
                key[len("encoder.") :]: value
                for key, value in checkpoint["model_state_dict"].items()
                if str(key).startswith("encoder.")
            }
        if not state_dict:
            raise ValueError(
                "Metric encoder checkpoint must contain encoder_state_dict or "
                f"an encoder.* model_state_dict: {path}"
            )
        encoder.load_state_dict(state_dict)
        encoder.eval()
        cached = {
            "encoder": encoder,
            "max_bits": max_bits,
            "embedding_dim": embedding_dim,
        }
        self._sample_metrics_metric_encoder_cache[path] = cached
        return cached

    def _sample_metrics_encode_metric_starts(self, start_trees):
        encoder_bundle = self._get_sample_metrics_metric_encoder()
        if encoder_bundle is None:
            return None, {}

        from scripts.pretrain_start_tree_metric_encoder import (
            build_tree_tensor_batch,
            canonical_internal_splits,
            embedding_stats,
        )

        split_sets = []
        n_taxa = []
        for tree in start_trees:
            splits, inferred_n_taxa = canonical_internal_splits(str(tree))
            split_sets.append(splits)
            n_taxa.append(int(inferred_n_taxa))

        device = torch.device("cpu")
        encoder = encoder_bundle["encoder"].to(device)
        split_bits, pad_mask, size_features = build_tree_tensor_batch(
            split_sets,
            n_taxa,
            max_bits=int(encoder_bundle["max_bits"]),
            device=device,
        )
        with torch.no_grad():
            embeddings = encoder(split_bits, pad_mask, size_features)
            embeddings = F.normalize(embeddings, dim=-1)
        stats = embedding_stats(embeddings)
        return embeddings.detach().cpu(), stats

    def _sample_metrics_replace_frozen_start_case_tables(self, embeddings):
        replacements = []
        if embeddings is None:
            return replacements
        for name in (
            "first_hit_frozen_start_case_embedding",
            "autoregressive_frozen_start_case_embedding",
        ):
            if not hasattr(self.model, name):
                continue
            current = getattr(self.model, name)
            replacements.append((name, current))
            table = embeddings.to(device=current.device, dtype=current.dtype)
            setattr(self.model, name, table)
        return replacements

    def _sample_metrics_restore_frozen_start_case_tables(self, replacements):
        for name, value in reversed(replacements):
            setattr(self.model, name, value)

    def _sample_compare_harness_unseen_starts(self, train=True):
        dataset_split = self._sample_metrics_dataset_split(train=train)
        pairs = self._sample_metrics_unseen_bank_pairs(dataset_split, train=train)
        if not pairs:
            return {"num_pairs": 0, "unseen_start_eval": 1.0}

        needs_case_indices = self._sample_metrics_model_needs_case_indices()
        uses_frozen_tables = self._sample_metrics_model_uses_frozen_start_case_table()
        if needs_case_indices and not uses_frozen_tables:
            raise RuntimeError(
                "Unseen-start eval does not support trainable case-ID conditioning; "
                "use no conditioning or a frozen start-case table."
            )

        embeddings = None
        embedding_stats_block = {}
        if uses_frozen_tables:
            embeddings, embedding_stats_block = self._sample_metrics_encode_metric_starts(
                [pair["start_tree"] for pair in pairs]
            )
            if embeddings is None:
                raise RuntimeError(
                    "sample_metrics_unseen_start_metric_encoder_path is required "
                    "when unseen-start eval is used with frozen case-table conditioning."
                )

        rows = []
        replacements = self._sample_metrics_replace_frozen_start_case_tables(embeddings)
        try:
            for pair in pairs:
                rows.append(self._sample_compare_harness_once(pair, train=train))
        finally:
            self._sample_metrics_restore_frozen_start_case_tables(replacements)

        metrics = self._summarize_sample_compare_harness_rows(rows, train=train)
        metrics.update(
            {
                "unseen_start_eval": 1.0,
                "unseen_start_count": float(len(pairs)),
                "unseen_start_unique_count": float(
                    len({str(pair["start_tree"]) for pair in pairs})
                ),
                "unseen_start_unique_topology_count": float(
                    len(
                        {
                            canonicalize_topology_newick(str(pair["start_tree"]))
                            for pair in pairs
                        }
                    )
                ),
                "unseen_source_bank_unique_count": float(
                    len({int(pair["source_bank_index"]) for pair in pairs})
                ),
            }
        )
        for key, value in embedding_stats_block.items():
            metrics[f"unseen_start_embedding_{key}"] = float(value)
        return metrics

    def _sample_metrics_should_iterate_dataset_indices(self, dataset_split):
        return bool(
            dataset_split is not None
            and (
                getattr(dataset_split, "topology_stream_index_jsonl_path", None)
                or getattr(dataset_split, "sample_metrics_iterate_dataset_indices", False)
            )
        )

    def _sample_metrics_dataset_index_pairs(self, dataset_split, train=True):
        num_pairs = max(1, int(getattr(self, "sample_metrics_num_pairs", 1)))
        bank_size = self._sample_metrics_bank_size(dataset_split)
        indices = self._sample_metrics_select_bank_indices(
            bank_size,
            num_pairs,
            train=train,
        )
        return [
            self._sample_metrics_build_bank_pair(dataset_split, source_index)
            for source_index in indices
        ]

    def _add_zero_shot_random_start_metrics(self, metrics, train=True):
        if not getattr(self, "sample_metrics_zero_shot_random_start_eval", False):
            return metrics
        previous_label = getattr(self, "_sample_metrics_tree_dump_label", None)
        self._sample_metrics_tree_dump_label = "zero_shot_random_start"
        try:
            zero_shot_metrics = self._sample_compare_harness_unseen_starts(
                train=train
            )
        finally:
            if previous_label is None:
                try:
                    delattr(self, "_sample_metrics_tree_dump_label")
                except AttributeError:
                    pass
            else:
                self._sample_metrics_tree_dump_label = previous_label
        metrics.update(
            {
                f"zero_shot_random_start/{key}": value
                for key, value in zero_shot_metrics.items()
            }
        )
        return metrics

    def sample_compare_harness(self, train=True):
        if self.use_historical_sampling_impl:
            return _call_historical_trainingmodule_method(
                "sample_compare_harness",
                self,
                train=train,
            )
        if getattr(self, "sample_metrics_unseen_start_eval", False):
            return self._sample_compare_harness_unseen_starts(train=train)
        num_pairs = max(1, int(getattr(self, "sample_metrics_num_pairs", 1)))
        rows = []
        dataset_split = self._sample_metrics_dataset_split(train=train)
        if self._sample_metrics_should_iterate_dataset_indices(dataset_split):
            for pair in self._sample_metrics_dataset_index_pairs(
                dataset_split,
                train=train,
            ):
                rows.append(self._sample_compare_harness_once(pair, train=train))
            metrics = self._summarize_sample_compare_harness_rows(rows, train=train)
            return self._add_zero_shot_random_start_metrics(metrics, train=train)
        if getattr(dataset_split, "overfit_fixed_pair", False) and int(num_pairs) == 1:
            fixed_pair = self._get_fixed_pair_sampling_details(train=train)
            if fixed_pair is not None:
                pair = {
                    "start_tree": fixed_pair.get("random_tree", fixed_pair.get("start_tree")),
                    "target_tree": fixed_pair.get(
                        "effective_target_tree", fixed_pair.get("target_tree")
                    ),
                    "bank_group_key": fixed_pair.get("bank_group_key")
                    or fixed_pair.get("group_key"),
                    "dataset_id": fixed_pair.get("dataset_id"),
                    "n_leaves": len(
                        EteTree(
                            fixed_pair.get("random_tree", fixed_pair.get("start_tree")),
                            format=1,
                        ).get_leaves()
                    ),
                    "max_events": int(
                        len(fixed_pair.get("final_labels", []) or [])
                        or fixed_pair.get("fixed_pair_num_events", 1024)
                    ),
                    "name_mapping": (
                        fixed_pair.get("name_mapping")
                        if fixed_pair.get("name_mapping") is not None
                        else (
                            dataset_split.return_nexus_number_to_name(0)
                            if hasattr(dataset_split, "return_nexus_number_to_name")
                            else None
                        )
                    ),
                }
                rows.append(self._sample_compare_harness_once(pair, train=train))
                metrics = self._summarize_sample_compare_harness_rows(rows, train=train)
                metrics.update(self._evaluate_fixed_pair_path_metrics(train=train))
                return self._add_zero_shot_random_start_metrics(metrics, train=train)
        if (
            getattr(dataset_split, "overfit_full_path_control_mode", False)
            and getattr(dataset_split, "_frozen_full_path_control_selections", None)
        ):
            max_pairs = min(
                int(num_pairs),
                len(getattr(dataset_split, "_frozen_full_path_control_selections", [])),
            )
            for pair_index in range(max_pairs):
                fixed_pair = None
                if hasattr(dataset_split, "get_overfit_fixed_pair"):
                    fixed_pair = dataset_split.get_overfit_fixed_pair(pair_index)
                sampled = None
                if fixed_pair is None:
                    sampled = dataset_split[pair_index]
                    if hasattr(dataset_split, "get_overfit_fixed_pair"):
                        fixed_pair = dataset_split.get_overfit_fixed_pair(pair_index)
                if fixed_pair is not None:
                    start_tree = fixed_pair.get("random_tree", fixed_pair.get("start_tree"))
                    target_tree = fixed_pair.get(
                        "effective_target_tree", fixed_pair.get("target_tree")
                    )
                    max_events_value = int(
                        len(fixed_pair.get("final_labels", []) or [])
                        or fixed_pair.get("fixed_pair_num_events", 1024)
                    )
                    name_mapping_value = fixed_pair.get("name_mapping")
                    bank_group_key_value = fixed_pair.get(
                        "bank_group_key"
                    ) or fixed_pair.get("group_key")
                    dataset_id_value = fixed_pair.get("dataset_id")
                else:
                    start_tree = sampled.get("start_tree")
                    target_tree = sampled.get("target_tree")
                    max_events_value = int(sampled.get("fixed_pair_num_events", 1024))
                    name_mapping_value = None
                    bank_group_key_value = sampled.get("bank_group_key") or sampled.get(
                        "group_key"
                    )
                    dataset_id_value = sampled.get("dataset_id")
                pair = {
                    "start_tree": start_tree,
                    "target_tree": target_tree,
                    "bank_group_key": bank_group_key_value,
                    "dataset_id": dataset_id_value,
                    "n_leaves": len(EteTree(start_tree, format=1).get_leaves()),
                    "max_events": max_events_value,
                    "name_mapping": (
                        name_mapping_value
                        if name_mapping_value is not None
                        else (
                            dataset_split.return_nexus_number_to_name(0)
                            if hasattr(dataset_split, "return_nexus_number_to_name")
                            else None
                        )
                    ),
                }
                rows.append(self._sample_compare_harness_once(pair, train=train))
        else:
            for _ in range(num_pairs):
                pair = self._get_harness_sampling_pair(
                    train=train,
                    frozen_start_bank=True,
                )
                rows.append(self._sample_compare_harness_once(pair, train=train))

        metrics = self._summarize_sample_compare_harness_rows(rows, train=train)
        if num_pairs == 1:
            metrics.update(self._evaluate_fixed_pair_path_metrics(train=train))
        return self._add_zero_shot_random_start_metrics(metrics, train=train)

    def _branch_relax_training_loss(self):
        if (
            self.branch_relax_head is None
            or self.branch_relax_head_weight <= 0.0
            or not self.branch_relax_samples
        ):
            return None, {}
        batch_size = min(int(self.branch_relax_batch_size), len(self.branch_relax_samples))
        samples = random.sample(self.branch_relax_samples, k=batch_size)
        tokenized = _move_tokenized_batch_to_device(
            self.model.tokenizer([sample["newick_tree"] for sample in samples]),
            self.device,
        )
        case_indices = torch.tensor(
            [int(sample["case_index"]) for sample in samples],
            dtype=torch.long,
            device=self.device,
        )
        (
            _velocity,
            edge_splits,
            _edge_mask,
            _first_hit_logits,
            _boundary_vanish_logits,
            edge_features,
        ) = self.forward(
            tokenized,
            torch.tensor([4.0], dtype=torch.float32, device=self.device),
            None,
            first_hit_case_indices=case_indices,
        )
        if edge_features is None:
            return None, {}
        preds = []
        labels = []
        for batch_idx, sample in enumerate(samples):
            entries, _lengths, _n_leaves, _mapping = _branch_relax_entries_for_tree(
                self,
                sample["newick_tree"],
                edge_splits[batch_idx],
                labels=sample["labels"],
            )
            if not entries:
                continue
            feature_block = torch.stack(
                [edge_features[batch_idx, entry["edge_index"]] for entry in entries],
                dim=0,
            )
            if self.branch_relax_detach_trunk:
                feature_block = feature_block.detach()
            numeric = torch.tensor(
                [entry["numeric"] for entry in entries],
                dtype=torch.float32,
                device=self.device,
            )
            case_block = torch.full(
                (len(entries),),
                int(sample["case_index"]),
                dtype=torch.long,
                device=self.device,
            )
            preds.append(self.branch_relax_head(feature_block, numeric, case_block))
            labels.append(
                torch.tensor(
                    [float(entry["label"]) for entry in entries],
                    dtype=torch.float32,
                    device=self.device,
                )
            )
        if not preds:
            return None, {}
        pred = torch.cat(preds)
        target = torch.cat(labels)
        diff = pred - target
        loss = diff.pow(2).mean()
        logs = {
            "train/branch_relax_loss_unscaled": loss.detach(),
            "train/branch_relax_mae": diff.abs().mean().detach(),
            "train/branch_relax_sign_acc": (
                ((pred > 0.0) == (target > 0.0)).float().mean().detach()
            ),
        }
        return loss, logs

    def _should_collect_rollout_replay(self):
        if (
            self.rollout_replay_velocity_weight <= 0.0
            and self.rollout_replay_autoregressive_weight <= 0.0
        ):
            return False
        if int(self.stepper) < self.rollout_replay_start_step:
            return False
        return (
            (int(self.stepper) - self.rollout_replay_start_step)
            % self.rollout_replay_frequency
        ) == 0

    def _should_collect_dynamic_start_bank_rollout(self):
        if not self.dynamic_start_bank_enabled:
            return False
        if int(self.stepper) < self.dynamic_start_bank_start_step:
            return False
        frequency = max(int(self.rollout_replay_frequency), 1)
        return (
            (int(self.stepper) - self.dynamic_start_bank_start_step) % frequency
        ) == 0

    def _cached_rollout_replay_entry(self, train=True):
        cache_key = "train" if train else "val"
        return self._cached_rollout_replay_batches[cache_key]

    def _build_rollout_replay_reuse_logs(
        self,
        cache_entry,
        *,
        candidate_sampled_rf_norm=None,
        cache_refreshed=0.0,
        cache_reused=1.0,
        cache_refresh_rejected=0.0,
    ):
        replay_logs = dict(cache_entry["logs"])
        replay_logs["replay/cache_refreshed"] = torch.tensor(
            float(cache_refreshed), dtype=torch.float32, device=self.device
        )
        replay_logs["replay/cache_reused"] = torch.tensor(
            float(cache_reused), dtype=torch.float32, device=self.device
        )
        replay_logs["replay/cache_refresh_rejected"] = torch.tensor(
            float(cache_refresh_rejected), dtype=torch.float32, device=self.device
        )
        cached_sampled_rf_norm = cache_entry.get("sampled_rf_norm")
        if cached_sampled_rf_norm is not None:
            replay_logs["replay/cached_sampled_rf_norm"] = torch.tensor(
                float(cached_sampled_rf_norm),
                dtype=torch.float32,
                device=self.device,
            )
        if candidate_sampled_rf_norm is not None:
            replay_logs["replay/candidate_sampled_rf_norm"] = torch.tensor(
                float(candidate_sampled_rf_norm),
                dtype=torch.float32,
                device=self.device,
            )
        return replay_logs

    def _collect_rollout_replay_batches(self, train=True):
        replay_enabled = (
            self.rollout_replay_velocity_weight > 0.0
            or self.rollout_replay_autoregressive_weight > 0.0
        )
        dynamic_bank_rollout_only = (
            (not replay_enabled) and self._should_collect_dynamic_start_bank_rollout()
        )
        if (
            (not replay_enabled)
            and (not dynamic_bank_rollout_only)
        ):
            return None, None, {}

        cache_entry = self._cached_rollout_replay_entry(train=train)
        should_refresh = self._should_collect_rollout_replay()
        if dynamic_bank_rollout_only:
            should_refresh = True
        use_bank_sampling = (
            self.rollout_replay_mode == "filtered_polytomy_bank_oracle"
        )
        if (
            (not dynamic_bank_rollout_only)
            and (not should_refresh)
            and (not self.rollout_replay_cache_reuse_every_step)
        ):
            return None, None, {}
        if (
            (not dynamic_bank_rollout_only)
            and (not should_refresh)
            and (
            cache_entry["velocity"] is not None
            or cache_entry["autoregressive"] is not None
            )
        ):
            replay_logs = self._build_rollout_replay_reuse_logs(cache_entry)
            if use_bank_sampling and (
                cache_entry["velocity_bank"] is not None
                or cache_entry["autoregressive_bank"] is not None
            ):
                velocity_samples = _sample_replay_bank_samples(
                    cache_entry["velocity_bank"],
                    self.rollout_replay_max_velocity_states,
                )
                autoregressive_samples = _sample_replay_bank_samples(
                    cache_entry["autoregressive_bank"],
                    self.rollout_replay_max_autoregressive_states,
                )
                replay_logs["replay/num_velocity_states"] = torch.tensor(
                    float(len(velocity_samples)),
                    dtype=torch.float32,
                    device=self.device,
                )
                replay_logs["replay/num_autoregressive_states"] = torch.tensor(
                    float(len(autoregressive_samples)),
                    dtype=torch.float32,
                    device=self.device,
                )
                return (
                    _build_velocity_replay_batch(self, velocity_samples),
                    _build_autoregressive_replay_batch(self, autoregressive_samples),
                    replay_logs,
                )
            return (
                cache_entry["velocity"],
                cache_entry["autoregressive"],
                replay_logs,
            )
        if not should_refresh:
            return None, None, {}

        pair = self._get_harness_sampling_pair(train=train)
        if pair is None:
            return None, None, {}

        sample_kwargs = self._build_harness_sample_kwargs(
            pair,
            train=train,
            rollout_kind="replay",
        )
        split_multi_label_events = bool(
            sample_kwargs.get("split_multi_label_events", False)
        )
        if (
            self.rollout_replay_mode == "legacy_prefix_oracle"
            and self.rollout_replay_prefix_stop_early
        ):
            sample_kwargs["prefix_replay_velocity_quota"] = int(
                self.rollout_replay_max_velocity_states
            )
            sample_kwargs["prefix_replay_autoregressive_quota"] = int(
                self.rollout_replay_max_autoregressive_states
            )
            sample_kwargs["prefix_replay_split_multi_label_events"] = bool(
                split_multi_label_events
            )
        was_training = self.model.training
        try:
            with torch.no_grad():
                sample_outputs = self.sample(
                    [pair["start_tree"]],
                    **sample_kwargs,
                    topology_repeat_cap=self.rollout_replay_topology_repeat_cap,
                )
        finally:
            self.model.train(was_training)

        sampled_trees, _, _, _, _, trace = sample_outputs
        sampled_rf_norm = float(calculate_norm_rf(sampled_trees[0], pair["target_tree"]))

        if dynamic_bank_rollout_only:
            replay_logs = {
                "replay/sampled_rf_norm": torch.tensor(
                    sampled_rf_norm,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "replay/num_velocity_states": torch.tensor(
                    float(len(trace.get("velocity", []))),
                    dtype=torch.float32,
                    device=self.device,
                ),
                "replay/num_autoregressive_states": torch.tensor(
                    float(len(trace.get("autoregressive", []))),
                    dtype=torch.float32,
                    device=self.device,
                ),
            }
            if self.sample_metrics_trace_topology_repeats_enabled:
                repeat_summary = _summarize_trace_topology_repeats(trace)
                for key, value in repeat_summary.items():
                    replay_logs[f"replay/{key}"] = torch.tensor(
                        float(value),
                        dtype=torch.float32,
                        device=self.device,
                    )
            replay_logs.update(
                self._maybe_update_dynamic_start_bank(
                    pair=pair,
                    sampled_tree=sampled_trees[0],
                    sampled_rf_norm=sampled_rf_norm,
                    trace=trace,
                    train=train,
                )
            )
            return None, None, replay_logs

        def _dedupe_samples(samples, key_fn):
            unique = []
            seen = set()
            for sample in samples:
                sample_key = key_fn(sample)
                if sample_key in seen:
                    continue
                seen.add(sample_key)
                unique.append(sample)
            return unique

        anchor_states = []
        first_wrong_velocity_replay = None
        if self.rollout_replay_mode == "anchor_oracle":
            anchor_states = _select_rollout_replay_anchors(
                trace,
                sampled_trees[0],
                pair["target_tree"],
                self.rollout_replay_anchor_states,
                include_autoregressive=self.rollout_replay_anchor_include_autoregressive,
            )
            velocity_samples, autoregressive_samples = (
                _collect_oracle_replay_samples_from_anchors(
                    self,
                    anchor_states,
                    self.rollout_replay_oracle_horizon,
                    split_multi_label_events=split_multi_label_events,
                )
            )
        elif self.rollout_replay_mode == "first_wrong_velocity_suffix_oracle":
            (
                velocity_samples,
                autoregressive_samples,
                first_wrong_velocity_replay,
            ) = _collect_first_wrong_velocity_suffix_replay_samples(
                self,
                pair=pair,
                trace=trace,
            )
        elif self.rollout_replay_mode == "legacy_prefix_oracle":
            velocity_samples, autoregressive_samples = (
                _collect_legacy_oracle_replay_samples_from_trace(
                    trace,
                    module=self,
                    split_multi_label_events=split_multi_label_events,
                )
            )
        elif self.rollout_replay_mode == "filtered_polytomy_bank_oracle":
            preselected_trace = dict(trace)
            preselected_trace["velocity"] = _sample_trace_states_uniform(
                trace.get("velocity", []),
                self.rollout_replay_max_velocity_states,
                time_key="timepoint",
            )
            preselected_trace["autoregressive"] = _sample_trace_states_uniform(
                trace.get("autoregressive", []),
                self.rollout_replay_max_autoregressive_states,
                time_key="time",
            )
            velocity_samples, autoregressive_samples = (
                _collect_legacy_oracle_replay_samples_from_trace(
                    preselected_trace,
                    module=self,
                    split_multi_label_events=split_multi_label_events,
                )
            )
        else:
            velocity_samples, autoregressive_samples = (
                _collect_legacy_oracle_replay_samples_from_trace(
                    trace,
                    module=self,
                    split_multi_label_events=split_multi_label_events,
                    terminal_tree=sampled_trees[0],
                    terminal_target_tree=pair["target_tree"],
                )
            )

        velocity_samples = _dedupe_samples(
            velocity_samples,
            key_fn=lambda sample: (
                str(sample["newick_tree"]),
                round(float(sample["timepoint"]), 8),
            ),
        )
        velocity_samples = [
            sample
            for sample in velocity_samples
            if float(sample.get("timepoint", 0.0)) <= 1.0
        ]
        velocity_oracle_orthant_relabel_stats = {
            "matched": 0,
            "unmatched": 0,
            "canonical_topologies": 0,
            "ambiguous_topologies": 0,
        }
        if self.rollout_replay_velocity_use_pair_oracle_orthant_labels:
            velocity_samples, velocity_oracle_orthant_relabel_stats = (
                _apply_pair_oracle_orthant_velocity_labels(velocity_samples, pair)
            )
        autoregressive_samples = _dedupe_samples(
            autoregressive_samples,
            key_fn=lambda sample: (
                str(sample["newick"]),
                round(float(sample["time"]), 8),
            ),
        )
        valid_autoregressive_samples = []
        invalid_autoregressive_samples = 0
        for sample in autoregressive_samples:
            structural_groups = {
                tuple(int(component) for component in group)
                for group in get_structural_polytomy_groups_from_newick(
                    sample["newick"]
                )
            }
            label_groups = {
                tuple(int(component) for component in label["components"])
                for label in sample["labels"]
            }
            if label_groups.issubset(structural_groups):
                valid_autoregressive_samples.append(sample)
            else:
                invalid_autoregressive_samples += 1
        autoregressive_samples = valid_autoregressive_samples
        bank_filtered_out_velocity_samples = 0
        bank_filtered_out_autoregressive_samples = 0
        if use_bank_sampling:
            filtered_velocity_samples = _filter_replay_samples_by_max_polytomy(
                velocity_samples,
                tree_key="newick_tree",
                max_polytomy_size=self.rollout_replay_bank_max_polytomy_size,
            )
            filtered_autoregressive_samples = _filter_replay_samples_by_max_polytomy(
                autoregressive_samples,
                tree_key="newick",
                max_polytomy_size=self.rollout_replay_bank_max_polytomy_size,
            )
            bank_filtered_out_velocity_samples = max(
                len(velocity_samples) - len(filtered_velocity_samples),
                0,
            )
            bank_filtered_out_autoregressive_samples = max(
                len(autoregressive_samples) - len(filtered_autoregressive_samples),
                0,
            )
            velocity_samples = filtered_velocity_samples
            autoregressive_samples = filtered_autoregressive_samples

        max_velocity_states = int(self.rollout_replay_max_velocity_states)
        max_autoregressive_states = int(self.rollout_replay_max_autoregressive_states)
        if self.rollout_replay_mode == "anchor_oracle":
            velocity_samples = _select_replay_samples_across_rollout(
                velocity_samples,
                max_velocity_states,
            )
            autoregressive_samples = _select_replay_samples_across_rollout(
                autoregressive_samples,
                max_autoregressive_states,
            )
        elif self.rollout_replay_mode in {"legacy_prefix_oracle", "first_wrong_velocity_suffix_oracle"}:
            velocity_samples = velocity_samples[:max_velocity_states]
            autoregressive_samples = autoregressive_samples[:max_autoregressive_states]
        elif use_bank_sampling:
            velocity_bank_samples = list(velocity_samples)
            autoregressive_bank_samples = list(autoregressive_samples)
            velocity_samples = _sample_replay_bank_samples(
                velocity_bank_samples,
                max_velocity_states,
            )
            autoregressive_samples = _sample_replay_bank_samples(
                autoregressive_bank_samples,
                max_autoregressive_states,
            )
        else:
            velocity_samples = _select_legacy_prefix_suffix_replay_samples(
                velocity_samples,
                max_velocity_states,
                tree_key="newick_tree",
            )
            autoregressive_samples = _select_legacy_prefix_suffix_replay_samples(
                autoregressive_samples,
                max_autoregressive_states,
                tree_key="newick",
            )

        all_velocity_samples = list(velocity_samples)
        all_autoregressive_samples = list(autoregressive_samples)
        if use_bank_sampling:
            all_velocity_samples = list(velocity_bank_samples)
            all_autoregressive_samples = list(autoregressive_bank_samples)

        replay_logs = {
            "replay/sampled_rf_norm": torch.tensor(
                sampled_rf_norm,
                dtype=torch.float32,
                device=self.device,
            ),
            "replay/num_velocity_states": torch.tensor(
                float(len(velocity_samples)),
                dtype=torch.float32,
                device=self.device,
            ),
            "replay/num_autoregressive_states": torch.tensor(
                float(len(autoregressive_samples)),
                dtype=torch.float32,
                device=self.device,
            ),
            "replay/num_invalid_autoregressive_states": torch.tensor(
                float(invalid_autoregressive_samples),
                dtype=torch.float32,
                device=self.device,
            ),
            "replay/bank_num_velocity_states": torch.tensor(
                float(len(all_velocity_samples)),
                dtype=torch.float32,
                device=self.device,
            ),
            "replay/bank_num_autoregressive_states": torch.tensor(
                float(len(all_autoregressive_samples)),
                dtype=torch.float32,
                device=self.device,
            ),
            "replay/bank_filtered_out_velocity_states": torch.tensor(
                float(bank_filtered_out_velocity_samples),
                dtype=torch.float32,
                device=self.device,
            ),
            "replay/bank_filtered_out_autoregressive_states": torch.tensor(
                float(bank_filtered_out_autoregressive_samples),
                dtype=torch.float32,
                device=self.device,
            ),
            "replay/num_anchor_states": torch.tensor(
                float(len(anchor_states)),
                dtype=torch.float32,
                device=self.device,
            ),
            "replay/oracle_orthant_velocity_label_matches": torch.tensor(
                float(velocity_oracle_orthant_relabel_stats["matched"]),
                dtype=torch.float32,
                device=self.device,
            ),
            "replay/oracle_orthant_velocity_label_unmatched": torch.tensor(
                float(velocity_oracle_orthant_relabel_stats["unmatched"]),
                dtype=torch.float32,
                device=self.device,
            ),
            "replay/oracle_orthant_velocity_label_topologies": torch.tensor(
                float(velocity_oracle_orthant_relabel_stats["canonical_topologies"]),
                dtype=torch.float32,
                device=self.device,
            ),
            "replay/oracle_orthant_velocity_label_ambiguous_topologies": torch.tensor(
                float(velocity_oracle_orthant_relabel_stats["ambiguous_topologies"]),
                dtype=torch.float32,
                device=self.device,
            ),
            "replay/cache_refreshed": torch.tensor(
                1.0, dtype=torch.float32, device=self.device
            ),
            "replay/cache_reused": torch.tensor(
                0.0, dtype=torch.float32, device=self.device
            ),
            "replay/stopped_for_repeated_topology": torch.tensor(
                1.0 if trace.get("stopped_for_repeated_topology", False) else 0.0,
                dtype=torch.float32,
                device=self.device,
            ),
            "replay/stopped_for_no_valid_merge": torch.tensor(
                1.0 if trace.get("stopped_for_no_valid_merge", False) else 0.0,
                dtype=torch.float32,
                device=self.device,
            ),
            "replay/skipped_no_valid_boundary_revisits": torch.tensor(
                float(trace.get("skipped_no_valid_boundary_revisits", 0.0)),
                dtype=torch.float32,
                device=self.device,
            ),
        }
        repeat_summary = {}
        if self.sample_metrics_trace_topology_repeats_enabled:
            repeat_summary = _summarize_trace_topology_repeats(trace)
            for key, value in repeat_summary.items():
                replay_logs[f"replay/{key}"] = torch.tensor(
                    float(value),
                    dtype=torch.float32,
                    device=self.device,
                )
        if first_wrong_velocity_replay is not None:
            replay_logs["replay/first_wrong_velocity_phase"] = torch.tensor(
                float(first_wrong_velocity_replay.get("first_wrong_phase_idx", -1)),
                dtype=torch.float32,
                device=self.device,
            )
            replay_logs["replay/found_first_wrong_velocity_state"] = torch.tensor(
                1.0,
                dtype=torch.float32,
                device=self.device,
            )
            replay_logs["replay/first_wrong_velocity_start_rf_to_oracle_start"] = torch.tensor(
                float(first_wrong_velocity_replay.get("sampled_start_rf_to_oracle_start", 0.0)),
                dtype=torch.float32,
                device=self.device,
            )
            if first_wrong_velocity_replay.get("sampled_boundary_rf_to_oracle_boundary") is not None:
                replay_logs["replay/first_wrong_velocity_boundary_rf_to_oracle_boundary"] = torch.tensor(
                    float(first_wrong_velocity_replay.get("sampled_boundary_rf_to_oracle_boundary", 0.0)),
                    dtype=torch.float32,
                    device=self.device,
                )
        elif self.rollout_replay_mode == "first_wrong_velocity_suffix_oracle":
            replay_logs["replay/first_wrong_velocity_phase"] = torch.tensor(
                -1.0,
                dtype=torch.float32,
                device=self.device,
            )
            replay_logs["replay/found_first_wrong_velocity_state"] = torch.tensor(
                0.0,
                dtype=torch.float32,
                device=self.device,
            )
        replay_logs.update(
            self._maybe_update_dynamic_start_bank(
                pair=pair,
                sampled_tree=sampled_trees[0],
                sampled_rf_norm=sampled_rf_norm,
                trace=trace,
                train=train,
            )
        )
        velocity_batch = _build_velocity_replay_batch(self, velocity_samples)
        autoregressive_batch = _build_autoregressive_replay_batch(
            self, autoregressive_samples
        )
        cached_sampled_rf_norm = cache_entry.get("sampled_rf_norm")
        keep_existing_cache = (
            self.rollout_replay_refresh_only_if_better_rf
            and cached_sampled_rf_norm is not None
            and sampled_rf_norm >= float(cached_sampled_rf_norm)
        )
        if keep_existing_cache:
            replay_logs = self._build_rollout_replay_reuse_logs(
                cache_entry,
                candidate_sampled_rf_norm=sampled_rf_norm,
                cache_refreshed=0.0,
                cache_reused=1.0,
                cache_refresh_rejected=1.0,
            )
            if use_bank_sampling and (
                cache_entry["velocity_bank"] is not None
                or cache_entry["autoregressive_bank"] is not None
            ):
                velocity_samples = _sample_replay_bank_samples(
                    cache_entry["velocity_bank"],
                    self.rollout_replay_max_velocity_states,
                )
                autoregressive_samples = _sample_replay_bank_samples(
                    cache_entry["autoregressive_bank"],
                    self.rollout_replay_max_autoregressive_states,
                )
                replay_logs["replay/num_velocity_states"] = torch.tensor(
                    float(len(velocity_samples)),
                    dtype=torch.float32,
                    device=self.device,
                )
                replay_logs["replay/num_autoregressive_states"] = torch.tensor(
                    float(len(autoregressive_samples)),
                    dtype=torch.float32,
                    device=self.device,
                )
                return (
                    _build_velocity_replay_batch(self, velocity_samples),
                    _build_autoregressive_replay_batch(
                        self, autoregressive_samples
                    ),
                    replay_logs,
                )
            return (
                cache_entry["velocity"],
                cache_entry["autoregressive"],
                replay_logs,
            )

        self._dump_rollout_replay_refresh(
            train=train,
            sampled_tree=sampled_trees[0],
            target_tree=pair["target_tree"],
            sampled_rf_norm=sampled_rf_norm,
            trace=trace,
            anchor_states=anchor_states,
            all_velocity_samples=all_velocity_samples,
            all_autoregressive_samples=all_autoregressive_samples,
            selected_velocity_samples=velocity_samples,
            selected_autoregressive_samples=autoregressive_samples,
            invalid_autoregressive_samples=invalid_autoregressive_samples,
            replay_logs=replay_logs,
            repeat_summary=repeat_summary,
        )
        cache_entry["velocity"] = velocity_batch
        cache_entry["autoregressive"] = autoregressive_batch
        cache_entry["velocity_bank"] = (
            list(all_velocity_samples) if use_bank_sampling else None
        )
        cache_entry["autoregressive_bank"] = (
            list(all_autoregressive_samples) if use_bank_sampling else None
        )
        cache_entry["sampled_rf_norm"] = float(sampled_rf_norm)
        cache_entry["logs"] = replay_logs
        return velocity_batch, autoregressive_batch, dict(replay_logs)

    def _rollout_replay_refresh_dir(self):
        if self.rollout_replay_dump_dir is not None:
            base_dir = self.rollout_replay_dump_dir
        else:
            checkpoint_callback = getattr(self.trainer, "checkpoint_callback", None)
            checkpoint_dir = getattr(checkpoint_callback, "dirpath", None)
            if checkpoint_dir:
                base_dir = os.path.join(checkpoint_dir, "replay_refreshes")
            else:
                base_dir = os.path.join(os.getcwd(), "replay_refreshes")
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    def _dump_rollout_replay_refresh(
        self,
        *,
        train,
        sampled_tree,
        target_tree,
        sampled_rf_norm,
        trace,
        anchor_states,
        all_velocity_samples,
        all_autoregressive_samples,
        selected_velocity_samples,
        selected_autoregressive_samples,
        invalid_autoregressive_samples,
        replay_logs,
        repeat_summary,
    ):
        if not self.rollout_replay_dump_refreshes:
            return

        try:
            dump_dir = self._rollout_replay_refresh_dir()
            phase = "train" if train else "val"
            file_step = int(self.stepper)
            self._rollout_replay_dump_counter += 1
            dump_name = (
                f"{phase}_step_{file_step:06d}_refresh_"
                f"{self._rollout_replay_dump_counter:06d}.json"
            )
            dump_path = os.path.join(dump_dir, dump_name)
            replay_logs_json = {}
            for key, value in replay_logs.items():
                if isinstance(value, torch.Tensor):
                    replay_logs_json[key] = _to_jsonable(value.detach().cpu())
                else:
                    replay_logs_json[key] = _to_jsonable(value)
            payload = {
                "phase": phase,
                "stepper": int(self.stepper),
                "global_step": int(self.global_step),
                "current_epoch": int(self.current_epoch),
                "rollout_replay_mode": str(self.rollout_replay_mode),
                "sampling_fixed_dt_base": (
                    None
                    if self.sampling_fixed_dt_base is None
                    else float(self.sampling_fixed_dt_base)
                ),
                "rollout_replay_fixed_dt_base": (
                    None
                    if self.rollout_replay_fixed_dt_base is None
                    else float(self.rollout_replay_fixed_dt_base)
                ),
                "sampled_tree": str(sampled_tree),
                "target_tree": str(target_tree),
                "sampled_rf_norm": float(sampled_rf_norm),
                "trace": _to_jsonable(trace),
                "anchor_states": _to_jsonable(anchor_states),
                "all_velocity_samples": _to_jsonable(all_velocity_samples),
                "all_autoregressive_samples": _to_jsonable(
                    all_autoregressive_samples
                ),
                "selected_velocity_samples": _to_jsonable(
                    selected_velocity_samples
                ),
                "selected_autoregressive_samples": _to_jsonable(
                    selected_autoregressive_samples
                ),
                "invalid_autoregressive_samples": int(
                    invalid_autoregressive_samples
                ),
                "repeat_summary": _to_jsonable(repeat_summary),
                "replay_logs": replay_logs_json,
            }
            tmp_path = dump_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            os.replace(tmp_path, dump_path)
        except Exception as exc:
            logger.warning("Failed to dump replay refresh: %s", exc)

    def compute_phyla_embeddings(self, sequences, names, device="cuda"):
        """
        Generates Phyla embeddings for a batch of sequences.
        """
        if self.phyla_model is None:
            raise ValueError("Phyla model not loaded.")

        # This utility handles tokenization, padding, and CLS token placement
        _, _, _, _encode_sequences_openfold_style = _load_phyla_runtime()
        batch, _ = _encode_sequences_openfold_style(sequences, names)

        # Generate Embeddings
        with torch.no_grad():
            encoded_seqs = batch["encoded_sequences"].to(device)
            sequence_mask = batch["sequence_mask"].to(device)
            cls_positions = batch["cls_positions"].bool().to(device)

            self.phyla_model.to(device)

            # Handle different forward pass signatures depending on model wrapper
            if "TrainingModule" in str(type(self.phyla_model)):
                embeddings = self.phyla_model(
                    encoded_seqs,
                    cls_token_mask=cls_positions,
                    sequence_mask=sequence_mask,
                )
            else:
                embeddings = self.phyla_model(
                    encoded_seqs,
                    sequence_mask,
                    cls_positions,
                )

        return embeddings

    @staticmethod
    def _strip_live_phyla_sequence(sequence):
        return str(sequence or "").replace("-", "").replace(".", "")

    def _ensure_live_phyla_device(self):
        if self.live_phyla_model is None:
            raise ValueError("Live Phyla model not loaded.")
        target = torch.device(
            self.live_phyla_device_config
            if self.live_phyla_device_config is not None
            else self.device
        )
        try:
            actual = next(self.live_phyla_model.parameters()).device
        except StopIteration:
            actual = torch.device(self._live_phyla_device or target)
        if str(actual) != str(target) or self._live_phyla_device != str(target):
            self.live_phyla_model.to(target)
            self.live_phyla_model.device = target
            self._live_phyla_device = str(target)
        return target

    def _compute_live_phyla_embeddings_case(
        self,
        sequences,
        names=None,
        *,
        grad: bool = True,
    ):
        if self.live_phyla_model is None:
            raise ValueError("Live Phyla model not loaded.")
        if sequences is None:
            return None
        sequences = [self._strip_live_phyla_sequence(sequence) for sequence in sequences]
        if names is None:
            names = [str(idx) for idx in range(len(sequences))]
        names = [str(name) for name in names]
        if len(names) != len(sequences):
            names = [str(idx) for idx in range(len(sequences))]
        input_tokens = int(sum(len(sequence) for sequence in sequences) + len(sequences))
        if (
            self.live_phyla_max_input_tokens > 0
            and input_tokens > self.live_phyla_max_input_tokens
        ):
            raise RuntimeError(
                f"Live Phyla input has {input_tokens} tokens, exceeding "
                f"live_phyla_max_input_tokens={self.live_phyla_max_input_tokens}."
            )
        device = self._ensure_live_phyla_device()
        use_grad = bool(grad and self.live_phyla_unfreeze)
        previous_training = self.live_phyla_model.training
        if use_grad:
            self.live_phyla_model.train()
            context = contextlib.nullcontext()
        else:
            self.live_phyla_model.eval()
            context = torch.no_grad()
        with context:
            encoded, cls_mask, seq_mask, _ = self.live_phyla_model.encode(
                sequences,
                names,
            )
            embeddings = self.live_phyla_model(
                encoded.to(device),
                seq_mask.to(device),
                cls_mask.to(device),
            )
        if previous_training:
            self.live_phyla_model.train()
        else:
            self.live_phyla_model.eval()
        if embeddings.dim() == 3 and embeddings.size(0) == 1:
            embeddings = embeddings.squeeze(0)
        return embeddings

    def _attach_shared_live_phyla_embeddings_for_batches(
        self,
        batches,
        *,
        grad: bool = True,
    ):
        if self.live_phyla_model is None:
            return False

        attach_specs = []
        unique_cases = {}
        for batch in batches:
            if not isinstance(batch, dict):
                continue
            if batch.get("phyla_embeddings") is not None:
                continue
            if batch.get("selected_sequences") is None:
                continue
            specs = _selected_sequence_case_specs(
                batch.get("selected_sequences"),
                batch.get("selected_sequence_names"),
            )
            if specs is None:
                return False
            for key, sequences, names in specs:
                unique_cases.setdefault(key, (sequences, names))
            attach_specs.append((batch, specs))

        if len(attach_specs) < 2 or not unique_cases:
            return False

        embeddings_by_key = {}
        for key, (sequences, names) in unique_cases.items():
            embeddings = self._compute_live_phyla_embeddings_case(
                sequences,
                names,
                grad=grad,
            )
            if embeddings is None:
                return False
            embeddings_by_key[key] = embeddings

        all_specs = [
            specs
            for _batch, specs in attach_specs
        ]
        if (
            len(unique_cases) == 1
            and all(
                specs
                and all(spec[0] == next(iter(unique_cases)) for spec in specs)
                for specs in all_specs
            )
        ):
            shared_embeddings = embeddings_by_key[next(iter(unique_cases))]
            for batch, _specs in attach_specs:
                batch["phyla_embeddings"] = shared_embeddings
            return True

        for batch, specs in attach_specs:
            batch["phyla_embeddings"] = [
                embeddings_by_key[key]
                for key, _sequences, _names in specs
            ]
        return True

    def _compute_live_phyla_embeddings_for_batch(self, batch, *, grad: bool = True):
        selected_sequences = batch.get("selected_sequences")
        if selected_sequences is None:
            return None
        selected_names = batch.get("selected_sequence_names")
        if isinstance(selected_sequences, tuple):
            selected_sequences = list(selected_sequences)
        if selected_sequences and all(
            isinstance(item, str) for item in selected_sequences
        ):
            selected_sequences = [selected_sequences]
        if selected_names is None:
            selected_names = [None] * len(selected_sequences)
        elif isinstance(selected_names, tuple):
            selected_names = list(selected_names)
        if selected_names and all(isinstance(item, str) for item in selected_names):
            selected_names = [selected_names]
        embeddings = []
        local_cache = {}
        for sequences, names in zip(selected_sequences, selected_names):
            if sequences is None:
                return None
            sequence_list = list(sequences)
            name_list = None if names is None else list(names)
            cache_key = (
                tuple(str(name) for name in (name_list or [])),
                tuple(str(sequence) for sequence in sequence_list),
                bool(grad),
            )
            if cache_key not in local_cache:
                local_cache[cache_key] = self._compute_live_phyla_embeddings_case(
                    sequence_list,
                    name_list,
                    grad=grad,
                )
            embeddings.append(local_cache[cache_key])
        return embeddings

    def _compute_live_phyla_embeddings_for_pair(self, pair):
        sequences = pair.get("selected_sequences")
        if sequences is None:
            return None
        names = pair.get("selected_sequence_names")
        embeddings = self._compute_live_phyla_embeddings_case(
            list(sequences),
            None if names is None else list(names),
            grad=False,
        )
        if embeddings is None:
            return None
        return embeddings.unsqueeze(0)

    def _infer_precomputed_phyla_dataset_ids(self):
        dataset_ids = set()
        data_module = getattr(self, "dataset", None)
        candidates = []
        for attr in (
            "dataset_train",
            "dataset_val",
            "dataset_test",
            "dataset_predict",
            "sample_metrics_dataset_train",
            "sample_metrics_dataset_val",
        ):
            dataset_obj = getattr(data_module, attr, None)
            if dataset_obj is not None:
                candidates.append(dataset_obj)
        if data_module is not None and not candidates:
            candidates.append(data_module)

        for dataset_obj in candidates:
            for meta in getattr(dataset_obj, "_index", []) or []:
                raw_id = meta.get("dataset_id")
                if raw_id is None:
                    raw_id = meta.get("id")
                if raw_id is not None:
                    dataset_ids.add(str(raw_id).upper())
            for raw_id in getattr(dataset_obj, "posterior_dataset_ids", []) or []:
                if raw_id is not None:
                    dataset_ids.add(str(raw_id).upper())

        return dataset_ids or None

    def _iter_precomputed_phyla_embedding_paths(self, path):
        raw_path = str(path)
        parts = [
            part
            for part in raw_path.split(os.pathsep)
            if part.strip()
        ]
        if len(parts) > 1 and not os.path.exists(raw_path):
            for part in parts:
                yield from self._iter_precomputed_phyla_embedding_paths(part)
            return

        if os.path.isdir(raw_path):
            dataset_ids = getattr(
                self,
                "phyla_precomputed_dataset_id_allowlist",
                None,
            )
            suffixes = (
                "_phyla_beta_embeddings.pt",
                "_phyla_beta_sitechunk_w256_s256_embeddings.pt",
            )
            if dataset_ids:
                for dataset_id in sorted(dataset_ids):
                    for suffix in suffixes:
                        candidate = os.path.join(raw_path, f"{dataset_id}{suffix}")
                        if os.path.exists(candidate):
                            yield candidate
                return

            candidates = []
            for filename in os.listdir(raw_path):
                if filename.endswith(suffixes):
                    candidates.append(os.path.join(raw_path, filename))
            for candidate in sorted(candidates):
                yield candidate
            return

        yield raw_path

    def _register_precomputed_phyla_embeddings(
        self,
        sequence_names,
        tensor,
        *,
        dataset_id=None,
    ):
        mapping = {
            str(name): tensor[idx].clone()
            for idx, name in enumerate(sequence_names)
        }
        if self.phyla_precomputed_name_to_embedding is None:
            self.phyla_precomputed_name_to_embedding = {}
        if dataset_id is None:
            self.phyla_precomputed_name_to_embedding.update(mapping)
            return
        dataset_key = str(dataset_id).upper()
        self.phyla_precomputed_dataset_name_to_embedding[dataset_key] = mapping
        self.phyla_precomputed_dataset_id_to_tensor[dataset_key] = tensor.clone()

    def _load_single_precomputed_phyla_embeddings(self, path):
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                "Expected a dict payload with 'sequence_names' and 'embeddings'."
            )

        sequence_names = payload.get("sequence_names")
        if sequence_names is None:
            sequence_names = payload.get("names")
        embeddings = payload.get("embeddings")
        if embeddings is None:
            embeddings = payload.get("phyla_embeddings")
        if sequence_names is None or embeddings is None:
            raise ValueError(
                "Precomputed Phyla file must contain 'sequence_names' and 'embeddings'."
            )

        if torch.is_tensor(embeddings):
            tensor = embeddings.detach().cpu().float()
        else:
            tensor = torch.as_tensor(embeddings, dtype=torch.float32)

        if tensor.dim() == 3:
            if tensor.size(0) != 1:
                raise ValueError(
                    f"Expected embeddings with leading batch size 1, got {tuple(tensor.shape)}."
                )
            tensor = tensor.squeeze(0)
        if tensor.dim() != 2:
            raise ValueError(
                f"Expected embeddings with shape [N, D], got {tuple(tensor.shape)}."
            )
        if len(sequence_names) != tensor.size(0):
            raise ValueError(
                f"Sequence name count {len(sequence_names)} does not match "
                f"embedding rows {tensor.size(0)}."
            )

        expected_dim = getattr(self.model, "phyla_dim", None)
        if expected_dim is None:
            phyla_proj = getattr(self.model, "phyla_proj", None)
            expected_dim = getattr(phyla_proj, "in_features", None)
            if expected_dim is None and phyla_proj is not None:
                for module in phyla_proj.modules():
                    if isinstance(module, nn.Linear):
                        expected_dim = module.in_features
                        break
        if expected_dim is None:
            raise ValueError("Could not infer model Phyla embedding dimension.")
        expected_dim = int(expected_dim)
        if tensor.size(1) != expected_dim:
            raise ValueError(
                f"Precomputed embedding dim {tensor.size(1)} does not match "
                f"model phyla_dim {expected_dim}."
            )

        dataset_id = payload.get("dataset_id")
        self._register_precomputed_phyla_embeddings(
            sequence_names,
            tensor,
            dataset_id=dataset_id,
        )

    def _load_precomputed_phyla_embeddings(self, path):
        loaded = 0
        for embedding_path in self._iter_precomputed_phyla_embedding_paths(path):
            if not os.path.exists(str(embedding_path)):
                raise FileNotFoundError(str(embedding_path))
            self._load_single_precomputed_phyla_embeddings(embedding_path)
            loaded += 1
        if loaded == 0:
            raise FileNotFoundError(f"No phyla embedding files found under {path!r}")
        dataset_count = len(
            getattr(self, "phyla_precomputed_dataset_name_to_embedding", {}) or {}
        )
        allowlist = getattr(self, "phyla_precomputed_dataset_id_allowlist", None)
        if allowlist:
            logging.info(
                "Loaded %d precomputed Phyla embedding file(s) for %d dataset bank(s) "
                "using a %d-ID dataset filter.",
                loaded,
                dataset_count,
                len(allowlist),
            )
        else:
            logging.info(
                "Loaded %d precomputed Phyla embedding file(s) for %d dataset bank(s).",
                loaded,
                dataset_count,
            )

    def _ordered_leaf_names_from_mapping(self, mapping, num_leaf=None):
        if mapping is None:
            return None
        ordered = []
        limit = None if num_leaf is None else int(num_leaf)
        for raw_idx, raw_name in mapping.items():
            if raw_name in (None, "", "ROOT_DUMMY"):
                continue
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            if limit is not None and idx >= limit:
                continue
            ordered.append((idx, str(raw_name)))
        if not ordered:
            return None
        ordered.sort(key=lambda item: item[0])
        return [name for _idx, name in ordered]

    def _ordered_leaf_names_from_newick(self, newick_tree):
        tree = Tree(newick_tree)
        names = []
        for idx in range(tree.n_leaves):
            name = str(tree.id_to_name[idx])
            if name != "ROOT_DUMMY":
                names.append(name)
        return names

    def _lookup_precomputed_phyla_embeddings(self, names, device=None, dataset_id=None):
        if not names:
            return None
        precomputed = None
        if dataset_id is not None:
            precomputed = getattr(
                self,
                "phyla_precomputed_dataset_name_to_embedding",
                {},
            ).get(str(dataset_id).upper())
        if not precomputed:
            precomputed = getattr(self, "phyla_precomputed_name_to_embedding", None)
        if not precomputed:
            return None
        missing = [
            str(name)
            for name in names
            if str(name) not in precomputed
        ]
        if missing:
            dataset_key = None if dataset_id is None else str(dataset_id).upper()
            tensor = getattr(
                self,
                "phyla_precomputed_dataset_id_to_tensor",
                {},
            ).get(dataset_key)
            if tensor is None:
                return None
            try:
                numeric_names = [int(str(name)) for name in names]
            except (TypeError, ValueError):
                return None
            if not numeric_names:
                return None
            min_idx = min(numeric_names)
            max_idx = max(numeric_names)
            if min_idx >= 1 and max_idx <= tensor.size(0):
                embeddings = torch.stack(
                    [tensor[idx - 1] for idx in numeric_names],
                    dim=0,
                )
            elif min_idx >= 0 and max_idx < tensor.size(0):
                embeddings = torch.stack(
                    [tensor[idx] for idx in numeric_names],
                    dim=0,
                )
            else:
                return None
            if device is not None:
                embeddings = embeddings.to(device)
            return embeddings
        embeddings = torch.stack(
            [
                precomputed[str(name)]
                for name in names
            ],
            dim=0,
        )
        if device is not None:
            embeddings = embeddings.to(device)
        return embeddings

    def _resolve_precomputed_phyla_embeddings_for_tree(
        self,
        newick_tree,
        mapping=None,
        num_leaf=None,
        device=None,
        dataset_id=None,
    ):
        names = self._ordered_leaf_names_from_mapping(mapping, num_leaf=num_leaf)
        if names is None and newick_tree is not None:
            names = self._ordered_leaf_names_from_newick(newick_tree)
        if not names:
            return None
        embeddings = self._lookup_precomputed_phyla_embeddings(
            names,
            device=device,
            dataset_id=dataset_id,
        )
        if embeddings is None:
            return None
        return embeddings.unsqueeze(0)

    def forward(
        self,
        batched_tokenized_trees,
        t,
        phyla_embeddings,
        autoregressive=False,
        autoregressive_component_groups=None,
        autoregressive_case_indices=None,
        autoregressive_start_topology_features=None,
        first_hit_case_indices=None,
        first_hit_start_topology_features=None,
        first_hit_start_topology_embeddings=None,
        first_hit_start_topology_pad_mask=None,
        first_hit_start_tree_graph_context=None,
    ):
        batched_tokenized_trees = _move_tokenized_batch_to_device(
            batched_tokenized_trees,
            self.device,
        )
        if torch.is_tensor(t):
            t = t.to(self.device)
        if not autoregressive:
            return_first_hit_logits = (
                self.velocity_first_hit_head_weight > 0.0
                or self.velocity_first_hit_head_use_at_sampling
            )
            return_boundary_vanish_logits = (
                self.velocity_boundary_vanish_head_weight > 0.0
                or self.velocity_boundary_vanish_head_use_at_sampling
            )
            return_edge_features = (
                self.velocity_first_hit_predictor_mode
                in {
                    "edge_length",
                    "edge_token_attention",
                    "edge_token_attention_replace",
                    "edge_token_attention_logitinput_replace",
                    "edge_token_attention_logitinput_replace_latelength",
                }
                or self.velocity_boundary_time_head_weight > 0.0
                or self.velocity_boundary_time_head_use_at_sampling
                or self.velocity_terminal_head_weight > 0.0
                or self.velocity_terminal_head_use_at_sampling
                or self.sampling_discrete_phase_rollout_use_at_sampling
                or self.velocity_refiner_mode == "edge_token_attention_delta"
                or self.branch_relax_head_weight > 0.0
                or self.branch_relax_head_use_at_sampling
            )
            edge_outputs = self.model(
                batched_tokenized_trees,
                t,
                phyla_embeddings=phyla_embeddings,
                return_leafs_only=False,
                return_edges_only=True,
                return_edge_features=return_edge_features,
                return_first_hit_logits=return_first_hit_logits,
                return_boundary_vanish_logits=return_boundary_vanish_logits,
                first_hit_case_indices=first_hit_case_indices,
                first_hit_start_topology_features=first_hit_start_topology_features,
                first_hit_start_topology_embeddings=first_hit_start_topology_embeddings,
                first_hit_start_topology_pad_mask=first_hit_start_topology_pad_mask,
                first_hit_start_tree_graph_context=first_hit_start_tree_graph_context,
            )
            edge_features = None
            first_hit_logits = None
            boundary_vanish_logits = None
            if return_first_hit_logits and return_boundary_vanish_logits:
                if return_edge_features:
                    (
                        velocity,
                        mask,
                        edge_features,
                        first_hit_logits,
                        boundary_vanish_logits,
                    ) = edge_outputs
                else:
                    velocity, mask, first_hit_logits, boundary_vanish_logits = edge_outputs
            elif return_first_hit_logits:
                if return_edge_features:
                    velocity, mask, edge_features, first_hit_logits = edge_outputs
                else:
                    velocity, mask, first_hit_logits = edge_outputs
            elif return_boundary_vanish_logits:
                if return_edge_features:
                    velocity, mask, edge_features, boundary_vanish_logits = edge_outputs
                else:
                    velocity, mask, boundary_vanish_logits = edge_outputs
            else:
                if return_edge_features:
                    velocity, mask, edge_features = edge_outputs
                else:
                    velocity, mask = edge_outputs
            edge_split_masks = batched_tokenized_trees[-1]
            edge_mask = batched_tokenized_trees[-2]
            return (
                velocity,
                edge_split_masks,
                edge_mask,
                first_hit_logits,
                boundary_vanish_logits,
                edge_features,
            )
        else:
            model_kwargs = dict(
                phyla_embeddings=phyla_embeddings,
                return_leafs_only=False,
                return_edges_only=True,
                autoregressive=True,
            )
            model_signature = inspect.signature(self.model.forward)
            supports_component_groups = (
                "autoregressive_component_groups" in model_signature.parameters
                or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in model_signature.parameters.values()
                )
            )
            if (
                autoregressive_component_groups is not None
                and supports_component_groups
            ):
                model_kwargs["autoregressive_component_groups"] = (
                    autoregressive_component_groups
                )
            supports_case_indices = (
                "autoregressive_case_indices" in model_signature.parameters
                or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in model_signature.parameters.values()
                )
            )
            if autoregressive_case_indices is not None and supports_case_indices:
                model_kwargs["autoregressive_case_indices"] = autoregressive_case_indices
            supports_start_topology_features = (
                "autoregressive_start_topology_features" in model_signature.parameters
                or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in model_signature.parameters.values()
                )
            )
            if (
                autoregressive_start_topology_features is not None
                and supports_start_topology_features
            ):
                model_kwargs["autoregressive_start_topology_features"] = (
                    autoregressive_start_topology_features
                )

            all_group_logits = self.model(
                batched_tokenized_trees,
                t,
                **model_kwargs,
            )
            return all_group_logits

    def _velocity_decode_flags(self):
        return_first_hit_logits = (
            self.velocity_first_hit_head_weight > 0.0
            or self.velocity_first_hit_head_use_at_sampling
        )
        return_boundary_vanish_logits = (
            self.velocity_boundary_vanish_head_weight > 0.0
            or self.velocity_boundary_vanish_head_use_at_sampling
        )
        return_edge_features = (
            self.velocity_first_hit_predictor_mode
            in {
                "edge_length",
                "edge_token_attention",
                "edge_token_attention_replace",
                "edge_token_attention_logitinput_replace",
                "edge_token_attention_logitinput_replace_latelength",
            }
            or self.velocity_boundary_time_head_weight > 0.0
            or self.velocity_boundary_time_head_use_at_sampling
            or self.velocity_terminal_head_weight > 0.0
            or self.velocity_terminal_head_use_at_sampling
            or self.sampling_discrete_phase_rollout_use_at_sampling
            or self.velocity_refiner_mode == "edge_token_attention_delta"
            or self.branch_relax_head_weight > 0.0
            or self.branch_relax_head_use_at_sampling
        )
        return (
            return_edge_features,
            return_first_hit_logits,
            return_boundary_vanish_logits,
        )

    def _unpack_velocity_edge_outputs(
        self,
        edge_outputs,
        tokenized_trees,
        *,
        return_edge_features,
        return_first_hit_logits,
        return_boundary_vanish_logits,
    ):
        edge_features = None
        first_hit_logits = None
        boundary_vanish_logits = None
        if return_first_hit_logits and return_boundary_vanish_logits:
            if return_edge_features:
                (
                    velocity,
                    _mask,
                    edge_features,
                    first_hit_logits,
                    boundary_vanish_logits,
                ) = edge_outputs
            else:
                velocity, _mask, first_hit_logits, boundary_vanish_logits = (
                    edge_outputs
                )
        elif return_first_hit_logits:
            if return_edge_features:
                velocity, _mask, edge_features, first_hit_logits = edge_outputs
            else:
                velocity, _mask, first_hit_logits = edge_outputs
        elif return_boundary_vanish_logits:
            if return_edge_features:
                velocity, _mask, edge_features, boundary_vanish_logits = edge_outputs
            else:
                velocity, _mask, boundary_vanish_logits = edge_outputs
        else:
            if return_edge_features:
                velocity, _mask, edge_features = edge_outputs
            else:
                velocity, _mask = edge_outputs
        edge_split_masks = tokenized_trees[-1]
        edge_mask = tokenized_trees[-2]
        return (
            velocity,
            edge_split_masks,
            edge_mask,
            first_hit_logits,
            boundary_vanish_logits,
            edge_features,
        )

    def _tokenized_batch_size(self, tokenized_trees):
        if tokenized_trees is None or not tokenized_trees:
            return 0
        return int(tokenized_trees[0].shape[0])

    def _tokenized_list_field(self, value):
        if isinstance(value, list):
            return list(value)
        if isinstance(value, tuple):
            return list(value)
        if torch.is_tensor(value):
            if value.dim() <= 1:
                return [value]
            return [value[idx] for idx in range(value.shape[0])]
        return list(value)

    def _pad_tokenized_tensor_for_concat(self, value, max_tokens, pad_value):
        if not torch.is_tensor(value):
            return value
        current_tokens = int(value.shape[1])
        if current_tokens == int(max_tokens):
            return value
        pad_shape = list(value.shape)
        pad_shape[1] = int(max_tokens) - current_tokens
        padding = value.new_full(pad_shape, pad_value)
        return torch.cat([value, padding], dim=1)

    def _concat_tokenized_tree_batches(self, tokenized_batches):
        tokenized_batches = [
            _move_tokenized_batch_to_device(batch, self.device)
            for batch in tokenized_batches
            if batch is not None
        ]
        if not tokenized_batches:
            return None
        max_tokens = max(int(batch[0].shape[1]) for batch in tokenized_batches)
        padded_features = []
        padding_masks = []
        padded_indices = []
        leaf_masks = []
        leaf_indices = []
        edge_masks = []
        edge_split_masks = []
        for batch in tokenized_batches:
            padded_features.append(
                self._pad_tokenized_tensor_for_concat(batch[0], max_tokens, 0.0)
            )
            padding_masks.append(
                self._pad_tokenized_tensor_for_concat(batch[1], max_tokens, True)
            )
            padded_indices.append(
                self._pad_tokenized_tensor_for_concat(batch[2], max_tokens, 0)
            )
            leaf_masks.append(
                self._pad_tokenized_tensor_for_concat(batch[3], max_tokens, False)
            )
            leaf_indices.extend(self._tokenized_list_field(batch[4]))
            edge_masks.append(
                self._pad_tokenized_tensor_for_concat(batch[5], max_tokens, False)
            )
            edge_split_masks.extend(self._tokenized_list_field(batch[6]))
        return (
            torch.cat(padded_features, dim=0),
            torch.cat(padding_masks, dim=0),
            torch.cat(padded_indices, dim=0),
            torch.cat(leaf_masks, dim=0),
            leaf_indices,
            torch.cat(edge_masks, dim=0),
            edge_split_masks,
        )

    def _pad_phyla_embeddings_for_concat(self, embeddings, max_leaves):
        if embeddings.size(1) == int(max_leaves):
            return embeddings
        pad_shape = (
            embeddings.size(0),
            int(max_leaves) - embeddings.size(1),
            embeddings.size(2),
        )
        padding = embeddings.new_zeros(pad_shape)
        return torch.cat([embeddings, padding], dim=1)

    def _concat_phyla_embeddings_for_batches(self, batches, batch_sizes):
        phyla_values = [batch.get("phyla_embeddings") for batch in batches]
        if all(value is None for value in phyla_values):
            return None
        if any(value is None for value in phyla_values):
            return None
        normalized = []
        for phyla_embeddings, batch_size in zip(phyla_values, batch_sizes):
            normalized.append(
                self.model._normalize_phyla_embeddings(
                    phyla_embeddings,
                    int(batch_size),
                )
            )
        max_leaves = max(int(value.size(1)) for value in normalized)
        normalized = [
            self._pad_phyla_embeddings_for_concat(value, max_leaves)
            for value in normalized
        ]
        return torch.cat(normalized, dim=0)

    def _encode_prepared_tokenized_trees_once(
        self,
        tokenized_trees,
        times,
        phyla_embeddings,
    ):
        required = (
            "_prepare_encoder_inputs",
            "_encode_with_layers",
            "_decode_outputs",
        )
        if not all(hasattr(self.model, name) for name in required):
            return None
        (
            x,
            padding_mask,
            leaf_mask,
            leaf_idx,
            edge_mask,
            edge_split_masks,
            phyla_global_context,
            phyla_clade_context,
            phyla_embeddings,
        ) = self.model._prepare_encoder_inputs(
            tokenized_trees,
            t=times,
            phyla_embeddings=phyla_embeddings,
        )
        encoder_input_x = x
        x = self.model._encode_with_layers(
            x,
            padding_mask=padding_mask,
            layers=self.model.layers,
            final_layer_norm=self.model.final_layer_norm,
        )
        if hasattr(self.model, "block2_layers"):
            block2_x = self.model._build_block2_input(x, encoder_input_x)
            x = self.model._encode_with_layers(
                block2_x,
                padding_mask=padding_mask,
                layers=self.model.block2_layers,
                final_layer_norm=self.model.block2_final_layer_norm,
            )
        elif hasattr(self.model, "refine_blocks"):
            for block in self.model.refine_blocks:
                block_input_x = self.model._build_refine_block_input(
                    x,
                    encoder_input_x,
                    block["bridge"],
                )
                x = self.model._encode_with_layers(
                    block_input_x,
                    padding_mask=padding_mask,
                    layers=block["layers"],
                    final_layer_norm=block["final_norm"],
                )
        return {
            "encoded": x,
            "leaf_mask": leaf_mask,
            "leaf_idx": leaf_idx,
            "edge_mask": edge_mask,
            "edge_split_masks": edge_split_masks,
            "phyla_global_context": phyla_global_context,
            "phyla_clade_context": phyla_clade_context,
            "phyla_embeddings": phyla_embeddings,
        }

    def _autoregressive_component_groups_for_batch(self, batch):
        cached_component_groups = batch.get("_cached_autoregressive_component_groups")
        if cached_component_groups is not None:
            return cached_component_groups
        if "newick_autoregressive_trees" in batch:
            return [
                get_structural_polytomy_groups_from_newick(newick_tree)
                for newick_tree in batch["newick_autoregressive_trees"]
            ]
        autoregressive_component_groups = []
        for labeled_merge_cluster in batch["batched_autoregressive_labels"]:
            seen_groups = set()
            groups = []
            for label in labeled_merge_cluster:
                components = tuple(int(component) for component in label["components"])
                if components in seen_groups:
                    continue
                seen_groups.add(components)
                groups.append(list(components))
            autoregressive_component_groups.append(groups)
        return autoregressive_component_groups

    def _joint_velocity_autoregressive_forward(
        self,
        velocity_batch,
        autoregressive_batch,
    ):
        velocity_batch, velocity_perturb_stats = self._prepare_velocity_training_batch(
            velocity_batch
        )
        autoregressive_batch, ar_prep_stats = (
            self._prepare_autoregressive_training_batch(autoregressive_batch)
        )
        velocity_tokenized = _move_tokenized_batch_to_device(
            velocity_batch["tokenized_trees"],
            self.device,
        )
        autoregressive_tokenized = _move_tokenized_batch_to_device(
            autoregressive_batch["tokenized_autoregressive_trees"],
            self.device,
        )
        velocity_batch_size = self._tokenized_batch_size(velocity_tokenized)
        autoregressive_batch_size = self._tokenized_batch_size(
            autoregressive_tokenized
        )
        if velocity_batch_size <= 0 or autoregressive_batch_size <= 0:
            return None

        velocity_times = velocity_batch["batched_time"]
        if torch.is_tensor(velocity_times):
            velocity_times = velocity_times.to(self.device).reshape(-1)
        else:
            velocity_times = torch.tensor(
                list(velocity_times),
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1)
        autoregressive_times = self._effective_autoregressive_time_tensor(
            autoregressive_batch["batched_autoregressive_time"]
        ).to(self.device).reshape(-1)
        if int(velocity_times.numel()) != velocity_batch_size:
            velocity_times = velocity_times[:1].expand(velocity_batch_size)
        if int(autoregressive_times.numel()) != autoregressive_batch_size:
            autoregressive_times = autoregressive_times[:1].expand(
                autoregressive_batch_size
            )
        combined_times = torch.cat([velocity_times, autoregressive_times], dim=0)

        combined_phyla_embeddings = self._concat_phyla_embeddings_for_batches(
            [velocity_batch, autoregressive_batch],
            [velocity_batch_size, autoregressive_batch_size],
        )
        if (
            combined_phyla_embeddings is None
            and (
                velocity_batch.get("phyla_embeddings") is not None
                or autoregressive_batch.get("phyla_embeddings") is not None
            )
        ):
            return None

        combined_tokenized = velocity_batch.get("_joint_tokenized_trees")
        if combined_tokenized is not None:
            combined_tokenized = _move_tokenized_batch_to_device(
                combined_tokenized,
                self.device,
            )
            combined_batch_size = self._tokenized_batch_size(combined_tokenized)
            expected_combined_batch_size = (
                velocity_batch_size + autoregressive_batch_size
            )
            if (
                combined_batch_size != expected_combined_batch_size
                or int(
                    velocity_batch.get(
                        "_joint_velocity_batch_size",
                        velocity_batch_size,
                    )
                )
                != velocity_batch_size
                or int(
                    velocity_batch.get(
                        "_joint_autoregressive_batch_size",
                        autoregressive_batch_size,
                    )
                )
                != autoregressive_batch_size
            ):
                combined_tokenized = None
        if combined_tokenized is None:
            combined_tokenized = self._concat_tokenized_tree_batches(
                [velocity_tokenized, autoregressive_tokenized]
            )
        encoded_payload = self._encode_prepared_tokenized_trees_once(
            combined_tokenized,
            combined_times,
            combined_phyla_embeddings,
        )
        if encoded_payload is None:
            return None

        encoded = encoded_payload["encoded"]
        leaf_mask = encoded_payload["leaf_mask"]
        leaf_idx = encoded_payload["leaf_idx"]
        edge_mask = encoded_payload["edge_mask"]
        edge_split_masks = encoded_payload["edge_split_masks"]
        phyla_global_context = encoded_payload["phyla_global_context"]
        phyla_clade_context = encoded_payload["phyla_clade_context"]
        encoded_phyla_embeddings = encoded_payload.get("phyla_embeddings")
        velocity_slice = slice(0, velocity_batch_size)
        autoregressive_slice = slice(
            velocity_batch_size,
            velocity_batch_size + autoregressive_batch_size,
        )
        (
            return_edge_features,
            return_first_hit_logits,
            return_boundary_vanish_logits,
        ) = self._velocity_decode_flags()
        velocity_decoded = self.model._decode_outputs(
            encoded[velocity_slice],
            leaf_mask=leaf_mask[velocity_slice],
            leaf_idx=leaf_idx[velocity_slice],
            edge_mask=edge_mask[velocity_slice],
            edge_split_masks=edge_split_masks[velocity_slice],
            t=velocity_times,
            return_leafs_only=False,
            return_edges_only=True,
            return_edge_features=return_edge_features,
            return_first_hit_logits=return_first_hit_logits,
            return_boundary_vanish_logits=return_boundary_vanish_logits,
            first_hit_case_indices=velocity_batch.get("_first_hit_case_indices"),
            first_hit_start_topology_features=velocity_batch.get(
                "_first_hit_start_topology_features"
            ),
            first_hit_start_topology_embeddings=velocity_batch.get(
                "_first_hit_start_topology_embeddings"
            ),
            first_hit_start_topology_pad_mask=velocity_batch.get(
                "_first_hit_start_topology_pad_mask"
            ),
            first_hit_start_tree_graph_context=velocity_batch.get(
                "_first_hit_start_tree_graph_context"
            ),
            phyla_global_context=None
            if phyla_global_context is None
            else phyla_global_context[velocity_slice],
            phyla_clade_context=None
            if phyla_clade_context is None
            else phyla_clade_context[velocity_slice],
            phyla_embeddings=None
            if encoded_phyla_embeddings is None
            else encoded_phyla_embeddings[velocity_slice],
        )
        velocity_outputs = self._unpack_velocity_edge_outputs(
            velocity_decoded,
            (
                velocity_tokenized[0],
                velocity_tokenized[1],
                velocity_tokenized[2],
                velocity_tokenized[3],
                velocity_tokenized[4],
                edge_mask[velocity_slice],
                edge_split_masks[velocity_slice],
            ),
            return_edge_features=return_edge_features,
            return_first_hit_logits=return_first_hit_logits,
            return_boundary_vanish_logits=return_boundary_vanish_logits,
        )

        autoregressive_component_groups = self._autoregressive_component_groups_for_batch(
            autoregressive_batch
        )
        if autoregressive_batch.get("_cached_autoregressive_component_groups") is None:
            autoregressive_batch = dict(autoregressive_batch)
            autoregressive_batch["_cached_autoregressive_component_groups"] = (
                autoregressive_component_groups
            )
        autoregressive_outputs = self.model._decode_outputs(
            encoded[autoregressive_slice],
            leaf_mask=leaf_mask[autoregressive_slice],
            leaf_idx=leaf_idx[autoregressive_slice],
            edge_mask=edge_mask[autoregressive_slice],
            edge_split_masks=edge_split_masks[autoregressive_slice],
            t=autoregressive_times,
            return_leafs_only=False,
            return_edges_only=True,
            autoregressive=True,
            autoregressive_component_groups=autoregressive_component_groups,
            autoregressive_case_indices=autoregressive_batch.get(
                "_autoregressive_case_indices"
            ),
            autoregressive_start_topology_features=autoregressive_batch.get(
                "_autoregressive_start_topology_features"
            ),
            phyla_global_context=None
            if phyla_global_context is None
            else phyla_global_context[autoregressive_slice],
            phyla_clade_context=None
            if phyla_clade_context is None
            else phyla_clade_context[autoregressive_slice],
            phyla_embeddings=None
            if encoded_phyla_embeddings is None
            else encoded_phyla_embeddings[autoregressive_slice],
        )
        return {
            "velocity_batch": velocity_batch,
            "autoregressive_batch": autoregressive_batch,
            "velocity_outputs": velocity_outputs,
            "autoregressive_outputs": autoregressive_outputs,
            "velocity_perturb_stats": velocity_perturb_stats,
            "ar_prep_stats": ar_prep_stats,
        }

    def _birthset_num_required_splits(
        self,
        num_components,
        component_masks=None,
        full_mask=None,
    ):
        parent_mask = 0
        for component in component_masks or []:
            parent_mask |= int(component)
        is_root_polytomy = bool(
            full_mask is not None
            and int(full_mask) != 0
            and (int(parent_mask) & int(full_mask)) == int(full_mask)
        )
        offset = 3 if is_root_polytomy else 2
        return max(int(num_components) - offset, 0)

    def _birthset_num_leaves_for_group(self, batch, batch_index, component_masks):
        num_leaves = batch.get("num_leaves")
        if isinstance(num_leaves, (list, tuple)) and len(num_leaves) > batch_index:
            try:
                return int(num_leaves[batch_index])
            except Exception:
                pass
        if torch.is_tensor(num_leaves):
            if num_leaves.numel() == 1:
                return int(num_leaves.item())
            if num_leaves.numel() > batch_index:
                return int(num_leaves[batch_index].item())
        newicks = batch.get("newick_autoregressive_trees")
        if newicks is not None and len(newicks) > batch_index:
            try:
                return int(Tree(newicks[batch_index]).n_leaves)
            except Exception:
                pass
        max_bit = 0
        for mask in component_masks:
            max_bit = max(max_bit, int(mask).bit_length())
        return max_bit + 1 if max_bit > 0 else 0

    def _birthset_update_split_bank_from_labels(self, labels_by_batch):
        if not self.birthset_use_train_birth_split_bank:
            return
        for labeled_merge_cluster in labels_by_batch or []:
            for label in labeled_merge_cluster or []:
                result_split = label.get("result_split")
                if result_split is not None:
                    self.birthset_split_bank.add(int(result_split))

    def _birthset_candidate_cap_reached(self, candidates_by_subset, force):
        return (
            not force
            and len(candidates_by_subset) >= self.birthset_max_candidates_per_polytomy
        )

    def _birthset_add_candidate(
        self,
        candidates_by_subset,
        local_subset,
        component_masks,
        full_mask,
        source,
        *,
        force=False,
    ):
        local_subset = int(local_subset)
        if not _birthset_valid_local_subset(local_subset, len(component_masks)):
            return False
        split_mask = _birthset_local_subset_to_split(local_subset, component_masks)
        if not _birthset_valid_rooted_split(split_mask, full_mask):
            return False
        if self._birthset_candidate_cap_reached(candidates_by_subset, force):
            return False
        existing = candidates_by_subset.get(local_subset)
        if existing is not None and existing.get("source") == "gold":
            return True
        candidates_by_subset[local_subset] = _birthset_candidate_record(
            local_subset,
            component_masks,
            source,
        )
        return True

    def _birthset_add_pair_prefix_candidates(
        self,
        candidates_by_subset,
        component_masks,
        full_mask,
        component_embeddings,
        component_phyla_embeddings=None,
        context=None,
    ):
        proposal_head = self.birthset_proposal_head or self.birthset_topology_head
        if proposal_head is None or component_embeddings is None:
            return 0
        G = len(component_masks)
        if G < 3:
            return 0

        pair_subsets = []
        for left_idx in range(G):
            for right_idx in range(left_idx + 1, G):
                subset = (1 << int(left_idx)) | (1 << int(right_idx))
                if _birthset_valid_local_subset(subset, G):
                    pair_subsets.append(int(subset))
        if not pair_subsets:
            return 0

        H = component_embeddings.detach()
        ctx = context.detach() if torch.is_tensor(context) else context
        phyla_ctx = (
            component_phyla_embeddings.detach()
            if torch.is_tensor(component_phyla_embeddings)
            else component_phyla_embeddings
        )
        with torch.inference_mode():
            pair_logits = proposal_head(
                H,
                pair_subsets,
                context=ctx,
                component_phyla_embeddings=phyla_ctx,
            )
        if pair_logits.numel() == 0:
            return 0

        top_k = min(
            int(self.birthset_pair_prefix_top_pairs),
            int(pair_logits.numel()),
        )
        top_pair_ids = torch.topk(pair_logits, k=top_k, largest=True).indices.tolist()
        pair_expansion_items = {}
        expansion_requests = []
        for pair_id in top_pair_ids:
            base_subset = int(pair_subsets[int(pair_id)])
            remaining = [
                idx for idx in range(G) if ((base_subset >> int(idx)) & 1) == 0
            ]
            items = []
            for idx in remaining:
                subset = int(base_subset | (1 << int(idx)))
                if not _birthset_valid_local_subset(subset, G):
                    continue
                items.append((int(idx), subset))
                expansion_requests.append((int(pair_id), subset))
            pair_expansion_items[int(pair_id)] = items

        expansion_scores_by_pair = {}
        if expansion_requests:
            expansion_subsets_all = [subset for _pair_id, subset in expansion_requests]
            with torch.inference_mode():
                expansion_logits_all = proposal_head(
                    H,
                    expansion_subsets_all,
                    context=ctx,
                    component_phyla_embeddings=phyla_ctx,
                )
            for (pair_id, subset), score in zip(
                expansion_requests,
                expansion_logits_all.detach().cpu().tolist(),
            ):
                expansion_scores_by_pair.setdefault(int(pair_id), {})[
                    int(subset)
                ] = float(score)

        added_count = 0
        for pair_id in top_pair_ids:
            if len(candidates_by_subset) >= self.birthset_max_candidates_per_polytomy:
                break
            base_subset = int(pair_subsets[int(pair_id)])
            if self._birthset_add_candidate(
                candidates_by_subset,
                base_subset,
                component_masks,
                full_mask,
                "pair_prefix",
            ):
                added_count += 1

            expansion_items = pair_expansion_items.get(int(pair_id), [])
            if not expansion_items:
                continue
            ordered_components = sorted(
                [idx for idx, _subset in expansion_items],
                key=lambda idx: expansion_scores_by_pair.get(int(pair_id), {}).get(
                    int(base_subset | (1 << int(idx))),
                    float("-inf"),
                ),
                reverse=True,
            )
            prefix_subset = int(base_subset)
            for idx in ordered_components:
                next_subset = int(prefix_subset | (1 << int(idx)))
                if not _birthset_valid_local_subset(next_subset, G):
                    break
                if self._birthset_add_candidate(
                    candidates_by_subset,
                    next_subset,
                    component_masks,
                    full_mask,
                    "pair_prefix",
                ):
                    added_count += 1
                prefix_subset = next_subset
                if len(candidates_by_subset) >= self.birthset_max_candidates_per_polytomy:
                    break
        return int(added_count)

    def _birthset_build_candidates(
        self,
        component_masks,
        num_leaves,
        *,
        gold_splits=None,
        gold_local_subsets=None,
        component_embeddings=None,
        component_phyla_embeddings=None,
        context=None,
        train=False,
    ):
        component_masks = [int(mask) for mask in component_masks]
        full_mask = _birthset_full_mask(num_leaves)
        candidates_by_subset = {}
        gold_splits = [int(split) for split in (gold_splits or [])]
        gold_local_subsets = [int(mask) for mask in (gold_local_subsets or [])]
        max_bit_length = max(
            [int(full_mask).bit_length()]
            + [int(mask).bit_length() for mask in component_masks]
            + [int(mask).bit_length() for mask in gold_splits]
        )
        if max_bit_length > int(full_mask).bit_length():
            full_mask = (1 << int(max_bit_length)) - 1
        gold_mismatches = 0

        if (
            self.birthset_use_train_birth_split_bank
            and self.birthset_split_bank
        ):
            for split in sorted(self.birthset_split_bank):
                local_subset = _birthset_map_split_to_local_subset(
                    split,
                    component_masks,
                )
                if local_subset is None:
                    continue
                self._birthset_add_candidate(
                    candidates_by_subset,
                    local_subset,
                    component_masks,
                    full_mask,
                    "bank",
                )

        if (
            self.birthset_use_small_polytomy_enumeration
            and len(component_masks) <= self.birthset_max_enum_components
        ):
            G = len(component_masks)
            for size in range(2, G):
                for combo in itertools.combinations(range(G), size):
                    local_subset = 0
                    for idx in combo:
                        local_subset |= 1 << int(idx)
                    self._birthset_add_candidate(
                        candidates_by_subset,
                        local_subset,
                        component_masks,
                        full_mask,
                        "enum",
                    )
                    if len(candidates_by_subset) >= self.birthset_max_candidates_per_polytomy:
                        break
                if len(candidates_by_subset) >= self.birthset_max_candidates_per_polytomy:
                    break

        if self.birthset_use_pair_prefix_candidates:
            self._birthset_add_pair_prefix_candidates(
                candidates_by_subset,
                component_masks,
                full_mask,
                component_embeddings,
                component_phyla_embeddings=component_phyla_embeddings,
                context=context,
            )

        pre_gold_candidate_splits = {
            _birthset_canonical_unrooted_split(item["split_mask"], full_mask)
            for item in candidates_by_subset.values()
        }
        pre_gold_candidate_local_subsets = {
            int(item["local_subset"]) for item in candidates_by_subset.values()
        }
        pre_gold_target_count = len(
            {
                ("split", int(split))
                for split in gold_splits
            }
            | {
                ("local", int(local_subset))
                for local_subset in gold_local_subsets
            }
        )
        pre_gold_target_hits = len(
            {
                ("split", int(split))
                for split in gold_splits
                if _birthset_canonical_unrooted_split(split, full_mask)
                in pre_gold_candidate_splits
            }
            | {
                ("local", int(local_subset))
                for local_subset in gold_local_subsets
                if int(local_subset) in pre_gold_candidate_local_subsets
            }
        )

        if train:
            for split in gold_splits:
                local_subset = _birthset_map_split_to_local_subset(
                    split,
                    component_masks,
                )
                if local_subset is None:
                    gold_mismatches += 1
                    continue
                self._birthset_add_candidate(
                    candidates_by_subset,
                    local_subset,
                    component_masks,
                    full_mask,
                    "gold",
                    force=True,
                )
            for local_subset in gold_local_subsets:
                added = self._birthset_add_candidate(
                    candidates_by_subset,
                    local_subset,
                    component_masks,
                    full_mask,
                    "gold",
                    force=True,
                )
                if not added:
                    gold_mismatches += 1

        candidates = list(candidates_by_subset.values())
        source_rank = {"gold": 0, "pair_prefix": 1, "bank": 2, "enum": 3}
        candidates.sort(
            key=lambda item: (
                source_rank.get(item["source"], 9),
                int(item["size"]),
                int(item["split_mask"]),
            )
        )
        return {
            "candidates": candidates,
            "full_mask": int(full_mask),
            "gold_mismatches": int(gold_mismatches),
            "pre_gold_target_count": int(pre_gold_target_count),
            "pre_gold_target_hits": int(pre_gold_target_hits),
        }

    def _birthset_rank_loss(self, logits, labels):
        if self.birthset_lambda_rank <= 0.0:
            return logits.new_tensor(0.0)
        pos_logits = logits[labels > 0.5]
        neg_logits = logits[labels <= 0.5]
        if pos_logits.numel() == 0 or neg_logits.numel() == 0:
            return logits.new_tensor(0.0)
        max_neg = min(
            int(neg_logits.numel()),
            max(1, int(pos_logits.numel()) * self.birthset_negatives_per_positive),
        )
        if int(neg_logits.numel()) > max_neg:
            neg_logits = torch.topk(neg_logits, k=max_neg, largest=True).values
        pairwise = (
            float(self.birthset_rank_margin)
            - pos_logits.unsqueeze(1)
            + neg_logits.unsqueeze(0)
        )
        return F.relu(pairwise).mean()

    def _birthset_positive_weight(self, labels):
        value = self.birthset_pos_weight
        if isinstance(value, str) and value.lower() == "auto":
            positives = labels.sum().clamp(min=1.0)
            negatives = (labels.numel() - labels.sum()).clamp(min=1.0)
            return (negatives / positives).detach()
        try:
            return labels.new_tensor(float(value))
        except Exception:
            return labels.new_tensor(1.0)

    def _birthset_proposal_positive_weight(self, labels):
        positives = labels.sum().clamp(min=1.0)
        negatives = (labels.numel() - labels.sum()).clamp(min=1.0)
        return (negatives / positives).detach()

    def _birthset_subset_inside_any_gold(self, local_subset, gold_local_subsets):
        local_subset = int(local_subset)
        for gold_subset in gold_local_subsets or []:
            gold_subset = int(gold_subset)
            if gold_subset and (local_subset & ~gold_subset) == 0:
                return True
        return False

    def _birthset_constructive_pair_targets(self, gold_local_subsets, num_components):
        gold = [
            int(mask)
            for mask in sorted({int(mask) for mask in gold_local_subsets or []})
            if _birthset_valid_local_subset(int(mask), int(num_components))
        ]
        if not gold:
            return set()

        direct_pair_targets = {
            int(mask)
            for mask in gold
            if _birthset_local_subset_size(int(mask)) == 2
        }
        if direct_pair_targets:
            return direct_pair_targets

        # Degenerate/rooted edge case: if the local split labels contain no explicit
        # two-component clade, use the smallest gold sides as the constructive seeds.
        min_size = min(_birthset_local_subset_size(mask) for mask in gold)
        targets = set()
        for gold_subset in gold:
            if _birthset_local_subset_size(gold_subset) != int(min_size):
                continue
            members = [
                idx
                for idx in range(int(num_components))
                if (int(gold_subset) >> int(idx)) & 1
            ]
            for left_pos, left_idx in enumerate(members):
                for right_idx in members[left_pos + 1 :]:
                    pair = (1 << int(left_idx)) | (1 << int(right_idx))
                    if _birthset_valid_local_subset(pair, int(num_components)):
                        targets.add(int(pair))
        return targets

    def _birthset_proposal_loss_from_precomputed(
        self,
        component_embeddings,
        *,
        component_phyla_embeddings=None,
        context=None,
        precomputed=None,
    ):
        proposal_head = self.birthset_proposal_head
        if proposal_head is None or component_embeddings is None or not precomputed:
            return None
        if precomputed.get("train_topk_dynamic"):
            return None

        device = component_embeddings.device
        proposal_profile = None
        if self._training_step_profile_enabled():
            self._training_step_profile_sync()
            proposal_profile = {"_last": time.perf_counter()}

        def _proposal_profile_mark(label):
            if proposal_profile is None:
                return
            self._training_step_profile_sync()
            now = time.perf_counter()
            last = proposal_profile.get("_last", now)
            proposal_profile[label] = (
                proposal_profile.get(label, 0.0) + (now - last)
            )
            proposal_profile["_last"] = now

        losses = []
        pair_loss = None
        expansion_loss = None
        order_loss = None
        pair_recall_at_topk = None

        pair_subsets = [int(x) for x in precomputed.get("pair_subsets") or []]
        pair_labels_values = [
            float(x) for x in precomputed.get("pair_labels") or []
        ]
        if len(pair_labels_values) != len(pair_subsets):
            return None
        _proposal_profile_mark("pair_prep")

        pair_score_by_subset = {}
        top_pair_subsets = set()
        if pair_subsets:
            pair_logits = proposal_head(
                component_embeddings,
                pair_subsets,
                context=context,
                component_phyla_embeddings=component_phyla_embeddings,
            )
            _proposal_profile_mark("pair_forward")
            pair_labels = torch.tensor(
                pair_labels_values,
                dtype=torch.float32,
                device=device,
            )
            pair_weights = torch.where(
                pair_labels > 0.5,
                self._birthset_proposal_positive_weight(pair_labels),
                pair_labels.new_tensor(1.0),
            )
            pair_loss = (
                F.binary_cross_entropy_with_logits(
                    pair_logits,
                    pair_labels,
                    reduction="none",
                )
                * pair_weights
            ).mean()
            losses.append(pair_loss)
            top_k = min(int(self.birthset_pair_prefix_top_pairs), len(pair_subsets))
            top_ids = torch.topk(pair_logits.detach(), k=top_k, largest=True).indices
            top_pair_subsets = {int(pair_subsets[int(idx)]) for idx in top_ids.tolist()}
            pair_score_by_subset = {
                int(pair_subsets[int(idx)]): float(score)
                for idx, score in enumerate(pair_logits.detach().cpu().tolist())
            }
            strict_pair_targets = {
                int(x) for x in precomputed.get("strict_pair_targets") or []
            }
            if strict_pair_targets:
                pair_recall_at_topk = float(
                    len(top_pair_subsets & strict_pair_targets)
                ) / float(len(strict_pair_targets))
            else:
                gold_local_subsets = [
                    int(x) for x in precomputed.get("gold_local_subsets") or []
                ]
                if gold_local_subsets:
                    gold_hits = 0
                    for gold_subset in gold_local_subsets:
                        hit = any(
                            (pair_subset & ~int(gold_subset)) == 0
                            for pair_subset in top_pair_subsets
                        )
                        gold_hits += 1 if hit else 0
                    pair_recall_at_topk = float(gold_hits) / float(
                        len(gold_local_subsets)
                    )
            _proposal_profile_mark("pair_loss")
        else:
            _proposal_profile_mark("pair_forward")
            _proposal_profile_mark("pair_loss")

        expansion_subsets = precomputed.get("expansion_subsets")
        expansion_labels_values = precomputed.get("expansion_labels")
        if expansion_subsets is not None:
            expansion_subsets = [int(x) for x in expansion_subsets]
            expansion_labels_values = [
                float(x) for x in (expansion_labels_values or [])
            ]
            if len(expansion_labels_values) != len(expansion_subsets):
                return None
        _proposal_profile_mark("expansion_prep")

        if expansion_subsets:
            expansion_logits = proposal_head(
                component_embeddings,
                expansion_subsets,
                context=context,
                component_phyla_embeddings=component_phyla_embeddings,
            )
            _proposal_profile_mark("expansion_forward")
            expansion_labels = torch.tensor(
                expansion_labels_values,
                dtype=torch.float32,
                device=device,
            )
            expansion_weights = torch.where(
                expansion_labels > 0.5,
                self._birthset_proposal_positive_weight(expansion_labels),
                expansion_labels.new_tensor(1.0),
            )
            expansion_loss = (
                F.binary_cross_entropy_with_logits(
                    expansion_logits,
                    expansion_labels,
                    reduction="none",
                )
                * expansion_weights
            ).mean()
            losses.append(expansion_loss)
            _proposal_profile_mark("expansion_loss")
        else:
            _proposal_profile_mark("expansion_forward")
            _proposal_profile_mark("expansion_loss")

        _proposal_profile_mark("order_seed_prep")
        order_losses = []
        order_candidate_subsets = [
            int(x) for x in precomputed.get("order_candidate_subsets") or []
        ]
        order_slices = list(precomputed.get("order_slices") or [])
        if order_candidate_subsets:
            order_logits_all = proposal_head(
                component_embeddings,
                order_candidate_subsets,
                context=context,
                component_phyla_embeddings=component_phyla_embeddings,
            )
            _proposal_profile_mark("order_forward")
            for start, end, target_ranks in order_slices:
                seed_logits = order_logits_all[int(start) : int(end)]
                if int(seed_logits.numel()) < 2:
                    continue
                rank_tensor = torch.tensor(
                    [float(x) for x in target_ranks],
                    dtype=torch.float32,
                    device=device,
                )
                better = rank_tensor.unsqueeze(1) < rank_tensor.unsqueeze(0)
                if not bool(better.any().item()):
                    continue
                pairwise_margin = (
                    float(self.birthset_rank_margin)
                    - seed_logits.unsqueeze(1)
                    + seed_logits.unsqueeze(0)
                )
                order_losses.append(F.relu(pairwise_margin[better]).mean())
            _proposal_profile_mark("order_loss")
        else:
            _proposal_profile_mark("order_forward")
            _proposal_profile_mark("order_loss")
        if order_losses:
            order_loss = torch.stack(order_losses).mean()
            losses.append(order_loss)
        _proposal_profile_mark("finalize")

        if not losses:
            return None
        result = {
            "loss": torch.stack(losses).mean(),
            "pair_loss": None if pair_loss is None else pair_loss.detach(),
            "expansion_loss": None
            if expansion_loss is None
            else expansion_loss.detach(),
            "order_loss": None if order_loss is None else order_loss.detach(),
            "pair_recall_at_topk": pair_recall_at_topk,
            "num_pair_examples": float(len(pair_subsets)),
            "num_expansion_examples": float(len(expansion_subsets or [])),
            "num_order_seed_pairs": float(
                precomputed.get("positive_pair_count", 0.0)
            ),
        }
        if proposal_profile is not None:
            result["profile"] = {
                key: float(value)
                for key, value in proposal_profile.items()
                if not key.startswith("_")
            }
        return result

    def _birthset_proposal_loss(
        self,
        component_embeddings,
        gold_local_subsets,
        *,
        component_phyla_embeddings=None,
        context=None,
        precomputed=None,
    ):
        proposal_head = self.birthset_proposal_head
        if (
            proposal_head is None
            or component_embeddings is None
            or float(self.birthset_lambda_proposal) <= 0.0
        ):
            return None

        G = int(component_embeddings.shape[0])
        gold_local_subsets = [
            int(mask)
            for mask in sorted({int(mask) for mask in gold_local_subsets or []})
            if _birthset_valid_local_subset(int(mask), G)
        ]
        if G < 3 or not gold_local_subsets:
            return None

        precomputed_result = self._birthset_proposal_loss_from_precomputed(
            component_embeddings,
            component_phyla_embeddings=component_phyla_embeddings,
            context=context,
            precomputed=precomputed,
        )
        if precomputed_result is not None:
            return precomputed_result

        device = component_embeddings.device
        proposal_profile = None
        if self._training_step_profile_enabled():
            self._training_step_profile_sync()
            proposal_profile = {"_last": time.perf_counter()}

        def _proposal_profile_mark(label):
            if proposal_profile is None:
                return
            self._training_step_profile_sync()
            now = time.perf_counter()
            last = proposal_profile.get("_last", now)
            proposal_profile[label] = (
                proposal_profile.get(label, 0.0) + (now - last)
            )
            proposal_profile["_last"] = now

        losses = []
        pair_loss = None
        expansion_loss = None
        order_loss = None
        pair_recall_at_topk = None
        strict_pair_targets = None
        constructive_pair_targets = self._birthset_constructive_pair_targets(
            gold_local_subsets,
            G,
        )
        if self.birthset_proposal_pair_target_mode == "strict_minimal":
            strict_pair_targets = constructive_pair_targets

        pair_subsets = []
        for left_idx in range(G):
            for right_idx in range(left_idx + 1, G):
                subset = (1 << int(left_idx)) | (1 << int(right_idx))
                if _birthset_valid_local_subset(subset, G):
                    pair_subsets.append(int(subset))
        _proposal_profile_mark("pair_prep")

        top_pair_subsets = set()
        pair_score_by_subset = {}
        if pair_subsets:
            pair_logits = proposal_head(
                component_embeddings,
                pair_subsets,
                context=context,
                component_phyla_embeddings=component_phyla_embeddings,
            )
            _proposal_profile_mark("pair_forward")
            pair_labels = torch.tensor(
                [
                    1.0
                    if (
                        int(subset) in strict_pair_targets
                        if strict_pair_targets is not None
                        else self._birthset_subset_inside_any_gold(
                            subset,
                            gold_local_subsets,
                        )
                    )
                    else 0.0
                    for subset in pair_subsets
                ],
                dtype=torch.float32,
                device=device,
            )
            pair_weights = torch.where(
                pair_labels > 0.5,
                self._birthset_proposal_positive_weight(pair_labels),
                pair_labels.new_tensor(1.0),
            )
            pair_loss = (
                F.binary_cross_entropy_with_logits(
                    pair_logits,
                    pair_labels,
                    reduction="none",
                )
                * pair_weights
            ).mean()
            losses.append(pair_loss)

            top_k = min(int(self.birthset_pair_prefix_top_pairs), len(pair_subsets))
            top_ids = torch.topk(pair_logits.detach(), k=top_k, largest=True).indices
            top_pair_subsets = {int(pair_subsets[int(idx)]) for idx in top_ids.tolist()}
            pair_score_by_subset = {
                int(pair_subsets[int(idx)]): float(score)
                for idx, score in enumerate(pair_logits.detach().cpu().tolist())
            }
            if strict_pair_targets is not None:
                if strict_pair_targets:
                    pair_recall_at_topk = float(
                        len(top_pair_subsets & strict_pair_targets)
                    ) / float(len(strict_pair_targets))
            else:
                gold_hits = 0
                for gold_subset in gold_local_subsets:
                    hit = any(
                        (pair_subset & ~int(gold_subset)) == 0
                        for pair_subset in top_pair_subsets
                    )
                    gold_hits += 1 if hit else 0
                pair_recall_at_topk = float(gold_hits) / float(len(gold_local_subsets))
            _proposal_profile_mark("pair_loss")

        expansion_subsets = []
        if self.birthset_proposal_train_topk and top_pair_subsets:
            seed_pair_subsets = set(top_pair_subsets) | set(constructive_pair_targets)
            seen_expansion_subsets = set()
            for pair_subset in sorted(seed_pair_subsets):
                pair_subset = int(pair_subset)
                for idx in range(G):
                    if (pair_subset >> int(idx)) & 1:
                        continue
                    subset = int(pair_subset | (1 << int(idx)))
                    if not _birthset_valid_local_subset(subset, G):
                        continue
                    if subset in seen_expansion_subsets:
                        continue
                    seen_expansion_subsets.add(subset)
                    expansion_subsets.append(subset)
        else:
            for combo in itertools.combinations(range(G), 3):
                subset = 0
                for idx in combo:
                    subset |= 1 << int(idx)
                expansion_subsets.append(int(subset))
        if expansion_subsets:
            positives = [
                subset
                for subset in expansion_subsets
                if self._birthset_subset_inside_any_gold(subset, gold_local_subsets)
            ]
            negatives = [
                subset
                for subset in expansion_subsets
                if not self._birthset_subset_inside_any_gold(subset, gold_local_subsets)
            ]
            cap = int(self.birthset_proposal_max_expansion_examples)
            if len(positives) + len(negatives) > cap:
                expansion_subsets = positives + negatives[: max(0, cap - len(positives))]
            else:
                expansion_subsets = positives + negatives
        _proposal_profile_mark("expansion_prep")

        if expansion_subsets:
            expansion_logits = proposal_head(
                component_embeddings,
                expansion_subsets,
                context=context,
                component_phyla_embeddings=component_phyla_embeddings,
            )
            _proposal_profile_mark("expansion_forward")
            expansion_labels = torch.tensor(
                [
                    1.0
                    if self._birthset_subset_inside_any_gold(
                        subset,
                        gold_local_subsets,
                    )
                    else 0.0
                    for subset in expansion_subsets
                ],
                dtype=torch.float32,
                device=device,
            )
            expansion_weights = torch.where(
                expansion_labels > 0.5,
                self._birthset_proposal_positive_weight(expansion_labels),
                expansion_labels.new_tensor(1.0),
            )
            expansion_loss = (
                F.binary_cross_entropy_with_logits(
                    expansion_logits,
                    expansion_labels,
                    reduction="none",
                )
                * expansion_weights
            ).mean()
            losses.append(expansion_loss)
            _proposal_profile_mark("expansion_loss")

        order_losses = []
        if self.birthset_proposal_train_topk and top_pair_subsets:
            positive_pair_subsets = sorted(
                {
                    int(subset)
                    for subset in set(constructive_pair_targets)
                    if _birthset_valid_local_subset(int(subset), G)
                },
                key=lambda subset: (
                    -float(pair_score_by_subset.get(int(subset), float("-inf"))),
                    int(subset),
                ),
            )
        else:
            if strict_pair_targets is not None:
                positive_pair_subsets = [
                    subset
                    for subset in pair_subsets
                    if int(subset) in strict_pair_targets
                ]
            else:
                positive_pair_subsets = [
                    subset
                    for subset in pair_subsets
                    if self._birthset_subset_inside_any_gold(
                        subset,
                        gold_local_subsets,
                    )
                ]
            positive_pair_subsets.sort(
                key=lambda subset: (
                    min(
                        (
                            _birthset_local_subset_size(gold_subset)
                            for gold_subset in gold_local_subsets
                            if (int(subset) & ~int(gold_subset)) == 0
                        ),
                        default=G + 1,
                    ),
                    int(subset),
                ),
            )
        positive_pair_subsets = positive_pair_subsets[
            : int(self.birthset_proposal_max_order_seed_pairs)
        ]
        _proposal_profile_mark("order_seed_prep")

        order_candidate_subsets = []
        order_slices = []
        for pair_subset in positive_pair_subsets:
            pair_subset = int(pair_subset)
            expansion_records = []
            target_ranks = []
            for idx in range(G):
                if (pair_subset >> int(idx)) & 1:
                    continue
                candidate_subset = int(pair_subset | (1 << int(idx)))
                if not _birthset_valid_local_subset(candidate_subset, G):
                    continue
                containing_sizes = [
                    _birthset_local_subset_size(gold_subset)
                    for gold_subset in gold_local_subsets
                    if (pair_subset & ~int(gold_subset)) == 0
                    and (candidate_subset & ~int(gold_subset)) == 0
                ]
                rank = min(containing_sizes) if containing_sizes else G + 1
                expansion_records.append(candidate_subset)
                target_ranks.append(float(rank))
            if len(expansion_records) < 2:
                continue
            start = len(order_candidate_subsets)
            order_candidate_subsets.extend(expansion_records)
            order_slices.append((start, len(order_candidate_subsets), target_ranks))

        if order_candidate_subsets:
            order_logits_all = proposal_head(
                component_embeddings,
                order_candidate_subsets,
                context=context,
                component_phyla_embeddings=component_phyla_embeddings,
            )
            _proposal_profile_mark("order_forward")
            for start, end, target_ranks in order_slices:
                seed_logits = order_logits_all[int(start) : int(end)]
                if int(seed_logits.numel()) < 2:
                    continue
                rank_tensor = torch.tensor(
                    target_ranks,
                    dtype=torch.float32,
                    device=device,
                )
                better = rank_tensor.unsqueeze(1) < rank_tensor.unsqueeze(0)
                if not bool(better.any().item()):
                    continue
                pairwise_margin = (
                    float(self.birthset_rank_margin)
                    - seed_logits.unsqueeze(1)
                    + seed_logits.unsqueeze(0)
                )
                order_losses.append(F.relu(pairwise_margin[better]).mean())
            _proposal_profile_mark("order_loss")
        if order_losses:
            order_loss = torch.stack(order_losses).mean()
            losses.append(order_loss)
        _proposal_profile_mark("finalize")

        if not losses:
            return None
        result = {
            "loss": torch.stack(losses).mean(),
            "pair_loss": None if pair_loss is None else pair_loss.detach(),
            "expansion_loss": None
            if expansion_loss is None
            else expansion_loss.detach(),
            "order_loss": None if order_loss is None else order_loss.detach(),
            "pair_recall_at_topk": pair_recall_at_topk,
            "num_pair_examples": float(len(pair_subsets)),
            "num_expansion_examples": float(len(expansion_subsets)),
            "num_order_seed_pairs": float(len(positive_pair_subsets)),
        }
        if proposal_profile is not None:
            result["profile"] = {
                key: float(value)
                for key, value in proposal_profile.items()
                if not key.startswith("_")
            }
        return result

    def _birthset_select_compatible_top_k(
        self,
        candidates,
        logits,
        k,
        existing_splits,
        full_mask,
    ):
        if int(k) <= 0 or not candidates:
            return []
        if self.birthset_decoder == "beam":
            return self._birthset_select_compatible_beam(
                candidates,
                logits,
                k,
                existing_splits,
                full_mask,
            )
        ordered_ids = sorted(
            range(len(candidates)),
            key=lambda idx: float(logits[idx].detach().cpu().item()),
            reverse=True,
        )
        selected = []
        selected_masks = set()
        selected_keys = set()
        existing_splits = {int(split) for split in existing_splits}
        existing_keys = {
            _birthset_canonical_unrooted_split(split, full_mask)
            for split in existing_splits
            if _birthset_valid_rooted_split(split, full_mask)
        }
        for idx in ordered_ids:
            if len(selected) >= int(k):
                break
            candidate = candidates[idx]
            split = int(candidate["split_mask"])
            if split in existing_splits or split in selected_masks:
                continue
            if not _birthset_valid_rooted_split(split, full_mask):
                continue
            split_key = _birthset_canonical_unrooted_split(split, full_mask)
            if split_key in existing_keys or split_key in selected_keys:
                continue
            if not all(
                _birthset_rooted_splits_compatible(split, other, full_mask)
                for other in existing_splits
            ):
                continue
            if not all(
                _birthset_rooted_splits_compatible(split, other, full_mask)
                for other in selected_masks
            ):
                continue
            item = dict(candidate)
            item["score"] = float(logits[idx].detach().cpu().item())
            selected.append(item)
            selected_masks.add(split)
            selected_keys.add(split_key)
        return selected

    def _birthset_select_compatible_beam(
        self,
        candidates,
        logits,
        k,
        existing_splits,
        full_mask,
    ):
        ordered_ids = sorted(
            range(len(candidates)),
            key=lambda idx: float(logits[idx].detach().cpu().item()),
            reverse=True,
        )
        existing_splits = {int(split) for split in existing_splits}
        existing_keys = {
            _birthset_canonical_unrooted_split(split, full_mask)
            for split in existing_splits
            if _birthset_valid_rooted_split(split, full_mask)
        }
        beam = [([], set(), set(), 0.0)]
        for idx in ordered_ids:
            candidate = candidates[idx]
            split = int(candidate["split_mask"])
            score = float(logits[idx].detach().cpu().item())
            next_beam = list(beam)
            for selected, selected_masks, selected_keys, total_score in beam:
                if len(selected) >= int(k):
                    continue
                if split in existing_splits or split in selected_masks:
                    continue
                if not _birthset_valid_rooted_split(split, full_mask):
                    continue
                split_key = _birthset_canonical_unrooted_split(split, full_mask)
                if split_key in existing_keys or split_key in selected_keys:
                    continue
                if not all(
                    _birthset_rooted_splits_compatible(split, other, full_mask)
                    for other in existing_splits
                ):
                    continue
                if not all(
                    _birthset_rooted_splits_compatible(split, other, full_mask)
                    for other in selected_masks
                ):
                    continue
                item = dict(candidate)
                item["score"] = score
                next_beam.append(
                    (
                        selected + [item],
                        set(selected_masks) | {split},
                        set(selected_keys) | {split_key},
                        total_score + score,
                    )
                )
            next_beam.sort(key=lambda state: (len(state[0]), state[3]), reverse=True)
            beam = next_beam[: self.birthset_beam_width]
        exact = [state for state in beam if len(state[0]) == int(k)]
        chosen = max(exact or beam, key=lambda state: (len(state[0]), state[3]))
        return chosen[0]

    def _birthset_balanced_completion(
        self,
        selected,
        component_masks,
        full_mask,
        k,
        existing_splits,
    ):
        if len(selected) >= int(k):
            return selected
        candidates_by_subset = {}
        G = len(component_masks)
        for size in range(2, G):
            for combo in itertools.combinations(range(G), size):
                local_subset = 0
                for idx in combo:
                    local_subset |= 1 << int(idx)
                self._birthset_add_candidate(
                    candidates_by_subset,
                    local_subset,
                    component_masks,
                    full_mask,
                    "balanced",
                    force=True,
                )
        selected_masks = {int(item["split_mask"]) for item in selected}
        existing_splits = {int(split) for split in existing_splits}
        ordered = sorted(
            candidates_by_subset.values(),
            key=lambda item: (
                abs(float(item["size"]) - (float(G) / 2.0)),
                int(item["size"]),
                int(item["split_mask"]),
            ),
        )
        completed = list(selected)
        selected_keys = {
            _birthset_canonical_unrooted_split(split, full_mask)
            for split in selected_masks
        }
        existing_keys = {
            _birthset_canonical_unrooted_split(split, full_mask)
            for split in existing_splits
            if _birthset_valid_rooted_split(split, full_mask)
        }
        for candidate in ordered:
            if len(completed) >= int(k):
                break
            split = int(candidate["split_mask"])
            if split in selected_masks or split in existing_splits:
                continue
            if not _birthset_valid_rooted_split(split, full_mask):
                continue
            split_key = _birthset_canonical_unrooted_split(split, full_mask)
            if split_key in selected_keys or split_key in existing_keys:
                continue
            if not all(
                _birthset_rooted_splits_compatible(split, other, full_mask)
                for other in existing_splits
            ):
                continue
            if not all(
                _birthset_rooted_splits_compatible(split, other, full_mask)
                for other in selected_masks
            ):
                continue
            item = dict(candidate)
            item["score"] = float("-inf")
            completed.append(item)
            selected_masks.add(split)
            selected_keys.add(split_key)
        return completed

    def _birthset_precomputed_group_for(
        self,
        batch,
        batch_index,
        component_masks,
    ):
        precomputed_by_batch = batch.get("batched_birthset_precomputed")
        if not isinstance(precomputed_by_batch, (list, tuple)):
            return None
        if int(batch_index) >= len(precomputed_by_batch):
            return None
        sample_precomputed = precomputed_by_batch[int(batch_index)]
        if not isinstance(sample_precomputed, dict):
            return None
        groups = sample_precomputed.get("groups_by_components")
        if not isinstance(groups, dict):
            return None
        key = tuple(int(mask) for mask in component_masks)
        group = groups.get(key)
        if group is None:
            # Be tolerant of future JSON-style materialization that may stringify keys.
            group = groups.get(str(key))
        if isinstance(group, dict):
            return group
        return None

    def _birthset_candidate_info_from_precomputed(
        self,
        batch,
        precomputed_group,
    ):
        if not bool(batch.get("_birthset_precomputed_candidate_info_enabled", False)):
            return None
        if self.birthset_use_train_birth_split_bank:
            return None
        if not isinstance(precomputed_group, dict):
            return None
        candidate_info = precomputed_group.get("candidate_info")
        if not isinstance(candidate_info, dict):
            return None
        candidates = candidate_info.get("candidates")
        if not isinstance(candidates, list):
            return None
        return candidate_info

    def _birthset_step_logs(
        self,
        batch,
        all_group_logits,
        ar_prep_stats,
        is_replay_batch,
        *,
        update_split_bank=True,
    ):
        logs = {}
        if self.birthset_topology_head is None:
            anchor_param = next(self.model.parameters())
            logs["loss"] = anchor_param.sum() * 0.0
            return logs

        should_update_split_bank = bool(update_split_bank) or bool(
            getattr(self, "training", False)
        )
        if should_update_split_bank:
            self._birthset_update_split_bank_from_labels(
                batch.get("batched_autoregressive_labels")
            )

        label_targets_by_batch = []
        found = {}
        for batch_index, labeled_merge_cluster in enumerate(
            batch["batched_autoregressive_labels"]
        ):
            group_targets = {}
            for label in labeled_merge_cluster:
                result_split = int(label["result_split"])
                components = tuple(int(component) for component in label["components"])
                merge_indices = [int(idx) for idx in label["merge_indices"]]
                local_subset = 0
                for idx in merge_indices:
                    local_subset |= 1 << int(idx)
                found[(batch_index, result_split)] = False
                group_targets.setdefault(components, []).append(
                    (result_split, local_subset)
                )
            label_targets_by_batch.append(group_targets)

        losses = []
        bce_losses = []
        rank_losses = []
        proposal_losses = []
        proposal_pair_losses = []
        proposal_expansion_losses = []
        proposal_order_losses = []
        proposal_profiles = []
        proposal_pair_recall_values = []
        proposal_pair_example_counts = []
        proposal_expansion_example_counts = []
        proposal_order_seed_pair_counts = []
        candidate_counts = []
        positive_counts = []
        required_counts = []
        observed_counts = []
        recall_values = []
        pre_gold_recall_values = []
        precision_values = []
        selected_recall_values = []
        f1_values = []
        fully_resolved_values = []
        mismatch_count = 0
        no_candidate_count = 0

        loss_device = self.device
        for group in all_group_logits:
            component_masks = [int(split) for split in group["splits_represented"]]
            batch_index = int(group["batch_index"])
            G = len(component_masks)
            if G <= 2:
                continue
            loss_device = group["group_embeddings"].device
            component_phyla_embeddings = (
                group.get("component_phyla_embeddings")
                if self.birthset_use_component_phyla_conditioning
                else None
            )
            group_targets = label_targets_by_batch[batch_index].get(
                tuple(component_masks),
                [],
            )
            gold_splits = sorted({int(split) for split, _ in group_targets})
            gold_local_subsets = sorted({int(mask) for _, mask in group_targets})
            for split in gold_splits:
                found[(batch_index, split)] = True

            precomputed_group = self._birthset_precomputed_group_for(
                batch,
                batch_index,
                component_masks,
            )
            num_leaves = self._birthset_num_leaves_for_group(
                batch,
                batch_index,
                component_masks,
            )
            candidate_info = self._birthset_candidate_info_from_precomputed(
                batch,
                precomputed_group,
            )
            if candidate_info is None:
                candidate_info = self._birthset_build_candidates(
                    component_masks,
                    num_leaves,
                    gold_splits=gold_splits,
                    gold_local_subsets=gold_local_subsets,
                    component_embeddings=group["group_embeddings"],
                    component_phyla_embeddings=component_phyla_embeddings,
                    context=group.get("graph_context"),
                    train=True,
                )
            candidates = candidate_info["candidates"]
            full_mask = candidate_info["full_mask"]
            mismatch_count += int(candidate_info["gold_mismatches"])
            candidate_counts.append(float(len(candidates)))
            positive_counts.append(float(len(gold_splits)))
            required_counts.append(
                float(
                    self._birthset_num_required_splits(
                        G,
                        component_masks=component_masks,
                        full_mask=full_mask,
                    )
                )
            )
            observed_counts.append(float(len(gold_splits)))
            if gold_splits:
                gold_split_keys = {
                    _birthset_canonical_unrooted_split(split, full_mask)
                    for split in gold_splits
                }
                pre_gold_target_count = int(
                    candidate_info.get("pre_gold_target_count", 0)
                )
                if pre_gold_target_count > 0:
                    pre_gold_recall_values.append(
                        float(candidate_info.get("pre_gold_target_hits", 0))
                        / float(pre_gold_target_count)
                    )
                candidate_split_set = {
                    _birthset_canonical_unrooted_split(item["split_mask"], full_mask)
                    for item in candidates
                }
                recall_values.append(
                    float(
                        sum(1 for split in gold_split_keys if split in candidate_split_set)
                    )
                    / float(len(gold_split_keys))
                )

            if not candidates:
                no_candidate_count += 1
                continue

            local_subsets = [int(item["local_subset"]) for item in candidates]
            logits = self.birthset_topology_head(
                group["group_embeddings"],
                local_subsets,
                context=group.get("graph_context"),
                component_phyla_embeddings=component_phyla_embeddings,
            )
            gold_split_keys = {
                _birthset_canonical_unrooted_split(split, full_mask)
                for split in gold_splits
            }
            precomputed_candidate_labels = candidate_info.get("candidate_labels")
            if (
                precomputed_candidate_labels is not None
                and len(precomputed_candidate_labels) == len(candidates)
            ):
                labels = torch.tensor(
                    [float(value) for value in precomputed_candidate_labels],
                    dtype=torch.float32,
                    device=logits.device,
                )
            else:
                labels = torch.tensor(
                    [
                        1.0
                        if (
                            _birthset_canonical_unrooted_split(
                                item["split_mask"],
                                full_mask,
                            )
                            in gold_split_keys
                            or int(item["local_subset"]) in gold_local_subsets
                        )
                        else 0.0
                        for item in candidates
                    ],
                    dtype=torch.float32,
                    device=logits.device,
                )
            pos_weight = self._birthset_positive_weight(labels)
            weights = torch.where(labels > 0.5, pos_weight, labels.new_tensor(1.0))
            bce = (
                F.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    reduction="none",
                )
                * weights
            ).mean()
            rank = self._birthset_rank_loss(logits, labels)
            group_loss = bce + (self.birthset_lambda_rank * rank)
            proposal = self._birthset_proposal_loss(
                group["group_embeddings"],
                gold_local_subsets,
                component_phyla_embeddings=component_phyla_embeddings,
                context=group.get("graph_context"),
                precomputed=None
                if precomputed_group is None
                else precomputed_group.get("proposal"),
            )
            if proposal is not None:
                proposal_loss = proposal["loss"]
                group_loss = group_loss + (
                    float(self.birthset_lambda_proposal) * proposal_loss
                )
                if proposal.get("profile"):
                    proposal_profiles.append(dict(proposal["profile"]))
                proposal_losses.append(proposal_loss.detach())
                if proposal.get("pair_loss") is not None:
                    proposal_pair_losses.append(proposal["pair_loss"])
                if proposal.get("expansion_loss") is not None:
                    proposal_expansion_losses.append(proposal["expansion_loss"])
                if proposal.get("order_loss") is not None:
                    proposal_order_losses.append(proposal["order_loss"])
                if proposal.get("pair_recall_at_topk") is not None:
                    proposal_pair_recall_values.append(
                        float(proposal["pair_recall_at_topk"])
                    )
                proposal_pair_example_counts.append(
                    float(proposal.get("num_pair_examples", 0.0))
                )
                proposal_expansion_example_counts.append(
                    float(proposal.get("num_expansion_examples", 0.0))
                )
                proposal_order_seed_pair_counts.append(
                    float(proposal.get("num_order_seed_pairs", 0.0))
                )
            losses.append(group_loss)
            bce_losses.append(bce.detach())
            rank_losses.append(rank.detach())

            required = self._birthset_num_required_splits(
                G,
                component_masks=component_masks,
                full_mask=full_mask,
            )
            selected = self._birthset_select_compatible_top_k(
                candidates,
                logits.detach(),
                required,
                existing_splits=set(),
                full_mask=full_mask,
            )
            selected_splits = {
                _birthset_canonical_unrooted_split(item["split_mask"], full_mask)
                for item in selected
            }
            gold_set = set(gold_split_keys)
            if selected:
                precision_values.append(
                    float(len(selected_splits & gold_set)) / float(len(selected_splits))
                )
            if gold_set:
                selected_recall = float(len(selected_splits & gold_set)) / float(
                    len(gold_set)
                )
                selected_recall_values.append(selected_recall)
                precision = precision_values[-1] if selected else 0.0
                if precision + selected_recall > 0.0:
                    f1_values.append(
                        2.0 * precision * selected_recall / (precision + selected_recall)
                    )
                else:
                    f1_values.append(0.0)
            fully_resolved_values.append(
                1.0 if len(selected) >= int(required) else 0.0
            )

        missing_explicit_targets = sum(
            1 for was_found in found.values() if not was_found
        )
        if missing_explicit_targets:
            logs["birthset_stats/missing_explicit_targets"] = torch.tensor(
                float(missing_explicit_targets),
                device=loss_device,
            )
            if self.verbose:
                logger.warning(
                    "Birthset target mapping missed %s explicit AR targets.",
                    missing_explicit_targets,
                )

        if losses:
            birth_loss = torch.stack(losses).mean()
        else:
            anchor_param = next(self.model.parameters())
            birth_loss = anchor_param.sum() * 0.0
            logs["birthset_stats/no_candidate_loss"] = torch.tensor(
                1.0,
                device=loss_device,
            )
        logs["loss"] = float(self.birthset_lambda_birth) * birth_loss
        logs["birthset_stats/loss_unscaled"] = birth_loss.detach()
        if bce_losses:
            logs["birthset_stats/bce_loss"] = torch.stack(bce_losses).mean().to(
                loss_device
            )
        if rank_losses:
            logs["birthset_stats/rank_loss"] = torch.stack(rank_losses).mean().to(
                loss_device
            )
        if proposal_losses:
            logs["birthset_stats/proposal_loss"] = torch.stack(
                proposal_losses
            ).mean().to(loss_device)
        if proposal_pair_losses:
            logs["birthset_stats/proposal_pair_loss"] = torch.stack(
                proposal_pair_losses
            ).mean().to(loss_device)
        if proposal_expansion_losses:
            logs["birthset_stats/proposal_expansion_loss"] = torch.stack(
                proposal_expansion_losses
            ).mean().to(loss_device)
        if proposal_order_losses:
            logs["birthset_stats/proposal_order_loss"] = torch.stack(
                proposal_order_losses
            ).mean().to(loss_device)
        if proposal_pair_recall_values:
            logs["birthset_stats/proposal_pair_recall_at_topk"] = torch.tensor(
                float(np.mean(proposal_pair_recall_values)),
                device=loss_device,
            )
        if proposal_pair_example_counts:
            logs["birthset_stats/proposal_pair_examples"] = torch.tensor(
                float(np.mean(proposal_pair_example_counts)),
                device=loss_device,
            )
        if proposal_expansion_example_counts:
            logs["birthset_stats/proposal_expansion_examples"] = torch.tensor(
                float(np.mean(proposal_expansion_example_counts)),
                device=loss_device,
            )
        if proposal_order_seed_pair_counts:
            logs["birthset_stats/proposal_order_seed_pairs"] = torch.tensor(
                float(np.mean(proposal_order_seed_pair_counts)),
                device=loss_device,
            )
        if proposal_profiles and self._training_step_profile_enabled():
            profile_keys = sorted(
                {
                    key
                    for profile in proposal_profiles
                    for key in profile.keys()
                    if key not in {"groups"}
                }
            )
            profile_summary = {
                "groups": float(len(proposal_profiles)),
            }
            profile_summary.update(
                {
                    key: float(
                        sum(float(profile.get(key, 0.0)) for profile in proposal_profiles)
                    )
                    for key in profile_keys
                }
            )
            for key, value in profile_summary.items():
                logs[f"birthset_stats/proposal_profile/{key}"] = torch.tensor(
                    float(value),
                    device=loss_device,
                )
            timing_text = " ".join(
                f"{key}={float(value):.4f}s"
                for key, value in profile_summary.items()
                if key != "groups"
            )
            logging.info(
                "BIRTHSET_PROPOSAL_PROFILE step=%s groups=%s %s",
                int(getattr(self, "stepper", 0) or 0),
                int(profile_summary["groups"]),
                timing_text,
            )
        logs["birthset_stats/lambda_birth"] = torch.tensor(
            float(self.birthset_lambda_birth),
            device=loss_device,
        )
        logs["birthset_stats/lambda_rank"] = torch.tensor(
            float(self.birthset_lambda_rank),
            device=loss_device,
        )
        logs["birthset_stats/lambda_proposal"] = torch.tensor(
            float(self.birthset_lambda_proposal),
            device=loss_device,
        )
        logs["birthset_stats/split_bank_size"] = torch.tensor(
            float(len(self.birthset_split_bank)),
            device=loss_device,
        )
        logs["birthset_stats/gold_mapping_mismatches"] = torch.tensor(
            float(mismatch_count),
            device=loss_device,
        )
        logs["birthset_stats/no_candidate_groups"] = torch.tensor(
            float(no_candidate_count),
            device=loss_device,
        )
        if candidate_counts:
            logs["birthset_stats/avg_candidates"] = torch.tensor(
                float(np.mean(candidate_counts)),
                device=loss_device,
            )
        if positive_counts:
            logs["birthset_stats/avg_positives"] = torch.tensor(
                float(np.mean(positive_counts)),
                device=loss_device,
            )
        if required_counts:
            logs["birthset_stats/avg_required_splits"] = torch.tensor(
                float(np.mean(required_counts)),
                device=loss_device,
            )
        if observed_counts:
            logs["birthset_stats/avg_observed_gold_splits"] = torch.tensor(
                float(np.mean(observed_counts)),
                device=loss_device,
            )
        if recall_values:
            logs["birthset_stats/candidate_recall"] = torch.tensor(
                float(np.mean(recall_values)),
                device=loss_device,
            )
        if pre_gold_recall_values:
            logs["birthset_stats/candidate_recall_pre_gold"] = torch.tensor(
                float(np.mean(pre_gold_recall_values)),
                device=loss_device,
            )
        if precision_values:
            logs["birthset_stats/selected_precision"] = torch.tensor(
                float(np.mean(precision_values)),
                device=loss_device,
            )
        if selected_recall_values:
            logs["birthset_stats/selected_recall"] = torch.tensor(
                float(np.mean(selected_recall_values)),
                device=loss_device,
            )
        if f1_values:
            logs["birthset_stats/selected_f1"] = torch.tensor(
                float(np.mean(f1_values)),
                device=loss_device,
            )
        if fully_resolved_values:
            logs["birthset_stats/fraction_fully_resolved"] = torch.tensor(
                float(np.mean(fully_resolved_values)),
                device=loss_device,
            )
        if ar_prep_stats is not None:
            logs["autoregressive_stats/rollin_attempted"] = torch.tensor(
                ar_prep_stats["rollin_attempted"],
                device=loss_device,
            )
            logs["autoregressive_stats/rollin_applied"] = torch.tensor(
                ar_prep_stats["rollin_applied"],
                device=loss_device,
            )
            logs["autoregressive_stats/dagger_attempted"] = torch.tensor(
                ar_prep_stats["dagger_attempted"],
                device=loss_device,
            )
            logs["autoregressive_stats/dagger_applied"] = torch.tensor(
                ar_prep_stats["dagger_applied"],
                device=loss_device,
            )

        if self.record and not is_replay_batch:
            wandb_metrics = {
                "train/birthset_loss": float(birth_loss.detach().item()),
                "train/birthset_loss_scaled": float(logs["loss"].detach().item()),
                "birthset_stats/split_bank_size": float(len(self.birthset_split_bank)),
                "birthset_stats/gold_mapping_mismatches": float(mismatch_count),
            }
            for key, value in logs.items():
                if key.startswith("birthset_stats/") and torch.is_tensor(value):
                    if value.numel() == 1:
                        wandb_metrics[key] = float(value.detach().cpu().item())
            self._wandb_log_filtered(wandb_metrics, step=self.stepper)
        return logs

    def _plan_birthset_boundary_splits(
        self,
        logit_outputs,
        existing_splits,
        num_leaves,
    ):
        if self.birthset_topology_head is None:
            return {
                "selected": [],
                "metrics": {"num_ar_fallback_calls": 1.0},
            }
        full_mask = _birthset_full_mask(num_leaves)
        max_bit_length = int(full_mask).bit_length()
        for output in logit_outputs or []:
            for split in output.get("splits_represented", []) or []:
                max_bit_length = max(max_bit_length, int(split).bit_length())
        for split in existing_splits or []:
            max_bit_length = max(max_bit_length, int(split).bit_length())
        if max_bit_length > int(full_mask).bit_length():
            full_mask = (1 << int(max_bit_length)) - 1
        existing = {int(split) for split in existing_splits}
        planned_existing = set(existing)
        selected_all = []
        metrics = {
            "num_polytomies": 0.0,
            "num_candidate_splits": 0.0,
            "num_selected_birth_splits": 0.0,
            "num_required_birth_splits": 0.0,
            "fraction_resolved_without_fallback": 0.0,
            "num_ar_fallback_calls": 0.0,
            "num_transformer_forwards": 1.0,
        }
        sorted_outputs = sorted(
            logit_outputs,
            key=lambda output: float(output["polytomy_pred"].detach().cpu().item()),
            reverse=True,
        )
        resolved_count = 0
        for group in sorted_outputs:
            component_masks = [int(split) for split in group["splits_represented"]]
            G = len(component_masks)
            component_phyla_embeddings = (
                group.get("component_phyla_embeddings")
                if self.birthset_use_component_phyla_conditioning
                else None
            )
            required = self._birthset_num_required_splits(
                G,
                component_masks=component_masks,
                full_mask=full_mask,
            )
            if required <= 0:
                continue
            metrics["num_polytomies"] += 1.0
            metrics["num_required_birth_splits"] += float(required)
            candidate_info = self._birthset_build_candidates(
                component_masks,
                num_leaves,
                component_embeddings=group["group_embeddings"],
                component_phyla_embeddings=component_phyla_embeddings,
                context=group.get("graph_context"),
                train=False,
            )
            candidates = candidate_info["candidates"]
            metrics["num_candidate_splits"] += float(len(candidates))
            if not candidates:
                metrics["num_ar_fallback_calls"] += 1.0
                continue
            local_subsets = [int(item["local_subset"]) for item in candidates]
            with torch.inference_mode():
                logits = self.birthset_topology_head(
                    group["group_embeddings"],
                    local_subsets,
                    context=group.get("graph_context"),
                    component_phyla_embeddings=component_phyla_embeddings,
                )
            selected = self._birthset_select_compatible_top_k(
                candidates,
                logits,
                required,
                planned_existing,
                full_mask,
            )
            if (
                len(selected) < int(required)
                and self.birthset_fallback == "balanced_completion"
            ):
                selected = self._birthset_balanced_completion(
                    selected,
                    component_masks,
                    full_mask,
                    required,
                    planned_existing,
                )
            if len(selected) < int(required):
                metrics["num_ar_fallback_calls"] += 1.0
            else:
                resolved_count += 1
            for item in selected:
                split = int(item["split_mask"])
                if split in planned_existing:
                    continue
                selected_all.append(item)
                planned_existing.add(split)
        metrics["num_selected_birth_splits"] = float(len(selected_all))
        if metrics["num_polytomies"] > 0.0:
            metrics["fraction_resolved_without_fallback"] = (
                float(resolved_count) / metrics["num_polytomies"]
            )
        return {"selected": selected_all, "metrics": metrics}

    def step_terminal(self, batch, eval=False):
        if batch is None:
            loss = torch.tensor(0.0, device=self.device, requires_grad=not eval)
            return {
                "loss": loss,
                "velocity/terminal_head_accuracy": torch.tensor(
                    0.0, dtype=torch.float32, device=self.device
                ),
                "velocity/terminal_head_target_rate": torch.tensor(
                    0.0, dtype=torch.float32, device=self.device
                ),
                "velocity/terminal_head_pred_rate": torch.tensor(
                    0.0, dtype=torch.float32, device=self.device
                ),
            }
        batch = self._attach_case_indices_to_batch(batch)
        batch = self._attach_start_topology_features_to_batch(batch)
        batch = self._attach_start_tree_graph_context_to_batch(batch)

        (
            v_pred,
            edge_split_masks,
            _edge_mask,
            first_hit_logits,
            boundary_vanish_logits,
            edge_features,
        ) = self.forward(
            batch["tokenized_trees"],
            batch["batched_time"],
            batch["phyla_embeddings"],
            first_hit_case_indices=batch.get("_first_hit_case_indices"),
            first_hit_start_topology_features=batch.get(
                "_first_hit_start_topology_features"
            ),
            first_hit_start_topology_embeddings=batch.get(
                "_first_hit_start_topology_embeddings"
            ),
            first_hit_start_topology_pad_mask=batch.get(
                "_first_hit_start_topology_pad_mask"
            ),
            first_hit_start_tree_graph_context=batch.get(
                "_first_hit_start_tree_graph_context"
            ),
        )

        logits = []
        targets = []
        original_trees = batch.get("original_trees", [])
        batched_time = batch.get("batched_time")
        batched_targets = batch.get("batched_terminal_stop")
        batched_case_indices = batch.get("_first_hit_case_indices")
        for num, current_newick in enumerate(original_trees):
            if current_newick is None:
                continue
            try:
                n_leaves = int(Tree(current_newick).n_leaves)
            except Exception:
                continue
            aligned = _align_model_outputs_to_tree_context(
                self,
                current_newick,
                n_leaves,
                edge_split_masks[num],
                v_pred[num, :, 0],
                first_hit_logits_tree=None
                if first_hit_logits is None
                else first_hit_logits[num, :, 0],
                boundary_vanish_logits_tree=None
                if boundary_vanish_logits is None
                else boundary_vanish_logits[num, :, 0],
                edge_features_tree=None if edge_features is None else edge_features[num],
                eps_len=1e-8,
            )
            aligned_first_hit_logits = self._compute_first_hit_logits(
                aligned["first_hit_logits"],
                lengths=aligned["lengths"],
                velocities=aligned["velocities"],
                edge_features=aligned["edge_features"],
                group_sizes=[int(aligned["lengths"].numel())],
            )
            term_logit = self._predict_terminal_stop_logit(
                aligned["lengths"],
                aligned["velocities"],
                time_value=float(batched_time[num].item()),
                first_hit_logits=aligned_first_hit_logits,
                boundary_vanish_logits=aligned["boundary_vanish_logits"],
                edge_features=aligned["edge_features"],
                aligned_model_masks=aligned["aligned_model_masks"],
                supervised_mask=aligned["supervised_mask"],
                case_index=None
                if batched_case_indices is None
                else int(batched_case_indices[num].item()),
            )
            if term_logit is None:
                continue
            logits.append(term_logit)
            targets.append(batched_targets[num])

        if not logits:
            loss = torch.tensor(0.0, device=self.device, requires_grad=not eval)
            return {
                "loss": loss,
                "velocity/terminal_head_accuracy": torch.tensor(
                    0.0, dtype=torch.float32, device=self.device
                ),
                "velocity/terminal_head_target_rate": torch.tensor(
                    0.0, dtype=torch.float32, device=self.device
                ),
                "velocity/terminal_head_pred_rate": torch.tensor(
                    0.0, dtype=torch.float32, device=self.device
                ),
            }

        logits_tensor = torch.stack(logits).reshape(-1)
        target_tensor = torch.stack(targets).to(
            logits_tensor.device, dtype=logits_tensor.dtype
        )
        pos_weight = None
        if bool(getattr(self, "velocity_terminal_head_balance_loss", False)):
            with torch.no_grad():
                pos = target_tensor.sum()
                neg = target_tensor.numel() - pos
                if float(pos.item()) > 0.0 and float(neg.item()) > 0.0:
                    pos_weight = torch.clamp(neg / pos, min=1.0).detach()
        loss = F.binary_cross_entropy_with_logits(
            logits_tensor,
            target_tensor,
            pos_weight=pos_weight,
        )
        with torch.no_grad():
            probs = torch.sigmoid(logits_tensor)
            preds = probs > 0.5
            target_bool = target_tensor > 0.5
            acc = (preds == target_bool).float().mean()
            target_rate = target_bool.float().mean()
            pred_rate = preds.float().mean()
        return {
            "loss": loss,
            "velocity/terminal_head_accuracy": acc.detach(),
            "velocity/terminal_head_target_rate": target_rate.detach(),
            "velocity/terminal_head_pred_rate": pred_rate.detach(),
        }

    def step(
        self,
        batch,
        eval=False,
        autoregressive=False,
        precomputed_outputs=None,
        prepared_batch=False,
        velocity_perturb_stats=None,
        ar_prep_stats=None,
    ):
        if self.use_historical_step_impl:
            if precomputed_outputs is not None or prepared_batch:
                raise ValueError(
                    "Historical step implementation does not support "
                    "precomputed prepared-batch outputs."
                )
            historical_module = _load_historical_training_module_for_step()
            return historical_module.TrainingModule.step(
                self, batch, eval=eval, autoregressive=autoregressive
            )
        logs = {}
        is_replay_batch = bool(batch.get("_is_replay_batch", False))
        if not eval and not autoregressive:
            self.current_step_value += 1
        if (
            batch["phyla_embeddings"] is None
            and self.live_phyla_model is not None
            and batch.get("selected_sequences") is not None
        ):
            live_embeddings = self._compute_live_phyla_embeddings_for_batch(
                batch,
                grad=not eval,
            )
            if live_embeddings is not None:
                batch["phyla_embeddings"] = live_embeddings
        if (
            batch["phyla_embeddings"] is None
            and "ids" in batch
        ):
            phyla_embeddings_list = []
            missing_precomputed = False
            for i in range(len(batch["ids"])):
                mapping = batch.get("mappings", [None] * len(batch["ids"]))[i]
                num_leaf = batch.get("num_leaves", [None] * len(batch["ids"]))[i]
                dataset_ids = batch.get("dataset_ids") or [None] * len(batch["ids"])
                dataset_id = dataset_ids[i]
                ordered_names = self._ordered_leaf_names_from_mapping(
                    mapping,
                    num_leaf=num_leaf,
                )
                embeddings = self._lookup_precomputed_phyla_embeddings(
                    ordered_names or [],
                    device=self.device,
                    dataset_id=dataset_id,
                )
                if embeddings is None:
                    missing_precomputed = True
                    phyla_embeddings_list = []
                    break
                phyla_embeddings_list.append(embeddings)

            if phyla_embeddings_list:
                batch["phyla_embeddings"] = phyla_embeddings_list
            elif self.phyla_model is not None and missing_precomputed:
                phyla_embeddings_list = []
                for i in range(len(batch["ids"])):
                    mapping = batch.get("mappings", [None] * len(batch["ids"]))[i]
                    num_leaf = batch.get("num_leaves", [0] * len(batch["ids"]))[i]
                    if mapping is None:
                        phyla_embeddings_list = []
                        break
                    seqs = []
                    names = []
                    for idx in range(num_leaf):
                        idx_str = str(idx)
                        taxon_name = mapping.get(idx_str)
                        if taxon_name:
                            seq = self.dataset.name_to_seq.get(taxon_name, "")
                            seqs.append(seq)
                            names.append(taxon_name)
                        else:
                            seqs.append("")
                            names.append("unknown")

                    embeddings = self.compute_phyla_embeddings(
                        seqs, names, device=str(self.device)
                    )
                    if embeddings.dim() == 3 and embeddings.size(0) == 1:
                        embeddings = embeddings.squeeze(0)
                    phyla_embeddings_list.append(embeddings)

                batch["phyla_embeddings"] = phyla_embeddings_list

        if not autoregressive:
            if prepared_batch:
                velocity_perturb_stats = velocity_perturb_stats or {
                    "attempted": 0.0,
                    "applied": 0.0,
                }
            else:
                batch, velocity_perturb_stats = self._prepare_velocity_training_batch(
                    batch
                )
            if precomputed_outputs is None:
                (
                    v_pred,
                    edge_split_masks,
                    edge_mask,
                    first_hit_logits,
                    boundary_vanish_logits,
                    edge_features,
                ) = self.forward(
                    batch["tokenized_trees"],
                    batch["batched_time"],
                    batch["phyla_embeddings"],
                    first_hit_case_indices=batch.get("_first_hit_case_indices"),
                    first_hit_start_topology_features=batch.get(
                        "_first_hit_start_topology_features"
                    ),
                    first_hit_start_topology_embeddings=batch.get(
                        "_first_hit_start_topology_embeddings"
                    ),
                    first_hit_start_topology_pad_mask=batch.get(
                        "_first_hit_start_topology_pad_mask"
                    ),
                    first_hit_start_tree_graph_context=batch.get(
                        "_first_hit_start_tree_graph_context"
                    ),
                )
            else:
                (
                    v_pred,
                    edge_split_masks,
                    edge_mask,
                    first_hit_logits,
                    boundary_vanish_logits,
                    edge_features,
                ) = precomputed_outputs

            if self.train_tokenized_trees is None:
                self.train_tokenized_trees = batch["tokenized_trees"]
                self.train_batched_time = batch["batched_time"]
                self.train_tree = batch["original_trees"]
            # else:
            #     if calculate_norm_rf(batch['original_trees'][0], self.train_tree[0]) != 0:
            #         raise Exception("Training tree topology changed during training!")
            #     elif not torch.equal(batch["tokenized_trees"][0], self.train_tokenized_trees[0]):
            #         import pdb; pdb.set_trace()
            #         raise Exception("Training tokenized trees changed during training!")

            direct_set_loss = None
            direct_set_mse_weight = float(
                getattr(self, "velocity_probe_direct_set_mse_weight", 0.0)
            )
            direct_set_loss_weight = float(
                getattr(self, "velocity_probe_direct_set_loss_weight", 1.0)
            )
            if bool(batch.get("_use_probe_parity_direct_set_loss", False)):
                direct_losses = []
                exact_flags = []
                jaccards = []
                direct_set_pos_weights = []
                target_negative_rates = []
                target_negative_losses = []
                nontarget_nonnegative_losses = []
                sample_mask = list(
                    batch.get(
                        "_probe_direct_set_sample_mask",
                        [True for _ in batch.get("original_trees", [])],
                    )
                )
                target_sets = list(
                    batch.get(
                        "_probe_direct_set_targets",
                        [[] for _ in batch.get("original_trees", [])],
                    )
                )
                direct_set_debug_enabled = (
                    os.environ.get("PHYLAFLOW_DEBUG_DIRECT_SET", "0") == "1"
                )
                autoregressive_first_hit_mode = (
                    getattr(self.model, "first_hit_head_mode", "base")
                    == "autoregressive_set"
                )
                direct_set_bce_weight = float(
                    getattr(self, "velocity_probe_direct_set_bce_weight", 1.0)
                )
                for num, current_newick in enumerate(batch.get("original_trees", [])):
                    if num >= len(sample_mask) or not bool(sample_mask[num]):
                        continue
                    if current_newick is None:
                        continue
                    if first_hit_logits is None and not autoregressive_first_hit_mode:
                        continue
                    if autoregressive_first_hit_mode and edge_features is None:
                        continue
                    try:
                        tree_obj = Tree(current_newick)
                        n_leaves = int(tree_obj.n_leaves)
                    except Exception:
                        continue
                    split_masks_num = [int(m) for m in edge_split_masks[num]]
                    split_masks_nonzero = [m for m in split_masks_num if m != 0]
                    if not split_masks_nonzero:
                        continue
                    real_max_bit = max(m.bit_length() for m in split_masks_nonzero)
                    full_mask = (1 << real_max_bit) - 1 if real_max_bit > 0 else 0
                    try:
                        encoder = BHVEncoder()
                        bhv_masks, bhv_lengths = encoder.return_BHV_encoding(tree_obj)
                        bhv_len_map = {
                            int(m): float(l)
                            for m, l in zip(bhv_masks, bhv_lengths)
                            if l is not None
                        }
                    except Exception:
                        continue
                    target_set = {int(x) for x in target_sets[num]}
                    logits = []
                    targets = []
                    velocity_preds = []
                    matched_masks = []
                    edge_feature_rows = []
                    for edge_idx, mask in enumerate(split_masks_num):
                        if mask == 0:
                            continue
                        k_bits = int(mask).bit_count()
                        if min(k_bits, real_max_bit - k_bits) == 1:
                            continue
                        edge_length = bhv_len_map.get(int(mask))
                        if edge_length is None and full_mask:
                            edge_length = bhv_len_map.get(full_mask ^ int(mask))
                        if edge_length is None or float(edge_length) <= 1e-8:
                            continue
                        if not autoregressive_first_hit_mode:
                            logits.append(first_hit_logits[num, edge_idx, 0])
                        velocity_preds.append(v_pred[num, edge_idx, 0])
                        matched_masks.append(int(mask))
                        targets.append(1.0 if int(mask) in target_set else 0.0)
                        if edge_features is not None:
                            edge_feature_rows.append(edge_features[num, edge_idx])
                    if not velocity_preds:
                        continue
                    velocity_tensor = torch.stack(velocity_preds).reshape(-1)
                    target_tensor = torch.tensor(
                        targets,
                        device=velocity_tensor.device,
                        dtype=velocity_tensor.dtype,
                    )
                    target_mask = target_tensor > 0.5
                    if autoregressive_first_hit_mode:
                        if not edge_feature_rows:
                            continue
                        edge_feature_tensor = torch.stack(edge_feature_rows, dim=0).to(
                            device=velocity_tensor.device,
                            dtype=edge_features.dtype,
                        )
                        sample_loss, ar_stats = self.model.first_hit_autoregressive_group_loss(
                            edge_feature_tensor,
                            target_mask,
                        )
                        sample_loss = direct_set_bce_weight * sample_loss
                        direct_set_pos_weights.append(1.0)
                    else:
                        logits_tensor = torch.stack(logits).reshape(-1)
                        pos_weight = None
                        if bool(
                            getattr(
                                self,
                                "velocity_probe_direct_set_positive_reweight",
                                False,
                            )
                        ):
                            pos = target_tensor.sum()
                            neg = target_tensor.numel() - pos
                            if (
                                float(pos.item()) > 0.0
                                and float(neg.item()) > 0.0
                            ):
                                pos_weight_value = torch.clamp(
                                    neg / pos, min=1.0
                                )
                                pos_weight_value = torch.pow(
                                    pos_weight_value,
                                    float(
                                        getattr(
                                            self,
                                            "velocity_probe_direct_set_positive_reweight_power",
                                            1.0,
                                        )
                                    ),
                                )
                                pos_weight_max = getattr(
                                    self,
                                    "velocity_probe_direct_set_positive_reweight_max",
                                    None,
                                )
                                if pos_weight_max is not None and pos_weight_max > 0.0:
                                    pos_weight_value = torch.clamp(
                                        pos_weight_value,
                                        min=1.0,
                                        max=float(pos_weight_max),
                                    )
                                pos_weight = pos_weight_value.detach()
                        sample_loss = direct_set_bce_weight * F.binary_cross_entropy_with_logits(
                            logits_tensor,
                            target_tensor,
                            pos_weight=pos_weight,
                        )
                        direct_set_pos_weights.append(
                            float(pos_weight.detach().item())
                            if pos_weight is not None
                            else 1.0
                        )
                    target_negative_weight = float(
                        getattr(
                            self,
                            "velocity_probe_direct_set_target_negative_weight",
                            1.0,
                        )
                    )
                    if bool(target_mask.any().item()) and target_negative_weight > 0.0:
                        target_negative_loss = F.softplus(
                            velocity_tensor[target_mask]
                        ).mean()
                        sample_loss = sample_loss + (
                            target_negative_weight * target_negative_loss
                        )
                        target_negative_losses.append(target_negative_loss.detach())
                        with torch.no_grad():
                            target_negative_rates.append(
                                float((velocity_tensor[target_mask] < 0.0).float().mean().item())
                            )
                    nontarget_mask = ~target_mask
                    nontarget_nonnegative_weight = float(
                        getattr(
                            self,
                            "velocity_probe_direct_set_nontarget_nonnegative_weight",
                            0.0,
                        )
                    )
                    if (
                        bool(nontarget_mask.any().item())
                        and nontarget_nonnegative_weight > 0.0
                    ):
                        nontarget_nonnegative_loss = F.softplus(
                            -velocity_tensor[nontarget_mask]
                        ).mean()
                        sample_loss = sample_loss + (
                            nontarget_nonnegative_weight
                            * nontarget_nonnegative_loss
                        )
                        nontarget_nonnegative_losses.append(
                            nontarget_nonnegative_loss.detach()
                        )
                    direct_losses.append(sample_loss)
                    with torch.no_grad():
                        if autoregressive_first_hit_mode:
                            pred_mask = self.model.predict_first_hit_autoregressive_mask(
                                edge_feature_tensor
                            )
                            pred_set = {
                                matched_masks[i]
                                for i in range(len(matched_masks))
                                if bool(pred_mask[i].item())
                            }
                        else:
                            pred_mask = torch.sigmoid(logits_tensor) > 0.5
                            pred_set = {
                                matched_masks[i]
                                for i in range(len(matched_masks))
                                if bool(pred_mask[i].item())
                            }
                        exact_flags.append(float(pred_set == target_set))
                        union = len(pred_set | target_set)
                        inter = len(pred_set & target_set)
                        jaccards.append(
                            float(inter / union) if union > 0 else 1.0
                        )

                if direct_losses:
                    loss = torch.stack(direct_losses).mean()
                else:
                    loss = torch.tensor(
                        0.0,
                        device=self.device,
                        requires_grad=not eval,
                    )
                zero = torch.zeros((), device=loss.device, dtype=loss.dtype)
                logs = {
                    "loss": loss,
                    "loss_regression": loss,
                    "loss_auxiliary": zero,
                    "velocity/probe_direct_set_exact_rate": torch.tensor(
                        float(sum(exact_flags) / len(exact_flags))
                        if exact_flags
                        else 0.0,
                        device=loss.device,
                        dtype=torch.float32,
                    ),
                    "velocity/probe_direct_set_mean_jaccard": torch.tensor(
                        float(sum(jaccards) / len(jaccards)) if jaccards else 0.0,
                        device=loss.device,
                        dtype=torch.float32,
                    ),
                    "velocity/probe_direct_set_target_negative_rate": torch.tensor(
                        float(sum(target_negative_rates) / len(target_negative_rates))
                        if target_negative_rates
                        else 0.0,
                        device=loss.device,
                        dtype=torch.float32,
                    ),
                    "velocity/probe_direct_set_target_negative_loss": torch.tensor(
                        float(
                            torch.stack(target_negative_losses).mean().item()
                        )
                        if target_negative_losses
                        else 0.0,
                        device=loss.device,
                        dtype=torch.float32,
                    ),
                    "velocity/probe_direct_set_nontarget_nonnegative_loss": torch.tensor(
                        float(
                            torch.stack(nontarget_nonnegative_losses).mean().item()
                        )
                        if nontarget_nonnegative_losses
                        else 0.0,
                        device=loss.device,
                        dtype=torch.float32,
                    ),
                    "velocity/probe_direct_set_pos_weight": torch.tensor(
                        float(sum(direct_set_pos_weights) / len(direct_set_pos_weights))
                        if direct_set_pos_weights
                        else 1.0,
                        device=loss.device,
                        dtype=torch.float32,
                    ),
                    "velocity/probe_direct_set_bce_weight": torch.tensor(
                        direct_set_bce_weight,
                        device=loss.device,
                        dtype=torch.float32,
                    ),
                    "velocity/full_path_control_mode": torch.tensor(
                        1.0
                        if batch.get("_use_full_path_control_velocity_loss", False)
                        else 0.0,
                        device=loss.device,
                        dtype=torch.float32,
                    ),
                }
                if direct_set_debug_enabled:
                    logging.info(
                        "DIRECT_SET_DEBUG use_probe=%s pos_reweight=%s mean_pos_weight=%.6f samples=%d exact_rate=%.6f mean_jaccard=%.6f",
                        bool(batch.get("_use_probe_parity_direct_set_loss", False)),
                        bool(self.velocity_probe_direct_set_positive_reweight),
                        float(logs["velocity/probe_direct_set_pos_weight"].detach().item()),
                        int(len(direct_losses)),
                        float(logs["velocity/probe_direct_set_exact_rate"].detach().item()),
                        float(logs["velocity/probe_direct_set_mean_jaccard"].detach().item()),
                    )
                if direct_set_mse_weight <= 0.0:
                    return logs
                direct_set_loss = loss

            velocity_labels = batch["batched_velocity"]
            num_leaves = batch["num_leaves"]
            enc = BHVEncoder()
            gathered_velocity_labels = []
            v_pred_indices = []
            gathered_velocity_lengths = []
            gathered_boundary_vanish_targets = []

            for num in range(len(velocity_labels)):
                sub_gathered_velocity_labels = []
                sub_v_pred_indices = []
                sub_gathered_velocity_lengths = []
                sub_boundary_vanish_targets = []

                num_leave = int(num_leaves[num])
                split_masks_num = [int(m) for m in edge_split_masks[num]]
                split_masks_nonzero = [m for m in split_masks_num if m != 0]
                if len(split_masks_nonzero) == 0:
                    gathered_velocity_labels.append(
                        torch.tensor(sub_gathered_velocity_labels)
                    )
                    v_pred_indices.append(torch.tensor(sub_v_pred_indices))
                    gathered_velocity_lengths.append(
                        torch.tensor(sub_gathered_velocity_lengths)
                    )
                    continue

                real_max_bit = max(m.bit_length() for m in split_masks_nonzero)
                full_mask = (1 << real_max_bit) - 1 if real_max_bit > 0 else 0
                mask_to_idx = {m: i for i, m in enumerate(split_masks_num)}
                tree_obj = Tree(batch["original_trees"][num])
                tree_masks, tree_lengths = enc.return_BHV_encoding(tree_obj)
                length_map = {
                    int(m): float(l)
                    for m, l in zip(tree_masks, tree_lengths)
                    if l is not None
                }
                next_boundary_tree = batch.get("velocity_next_boundary_trees", [None])[
                    num
                ]
                next_boundary_active_masks = None
                if next_boundary_tree:
                    boundary_tree_obj = Tree(next_boundary_tree)
                    boundary_masks, boundary_lengths = enc.return_BHV_encoding(
                        boundary_tree_obj
                    )
                    next_boundary_active_masks = set()
                    for boundary_mask, boundary_length in zip(
                        boundary_masks, boundary_lengths
                    ):
                        if boundary_length is None or float(boundary_length) <= 1e-8:
                            continue
                        boundary_mask = int(boundary_mask)
                        if boundary_mask.bit_length() == real_max_bit + 1:
                            boundary_mask = remove_bit(boundary_mask, num_leave - 1)
                        elif boundary_mask.bit_length() > real_max_bit + 1:
                            continue

                        matched_boundary_mask = boundary_mask
                        if (
                            matched_boundary_mask not in mask_to_idx
                            and full_mask
                            and (full_mask ^ matched_boundary_mask) in mask_to_idx
                        ):
                            matched_boundary_mask = full_mask ^ matched_boundary_mask
                        next_boundary_active_masks.add(int(matched_boundary_mask))
                for vel in velocity_labels[num]:
                    original_vel = vel
                    if vel.bit_length() == real_max_bit + 1:
                        vel = remove_bit(vel, num_leave - 1)
                    elif vel.bit_length() > real_max_bit + 1:
                        raise Exception(
                            f"Whoa there is a big problem with this split mask {vel} vs real max {real_max_bit}!"
                        )

                    matched_vel = vel
                    if matched_vel not in mask_to_idx:
                        # Split orientation can flip after dummy-root removal; allow complement match.
                        complement_vel = full_mask ^ vel
                        if complement_vel in mask_to_idx:
                            matched_vel = complement_vel
                        else:
                            print(
                                f"This split {vel} from velocity labels is not in edge splits {split_masks_num}!"
                            )
                            print([i for i in range(vel.bit_length()) if (vel >> i) & 1])
                            raise Exception("Split not found in edge splits")
                    
                    #Ignore leaf edges
                    n_bits = real_max_bit
                    k = int(matched_vel).bit_count()
                    is_pendant = min(k, n_bits - k) == 1
                    if is_pendant:
                        continue

                    edge_len = length_map.get(int(matched_vel))
                    if edge_len is None and full_mask:
                        edge_len = length_map.get(full_mask ^ int(matched_vel))
                    if edge_len is None:
                        print(
                            f"Edge length not found for split {matched_vel} (original {original_vel}) in tree {batch['original_trees'][num]}"
                        )
                        print(f"Available splits: {split_masks_num}")
                        print(f"Length map keys: {list(length_map.keys())}")
                        raise Exception("Edge length not found for matched split")
                    if edge_len is None or float(edge_len) <= 1e-8:
                        continue

                    sub_gathered_velocity_labels.append(velocity_labels[num][original_vel])
                    sub_v_pred_indices.append(mask_to_idx[int(matched_vel)])
                    sub_gathered_velocity_lengths.append(float(edge_len))
                    if next_boundary_active_masks is not None:
                        sub_boundary_vanish_targets.append(
                            0.0 if int(matched_vel) in next_boundary_active_masks else 1.0
                        )


                gathered_velocity_labels.append(
                    torch.tensor(sub_gathered_velocity_labels)
                )
                v_pred_indices.append(torch.tensor(sub_v_pred_indices))
                gathered_velocity_lengths.append(torch.tensor(sub_gathered_velocity_lengths))
                gathered_boundary_vanish_targets.append(
                    torch.tensor(sub_boundary_vanish_targets, dtype=torch.float32)
                )

            # gathered_velocity_labels = torch.stack(gathered_velocity_labels)
            # v_pred_indices = torch.stack(v_pred_indices)

            # Fix: Flatten tensors to handle variable number of edges per tree
            preds_list = []
            first_hit_logits_list = []
            edge_features_list = []
            boundary_vanish_logits_list = []
            for b_idx in range(len(v_pred_indices)):
                indices = v_pred_indices[b_idx].to(v_pred.device)
                if indices.numel() > 0:
                    preds = v_pred[b_idx].index_select(0, indices)
                    preds_list.append(preds)
                    if first_hit_logits is not None:
                        first_hit_logits_list.append(
                            first_hit_logits[b_idx].index_select(0, indices)
                        )
                    if edge_features is not None:
                        edge_features_list.append(
                            edge_features[b_idx].index_select(0, indices)
                        )
                    if boundary_vanish_logits is not None:
                        boundary_vanish_logits_list.append(
                            boundary_vanish_logits[b_idx].index_select(0, indices)
                        )

            if len(preds_list) > 0:
                v_pred_gathered = torch.cat(preds_list).squeeze(-1)
                gathered_velocity_labels_flat = torch.cat(gathered_velocity_labels).to(
                    v_pred_gathered.device
                )
                gathered_velocity_lengths_flat = torch.cat(gathered_velocity_lengths).to(
                    v_pred_gathered.device
                )
                y = gathered_velocity_labels_flat
                p = v_pred_gathered
                lengths = gathered_velocity_lengths_flat
                first_hit_logits_gathered = (
                    torch.cat(first_hit_logits_list).squeeze(-1)
                    if first_hit_logits_list
                    else None
                )
                edge_features_gathered = (
                    torch.cat(edge_features_list, dim=0)
                    if edge_features_list
                    else None
                )
                velocity_group_sizes = [
                    int(indices.numel())
                    for indices in v_pred_indices
                    if int(indices.numel()) > 0
                ]
                if edge_features_gathered is not None:
                    p = self._refine_velocity_predictions(
                        p,
                        lengths=lengths,
                        edge_features=edge_features_gathered,
                        group_sizes=velocity_group_sizes,
                    )
                if first_hit_logits_gathered is not None or edge_features_gathered is not None:
                    first_hit_logits_gathered = self._compute_first_hit_logits(
                        first_hit_logits_gathered,
                        lengths=lengths,
                        velocities=p,
                        edge_features=edge_features_gathered,
                        group_sizes=velocity_group_sizes,
                    )
                boundary_vanish_logits_gathered = (
                    torch.cat(boundary_vanish_logits_list).squeeze(-1)
                    if boundary_vanish_logits_list
                    else None
                )
                boundary_vanish_targets_flat = (
                    torch.cat(gathered_boundary_vanish_targets).to(v_pred_gathered.device)
                    if (
                        boundary_vanish_logits_gathered is not None
                        and gathered_boundary_vanish_targets
                        and sum(int(t.numel()) for t in gathered_boundary_vanish_targets)
                        == int(boundary_vanish_logits_gathered.numel())
                    )
                    else None
                )

                # --- Velocity diagnostics ---
                with torch.no_grad():
                    vel_metrics = _velocity_diagnostics(
                        p,
                        y,
                        topk=3,
                        sign_eps=self.velocity_sign_eps,
                        lengths=gathered_velocity_lengths_flat,
                    )
                if self.verbose:
                    logger.info(
                        f"Velocity metrics: MSE={vel_metrics['mse']:.6f}  "
                        f"Cosine={vel_metrics['cosine']:.4f}  "
                        f"Pearson={vel_metrics['pearson']:.4f}  "
                        f"Spearman={vel_metrics['spearman']:.4f}  "
                        f"SignAcc={vel_metrics['sign_acc']:.4f}  "
                        f"TopK={vel_metrics['topk_overlap']:.4f}  "
                        f"dtTopK={vel_metrics['dt_topk_overlap']:.4f}  "
                        f"dtFirstHitRecall={vel_metrics['dt_first_hit_recall']:.4f}  "
                        f"dtHitRelErr={vel_metrics['dt_hit_rel_err']:.4f}  "
                        f"N={vel_metrics['n_edges']}"
                    )
                logs.update(
                    {
                        "velocity/cosine": torch.tensor(
                            vel_metrics["cosine"], device=v_pred.device
                        ),
                        "velocity/pearson": torch.tensor(
                            vel_metrics["pearson"], device=v_pred.device
                        ),
                        "velocity/spearman": torch.tensor(
                            vel_metrics["spearman"], device=v_pred.device
                        ),
                        "velocity/sign_acc": torch.tensor(
                            vel_metrics["sign_acc"], device=v_pred.device
                        ),
                        "velocity/topk_overlap": torch.tensor(
                            vel_metrics["topk_overlap"], device=v_pred.device
                        ),
                        "velocity/dt_topk_overlap": torch.tensor(
                            vel_metrics["dt_topk_overlap"], device=v_pred.device
                        ),
                        "velocity/dt_first_hit_recall": torch.tensor(
                            vel_metrics["dt_first_hit_recall"], device=v_pred.device
                        ),
                        "velocity/dt_first_hit_precision": torch.tensor(
                            vel_metrics["dt_first_hit_precision"], device=v_pred.device
                        ),
                        "velocity/dt_neg_jaccard": torch.tensor(
                            vel_metrics["dt_neg_jaccard"], device=v_pred.device
                        ),
                        "velocity/length_jitter_attempted": torch.tensor(
                            velocity_perturb_stats["attempted"],
                            device=v_pred.device,
                        ),
                        "velocity/length_jitter_applied": torch.tensor(
                            velocity_perturb_stats["applied"],
                            device=v_pred.device,
                        ),
                    }
                )

                # eps = 1e-6
                # first_hit_tol = 0.01
                # contract = (y < -self.velocity_sign_eps) & (lengths > 1e-8)

                # Lc = lengths[contract].clamp_min(eps)
                # yc = y[contract]
                # pc = p[contract]

                # tau_true = Lc / (-yc).clamp_min(eps)
                # w = (tau_true.median().clamp_min(eps) / tau_true).clamp(max=20.0)

                # tau_min = tau_true.min()
                # first = (tau_true - tau_min).abs() <= 0.01  # true tol (could be 0)

                # boost = 5.0
                # w = w * (1.0 + boost * first.float())

                # loss = (w * (pc - yc)**2).mean()
                # import pdb; pdb.set_trace()

                # tau_true = lengths[contract] / (-y[contract]).clamp_min(eps)
                # tau_min = tau_true.min()
                # first = torch.abs(tau_true - tau_min) <= first_hit_tol
                # w = torch.ones_like(y[contract])
                # alpha = 10
                # w[first] = 1.0 + alpha
                # loss = (w * (p[contract] - y[contract])**2).mean()

                # ####OG LOSS HERE
                residual_sq = (p - y).pow(2)
                plain_mse = residual_sq.mean()

                abs_y = y.abs()
                eps = 1e-6
                scale = abs_y.median().clamp_min(eps)  # robust scale
                w = (abs_y / scale).clamp(min=0.0, max=20.0)
                weighted_mse = (w * residual_sq).sum() / w.sum().clamp_min(eps)
                if self.velocity_loss_mode == "plain":
                    loss = plain_mse
                elif self.velocity_loss_mode == "weighted":
                    loss = weighted_mse
                else:
                    loss = (
                        self.velocity_loss_plain_weight * plain_mse
                        + (1.0 - self.velocity_loss_plain_weight) * weighted_mse
                    )
                regression_loss = loss
                auxiliary_loss = p.new_tensor(0.0)

                # ------------------------------------------------------------------
                # First-hit structured loss on true contracting edges:
                #   1) keep the fastest true contracting edges accurate in rate space
                #   2) ensure true first-hit edges remain earlier than later edges
                #   3) keep the tied first-hit set collapsed together
                # ------------------------------------------------------------------
                # contract_mask = (y < -self.velocity_sign_eps) & (lengths > 1e-8)
                # fast_rate_loss = p.new_tensor(0.0)
                # first_hit_dt_loss = p.new_tensor(0.0)

                # if int(contract_mask.sum()) > 0:
                #     Lc = lengths[contract_mask].clamp_min(eps)
                #     yc = y[contract_mask]
                #     pc = p[contract_mask]

                #     # First-hit ordering is governed by contraction rate (-v / length).
                #     rate_true = (-yc).clamp_min(eps) / Lc
                #     rate_pred = F.softplus(
                #         -pc, beta=self.velocity_event_rate_beta
                #     ) / Lc

                #     # True collapse times for truly contracting edges
                #     tau_true = 1.0 / rate_true.clamp_min(eps)
                #     tau_pred = 1.0 / rate_pred.clamp_min(eps)

                #     # Identify the true first-hit set with tolerance
                #     tau_true_min = tau_true.min()
                #     first_mask = torch.abs(tau_true - tau_true_min) <= 0.01
                #     later_mask = ~first_mask
                #     fast_k = min(8, int(tau_true.numel()))
                #     fast_idx = torch.argsort(tau_true)[:fast_k]
                #     fast_mask = torch.zeros_like(first_mask)
                #     fast_mask[fast_idx] = True
                #     rate_scale = rate_true[fast_mask].median().clamp_min(1.0)
                #     fast_rate_loss = F.smooth_l1_loss(
                #         rate_pred[fast_mask] / rate_scale,
                #         rate_true[fast_mask] / rate_scale,
                #     )

                #     z_pred = torch.log(tau_pred.clamp_min(eps))

                #     # 1) Tie loss: first-hit edges should have similar predicted dt
                #     if int(first_mask.sum()) > 1:
                #         z_first = z_pred[first_mask]
                #         first_hit_tie_loss = ((z_first - z_first.mean()) ** 2).mean()
                #     else:
                #         first_hit_tie_loss = p.new_tensor(0.0)
                #     if int(first_mask.sum()) > 0:
                #         first_hit_dt_loss = F.smooth_l1_loss(
                #             z_pred[first_mask],
                #             torch.log(tau_true[first_mask].clamp_min(eps)),
                #         )

                #     # 2) Rank loss: first-hit edges should be earlier than later contracting edges
                #     if int(first_mask.sum()) > 0 and int(later_mask.sum()) > 0:
                #         z_first = z_pred[first_mask][:, None]   # shape [F, 1]
                #         z_later = z_pred[later_mask][None, :]   # shape [1, L]
                #         first_hit_rank_loss = F.relu(
                #             z_later - z_first + 0.02
                #         ).mean()
                #     else:
                #         first_hit_rank_loss = p.new_tensor(0.0)

                #     first_hit_loss = (
                #         first_hit_tie_loss
                #         + first_hit_rank_loss
                #     )

                #     n_contract = int(contract_mask.sum())
                #     n_first = int(first_mask.sum())
                #     n_later = int(later_mask.sum())
                # else:
                #     first_hit_tie_loss = p.new_tensor(0.0)
                #     first_hit_rank_loss = p.new_tensor(0.0)
                #     first_hit_loss = p.new_tensor(0.0)
                #     n_contract = 0
                #     n_first = 0
                #     n_later = 0

                # first_hit_aux_weight = float(
                #     min(max((self.current_step_value - 100) / 200.0, 0.0), 1.0)
                # )

                first_hit_velocity_loss = p.new_tensor(0.0)
                logtau_all_loss_raw = p.new_tensor(0.0)
                logtau_all_loss = p.new_tensor(0.0)
                logtau_first_over_loss_raw = p.new_tensor(0.0)
                logtau_first_over_loss = p.new_tensor(0.0)
                logtau_first_tie_loss_raw = p.new_tensor(0.0)
                logtau_first_tie_loss = p.new_tensor(0.0)
                logtau_predset_over_loss_raw = p.new_tensor(0.0)
                logtau_predset_over_loss = p.new_tensor(0.0)
                event_loss_raw = p.new_tensor(0.0)
                event_loss = p.new_tensor(0.0)
                event_precision_loss_raw = p.new_tensor(0.0)
                event_precision_loss = p.new_tensor(0.0)
                first_hit_head_loss_raw = p.new_tensor(0.0)
                first_hit_head_loss = p.new_tensor(0.0)
                boundary_vanish_head_loss_raw = p.new_tensor(0.0)
                boundary_vanish_head_loss = p.new_tensor(0.0)
                boundary_time_head_loss_raw = p.new_tensor(0.0)
                boundary_time_head_loss = p.new_tensor(0.0)
                event_stats = {
                    "n_candidates": 0,
                    "target_first_size": 0,
                    "pred_first_mass": 0.0,
                    "top1_hits_first_set": 0.0,
                }
                event_precision_stats = {
                    "margin_gap": 0.0,
                    "n_pos": 0,
                    "n_neg": 0,
                    "violated": 0.0,
                }
                first_hit_head_stats = {
                    "n_candidates": 0,
                    "target_first_size": 0,
                    "pred_first_size": 0,
                    "top1_hits_first_set": 0.0,
                    "recall": 0.0,
                    "precision": 0.0,
                    "jaccard": 0.0,
                }
                boundary_vanish_head_stats = {
                    "n_candidates": 0,
                    "target_size": 0,
                    "pred_size": 0,
                    "top1_hits_target_set": 0.0,
                    "recall": 0.0,
                    "precision": 0.0,
                    "jaccard": 0.0,
                }
                boundary_time_head_stats = {
                    "n_groups": 0,
                    "n_valid": 0,
                    "dt_pred_mean": 0.0,
                    "dt_true_mean": 0.0,
                    "dt_rel_err_mean": 0.0,
                }
                first_hit_extra_penalty_raw = p.new_tensor(0.0)
                first_hit_fp_mass = p.new_tensor(0.0)
                first_hit_fn_mass = p.new_tensor(0.0)
                use_full_path_control_velocity_loss = bool(
                    batch.get("_use_full_path_control_velocity_loss", False)
                )
                if use_full_path_control_velocity_loss:
                    control_loss, control_parts = _full_path_control_velocity_loss(
                        p=p,
                        y=y,
                        lengths=lengths,
                        first_hit_logits=first_hit_logits_gathered,
                        group_sizes=velocity_group_sizes,
                        velocity_sign_eps=self.velocity_sign_eps,
                        dt_eps=self.velocity_dt_eps,
                        first_hit_tol=self.velocity_first_hit_loss_tol,
                        first_hit_head_weight=self.velocity_first_hit_head_weight,
                        logtau_all_weight=self.velocity_logtau_all_weight,
                        logtau_first_over_weight=self.velocity_logtau_first_over_weight,
                        logtau_first_tie_weight=self.velocity_logtau_first_tie_weight,
                        logtau_predset_over_weight=self.velocity_logtau_predset_over_weight,
                    )
                    loss = control_loss
                    regression_loss = control_parts["mse"]
                    auxiliary_loss = loss - regression_loss
                    logtau_all_loss_raw = control_parts["logtau_all_raw"]
                    logtau_all_loss = control_parts["logtau_all"]
                    logtau_first_over_loss_raw = control_parts["logtau_first_over_raw"]
                    logtau_first_over_loss = control_parts["logtau_first_over"]
                    logtau_first_tie_loss_raw = control_parts["logtau_first_tie_raw"]
                    logtau_first_tie_loss = control_parts["logtau_first_tie"]
                    logtau_predset_over_loss_raw = control_parts[
                        "logtau_predset_over_raw"
                    ]
                    logtau_predset_over_loss = control_parts["logtau_predset_over"]
                    first_hit_head_loss_raw = control_parts["firsthit_bce_raw"]
                    first_hit_head_loss = control_parts["firsthit_bce"]
                    logs["velocity/full_path_control_mode"] = torch.tensor(
                        1.0, dtype=torch.float32, device=v_pred.device
                    )
                contract_mask = (y < -self.velocity_sign_eps) & (lengths > 1e-8)
                if (
                    not use_full_path_control_velocity_loss
                    and (
                    self.velocity_logtau_all_weight > 0.0
                    and int(contract_mask.sum()) > 0
                    )
                ):
                    dt_eps = float(self.velocity_dt_eps)
                    Lc = lengths[contract_mask].clamp_min(dt_eps)
                    yc = y[contract_mask]
                    pc = p[contract_mask]
                    tau_true = Lc / (-yc).clamp_min(dt_eps)
                    tau_pred = Lc / (-pc).clamp_min(dt_eps)
                    logtau_all_loss_raw = F.smooth_l1_loss(
                        torch.log(tau_pred.clamp_min(dt_eps)),
                        torch.log(tau_true.clamp_min(dt_eps)),
                    )
                    logtau_all_loss = (
                        self.velocity_logtau_all_weight * logtau_all_loss_raw
                    )
                    loss = loss + logtau_all_loss
                    auxiliary_loss = auxiliary_loss + logtau_all_loss
                if (
                    not use_full_path_control_velocity_loss
                    and (
                    (
                        self.velocity_logtau_first_over_weight > 0.0
                        or self.velocity_logtau_first_tie_weight > 0.0
                    )
                    and int(contract_mask.sum()) > 0
                    and velocity_group_sizes
                    )
                ):
                    dt_eps = float(self.velocity_dt_eps)
                    over_losses = []
                    tie_losses = []
                    start_idx = 0
                    for group_size in velocity_group_sizes:
                        end_idx = start_idx + int(group_size)
                        Lg = lengths[start_idx:end_idx]
                        yg = y[start_idx:end_idx]
                        pg = p[start_idx:end_idx]
                        start_idx = end_idx
                        group_contract = (yg < -self.velocity_sign_eps) & (Lg > 1e-8)
                        if not bool(group_contract.any().item()):
                            continue
                        tau_true = Lg[group_contract].clamp_min(dt_eps) / (
                            -yg[group_contract]
                        ).clamp_min(dt_eps)
                        tau_min = tau_true.min()
                        group_contract_idx = torch.nonzero(
                            group_contract, as_tuple=False
                        ).reshape(-1)
                        first_idx = group_contract_idx[
                            torch.abs(tau_true - tau_min)
                            <= float(self.velocity_first_hit_loss_tol)
                        ]
                        if int(first_idx.numel()) == 0:
                            continue
                        tau_true_first = Lg[first_idx].clamp_min(dt_eps) / (
                            -yg[first_idx]
                        ).clamp_min(dt_eps)
                        tau_pred_first = Lg[first_idx].clamp_min(dt_eps) / (
                            -pg[first_idx]
                        ).clamp_min(dt_eps)
                        pred_max = torch.log(tau_pred_first.clamp_min(dt_eps)).max()
                        true_max = torch.log(tau_true_first.clamp_min(dt_eps)).max()
                        over_losses.append(F.relu(pred_max - true_max).pow(2))
                        if int(tau_pred_first.numel()) > 1:
                            log_pred_first = torch.log(
                                tau_pred_first.clamp_min(dt_eps)
                            )
                            tie_losses.append(
                                (
                                    log_pred_first - log_pred_first.mean()
                                ).pow(2).mean()
                            )
                    if over_losses and self.velocity_logtau_first_over_weight > 0.0:
                        logtau_first_over_loss_raw = torch.stack(over_losses).mean()
                        logtau_first_over_loss = (
                            self.velocity_logtau_first_over_weight
                            * logtau_first_over_loss_raw
                        )
                        loss = loss + logtau_first_over_loss
                        auxiliary_loss = auxiliary_loss + logtau_first_over_loss
                    if tie_losses and self.velocity_logtau_first_tie_weight > 0.0:
                        logtau_first_tie_loss_raw = torch.stack(tie_losses).mean()
                        logtau_first_tie_loss = (
                            self.velocity_logtau_first_tie_weight
                            * logtau_first_tie_loss_raw
                        )
                        loss = loss + logtau_first_tie_loss
                        auxiliary_loss = auxiliary_loss + logtau_first_tie_loss
                if (
                    not use_full_path_control_velocity_loss
                    and self.velocity_dt_hit_weight > 0.0
                    and int(contract_mask.sum()) > 0
                ):
                    Lc = lengths[contract_mask].clamp_min(eps)
                    yc = y[contract_mask]
                    pc = p[contract_mask]
                    tau_true = Lc / (-yc).clamp_min(eps)
                    tau_true_min = tau_true.min()
                    first_mask = (
                        torch.abs(tau_true - tau_true_min)
                        <= float(self.velocity_first_hit_loss_tol)
                    )
                    if int(first_mask.sum()) > 0:
                        first_hit_velocity_loss = F.smooth_l1_loss(
                            pc[first_mask], yc[first_mask]
                        )

                # first_hit_aux_weight = float(
                #     min(max((self.current_step_value - 100) / 200.0, 0.0), 1.0)
                # )

                # loss = (
                #     loss
                #     + 1
                #     * (0.5 * self.velocity_dt_hit_weight * first_hit_velocity_loss)
                # )

                # loss = loss

                # loss = (
                #     loss
                #     + self.velocity_dt_hit_weight * first_hit_velocity_loss
                # )

                first_hit_velocity_aux = (
                    self.velocity_dt_hit_weight * first_hit_velocity_loss
                )
                if not use_full_path_control_velocity_loss:
                    loss = loss + first_hit_velocity_aux
                    auxiliary_loss = auxiliary_loss + first_hit_velocity_aux
                if (
                    not use_full_path_control_velocity_loss
                    and self.velocity_event_weight > 0.0
                ):
                    event_loss_raw, event_stats = _boundary_event_distribution_loss(
                        lengths=lengths,
                        y_true=y,
                        y_pred=p,
                        velocity_sign_eps=self.velocity_sign_eps,
                        dt_eps=self.velocity_dt_eps,
                        temp=self.velocity_event_temp,
                        rate_beta=self.velocity_event_rate_beta,
                        normalize_by_log_candidates=self.velocity_event_normalize_by_log_candidates,
                    )
                    event_loss = self.velocity_event_weight * event_loss_raw
                    loss = loss + event_loss
                    auxiliary_loss = auxiliary_loss + event_loss
                if (
                    not use_full_path_control_velocity_loss
                    and self.velocity_event_precision_weight > 0.0
                ):
                    (
                        event_precision_loss_raw,
                        event_precision_stats,
                    ) = _boundary_event_precision_margin_loss(
                        lengths=lengths,
                        y_true=y,
                        y_pred=p,
                        velocity_sign_eps=self.velocity_sign_eps,
                        dt_eps=self.velocity_dt_eps,
                        temp=self.velocity_event_temp,
                        rate_beta=self.velocity_event_rate_beta,
                        margin=self.velocity_event_precision_margin,
                    )
                    event_precision_loss = (
                        self.velocity_event_precision_weight
                        * event_precision_loss_raw
                    )
                    loss = loss + event_precision_loss
                    auxiliary_loss = auxiliary_loss + event_precision_loss
                if (
                    not use_full_path_control_velocity_loss
                    and (
                    self.velocity_first_hit_head_weight > 0.0
                    and first_hit_logits_gathered is not None
                    )
                ):
                    (
                        first_hit_head_loss_raw,
                        first_hit_head_stats,
                    ) = _first_hit_grouped_set_bce_loss(
                        lengths=lengths,
                        y_true=y,
                        first_hit_logits=first_hit_logits_gathered,
                        group_sizes=velocity_group_sizes,
                        velocity_sign_eps=self.velocity_sign_eps,
                        dt_eps=self.velocity_dt_eps,
                        first_hit_tol=self.velocity_first_hit_loss_tol,
                    )
                    first_hit_extra_penalty_raw = p.new_tensor(0.0)
                    first_hit_fp_mass = p.new_tensor(0.0)
                    first_hit_fn_mass = p.new_tensor(0.0)
                    if (
                        self.velocity_first_hit_false_positive_mass_weight > 0.0
                        or self.velocity_first_hit_false_negative_mass_weight > 0.0
                    ):
                        first_hit_mass_penalty_raw, first_hit_mass_stats = (
                            _first_hit_grouped_soft_mass_penalty(
                                lengths=lengths,
                                y_true=y,
                                first_hit_logits=first_hit_logits_gathered,
                                group_sizes=velocity_group_sizes,
                                velocity_sign_eps=self.velocity_sign_eps,
                                dt_eps=self.velocity_dt_eps,
                                first_hit_tol=self.velocity_first_hit_loss_tol,
                            )
                        )
                        first_hit_fp_mass = first_hit_mass_stats.get(
                            "fp_mass_tensor", p.new_tensor(0.0)
                        )
                        first_hit_fn_mass = first_hit_mass_stats.get(
                            "fn_mass_tensor", p.new_tensor(0.0)
                        )
                        first_hit_extra_penalty_raw = (
                            self.velocity_first_hit_false_positive_mass_weight
                            * first_hit_fp_mass
                            + self.velocity_first_hit_false_negative_mass_weight
                            * first_hit_fn_mass
                        )
                    first_hit_head_loss = (
                        self.velocity_first_hit_head_weight
                        * (first_hit_head_loss_raw + first_hit_extra_penalty_raw)
                    )
                    loss = loss + first_hit_head_loss
                    auxiliary_loss = auxiliary_loss + first_hit_head_loss
                    logs["velocity/first_hit_fp_mass_raw"] = first_hit_fp_mass
                    logs["velocity/first_hit_fn_mass_raw"] = first_hit_fn_mass
                    logs["velocity/first_hit_extra_penalty_raw"] = (
                        first_hit_extra_penalty_raw
                    )
                if (
                    not use_full_path_control_velocity_loss
                    and (
                    self.velocity_boundary_time_head_weight > 0.0
                    and edge_features_gathered is not None
                    and velocity_group_sizes
                    )
                ):
                    boundary_time_pred_log = self._predict_boundary_time_log(
                        lengths=lengths,
                        velocities=p,
                        edge_features=edge_features_gathered,
                        group_sizes=velocity_group_sizes,
                    )
                    if boundary_time_pred_log is not None:
                        target_logs = []
                        pred_logs = []
                        pred_dts = []
                        true_dts = []
                        start_idx = 0
                        for group_idx, group_size in enumerate(velocity_group_sizes):
                            end_idx = start_idx + int(group_size)
                            group_lengths = lengths[start_idx:end_idx]
                            group_targets = y[start_idx:end_idx]
                            contract_group = (
                                (group_targets < -self.velocity_sign_eps)
                                & (group_lengths > 1e-8)
                            )
                            if int(contract_group.sum()) > 0:
                                tau_true_group = (
                                    group_lengths[contract_group].clamp_min(eps)
                                    / (-group_targets[contract_group]).clamp_min(eps)
                                )
                                target_logs.append(
                                    torch.log(tau_true_group.min().clamp_min(eps))
                                )
                                pred_logs.append(boundary_time_pred_log[group_idx])
                                pred_dts.append(
                                    torch.exp(boundary_time_pred_log[group_idx])
                                )
                                true_dts.append(tau_true_group.min())
                            start_idx = end_idx

                        if pred_logs:
                            pred_log_tensor = torch.stack(pred_logs)
                            target_log_tensor = torch.stack(target_logs)
                            pred_dt_tensor = torch.stack(pred_dts)
                            true_dt_tensor = torch.stack(true_dts)
                            boundary_time_head_loss_raw = F.smooth_l1_loss(
                                pred_log_tensor,
                                target_log_tensor,
                            )
                            boundary_time_head_loss = (
                                self.velocity_boundary_time_head_weight
                                * boundary_time_head_loss_raw
                            )
                            loss = loss + boundary_time_head_loss
                            auxiliary_loss = auxiliary_loss + boundary_time_head_loss
                            rel_err = (
                                (pred_dt_tensor - true_dt_tensor).abs()
                                / true_dt_tensor.clamp_min(eps)
                            )
                            boundary_time_head_stats = {
                                "n_groups": int(len(velocity_group_sizes)),
                                "n_valid": int(pred_log_tensor.numel()),
                                "dt_pred_mean": float(pred_dt_tensor.mean().item()),
                                "dt_true_mean": float(true_dt_tensor.mean().item()),
                                "dt_rel_err_mean": float(rel_err.mean().item()),
                            }
                if (
                    not use_full_path_control_velocity_loss
                    and (
                    self.velocity_boundary_vanish_head_weight > 0.0
                    and boundary_vanish_logits_gathered is not None
                    and boundary_vanish_targets_flat is not None
                    )
                ):
                    (
                        boundary_vanish_head_loss_raw,
                        boundary_vanish_head_stats,
                    ) = _edge_set_bce_loss(
                        boundary_vanish_logits_gathered,
                        boundary_vanish_targets_flat,
                    )
                    boundary_vanish_head_loss = (
                        self.velocity_boundary_vanish_head_weight
                        * boundary_vanish_head_loss_raw
                    )
                    loss = loss + boundary_vanish_head_loss
                    auxiliary_loss = auxiliary_loss + boundary_vanish_head_loss

                # loss = (
                #     loss
                #     + first_hit_aux_weight
                #     * (
                #         0.02 * fast_rate_loss
                #         + 0.05 * first_hit_dt_loss
                #         + 0.02 * first_hit_loss
                #     )
                # )
    
            else:
                loss = torch.tensor(0.0, device=v_pred.device, requires_grad=True)
                regression_loss = loss.detach() * 0.0
                auxiliary_loss = loss.detach() * 0.0
                plain_mse = loss.detach() * 0.0
                weighted_mse = loss.detach() * 0.0
                first_hit_velocity_loss = loss.detach() * 0.0
                logtau_all_loss_raw = loss.detach() * 0.0
                logtau_all_loss = loss.detach() * 0.0
                logtau_first_over_loss_raw = loss.detach() * 0.0
                logtau_first_over_loss = loss.detach() * 0.0
                logtau_first_tie_loss_raw = loss.detach() * 0.0
                logtau_first_tie_loss = loss.detach() * 0.0
                logtau_predset_over_loss_raw = loss.detach() * 0.0
                logtau_predset_over_loss = loss.detach() * 0.0
                event_loss_raw = loss.detach() * 0.0
                event_loss = loss.detach() * 0.0
                event_precision_loss_raw = loss.detach() * 0.0
                event_precision_loss = loss.detach() * 0.0
                first_hit_head_loss_raw = loss.detach() * 0.0
                first_hit_head_loss = loss.detach() * 0.0
                boundary_vanish_head_loss_raw = loss.detach() * 0.0
                boundary_vanish_head_loss = loss.detach() * 0.0
                boundary_time_head_loss_raw = loss.detach() * 0.0
                boundary_time_head_loss = loss.detach() * 0.0
                event_stats = {
                    "n_candidates": 0,
                    "target_first_size": 0,
                    "pred_first_mass": 0.0,
                    "top1_hits_first_set": 0.0,
                }
                event_precision_stats = {
                    "margin_gap": 0.0,
                    "n_pos": 0,
                    "n_neg": 0,
                    "violated": 0.0,
                }
                first_hit_head_stats = {
                    "n_candidates": 0,
                    "target_first_size": 0,
                    "pred_first_size": 0,
                    "top1_hits_first_set": 0.0,
                    "recall": 0.0,
                    "precision": 0.0,
                    "jaccard": 0.0,
                }
                boundary_vanish_head_stats = {
                    "n_candidates": 0,
                    "target_size": 0,
                    "pred_size": 0,
                    "top1_hits_target_set": 0.0,
                    "recall": 0.0,
                    "precision": 0.0,
                    "jaccard": 0.0,
                }
                boundary_time_head_stats = {
                    "n_groups": 0,
                    "n_valid": 0,
                    "dt_pred_mean": 0.0,
                    "dt_true_mean": 0.0,
                    "dt_rel_err_mean": 0.0,
                }
                n_contract = 0
            mse_branch_loss = loss
            mse_branch_regression_loss = regression_loss
            mse_branch_auxiliary_loss = auxiliary_loss
            if direct_set_loss is not None:
                loss = (
                    direct_set_loss_weight * direct_set_loss
                    + direct_set_mse_weight * mse_branch_loss
                )
                regression_loss = direct_set_mse_weight * mse_branch_regression_loss
                auxiliary_loss = loss - regression_loss
            # print("Wow congrats")
            logs.update(
                {
                    "velocity/loss_plain_mse": plain_mse.detach(),
                    "velocity/loss_weighted_mse": weighted_mse.detach(),
                    "velocity/mse_branch_loss_unscaled": mse_branch_loss.detach(),
                    "velocity/mse_branch_regression_unscaled": mse_branch_regression_loss.detach(),
                    "velocity/mse_branch_auxiliary_unscaled": mse_branch_auxiliary_loss.detach(),
                    "velocity/loss_regression_unscaled": regression_loss.detach(),
                    "velocity/loss_auxiliary_unscaled": auxiliary_loss.detach(),
                    "velocity/probe_direct_set_loss": (
                        torch.zeros((), device=v_pred.device, dtype=v_pred.dtype)
                        if direct_set_loss is None
                        else direct_set_loss.detach()
                    ),
                    "velocity/probe_direct_set_loss_weight": torch.tensor(
                        direct_set_loss_weight, device=v_pred.device
                    ),
                    "velocity/probe_direct_set_mse_weight": torch.tensor(
                        direct_set_mse_weight, device=v_pred.device
                    ),
                    "velocity/first_hit_velocity_loss": first_hit_velocity_loss.detach(),
                    "velocity/logtau_all_loss_raw": logtau_all_loss_raw.detach(),
                    "velocity/logtau_all_loss": logtau_all_loss.detach(),
                    "velocity/logtau_first_over_loss_raw": logtau_first_over_loss_raw.detach(),
                    "velocity/logtau_first_over_loss": logtau_first_over_loss.detach(),
                    "velocity/logtau_first_tie_loss_raw": logtau_first_tie_loss_raw.detach(),
                    "velocity/logtau_first_tie_loss": logtau_first_tie_loss.detach(),
                    "velocity/logtau_predset_over_loss_raw": logtau_predset_over_loss_raw.detach(),
                    "velocity/logtau_predset_over_loss": logtau_predset_over_loss.detach(),
                    "velocity/event_loss_raw": event_loss_raw.detach(),
                    "velocity/event_loss": event_loss.detach(),
                    "velocity/event_precision_loss_raw": event_precision_loss_raw.detach(),
                    "velocity/event_precision_loss": event_precision_loss.detach(),
                    "velocity/first_hit_head_loss_raw": first_hit_head_loss_raw.detach(),
                    "velocity/first_hit_head_loss": first_hit_head_loss.detach(),
                    "velocity/boundary_vanish_head_loss_raw": boundary_vanish_head_loss_raw.detach(),
                    "velocity/boundary_vanish_head_loss": boundary_vanish_head_loss.detach(),
                    "velocity/boundary_time_head_loss_raw": boundary_time_head_loss_raw.detach(),
                    "velocity/boundary_time_head_loss": boundary_time_head_loss.detach(),
                    "velocity/boundary_time_head_dt_pred_mean": torch.tensor(
                        boundary_time_head_stats["dt_pred_mean"], device=v_pred.device
                    ),
                    "velocity/boundary_time_head_dt_true_mean": torch.tensor(
                        boundary_time_head_stats["dt_true_mean"], device=v_pred.device
                    ),
                    "velocity/boundary_time_head_dt_rel_err_mean": torch.tensor(
                        boundary_time_head_stats["dt_rel_err_mean"], device=v_pred.device
                    ),
                }
            )
            logs["loss_regression"] = regression_loss
            logs["loss_auxiliary"] = auxiliary_loss
            logs["loss"] = loss
            # if len(preds_list) > 0:
            #     logger.info(
            #         f"Velocity loss ({self.velocity_loss_mode}): total={loss.item():.6f} "
            #         f"plain={plain_mse.item():.6f} weighted={weighted_mse.item():.6f} "
            #         # f"dt_gate={dt_gate.item():.4f} dt_candidates={dt_candidates_loss.item():.6f} "
            #         # f"dt_hit={dt_hit_loss.item():.6f}"
            #     )
            # else:

            if self.record and not is_replay_batch:
                dt_hit_pred_log = (
                    vel_metrics["dt_hit_pred"]
                    if np.isfinite(vel_metrics["dt_hit_pred"])
                    else -1.0
                )
                dt_hit_true_log = (
                    vel_metrics["dt_hit_true"]
                    if np.isfinite(vel_metrics["dt_hit_true"])
                    else -1.0
                )
                dt_hit_abs_err_log = (
                    vel_metrics["dt_hit_abs_err"]
                    if np.isfinite(vel_metrics["dt_hit_abs_err"])
                    else 1e6
                )
                dt_hit_rel_err_log = (
                    vel_metrics["dt_hit_rel_err"]
                    if np.isfinite(vel_metrics["dt_hit_rel_err"])
                    else 1e6
                )
                vel_wandb = {"train/velocity_loss": loss.item()}
                if len(preds_list) > 0:
                    vel_wandb.update({
                        "velocity/loss_plain_mse": plain_mse.item(),
                        "velocity/loss_weighted_mse": weighted_mse.item(),
                        "velocity/mse": vel_metrics["mse"],
                        "velocity/mse_vs_zero": vel_metrics["mse_vs_zero"],
                        "velocity/mse_vs_mean": vel_metrics["mse_vs_mean"],
                        "velocity/zero_baseline_mse": vel_metrics["zero_baseline_mse"],
                        "velocity/mean_baseline_mse": vel_metrics["mean_baseline_mse"],
                        "velocity/cosine": vel_metrics["cosine"],
                        "velocity/pearson": vel_metrics["pearson"],
                        "velocity/spearman": vel_metrics["spearman"],
                        "velocity/sign_acc": vel_metrics["sign_acc"],
                        "velocity/topk_overlap": vel_metrics["topk_overlap"],
                        "velocity/dt_hit_pred": dt_hit_pred_log,
                        "velocity/dt_hit_true": dt_hit_true_log,
                        "velocity/dt_hit_abs_err": dt_hit_abs_err_log,
                        "velocity/dt_hit_rel_err": dt_hit_rel_err_log,
                        "velocity/dt_first_hit_match": vel_metrics["dt_first_hit_match"],
                        "velocity/dt_first_hit_recall": vel_metrics["dt_first_hit_recall"],
                        "velocity/dt_first_hit_precision": vel_metrics["dt_first_hit_precision"],
                        "velocity/dt_topk_overlap": vel_metrics["dt_topk_overlap"],
                        "velocity/event_loss_raw": float(event_loss_raw.detach().item()),
                        "velocity/event_loss": float(event_loss.detach().item()),
                        "velocity/event_n_candidates": float(event_stats["n_candidates"]),
                        "velocity/event_target_first_size": float(event_stats["target_first_size"]),
                        "velocity/event_pred_first_mass": float(event_stats["pred_first_mass"]),
                        "velocity/event_top1_hits_first_set": float(event_stats["top1_hits_first_set"]),
                        "velocity/event_precision_loss_raw": float(event_precision_loss_raw.detach().item()),
                        "velocity/event_precision_loss": float(event_precision_loss.detach().item()),
                        "velocity/event_precision_margin_gap": float(event_precision_stats["margin_gap"]),
                        "velocity/event_precision_n_pos": float(event_precision_stats["n_pos"]),
                        "velocity/event_precision_n_neg": float(event_precision_stats["n_neg"]),
                        "velocity/event_precision_violated": float(event_precision_stats["violated"]),
                        "velocity/logtau_all_loss_raw": float(logtau_all_loss_raw.detach().item()),
                        "velocity/logtau_all_loss": float(logtau_all_loss.detach().item()),
                        "velocity/logtau_first_over_loss_raw": float(logtau_first_over_loss_raw.detach().item()),
                        "velocity/logtau_first_over_loss": float(logtau_first_over_loss.detach().item()),
                        "velocity/logtau_first_tie_loss_raw": float(logtau_first_tie_loss_raw.detach().item()),
                        "velocity/logtau_first_tie_loss": float(logtau_first_tie_loss.detach().item()),
                        "velocity/logtau_predset_over_loss_raw": float(logtau_predset_over_loss_raw.detach().item()),
                        "velocity/logtau_predset_over_loss": float(logtau_predset_over_loss.detach().item()),
                        "velocity/first_hit_head_loss_raw": float(first_hit_head_loss_raw.detach().item()),
                        "velocity/first_hit_head_loss": float(first_hit_head_loss.detach().item()),
                        "velocity/first_hit_head_target_size": float(first_hit_head_stats["target_first_size"]),
                        "velocity/first_hit_head_pred_size": float(first_hit_head_stats["pred_first_size"]),
                        "velocity/first_hit_head_top1_hits": float(first_hit_head_stats["top1_hits_first_set"]),
                        "velocity/first_hit_head_recall": float(first_hit_head_stats["recall"]),
                        "velocity/first_hit_head_precision": float(first_hit_head_stats["precision"]),
                        "velocity/first_hit_head_jaccard": float(first_hit_head_stats["jaccard"]),
                        "velocity/first_hit_fp_mass_raw": float(first_hit_fp_mass.detach().item()),
                        "velocity/first_hit_fn_mass_raw": float(first_hit_fn_mass.detach().item()),
                        "velocity/first_hit_extra_penalty_raw": float(first_hit_extra_penalty_raw.detach().item()),
                        "velocity/boundary_vanish_head_loss_raw": float(boundary_vanish_head_loss_raw.detach().item()),
                        "velocity/boundary_vanish_head_loss": float(boundary_vanish_head_loss.detach().item()),
                        "velocity/boundary_vanish_head_target_size": float(boundary_vanish_head_stats["target_size"]),
                        "velocity/boundary_vanish_head_pred_size": float(boundary_vanish_head_stats["pred_size"]),
                        "velocity/boundary_vanish_head_top1_hits": float(boundary_vanish_head_stats["top1_hits_target_set"]),
                        "velocity/boundary_vanish_head_recall": float(boundary_vanish_head_stats["recall"]),
                        "velocity/boundary_vanish_head_precision": float(boundary_vanish_head_stats["precision"]),
                        "velocity/boundary_vanish_head_jaccard": float(boundary_vanish_head_stats["jaccard"]),
                        "velocity/length_jitter_attempted": velocity_perturb_stats["attempted"],
                        "velocity/length_jitter_applied": velocity_perturb_stats["applied"],
                    })
                self._wandb_log_filtered(vel_wandb, step=self.stepper)
            # import pdb

            # pdb.set_trace()
        else:
            if prepared_batch:
                ar_prep_stats = ar_prep_stats or {
                    "rollin_attempted": 0.0,
                    "rollin_applied": 0.0,
                    "dagger_attempted": 0.0,
                    "dagger_applied": 0.0,
                    "dagger_rollout_steps": 0.0,
                    "structure_perturb_attempted": 0.0,
                    "structure_perturb_applied": 0.0,
                }
            else:
                batch, ar_prep_stats = self._prepare_autoregressive_training_batch(
                    batch
                )
            skip_autoregressive_merge_metrics = bool(
                batch.get("_skip_autoregressive_merge_metrics", False)
            )
            autoregressive_component_groups = (
                self._autoregressive_component_groups_for_batch(batch)
            )

            autoregressive_times = self._effective_autoregressive_time_tensor(
                batch["batched_autoregressive_time"]
            )
            if precomputed_outputs is None:
                all_group_logits = self.forward(
                    batch["tokenized_autoregressive_trees"],
                    autoregressive_times,
                    batch["phyla_embeddings"],
                    autoregressive=True,
                    autoregressive_component_groups=autoregressive_component_groups,
                    autoregressive_case_indices=batch.get("_autoregressive_case_indices"),
                    autoregressive_start_topology_features=batch.get(
                        "_autoregressive_start_topology_features"
                    ),
                )
            else:
                all_group_logits = precomputed_outputs

            if self.topology_decoder in {"birthset", "birthset_with_ar_fallback"}:
                return self._birthset_step_logs(
                    batch,
                    all_group_logits,
                    ar_prep_stats,
                    is_replay_batch,
                    update_split_bank=not eval,
                )

            found = {}
            label_targets_by_batch = []
            for batch_index, labeled_merge_cluster in enumerate(batch["batched_autoregressive_labels"]):
                group_targets = {}
                for label in labeled_merge_cluster:
                    result_split = int(label["result_split"])
                    components = tuple(int(component) for component in label["components"])
                    merge_indices = [int(idx) for idx in label["merge_indices"]]
                    found[(batch_index, result_split)] = False
                    group_targets.setdefault(components, []).append(
                        (result_split, merge_indices)
                    )
                label_targets_by_batch.append(group_targets)

            losses = []

            total_metrics = []
            alternative_target_counts = []
            stop_after_merge_losses = []
            stop_after_merge_accuracies = []
            stop_after_merge_targets = []
            stop_after_merge_predictions = []
            subset_size_losses = []
            subset_size_accuracies = []
            subset_size_target_means = []
            subset_size_prediction_means = []

            chosen_polytomies = []
            polytomy_logits = []
            polytomy_sizes = []  # Track size of each polytomy encountered

            for group in all_group_logits:
                logits = group["logits"]
                splits_in_polytomy = tuple(int(split) for split in group["splits_represented"])
                batch_index = int(group["batch_index"])
                decoder_mode = str(group.get("decoder_mode", "pairwise_threshold"))
                
                # Track polytomy size (number of splits in the polytomy)
                polytomy_sizes.append(len(splits_in_polytomy))

                explicit_subsets = []
                for resulting_split, idxs in label_targets_by_batch[batch_index].get(
                    splits_in_polytomy,
                    [],
                ):
                    found[(batch_index, resulting_split)] = True
                    explicit_subsets.append(
                        tuple(sorted(int(splits_in_polytomy[i]) for i in idxs))
                    )
                explicit_subsets = list(dict.fromkeys(explicit_subsets))

                candidate_subsets = list(dict.fromkeys(explicit_subsets))
                if (
                    self.autoregressive_target_mode == "ready_alternatives"
                    and "target_trees" in batch
                ):
                    ready_subsets = _ready_target_merge_subsets_for_group(
                        splits_in_polytomy,
                        batch["target_trees"][batch_index],
                        Tree(batch["newick_autoregressive_trees"][batch_index]).n_leaves,
                    )
                    for subset in ready_subsets:
                        subset = tuple(sorted(int(split) for split in subset))
                        if subset not in candidate_subsets:
                            candidate_subsets.append(subset)

                alternative_target_counts.append(float(len(candidate_subsets)))

                if not candidate_subsets:
                    chosen_polytomies.append(torch.tensor(0.0))
                else:
                    chosen_polytomies.append(torch.tensor(1.0))

                polytomy_logits.append(group["polytomy_pred"])

                size_info = None
                if decoder_mode == "structured_subset" and len(explicit_subsets) <= 1:
                    size_targets = (
                        [len(subset) for subset in candidate_subsets]
                        if candidate_subsets
                        else [0]
                    )
                    size_info = _structured_size_loss_and_prediction(
                        group.get("subset_size_logits"),
                        target_sizes=size_targets,
                        max_group_size=len(splits_in_polytomy),
                    )
                    if size_info is not None:
                        subset_size_losses.append(size_info["loss"].detach())
                        subset_size_target_means.append(
                            float(np.mean(size_info["target_sizes"]))
                        )
                        subset_size_prediction_means.append(
                            float(size_info["predicted_size"])
                        )
                        subset_size_accuracies.append(
                            1.0
                            if int(size_info["predicted_size"])
                            in {int(size) for size in size_info["target_sizes"]}
                            else 0.0
                        )

                if candidate_subsets:
                    candidate_losses = []
                    candidate_targets = []
                    candidate_pred_logits = []
                    for subset in candidate_subsets:
                        if decoder_mode == "structured_subset":
                            structured = _structured_subset_loss_and_prediction(
                                group,
                                splits_in_polytomy,
                                subset,
                                include_metric_logits=(
                                    not skip_autoregressive_merge_metrics
                                ),
                            )
                            if structured is None:
                                continue
                            candidate_losses.append(structured["loss"])
                            candidate_targets.append(structured["target_logits"])
                            candidate_pred_logits.append(structured["predicted_logits"])
                        else:
                            G = logits.size(0)
                            mask = ~torch.eye(
                                G, dtype=torch.bool, device=logits.device
                            )
                            tri = torch.triu(mask, diagonal=1)

                            y = _subset_target_matrix(
                                splits_in_polytomy,
                                subset,
                                logits.device,
                            )
                            y_vec = y[tri]
                            candidate_targets.append(y)

                            logits_vec = logits[tri]
                            finite = torch.isfinite(logits_vec)
                            logits_vec_f = logits_vec[finite]
                            y_vec_f = y_vec[finite]

                            pos = y_vec_f.sum().clamp(min=1.0)
                            neg = (y_vec_f.numel() - y_vec_f.sum()).clamp(min=1.0)
                            pos_weight = (neg / pos).detach()

                            candidate_losses.append(
                                F.binary_cross_entropy_with_logits(
                                    logits_vec_f,
                                    y_vec_f,
                                    pos_weight=pos_weight,
                                    reduction="mean",
                                )
                            )
                            candidate_pred_logits.append(logits)

                    if candidate_losses:
                        loss_stack = torch.stack(candidate_losses)
                        best_candidate_index = int(torch.argmin(loss_stack).item())
                        loss = loss_stack[best_candidate_index]
                        best_target = candidate_targets[best_candidate_index]
                        best_pred_logits = candidate_pred_logits[best_candidate_index]

                        if (
                            decoder_mode == "structured_subset"
                            and self.autoregressive_stop_after_merge_weight > 0.0
                            and "batched_autoregressive_stop_after_merge" in batch
                            and group.get("stop_after_merge_logit") is not None
                        ):
                            stop_target = batch[
                                "batched_autoregressive_stop_after_merge"
                            ][batch_index].to(group["stop_after_merge_logit"].device)
                            stop_loss = F.binary_cross_entropy_with_logits(
                                group["stop_after_merge_logit"].view(()),
                                stop_target.view(()),
                            )
                            loss = (
                                loss
                                + self.autoregressive_stop_after_merge_weight
                                * stop_loss
                            )
                            stop_after_merge_losses.append(stop_loss.detach())
                            stop_prob = torch.sigmoid(
                                group["stop_after_merge_logit"].detach()
                            )
                            stop_after_merge_targets.append(float(stop_target.item()))
                            stop_after_merge_predictions.append(float(stop_prob.item()))
                            stop_after_merge_accuracies.append(
                                1.0
                                if ((stop_prob > 0.5).float() == stop_target).item()
                                else 0.0
                            )

                        if best_pred_logits is not None and best_target is not None:
                            metrics = compute_merge_metrics(
                                best_pred_logits,
                                best_target,
                                threshold_logit=0.0,
                            )
                            total_metrics.append(metrics)

                        losses.append(loss)
                elif size_info is not None:
                    losses.append(size_info["loss"])

            loss_device = (
                all_group_logits[0]["logits"].device if all_group_logits else self.device
            )

            missing_explicit_targets = sum(
                1 for was_found in found.values() if not was_found
            )
            if missing_explicit_targets > 0:
                if self.autoregressive_target_mode == "ready_alternatives":
                    if self.verbose:
                        logger.info(
                            "Autoregressive explicit-target misses under ready-alternatives: %s",
                            missing_explicit_targets,
                        )
                    logs["autoregressive_stats/missing_explicit_targets"] = torch.tensor(
                        float(missing_explicit_targets),
                        device=loss_device,
                    )
                else:
                    for (batch_index, split_mask), was_found in found.items():
                        if not was_found:
                            print(
                                "Missing split: ",
                                [
                                    j
                                    for j in range(int(split_mask).bit_length())
                                    if (int(split_mask) >> j) & 1
                                ],
                            )
                            raise Exception(
                                f"Did not find merge for split {split_mask} in batch element {batch_index}!"
                            )

            L_polytomy_choosing = None

            if len(chosen_polytomies) > 1:
                polytomy_logits_tensor = torch.stack(polytomy_logits).squeeze(1)
                chosen_polytomies_tensor = torch.stack(chosen_polytomies).to(polytomy_logits_tensor.device)
                L_polytomy_choosing = F.binary_cross_entropy_with_logits(
                    polytomy_logits_tensor,
                    chosen_polytomies_tensor,
                ) 

                if self.training_step_verbose_logging_enabled:
                    logger.info(f"Polytomy choosing loss: {L_polytomy_choosing.item()}")
                if self.record:
                    self._wandb_log_filtered(
                        {
                            "train/polytomy_choosing_loss": L_polytomy_choosing.item(),
                            "train/polytomy_choosing_loss_weighted": (
                                self.autoregressive_polytomy_choosing_weight
                                * L_polytomy_choosing.item()
                            ),
                        },
                        step=self.stepper,
                    )

            if losses:
                L_merging = torch.stack(losses).mean()
            else:
                anchor_param = next(self.model.parameters())
                L_merging = anchor_param.sum() * 0.0
                if self.training_step_verbose_logging_enabled:
                    logger.info(
                        "Autoregressive loss skipped because no candidate merge targets were available."
                    )
                logs["autoregressive_stats/no_candidate_merge_loss"] = torch.tensor(
                    1.0,
                    device=loss_device,
                )
            logs["loss"] = _combine_autoregressive_losses(
                L_merging,
                L_polytomy_choosing,
                self.autoregressive_polytomy_choosing_weight,
            )
            if self.training_step_verbose_logging_enabled:
                logger.info(f"Autoregressive loss: {L_merging.item()}")

            aggregated_metrics = {}
            if len(total_metrics) > 0:
                for key in total_metrics[0]:
                    aggregated_metrics[key] = sum(
                        m[key] for m in total_metrics
                    ) / len(total_metrics)

                if self.training_step_verbose_logging_enabled:
                    for key in aggregated_metrics:
                        logger.info(f"{key}: {aggregated_metrics[key]}")

            if L_polytomy_choosing is not None:
                logs["autoregressive_stats/polytomy_choosing_weight"] = torch.tensor(
                    float(self.autoregressive_polytomy_choosing_weight),
                    device=loss_device,
                )

            if stop_after_merge_losses:
                logs["autoregressive_stats/stop_after_merge_loss"] = torch.stack(
                    stop_after_merge_losses
                ).mean().to(loss_device)
                logs["autoregressive_stats/stop_after_merge_accuracy"] = torch.tensor(
                    float(np.mean(stop_after_merge_accuracies)),
                    device=loss_device,
                )
                logs["autoregressive_stats/stop_after_merge_target_rate"] = torch.tensor(
                    float(np.mean(stop_after_merge_targets)),
                    device=loss_device,
                )
                logs["autoregressive_stats/stop_after_merge_pred_rate"] = torch.tensor(
                    float(np.mean(stop_after_merge_predictions)),
                    device=loss_device,
                )
            if subset_size_losses:
                logs["autoregressive_stats/subset_size_loss"] = torch.stack(
                    subset_size_losses
                ).mean().to(loss_device)
                logs["autoregressive_stats/subset_size_accuracy"] = torch.tensor(
                    float(np.mean(subset_size_accuracies)),
                    device=loss_device,
                )
                logs["autoregressive_stats/subset_size_target_mean"] = torch.tensor(
                    float(np.mean(subset_size_target_means)),
                    device=loss_device,
                )
                logs["autoregressive_stats/subset_size_pred_mean"] = torch.tensor(
                    float(np.mean(subset_size_prediction_means)),
                    device=loss_device,
                )

            # Calculate average polytomy size
            avg_polytomy_size = np.mean(polytomy_sizes) if polytomy_sizes else 0.0
            num_polytomies = len(polytomy_sizes)
            avg_alternative_targets = (
                float(np.mean(alternative_target_counts))
                if alternative_target_counts
                else 0.0
            )
            if self.training_step_verbose_logging_enabled:
                logger.info(f"Average polytomy size: {avg_polytomy_size}")
                logger.info(
                    f"Average alternative autoregressive targets: {avg_alternative_targets}"
                )
            logs["autoregressive_stats/avg_candidate_targets"] = torch.tensor(
                avg_alternative_targets,
                device=loss_device,
            )
            logs["autoregressive_stats/rollin_attempted"] = torch.tensor(
                ar_prep_stats["rollin_attempted"],
                device=loss_device,
            )
            logs["autoregressive_stats/rollin_applied"] = torch.tensor(
                ar_prep_stats["rollin_applied"],
                device=loss_device,
            )
            logs["autoregressive_stats/dagger_attempted"] = torch.tensor(
                ar_prep_stats["dagger_attempted"],
                device=loss_device,
            )
            logs["autoregressive_stats/dagger_applied"] = torch.tensor(
                ar_prep_stats["dagger_applied"],
                device=loss_device,
            )
            dagger_avg_steps = (
                ar_prep_stats["dagger_rollout_steps"] / ar_prep_stats["dagger_applied"]
                if ar_prep_stats["dagger_applied"] > 0.0
                else 0.0
            )
            logs["autoregressive_stats/dagger_avg_rollout_steps"] = torch.tensor(
                dagger_avg_steps,
                device=loss_device,
            )
            logs["autoregressive_stats/structure_perturb_attempted"] = torch.tensor(
                ar_prep_stats["structure_perturb_attempted"],
                device=loss_device,
            )
            logs["autoregressive_stats/structure_perturb_applied"] = torch.tensor(
                ar_prep_stats["structure_perturb_applied"],
                device=loss_device,
            )

            if self.record and not is_replay_batch:
                # Batch all metrics into a single wandb.log call to avoid step conflicts
                wandb_metrics = {
                    "train/autoregressive_loss": L_merging.item(),
                    "autoregressive_stats/avg_polytomy_size": avg_polytomy_size,
                    "autoregressive_stats/num_polytomies": num_polytomies,
                    "autoregressive_stats/avg_candidate_targets": avg_alternative_targets,
                    "autoregressive_stats/rollin_attempted": ar_prep_stats["rollin_attempted"],
                    "autoregressive_stats/rollin_applied": ar_prep_stats["rollin_applied"],
                    "autoregressive_stats/dagger_attempted": ar_prep_stats["dagger_attempted"],
                    "autoregressive_stats/dagger_applied": ar_prep_stats["dagger_applied"],
                    "autoregressive_stats/dagger_avg_rollout_steps": dagger_avg_steps,
                    "autoregressive_stats/structure_perturb_attempted": ar_prep_stats["structure_perturb_attempted"],
                    "autoregressive_stats/structure_perturb_applied": ar_prep_stats["structure_perturb_applied"],
                }
                wandb_metrics.update(
                    {f"{key}": aggregated_metrics[key] for key in aggregated_metrics}
                )
                self._wandb_log_filtered(wandb_metrics, step=self.stepper)

        return logs

    def sample(
        self,
        newick_starting_trees: list[str],
        phyla_embeddings,
        case_indices=None,
        num_samples=None,
        mapping=None,
        T=1.0,
        dt_base=0.02,
        eps_len=1e-8,
        hit_tol=1e-10,
        first_hit_tol=1e-4,
        autoregressive_birth_length=1e-3,
        stop_on_no_valid_merge=False,
        max_events=1000,
        max_steps=20000,
        topology_repeat_cap=0,
        KNN_TOPM = 32,
        KNN_TAU = 0.05,
        KNN_STOCHASTIC = False,
        debug_real_tree=None,
        return_trace: bool = False,
        target_trees: list[str] | None = None,
        first_hit_start_topology_features=None,
        autoregressive_start_topology_features=None,
        first_hit_start_topology_embeddings=None,
        first_hit_start_topology_pad_mask=None,
        first_hit_start_tree_graph_context=None,
        split_multi_label_events: bool = False,
        max_allowed_polytomy_size: int = -1,
        oversize_polytomy_policy: str = "none",
        oversize_polytomy_blacklist_revisits: bool = False,
        oversize_polytomy_min_dt_escape: float = 0.0,
        fixed_dt_sampling: bool = False,
        max_autoregressive_merges_per_boundary: int = -1,
        prefix_replay_velocity_quota: int = 0,
        prefix_replay_autoregressive_quota: int = 0,
        prefix_replay_split_multi_label_events: bool = False,
        oracle_first_hit_use_at_sampling: bool = False,
        oracle_gate_first_hit_use_at_sampling: bool = False,
        oracle_boundary_vanish_use_at_sampling: bool = False,
        trace_state_rf: bool = True,
        explicit_autoregressive_component_groups: bool = True,
    ):
        if self.use_historical_sampling_impl:
            return _call_historical_trainingmodule_method(
                "sample",
                self,
                newick_starting_trees,
                phyla_embeddings,
                case_indices=case_indices,
                num_samples=num_samples,
                mapping=mapping,
                T=T,
                dt_base=dt_base,
                eps_len=eps_len,
                hit_tol=hit_tol,
                first_hit_tol=first_hit_tol,
                autoregressive_birth_length=autoregressive_birth_length,
                stop_on_no_valid_merge=stop_on_no_valid_merge,
                max_events=max_events,
                max_steps=max_steps,
                topology_repeat_cap=topology_repeat_cap,
                KNN_TOPM=KNN_TOPM,
                KNN_TAU=KNN_TAU,
                KNN_STOCHASTIC=KNN_STOCHASTIC,
                debug_real_tree=debug_real_tree,
                return_trace=return_trace,
                target_trees=target_trees,
                split_multi_label_events=split_multi_label_events,
                max_allowed_polytomy_size=max_allowed_polytomy_size,
                oversize_polytomy_policy=oversize_polytomy_policy,
                oversize_polytomy_blacklist_revisits=oversize_polytomy_blacklist_revisits,
                oversize_polytomy_min_dt_escape=oversize_polytomy_min_dt_escape,
                fixed_dt_sampling=fixed_dt_sampling,
                max_autoregressive_merges_per_boundary=max_autoregressive_merges_per_boundary,
                prefix_replay_velocity_quota=prefix_replay_velocity_quota,
                prefix_replay_autoregressive_quota=prefix_replay_autoregressive_quota,
                prefix_replay_split_multi_label_events=prefix_replay_split_multi_label_events,
                oracle_first_hit_use_at_sampling=oracle_first_hit_use_at_sampling,
                oracle_gate_first_hit_use_at_sampling=oracle_gate_first_hit_use_at_sampling,
                oracle_boundary_vanish_use_at_sampling=oracle_boundary_vanish_use_at_sampling,
                trace_state_rf=trace_state_rf,
                explicit_autoregressive_component_groups=(
                    explicit_autoregressive_component_groups
                ),
            )
        shared_start_topology_features = first_hit_start_topology_features
        if shared_start_topology_features is None:
            shared_start_topology_features = autoregressive_start_topology_features
        if (
            self.sampling_discrete_phase_rollout_use_at_sampling
            and len(newick_starting_trees) > 1
            and target_trees is not None
            and len(target_trees) == len(newick_starting_trees)
            and self.topology_decoder == "birthset"
            and self.birthset_fallback == "none"
            and not self.velocity_terminal_head_use_at_sampling
        ):
            out = _discrete_phase_rollout_batched_birthset(
                self,
                newick_starting_trees,
                target_trees,
                phyla_embeddings,
                case_indices=case_indices,
                start_topology_features=shared_start_topology_features,
                start_topology_embeddings=first_hit_start_topology_embeddings,
                start_topology_pad_mask=first_hit_start_topology_pad_mask,
                dt_base=float(dt_base),
                eps_len=float(eps_len),
                max_events=max_events,
                max_steps=max_steps,
                max_phases=int(
                    getattr(self, "sampling_discrete_phase_max_phases", 8)
                ),
                return_trace=return_trace,
                trace_state_rf=bool(trace_state_rf),
                explicit_autoregressive_component_groups=bool(
                    explicit_autoregressive_component_groups
                ),
            )
            result = (
                out["final_trees"],
                int(out["num_ar_states"]),
                0.0,
                0.0,
                int(out["num_ar_states"]),
            )
            if return_trace:
                return result + (out["trace"],)
            return result
        if (
            self.sampling_discrete_phase_rollout_use_at_sampling
            and len(newick_starting_trees) == 1
            and target_trees is not None
            and len(target_trees) == 1
        ):
            out = _discrete_phase_rollout(
                self,
                newick_starting_trees[0],
                target_trees[0],
                phyla_embeddings,
                case_index=None
                if case_indices is None
                else int(torch.as_tensor(case_indices).reshape(-1)[0].item()),
                start_topology_features=shared_start_topology_features,
                start_topology_embeddings=first_hit_start_topology_embeddings,
                start_topology_pad_mask=first_hit_start_topology_pad_mask,
                dt_base=float(dt_base),
                eps_len=float(eps_len),
                autoregressive_birth_length=float(autoregressive_birth_length),
                max_events=max_events,
                max_steps=max_steps,
                max_phases=int(
                    getattr(self, "sampling_discrete_phase_max_phases", 8)
                ),
                return_trace=return_trace,
                trace_state_rf=bool(trace_state_rf),
                explicit_autoregressive_component_groups=bool(
                    explicit_autoregressive_component_groups
                ),
            )
            result = (
                [out["final_tree"]],
                int(out["num_ar_states"]),
                0.0,
                0.0,
                int(out["num_ar_states"]),
            )
            if return_trace:
                return result + (out["trace"],)
            return result
        if (
            self.sampling_predsim_overrun_use_at_sampling
            and len(newick_starting_trees) == 1
            and target_trees is not None
            and len(target_trees) == 1
        ):
            effective_max_steps = (
                1000000 if max_steps is None or int(max_steps) < 0 else int(max_steps)
            )
            effective_max_events = (
                1000000
                if max_events is None or int(max_events) < 0
                else int(max_events)
            )
            out = _predsim_overrun_rollout(
                self,
                newick_starting_trees[0],
                target_trees[0],
                phyla_embeddings,
                case_index=None
                if case_indices is None
                else int(torch.as_tensor(case_indices).reshape(-1)[0].item()),
                start_topology_features=shared_start_topology_features,
                T=float(T),
                eps_len=float(eps_len),
                first_hit_tol=float(first_hit_tol),
                max_steps=effective_max_steps,
                max_events=effective_max_events,
                max_autoregressive_merges_per_boundary=int(
                    max_autoregressive_merges_per_boundary
                ),
                raw_no_fallback=True,
                autoregressive_birth_length=float(autoregressive_birth_length),
                boundary_mode=(
                    "pred_simultaneous_time_head"
                    if getattr(self, "velocity_boundary_time_head_use_at_sampling", False)
                    else getattr(
                        self,
                        "sampling_predsim_boundary_mode",
                        "pred_simultaneous",
                    )
                ),
                allow_time_overrun=bool(
                    getattr(self, "sampling_predsim_allow_time_overrun", True)
                ),
                blocked_edge_floor=getattr(
                    self, "sampling_blocked_edge_floor", None
                ),
            )
            result = (
                [out["final_tree"]],
                int(out["num_ar_states"]),
                0.0,
                0.0,
                int(out["num_ar_states"]),
            )
            if return_trace:
                return result + (
                    _predsim_overrun_trace_to_sampling_trace(out, target_trees[0]),
                )
            return result
        if num_samples is None:
            num_samples = self.num_samples

        case_index_tensor = None
        if case_indices is not None:
            case_index_tensor = torch.as_tensor(
                case_indices,
                dtype=torch.long,
                device=self.device,
            ).reshape(-1)
            if int(case_index_tensor.shape[0]) != int(len(newick_starting_trees)):
                raise ValueError(
                    "case_indices must have one entry per starting tree in sample()."
                )
        start_topology_feature_tensor = shared_start_topology_features
        if (
            start_topology_feature_tensor is None
            and (
                getattr(self.model, "first_hit_head_mode", "base")
                in {
                    "start_topology_adapter_mlp",
                    "start_topology_raw_pool_concat_mlp",
                }
                or getattr(
                    self.model,
                    "autoregressive_use_start_topology_conditioning",
                    False,
                )
            )
        ):
            start_topology_feature_tensor = _build_start_topology_feature_tensor(
                self,
                newick_starting_trees,
                device=self.device,
            )
        start_topology_embeddings_tensor = first_hit_start_topology_embeddings
        start_topology_pad_mask_tensor = first_hit_start_topology_pad_mask
        if (
            (
                start_topology_embeddings_tensor is None
                or start_topology_pad_mask_tensor is None
            )
            and getattr(self.model, "first_hit_head_mode", "base")
            == "start_topology_cross_attn_mlp"
        ):
            (
                start_topology_embeddings_tensor,
                start_topology_pad_mask_tensor,
            ) = _build_start_topology_identity_batch(
                self,
                newick_starting_trees,
                device=self.device,
            )
        start_tree_graph_context_tensor = first_hit_start_tree_graph_context
        if (
            start_tree_graph_context_tensor is None
            and getattr(self.model, "first_hit_head_mode", "base")
            == "start_tree_graph_token_mlp"
        ):
            start_tree_graph_context_tensor = _build_start_tree_graph_context(
                self,
                newick_starting_trees,
                phyla_embeddings,
                device=self.device,
                detach=getattr(self.model, "first_hit_start_tree_graph_detach", False),
            )

        self.model.eval()
        max_logits = []
        trace = None
        if return_trace:
            trace = {
                "velocity": [],
                "autoregressive": [],
                "stopped_for_no_valid_merge": False,
                "stopped_for_repeated_topology": False,
                "skipped_no_valid_boundary_revisits": 0.0,
                "stopped_for_prefix_replay_quota": False,
                "silent_boundary_recoveries": 0.0,
            }
            if target_trees is None:
                target_trees = [None] * len(newick_starting_trees)
        prefix_replay_velocity_quota = max(int(prefix_replay_velocity_quota), 0)
        prefix_replay_autoregressive_quota = max(
            int(prefix_replay_autoregressive_quota), 0
        )
        prefix_replay_valid_velocity_counts = [0] * len(newick_starting_trees)
        prefix_replay_valid_autoregressive_counts = [0] * len(newick_starting_trees)
        prefix_replay_stop_requested = False
        sampling_disable_inner_logging = bool(
            getattr(self, "sampling_disable_inner_logging", False)
        )
        sampling_cache_autoregressive_state = bool(
            getattr(self, "sampling_cache_autoregressive_state", False)
        )
        sampling_use_top_merge_planner = bool(
            getattr(self, "sampling_use_top_merge_planner", False)
        )
        sampling_use_inference_mode = bool(
            getattr(self, "sampling_use_inference_mode", False)
        )
        sampling_actual_event_boundary_use_at_sampling = bool(
            getattr(self, "sampling_actual_event_boundary_use_at_sampling", False)
        )
        sampling_actual_event_boundary_include_predicted_first_hit = bool(
            getattr(
                self,
                "sampling_actual_event_boundary_include_predicted_first_hit",
                False,
            )
        )
        sampling_cache_tri_mask = bool(
            getattr(self, "sampling_cache_tri_mask", False)
        )
        sampling_cache_polytomy_groups = bool(
            getattr(self, "sampling_cache_polytomy_groups", False)
        )
        ar_single_tree_cache = {}
        tri_mask_cache = {}
        polytomy_group_cache = {}
        birthset_polytomy_unrooted_ok = self.topology_decoder in {
            "birthset",
            "birthset_with_ar_fallback",
        }

        def _sampling_log_info(message):
            if not sampling_disable_inner_logging:
                logger.info(message)

        def _get_tri_mask(size, device):
            if not sampling_cache_tri_mask:
                return torch.triu(
                    ~torch.eye(size, dtype=torch.bool, device=device),
                    diagonal=1,
                )
            key = (int(size), str(device))
            tri = tri_mask_cache.get(key)
            if tri is None:
                tri = torch.triu(
                    ~torch.eye(size, dtype=torch.bool, device=device),
                    diagonal=1,
                )
                tri_mask_cache[key] = tri
            return tri

        def _get_polytomy_group_artifacts(newick_value):
            if not sampling_cache_polytomy_groups:
                return None, has_polytomy_fast(
                    newick_value,
                    unrooted_ok=birthset_polytomy_unrooted_ok,
                )
            cached = polytomy_group_cache.get(newick_value)
            if cached is None:
                groups = get_structural_polytomy_groups_from_newick(newick_value)
                cached = {
                    "groups": groups,
                    "has_polytomy": has_polytomy_fast(
                        newick_value,
                        unrooted_ok=birthset_polytomy_unrooted_ok,
                    ),
                }
                polytomy_group_cache[newick_value] = cached
            return cached["groups"], cached["has_polytomy"]

        def _get_ar_single_tree_artifacts(td_state, n_leaves_state, mapping_state):
            graph_local, newick_local = build_tree_from_splits(
                list(td_state.keys()),
                td_state,
                n_leaves_state,
                root_leaf=n_leaves_state - 1,
                mapping=mapping_state,
            )
            if not sampling_cache_autoregressive_state:
                return graph_local, newick_local, None, None, None

            cached = ar_single_tree_cache.get(newick_local)
            if cached is None:
                cached = {
                    "tokenized_trees": self.model.tokenizer([newick_local]),
                    "component_groups": [
                        get_structural_polytomy_groups_from_newick(newick_local)
                    ],
                    "structural_cache_item": self.model.tokenizer.compute_structural_cache(
                        [newick_local]
                    )[0],
                }
                ar_single_tree_cache[newick_local] = cached
            return (
                graph_local,
                newick_local,
                cached["tokenized_trees"],
                cached["component_groups"],
                cached["structural_cache_item"],
            )

        def _slice_single_tree_conditioning(value, tree_index):
            if value is None:
                return None
            if torch.is_tensor(value):
                if value.ndim >= 1 and int(value.shape[0]) == len(newick_starting_trees):
                    return value[int(tree_index) : int(tree_index) + 1]
                return value
            if isinstance(value, (list, tuple)) and len(value) == len(newick_starting_trees):
                return [value[int(tree_index)]]
            return value

        def _prefix_replay_quota_satisfied():
            if (
                prefix_replay_velocity_quota <= 0
                and prefix_replay_autoregressive_quota <= 0
            ):
                return False
            for idx in range(len(newick_starting_trees)):
                if (
                    prefix_replay_velocity_quota > 0
                    and prefix_replay_valid_velocity_counts[idx]
                    < prefix_replay_velocity_quota
                ):
                    return False
                if (
                    prefix_replay_autoregressive_quota > 0
                    and prefix_replay_valid_autoregressive_counts[idx]
                    < prefix_replay_autoregressive_quota
                ):
                    return False
            return True

        if (
            phyla_embeddings is None
            and self.phyla_precomputed_name_to_embedding is not None
        ):
            batch_embeddings = []
            for tree_idx, tree_newick in enumerate(newick_starting_trees):
                tree_mapping = None
                if isinstance(mapping, list):
                    tree_mapping = mapping[tree_idx]
                elif isinstance(mapping, dict):
                    tree_mapping = mapping
                resolved = self._resolve_precomputed_phyla_embeddings_for_tree(
                    tree_newick,
                    mapping=tree_mapping,
                    device=self.device,
                )
                if resolved is None:
                    batch_embeddings = []
                    break
                batch_embeddings.append(resolved.squeeze(0))
            if batch_embeddings:
                phyla_embeddings = torch.stack(batch_embeddings, dim=0)

        if (
            phyla_embeddings is None
            and self.phyla_model is not None
            and self.dataset is not None
        ):
            # Calculate embeddings on the fly
            t_temp = Tree(newick_starting_trees[0])
            sorted_names = [t_temp.id_to_name[i] for i in range(t_temp.n_leaves)]

            # Filter out ROOT_DUMMY as it has no sequence
            valid_names = [n for n in sorted_names if n != "ROOT_DUMMY"]
            sorted_seqs = [self.dataset.name_to_seq[name] for name in valid_names]

            raw_emb = self.compute_phyla_embeddings(
                sorted_seqs, valid_names, device=self.device
            )
            # raw_emb is (1, N, D). We want (B, N, D).
            if raw_emb.size(0) == 1:
                phyla_embeddings = raw_emb.expand(len(newick_starting_trees), -1, -1)
            else:
                phyla_embeddings = raw_emb.expand(len(newick_starting_trees), -1, -1)

        # SPEED UP SAMPLING
        # 1) init: parse tree -> {mask: length}
        trees = []
        num_leaves = []
        mapping = []
        # Precompute cache for initial trees
        # Since topology changes in the loop, we will update this cache dynamically
        # Initialize tokenized structure cache
        current_newicks = list(newick_starting_trees)
        token_cache = self.model.tokenizer.create_batched_cache(current_newicks)
        #tokenized = self.dataset.tree_tokenizer(current_newicks[0])
        # new_tokenized = ()
        # for i in tokenized:
        #     if torch.is_tensor(i):
        #         new_tokenized += (i.to(self.device),)
        #     else:
        #         new_tokenized += (i,)


        for b_idx, nw in enumerate(newick_starting_trees):
            t = Tree(nw)
            enc = BHVEncoder()
            masks, lens = enc.return_BHV_encoding(t)
            # BHV encoder uses canonical split orientation (with dummy-root influence),
            # while model/tokenizer uses directed edge masks on dummy-free Newick.
            # Convert initial lengths into tokenizer split-mask space once, so the
            # sampler state remains in one consistent representation.
            bhv_lengths = {int(m): float(l) for m, l in zip(masks, lens) if l is not None}
            model_masks_init = [int(m) for m in token_cache.edge_split_masks_list[b_idx] if int(m) != 0]

            biological_bits = max(t.n_leaves - 1, 0)
            full_model_mask = (1 << biological_bits) - 1 if biological_bits > 0 else 0

            td_init = {}
            for m_model in model_masks_init:
                length = bhv_lengths.get(m_model)
                if length is None and full_model_mask:
                    length = bhv_lengths.get(full_model_mask ^ m_model)

                if length is None:
                    raise Exception(
                        f"Could not map initial split {m_model} from BHV encoding to tokenizer mask space."
                    )
                # Keep sampler state aligned with active edges only; zero-length edges
                # are represented as absent and do not participate in dynamics.
                if float(length) > eps_len:
                    td_init[m_model] = float(length)

            trees.append(td_init)
            num_leaves.append(t.n_leaves)
            mapping.append(t.id_to_name)

        t = 0.0
        n_events = 0
        n_events_by_tree = [0] * len(newick_starting_trees)
        n_steps = 0
        n_topology_changes = 0
        num_topology_changes = 0
        num_silent_boundary_recoveries = 0
        polytomy_sizes = []  # Track sizes of polytomies encountered during sampling
        boundary_topology_counts = [dict() for _ in newick_starting_trees]
        no_valid_boundary_topologies = [set() for _ in newick_starting_trees]
        oversize_boundary_topologies = [set() for _ in newick_starting_trees]
        stop_for_repeated_topology = False
        stop_for_no_valid_merge = False
        def _event_budget_remaining():
            if max_events is None:
                return True
            return any(int(count) < int(max_events) for count in n_events_by_tree)

        while (
            t < T
            and (max_steps is None or n_steps < max_steps)
            and _event_budget_remaining()
        ):
            n_steps += 1

            # --- encode/tokenize current trees for the model ---

            # Use CACHED tokenizer
            tokenized = self.model.tokenizer.forward_batched(token_cache, trees)
            #import pdb; pdb.set_trace()

            # if calculate_norm_rf(current_newicks[0], self.train_tree[0]) != 0:
            #     raise Exception("Current tree does not match training tree topology!")
            # #import pdb; pdb.set_trace()
            # if tokenized[0].shape[1] != self.train_tokenized_trees[0].shape[1]:
            #     raise Exception("Tokenized tree length mismatch!")
            # elif (new_tokenized[0] == self.train_tokenized_trees[0]).all().item() is False:
            #     raise Exception("Tokenized trees do not match!")
            
 
            sampling_context = (
                torch.inference_mode()
                if sampling_use_inference_mode
                else torch.no_grad()
            )
            with sampling_context:
                (
                    velocity,
                    edge_splits,
                    edge_split_mask,
                    first_hit_logits,
                    boundary_vanish_logits,
                    edge_features,
                ) = self.forward(
                    tokenized,
                    t,
                    phyla_embeddings,
                    first_hit_case_indices=case_index_tensor,
                    first_hit_start_topology_features=start_topology_feature_tensor,
                    first_hit_start_topology_embeddings=start_topology_embeddings_tensor,
                    first_hit_start_topology_pad_mask=start_topology_pad_mask_tensor,
                    first_hit_start_tree_graph_context=start_tree_graph_context_tensor,
                )

            # ---- FIRST PASS: compute per-tree dt_hit, cache per-tree arrays ----

            dt_hit_list = []
            actual_dt_hit_list = []
            cache = []
            for b_idx, (td, v, n_leaves, mapp) in enumerate(zip(trees, velocity, num_leaves, mapping)):
                model_masks = edge_splits[b_idx]
                mask_idx = {mask: i for i, mask in enumerate(model_masks)}
                # Use biological leaf universe (exclude dummy root leaf) for canonicalization.
                # Deriving bit-width from observed model masks can undercount when a high-index
                # leaf is absent in the current split set, which breaks complement matching.
                biological_bits = max(n_leaves - 1, 0)
                full_model_mask = (1 << biological_bits) - 1 if biological_bits > 0 else 0
                # In Tree(...) representation we carry a dummy leaf at index n_leaves-1.
                # The edge incident to the dummy can appear as "all biological leaves" split;
                # tokenizer masks (built on dummy-free Newick) do not include this split.
                dummy_artifact_mask = full_model_mask
                V_model = v.squeeze(1).detach().cpu().numpy()
                H_model = None
                if first_hit_logits is not None:
                    H_model = first_hit_logits[b_idx].squeeze(1).detach().cpu().numpy()
                B_model = None
                if boundary_vanish_logits is not None:
                    B_model = (
                        boundary_vanish_logits[b_idx].squeeze(1).detach().cpu().numpy()
                    )
                E_model = None
                if edge_features is not None:
                    E_model = edge_features[b_idx].detach().cpu()

                L = []
                V_val = []
                H_val = []
                B_val = []
                E_val = []
                masks = []
                supervised_edge_flags = []
                aligned_model_masks = []
                for m in td:
                    if m == dummy_artifact_mask:
                        continue

                    matched_m = m
                    dummy_bit_idx = n_leaves - 1
                    if biological_bits > 0 and ((m >> dummy_bit_idx) & 1):
                        matched_m = remove_bit(m, dummy_bit_idx)

                    if biological_bits > 0 and matched_m.bit_length() > biological_bits:
                        print(
                            f"Skipping split with unexpected bit_length: {m} "
                            f"(bit_length={m.bit_length()}, expected <= {biological_bits + 1})"
                        )
                        raise Exception("Unexpected split in tree while sampling that cannot be matched to model masks!")

                    idx = mask_idx.get(matched_m)
                    if idx is None and full_model_mask:
                        complement_m = full_model_mask ^ matched_m
                        idx = mask_idx.get(complement_m)
                        if idx is not None:
                            matched_m = complement_m

                    if idx is None:
                        print(
                            f"Whoa there is a split missing in velocity masks! {m} or "
                            f"{[i for i in range(m.bit_length()) if (m >> i) & 1]}"
                        )
                        raise Exception("Missing split in velocity masks!")

                    curr_len = float(td[m])
                    if curr_len <= eps_len:
                        continue
                    L.append(curr_len)

                    #We should not be making moves based on leafs! If leaf, velocity is 0
                    k_bits = int(matched_m).bit_count()
                    is_pendant = biological_bits > 0 and min(
                        k_bits, biological_bits - k_bits
                    ) == 1
                    if is_pendant:
                        V_val.append(0.0)
                        if H_model is not None:
                            # Historical sampler behavior excluded pendant edges from
                            # first-hit competition entirely. Keeping their learned
                            # logits here lets pendant edges spuriously win boundary
                            # selection, which regressed the replayfast vanish/smallstep
                            # random-start line.
                            H_val.append(float("-inf"))
                        if B_model is not None:
                            B_val.append(float("-inf"))
                        supervised_edge_flags.append(False)
                    else:
                        V_val.append(V_model[idx])
                        if H_model is not None:
                            H_val.append(float(H_model[idx]))
                        if B_model is not None:
                            B_val.append(float(B_model[idx]))
                        if E_model is not None:
                            E_val.append(E_model[idx])
                        supervised_edge_flags.append(True)
                    if E_model is not None and len(E_val) < len(L):
                        E_val.append(E_model[idx])

                    masks.append(m)
                    aligned_model_masks.append(int(matched_m))


                V = np.array(V_val, dtype=np.float64)
                L = np.array(L, dtype=np.float64)
                H = (
                    np.array(H_val, dtype=np.float64)
                    if H_model is not None
                    else None
                )
                E = None
                if E_val:
                    E = torch.stack(E_val, dim=0)
                supervised_mask = np.array(supervised_edge_flags, dtype=bool)
                if self.velocity_refiner_mode == "edge_token_attention_delta" and E is not None:
                    refine_mask = supervised_mask & (L > eps_len)
                    if bool(np.any(refine_mask)):
                        refine_indices = np.nonzero(refine_mask)[0]
                        V_refined = self._refine_velocity_predictions(
                            torch.from_numpy(V[refine_mask]).to(
                                self.device, dtype=torch.float32
                            ),
                            lengths=torch.from_numpy(L[refine_mask]).to(
                                self.device, dtype=torch.float32
                            ),
                            edge_features=E.index_select(
                                0,
                                torch.as_tensor(
                                    refine_indices,
                                    dtype=torch.long,
                                    device=E.device,
                                ),
                            ).to(self.device, dtype=torch.float32),
                            group_sizes=[int(len(refine_indices))],
                        )
                        V = V.copy()
                        V[refine_mask] = V_refined.detach().cpu().numpy()
                if H is not None or E is not None:
                    H_tensor = (
                        torch.from_numpy(H).to(self.device, dtype=torch.float32)
                        if H is not None
                        else None
                    )
                    H = (
                        self._compute_first_hit_logits(
                            H_tensor,
                            lengths=torch.from_numpy(L).to(
                                self.device, dtype=torch.float32
                            ),
                            velocities=torch.from_numpy(V).to(
                                self.device, dtype=torch.float32
                            ),
                            edge_features=(
                                E.to(self.device, dtype=torch.float32)
                                if E is not None
                                else None
                            ),
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                B = (
                    np.array(B_val, dtype=np.float64)
                    if B_model is not None
                    else None
                )
                
                if len(V) != len(L):
                    raise Exception("I assume these two things are equal length!")
                if len(supervised_mask) != len(L):
                    raise Exception("Supervised-mask and edge-length arrays must align!")

                if (L < 0).any():
                    raise Exception("There are negative lengths that is not possible!")

                # --- DEBUG: compare predicted vs true velocity at t=0 ---
                if (
                    debug_real_tree is not None
                    and n_steps == 1
                    and not sampling_disable_inner_logging
                ):
                    try:
                        _, true_velocity = return_sampled_tree_orthant_velocity(
                            newick_starting_trees[b_idx], debug_real_tree, 0.0
                        )
                        # Match exactly the supervised subset used during training:
                        # remove dummy bit if present, allow complement orientation,
                        # and drop pendant edges.
                        v_pred_arr = []
                        v_true_arr = []
                        matched_masks_dbg = []
                        true_vel_by_model_mask = {}
                        skipped_pendant = 0
                        unmatched = 0
                        for vel_mask, tv in true_velocity.items():
                            vel = int(vel_mask)
                            if biological_bits > 0 and vel.bit_length() == biological_bits + 1:
                                vel = remove_bit(vel, n_leaves - 1)
                            elif biological_bits > 0 and vel.bit_length() > biological_bits + 1:
                                unmatched += 1
                                continue

                            matched_vel = vel
                            idx = mask_idx.get(matched_vel)
                            if idx is None and full_model_mask:
                                complement_vel = full_model_mask ^ matched_vel
                                idx = mask_idx.get(complement_vel)
                                if idx is not None:
                                    matched_vel = complement_vel

                            if idx is None:
                                unmatched += 1
                                continue

                            k_bits = int(matched_vel).bit_count()
                            is_pendant = biological_bits > 0 and min(
                                k_bits, biological_bits - k_bits
                            ) == 1
                            if is_pendant:
                                skipped_pendant += 1
                                continue

                            v_pred_arr.append(float(V_model[idx]))
                            v_true_arr.append(float(tv))
                            matched_masks_dbg.append(int(matched_vel))
                            true_vel_by_model_mask[int(matched_vel)] = float(tv)

                        if len(v_pred_arr) > 0:
                            v_pred_np = np.array(v_pred_arr)
                            v_true_np = np.array(v_true_arr)
                            mse = float(np.mean((v_pred_np - v_true_np) ** 2))
                            mae = float(np.mean(np.abs(v_pred_np - v_true_np)))
                            cos_num = np.dot(v_pred_np, v_true_np)
                            cos_den = (np.linalg.norm(v_pred_np) * np.linalg.norm(v_true_np))
                            cosine_sim = float(cos_num / cos_den) if cos_den > 0 else 0.0
                            print(f"\n===== DEBUG: Predicted vs True velocity at t=0 (tree {b_idx}) =====")
                            print(
                                f"  Matched supervised internal edges: {len(v_pred_arr)} "
                                f"(of {len(true_velocity)} true; skipped pendant={skipped_pendant}, unmatched={unmatched})"
                            )
                            print(f"  MSE:  {mse:.6e}")
                            print(f"  MAE:  {mae:.6e}")
                            print(f"  Cosine similarity: {cosine_sim:.6f}")
                            print(f"  Pred  range: [{v_pred_np.min():.6e}, {v_pred_np.max():.6e}]")
                            print(f"  True  range: [{v_true_np.min():.6e}, {v_true_np.max():.6e}]")

                            # Compare dt_hit on matched supervised edges using the same lengths.
                            matched_idx = [
                                i
                                for i, mm in enumerate(aligned_model_masks)
                                if supervised_mask[i] and (mm in true_vel_by_model_mask)
                            ]
                            if len(matched_idx) > 0:
                                L_match = L[matched_idx]
                                v_pred_match = V[matched_idx]
                                v_true_match = np.array(
                                    [true_vel_by_model_mask[aligned_model_masks[i]] for i in matched_idx],
                                    dtype=np.float64,
                                )

                                pred_neg_match = (v_pred_match < 0.0) & (L_match > eps_len)
                                true_neg_match = (v_true_match < 0.0) & (L_match > eps_len)

                                pred_dt_candidates = (
                                    L_match[pred_neg_match] / -v_pred_match[pred_neg_match]
                                    if np.any(pred_neg_match)
                                    else np.array([], dtype=np.float64)
                                )
                                true_dt_candidates = (
                                    L_match[true_neg_match] / -v_true_match[true_neg_match]
                                    if np.any(true_neg_match)
                                    else np.array([], dtype=np.float64)
                                )

                                dt_hit_pred_dbg = (
                                    float(np.min(pred_dt_candidates))
                                    if pred_dt_candidates.size > 0
                                    else float("inf")
                                )
                                dt_hit_true_dbg = (
                                    float(np.min(true_dt_candidates))
                                    if true_dt_candidates.size > 0
                                    else float("inf")
                                )

                                print(
                                    f"  dt_hit(pred, matched supervised): {dt_hit_pred_dbg:.6e} "
                                    f"(neg={int(pred_neg_match.sum())}, candidates={pred_dt_candidates.size})"
                                )
                                print(
                                    f"  dt_hit(true, matched supervised): {dt_hit_true_dbg:.6e} "
                                    f"(neg={int(true_neg_match.sum())}, candidates={true_dt_candidates.size})"
                                )
                                if pred_dt_candidates.size > 0:
                                    print(
                                        f"  Pred dt candidates (min-5): {np.sort(pred_dt_candidates)[:5]}"
                                    )
                                if true_dt_candidates.size > 0:
                                    print(
                                        f"  True dt candidates (min-5): {np.sort(true_dt_candidates)[:5]}"
                                    )
                            else:
                                print(
                                    "  dt_hit debug: no matched supervised edges with both length and true velocity."
                                )
                            # Show top-5 worst mismatches
                            abs_err = np.abs(v_pred_np - v_true_np)
                            worst_idx = np.argsort(abs_err)[::-1][:5]
                            print(f"  Top-5 worst mismatches:")
                            for wi in worst_idx:
                                print(f"    split={matched_masks_dbg[wi]:>12}  pred={v_pred_np[wi]:+.6e}  true={v_true_np[wi]:+.6e}  err={abs_err[wi]:.6e}")
                            print(f"============================================================\n")
                            # import pdb; pdb.set_trace()
                        else:
                            print(f"DEBUG: Could not match any velocity splits for tree {b_idx}")
                    except Exception as e:
                        print(f"DEBUG: Failed to compute true velocity for comparison: {e}")

                # --- compute dt_hit ---
                predicted_first_mask = None
                predicted_vanish_mask = None
                actual_hit_mask = np.zeros_like(supervised_mask, dtype=bool)
                actual_dt_hit = float("inf")
                use_boundary_vanish_one_step = (
                    self.velocity_boundary_vanish_one_step_use_at_sampling
                    and predicted_vanish_mask is None
                )
                current_boundary_state_key = tuple(sorted(int(mask) for mask in td.keys()))
                boundary_blacklisted = bool(
                    oversize_polytomy_blacklist_revisits
                    and current_boundary_state_key in oversize_boundary_topologies[b_idx]
                )
                if (
                    oracle_boundary_vanish_use_at_sampling
                    and target_trees is not None
                ):
                    candidate_mask = supervised_mask & (L > eps_len)
                    try:
                        _, current_newick_oracle = build_tree_from_splits(
                            list(td.keys()),
                            td,
                            n_leaves,
                            root_leaf=n_leaves - 1,
                            mapping=mapp,
                        )
                    except Exception:
                        current_newick_oracle = None
                    target_tree_oracle = (
                        target_trees[b_idx]
                        if b_idx < len(target_trees)
                        else None
                    )
                    predicted_vanish_mask = _oracle_boundary_vanish_mask_for_sampling(
                        current_newick_oracle,
                        target_tree_oracle,
                        masks=masks,
                        n_leaves=n_leaves,
                        candidate_mask=candidate_mask,
                    )
                elif (
                    self.velocity_boundary_vanish_head_use_at_sampling
                    and B is not None
                ):
                    candidate_mask = supervised_mask & (L > eps_len)
                    predicted_vanish_mask = _predict_boundary_vanish_mask_from_logits(
                        B,
                        candidate_mask,
                    )
                use_boundary_vanish_one_step = (
                    self.velocity_boundary_vanish_one_step_use_at_sampling
                    and predicted_vanish_mask is not None
                )
                if boundary_blacklisted:
                    dt_hit = float("inf")
                    predicted_first_mask = np.zeros_like(supervised_mask, dtype=bool)
                elif use_boundary_vanish_one_step:
                    _, dt_hit, _ = _apply_boundary_vanish_one_step(
                        lengths=L,
                        velocities=V,
                        predicted_vanish_mask=predicted_vanish_mask,
                        supervised_mask=supervised_mask,
                        dt_cap=float("inf"),
                        eps_len=eps_len,
                    )
                    predicted_first_mask = predicted_vanish_mask.copy()
                elif (
                    self.velocity_first_hit_head_use_at_sampling
                    and H is not None
                    and not (
                        oracle_gate_first_hit_use_at_sampling
                        and target_trees is not None
                    )
                ):
                    candidate_mask = supervised_mask & (L > eps_len)
                    if oracle_first_hit_use_at_sampling and target_trees is not None:
                        try:
                            _, current_newick_oracle = build_tree_from_splits(
                                list(td.keys()),
                                td,
                                n_leaves,
                                root_leaf=n_leaves - 1,
                                mapping=mapp,
                            )
                        except Exception:
                            current_newick_oracle = None
                        target_tree_oracle = (
                            target_trees[b_idx]
                            if b_idx < len(target_trees)
                            else None
                        )
                        predicted_first_mask = _oracle_first_hit_mask_for_sampling(
                            current_newick_oracle,
                            target_tree_oracle,
                            masks=masks,
                            lengths=L,
                            n_leaves=n_leaves,
                            supervised_mask=candidate_mask,
                            velocity_sign_eps=self.velocity_sign_eps,
                            dt_eps=self.velocity_dt_eps,
                            first_hit_tol=first_hit_tol,
                        )
                    else:
                        if (
                            getattr(self.model, "first_hit_head_mode", "base")
                            == "autoregressive_set"
                            and E is not None
                        ):
                            predicted_first_mask = self.model.predict_first_hit_autoregressive_mask(
                                E.to(self.device, dtype=torch.float32),
                                candidate_mask=torch.as_tensor(
                                    candidate_mask,
                                    device=self.device,
                                    dtype=torch.bool,
                                ),
                                max_steps=getattr(
                                    self,
                                    "velocity_first_hit_sampling_max_edges",
                                    -1,
                                ),
                            ).detach().cpu().numpy().astype(bool)
                        else:
                            predicted_first_mask, _raw_first_count, _used_first_fallback = _predict_first_hit_mask_with_fallback(
                                H,
                                candidate_mask,
                                max_edges=getattr(
                                    self, "velocity_first_hit_sampling_max_edges", -1
                                ),
                                fallback_threshold=getattr(
                                    self,
                                    "velocity_first_hit_sampling_fallback_threshold",
                                    -1,
                                ),
                                fallback_top_k=getattr(
                                    self,
                                    "velocity_first_hit_sampling_fallback_top_k",
                                    -1,
                                ),
                            )
                    if np.any(predicted_first_mask):
                        rates = (
                            F.softplus(
                                torch.from_numpy(-V[predicted_first_mask]).float(),
                                beta=float(self.velocity_event_rate_beta),
                            )
                            .cpu()
                            .numpy()
                        ) / np.maximum(L[predicted_first_mask], eps_len)
                        dt_candidates = 1.0 / np.maximum(rates, eps_len)
                        dt_hit = float(np.min(dt_candidates))
                    else:
                        dt_hit = float("inf")
                else:
                    oracle_first_hit_mask = None
                    if (
                        oracle_gate_first_hit_use_at_sampling
                        and target_trees is not None
                    ):
                        try:
                            _, current_newick_oracle = build_tree_from_splits(
                                list(td.keys()),
                                td,
                                n_leaves,
                                root_leaf=n_leaves - 1,
                                mapping=mapp,
                            )
                        except Exception:
                            current_newick_oracle = None
                        target_tree_oracle = (
                            target_trees[b_idx]
                            if b_idx < len(target_trees)
                            else None
                        )
                        oracle_first_hit_mask = _oracle_first_hit_mask_for_sampling(
                            current_newick_oracle,
                            target_tree_oracle,
                            masks=masks,
                            lengths=L,
                            n_leaves=n_leaves,
                            supervised_mask=supervised_mask & (L > eps_len),
                            velocity_sign_eps=self.velocity_sign_eps,
                            dt_eps=self.velocity_dt_eps,
                            first_hit_tol=first_hit_tol,
                        )
                    if (
                        self.velocity_first_hit_head_use_at_sampling
                        and (
                            H is not None
                            or (
                                getattr(self.model, "first_hit_head_mode", "base")
                                == "autoregressive_set"
                                and E is not None
                            )
                        )
                        and oracle_first_hit_mask is not None
                        and np.any(oracle_first_hit_mask)
                    ):
                        candidate_mask = supervised_mask & (L > eps_len)
                        if (
                            getattr(self.model, "first_hit_head_mode", "base")
                            == "autoregressive_set"
                            and E is not None
                        ):
                            predicted_first_mask = self.model.predict_first_hit_autoregressive_mask(
                                E.to(self.device, dtype=torch.float32),
                                candidate_mask=torch.as_tensor(
                                    candidate_mask,
                                    device=self.device,
                                    dtype=torch.bool,
                                ),
                                max_steps=getattr(
                                    self,
                                    "velocity_first_hit_sampling_max_edges",
                                    -1,
                                ),
                            ).detach().cpu().numpy().astype(bool)
                        else:
                            predicted_first_mask, _raw_first_count, _used_first_fallback = _predict_first_hit_mask_with_fallback(
                                H,
                                candidate_mask,
                                max_edges=getattr(
                                    self, "velocity_first_hit_sampling_max_edges", -1
                                ),
                                fallback_threshold=getattr(
                                    self,
                                    "velocity_first_hit_sampling_fallback_threshold",
                                    -1,
                                ),
                                fallback_top_k=getattr(
                                    self,
                                    "velocity_first_hit_sampling_fallback_top_k",
                                    -1,
                                ),
                            )
                        if np.any(predicted_first_mask):
                            rates = (
                                F.softplus(
                                    torch.from_numpy(-V[predicted_first_mask]).float(),
                                    beta=float(self.velocity_event_rate_beta),
                                )
                                .cpu()
                                .numpy()
                            ) / np.maximum(L[predicted_first_mask], eps_len)
                            dt_candidates = 1.0 / np.maximum(rates, eps_len)
                            dt_hit = float(np.min(dt_candidates))
                        else:
                            dt_hit = float("inf")
                    else:
                        moving_neg = supervised_mask & (V < 0.0) & (L > eps_len)
                        if predicted_vanish_mask is not None and np.any(predicted_vanish_mask):
                            moving_neg = moving_neg & predicted_vanish_mask
                        if np.any(moving_neg):
                            dt_candidates = L[moving_neg] / -V[moving_neg]
                            dt_hit = float(np.min(dt_candidates))
                        else:
                            dt_hit = float("inf")
                        predicted_first_mask = moving_neg

                if (
                    sampling_actual_event_boundary_use_at_sampling
                    and not boundary_blacklisted
                    and not use_boundary_vanish_one_step
                ):
                    actual_moving_neg = supervised_mask & (V < 0.0) & (L > eps_len)
                    if np.any(actual_moving_neg):
                        actual_dt_candidates = np.full_like(L, np.inf, dtype=np.float64)
                        actual_dt_candidates[actual_moving_neg] = (
                            L[actual_moving_neg]
                            / np.maximum(-V[actual_moving_neg], eps_len)
                        )
                        actual_dt_hit = float(np.min(actual_dt_candidates[actual_moving_neg]))
                        actual_hit_mask = actual_moving_neg & (
                            np.abs(actual_dt_candidates - actual_dt_hit)
                            <= float(first_hit_tol)
                        )

                cache.append(
                    (
                        td,
                        L,
                        V,
                        n_leaves,
                        mapp,
                        dt_hit,
                        supervised_mask,
                        masks,
                        predicted_first_mask,
                        predicted_vanish_mask,
                        actual_dt_hit,
                        actual_hit_mask,
                        use_boundary_vanish_one_step,
                        current_boundary_state_key,
                        boundary_blacklisted,
                    )
                )
                dt_hit_list.append(dt_hit)
                actual_dt_hit_list.append(actual_dt_hit)

            # ---- GLOBAL dt across the batch ----
            dt_hit_global = min(dt_hit_list) if len(dt_hit_list) else float("inf")
            actual_dt_hit_global = (
                min(actual_dt_hit_list) if len(actual_dt_hit_list) else float("inf")
            )
            # Experimenting here, dt_hit_global is not a good metric we just jump, jump, jump, so why not use dt_base
            if sampling_actual_event_boundary_use_at_sampling:
                dt = min(dt_base, actual_dt_hit_global, T - t)
            elif fixed_dt_sampling:
                dt = min(dt_base, T - t)
            elif self.velocity_boundary_vanish_one_step_use_at_sampling:
                dt = min(dt_hit_global, T - t)
            else:
                dt = min(dt_base, dt_hit_global, T - t)
            #dt = min(dt_base, T-t)

            # defensive: prevent hard stall
            if dt <= 0:
                dt = min(dt_base, T - t)


            # ---- SECOND PASS: advance everyone with the SAME dt ----
            new_trees = []

            # Since update of token_cache happens per tree potentially, we need to defer it or track which ones changed.
            # However, batch indices align with zip(trees...), so we can update token_cache[i] if needed.

            for b_idx, (
                td,
                L,
                V,
                n_leaves,
                mapp,
                dt_hit,
                supervised_mask,
                masks,
                predicted_first_mask,
                predicted_vanish_mask,
                actual_dt_hit,
                actual_hit_mask,
                use_boundary_vanish_one_step,
                current_boundary_state_key,
                boundary_blacklisted,
            ) in enumerate(
                cache
            ):
                model_masks = edge_splits[b_idx]
                target_tree_for_trace = None
                if trace is not None and target_trees is not None and b_idx < len(target_trees):
                    target_tree_for_trace = target_trees[b_idx]
                if trace is not None and target_tree_for_trace:
                    try:
                        _, current_newick = build_tree_from_splits(
                            list(td.keys()),
                            td,
                            n_leaves,
                            root_leaf=n_leaves - 1,
                            mapping=mapp,
                        )
                        trace["velocity"].append(
                            {
                                "newick_tree": current_newick,
                                "target_tree": target_tree_for_trace,
                                "timepoint": float(t),
                                "num_leaves": int(n_leaves),
                            }
                        )
                        if (
                            prefix_replay_velocity_quota > 0
                            and _build_legacy_velocity_oracle_sample(
                                current_newick,
                                target_tree_for_trace,
                                timepoint=float(t),
                                num_leaves=int(n_leaves),
                            )
                            is not None
                        ):
                            prefix_replay_valid_velocity_counts[b_idx] += 1
                            if _prefix_replay_quota_satisfied():
                                prefix_replay_stop_requested = True
                                trace["stopped_for_prefix_replay_quota"] = True
                    except Exception:
                        pass
                if prefix_replay_stop_requested:
                    new_trees.append(td)
                    break
                # --- advance ---
                if boundary_blacklisted:
                    escape_dt = float(dt)
                    min_escape = float(oversize_polytomy_min_dt_escape)
                    if min_escape > 0.0:
                        escape_dt = min(max(escape_dt, min_escape), T - t)
                    L_new = L + escape_dt * V
                    if np.any(supervised_mask):
                        L_new[supervised_mask] = np.maximum(
                            L_new[supervised_mask], eps_len * 10.0
                        )
                elif use_boundary_vanish_one_step and np.isfinite(dt_hit) and dt >= dt_hit:
                    L_new, _, _ = _apply_boundary_vanish_one_step(
                        lengths=L,
                        velocities=V,
                        predicted_vanish_mask=predicted_vanish_mask,
                        supervised_mask=supervised_mask,
                        dt_cap=dt,
                        eps_len=eps_len,
                    )
                else:
                    L_new = L + dt * V
                # import pdb; pdb.set_trace()

                collapse_mask = None
                if (
                    sampling_actual_event_boundary_use_at_sampling
                    and not boundary_blacklisted
                    and not use_boundary_vanish_one_step
                ):
                    hit_boundary = bool(np.isfinite(actual_dt_hit) and (dt + hit_tol) >= actual_dt_hit)
                    collapse_mask = actual_hit_mask.copy()
                    if (
                        hit_boundary
                        and sampling_actual_event_boundary_include_predicted_first_hit
                        and predicted_first_mask is not None
                        and np.any(predicted_first_mask)
                    ):
                        collapse_mask = collapse_mask | predicted_first_mask
                    if hit_boundary and np.any(collapse_mask):
                        L_new[collapse_mask] = 0.0
                else:
                    # treat as boundary if we stepped past the first hit time for THIS tree
                    # (float equality with hit_tol=1e-10 is too strict)
                    hit_boundary = np.isfinite(dt_hit) and dt >= dt_hit
                    if not hit_boundary and np.any(predicted_first_mask):
                        # Numerical fallback: only moving supervised edges can trigger hits.
                        hit_boundary = bool((L_new[predicted_first_mask] <= eps_len).any())

                    if (
                        hit_boundary
                        and np.isfinite(dt_hit)
                        and np.any(predicted_first_mask)
                        and not use_boundary_vanish_one_step
                    ):
                        if self.velocity_first_hit_head_use_at_sampling and first_hit_logits is not None:
                            L_new[predicted_first_mask] = 0.0
                        else:
                            # Collapse near-simultaneous first-hit edges into the same boundary event.
                            neg_idx = np.where(predicted_first_mask)[0]
                            dt_candidates = L[neg_idx] / np.maximum(-V[neg_idx], eps_len)
                            near_first_hit = np.abs(dt_candidates - dt_hit) <= float(first_hit_tol)
                            if np.any(near_first_hit):
                                L_new[neg_idx[near_first_hit]] = 0.0
                    collapse_mask = predicted_first_mask

                if predicted_vanish_mask is not None:
                    allowed_collapse_mask = predicted_vanish_mask.copy()
                    if collapse_mask is not None:
                        allowed_collapse_mask = (
                            allowed_collapse_mask | collapse_mask
                        )
                    blocked_collapse_mask = supervised_mask & (
                        ~allowed_collapse_mask
                    )
                    if np.any(blocked_collapse_mask):
                        L_new[blocked_collapse_mask] = np.maximum(
                            L_new[blocked_collapse_mask],
                            eps_len * 10.0,
                        )
                elif (
                    (
                        self.sampling_only_first_hit_collapse
                        or sampling_actual_event_boundary_use_at_sampling
                    )
                    and collapse_mask is not None
                ):
                    blocked_collapse_mask = supervised_mask & (
                        ~collapse_mask
                    )
                    if np.any(blocked_collapse_mask):
                        L_new[blocked_collapse_mask] = np.maximum(
                            L_new[blocked_collapse_mask],
                            eps_len * 10.0,
                        )

                
                # update dict
                td2 = {m: float(l) for m, l in zip(masks, L_new) if l > eps_len}

                # If the active split set changed, we have crossed a boundary even if
                # the learned first-hit/vanish timing did not explicitly flag it.
                # Without this recovery, the sampler can silently drop a split,
                # return an unresolved tree, skip AR entirely, and leave the
                # structural cache stale for the next step.
                if not hit_boundary:
                    next_boundary_state_key = tuple(
                        sorted(int(mask) for mask in td2.keys())
                    )
                    if next_boundary_state_key != current_boundary_state_key:
                        hit_boundary = True
                        num_silent_boundary_recoveries += 1
                        if trace is not None:
                            trace["silent_boundary_recoveries"] += 1.0

                # Optional analysis-time safeguard: if the next boundary event would create
                # an oversized polytomy, either shrink the collapse set or skip the event.
                if hit_boundary and int(max_allowed_polytomy_size) >= 0:
                    try:
                        _, td2_newick_oversize = build_tree_from_splits(
                            list(td2.keys()),
                            td2,
                            n_leaves,
                            root_leaf=n_leaves - 1,
                            mapping=mapp,
                        )
                        oversize_poly = (
                            _max_polytomy_size_from_newick(td2_newick_oversize)
                            > int(max_allowed_polytomy_size)
                        )
                    except Exception:
                        oversize_poly = False

                    if oversize_poly:
                        policy = str(oversize_polytomy_policy or "none")
                        handled = False

                        if (
                            policy == "single_first_then_skip"
                            and predicted_first_mask is not None
                            and np.any(predicted_first_mask)
                        ):
                            candidate_idx = np.where(predicted_first_mask)[0]
                            if candidate_idx.size > 0:
                                rates = np.maximum(-V[candidate_idx], eps_len)
                                dt_candidates = L[candidate_idx] / rates
                                candidate_idx = candidate_idx[np.argsort(dt_candidates)]
                                for keep_idx in candidate_idx:
                                    L_trial = L + dt * V
                                    L_trial[keep_idx] = 0.0
                                    if predicted_vanish_mask is not None:
                                        allowed_trial = predicted_vanish_mask.copy()
                                        allowed_trial[keep_idx] = True
                                        blocked_trial = supervised_mask & (~allowed_trial)
                                        if np.any(blocked_trial):
                                            L_trial[blocked_trial] = np.maximum(
                                                L_trial[blocked_trial],
                                                eps_len * 10.0,
                                            )
                                    td2_trial = {
                                        m: float(l)
                                        for m, l in zip(masks, L_trial)
                                        if l > eps_len
                                    }
                                    try:
                                        _, td2_newick_trial = build_tree_from_splits(
                                            list(td2_trial.keys()),
                                            td2_trial,
                                            n_leaves,
                                            root_leaf=n_leaves - 1,
                                            mapping=mapp,
                                        )
                                        if (
                                            _max_polytomy_size_from_newick(td2_newick_trial)
                                            <= int(max_allowed_polytomy_size)
                                        ):
                                            L_new = L_trial
                                            td2 = td2_trial
                                            hit_boundary = True
                                            if predicted_first_mask is not None:
                                                predicted_first_mask = np.zeros_like(
                                                    predicted_first_mask, dtype=bool
                                                )
                                                predicted_first_mask[keep_idx] = True
                                            handled = True
                                            break
                                    except Exception:
                                        continue

                        if not handled and policy in {"skip", "single_first_then_skip"}:
                            if oversize_polytomy_blacklist_revisits:
                                oversize_boundary_topologies[b_idx].add(
                                    current_boundary_state_key
                                )
                            # Ignore the oversized boundary event and keep flowing in the
                            # current orthant by clipping any would-be collapsed edges.
                            escape_dt = float(dt)
                            min_escape = float(oversize_polytomy_min_dt_escape)
                            if min_escape > 0.0:
                                escape_dt = min(max(escape_dt, min_escape), T - t)
                            L_new = np.maximum(L + escape_dt * V, eps_len * 10.0)
                            td2 = {m: float(l) for m, l in zip(masks, L_new) if l > eps_len}
                            hit_boundary = False

                # We only need to rebuild Newick/Graph if we hit a boundary (topology changed)
                if hit_boundary:
                    num_merges = 0
                    topology_changed = True
                    stop_after_no_valid_merge_requested = False
                    boundary_merge_cap = (
                        float("inf")
                        if max_events is None
                        else max(0, int(max_events) - int(n_events_by_tree[b_idx]))
                    )
                    if int(max_autoregressive_merges_per_boundary) >= 0:
                        boundary_merge_cap = min(
                            boundary_merge_cap,
                            int(max_autoregressive_merges_per_boundary),
                        )
                    birthset_attempted_this_boundary = False
                    while (
                        topology_changed
                        and (
                            max_events is None
                            or int(n_events_by_tree[b_idx]) < int(max_events)
                        )
                        and num_merges < boundary_merge_cap
                    ):
                        boundary_state_key = tuple(sorted(int(mask) for mask in td2.keys()))
                        if (
                            self.skip_repeated_no_valid_boundary_use_at_sampling
                            and boundary_state_key in no_valid_boundary_topologies[b_idx]
                        ):
                            topology_changed = False
                            if trace is not None:
                                trace["skipped_no_valid_boundary_revisits"] += 1.0
                            break
                        (
                            graph,
                            td2_newick,
                            cached_tokenized_trees,
                            cached_component_groups,
                            cached_structural_cache_item,
                        ) = _get_ar_single_tree_artifacts(td2, n_leaves, mapp)

                        cached_groups_from_newick = None
                        cached_has_polytomy = None
                        if sampling_cache_polytomy_groups:
                            (
                                cached_groups_from_newick,
                                cached_has_polytomy,
                            ) = _get_polytomy_group_artifacts(td2_newick)
                        polytomy_nodes = (
                            cached_has_polytomy
                            if cached_has_polytomy is not None
                            else has_polytomy_fast(
                                td2_newick,
                                unrooted_ok=birthset_polytomy_unrooted_ok,
                            )
                        )
                        # td2 = {m: float(l) for m, l in zip(active_masks, L_new)}

                        trace_event = None
                        if polytomy_nodes:
                            # For autoregressive step, we just use standard tokenizer for now as it's rare event
                            tokenized_trees = (
                                cached_tokenized_trees
                                if cached_tokenized_trees is not None
                                else self.model.tokenizer([td2_newick])
                            )
                            if trace is not None and target_tree_for_trace:
                                try:
                                    autoregressive_time_value = (
                                        self._sampling_autoregressive_time_value(
                                            t,
                                            event_index=n_events_by_tree[b_idx],
                                            max_events=max_events,
                                        )
                                    )
                                    trace_event = {
                                        "source_newick": td2_newick,
                                        "newick": td2_newick,
                                        "target_tree": target_tree_for_trace,
                                        "time": autoregressive_time_value,
                                        "decoder_mode": "boundary_state",
                                    }
                                    trace["autoregressive"].append(trace_event)
                                    if (
                                        prefix_replay_autoregressive_quota > 0
                                        and _build_legacy_autoregressive_oracle_sample(
                                            td2_newick,
                                            target_tree_for_trace,
                                            module=self,
                                            time=autoregressive_time_value,
                                            split_multi_label_events=prefix_replay_split_multi_label_events,
                                        )
                                        is not None
                                    ):
                                        prefix_replay_valid_autoregressive_counts[
                                            b_idx
                                        ] += 1
                                        if _prefix_replay_quota_satisfied():
                                            prefix_replay_stop_requested = True
                                            trace[
                                                "stopped_for_prefix_replay_quota"
                                            ] = True
                                except Exception:
                                    pass
                            # import pdb; pdb.set_trace()
                        else:
                            break
                        if prefix_replay_stop_requested:
                            topology_changed = False
                            break

                        ar_context = (
                            torch.inference_mode()
                            if sampling_use_inference_mode
                            else torch.no_grad()
                        )
                        with ar_context:
                            autoregressive_component_groups = (
                                cached_component_groups
                                if cached_component_groups is not None
                                else [
                                    cached_groups_from_newick
                                    if cached_groups_from_newick is not None
                                    else get_structural_polytomy_groups_from_newick(
                                        td2_newick
                                    )
                                ]
                            )
                            logit_outputs = self.forward(
                                tokenized_trees,
                                self._sampling_autoregressive_time_tensor(
                                    t,
                                    event_index=n_events_by_tree[b_idx],
                                    max_events=max_events,
                                ),
                                _slice_single_tree_conditioning(
                                    phyla_embeddings,
                                    b_idx,
                                ),
                                autoregressive=True,
                                autoregressive_component_groups=autoregressive_component_groups,
                                autoregressive_case_indices=_slice_single_tree_conditioning(
                                    case_index_tensor,
                                    b_idx,
                                ),
                                autoregressive_start_topology_features=_slice_single_tree_conditioning(
                                    start_topology_feature_tensor,
                                    b_idx,
                                ),
                            )

                        birthset_allows_ar_fallback = (
                            self.topology_decoder == "birthset_with_ar_fallback"
                            and self.birthset_fallback == "ar"
                        )
                        use_birthset_decoder = (
                            self.topology_decoder
                            in {"birthset", "birthset_with_ar_fallback"}
                            and not birthset_attempted_this_boundary
                        )
                        if use_birthset_decoder:
                            birthset_attempted_this_boundary = True
                            birthset_plan = self._plan_birthset_boundary_splits(
                                logit_outputs,
                                td2.keys(),
                                n_leaves,
                            )
                            selected_births = list(birthset_plan.get("selected", []))
                            if max_events is not None:
                                remaining_events = max(
                                    int(max_events) - int(n_events_by_tree[b_idx]),
                                    0,
                                )
                                selected_births = selected_births[:remaining_events]
                            if selected_births:
                                top_change = False
                                selected_splits = []
                                for item in selected_births:
                                    new_split = int(item["split_mask"])
                                    to_print = [
                                        i
                                        for i in range(new_split.bit_length())
                                        if (new_split >> i) & 1
                                    ]
                                    _sampling_log_info(
                                        f"Birthset inserted split {new_split}: {to_print}"
                                    )
                                    td2[new_split] = float(
                                        self.birthset_birth_length
                                    )
                                    n_events += 1
                                    n_events_by_tree[b_idx] += 1
                                    num_topology_changes += 1
                                    top_change = True
                                    selected_splits.append(new_split)
                                if top_change:
                                    num_merges += 1
                                    topology_changed = True
                                    post_birthset_newick = build_tree_from_splits(
                                        list(td2.keys()),
                                        td2,
                                        n_leaves,
                                        root_leaf=n_leaves - 1,
                                        mapping=mapp,
                                    )[1]
                                    still_polytomy = has_polytomy_fast(
                                        post_birthset_newick,
                                        unrooted_ok=birthset_polytomy_unrooted_ok,
                                    )
                                    if trace_event is not None:
                                        trace_event.update(
                                            {
                                                "newick": post_birthset_newick,
                                                "decoder_mode": "birthset",
                                                "planned_merge_count": int(
                                                    len(selected_splits)
                                                ),
                                                "selected_result_splits": [
                                                    int(split)
                                                    for split in selected_splits
                                                ],
                                                "birthset_metrics": birthset_plan.get(
                                                    "metrics",
                                                    {},
                                                ),
                                                "birthset_unresolved_after_insert": float(
                                                    1.0 if still_polytomy else 0.0
                                                ),
                                            }
                                        )
                                    if still_polytomy and not birthset_allows_ar_fallback:
                                        no_valid_boundary_topologies[b_idx].add(
                                            boundary_state_key
                                        )
                                        if stop_on_no_valid_merge:
                                            stop_after_no_valid_merge_requested = True
                                        if trace is not None:
                                            trace[
                                                "birthset_incomplete_without_fallback"
                                            ] = True
                                        topology_changed = False
                                        break
                                    continue
                            if not birthset_allows_ar_fallback:
                                if trace_event is not None:
                                    trace_event.update(
                                        {
                                            "decoder_mode": "birthset",
                                            "planned_merge_count": 0,
                                            "selected_result_split": None,
                                            "birthset_metrics": birthset_plan.get(
                                                "metrics",
                                                {},
                                            ),
                                            "birthset_unresolved_after_insert": 1.0,
                                        }
                                    )
                                no_valid_boundary_topologies[b_idx].add(
                                    boundary_state_key
                                )
                                if stop_on_no_valid_merge:
                                    stop_after_no_valid_merge_requested = True
                                if trace is not None:
                                    trace["birthset_incomplete_without_fallback"] = True
                                topology_changed = False
                                break
                        
                        planned_merges = _plan_autoregressive_boundary_merges(
                            logit_outputs,
                            td2.keys(),
                            top_only=sampling_use_top_merge_planner,
                        )
                        if planned_merges:
                            planned_merges = planned_merges[:1]

                        top_change = False
                        stop_after_merge_requested = False
                        if not planned_merges:
                            _sampling_log_info("No valid merges found!")
                            if trace_event is not None:
                                trace_event.update(
                                    {
                                        "decoder_mode": "ar_fallback"
                                        if birthset_attempted_this_boundary
                                        else "ar",
                                        "planned_merge_count": 0,
                                        "selected_result_split": None,
                                    }
                                )
                            no_valid_boundary_topologies[b_idx].add(boundary_state_key)
                            if stop_on_no_valid_merge:
                                stop_after_no_valid_merge_requested = True
                        else:
                            selected_result_splits_for_trace = []
                            for planned in planned_merges:
                                polytomy_sizes.append(len(planned["splits_represented"]))

                                logits = planned["logits"]
                                G = logits.size(0)
                                tri = _get_tri_mask(G, logits.device)
                                logits_vec = logits[tri]
                                finite_logits = logits_vec[torch.isfinite(logits_vec)]
                                if finite_logits.numel() > 0:
                                    max_logits.append(
                                        float(torch.sigmoid(finite_logits).max().item())
                                    )

                                for subset, new_split in planned["subsets"]:
                                    to_print = [
                                        i
                                        for i in range(new_split.bit_length())
                                        if (new_split >> i) & 1
                                    ]
                                    _sampling_log_info(
                                        f"Merging subset {list(subset)} to create split {new_split}: {to_print}"
                                    )

                                    # New splits are born at the boundary; seed them
                                    # with a small positive length to avoid an
                                    # immediate re-collapse while keeping geometry local.
                                    td2[new_split] = float(autoregressive_birth_length)
                                    n_events += 1
                                    n_events_by_tree[b_idx] += 1
                                    num_topology_changes += 1
                                    top_change = True
                                    selected_result_splits_for_trace.append(
                                        int(new_split)
                                    )
                                    if (
                                        self.autoregressive_stop_after_merge_use_at_sampling
                                        and planned.get("decoder_mode")
                                        == "structured_subset"
                                        and float(planned.get("stop_after_merge_logit", 0.0))
                                        > 0.0
                                    ):
                                        stop_after_merge_requested = True

                            if top_change:
                                num_merges += 1
                                _sampling_log_info(
                                    "Merge step performed from decoded subsets"
                                )
                                if stop_after_merge_requested:
                                    _sampling_log_info(
                                        "Structured AR requested boundary stop after the applied merge."
                                    )
                            if trace_event is not None:
                                trace_event.update(
                                    {
                                        "decoder_mode": "ar_fallback"
                                        if birthset_attempted_this_boundary
                                        else "ar",
                                        "planned_merge_count": int(
                                            len(planned_merges)
                                        ),
                                        "selected_result_splits": selected_result_splits_for_trace,
                                    }
                                )
                        topology_changed = top_change
                        if stop_after_merge_requested:
                            topology_changed = False
                        if stop_after_no_valid_merge_requested:
                            topology_changed = False

                    if topology_changed and (
                        (
                            max_events is not None
                            and int(n_events_by_tree[b_idx]) >= int(max_events)
                        )
                        or num_merges >= boundary_merge_cap
                    ):
                        _sampling_log_info(
                            "Stopping boundary-resolution loop after hitting the merge-event cap."
                        )

                        # if not top_change:
                        #     logger.info("No more merges possible, pick a random polytomy and do a KNN merge")
                        #     output = random.choice(logit_outputs)
                        #     split_embeddings = output['group_embeddings']
                        #     group_represented = output['splits_represented']

                        #     if len(group_represented) != split_embeddings.size(0):
                        #         raise Exception("Whoa size mismatch between groups and split embeddings")
                            
                        #     i, j = _pick_knn_pair(split_embeddings, topM=KNN_TOPM, tau=KNN_TAU, stochastic=KNN_STOCHASTIC)

                        #     sm_i, sm_j = group_represented[i], group_represented[j]
                        #     new_split = int(sm_i) | int(sm_j)

                        #     if new_split not in td2:
                        #         # td2[new_split] = 1e-3  # tiny length
                        #         curr_lens = list(td2.values())
                        #         if len(curr_lens) > 0:
                        #             td2[new_split] = float(np.median(curr_lens))
                        #         else:
                        #             td2[new_split] = 1e-3
                        #     else:
                        #         # import pdb; pdb.set_trace()
                        #         raise Exception("Not possible to merge into a split that already exists...")

                        #     top_change = True
                        #     num_merges += 1
                        #     n_events += 1
                        #     num_topology_changes += 1

                        
                    (
                        _,
                        td2_newick_final,
                        _cached_tokenized_trees_final,
                        _cached_component_groups_final,
                        cached_structural_cache_item_final,
                    ) = _get_ar_single_tree_artifacts(td2, n_leaves, mapp)
                    # Update the cache for this batch index
                    new_item = (
                        cached_structural_cache_item_final
                        if cached_structural_cache_item_final is not None
                        else self.model.tokenizer.compute_structural_cache(
                            [td2_newick_final]
                        )[0]
                    )

                    token_cache.update(b_idx, new_item)

                    if _record_repeated_topology_visit(
                        boundary_topology_counts[b_idx],
                        tuple(sorted(int(mask) for mask in td2.keys())),
                        topology_repeat_cap,
                    ):
                        _sampling_log_info(
                            "Stopping sampling early after repeated boundary topology visit."
                        )
                        stop_for_repeated_topology = True
                        if trace is not None:
                            trace["stopped_for_repeated_topology"] = True
                    if stop_after_no_valid_merge_requested:
                        stop_for_no_valid_merge = True
                        if trace is not None:
                            trace["stopped_for_no_valid_merge"] = True

                new_trees.append(td2)
                if prefix_replay_stop_requested:
                    break

            trees = new_trees
            t += dt
            if prefix_replay_stop_requested:
                break
            if stop_for_repeated_topology:
                break
            if stop_for_no_valid_merge:
                break

            if n_steps % 100 == 0:
                _sampling_log_info(f"Step {n_steps}: dt={dt:.2e}, t={t:.2f}/{T}")

        # print(f"Sampling finished in {n_steps} steps. Total events: {n_events}")
        avg_polytomy_size = np.mean(polytomy_sizes) if polytomy_sizes else 0.0
        # if num_topology_changes > 0:
        #     import pdb; pdb.set_trace()
    
        _sampling_log_info(
            f"Sampling finished in {n_steps} steps. Total events: {n_events}, topology changes: {num_topology_changes}, "
            f"silent boundary recoveries: {num_silent_boundary_recoveries}, average polytomy size: {avg_polytomy_size:.2f}"
        )

        sampled_newicks = [
            build_tree_from_splits(
                list(td.keys()),
                td,
                n_leaves=n_leaves,
                root_leaf=n_leaves - 1,
                mapping=mapp,
            )[1]
            for td, n_leaves, mapp in zip(trees, num_leaves, mapping)
        ]
        sampled_newicks = [
            align_numeric_leaf_labels_to_reference(
                sampled_newick,
                start_newick,
                target_tree=(
                    target_trees[idx]
                    if target_trees is not None and idx < len(target_trees)
                    else None
                ),
            )[0]
            for idx, (sampled_newick, start_newick) in enumerate(
                zip(sampled_newicks, newick_starting_trees)
            )
        ]
        result = (
            sampled_newicks,
            num_topology_changes,
            sum(max_logits) / len(max_logits) if len(max_logits) > 0 else 0.0,
            avg_polytomy_size,
            len(polytomy_sizes),
        )
        if return_trace:
            return result + (trace,)
        return result

    def sample_compare(self, batch, train=True, num_samples=None, dt=0.02, save = True):
        if num_samples is None:
            num_samples = self.num_samples
        sampling_disable_inner_logging = bool(
            getattr(self, "sampling_disable_inner_logging", False)
        )
        nexus_filepaths = batch["nexus_filepaths"]
        tree_paths = batch["tree_paths"]
        ids = batch["ids"]

        if len(set(nexus_filepaths)) != 1 or len(set(ids)) != 1:
            raise Exception(
                "Each batch should correspond to one ID, not multiple different IDs, logic is inconsitent somewhere"
            )

        nexus_filepath = batch["nexus_filepaths"][0]
        id = batch["ids"][0]
        mapping = batch["mappings"][0]
        seq_ordering_map = batch["sequence_ordering_maps"][0]

        if train:
            real_trees = self.dataset.dataset_train.return_posterior_trees(id)
            num_leaves = self.dataset.dataset_train.return_number_leaves(id)
        else:
            real_trees = self.dataset.dataset_val.return_posterior_trees(id)
            num_leaves = self.dataset.dataset_val.return_number_leaves(id)

        if len(real_trees) > num_samples:
            pot_real_trees = random.sample(real_trees, num_samples)
        else:
            pot_real_trees = real_trees

        sanity_check = self.dataset.dataset_train.sanity_check if train else self.dataset.dataset_val.sanity_check
        random_sanity_check = self.dataset.dataset_train.random_sanity_check if train else self.dataset.dataset_val.random_sanity_check

        def _remap_tree_to_batch_indexing(tree_newick, offset=0, tree_kind="tree"):
            t_obj = EteTree(tree_newick, format=1)
            for leaf in t_obj.get_leaves():
                lookup_name = leaf.name
                if offset:
                    try:
                        lookup_name = str(int(lookup_name) + offset)
                    except ValueError:
                        raise Exception(
                            f"Non-integer leaf name '{leaf.name}' encountered in {tree_kind} while applying offset {offset}."
                        )

                mapped_name = seq_ordering_map.get(lookup_name)
                if mapped_name is None:
                    raise Exception(
                        f"Leaf name '{lookup_name}' in {tree_kind} not found in sequence ordering map."
                    )
                leaf.name = mapped_name

            return t_obj.write(format=1)

        real_trees = []
        for i in pot_real_trees:
            real_trees.append(_remap_tree_to_batch_indexing(i, offset=0, tree_kind="real tree"))

        for i in real_trees:
            if has_polytomy_fast(i):
                raise Exception(
                    "Whoa there is a polytomy in the real trees, need to resolve first!"
                )

        sampled_trees = []
        num_topology_changes = []
        avg_max_logits = []
        num_polytomies = 0
        avg_polytomy_sizes = []
        num_polytomies_resolved = []

        for _ in tqdm(range(num_samples)):
            # rt = Tree(num_leaves=num_leaves, random=True)
            # starting_tree = str(rt)
            if train:
                starting_tree = self.dataset.dataset_train.sample_random_tree(
                    real_trees[0]
                )
            else:
                starting_tree = self.dataset.dataset_val.sample_random_tree(
                    real_trees[0]
                )

            starting_leaf_names = {
                leaf.name for leaf in EteTree(starting_tree, format=1).get_leaves()
            }
            already_batch_indexed = starting_leaf_names.issubset(
                {str(value) for value in seq_ordering_map.values()}
            )
            random_tree_offset = (
                0
                if already_batch_indexed
                else 1 if (not sanity_check and not random_sanity_check) else 0
            )
            starting_tree = _remap_tree_to_batch_indexing(
                starting_tree, offset=random_tree_offset, tree_kind="random tree"
            )


            #### DEBUG CHANGE LATER MADE ONE TIMEPOINT ####
            timepoint = random.uniform(0, 1)

            start_time = time.time()
            sampled_tree, n_topology_changes_one, avg_max_logit, avg_polytomy_size, n_polytomies_resolved_one = self.sample(
                [starting_tree], batch["phyla_embeddings"], num_samples=1, dt_base=dt,
                debug_real_tree=(
                    real_trees[0] if not sampling_disable_inner_logging else None
                ),
            )
            if not sampling_disable_inner_logging:
                print(f"Sampling a single tree took {time.time() - start_time} seconds")

            avg_polytomy_sizes.append(avg_polytomy_size)
            num_polytomies_resolved.append(n_polytomies_resolved_one)

            sampled_tree = sampled_tree[0]
            num_topology_changes.append(n_topology_changes_one)
            avg_max_logits.append(avg_max_logit)
            if has_polytomy_fast(sampled_tree):
                sampled_tree = resolve_polytomies_random_deterministic(sampled_tree)
                if has_polytomy_fast(sampled_tree):
                    raise Exception(
                        "Whoa there is STILL a polytomy in the sampled tree, something is wrong!"
                    )
                num_polytomies += 1

            # Now do something with the sampled tree and the real trees
            sampled_trees.append(sampled_tree)

        sampled = [number_to_name_newick(i, {int(i):v for i, v in mapping.items()}, True) for i in sampled_trees]
        posterior_trees = [number_to_name_newick(i, {int(i):v for i, v in mapping.items()}, True) for i in real_trees]

        rf_dists = []
        n_pairs = min(len(sampled), len(posterior_trees))
        for i in range(n_pairs):
            rf_dists.append(calculate_norm_rf(sampled[i], posterior_trees[i]))
        
        rf_norm_val = np.mean(rf_dists) if rf_dists else 0.0

        if save:
            import pickle
            os.makedirs("samples", exist_ok=True)
            with open(f"samples/sample_trees_{self.global_step}.pkl", "wb") as f:
                pickle.dump((sampled, posterior_trees), f)

        try:
            metrics = compare_likelihood_distributions(
                nexus_filepath, true_trees=posterior_trees, sampled_trees=sampled, threads=1
            )
        except Exception as e:
            print(f"An error occurred during likelihood comparison: {e}")
            metrics = {}
        
        metrics["rf_norm"] = float(rf_norm_val)

        metrics.update(
            kl_divergence_topological_distributions(
                posterior_trees, sampled, num_leaves=num_leaves
            )
        )
        metrics.update(
            kl_divergence_tree_topology_distributions(
                posterior_trees, sampled
            )
        )
        metrics.update(
            topk_posterior_tree_recall(
                posterior_trees, sampled
            )
        )
        metrics.update(
            split_bipartition_frequency_correlation(
                posterior_trees, sampled, num_leaves=num_leaves
            )
        )
        metrics.update(compare_branch_length_distributions(posterior_trees, sampled))
        if not sampling_disable_inner_logging:
            print(f"Num polytomies resolved in sampling: {num_polytomies} out of {num_samples}")
            print("Average topology changes during sampling: ", np.mean(num_topology_changes))
            print("Average max logits during sampling: ", np.mean(avg_max_logits))
        overall_avg_polytomy_size = np.mean([s for s in avg_polytomy_sizes if s > 0]) if any(s > 0 for s in avg_polytomy_sizes) else 0.0
        if not sampling_disable_inner_logging:
            print(f"Average polytomy size during sampling: {overall_avg_polytomy_size:.2f}")
        
        avg_num_polytomies_resolved = np.mean(num_polytomies_resolved)
        if not sampling_disable_inner_logging:
            print(f"Average number of polytomies resolved during sampling: {avg_num_polytomies_resolved}")
        if self.record:
            self._wandb_log_filtered(
                {
                    "samples/number_of_polytomies_resolved": num_polytomies,
                    "samples/average_topology_changes": np.mean(num_topology_changes),
                    "samples/average_max_logits": np.mean(avg_max_logits),
                    "samples/average_num_polytomies_resolved": avg_num_polytomies_resolved,
                    "samples/average_polytomy_size": overall_avg_polytomy_size,
                },
                step=self.stepper,
            )

        return metrics
        
    def on_train_end(self):
        if self.record:
            wandb.finish()

    def _training_step_profile_enabled(self):
        frequency = int(getattr(self, "training_step_profile_frequency", 0) or 0)
        if frequency <= 0:
            return False
        step = int(getattr(self, "stepper", 0) or 0)
        warmup = int(getattr(self, "training_step_profile_warmup_steps", 0) or 0)
        return step >= warmup and (step % frequency) == 0

    def _training_step_profile_sync(self):
        if (
            not bool(getattr(self, "training_step_profile_sync_cuda", True))
            or not torch.cuda.is_available()
        ):
            return
        try:
            device = getattr(self, "device", None)
            if device is not None and getattr(device, "type", None) == "cuda":
                torch.cuda.synchronize(device)
            else:
                torch.cuda.synchronize()
        except Exception:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    def _training_step_profile_start(self):
        if not self._training_step_profile_enabled():
            return None
        self._training_step_profile_sync()
        now = time.perf_counter()
        return {"_start": now, "_last": now}

    def _training_step_profile_mark(self, profile, label):
        if profile is None:
            return
        self._training_step_profile_sync()
        now = time.perf_counter()
        last = profile.get("_last", profile["_start"])
        profile[label] = profile.get(label, 0.0) + (now - last)
        profile["_last"] = now

    def _training_step_profile_log(self, profile, logs=None):
        if profile is None:
            return
        self._training_step_profile_sync()
        now = time.perf_counter()
        total = now - profile["_start"]
        profile["total"] = total
        timing_items = [
            (key, value)
            for key, value in profile.items()
            if not key.startswith("_")
        ]
        timing_text = " ".join(
            f"{key}={float(value):.4f}s" for key, value in timing_items
        )
        logging.info(
            "TRAINING_STEP_PROFILE step=%s %s",
            int(getattr(self, "stepper", 0) or 0),
            timing_text,
        )
        if logs is None:
            return
        device = getattr(self, "device", torch.device("cpu"))
        for value in logs.values():
            if torch.is_tensor(value):
                device = value.device
                break
        for key, value in timing_items:
            logs[f"profile/{key}_sec"] = torch.tensor(float(value), device=device)

    def training_step(self, batch, _):
        # Skip if batch is None (all items failed tokenization in collate_fn)
        if batch is None:
            logging.warning("Skipping training step: batch is None (tokenization failed for all items)")
            print("Skipping training step: batch is None (tokenization failed for all items)")
            return None
        
        # Increment stepper at the START to ensure all logs in this step use the same step number
        self.stepper += 1
        step_profile = self._training_step_profile_start()
        
        opt = self.optimizers()
        opt.zero_grad()
        full_path_retry_base_batch = _clone_full_path_replay_retry_base(batch)
        initial_replay_retry_attempt = int(
            getattr(self, "training_step_full_path_replay_initial_retry_attempt", 0)
        )
        if initial_replay_retry_attempt > 0:
            (
                initial_capped_batch,
                initial_replay_reduced,
                initial_replay_counts,
            ) = _subsample_full_path_replay_for_oom_retry(
                full_path_retry_base_batch,
                stepper=self.stepper,
                retry_attempt=initial_replay_retry_attempt,
            )
            if initial_replay_reduced:
                logging.info(
                    "Initial full-path replay cap %s: %s",
                    initial_replay_retry_attempt,
                    initial_replay_counts,
                )
                batch = initial_capped_batch
                full_path_retry_base_batch = _clone_full_path_replay_retry_base(
                    initial_capped_batch
                )
        self._training_step_profile_mark(step_profile, "retry_setup")

        if self.deepspeed:

            success = False
            num = 0
            failed = False

            # Logic, if we have an out of memmory error we just resample with a smaller subtree and rerun
            while not success:
                self.logger_.log(f"Entering step {num}", level=logging.INFO)
                error_tensor = torch.zeros(1).cuda()
                if num > 1:
                    self.logger_.log(
                        "Batch is too large; subsampling full-path replay items "
                        "instead of pruning leaves",
                        level=logging.INFO,
                    )
                    if "loss" in locals():
                        loss = loss.detach()
                        del loss
                        gc.collect()

                    if num > 10:
                        return torch.tensor(0)

                    torch.cuda.empty_cache()
                    torch.distributed.barrier()
                    retry_attempt = max(1, int(num) - 1)
                    batch, replay_reduced, replay_counts = (
                        _subsample_full_path_replay_for_oom_retry(
                            full_path_retry_base_batch,
                            stepper=self.stepper,
                            retry_attempt=retry_attempt,
                        )
                    )
                    self.logger_.log(
                        f"Full-path replay retry {retry_attempt}: "
                        f"{replay_counts if replay_counts else 'no replay samples to reduce'}",
                        level=logging.INFO,
                    )
                    if not replay_reduced:
                        self.logger_.log(
                            "No full-path replay items were reduced on OOM retry; "
                            "retrying without leaf pruning.",
                            level=logging.INFO,
                        )

                    torch.distributed.barrier()
                    self.logger_.log(
                        f"We have all recreated our batches now moving on",
                        level=logging.INFO,
                    )
                try:
                    loss_status_tensor = torch.zeros(
                        torch.distributed.get_world_size()
                    ).cuda()
                    logs = self.step(batch)
                    if logs is not None:
                        loss_unscaled = logs["loss"]
                        loss = (
                            loss_unscaled * self.training_step_velocity_weight
                        )
                        logs["train/velocity_loss_unscaled"] = loss_unscaled.detach()
                        logs["train/velocity_loss_scaled"] = loss.detach()
                        logs["loss"] = loss
                        memory_error_tensor = torch.zeros(1).cuda()

                        # Go through every GPU get memmory used, if it is above 70% we will abort the manual backward and fail
                        stop_manual_backward = False
                        for i in range(torch.cuda.device_count()):
                            # Get the current memory usage
                            current_memory = torch.cuda.memory_allocated(i)
                            # Get the total memory
                            total_memory = torch.cuda.get_device_properties(
                                i
                            ).total_memory
                            fraction = current_memory / total_memory
                            if fraction > 0.75:
                                self.logger_.log(
                                    f"We detected that {i} device is above 75% memory usage!, will avoid manual backward!",
                                    level=logging.INFO,
                                )
                                stop_manual_backward = True
                                memory_error_tensor[0] = 1
                            self.logger_.log(
                                f"Device {i} is using {fraction} of its memory",
                                level=logging.INFO,
                            )
                            torch.distributed.barrier()

                        # If one at least fails the memory check then we will scuttle the backward
                        torch.distributed.all_reduce(memory_error_tensor)
                        if memory_error_tensor[0] > 0:
                            self.logger_.log(
                                f"Wow some is about to OOM we are scuttling the backward",
                                level=logging.INFO,
                            )
                            stop_manual_backward = True

                        # Okay what if one passes the memory check and still fails?
                        # loss_status_tensor = torch.zeros(1).cuda()

                        if not stop_manual_backward:
                            self.manual_backward(loss)
                            success = True
                            failed = False
                            self.logger_.log(f"Succeded!", level=logging.INFO)
                            loss_status_tensor[torch.distributed.get_rank()] = 1
                        else:
                            self.logger_.log(f"Skipping backward!", level=logging.INFO)
                            failed = True
                            success = False
                            num += 1
                            logs = None
                            loss_status_tensor[torch.distributed.get_rank()] = 1

                    else:
                        self.logger_.log(f"Failed!", level=logging.INFO)
                        num += 1
                        loss_status_tensor[torch.distributed.get_rank()] = 1
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.logger_.log(f"WARNING: out of memory", level=logging.INFO)
                        error_tensor[0] = 1
                        failed = True
                        logs = {"loss": torch.tensor(0)}
                        num += 1
                        loss_status_tensor[torch.distributed.get_rank()] = 1
                        self.logger_.log(f"Set up my status", level=logging.INFO)
                    else:
                        self.logger_.log(f"RAISING NEW ERROR {e}", level=logging.INFO)
                        raise e
                finally:
                    self.logger_.log(f"Entering check for the loss", level=logging.INFO)

                    while (
                        loss_status_tensor.sum() != torch.distributed.get_world_size()
                    ):
                        torch.distributed.all_reduce(loss_status_tensor)
                        self.logger_.log(
                            f"Waiting for everyone to finish\t{loss_status_tensor.sum()}\t{loss_status_tensor}",
                            level=logging.INFO,
                        )

                    torch.distributed.barrier()
                    torch.distributed.all_reduce(error_tensor)
                    if error_tensor[0] > 0:
                        self.logger_.log(
                            "Ooops someone had a OOM we should scuttle",
                            level=logging.INFO,
                        )
                        failed = True
                        success = False
                        num += 1

                    # print("Waiting")
                    torch.distributed.barrier()

                num += 1
                torch.distributed.barrier()
        else:
            success = False
            num = 0

            # Logic, if we have an out of memmory error we just resample with a smaller subtree and rerun
            while not success:

                # If fail will call zero grad again, may need this for deepspeed?
                opt.zero_grad()
                if num > 0:
                    logging.info(
                        "Batch is too large; subsampling full-path replay items "
                        "instead of pruning leaves"
                    )
                    if num > 10:
                        logging.info("We are spiraling, moving on")
                        return torch.tensor(0)

                    batch, replay_reduced, replay_counts = (
                        _subsample_full_path_replay_for_oom_retry(
                            full_path_retry_base_batch,
                            stepper=self.stepper,
                            retry_attempt=num,
                        )
                    )
                    logging.info(
                        "Full-path replay retry %s: %s",
                        num,
                        replay_counts if replay_counts else "no replay samples to reduce",
                    )
                    if not replay_reduced:
                        logging.info(
                            "No full-path replay items were reduced on OOM retry; "
                            "retrying without leaf pruning."
                        )
                    if self.training_step_verbose_logging_enabled:
                        logging.info(
                            f"Memory allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB"
                        )
                        logging.info(
                            f"Memory reserved: {torch.cuda.memory_reserved() / 1024 ** 2} MB"
                        )

                    gc.collect()
                try:
                    if self.training_step_verbose_logging_enabled:
                        logging.info(
                            f"Memory allocated before step: {torch.cuda.memory_allocated() / 1024 ** 2} MB"
                        )
                        logging.info(
                            f"Memory reserved before step: {torch.cuda.memory_reserved() / 1024 ** 2} MB"
                        )
                    replay_velocity_batch = None
                    replay_autoregressive_batch = None
                    replay_metric_logs = {}
                    if (
                        self.rollout_replay_velocity_weight > 0.0
                        or self.rollout_replay_autoregressive_weight > 0.0
                        or self.dynamic_start_bank_enabled
                    ):
                        (
                            replay_velocity_batch,
                            replay_autoregressive_batch,
                            replay_metric_logs,
                        ) = self._collect_rollout_replay_batches(train=True)
                    self._training_step_profile_mark(
                        step_profile,
                        "rollout_replay_collect",
                    )
                    velocity_training_batch = batch
                    autoregressive_training_batch = batch
                    terminal_training_batch = None
                    joint_forward = None
                    control_mode = bool(batch.get("_full_path_control_mode", False))
                    probe_parity_joint = bool(
                        control_mode and self.training_step_probe_parity_joint_update
                    )
                    if control_mode:
                        full_path_velocity_samples = (
                            batch.get("full_path_velocity_samples") or []
                        )
                        full_path_autoregressive_samples = (
                            batch.get("full_path_autoregressive_samples") or []
                        )
                        full_path_terminal_samples = (
                            batch.get("full_path_terminal_samples") or []
                        )
                        if (
                            full_path_velocity_samples
                            and full_path_autoregressive_samples
                            and self.training_step_joint_tokenize_velocity_ar
                        ):
                            (
                                velocity_training_batch,
                                autoregressive_training_batch,
                            ) = _build_velocity_autoregressive_replay_batches(
                                self,
                                full_path_velocity_samples,
                                full_path_autoregressive_samples,
                                joint_raw_graph_batch=batch.get(
                                    "full_path_joint_tokenizer_raw_graph_batch"
                                ),
                            )
                            if velocity_training_batch is not None:
                                velocity_training_batch[
                                    "_use_full_path_control_velocity_loss"
                                ] = True
                        else:
                            if full_path_velocity_samples:
                                velocity_training_batch = _build_velocity_replay_batch(
                                    self,
                                    full_path_velocity_samples,
                                )
                                if velocity_training_batch is not None:
                                    velocity_training_batch[
                                        "_use_full_path_control_velocity_loss"
                                    ] = True
                            if full_path_autoregressive_samples:
                                autoregressive_training_batch = (
                                    _build_autoregressive_replay_batch(
                                        self,
                                        full_path_autoregressive_samples,
                                    )
                                )
                        if (
                            full_path_terminal_samples
                            and self.velocity_terminal_head_weight > 0.0
                        ):
                            terminal_training_batch = _build_terminal_replay_batch(
                                self,
                                full_path_terminal_samples,
                            )
                        self._training_step_profile_mark(
                            step_profile,
                            "full_path_batch_build",
                        )
                        if (
                            probe_parity_joint
                            and self.live_phyla_model is not None
                        ):
                            shared_live_batches = [
                                velocity_training_batch,
                                autoregressive_training_batch,
                            ]
                            if (
                                terminal_training_batch is not None
                                and self.velocity_terminal_head_weight > 0.0
                            ):
                                shared_live_batches.append(terminal_training_batch)
                            self._attach_shared_live_phyla_embeddings_for_batches(
                                shared_live_batches,
                                grad=True,
                            )
                            self._training_step_profile_mark(
                                step_profile,
                                "live_phyla_shared",
                            )
                        if probe_parity_joint:
                            joint_forward = (
                                self._joint_velocity_autoregressive_forward(
                                    velocity_training_batch,
                                    autoregressive_training_batch,
                                )
                            )
                            if joint_forward is not None:
                                velocity_training_batch = joint_forward[
                                    "velocity_batch"
                                ]
                                autoregressive_training_batch = joint_forward[
                                    "autoregressive_batch"
                                ]
                            self._training_step_profile_mark(
                                step_profile,
                                "joint_trunk_forward",
                            )
                    else:
                        self._training_step_profile_mark(
                            step_profile,
                            "full_path_batch_build",
                        )

                    # --- HEAD 1: VELOCITY ---
                    if self.training_step_verbose_logging_enabled:
                        logging.info("DEBUG: Starting Velocity Head Training")
                    velocity_step_kwargs = {}
                    if joint_forward is not None:
                        velocity_step_kwargs = {
                            "precomputed_outputs": joint_forward[
                                "velocity_outputs"
                            ],
                            "prepared_batch": True,
                            "velocity_perturb_stats": joint_forward[
                                "velocity_perturb_stats"
                            ],
                        }
                    logs_vel = self.step(
                        velocity_training_batch,
                        autoregressive=False,
                        **velocity_step_kwargs,
                    )
                    self._training_step_profile_mark(step_profile, "velocity_step")
                    velocity_metric_logs = {
                        k: v for k, v in logs_vel.items() if k.startswith("velocity/")
                    }
                    loss_vel_unscaled = logs_vel["loss"]
                    loss_vel_regression_unscaled = logs_vel.get(
                        "loss_regression", loss_vel_unscaled
                    )
                    loss_vel_auxiliary_unscaled = logs_vel.get(
                        "loss_auxiliary",
                        loss_vel_unscaled - loss_vel_regression_unscaled,
                    )
                    replay_velocity_loss_unscaled = None
                    replay_velocity_regression_unscaled = None
                    replay_velocity_auxiliary_unscaled = None
                    replay_velocity_loss = None
                    if (
                        replay_velocity_batch is not None
                        and self.rollout_replay_velocity_weight > 0.0
                    ):
                        replay_velocity_logs = self.step(
                            replay_velocity_batch,
                            eval=True,
                            autoregressive=False,
                        )
                        replay_velocity_loss_unscaled = replay_velocity_logs["loss"]
                        replay_velocity_regression_unscaled = replay_velocity_logs.get(
                            "loss_regression",
                            replay_velocity_loss_unscaled,
                        )
                        replay_velocity_auxiliary_unscaled = replay_velocity_logs.get(
                            "loss_auxiliary",
                            replay_velocity_loss_unscaled
                            - replay_velocity_regression_unscaled,
                        )
                        replay_metric_logs["replay/velocity_loss_unscaled"] = (
                            replay_velocity_loss_unscaled.detach()
                        )
                        replay_metric_logs[
                            "replay/velocity_loss_regression_unscaled"
                        ] = replay_velocity_regression_unscaled.detach()
                        replay_metric_logs[
                            "replay/velocity_loss_auxiliary_unscaled"
                        ] = replay_velocity_auxiliary_unscaled.detach()
                    self._training_step_profile_mark(
                        step_profile,
                        "velocity_loss_prep",
                    )
                    if self.rollout_replay_legacy_loss_structure:
                        loss_vel = (
                            self.training_step_velocity_weight
                            * loss_vel_regression_unscaled
                        ) + loss_vel_auxiliary_unscaled
                        if replay_velocity_loss_unscaled is not None:
                            replay_velocity_loss = (
                                replay_velocity_loss_unscaled
                                * self.rollout_replay_velocity_weight
                            )
                            loss_vel = loss_vel + replay_velocity_loss
                    else:
                        loss_vel = (
                            (
                                loss_vel_regression_unscaled
                                + (
                                    self.rollout_replay_velocity_weight
                                    * replay_velocity_regression_unscaled
                                    if replay_velocity_regression_unscaled is not None
                                    else 0.0
                                )
                            )
                            * self.training_step_velocity_weight
                        ) + loss_vel_auxiliary_unscaled
                        if replay_velocity_auxiliary_unscaled is not None:
                            replay_velocity_loss = (
                                (
                                    self.training_step_velocity_weight
                                    * replay_velocity_regression_unscaled
                                )
                                + replay_velocity_auxiliary_unscaled
                            ) * self.rollout_replay_velocity_weight
                            loss_vel = loss_vel + (
                                self.rollout_replay_velocity_weight
                                * replay_velocity_auxiliary_unscaled
                            )
                    if replay_velocity_loss is not None:
                        replay_metric_logs["replay/velocity_loss_scaled"] = (
                            replay_velocity_loss.detach()
                        )
                    terminal_loss_unscaled = None
                    terminal_loss = None
                    if (
                        terminal_training_batch is not None
                        and self.velocity_terminal_head_weight > 0.0
                    ):
                        terminal_logs = self.step_terminal(
                            terminal_training_batch,
                            eval=control_mode,
                        )
                        terminal_loss_unscaled = terminal_logs["loss"]
                        terminal_loss = (
                            terminal_loss_unscaled
                            * self.velocity_terminal_head_weight
                        )
                        loss_vel = loss_vel + terminal_loss
                        replay_metric_logs["train/terminal_loss_unscaled"] = (
                            terminal_loss_unscaled.detach()
                        )
                        replay_metric_logs["train/terminal_loss_scaled"] = (
                            terminal_loss.detach()
                        )
                        replay_metric_logs.update(
                            {
                                k: v
                                for k, v in terminal_logs.items()
                                if k != "loss"
                            }
                        )
                        self._training_step_profile_mark(step_profile, "terminal_step")
                    loss_vel_unscaled_detached = loss_vel_unscaled.detach()
                    loss_vel_regression_unscaled_detached = (
                        loss_vel_regression_unscaled.detach()
                    )
                    loss_vel_auxiliary_unscaled_detached = (
                        loss_vel_auxiliary_unscaled.detach()
                    )
                    loss_vel_scaled_detached = loss_vel.detach()
                    if self.training_step_verbose_logging_enabled:
                        logging.info(
                            "Velocity head loss: total_raw=%.6f regression_raw=%.6f auxiliary_raw=%.6f scaled=%.6f weight=%.4f",
                            float(loss_vel_unscaled_detached.item()),
                            float(loss_vel_regression_unscaled_detached.item()),
                            float(loss_vel_auxiliary_unscaled_detached.item()),
                            float(loss_vel_scaled_detached.item()),
                            float(self.training_step_velocity_weight),
                        )
                    pre_ar_grads = None
                    velocity_grad_norm = None
                    if not probe_parity_joint:
                        self.manual_backward(loss_vel)
                        if self.training_step_separate_optimizer_steps:
                            if self.training_step_gradient_clip_val > 0.0:
                                self.clip_gradients(
                                    opt,
                                    gradient_clip_val=self.training_step_gradient_clip_val,
                                    gradient_clip_algorithm="norm",
                                )
                            opt.step()
                            opt.zero_grad()
                        elif self.training_step_autoregressive_grad_ratio is not None:
                            pre_ar_grads = {}
                            vel_sq = 0.0
                            for p in self.model.parameters():
                                if p.grad is None:
                                    continue
                                g_prev = p.grad.detach().clone()
                                pre_ar_grads[p] = g_prev
                                vel_sq += float(torch.sum(g_prev * g_prev))
                            velocity_grad_norm = vel_sq ** 0.5
                    self._training_step_profile_mark(
                        step_profile,
                        "velocity_backward_or_grad_prep",
                    )
                    if self.training_step_verbose_logging_enabled:
                        logging.info("DEBUG: Finished Velocity Head Training")

                    del logs_vel
                    if not probe_parity_joint:
                        del loss_vel_unscaled
                        del loss_vel
                        if hasattr(torch.cuda, "empty_cache"):
                            torch.cuda.empty_cache()

                    # --- HEAD 2: AUTOREGRESSIVE ---
                    if self.training_step_verbose_logging_enabled:
                        logging.info("DEBUG: Starting Autoregressive Head Training")
                    autoregressive_step_kwargs = {}
                    if joint_forward is not None:
                        autoregressive_step_kwargs = {
                            "precomputed_outputs": joint_forward[
                                "autoregressive_outputs"
                            ],
                            "prepared_batch": True,
                            "ar_prep_stats": joint_forward["ar_prep_stats"],
                        }
                    logs = self.step(
                        autoregressive_training_batch,
                        eval=control_mode,
                        autoregressive=True,
                        **autoregressive_step_kwargs,
                    )
                    self._training_step_profile_mark(
                        step_profile,
                        "autoregressive_step",
                    )
                    if "loss" not in logs:
                        import pickle

                        with open("debug_batch.pkl", "wb") as f:
                            pickle.dump(batch, f)
                        raise Exception(
                            "Loss not found in logs for autoregressive head!"
                        )
                    loss_unscaled = logs["loss"]
                    replay_autoregressive_loss_unscaled = None
                    replay_autoregressive_loss = None
                    if (
                        replay_autoregressive_batch is not None
                        and self.rollout_replay_autoregressive_weight > 0.0
                    ):
                        replay_autoregressive_logs = self.step(
                            replay_autoregressive_batch,
                            eval=True,
                            autoregressive=True,
                        )
                        replay_autoregressive_loss_unscaled = (
                            replay_autoregressive_logs["loss"]
                        )
                        replay_autoregressive_loss = (
                            replay_autoregressive_loss_unscaled
                            * self.rollout_replay_autoregressive_weight
                        )
                        replay_metric_logs["replay/autoregressive_loss_unscaled"] = (
                            replay_autoregressive_loss_unscaled.detach()
                        )
                    if self.rollout_replay_legacy_loss_structure:
                        loss = loss_unscaled * self.training_step_autoregressive_weight
                        if replay_autoregressive_loss is not None:
                            loss = loss + replay_autoregressive_loss
                    else:
                        loss = (
                            (
                                loss_unscaled
                                + (
                                    self.rollout_replay_autoregressive_weight
                                    * replay_autoregressive_loss_unscaled
                                    if replay_autoregressive_loss_unscaled is not None
                                    else 0.0
                                )
                            )
                            * self.training_step_autoregressive_weight
                        )
                        if replay_autoregressive_loss is not None:
                            replay_autoregressive_loss = (
                                self.training_step_autoregressive_weight
                                * replay_autoregressive_loss_unscaled
                                * self.rollout_replay_autoregressive_weight
                            )
                        if replay_autoregressive_loss is not None:
                            replay_metric_logs["replay/autoregressive_loss_scaled"] = (
                                replay_autoregressive_loss.detach()
                            )
                    logs["train/autoregressive_loss_unscaled"] = (
                        loss_unscaled.detach()
                    )
                    logs["train/autoregressive_loss_scaled"] = loss.detach()
                    logs["train/velocity_loss_unscaled"] = (
                        loss_vel_unscaled_detached
                    )
                    logs["train/velocity_loss_regression_unscaled"] = (
                        loss_vel_regression_unscaled_detached
                    )
                    logs["train/velocity_loss_auxiliary_unscaled"] = (
                        loss_vel_auxiliary_unscaled_detached
                    )
                    logs["train/velocity_loss_scaled"] = (
                        loss_vel_scaled_detached
                    )
                    logs.update(velocity_metric_logs)
                    logs.update(replay_metric_logs)
                    self._training_step_profile_mark(
                        step_profile,
                        "autoregressive_loss_prep",
                    )
                    branch_relax_loss_unscaled, branch_relax_logs = (
                        self._branch_relax_training_loss()
                    )
                    if branch_relax_loss_unscaled is not None:
                        branch_relax_loss = (
                            branch_relax_loss_unscaled
                            * self.branch_relax_head_weight
                        )
                        loss = loss + branch_relax_loss
                        logs["train/branch_relax_loss_unscaled"] = (
                            branch_relax_loss_unscaled.detach()
                        )
                        logs["train/branch_relax_loss_scaled"] = (
                            branch_relax_loss.detach()
                        )
                        logs.update(branch_relax_logs)
                    self._training_step_profile_mark(step_profile, "branch_relax")
                    if self.training_step_verbose_logging_enabled:
                        logging.info(
                            "Autoregressive head loss: raw=%.6f scaled=%.6f weight=%.4f",
                            float(loss_unscaled.detach().item()),
                            float(loss.detach().item()),
                            float(self.training_step_autoregressive_weight),
                        )
                        logging.info(
                            f"Memory allocated before backward: {torch.cuda.memory_allocated() / 1024 ** 2} MB"
                        )
                        logging.info(
                            f"Memory reserved before backward: {torch.cuda.memory_reserved() / 1024 ** 2} MB"
                        )

                    if probe_parity_joint:
                        joint_loss = loss_vel + loss
                        logs["train/probe_parity_joint_loss_scaled"] = (
                            joint_loss.detach()
                        )
                        logs["loss"] = joint_loss
                        self.manual_backward(joint_loss)
                    else:
                        logs["loss"] = loss
                        self.manual_backward(loss)
                        if (
                            not self.training_step_separate_optimizer_steps
                            and self.training_step_autoregressive_grad_ratio is not None
                            and pre_ar_grads is not None
                            and velocity_grad_norm is not None
                        ):
                            ar_sq = 0.0
                            for p in self.model.parameters():
                                if p.grad is None:
                                    continue
                                g_prev = pre_ar_grads.get(p)
                                if g_prev is None:
                                    ar_delta = p.grad.detach()
                                else:
                                    ar_delta = p.grad.detach() - g_prev
                                ar_sq += float(torch.sum(ar_delta * ar_delta))
                            autoregressive_grad_norm = ar_sq ** 0.5
                            grad_scale = 1.0
                            if (
                                autoregressive_grad_norm > 1e-12
                                and velocity_grad_norm > 1e-12
                            ):
                                target_norm = (
                                    velocity_grad_norm
                                    * self.training_step_autoregressive_grad_ratio
                                )
                                if autoregressive_grad_norm > target_norm:
                                    grad_scale = target_norm / (
                                        autoregressive_grad_norm + 1e-12
                                    )
                                    for p in self.model.parameters():
                                        g_prev = pre_ar_grads.get(p)
                                        if p.grad is None:
                                            if g_prev is not None:
                                                p.grad = g_prev.clone()
                                            continue
                                        if g_prev is None:
                                            p.grad.mul_(grad_scale)
                                        else:
                                            p.grad.copy_(
                                                g_prev + (p.grad - g_prev) * grad_scale
                                            )
                            device_for_logs = loss.device
                            logs["train/velocity_grad_norm"] = torch.tensor(
                                velocity_grad_norm, device=device_for_logs
                            )
                            logs["train/autoregressive_grad_norm"] = torch.tensor(
                                autoregressive_grad_norm, device=device_for_logs
                            )
                            logs["train/autoregressive_grad_scale"] = torch.tensor(
                                grad_scale, device=device_for_logs
                            )
                    self._training_step_profile_mark(step_profile, "backward")
                    if self.training_step_gradient_clip_val > 0.0:
                        self.clip_gradients(
                            opt,
                            gradient_clip_val=self.training_step_gradient_clip_val,
                            gradient_clip_algorithm="norm",
                        )
                    opt.step()
                    opt.zero_grad()
                    self._training_step_profile_mark(step_profile, "optimizer")

                    success = True
                    failed = False

                    if self.training_step_verbose_logging_enabled:
                        logging.info(
                            f"Memory allocated after backward: {torch.cuda.memory_allocated() / 1024 ** 2} MB"
                        )
                        logging.info(
                            f"Memory reserved after backward: {torch.cuda.memory_reserved() / 1024 ** 2} MB"
                        )
                    if probe_parity_joint:
                        del loss_vel_unscaled
                        del loss_vel

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logging.warning("WARNING: out of memory")
                        if hasattr(torch.cuda, "empty_cache"):
                            # Not sure about this
                            torch.cuda.empty_cache()

                        logging.info(
                            f"Memory allocated after OOM: {torch.cuda.memory_allocated() / 1024 ** 2} MB"
                        )
                        logging.info(
                            f"Memory reserved after OOM: {torch.cuda.memory_reserved() / 1024 ** 2} MB"
                        )

                        num += 1
                    else:
                        raise e

        # print(f"Entering a new world with status {failed}")
        if not failed and logs is not None:
            for k, v in logs.items():
                self._log_scalar_filtered(
                    k,
                    v.to("cuda"),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    logger=True,
                    sync_dist=True,
                )

            index, sub_tree_size, num_subtrees = self.dataset.chosen_tree
            lr = opt.optimizer.param_groups[0]["lr"]
            self._log_scalar_filtered("num_seq_per_subtree", sub_tree_size)
            logs["num_seq_per_subtree"] = sub_tree_size
            self._log_scalar_filtered("num_subtrees", num_subtrees)
            logs["num_subtrees"] = num_subtrees
            self._log_scalar_filtered("lr", lr)
            logs["lr"] = lr
            if self.logger_ is not None:
                self.logger_.log(logs, level=logging.INFO)
            self._training_step_profile_mark(step_profile, "scalar_logging")
        else:
            print(logs)

        if logs is not None:
            if self.record:
                self._wandb_log_filtered(logs, step=self.stepper)
            if not self.dataset.msa_distance:
                self.dataset.update_normrf(logs["norm_rf_distance"])

            if self.deepspeed:
                if self.training_step_gradient_clip_val > 0.0:
                    self.clip_gradients(
                        opt,
                        gradient_clip_val=self.training_step_gradient_clip_val,
                        gradient_clip_algorithm="norm",
                    )

            self.current_step_value += 1
            if self.deepspeed:
                opt.step()
            # print("Hi Im here waiting!")
            if self.deepspeed:
                torch.distributed.barrier()

            # Perform learning rate schedling
            if self.lr_scheduler == "cosine":
                sch1 = self.lr_schedulers()
                sch1.step()
            elif self.lr_scheduler == "cosine_warmup":
                sch1, sch2 = self.lr_schedulers()
                # Perform warmup
                if self.num_warmup_steps > 0:
                    sch1.step()
                    self.num_warmup_steps -= 1
                # Perform cosine annealing
                else:
                    sch2.step()
            elif self.lr_scheduler == "warmup":
                sch1 = self.lr_schedulers()
                # Perform warmup
                if self.num_warmup_steps > 0:
                    sch1.step()
                    self.num_warmup_steps -= 1
            self._training_step_profile_mark(step_profile, "scheduler")

            # ADD CODE HERE TO UPDATE ADAPTIVE BATCH SIZE SAMPLER

            if self._training_sample_due():
                if self.training_sampling_mode == "harness_sanity":
                    metrics = self.sample_compare_harness(train=True)
                else:
                    metrics = self.sample_compare(batch, train=True, dt=self.dt)
                
                for k, v in metrics.items():
                    self._log_scalar_filtered(
                        f"sample_metrics/{k}",
                        v,
                        on_step=True,
                        logger=True,
                    )
                self._append_sample_metrics_trace(metrics)
                self._save_sample_metrics_checkpoint()
                self._advance_training_sampling_schedule()
                if self.record:
                    self._wandb_log_filtered(
                        {f"sample_metrics/{k}": v for k, v in metrics.items()},
                        step=self.stepper,
                    )
                if self.training_step_verbose_logging_enabled:
                    print(metrics)
                rf_norm = metrics.get("rf_norm")
                stop_threshold = self.training_sampling_stop_rf_threshold
                if stop_threshold is None and self.training_sampling_stop_on_zero_rf:
                    stop_threshold = 0.0
                if (
                    stop_threshold is not None
                    and rf_norm is not None
                    and float(rf_norm) <= float(stop_threshold)
                    and self.trainer is not None
                ):
                    logging.info(
                        "Stopping early because sampled rf_norm reached %.6f (threshold=%.6f) at global_step=%s",
                        float(rf_norm),
                        float(stop_threshold),
                        self.global_step,
                    )
                    self.trainer.should_stop = True

            self._training_step_profile_mark(step_profile, "sample_metrics")
            self._training_step_profile_log(step_profile, logs)
            return logs["loss"]
        else:
            self._training_step_profile_log(step_profile, logs)
            return torch.tensor(0)

    def validation_step(self, batch, batch_idx):
        pass

    def on_before_optimizer_step(self, optimizer):
        frequency = int(getattr(self, "grad_norm_log_frequency", 1) or 0)
        if frequency <= 0 or (int(self.stepper) % frequency) != 0:
            return

        # Compute the 2-norm for each layer
        norms = grad_norm(self, norm_type=2)

        def _grad_norm_scalar(value):
            if torch.is_tensor(value):
                return float(value.detach().cpu().item())
            return float(value)

        if "grad_2.0_norm_total" in norms:
            total = norms["grad_2.0_norm_total"]
        else:
            total = norms.get("total_grad_norm", 0.0)  # hypothetical fallback
            if total == 0.0:
                # Just take the first key that looks like total if exists
                keys = [k for k in norms.keys() if "total" in k]
                if keys:
                    total = norms[keys[0]]
        total = _grad_norm_scalar(total)

        # total = norms.get("grad_2.0_norm_total", 0.0)

        layer_norms = {k: v for k, v in norms.items() if "total" not in k}
        if layer_norms:
            layer_norm_values = [
                _grad_norm_scalar(value)
                for value in layer_norms.values()
            ]
            max_grad = max(layer_norm_values)
            mean_grad = sum(layer_norm_values) / max(len(layer_norm_values), 1)
        else:
            max_grad = 0.0
            mean_grad = 0.0

        self._log_scalar_filtered(
            "grad_norm_max", max_grad, prog_bar=True, on_step=True
        )
        self._log_scalar_filtered(
            "grad_norm_mean", mean_grad, prog_bar=False, on_step=True
        )

        # Print a warning if exploding
        if self.training_step_verbose_logging_enabled and max_grad > 1:
            print(
                f"[Warning] Gradient norm unusually high: max={max_grad:.2e}, mean={mean_grad:.2e}"
            )

        self._log_scalar_filtered("grad_norm_total", total)
        if self.training_step_verbose_logging_enabled:
            print(
                f"step {self.global_step:4d}  total_grad_norm = {total:.2f} mean is {mean_grad:.2f} max is {max_grad:.2f}"
            )
        if self.record:
            self._wandb_log_filtered(
                {
                    "grad/grad_norm_total": total,
                    "grad/grad_norm_max": max_grad,
                    "grad/grad_norm_mean": mean_grad,
                },
                step=self.stepper,
            )

    def configure_optimizers(self):
        parameters = self.parameters()
        if (
            self.live_phyla_model is not None
            and self.live_phyla_unfreeze
            and self.live_phyla_lr is not None
        ):
            live_ids = {id(param) for param in self.live_phyla_model.parameters()}
            live_params = [
                param
                for param in self.live_phyla_model.parameters()
                if param.requires_grad
            ]
            other_params = [
                param
                for param in self.parameters()
                if id(param) not in live_ids and param.requires_grad
            ]
            parameters = [
                {"params": other_params, "lr": self.lr},
                {"params": live_params, "lr": self.live_phyla_lr},
            ]
        if self.deepspeed:
            optimizer = FusedAdam(parameters, lr=self.lr)
        elif self.optimizer_name == "adam":
            optimizer = optim.Adam(parameters, lr=self.lr)
        else:
            optimizer = optim.AdamW(parameters, lr=self.lr)

        if self.lr_scheduler == "cosine":
            sch1 = CosineAnnealingLR(
                optimizer, T_max=self.num_annealing_steps
            )  # Set to current number of steps for training 7 days
            return [optimizer], [sch1]
        elif self.lr_scheduler == "cosine_warmup":
            sch1 = LinearLR(
                optimizer, start_factor=self.lr, total_iters=self.num_warmup_steps
            )
            sch2 = CosineAnnealingLR(optimizer, T_max=self.num_annealing_steps)
            return [optimizer], [sch1, sch2]
        elif self.lr_scheduler == "warmup":
            sch1 = LinearLR(
                optimizer, start_factor=self.lr, total_iters=self.num_warmup_steps
            )
            return [optimizer], [sch1]
        else:
            scheduler = []
            return optimizer
