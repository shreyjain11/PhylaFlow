#!/usr/bin/env python3
"""Overfit one 10-leaf tree using Phyla-beta sequence representations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import types
import urllib.request
from io import StringIO
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from ete3 import Tree as EteTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.full_sanity_fixedpair_20260401.benchmark_phylaflow_nexus_sitechunk_phyla import (
    iter_windows,
    load_nexus_alignment,
)
from scripts.probe_phyla_10leaf_pairwise import (
    _load_tree_payload,
    _numeric_leaf_key,
    _pruned_target_tree,
    _read_jsonl,
)
from utils.metric_utils import calculate_norm_rf


DEFAULT_INDEX = Path(
    "/ewsc/yektefai/phylaflow_datasets/"
    "orthomam10leaf_train80_datasetsplit_8000cases_20260511_index.jsonl"
)
DEFAULT_NEXUS_ROOT = Path("/ewsc/yektefai/phylaflow_datasets/nexus")
DEFAULT_EMBEDDING_ROOT = Path(
    "/ewsc/yektefai/phylaflow_datasets/phyla_embeddings_sitechunk_cpu_20260428"
)
DEFAULT_CHECKPOINT = Path("weights/11564369")
PHYLA_BETA_URL = "https://dataverse.harvard.edu/api/access/datafile/11564369"


def _strip_alignment_gaps(sequence: str) -> str:
    return str(sequence).replace("-", "").replace(".", "")


def _sequence_windows(
    sequences: list[str],
    input_mode: str,
    window_size: int,
    stride: int,
    max_windows: int,
) -> tuple[list[str], list[tuple[int, int]]]:
    if input_mode == "aligned-windows":
        windows = iter_windows(len(sequences[0]), int(window_size), int(stride))
    elif input_mode == "raw-full":
        sequences = [_strip_alignment_gaps(sequence) for sequence in sequences]
        windows = [(0, max(len(sequence) for sequence in sequences))]
    elif input_mode == "raw-windows":
        sequences = [_strip_alignment_gaps(sequence) for sequence in sequences]
        windows = iter_windows(
            max(len(sequence) for sequence in sequences),
            int(window_size),
            int(stride),
        )
    else:
        raise ValueError(f"Unknown input mode: {input_mode}")
    if int(max_windows) > 0:
        windows = windows[: int(max_windows)]
    return sequences, windows


def _install_skbio_stub() -> None:
    if "skbio" in sys.modules:
        return
    skbio = types.ModuleType("skbio")
    skbio_tree = types.ModuleType("skbio.tree")

    class _UnusedDistanceMatrix:
        def __init__(self, *args, **kwargs):
            raise ImportError("scikit-bio is not required for this probe")

    def _unused_nj(*args, **kwargs):
        raise ImportError("scikit-bio is not required for this probe")

    skbio.DistanceMatrix = _UnusedDistanceMatrix
    skbio_tree.nj = _unused_nj
    sys.modules["skbio"] = skbio
    sys.modules["skbio.tree"] = skbio_tree


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _embedding_path(root: Path, dataset_id: str) -> Path:
    dataset_id = str(dataset_id).upper()
    suffixes = (
        "_phyla_beta_sitechunk_w256_s256_embeddings.pt",
        "_phyla_beta_embeddings.pt",
        "_embeddings.pt",
    )
    for suffix in suffixes:
        candidate = root / f"{dataset_id}{suffix}"
        if candidate.exists():
            return candidate
    matches = sorted(root.glob(f"{dataset_id}*_embeddings.pt"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No Phyla embedding file found for {dataset_id} in {root}")


def _leaf_to_index(leaf_name: str, names: List[str]) -> int:
    name_to_idx = {str(name): idx for idx, name in enumerate(names)}
    if str(leaf_name) in name_to_idx:
        return name_to_idx[str(leaf_name)]
    idx = int(str(leaf_name))
    if 1 <= idx <= len(names):
        return idx - 1
    if 0 <= idx < len(names):
        return idx
    raise IndexError(f"Leaf {leaf_name} out of range for {len(names)} sequences")


def _select_case(args: argparse.Namespace) -> tuple[dict, str, list[str]]:
    rows = _read_jsonl(args.index, limit=int(args.row_index) + 1)
    if int(args.row_index) >= len(rows):
        raise IndexError(f"row-index {args.row_index} outside {args.index}")
    row = rows[int(args.row_index)]
    tree_newick, leaves = _pruned_target_tree(
        row,
        subset_size=int(args.subset_size),
        seed=int(args.seed),
    )
    return row, tree_newick, leaves


def _target_pair_data(tree_newick: str, leaves: list[str], positive_quantile: float):
    tree = EteTree(tree_newick, format=1)
    leaf_nodes = {str(leaf.name): leaf for leaf in tree.iter_leaves()}
    pair_indices = []
    distances = []
    for i, left in enumerate(leaves):
        for j, right in enumerate(leaves[i + 1 :], start=i + 1):
            pair_indices.append((i, j))
            distances.append(
                float(leaf_nodes[left].get_distance(leaf_nodes[right], topology_only=True))
            )
    dist = torch.tensor(distances, dtype=torch.float32)
    norm_dist = dist / dist.max().clamp_min(1.0)
    threshold = float(
        np.quantile(dist.detach().cpu().numpy(), float(positive_quantile), method="higher")
    )
    close = (dist <= threshold).float()
    return pair_indices, dist, norm_dist, close, threshold


def _load_cached_representations(
    row: dict,
    leaves: list[str],
    embedding_root: Path,
) -> tuple[list[str], torch.Tensor, torch.Tensor, dict]:
    payload = torch.load(_embedding_path(embedding_root, row["dataset_id"]), map_location="cpu")
    names = [str(name) for name in payload.get("sequence_names") or payload.get("names")]
    pooled = payload.get("embeddings")
    if pooled is None:
        pooled = payload.get("phyla_embeddings")
    chunks = payload.get("chunk_embeddings")
    if pooled is None:
        raise ValueError("Cached payload does not contain pooled embeddings")
    pooled = pooled.detach().cpu().float() if torch.is_tensor(pooled) else torch.as_tensor(pooled).float()
    if pooled.dim() == 3:
        pooled = pooled.squeeze(0)
    if chunks is None:
        chunks = pooled.unsqueeze(0)
    chunks = chunks.detach().cpu().float() if torch.is_tensor(chunks) else torch.as_tensor(chunks).float()
    indices = [_leaf_to_index(leaf, names) for leaf in leaves]
    selected_names = [names[idx] for idx in indices]
    return selected_names, pooled[indices], chunks[:, indices], {
        "source": "cached",
        "embedding_path": str(_embedding_path(embedding_root, row["dataset_id"])),
        "num_windows": int(chunks.size(0)),
    }


def _download_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(f"Downloading Phyla-beta checkpoint to {path}", flush=True)
    try:
        urllib.request.urlretrieve(PHYLA_BETA_URL, tmp)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        subprocess.check_call(["wget", "-O", str(tmp), PHYLA_BETA_URL])
    os.replace(tmp, path)


def _load_phyla_beta_state_dict(checkpoint: Path, device: str) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    normalized = {}
    for key, value in state_dict.items():
        for prefix in ("model_name.", "_forward_module.model.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        normalized[key] = value
    return normalized


def _load_live_phyla(checkpoint: Path, device: str):
    _install_skbio_stub()
    import mamba_ssm.modules.mamba_simple as mamba_simple
    import mamba_ssm.ops.selective_scan_interface as selective_scan_interface
    import phyla.model.model as phyla_model
    from phyla.model.model import Config, Phyla

    if not checkpoint.exists():
        _download_checkpoint(checkpoint)

    if str(device).startswith("cpu"):
        phyla_model.RMSNorm = torch.nn.RMSNorm
        mamba_simple.RMSNorm = torch.nn.RMSNorm
        mamba_simple.causal_conv1d_fn = None
        selective_scan_interface.selective_scan_cuda = None

    cfg = Config()
    cfg.model.model_name = "phyla-beta"
    if str(device).startswith("cpu"):
        cfg.model.fused_add_norm = False
    model = Phyla(cfg, name="phyla-beta", device=device)
    model.load_state_dict(_load_phyla_beta_state_dict(checkpoint, device), strict=True)
    model.to(device)
    model.eval()
    return model


def _load_live_representations(
    row: dict,
    leaves: list[str],
    nexus_root: Path,
    checkpoint: Path,
    device: str,
    window_size: int,
    stride: int,
    max_windows: int,
    input_mode: str,
) -> tuple[list[str], torch.Tensor, torch.Tensor, dict]:
    names, sequences = load_nexus_alignment(nexus_root / f"{row['dataset_id']}.nex")
    indices = [_leaf_to_index(leaf, names) for leaf in leaves]
    selected_names = [names[idx] for idx in indices]
    selected_sequences = [sequences[idx] for idx in indices]
    model = _load_live_phyla(checkpoint, device)
    selected_sequences, windows = _sequence_windows(
        selected_sequences,
        input_mode,
        int(window_size),
        int(stride),
        int(max_windows),
    )
    if not windows:
        raise ValueError("No windows selected")

    chunk_embeddings = []
    weighted_sum = None
    total_weight = 0.0
    with torch.inference_mode():
        for window_idx, (start, end) in enumerate(windows):
            chunk_sequences = [seq[start:end] for seq in selected_sequences]
            encoded, cls_mask, seq_mask, _ = model.encode(chunk_sequences, selected_names)
            embedding = model(
                encoded.to(device),
                seq_mask.to(device),
                cls_mask.to(device),
            ).detach().cpu().float()
            if embedding.dim() == 3:
                embedding = embedding.squeeze(0)
            chunk_embeddings.append(embedding)
            weight = float(end - start)
            weighted_sum = embedding * weight if weighted_sum is None else weighted_sum + embedding * weight
            total_weight += weight
            print(
                f"live Phyla window {window_idx + 1}/{len(windows)} columns {start}:{end}",
                flush=True,
            )
    chunks = torch.stack(chunk_embeddings, dim=0)
    pooled = (weighted_sum / max(total_weight, 1.0)).float()
    return selected_names, pooled, chunks, {
        "source": "live",
        "checkpoint": str(checkpoint),
        "num_windows": int(chunks.size(0)),
        "window_size": int(window_size),
        "stride": int(stride),
        "input_mode": input_mode,
    }


def _load_selected_sequences(
    row: dict,
    leaves: list[str],
    nexus_root: Path,
) -> tuple[list[str], list[str]]:
    names, sequences = load_nexus_alignment(nexus_root / f"{row['dataset_id']}.nex")
    indices = [_leaf_to_index(leaf, names) for leaf in leaves]
    return [names[idx] for idx in indices], [sequences[idx] for idx in indices]


def _select_phyla_trainable_params(model: nn.Module, scope: str) -> tuple[list[nn.Parameter], int]:
    for param in model.parameters():
        param.requires_grad_(False)
    if scope == "all":
        modules = [model]
    elif scope == "last-module":
        modules = [model.modul[-1]]
    elif scope == "tree-head":
        modules = [module.tree_head for module in model.modul if hasattr(module, "tree_head")]
    else:
        raise ValueError(f"Unknown Phyla train scope: {scope}")
    params: list[nn.Parameter] = []
    for module in modules:
        for param in module.parameters():
            param.requires_grad_(True)
            params.append(param)
    return params, sum(param.numel() for param in params)


def _live_pooled_with_grad(
    model,
    selected_sequences: list[str],
    selected_names: list[str],
    windows: list[tuple[int, int]],
    device: str,
) -> torch.Tensor:
    weighted_sum = None
    total_weight = 0.0
    for start, end in windows:
        chunk_sequences = [seq[start:end] for seq in selected_sequences]
        encoded, cls_mask, seq_mask, _ = model.encode(chunk_sequences, selected_names)
        embedding = model(
            encoded.to(device),
            seq_mask.to(device),
            cls_mask.to(device),
        )
        if embedding.dim() == 3:
            embedding = embedding.squeeze(0)
        weight = float(end - start)
        weighted_sum = embedding * weight if weighted_sum is None else weighted_sum + embedding * weight
        total_weight += weight
    if weighted_sum is None:
        raise ValueError("No windows selected")
    return weighted_sum / max(total_weight, 1.0)


def _train_live_phyla_pair_head(
    row: dict,
    target_tree: str,
    leaves: list[str],
    args: argparse.Namespace,
) -> dict:
    if not str(args.device).startswith("cuda"):
        raise ValueError("Unfreezing Phyla requires a CUDA device in this environment")
    selected_names, selected_sequences = _load_selected_sequences(row, leaves, args.nexus_root)
    model = _load_live_phyla(args.checkpoint, args.device)
    trainable_params, trainable_count = _select_phyla_trainable_params(
        model,
        args.phyla_train_scope,
    )
    model.train()

    selected_sequences, windows = _sequence_windows(
        selected_sequences,
        args.input_mode,
        int(args.window_size),
        int(args.stride),
        int(args.max_windows),
    )
    if not windows:
        raise ValueError("No windows selected")

    pair_indices, _, norm_dist, close, close_threshold = _target_pair_data(
        target_tree,
        leaves,
        args.positive_quantile,
    )
    norm_dist = norm_dist.to(args.device)
    close = close.to(args.device)
    pair_head = PairHead(4 * 256, int(args.hidden_dim)).to(args.device)
    opt = torch.optim.AdamW(
        [
            {"params": trainable_params, "lr": float(args.phyla_lr)},
            {"params": pair_head.parameters(), "lr": float(args.lr)},
        ],
        weight_decay=float(args.weight_decay),
    )

    history = []
    for step in range(int(args.epochs)):
        pooled = _live_pooled_with_grad(
            model,
            selected_sequences,
            selected_names,
            windows,
            args.device,
        )
        pooled = F.normalize(pooled.float(), dim=-1)
        features = _pair_features(pooled, pair_indices)
        pred_dist, logits = pair_head(features)
        pred_dist = torch.sigmoid(pred_dist)
        loss_dist = F.mse_loss(pred_dist, norm_dist)
        loss_close = F.binary_cross_entropy_with_logits(logits, close)
        loss = loss_dist + loss_close
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(
                list(trainable_params) + list(pair_head.parameters()),
                float(args.grad_clip),
            )
        opt.step()
        if step == 0 or (step + 1) % int(args.log_every) == 0 or step + 1 == int(args.epochs):
            with torch.no_grad():
                acc = ((torch.sigmoid(logits) >= 0.5).float() == close).float().mean()
            entry = {
                "step": int(step + 1),
                "loss": float(loss.detach().cpu().item()),
                "mse": float(loss_dist.detach().cpu().item()),
                "bce": float(loss_close.detach().cpu().item()),
                "close_accuracy": float(acc.detach().cpu().item()),
            }
            history.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)

    model.eval()
    pair_head.eval()
    with torch.no_grad():
        pooled = _live_pooled_with_grad(
            model,
            selected_sequences,
            selected_names,
            windows,
            args.device,
        )
        pooled = F.normalize(pooled.float(), dim=-1)
        features = _pair_features(pooled, pair_indices)
        pred_dist, logits = pair_head(features)
        pred_dist = torch.sigmoid(pred_dist).detach().cpu()
        logits = logits.detach().cpu()
        raw_dist = torch.pdist(pooled.detach().cpu())
        raw_dist = raw_dist / raw_dist.max().clamp_min(1e-6)

    result = {
        "dataset_id": row["dataset_id"],
        "row_index": int(args.row_index),
        "subset_size": int(args.subset_size),
        "leaves": leaves,
        "sequence_names": selected_names,
        "target_tree": target_tree,
        "source": {
            "source": "live_unfrozen",
            "checkpoint": str(args.checkpoint),
            "num_windows": int(len(windows)),
            "window_size": int(args.window_size),
            "stride": int(args.stride),
            "input_mode": args.input_mode,
            "phyla_train_scope": args.phyla_train_scope,
            "trainable_phyla_params": int(trainable_count),
        },
        "num_pairs": int(len(pair_indices)),
        "close_threshold_topological_distance": close_threshold,
        "positive_rate": float(close.detach().cpu().mean().item()),
        "history": history,
        "final_pair_head_mse": float(F.mse_loss(pred_dist, norm_dist.detach().cpu()).item()),
        "final_pair_head_bce": float(F.binary_cross_entropy_with_logits(logits, close.detach().cpu()).item()),
        "final_pair_head_close_accuracy": float(
            ((torch.sigmoid(logits) >= 0.5).float() == close.detach().cpu()).float().mean().item()
        ),
        "final_raw_pooled_mse": float(F.mse_loss(raw_dist, norm_dist.detach().cpu()).item()),
    }
    for name, pred in [("final_pair_head", pred_dist), ("final_raw_pooled", raw_dist)]:
        try:
            newick = _nj_newick(leaves, pred, pair_indices)
            result[f"{name}_nj_tree"] = newick
            result[f"{name}_nj_rf_norm"] = float(calculate_norm_rf(newick, target_tree))
        except Exception as exc:
            result[f"{name}_nj_error"] = str(exc)
    return result


def _pair_features(leaf_features: torch.Tensor, pair_indices: list[tuple[int, int]]) -> torch.Tensor:
    features = []
    for i, j in pair_indices:
        left = leaf_features[i]
        right = leaf_features[j]
        features.append(torch.cat([left, right, (left - right).abs(), left * right], dim=0))
    return torch.stack(features, dim=0)


class PairHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        return out[:, 0], out[:, 1]


class MetricProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, leaf_features: torch.Tensor, pair_indices: list[tuple[int, int]]) -> torch.Tensor:
        z = self.net(leaf_features)
        preds = []
        for i, j in pair_indices:
            preds.append(torch.linalg.vector_norm(z[i] - z[j], dim=-1))
        preds = torch.stack(preds, dim=0)
        return preds / preds.max().clamp_min(1e-6)


def _nj_newick(names: list[str], distances: torch.Tensor, pair_indices: list[tuple[int, int]]) -> str:
    n = len(names)
    matrix = [[0.0 for _ in range(i + 1)] for i in range(n)]
    for value, (i, j) in zip(distances.detach().cpu().float().tolist(), pair_indices):
        a, b = max(i, j), min(i, j)
        matrix[a][b] = max(float(value), 1e-6)
    dm = DistanceMatrix(names, matrix)
    tree = DistanceTreeConstructor().nj(dm)
    handle = StringIO()
    Phylo.write(tree, handle, "newick")
    return handle.getvalue().strip()


def _train_pair_head(
    features: torch.Tensor,
    norm_dist: torch.Tensor,
    close: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    device = torch.device(args.train_device)
    features = features.to(device)
    norm_dist = norm_dist.to(device)
    close = close.to(device)
    model = PairHead(features.size(1), int(args.hidden_dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.0)
    for step in range(int(args.epochs)):
        pred_dist, logits = model(features)
        pred_dist = torch.sigmoid(pred_dist)
        loss_dist = F.mse_loss(pred_dist, norm_dist)
        loss_close = F.binary_cross_entropy_with_logits(logits, close)
        loss = loss_dist + loss_close
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred_dist, logits = model(features)
        pred_dist = torch.sigmoid(pred_dist).detach().cpu()
        logits = logits.detach().cpu()
        preds = (torch.sigmoid(logits) >= 0.5).float()
    metrics = {
        "pair_head_mse": float(F.mse_loss(pred_dist, norm_dist.cpu()).item()),
        "pair_head_bce": float(F.binary_cross_entropy_with_logits(logits, close.cpu()).item()),
        "pair_head_close_accuracy": float((preds == close.cpu()).float().mean().item()),
    }
    return metrics, pred_dist, logits


def _train_metric_projector(
    leaf_features: torch.Tensor,
    pair_indices: list[tuple[int, int]],
    norm_dist: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict, torch.Tensor]:
    device = torch.device(args.train_device)
    leaf_features = leaf_features.to(device)
    norm_dist = norm_dist.to(device)
    model = MetricProjector(
        leaf_features.size(1),
        int(args.hidden_dim),
        int(args.latent_dim),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.0)
    for _ in range(int(args.epochs)):
        pred = model(leaf_features, pair_indices)
        loss = F.mse_loss(pred, norm_dist)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(leaf_features, pair_indices).detach().cpu()
    metrics = {
        "metric_projector_mse": float(F.mse_loss(pred, norm_dist.cpu()).item()),
    }
    return metrics, pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--subset-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--positive-quantile", type=float, default=0.25)
    parser.add_argument("--source", choices=("cached", "live"), default="cached")
    parser.add_argument("--embedding-root", type=Path, default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--nexus-root", type=Path, default=DEFAULT_NEXUS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-device", default="cpu")
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument(
        "--input-mode",
        choices=("aligned-windows", "raw-full", "raw-windows"),
        default="aligned-windows",
    )
    parser.add_argument("--unfreeze-phyla", action="store_true")
    parser.add_argument(
        "--phyla-train-scope",
        choices=("all", "last-module", "tree-head"),
        default="all",
    )
    parser.add_argument("--phyla-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    row, target_tree, leaves = _select_case(args)
    if args.unfreeze_phyla:
        if args.source != "live":
            raise ValueError("--unfreeze-phyla requires --source live")
        result = _train_live_phyla_pair_head(row, target_tree, leaves, args)
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return

    if args.source == "live":
        sequence_names, pooled, chunks, source_meta = _load_live_representations(
            row,
            leaves,
            args.nexus_root,
            args.checkpoint,
            args.device,
            args.window_size,
            args.stride,
            args.max_windows,
            args.input_mode,
        )
    else:
        sequence_names, pooled, chunks, source_meta = _load_cached_representations(
            row,
            leaves,
            args.embedding_root,
        )

    pooled = F.normalize(pooled.float(), dim=-1)
    flat_chunks = F.normalize(chunks.transpose(0, 1).reshape(len(leaves), -1).float(), dim=-1)
    pair_indices, dist, norm_dist, close, close_threshold = _target_pair_data(
        target_tree,
        leaves,
        args.positive_quantile,
    )

    pooled_features = _pair_features(pooled, pair_indices)
    chunk_features = _pair_features(flat_chunks, pair_indices)
    pooled_pair_metrics, pooled_pair_dist, _ = _train_pair_head(
        pooled_features,
        norm_dist,
        close,
        args,
    )
    chunk_pair_metrics, chunk_pair_dist, _ = _train_pair_head(
        chunk_features,
        norm_dist,
        close,
        args,
    )
    pooled_metric_metrics, pooled_metric_dist = _train_metric_projector(
        pooled,
        pair_indices,
        norm_dist,
        args,
    )
    chunk_metric_metrics, chunk_metric_dist = _train_metric_projector(
        flat_chunks,
        pair_indices,
        norm_dist,
        args,
    )

    raw_pooled = torch.pdist(pooled).detach().cpu()
    raw_pooled = raw_pooled / raw_pooled.max().clamp_min(1e-6)
    raw_chunks = torch.pdist(flat_chunks).detach().cpu()
    raw_chunks = raw_chunks / raw_chunks.max().clamp_min(1e-6)

    result = {
        "dataset_id": row["dataset_id"],
        "row_index": int(args.row_index),
        "subset_size": int(args.subset_size),
        "leaves": leaves,
        "sequence_names": sequence_names,
        "target_tree": target_tree,
        "source": source_meta,
        "num_pairs": int(len(pair_indices)),
        "close_threshold_topological_distance": close_threshold,
        "positive_rate": float(close.mean().item()),
        "raw_pooled_mse": float(F.mse_loss(raw_pooled, norm_dist).item()),
        "raw_chunk_flat_mse": float(F.mse_loss(raw_chunks, norm_dist).item()),
        **{f"pooled_{k}": v for k, v in pooled_pair_metrics.items()},
        **{f"chunk_flat_{k}": v for k, v in chunk_pair_metrics.items()},
        **{f"pooled_{k}": v for k, v in pooled_metric_metrics.items()},
        **{f"chunk_flat_{k}": v for k, v in chunk_metric_metrics.items()},
    }

    for name, pred in [
        ("raw_pooled", raw_pooled),
        ("raw_chunk_flat", raw_chunks),
        ("pooled_pair_head", pooled_pair_dist),
        ("chunk_flat_pair_head", chunk_pair_dist),
        ("pooled_metric_projector", pooled_metric_dist),
        ("chunk_flat_metric_projector", chunk_metric_dist),
    ]:
        try:
            newick = _nj_newick(leaves, pred, pair_indices)
            result[f"{name}_nj_tree"] = newick
            result[f"{name}_nj_rf_norm"] = float(calculate_norm_rf(newick, target_tree))
        except Exception as exc:
            result[f"{name}_nj_error"] = str(exc)

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
