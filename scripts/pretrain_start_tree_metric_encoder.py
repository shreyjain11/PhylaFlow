#!/usr/bin/env python
"""Pretrain a start-tree topology encoder from the known random start generator.

The training target is exact split/RF geometry: generate pairs of random start
trees at the same taxon count, encode each tree as a set of internal split
bitmasks, and train an embedding whose pairwise geometry predicts normalized
RF distance.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from ete3 import Tree as EteTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.random_tree import Tree as RandomTree  # noqa: E402


def parse_sizes(raw: str) -> list[int]:
    sizes: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        range_match = re.fullmatch(r"(\d+)-(\d+)(?::(\d+))?", part)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            step = int(range_match.group(3) or 1)
            if step <= 0:
                raise ValueError(f"Invalid size step in {part!r}")
            direction = 1 if end >= start else -1
            sizes.extend(range(start, end + direction, direction * step))
        else:
            sizes.append(int(part))
    sizes = sorted({int(size) for size in sizes})
    if any(size < 4 for size in sizes):
        raise ValueError("All taxon counts must be >= 4 for internal RF splits.")
    return sizes


def canonical_internal_splits(newick: str) -> tuple[tuple[int, ...], int]:
    tree = EteTree(str(newick).strip(), format=1)
    leaves = list(tree.iter_leaves())
    leaf_names = [str(leaf.name) for leaf in leaves]
    try:
        ordered_names = sorted(leaf_names, key=lambda name: int(name))
    except ValueError:
        ordered_names = sorted(leaf_names)
    name_to_idx = {name: idx for idx, name in enumerate(ordered_names)}
    n_taxa = len(ordered_names)
    full_mask = (1 << n_taxa) - 1
    splits: set[int] = set()

    def visit(node) -> int:
        if node.is_leaf():
            return 1 << name_to_idx[str(node.name)]
        mask = 0
        for child in node.children:
            mask |= visit(child)
        if node.up is not None:
            side = mask
            other = full_mask ^ side
            if 1 < side.bit_count() < n_taxa - 1:
                splits.add(min(int(side), int(other)))
        return mask

    visit(tree)
    return tuple(sorted(splits)), n_taxa


def normalized_rf_from_splits(
    splits_a: Iterable[int],
    splits_b: Iterable[int],
    n_taxa: int,
) -> float:
    max_rf = max(2 * (int(n_taxa) - 3), 1)
    rf = len(set(int(x) for x in splits_a) ^ set(int(x) for x in splits_b))
    return float(rf) / float(max_rf)


def sample_start_tree(n_taxa: int) -> str:
    return str(RandomTree(num_leaves=int(n_taxa), random=True))


def swap_leaf_labels(newick: str, *, num_swaps: int) -> str:
    tree = EteTree(str(newick).strip(), format=1)
    leaves = list(tree.iter_leaves())
    if len(leaves) < 2:
        return tree.write(format=1)
    for _ in range(max(1, int(num_swaps))):
        a, b = random.sample(leaves, 2)
        a.name, b.name = b.name, a.name
    return tree.write(format=1)


def mask_to_bits(mask: int, max_bits: int) -> list[float]:
    mask = int(mask)
    if mask.bit_length() > int(max_bits):
        raise ValueError(
            f"Split mask needs {mask.bit_length()} bits but max_bits={max_bits}."
        )
    return [float((mask >> bit) & 1) for bit in range(int(max_bits))]


def build_tree_tensor_batch(
    split_sets: list[tuple[int, ...]],
    n_taxa: list[int],
    *,
    max_bits: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_splits = max(1, max((len(splits) for splits in split_sets), default=0))
    split_bits = torch.zeros(
        len(split_sets),
        max_splits,
        int(max_bits),
        dtype=torch.float32,
        device=device,
    )
    pad_mask = torch.ones(
        len(split_sets),
        max_splits,
        dtype=torch.bool,
        device=device,
    )
    size_features = torch.zeros(len(split_sets), 2, dtype=torch.float32, device=device)
    max_bits_float = float(max_bits)
    log_den = math.log(max(float(max_bits), 2.0))

    for row_idx, splits in enumerate(split_sets):
        n = int(n_taxa[row_idx])
        size_features[row_idx, 0] = float(n) / max_bits_float
        size_features[row_idx, 1] = math.log(max(float(n), 2.0)) / log_den
        for split_idx, split in enumerate(splits):
            split_bits[row_idx, split_idx] = torch.tensor(
                mask_to_bits(int(split), max_bits),
                dtype=torch.float32,
                device=device,
            )
            pad_mask[row_idx, split_idx] = False
    return split_bits, pad_mask, size_features


class SplitSetEncoder(nn.Module):
    def __init__(
        self,
        *,
        max_bits: int,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_bits = int(max_bits)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.split_encoder = nn.Sequential(
            nn.LayerNorm(self.max_bits),
            nn.Linear(self.max_bits, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.size_encoder = nn.Sequential(
            nn.LayerNorm(2),
            nn.Linear(2, self.hidden_dim),
            nn.GELU(),
        )
        self.tree_encoder = nn.Sequential(
            nn.LayerNorm(4 * self.hidden_dim),
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.embedding_dim),
        )
        self.embedding_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
        self,
        split_bits: torch.Tensor,
        pad_mask: torch.Tensor,
        size_features: torch.Tensor,
    ) -> torch.Tensor:
        split_h = self.split_encoder(split_bits)
        valid = (~pad_mask).unsqueeze(-1)
        split_h = split_h.masked_fill(~valid, 0.0)
        counts = valid.sum(dim=1).clamp_min(1).to(dtype=split_h.dtype)
        pooled_sum = split_h.sum(dim=1)
        pooled_mean = pooled_sum / counts
        pooled_max = split_h.masked_fill(~valid, -torch.inf).max(dim=1).values
        pooled_max = torch.where(
            torch.isfinite(pooled_max),
            pooled_max,
            torch.zeros_like(pooled_max),
        )
        size_h = self.size_encoder(size_features)
        emb = self.tree_encoder(
            torch.cat([pooled_sum, pooled_mean, pooled_max, size_h], dim=-1)
        )
        return self.embedding_norm(emb)


class PairMetricModel(nn.Module):
    def __init__(
        self,
        *,
        max_bits: int,
        hidden_dim: int,
        embedding_dim: int,
        num_bins: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = SplitSetEncoder(
            max_bits=max_bits,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        pair_dim = 4 * int(embedding_dim)
        self.distance_head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )
        self.bin_head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, int(num_bins)),
        )

    def encode(
        self,
        split_bits: torch.Tensor,
        pad_mask: torch.Tensor,
        size_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(split_bits, pad_mask, size_features)

    def pair_outputs(self, z_a: torch.Tensor, z_b: torch.Tensor) -> dict[str, torch.Tensor]:
        pair = torch.cat([z_a, z_b, torch.abs(z_a - z_b), z_a * z_b], dim=-1)
        dist = torch.sigmoid(self.distance_head(pair)).squeeze(-1)
        bins = self.bin_head(pair)
        sim = (F.normalize(z_a, dim=-1) * F.normalize(z_b, dim=-1)).sum(dim=-1)
        return {"dist": dist, "bins": bins, "sim": sim}


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    n, m = matrix.shape
    if n != m:
        raise ValueError("off_diagonal expects a square matrix")
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(z: torch.Tensor, *, covariance_weight: float) -> torch.Tensor:
    if z.shape[0] <= 1:
        return z.new_tensor(0.0)
    centered = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + 1e-4)
    variance_loss = F.relu(1.0 - std).mean()
    cov = centered.T @ centered / max(z.shape[0] - 1, 1)
    covariance_loss = off_diagonal(cov).pow(2).mean()
    return variance_loss + float(covariance_weight) * covariance_loss


def embedding_stats(embeddings: torch.Tensor) -> dict[str, float]:
    emb = embeddings.detach().float().cpu()
    centered = emb - emb.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    variance = singular.square()
    variance = variance / variance.sum().clamp_min(1e-12)
    normalized = F.normalize(emb, dim=-1)
    cosine = normalized @ normalized.T
    offdiag_mask = ~torch.eye(cosine.shape[0], dtype=torch.bool)
    offdiag = cosine[offdiag_mask]
    return {
        "norm_mean": float(emb.norm(dim=-1).mean().item()),
        "effective_rank": float(
            torch.exp(-(variance * variance.clamp_min(1e-12).log()).sum()).item()
        ),
        "top_pc_variance": float(variance[0].item()) if variance.numel() else 0.0,
        "offdiag_cosine_mean": float(offdiag.mean().item()) if offdiag.numel() else 0.0,
        "offdiag_cosine_min": float(offdiag.min().item()) if offdiag.numel() else 0.0,
        "offdiag_cosine_max": float(offdiag.max().item()) if offdiag.numel() else 0.0,
    }


def make_pair_batch(
    *,
    batch_size: int,
    sizes: list[int],
    max_label_swaps: int,
    same_pair_prob: float,
    label_swap_pair_prob: float,
) -> dict[str, object]:
    split_sets: list[tuple[int, ...]] = []
    n_taxa_rows: list[int] = []
    distances: list[float] = []
    modes: list[str] = []

    for _ in range(int(batch_size)):
        n_taxa = int(random.choice(sizes))
        tree_a = sample_start_tree(n_taxa)
        roll = random.random()
        if roll < float(same_pair_prob):
            tree_b = tree_a
            mode = "same"
        elif roll < float(same_pair_prob) + float(label_swap_pair_prob):
            tree_b = swap_leaf_labels(
                tree_a,
                num_swaps=random.randint(1, max(1, int(max_label_swaps))),
            )
            mode = "label_swap"
        else:
            tree_b = sample_start_tree(n_taxa)
            mode = "independent"

        splits_a, inferred_a = canonical_internal_splits(tree_a)
        splits_b, inferred_b = canonical_internal_splits(tree_b)
        if inferred_a != n_taxa or inferred_b != n_taxa:
            raise RuntimeError(
                f"Taxon count mismatch: requested {n_taxa}, got {inferred_a}/{inferred_b}"
            )
        distances.append(normalized_rf_from_splits(splits_a, splits_b, n_taxa))
        split_sets.extend([splits_a, splits_b])
        n_taxa_rows.extend([n_taxa, n_taxa])
        modes.append(mode)

    return {
        "split_sets": split_sets,
        "n_taxa": n_taxa_rows,
        "distances": distances,
        "modes": modes,
    }


def distance_bins(distances: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    return torch.bucketize(distances, thresholds.to(distances.device))


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float()
    b = b.detach().float()
    if a.numel() < 2:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float((a @ b / denom).item())


def case_index_from_path(path: str, fallback: int) -> int:
    match = re.search(r"case(\d+)", str(path))
    if match is None:
        return int(fallback)
    return int(match.group(1))


def load_export_trees(patterns: list[str]) -> list[dict[str, object]]:
    paths: list[str] = []
    for pattern in patterns:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            paths.extend(expanded)
        elif Path(pattern).is_file():
            paths.append(pattern)
    rows = []
    for fallback, path in enumerate(sorted(set(paths))):
        payload = json.loads(Path(path).read_text())
        tree = payload.get("tree") if isinstance(payload, dict) else payload
        if not tree:
            continue
        rows.append(
            {
                "index": case_index_from_path(path, fallback),
                "path": str(path),
                "tree": str(tree),
                "group_key": payload.get("group_key") if isinstance(payload, dict) else None,
            }
        )
    rows.sort(key=lambda row: int(row["index"]))
    return rows


@torch.no_grad()
def encode_export_rows(
    model: PairMetricModel,
    rows: list[dict[str, object]],
    *,
    max_bits: int,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    embeddings: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(rows), int(chunk_size)):
        chunk = rows[start : start + int(chunk_size)]
        split_sets: list[tuple[int, ...]] = []
        n_taxa: list[int] = []
        for row in chunk:
            splits, n = canonical_internal_splits(str(row["tree"]))
            split_sets.append(splits)
            n_taxa.append(n)
        split_bits, pad_mask, size_features = build_tree_tensor_batch(
            split_sets,
            n_taxa,
            max_bits=max_bits,
            device=device,
        )
        emb = model.encode(split_bits, pad_mask, size_features)
        embeddings.append(F.normalize(emb, dim=-1).detach().cpu())
    return torch.cat(embeddings, dim=0) if embeddings else torch.zeros(0, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes", default="8,16,32,50")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--max-bits", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dist-loss-weight", type=float, default=1.0)
    parser.add_argument("--bin-loss-weight", type=float, default=0.25)
    parser.add_argument("--vicreg-weight", type=float, default=0.02)
    parser.add_argument("--vicreg-covariance-weight", type=float, default=1.0)
    parser.add_argument("--same-pair-prob", type=float, default=0.10)
    parser.add_argument("--label-swap-pair-prob", type=float, default=0.40)
    parser.add_argument("--max-label-swaps", type=int, default=3)
    parser.add_argument("--bin-thresholds", default="0.05,0.15,0.30,0.50,0.75")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--export-start-json-glob", action="append", default=[])
    parser.add_argument("--export-start-table", default=None)
    parser.add_argument("--export-chunk-size", type=int, default=128)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    sizes = parse_sizes(args.sizes)
    thresholds = torch.tensor(
        [float(x) for x in str(args.bin_thresholds).split(",") if x.strip()],
        dtype=torch.float32,
    )
    if thresholds.ndim != 1 or thresholds.numel() == 0:
        raise ValueError("--bin-thresholds must define at least one threshold")
    num_bins = int(thresholds.numel()) + 1

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = PairMetricModel(
        max_bits=args.max_bits,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_bins=num_bins,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    final_metrics: dict[str, float] = {}
    mode_counts = {"same": 0, "label_swap": 0, "independent": 0}

    for step in range(1, int(args.steps) + 1):
        batch = make_pair_batch(
            batch_size=args.batch_size,
            sizes=sizes,
            max_label_swaps=args.max_label_swaps,
            same_pair_prob=args.same_pair_prob,
            label_swap_pair_prob=args.label_swap_pair_prob,
        )
        for mode in batch["modes"]:
            mode_counts[str(mode)] = mode_counts.get(str(mode), 0) + 1

        split_bits, pad_mask, size_features = build_tree_tensor_batch(
            batch["split_sets"],
            batch["n_taxa"],
            max_bits=args.max_bits,
            device=device,
        )
        distances = torch.tensor(
            batch["distances"],
            dtype=torch.float32,
            device=device,
        )

        z_all = model.encode(split_bits, pad_mask, size_features)
        z_a = z_all[0::2]
        z_b = z_all[1::2]
        outputs = model.pair_outputs(z_a, z_b)

        sim_target = torch.exp(-distances / float(args.tau))
        bins_target = distance_bins(distances, thresholds)
        sim_loss = F.mse_loss(outputs["sim"], sim_target)
        dist_loss = F.mse_loss(outputs["dist"], distances)
        bin_loss = F.cross_entropy(outputs["bins"], bins_target)
        reg_loss = vicreg_loss(
            z_all,
            covariance_weight=float(args.vicreg_covariance_weight),
        )
        loss = (
            sim_loss
            + float(args.dist_loss_weight) * dist_loss
            + float(args.bin_loss_weight) * bin_loss
            + float(args.vicreg_weight) * reg_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            bin_acc = (outputs["bins"].argmax(dim=-1) == bins_target).float().mean()
            final_metrics = {
                "loss": float(loss.item()),
                "sim_loss": float(sim_loss.item()),
                "dist_loss": float(dist_loss.item()),
                "bin_loss": float(bin_loss.item()),
                "vicreg_loss": float(reg_loss.item()),
                "bin_acc": float(bin_acc.item()),
                "distance_mean": float(distances.mean().item()),
                "distance_min": float(distances.min().item()),
                "distance_max": float(distances.max().item()),
                "pred_distance_mean": float(outputs["dist"].mean().item()),
                "cosine_mean": float(outputs["sim"].mean().item()),
                "distance_pearson": pearson_corr(outputs["dist"], distances),
                "cosine_target_pearson": pearson_corr(outputs["sim"], sim_target),
            }

        if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
            print(
                " ".join(
                    [
                        f"step={step}",
                        f"loss={final_metrics['loss']:.5f}",
                        f"dist={final_metrics['dist_loss']:.5f}",
                        f"sim={final_metrics['sim_loss']:.5f}",
                        f"bin_acc={final_metrics['bin_acc']:.3f}",
                        f"corr_d={final_metrics['distance_pearson']:.3f}",
                        f"d_mean={final_metrics['distance_mean']:.3f}",
                    ]
                ),
                flush=True,
            )

    metadata = {
        "sizes": sizes,
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "max_bits": int(args.max_bits),
        "hidden_dim": int(args.hidden_dim),
        "embedding_dim": int(args.embedding_dim),
        "tau": float(args.tau),
        "bin_thresholds": [float(x) for x in thresholds.tolist()],
        "same_pair_prob": float(args.same_pair_prob),
        "label_swap_pair_prob": float(args.label_swap_pair_prob),
        "max_label_swaps": int(args.max_label_swaps),
        "seed": int(args.seed),
        "mode_counts": mode_counts,
        "final_metrics": final_metrics,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "metadata": metadata,
        },
        output,
    )
    print(f"saved_checkpoint={output}")

    if args.export_start_json_glob or args.export_start_table:
        if not args.export_start_table:
            raise ValueError("--export-start-table is required when exporting start JSONs")
        rows = load_export_trees(args.export_start_json_glob)
        if not rows:
            raise ValueError(
                "No start trees matched --export-start-json-glob; nothing to export."
            )
        embeddings = encode_export_rows(
            model,
            rows,
            max_bits=args.max_bits,
            device=device,
            chunk_size=args.export_chunk_size,
        )
        export_metadata = {
            "source_checkpoint": str(output.resolve()),
            "num_cases": len(rows),
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "start_tree_hashes": [
                hashlib.sha1(str(row["tree"]).encode("utf-8")).hexdigest()
                for row in rows
            ],
            "case_indices": [int(row["index"]) for row in rows],
            "paths": [str(row["path"]) for row in rows],
            "embedding_stats": embedding_stats(embeddings),
        }
        export_path = Path(args.export_start_table)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeddings": embeddings,
                "metadata": export_metadata,
            },
            export_path,
        )
        print(f"saved_start_table={export_path}")
        print(json.dumps(export_metadata["embedding_stats"], sort_keys=True))


if __name__ == "__main__":
    main()
