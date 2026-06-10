#!/usr/bin/env python
import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.treeTokenizer import TreeFeatureTokenizer


def _case_index(group_key, fallback):
    if group_key is None:
        return int(fallback)
    match = re.search(r"case(\d+)$", str(group_key))
    if match is None:
        return int(fallback)
    return int(match.group(1))


def _load_start_cases(config):
    paths = config["data"].get("overfit_fixed_pair_start_tree_json_paths")
    if not paths:
        raise ValueError("Config does not define overfit_fixed_pair_start_tree_json_paths")
    rows = []
    for fallback_idx, path in enumerate(paths):
        with open(path, "r") as handle:
            payload = json.load(handle)
        tree = payload["tree"] if isinstance(payload, dict) else payload
        group_key = payload.get("group_key") if isinstance(payload, dict) else None
        rows.append(
            {
                "index": _case_index(group_key, fallback_idx),
                "group_key": group_key or f"case{fallback_idx:02d}",
                "tree": str(tree),
                "path": str(path),
            }
        )
    rows.sort(key=lambda row: row["index"])
    expected = list(range(len(rows)))
    found = [int(row["index"]) for row in rows]
    if found != expected:
        raise ValueError(f"Case indices are not contiguous: expected {expected[:5]}..., got {found[:5]}...")
    return rows


def _make_tokenizer(config):
    model_cfg = config["model"]
    return TreeFeatureTokenizer(
        num_node_types=model_cfg["num_node_types"],
        num_edge_types=model_cfg["num_edge_types"],
        hidden_dim=model_cfg["embed_dim"],
        n_layers=model_cfg.get("tokenizer_n_layers", 4),
        lap_dim=model_cfg.get("tokenizer_lap_dim", 8),
        lap_dropout=model_cfg.get("tokenizer_lap_dropout", 0.0),
        branch_length_mode=model_cfg.get("tokenizer_branch_length_mode", "linear"),
        branch_length_num_buckets=model_cfg.get("tokenizer_branch_length_num_buckets", 64),
        branch_length_log_min=model_cfg.get("tokenizer_branch_length_log_min", -8.0),
        branch_length_log_max=model_cfg.get("tokenizer_branch_length_log_max", 1.0),
    )


def _split_masks_for_tree(tokenizer, tree):
    tokenized = tokenizer([tree])
    raw_masks = tokenized[-1][0]
    return [int(mask) for mask in raw_masks if int(mask) != 0]


def _mask_bits(mask, max_bits):
    return [(int(mask) >> bit) & 1 for bit in range(max_bits)]


def _build_split_bit_batch(rows, tokenizer, max_bits):
    masks_by_case = [_split_masks_for_tree(tokenizer, row["tree"]) for row in rows]
    max_splits = max(max(len(masks) for masks in masks_by_case), 1)
    bits = torch.zeros(len(rows), max_splits, max_bits, dtype=torch.float32)
    pad_mask = torch.ones(len(rows), max_splits, dtype=torch.bool)
    for row_idx, masks in enumerate(masks_by_case):
        for split_idx, mask in enumerate(masks):
            bits[row_idx, split_idx] = torch.tensor(
                _mask_bits(mask, max_bits),
                dtype=torch.float32,
            )
            pad_mask[row_idx, split_idx] = False
    return bits, pad_mask, masks_by_case


class StartCaseProbe(nn.Module):
    def __init__(self, *, max_bits, hidden_dim, embedding_dim, num_cases, dropout):
        super().__init__()
        self.split_encoder = nn.Sequential(
            nn.LayerNorm(max_bits),
            nn.Linear(max_bits, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.tree_encoder = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_cases)

    def encode(self, split_bits, pad_mask):
        split_h = self.split_encoder(split_bits)
        valid = (~pad_mask).unsqueeze(-1)
        split_h = split_h.masked_fill(~valid, 0.0)
        counts = valid.sum(dim=1).clamp_min(1).to(dtype=split_h.dtype)
        pooled_sum = split_h.sum(dim=1)
        pooled_mean = pooled_sum / counts
        split_h_for_max = split_h.masked_fill(~valid, -torch.inf)
        pooled_max = split_h_for_max.max(dim=1).values
        pooled_max = torch.where(
            torch.isfinite(pooled_max),
            pooled_max,
            torch.zeros_like(pooled_max),
        )
        embedding = self.tree_encoder(torch.cat([pooled_sum, pooled_mean, pooled_max], dim=-1))
        return self.embedding_norm(embedding)

    def forward(self, split_bits, pad_mask):
        return self.classifier(self.encode(split_bits, pad_mask))


def _embedding_stats(embeddings):
    emb = embeddings.detach().float()
    centered = emb - emb.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    variance = singular.square()
    variance = variance / variance.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(variance * variance.clamp_min(1e-12).log()).sum())
    normalized = F.normalize(emb, dim=-1)
    cosine = normalized @ normalized.T
    offdiag = cosine[~torch.eye(cosine.shape[0], dtype=torch.bool)]
    return {
        "norm_mean": float(emb.norm(dim=-1).mean().item()),
        "offdiag_cosine_mean": float(offdiag.mean().item()),
        "offdiag_cosine_min": float(offdiag.min().item()),
        "offdiag_cosine_max": float(offdiag.max().item()),
        "effective_rank": float(effective_rank.item()),
        "top_pc_variance": float(variance[0].item()) if variance.numel() else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--min-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--max-bits", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
    rows = _load_start_cases(config)
    tokenizer = _make_tokenizer(config)
    split_bits, pad_mask, masks_by_case = _build_split_bit_batch(
        rows,
        tokenizer,
        args.max_bits,
    )
    labels = torch.arange(len(rows), dtype=torch.long)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    split_bits = split_bits.to(device)
    pad_mask = pad_mask.to(device)
    labels = labels.to(device)

    model = StartCaseProbe(
        max_bits=args.max_bits,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_cases=len(rows),
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    final_loss = math.inf
    final_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(split_bits, pad_mask)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            final_acc = float((pred == labels).float().mean().item())
            final_loss = float(loss.item())
        if epoch % 100 == 0 or epoch == 1:
            print(f"epoch={epoch} loss={final_loss:.6f} acc={final_acc:.4f}", flush=True)
        if epoch >= args.min_epochs and final_acc >= 1.0 and final_loss <= 1e-4:
            print(f"early_stop epoch={epoch} loss={final_loss:.6f} acc={final_acc:.4f}", flush=True)
            break

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(split_bits, pad_mask).detach().cpu()
        logits = model(split_bits, pad_mask).detach().cpu()
        preds = logits.argmax(dim=-1)
        accuracy = float((preds == torch.arange(len(rows))).float().mean().item())

    metadata = {
        "config": str(Path(args.config).resolve()),
        "num_cases": len(rows),
        "max_bits": args.max_bits,
        "hidden_dim": args.hidden_dim,
        "embedding_dim": args.embedding_dim,
        "epochs_requested": args.epochs,
        "final_loss": final_loss,
        "accuracy": accuracy,
        "case_keys": [row["group_key"] for row in rows],
        "start_tree_hashes": [
            hashlib.sha1(row["tree"].encode("utf-8")).hexdigest() for row in rows
        ],
        "num_splits_by_case": [len(masks) for masks in masks_by_case],
        "embedding_stats": _embedding_stats(embeddings),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings,
            "logits": logits,
            "state_dict": model.state_dict(),
            "metadata": metadata,
        },
        output,
    )
    print(f"saved={output}")
    print(json.dumps(metadata["embedding_stats"], sort_keys=True))
    print(f"accuracy={accuracy:.6f} loss={final_loss:.6f}")


if __name__ == "__main__":
    main()
