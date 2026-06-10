#!/usr/bin/env python3
"""Probe whether frozen Phyla embeddings predict heldout 10-leaf topology."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ete3 import Tree as EteTree


def _read_jsonl(path: Path, limit: int | None = None) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def _load_tree_payload(path: str) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tree = str(
        payload.get("topology_key")
        or payload.get("tree")
        or payload.get("newick")
        or ""
    ).strip()
    if not tree:
        raise ValueError(f"No tree/newick field in {path}")
    return tree if tree.endswith(";") else f"{tree};"


def _numeric_leaf_key(name: str) -> Tuple[int, str]:
    try:
        return (0, f"{int(str(name)):020d}")
    except Exception:
        return (1, str(name))


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _pruned_target_tree(row: dict, subset_size: int, seed: int) -> Tuple[str, List[str]]:
    tree = EteTree(_load_tree_payload(row["target_path"]), format=1)
    leaves = [str(leaf.name) for leaf in tree.iter_leaves()]
    leaves = sorted(leaves, key=_numeric_leaf_key)
    rng = random.Random(_stable_seed(seed, row.get("dataset_id"), row.get("target_path")))
    keep = sorted(rng.sample(leaves, int(subset_size)), key=_numeric_leaf_key)
    tree.prune(keep, preserve_branch_length=True)
    return tree.write(format=1), keep


class EmbeddingCache:
    def __init__(self, root: Path):
        self.root = root
        self.cache: Dict[str, Tuple[List[str], torch.Tensor]] = {}

    def _path_for_dataset(self, dataset_id: str) -> Path:
        dataset_id = str(dataset_id).upper()
        suffixes = (
            "_phyla_beta_sitechunk_w256_s256_embeddings.pt",
            "_phyla_beta_embeddings.pt",
            "_embeddings.pt",
        )
        for suffix in suffixes:
            candidate = self.root / f"{dataset_id}{suffix}"
            if candidate.exists():
                return candidate
        matches = sorted(self.root.glob(f"{dataset_id}*_embeddings.pt"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"No embedding file found for {dataset_id} in {self.root}")

    def get(self, dataset_id: str) -> Tuple[List[str], torch.Tensor]:
        dataset_id = str(dataset_id).upper()
        cached = self.cache.get(dataset_id)
        if cached is not None:
            return cached
        payload = torch.load(self._path_for_dataset(dataset_id), map_location="cpu")
        names = payload.get("sequence_names") or payload.get("names")
        embeddings = payload.get("embeddings")
        if embeddings is None:
            embeddings = payload.get("phyla_embeddings")
        if names is None or embeddings is None:
            raise ValueError(f"Bad embedding payload for {dataset_id}")
        tensor = embeddings.detach().cpu().float() if torch.is_tensor(embeddings) else torch.as_tensor(embeddings).float()
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        tensor = F.normalize(tensor, dim=-1)
        result = ([str(name) for name in names], tensor)
        self.cache[dataset_id] = result
        return result


def _embedding_for_leaf(
    cache: EmbeddingCache,
    dataset_id: str,
    leaf_name: str,
) -> torch.Tensor:
    names, tensor = cache.get(dataset_id)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    if str(leaf_name) in name_to_idx:
        return tensor[name_to_idx[str(leaf_name)]]
    idx = int(str(leaf_name))
    if 1 <= idx <= tensor.size(0):
        return tensor[idx - 1]
    if 0 <= idx < tensor.size(0):
        return tensor[idx]
    raise IndexError(f"Leaf {leaf_name} out of range for {dataset_id} ({tensor.size(0)} rows)")


def _pair_distances(tree_newick: str, leaves: List[str]) -> Dict[Tuple[str, str], float]:
    tree = EteTree(tree_newick, format=1)
    leaf_nodes = {str(leaf.name): leaf for leaf in tree.iter_leaves()}
    distances = {}
    for i, left in enumerate(leaves):
        for right in leaves[i + 1 :]:
            distances[(left, right)] = float(
                leaf_nodes[left].get_distance(leaf_nodes[right], topology_only=True)
            )
    return distances


def _row_pair_examples(
    row: dict,
    cache: EmbeddingCache,
    subset_size: int,
    seed: int,
    positive_quantile: float,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    tree_newick, leaves = _pruned_target_tree(row, subset_size=subset_size, seed=seed)
    embeddings = {
        leaf: _embedding_for_leaf(cache, row["dataset_id"], leaf)
        for leaf in leaves
    }
    distances = _pair_distances(tree_newick, leaves)
    values = np.asarray(list(distances.values()), dtype=np.float64)
    threshold = float(np.quantile(values, float(positive_quantile), method="higher"))
    features = []
    labels = []
    for (left, right), distance in distances.items():
        e_left = embeddings[left]
        e_right = embeddings[right]
        features.append(
            torch.cat(
                [
                    e_left,
                    e_right,
                    torch.abs(e_left - e_right),
                    e_left * e_right,
                ],
                dim=0,
            )
        )
        labels.append(1.0 if float(distance) <= threshold else 0.0)
    return torch.stack(features, dim=0), torch.tensor(labels, dtype=torch.float32), len(leaves)


def _build_examples(
    rows: List[dict],
    cache: EmbeddingCache,
    subset_size: int,
    seed: int,
    positive_quantile: float,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    xs = []
    ys = []
    skipped = 0
    for row in rows:
        try:
            x, y, _n = _row_pair_examples(
                row,
                cache,
                subset_size=subset_size,
                seed=seed,
                positive_quantile=positive_quantile,
            )
            xs.append(x)
            ys.append(y)
        except Exception:
            skipped += 1
    if not xs:
        raise RuntimeError("No examples built")
    y_all = torch.cat(ys, dim=0)
    stats = {
        "rows": len(rows),
        "skipped_rows": skipped,
        "pair_examples": int(y_all.numel()),
        "positive_rate": float(y_all.mean().item()),
    }
    return torch.cat(xs, dim=0), y_all, stats


class PairProbe(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(len(labels)) + 1.0)
    return float((precision * sorted_labels).sum() / n_pos)


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    scores = torch.sigmoid(logits).detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    preds = (scores >= 0.5).astype(np.float32)
    return {
        "loss": float(F.binary_cross_entropy_with_logits(logits.cpu(), labels.cpu()).item()),
        "auc": _roc_auc(scores, y),
        "average_precision": _average_precision(scores, y),
        "accuracy": float((preds == y).mean()),
        "positive_rate": float(y.mean()),
        "score_mean": float(scores.mean()),
    }


def _train_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: torch.Tensor,
    y_eval: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> Tuple[PairProbe, dict]:
    model = PairProbe(x_train.size(1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_eval_device = x_eval.to(device)
    n = y_train.numel()
    pos_weight = ((n - y_train.sum()) / y_train.sum().clamp_min(1.0)).detach()
    best = None
    best_state = None
    generator = torch.Generator(device="cpu").manual_seed(1234)
    for epoch in range(int(epochs)):
        model.train()
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, int(batch_size)):
            idx = perm[start : start + int(batch_size)].to(device)
            logits = model(x_train.index_select(0, idx))
            loss = F.binary_cross_entropy_with_logits(
                logits,
                y_train.index_select(0, idx),
                pos_weight=pos_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            eval_metrics = _metrics(model(x_eval_device).cpu(), y_eval)
        if best is None or eval_metrics["auc"] > best["auc"]:
            best = {"epoch": epoch + 1, **eval_metrics}
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--eval-index", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--train-rows", type=int, default=2000)
    parser.add_argument("--eval-rows", type=int, default=400)
    parser.add_argument("--subset-size", type=int, default=10)
    parser.add_argument("--positive-quantile", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache = EmbeddingCache(args.embedding_root)
    train_rows = _read_jsonl(args.train_index, limit=args.train_rows)
    eval_rows = _read_jsonl(args.eval_index, limit=args.eval_rows)
    x_train, y_train, train_stats = _build_examples(
        train_rows,
        cache,
        subset_size=args.subset_size,
        seed=args.seed,
        positive_quantile=args.positive_quantile,
    )
    x_eval, y_eval, eval_stats = _build_examples(
        eval_rows,
        cache,
        subset_size=args.subset_size,
        seed=args.seed + 17,
        positive_quantile=args.positive_quantile,
    )

    model, best_eval = _train_probe(
        x_train,
        y_train,
        x_eval,
        y_eval,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    model.eval()
    with torch.no_grad():
        train_metrics = _metrics(model(x_train.to(args.device)).cpu(), y_train)
        eval_metrics = _metrics(model(x_eval.to(args.device)).cpu(), y_eval)

    rng = np.random.default_rng(args.seed)
    random_scores = rng.normal(size=int(y_eval.numel()))
    random_metrics = {
        "auc": _roc_auc(random_scores, y_eval.numpy()),
        "average_precision": _average_precision(random_scores, y_eval.numpy()),
    }
    result = {
        "train_stats": train_stats,
        "eval_stats": eval_stats,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "best_eval": best_eval,
        "random_score_eval": random_metrics,
        "config": {
            "subset_size": args.subset_size,
            "positive_quantile": args.positive_quantile,
            "epochs": args.epochs,
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
        },
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
