import argparse
import copy
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml

from model.model import return_model
from run.TrainingModule import (
    _branch_relax_entries_for_tree,
    _build_branch_relax_samples_for_module,
    _move_tokenized_batch_to_device,
)
from utils.bhv_movie import build_tree_from_splits
from scripts.jc_likelihood import GenericJCLikelihood


DEFAULT_BASE_CONFIG = (
    str(REPO_ROOT / "configs" / "final_release.yaml")
)
DEFAULT_START_TREES = (
    "${PHYLAFLOW_DATA_ROOT}/branch_relaxer/train_start_trees.txt"
)
DEFAULT_TARGET_TREES = (
    "${PHYLAFLOW_DATA_ROOT}/branch_relaxer/train_warmed_trees.txt"
)
DEFAULT_OUT_DIR = (
    "./outputs/branch_relaxer"
)


class BranchDeltaHead(nn.Module):
    def __init__(self, edge_dim, *, hidden_dim=128, case_dim=0, num_cases=0):
        super().__init__()
        self.case_dim = int(case_dim)
        self.case_embedding = None
        if self.case_dim > 0:
            self.case_embedding = nn.Embedding(max(1, int(num_cases)), self.case_dim)
        input_dim = int(edge_dim) + 3 + self.case_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, edge_features, numeric_features, case_indices=None):
        parts = [edge_features, numeric_features]
        if self.case_embedding is not None:
            if case_indices is None:
                case_indices = torch.zeros(
                    edge_features.shape[0],
                    dtype=torch.long,
                    device=edge_features.device,
                )
            case_indices = torch.clamp(
                case_indices.to(device=edge_features.device, dtype=torch.long),
                min=0,
                max=self.case_embedding.num_embeddings - 1,
            )
            parts.append(self.case_embedding(case_indices))
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)


class StandaloneRelaxer(nn.Module):
    def __init__(self, model, head):
        super().__init__()
        self.model = model
        self.head = head

    def forward_batch(self, tokenized, newicks, samples, device, phyla_bank=None):
        tokenized = _move_tokenized_batch_to_device(tokenized, device)
        phyla_embeddings = None
        if phyla_bank:
            phyla_embeddings = [
                phyla_bank[str(sample["dataset_id"]).upper()]
                for sample in samples
            ]
        edge_outputs = self.model(
            tokenized,
            torch.full((len(newicks),), 4.0, dtype=torch.float32, device=device),
            phyla_embeddings=phyla_embeddings,
            return_leafs_only=False,
            return_edges_only=True,
            return_edge_features=True,
        )
        _edge_values, _edge_pad_mask, edge_features = edge_outputs
        edge_split_masks = tokenized[-1]

        preds = []
        labels = []
        for batch_idx, sample in enumerate(samples):
            entries, _lengths, _n_leaves, _mapping = _branch_relax_entries_for_tree(
                self,
                sample["newick_tree"],
                edge_split_masks[batch_idx],
                labels=sample["labels"],
            )
            if not entries:
                continue
            feature_block = torch.stack(
                [edge_features[batch_idx, entry["edge_index"]] for entry in entries],
                dim=0,
            )
            numeric = torch.tensor(
                [entry["numeric"] for entry in entries],
                dtype=torch.float32,
                device=device,
            )
            case_indices = torch.full(
                (len(entries),),
                int(sample["case_index"]),
                dtype=torch.long,
                device=device,
            )
            preds.append(self.head(feature_block, numeric, case_indices))
            labels.append(
                torch.tensor(
                    [float(entry["label"]) for entry in entries],
                    dtype=torch.float32,
                    device=device,
                )
            )
        if not preds:
            return None, None
        return torch.cat(preds), torch.cat(labels)


def _small_model_config(base_config_path, args):
    with open(base_config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg = copy.deepcopy(cfg)
    model_cfg = cfg["model"]
    model_cfg["embed_dim"] = int(args.embed_dim)
    model_cfg["hidden_dim"] = int(args.embed_dim)
    model_cfg["n_layers"] = int(args.n_layers)
    model_cfg["n_heads"] = int(args.n_heads)
    model_cfg["dropout"] = float(args.dropout)
    model_cfg["attention_dropout"] = float(args.dropout)
    model_cfg["activation_dropout"] = float(args.dropout)
    model_cfg["drop_path_rate"] = 0.0
    model_cfg["phyla_dim"] = int(args.phyla_dim)
    model_cfg["phyla_use_leaf_tokens"] = bool(args.phyla_use_leaf_tokens)
    model_cfg["phyla_use_split_tokens"] = bool(args.phyla_use_split_tokens)
    model_cfg["use_performer"] = False
    model_cfg["autoregressive_use_case_conditioning"] = False
    model_cfg["autoregressive_use_start_topology_conditioning"] = False
    model_cfg["autoregressive_frozen_start_case_embedding_path"] = None
    model_cfg["first_hit_head_mode"] = "base"
    model_cfg["first_hit_head_num_cases"] = None
    model_cfg["first_hit_frozen_start_case_embedding_path"] = None
    return cfg


def _parse_dataset_spec(raw):
    parts = str(raw).split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            "dataset specs must be DATASET_ID:START_TREES:TARGET_TREES; "
            f"got {raw!r}"
        )
    dataset_id, start_trees, target_trees = parts
    return str(dataset_id).upper(), start_trees, target_trees


def _parse_eval_spec(raw):
    parts = str(raw).split(":", 3)
    if len(parts) != 4:
        raise ValueError(
            "eval specs must be NAME:DATASET_ID:START_TREES:TARGET_TREES; "
            f"got {raw!r}"
        )
    eval_name, dataset_id, start_trees, target_trees = parts
    return str(eval_name), str(dataset_id).upper(), start_trees, target_trees


def _load_phyla_embeddings(path, device):
    payload = torch.load(path, map_location="cpu")
    embeddings = payload.get("embeddings")
    if embeddings is None:
        embeddings = payload.get("phyla_embeddings")
    if embeddings is None:
        raise ValueError(f"No embeddings tensor found in {path}")
    embeddings = embeddings.float()
    if embeddings.dim() == 3:
        if embeddings.shape[0] != 1:
            raise ValueError(f"Expected batch dimension 1 in {path}, got {tuple(embeddings.shape)}")
        embeddings = embeddings.squeeze(0)
    if embeddings.dim() != 2:
        raise ValueError(f"Expected embeddings shape (taxa, dim) in {path}, got {tuple(embeddings.shape)}")
    return embeddings.to(device)


def _load_phyla_embedding_bank(datasets, embedding_dir, device):
    if not datasets:
        return {}
    embedding_dir = Path(embedding_dir)
    bank = {}
    for dataset_id in sorted({str(item).upper() for item in datasets}):
        path = embedding_dir / f"{dataset_id}_phyla_beta_embeddings.pt"
        bank[dataset_id] = _load_phyla_embeddings(path, device)
    return bank


def _add_sample_metadata(samples, *, dataset_id, case_offset):
    updated = []
    for local_index, sample in enumerate(samples):
        item = dict(sample)
        item["dataset_id"] = str(dataset_id).upper()
        item["case_index"] = int(case_offset) + int(local_index)
        updated.append(item)
    return updated


def _build_samples_from_specs(module_like, specs, default_dataset_id):
    all_samples = []
    if specs:
        for raw_spec in specs:
            dataset_id, start_trees, target_trees = _parse_dataset_spec(raw_spec)
            samples = _build_branch_relax_samples_for_module(
                module_like,
                start_trees,
                target_trees,
            )
            all_samples.extend(
                _add_sample_metadata(
                    samples,
                    dataset_id=dataset_id,
                    case_offset=len(all_samples),
                )
            )
    else:
        samples = _build_branch_relax_samples_for_module(
            module_like,
            default_dataset_id[1],
            default_dataset_id[2],
        )
        all_samples.extend(
            _add_sample_metadata(
                samples,
                dataset_id=default_dataset_id[0],
                case_offset=0,
            )
        )
    if not all_samples:
        raise ValueError("No branch relax samples were built")
    return all_samples


def _apply_relaxer(relaxer, sample, device, *, scale=1.0, edge_floor=1e-8, phyla_bank=None):
    relaxer.eval()
    tokenized = relaxer.model.tokenizer([sample["newick_tree"]])
    tokenized = _move_tokenized_batch_to_device(tokenized, device)
    phyla_embeddings = None
    if phyla_bank:
        phyla_embeddings = phyla_bank[str(sample["dataset_id"]).upper()]
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
        entries, lengths, n_leaves, mapping = _branch_relax_entries_for_tree(
            relaxer,
            sample["newick_tree"],
            edge_split_masks[0],
            labels=None,
        )
        if not entries:
            return sample["newick_tree"], {"applied": False}
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
            int(sample["case_index"]),
            dtype=torch.long,
            device=device,
        )
        deltas = relaxer.head(features, numeric, case_indices).detach().cpu().numpy()

    next_lengths = {int(mask): float(length) for mask, length in lengths.items()}
    max_abs_delta = 0.0
    for entry, delta in zip(entries, deltas):
        next_len = max(
            float(entry["length"]) + float(scale) * float(delta),
            float(edge_floor),
        )
        next_lengths[int(entry.get("source_mask", entry["mask"]))] = next_len
        max_abs_delta = max(max_abs_delta, abs(next_len - float(entry["length"])))
    td_next = {
        int(mask): float(length)
        for mask, length in next_lengths.items()
        if float(length) > float(edge_floor)
    }
    newick = build_tree_from_splits(
        list(td_next.keys()),
        td_next,
        int(n_leaves),
        root_leaf=int(n_leaves) - 1,
        mapping=mapping,
    )[1]
    return newick, {"applied": True, "max_abs_delta": float(max_abs_delta)}


def _evaluate(relaxer, samples, scorers, device, *, eval_count, rng, scale, phyla_bank=None):
    if eval_count is not None and int(eval_count) > 0 and len(samples) > int(eval_count):
        selected = rng.sample(samples, int(eval_count))
    else:
        selected = list(samples)
    before = []
    after = []
    target = []
    applied = 0
    for sample in selected:
        scorer = scorers[str(sample["dataset_id"]).upper()]
        before.append(scorer.log_likelihood(sample["newick_tree"]))
        relaxed_tree, info = _apply_relaxer(
            relaxer,
            sample,
            device,
            scale=scale,
            phyla_bank=phyla_bank,
        )
        after.append(scorer.log_likelihood(relaxed_tree))
        target.append(scorer.log_likelihood(sample["target_tree"]))
        applied += int(bool(info.get("applied")))
    return {
        "eval_count": len(selected),
        "applied_count": int(applied),
        "before_log_likelihood_mean": float(np.mean(before)),
        "after_log_likelihood_mean": float(np.mean(after)),
        "target_log_likelihood_mean": float(np.mean(target)),
        "after_minus_before_mean": float(np.mean(np.asarray(after) - np.asarray(before))),
        "target_minus_before_mean": float(np.mean(np.asarray(target) - np.asarray(before))),
    }


def _prefix_metrics(prefix, metrics):
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _write_jsonl(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--start-trees", default=DEFAULT_START_TREES)
    parser.add_argument("--target-trees", default=DEFAULT_TARGET_TREES)
    parser.add_argument("--dataset-spec", action="append", default=[])
    parser.add_argument("--eval-start-trees", default=None)
    parser.add_argument("--eval-target-trees", default=None)
    parser.add_argument("--eval-name", default="zero_eval")
    parser.add_argument("--eval-spec", action="append", default=[])
    parser.add_argument("--extra-eval-name", action="append", default=[])
    parser.add_argument("--extra-eval-start-trees", action="append", default=[])
    parser.add_argument("--extra-eval-target-trees", action="append", default=[])
    parser.add_argument("--score-prefix", default=None)
    parser.add_argument("--score-prefixes", default=None)
    parser.add_argument("--load-checkpoint", default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--balanced-dataset-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-train-eval", action="store_true")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset-id", default="DS1")
    parser.add_argument(
        "--phyla-embedding-dir",
        default="${PHYLAFLOW_PHYLA_EMBEDDING_DIR}",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--case-dim", type=int, default=0)
    parser.add_argument("--phyla-dim", type=int, default=256)
    parser.add_argument("--phyla-use-leaf-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phyla-use-split-tokens", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--relax-scale", type=float, default=1.0)
    args = parser.parse_args()
    args.base_config = os.path.expandvars(args.base_config)
    args.start_trees = os.path.expandvars(args.start_trees)
    args.target_trees = os.path.expandvars(args.target_trees)
    args.out_dir = os.path.expandvars(args.out_dir)
    args.phyla_embedding_dir = os.path.expandvars(args.phyla_embedding_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = str(out_dir / "metrics.jsonl")
    ckpt_path = str(out_dir / "best.pt")

    cfg = _small_model_config(args.base_config, args)
    model = return_model(cfg).to(device)
    module_like = SimpleNamespace(model=model)
    samples = _build_samples_from_specs(
        module_like,
        args.dataset_spec,
        (str(args.dataset_id).upper(), args.start_trees, args.target_trees),
    )
    eval_sets = []
    for raw_spec in args.eval_spec:
        eval_name, dataset_id, eval_start_trees, eval_target_trees = _parse_eval_spec(raw_spec)
        eval_sets.append(
            (
                str(eval_name),
                _add_sample_metadata(
                    _build_branch_relax_samples_for_module(
                        module_like,
                        eval_start_trees,
                        eval_target_trees,
                    ),
                    dataset_id=dataset_id,
                    case_offset=len(samples) + sum(len(items) for _name, items in eval_sets),
                ),
            )
        )
    if args.eval_start_trees or args.eval_target_trees:
        if not (args.eval_start_trees and args.eval_target_trees):
            raise ValueError("--eval-start-trees and --eval-target-trees must be provided together")
        eval_sets.append(
            (
                str(args.eval_name),
                _add_sample_metadata(
                    _build_branch_relax_samples_for_module(
                        module_like,
                        args.eval_start_trees,
                        args.eval_target_trees,
                    ),
                    dataset_id=str(args.dataset_id).upper(),
                    case_offset=len(samples) + sum(len(items) for _name, items in eval_sets),
                ),
            )
        )
    if not (
        len(args.extra_eval_name)
        == len(args.extra_eval_start_trees)
        == len(args.extra_eval_target_trees)
    ):
        raise ValueError(
            "--extra-eval-name, --extra-eval-start-trees, and "
            "--extra-eval-target-trees must be repeated the same number of times"
        )
    for eval_name, eval_start_trees, eval_target_trees in zip(
        args.extra_eval_name,
        args.extra_eval_start_trees,
        args.extra_eval_target_trees,
    ):
        eval_sets.append(
            (
                str(eval_name),
                _add_sample_metadata(
                    _build_branch_relax_samples_for_module(
                        module_like,
                        eval_start_trees,
                        eval_target_trees,
                    ),
                    dataset_id=str(args.dataset_id).upper(),
                    case_offset=len(samples) + sum(len(items) for _name, items in eval_sets),
                ),
            )
        )
    head = BranchDeltaHead(
        int(model.embed_dim),
        hidden_dim=int(args.head_hidden_dim),
        case_dim=int(args.case_dim),
        num_cases=len(samples),
    ).to(device)
    relaxer = StandaloneRelaxer(model, head).to(device)
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        head.load_state_dict(checkpoint["head"])

    optimizer = torch.optim.AdamW(
        relaxer.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    dataset_ids = {
        str(sample["dataset_id"]).upper()
        for sample in samples
    }
    for _eval_name, eval_samples in eval_sets:
        dataset_ids.update(str(sample["dataset_id"]).upper() for sample in eval_samples)
    scorers = {
        dataset_id: GenericJCLikelihood(dataset_id=dataset_id)
        for dataset_id in sorted(dataset_ids)
    }
    phyla_bank = _load_phyla_embedding_bank(
        dataset_ids,
        args.phyla_embedding_dir,
        device,
    )
    rng = random.Random(args.seed + 17)
    best_after = -float("inf")

    print(
        json.dumps(
            {
                "event": "start",
                "num_samples": len(samples),
                "datasets": sorted(dataset_ids),
                "eval_sets": {
                    str(name): len(eval_samples)
                    for name, eval_samples in eval_sets
                },
                "device": str(device),
                "out_dir": str(out_dir),
                "case_dim": int(args.case_dim),
                "embed_dim": int(args.embed_dim),
                "n_layers": int(args.n_layers),
                "n_heads": int(args.n_heads),
                "phyla_dim": int(args.phyla_dim),
                "phyla_embedding_dir": str(args.phyla_embedding_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if args.eval_only:
        eval_only_metrics = {"event": "eval_only", "timestamp": time.time()}
        if not bool(args.skip_train_eval):
            eval_only_metrics.update(
                _prefix_metrics(
                    "train_eval",
                    _evaluate(
                        relaxer,
                        samples,
                        scorers,
                        device,
                        eval_count=int(args.eval_count),
                        rng=rng,
                        scale=float(args.relax_scale),
                        phyla_bank=phyla_bank,
                    ),
                )
            )
        if eval_sets:
            for eval_name, eval_samples in eval_sets:
                eval_only_metrics.update(
                    _prefix_metrics(
                        str(eval_name),
                        _evaluate(
                            relaxer,
                            eval_samples,
                            scorers,
                            device,
                            eval_count=int(args.eval_count),
                            rng=rng,
                            scale=float(args.relax_scale),
                            phyla_bank=phyla_bank,
                        ),
                    )
                )
        else:
            eval_only_metrics.update(
                _evaluate(
                    relaxer,
                    samples,
                    scorers,
                    device,
                    eval_count=int(args.eval_count),
                    rng=rng,
                    scale=float(args.relax_scale),
                    phyla_bank=phyla_bank,
                )
            )
        _write_jsonl(metrics_path, eval_only_metrics)
        print(json.dumps(eval_only_metrics, sort_keys=True), flush=True)
        return

    samples_by_dataset = defaultdict(list)
    for sample in samples:
        samples_by_dataset[str(sample["dataset_id"]).upper()].append(sample)
    train_dataset_ids = sorted(samples_by_dataset)

    for step in range(1, int(args.max_steps) + 1):
        relaxer.train()
        if bool(args.balanced_dataset_sampling) and train_dataset_ids:
            batch = []
            for _ in range(min(int(args.batch_size), len(samples))):
                dataset_id = rng.choice(train_dataset_ids)
                batch.append(rng.choice(samples_by_dataset[dataset_id]))
        else:
            batch = rng.sample(samples, min(int(args.batch_size), len(samples)))
        tokenized = model.tokenizer([sample["newick_tree"] for sample in batch])
        pred, target = relaxer.forward_batch(
            tokenized,
            [sample["newick_tree"] for sample in batch],
            batch,
            device,
            phyla_bank=phyla_bank,
        )
        if pred is None:
            continue
        loss = torch.mean((pred - target) ** 2)
        mae = torch.mean(torch.abs(pred - target))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = None
        if float(args.grad_clip) > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                relaxer.parameters(),
                float(args.grad_clip),
            )
        optimizer.step()

        if step == 1 or step % int(args.eval_every) == 0:
            metrics = {
                "step": int(step),
                "timestamp": time.time(),
                "train_mse": float(loss.detach().cpu()),
                "train_mae": float(mae.detach().cpu()),
            }
            if grad_norm is not None:
                metrics["grad_norm_before_clip"] = float(grad_norm)
            metrics.update(
                _prefix_metrics(
                    "train_eval",
                    _evaluate(
                        relaxer,
                        samples,
                        scorers,
                        device,
                        eval_count=int(args.eval_count),
                        rng=rng,
                        scale=float(args.relax_scale),
                        phyla_bank=phyla_bank,
                    ),
                )
            )
            if eval_sets:
                for eval_name, eval_samples in eval_sets:
                    metrics.update(
                        _prefix_metrics(
                            str(eval_name),
                            _evaluate(
                                relaxer,
                                eval_samples,
                                scorers,
                                device,
                                eval_count=int(args.eval_count),
                                rng=rng,
                                scale=float(args.relax_scale),
                                phyla_bank=phyla_bank,
                            ),
                        )
                    )
            else:
                metrics.update(
                    _evaluate(
                        relaxer,
                        samples,
                        scorers,
                        device,
                        eval_count=int(args.eval_count),
                        rng=rng,
                        scale=float(args.relax_scale),
                        phyla_bank=phyla_bank,
                    )
                )
            _write_jsonl(metrics_path, metrics)
            print(json.dumps(metrics, sort_keys=True), flush=True)
            if args.score_prefixes:
                score_keys = [
                    f"{prefix.strip()}_after_log_likelihood_mean"
                    for prefix in str(args.score_prefixes).split(",")
                    if prefix.strip()
                ]
                score_value = float(np.mean([metrics[key] for key in score_keys]))
            elif args.score_prefix:
                score_key = f"{args.score_prefix}_after_log_likelihood_mean"
                score_value = metrics[score_key]
            elif eval_sets:
                score_key = f"{eval_sets[0][0]}_after_log_likelihood_mean"
                score_value = metrics[score_key]
            else:
                score_key = "after_log_likelihood_mean"
                score_value = metrics[score_key]
            if score_value > best_after:
                best_after = score_value
                torch.save(
                    {
                        "model": model.state_dict(),
                        "head": head.state_dict(),
                        "args": vars(args),
                        "metrics": metrics,
                    },
                    ckpt_path,
                )

    final_metrics = {}
    final_metrics.update(
        _prefix_metrics(
            "train_eval",
            _evaluate(
                relaxer,
                samples,
                scorers,
                device,
                eval_count=0,
                rng=rng,
                scale=float(args.relax_scale),
                phyla_bank=phyla_bank,
            ),
        )
    )
    if eval_sets:
        for eval_name, eval_samples in eval_sets:
            final_metrics.update(
                _prefix_metrics(
                    str(eval_name),
                    _evaluate(
                        relaxer,
                        eval_samples,
                        scorers,
                        device,
                        eval_count=0,
                        rng=rng,
                        scale=float(args.relax_scale),
                        phyla_bank=phyla_bank,
                    )
                )
            )
    else:
        final_metrics.update(
            _evaluate(
                relaxer,
                samples,
                scorers,
                device,
                eval_count=0,
                rng=rng,
                scale=float(args.relax_scale),
                phyla_bank=phyla_bank,
            )
        )
    final_metrics.update({"step": int(args.max_steps), "event": "final", "timestamp": time.time()})
    _write_jsonl(metrics_path, final_metrics)
    print(json.dumps(final_metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
