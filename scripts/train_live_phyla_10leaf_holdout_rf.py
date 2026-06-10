#!/usr/bin/env python3
"""Fine-tune live Phyla on 10-leaf cases and evaluate held-out RF."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.full_sanity_fixedpair_20260401.benchmark_phylaflow_nexus_sitechunk_phyla import (  # noqa: E501
    iter_windows,
    load_nexus_alignment,
)
from scripts.overfit_phyla_10leaf_clade_probe import (
    DEFAULT_CHECKPOINT,
    DEFAULT_NEXUS_ROOT,
    PairHead,
    _leaf_to_index,
    _live_pooled_with_grad,
    _load_live_phyla,
    _nj_newick,
    _pair_features,
    _select_phyla_trainable_params,
    _sequence_windows,
    _target_pair_data,
)
from scripts.probe_phyla_10leaf_pairwise import _pruned_target_tree, _read_jsonl
from utils.metric_utils import calculate_norm_rf


DEFAULT_TRAIN_INDEX = Path(
    "/ewsc/yektefai/phylaflow_datasets/"
    "orthomam10leaf_train80_datasetsplit_8000cases_20260511_index.jsonl"
)
DEFAULT_EVAL_INDEX = Path(
    "/ewsc/yektefai/phylaflow_datasets/"
    "orthomam10leaf_holdout20_datasetsplit_1600cases_20260511_index.jsonl"
)


@dataclass
class Case:
    row_index: int
    dataset_id: str
    leaves: list[str]
    sequence_names: list[str]
    sequences: list[str]
    target_tree: str
    windows: list[tuple[int, int]]
    pair_indices: list[tuple[int, int]]
    norm_dist: torch.Tensor
    close: torch.Tensor
    input_tokens: int


class AlignmentCache:
    def __init__(self, nexus_root: Path):
        self.nexus_root = nexus_root
        self.cache: dict[str, tuple[list[str], list[str]]] = {}

    def get(self, dataset_id: str) -> tuple[list[str], list[str]]:
        dataset_id = str(dataset_id)
        cached = self.cache.get(dataset_id)
        if cached is None:
            cached = load_nexus_alignment(self.nexus_root / f"{dataset_id}.nex")
            self.cache[dataset_id] = cached
        return cached


def _prepare_cases(
    rows: list[dict],
    *,
    alignment_cache: AlignmentCache,
    subset_size: int,
    seed: int,
    positive_quantile: float,
    window_size: int,
    stride: int,
    max_windows: int,
    input_mode: str,
    max_input_tokens: int,
) -> list[Case]:
    cases: list[Case] = []
    skipped = 0
    for row_index, row in enumerate(rows):
        try:
            target_tree, leaves = _pruned_target_tree(row, subset_size=subset_size, seed=seed)
            names, sequences = alignment_cache.get(row["dataset_id"])
            indices = [_leaf_to_index(leaf, names) for leaf in leaves]
            selected_names = [names[idx] for idx in indices]
            selected_sequences = [sequences[idx] for idx in indices]
            selected_sequences, windows = _sequence_windows(
                selected_sequences,
                input_mode,
                window_size,
                stride,
                max_windows,
            )
            if not windows:
                raise ValueError("No sequence windows selected")
            if input_mode == "aligned-windows":
                input_tokens = max((end - start) * len(selected_sequences) + len(selected_sequences) for start, end in windows)
            else:
                input_tokens = max(
                    sum(len(sequence[start:end]) for sequence in selected_sequences) + len(selected_sequences)
                    for start, end in windows
                )
            if max_input_tokens > 0 and input_tokens > max_input_tokens:
                raise ValueError(f"input_tokens {input_tokens} exceeds max_input_tokens {max_input_tokens}")
            pair_indices, _dist, norm_dist, close, _threshold = _target_pair_data(
                target_tree,
                leaves,
                positive_quantile,
            )
            cases.append(
                Case(
                    row_index=row_index,
                    dataset_id=str(row["dataset_id"]),
                    leaves=leaves,
                    sequence_names=selected_names,
                    sequences=selected_sequences,
                    target_tree=target_tree,
                    windows=windows,
                    pair_indices=pair_indices,
                    norm_dist=norm_dist,
                    close=close,
                    input_tokens=int(input_tokens),
                )
            )
        except Exception as exc:
            skipped += 1
            print(
                json.dumps(
                    {
                        "event": "skip_case",
                        "row_index": row_index,
                        "dataset_id": row.get("dataset_id"),
                        "error": str(exc),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if not cases:
        raise RuntimeError(f"No cases prepared from {len(rows)} rows")
    print(
        json.dumps(
            {
                "event": "prepared_cases",
                "rows": len(rows),
                "cases": len(cases),
                "skipped": skipped,
                "unique_datasets": len({case.dataset_id for case in cases}),
                "max_input_tokens": max(case.input_tokens for case in cases),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return cases


def _predict_distances(
    phyla,
    head: PairHead,
    case: Case,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pooled = _live_pooled_with_grad(
        phyla,
        case.sequences,
        case.sequence_names,
        case.windows,
        device,
    )
    pooled = F.normalize(pooled.float(), dim=-1)
    features = _pair_features(pooled, case.pair_indices)
    pred_dist, logits = head(features)
    pred_dist = torch.sigmoid(pred_dist)
    raw_dist = torch.pdist(pooled.detach())
    raw_dist = raw_dist / raw_dist.max().clamp_min(1e-6)
    return pred_dist, logits, raw_dist


def _case_losses(
    pred_dist: torch.Tensor,
    logits: torch.Tensor,
    case: Case,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    norm_dist = case.norm_dist.to(device)
    close = case.close.to(device)
    loss_dist = F.mse_loss(pred_dist, norm_dist)
    loss_close = F.binary_cross_entropy_with_logits(logits, close)
    return loss_dist + loss_close, loss_dist, loss_close


def _evaluate_cases(
    phyla,
    head: PairHead,
    cases: list[Case],
    device: str,
    max_cases: int,
) -> dict:
    selected = cases[: max_cases if max_cases > 0 else len(cases)]
    phyla.eval()
    head.eval()
    pair_rfs = []
    raw_rfs = []
    mses = []
    close_correct = 0.0
    close_total = 0
    with torch.no_grad():
        for case in selected:
            pred_dist, logits, raw_dist = _predict_distances(phyla, head, case, device)
            target = case.norm_dist.to(pred_dist.device)
            mses.append(float(F.mse_loss(pred_dist, target).detach().cpu().item()))
            close = case.close.to(logits.device)
            close_correct += float(((torch.sigmoid(logits) >= 0.5).float() == close).float().sum().item())
            close_total += int(close.numel())
            try:
                pair_tree = _nj_newick(case.leaves, pred_dist.detach().cpu(), case.pair_indices)
                pair_rfs.append(float(calculate_norm_rf(pair_tree, case.target_tree)))
            except Exception:
                pair_rfs.append(float("nan"))
            try:
                raw_tree = _nj_newick(case.leaves, raw_dist.detach().cpu(), case.pair_indices)
                raw_rfs.append(float(calculate_norm_rf(raw_tree, case.target_tree)))
            except Exception:
                raw_rfs.append(float("nan"))

    def summarize(values: list[float]) -> dict:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"mean": float("nan"), "median": float("nan"), "n": 0}
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "n": int(arr.size),
        }

    return {
        "cases": len(selected),
        "pair_head_rf_norm": summarize(pair_rfs),
        "raw_pooled_rf_norm": summarize(raw_rfs),
        "pair_head_mse_mean": float(np.mean(mses)) if mses else float("nan"),
        "close_accuracy": float(close_correct / max(close_total, 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", type=Path, default=DEFAULT_TRAIN_INDEX)
    parser.add_argument("--eval-index", type=Path, default=DEFAULT_EVAL_INDEX)
    parser.add_argument("--nexus-root", type=Path, default=DEFAULT_NEXUS_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-rows", type=int, default=128)
    parser.add_argument("--eval-rows", type=int, default=64)
    parser.add_argument("--eval-cases", type=int, default=64)
    parser.add_argument("--subset-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--positive-quantile", type=float, default=0.25)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=0,
        help="Skip cases whose maximum Phyla forward input length exceeds this many tokens.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("aligned-windows", "raw-full", "raw-windows"),
        default="aligned-windows",
    )
    parser.add_argument(
        "--sample-train-window",
        action="store_true",
        help="Use one random sequence window for each train update; evaluation still uses all prepared windows.",
    )
    parser.add_argument(
        "--phyla-train-scope",
        choices=("all", "last-module", "tree-head"),
        default="all",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--phyla-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--output-final", type=Path, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows = _read_jsonl(args.train_index, limit=args.train_rows)
    eval_rows = _read_jsonl(args.eval_index, limit=args.eval_rows)
    alignment_cache = AlignmentCache(args.nexus_root)
    train_cases = _prepare_cases(
        train_rows,
        alignment_cache=alignment_cache,
        subset_size=args.subset_size,
        seed=args.seed,
        positive_quantile=args.positive_quantile,
        window_size=args.window_size,
        stride=args.stride,
        max_windows=args.max_windows,
        input_mode=args.input_mode,
        max_input_tokens=args.max_input_tokens,
    )
    eval_cases = _prepare_cases(
        eval_rows,
        alignment_cache=alignment_cache,
        subset_size=args.subset_size,
        seed=args.seed + 17,
        positive_quantile=args.positive_quantile,
        window_size=args.window_size,
        stride=args.stride,
        max_windows=args.max_windows,
        input_mode=args.input_mode,
        max_input_tokens=args.max_input_tokens,
    )

    phyla = _load_live_phyla(args.checkpoint, args.device)
    trainable_params, trainable_count = _select_phyla_trainable_params(
        phyla,
        args.phyla_train_scope,
    )
    head = PairHead(4 * 256, args.hidden_dim).to(args.device)
    opt = torch.optim.AdamW(
        [
            {"params": trainable_params, "lr": args.phyla_lr},
            {"params": head.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    rng = random.Random(args.seed)

    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records = []

    def emit(record: dict) -> None:
        record = dict(record)
        record.setdefault("trainable_phyla_params", int(trainable_count))
        print(json.dumps(record, sort_keys=True), flush=True)
        records.append(record)
        if args.output_jsonl is not None:
            with args.output_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    emit(
        {
            "event": "start",
            "train_cases": len(train_cases),
            "eval_cases": len(eval_cases),
            "eval_subset_cases": args.eval_cases,
            "steps": args.steps,
            "phyla_train_scope": args.phyla_train_scope,
            "max_windows": args.max_windows,
            "input_mode": args.input_mode,
            "max_input_tokens": args.max_input_tokens,
            "sample_train_window": bool(args.sample_train_window),
            "device": args.device,
        }
    )
    emit({"event": "eval", "step": 0, "split": "train", **_evaluate_cases(phyla, head, train_cases, args.device, min(16, len(train_cases)))})
    emit({"event": "eval", "step": 0, "split": "heldout", **_evaluate_cases(phyla, head, eval_cases, args.device, args.eval_cases)})

    for step in range(1, args.steps + 1):
        case = rng.choice(train_cases)
        train_case = case
        if args.sample_train_window and len(case.windows) > 1:
            train_case = replace(case, windows=[rng.choice(case.windows)])
        phyla.train()
        head.train()
        pred_dist, logits, _raw_dist = _predict_distances(phyla, head, train_case, args.device)
        loss, loss_dist, loss_close = _case_losses(pred_dist, logits, train_case, args.device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(trainable_params) + list(head.parameters()),
                args.grad_clip,
            )
        opt.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            emit(
                {
                    "event": "train_step",
                    "step": step,
                    "dataset_id": case.dataset_id,
                    "loss": float(loss.detach().cpu().item()),
                    "mse": float(loss_dist.detach().cpu().item()),
                    "bce": float(loss_close.detach().cpu().item()),
                }
            )
            emit({"event": "eval", "step": step, "split": "train", **_evaluate_cases(phyla, head, train_cases, args.device, min(16, len(train_cases)))})
            emit({"event": "eval", "step": step, "split": "heldout", **_evaluate_cases(phyla, head, eval_cases, args.device, args.eval_cases)})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    final = {
        "config": config,
        "trainable_phyla_params": int(trainable_count),
        "records": records,
    }
    if args.output_final is not None:
        args.output_final.parent.mkdir(parents=True, exist_ok=True)
        args.output_final.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
