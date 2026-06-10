#!/usr/bin/env python
"""Export a frozen start-tree metric table aligned to a compact JSONL index."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pretrain_start_tree_metric_encoder import (  # noqa: E402
    PairMetricModel,
    build_tree_tensor_batch,
    canonical_internal_splits,
    embedding_stats,
)


_TOKEN_DELIMS = set("(),:;")


def _leaf_names_fast(newick: str) -> list[str]:
    names = []
    text = str(newick)
    n = len(text)
    i = 0
    while i < n:
        if text[i] not in "(,":
            i += 1
            continue
        i += 1
        while i < n and text[i].isspace():
            i += 1
        if i >= n or text[i] == "(":
            continue
        start = i
        while i < n and text[i] not in _TOKEN_DELIMS and not text[i].isspace():
            i += 1
        name = text[start:i].strip()
        if name:
            names.append(name)
    return names


def _skip_metadata(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    while i < n and text[i] not in ",);":
        i += 1
    return i


def _canonical_internal_splits_fast(newick: str) -> tuple[tuple[int, ...], int]:
    text = str(newick).strip()
    leaf_names = _leaf_names_fast(text)
    if not leaf_names:
        raise ValueError("No leaves found in Newick")
    try:
        ordered_names = sorted(leaf_names, key=lambda name: int(name))
    except ValueError:
        ordered_names = sorted(leaf_names)
    name_to_idx = {name: idx for idx, name in enumerate(ordered_names)}
    n_taxa = len(ordered_names)
    full_mask = (1 << n_taxa) - 1
    splits: set[int] = set()

    def parse_subtree(i: int, *, is_root: bool = False) -> tuple[int, int]:
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            raise ValueError("Unexpected end of Newick")
        if text[i] == "(":
            i += 1
            mask = 0
            while True:
                child_mask, i = parse_subtree(i, is_root=False)
                mask |= child_mask
                while i < len(text) and text[i].isspace():
                    i += 1
                if i < len(text) and text[i] == ",":
                    i += 1
                    continue
                if i < len(text) and text[i] == ")":
                    i += 1
                    break
                raise ValueError(f"Unexpected character while parsing Newick: {text[i:i+20]!r}")
            i = _skip_metadata(text, i)
            if not is_root:
                other = full_mask ^ mask
                if 1 < mask.bit_count() < n_taxa - 1:
                    splits.add(min(int(mask), int(other)))
            return mask, i

        start = i
        while i < len(text) and text[i] not in _TOKEN_DELIMS and not text[i].isspace():
            i += 1
        name = text[start:i].strip()
        if name not in name_to_idx:
            raise ValueError(f"Unknown leaf name while parsing Newick: {name!r}")
        i = _skip_metadata(text, i)
        return 1 << name_to_idx[name], i

    mask, _ = parse_subtree(0, is_root=True)
    if mask != full_mask:
        raise ValueError("Parsed Newick did not cover all leaves")
    return tuple(sorted(splits)), n_taxa


def _canonical_internal_splits_export(newick: str) -> tuple[tuple[int, ...], int]:
    try:
        return _canonical_internal_splits_fast(newick)
    except Exception:
        return canonical_internal_splits(newick)


def _read_tree(path: str) -> str:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    tree = (
        payload.get("tree")
        or payload.get("start_tree")
        or payload.get("final_tree")
        or payload.get("target_tree")
    )
    if not tree:
        raise ValueError(f"No tree found in {path}")
    tree = str(tree).strip()
    return tree if tree.endswith(";") else tree + ";"


def _load_metric_model(checkpoint_path: Path, device: torch.device) -> tuple[PairMetricModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metadata = dict(checkpoint.get("metadata", {}))
    thresholds = metadata.get("bin_thresholds", [0.05, 0.15, 0.30, 0.50, 0.75])
    model = PairMetricModel(
        max_bits=int(metadata.get("max_bits", 256)),
        hidden_dim=int(metadata.get("hidden_dim", 256)),
        embedding_dim=int(metadata.get("embedding_dim", 64)),
        num_bins=len(thresholds) + 1,
        dropout=0.0,
    )
    state_dict = checkpoint.get("model_state_dict")
    if not state_dict:
        raise ValueError(f"{checkpoint_path} does not contain model_state_dict")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, metadata


def _index_records(index_path: Path, max_rows: int = 0):
    emitted = 0
    with index_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if max_rows and emitted >= int(max_rows):
                break
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            case_index = int(record.get("case_index", line_number - 1))
            if case_index != line_number - 1:
                raise ValueError(
                    f"{index_path}:{line_number} has case_index={case_index}, "
                    f"expected {line_number - 1}"
                )
            start_path = record.get("start_path") or record.get("start_json")
            if not start_path:
                raise ValueError(f"{index_path}:{line_number} missing start_path")
            emitted += 1
            yield {
                "index": case_index,
                "path": str(start_path),
                "dataset_id": str(record.get("dataset_id", "")),
            }


def _prepare_export_record(record: dict[str, object]) -> dict[str, object]:
    tree = _read_tree(str(record["path"]))
    splits, n_taxa = _canonical_internal_splits_export(tree)
    return {
        "index": int(record["index"]),
        "path": str(record["path"]),
        "dataset_id": str(record.get("dataset_id", "")),
        "split_set": splits,
        "n_taxa": int(n_taxa),
        "start_tree_hash": hashlib.sha1(tree.encode("utf-8")).hexdigest(),
    }


def _iter_prepared_chunks(
    index_path: Path,
    chunk_size: int,
    *,
    max_rows: int = 0,
    num_workers: int = 0,
):
    rows = []
    records = _index_records(index_path, max_rows=max_rows)
    if int(num_workers) > 0:
        with mp.Pool(processes=int(num_workers)) as pool:
            iterator = pool.imap(
                _prepare_export_record,
                records,
                chunksize=max(1, int(chunk_size) // max(1, int(num_workers) * 4)),
            )
            for row in iterator:
                rows.append(row)
                if len(rows) >= int(chunk_size):
                    yield rows
                    rows = []
    else:
        for record in records:
            rows.append(_prepare_export_record(record))
            if len(rows) >= int(chunk_size):
                yield rows
                rows = []
    if rows:
        yield rows


def _encode_prepared_rows(
    model: PairMetricModel,
    rows: list[dict[str, object]],
    *,
    max_bits: int,
    device: torch.device,
) -> torch.Tensor:
    split_sets = [row["split_set"] for row in rows]
    n_taxa = [int(row["n_taxa"]) for row in rows]
    split_bits, pad_mask, size_features = build_tree_tensor_batch(
        split_sets,
        n_taxa,
        max_bits=int(max_bits),
        device=device,
    )
    with torch.no_grad():
        emb = model.encode(split_bits, pad_mask, size_features)
        emb = F.normalize(emb, dim=-1)
    return emb.detach().cpu()


def _bounded_embedding_stats(embeddings: torch.Tensor, max_rows: int) -> dict:
    if not embeddings.numel():
        return {}
    max_rows = max(1, int(max_rows))
    if int(embeddings.shape[0]) <= max_rows:
        return embedding_stats(embeddings)
    sample = embeddings[:max_rows].contiguous()
    stats = embedding_stats(sample)
    stats["sample_size"] = int(sample.shape[0])
    stats["full_num_cases"] = int(embeddings.shape[0])
    return stats


def export(args: argparse.Namespace) -> dict:
    index_path = args.index.expanduser().resolve()
    checkpoint_path = args.metric_encoder.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    device = torch.device(args.device)
    model, metadata = _load_metric_model(checkpoint_path, device)

    started = time.time()
    chunks = []
    start_tree_hashes = []
    dataset_ids = []
    paths = []
    count = 0
    for chunk in _iter_prepared_chunks(
        index_path,
        int(args.chunk_size),
        max_rows=int(args.max_rows),
        num_workers=int(args.num_workers),
    ):
        emb = _encode_prepared_rows(
            model,
            chunk,
            max_bits=int(metadata.get("max_bits", 256)),
            device=device,
        )
        chunks.append(emb.cpu())
        for row in chunk:
            start_tree_hashes.append(str(row["start_tree_hash"]))
            dataset_ids.append(str(row["dataset_id"]))
            paths.append(str(row["path"]))
        count += len(chunk)
        if args.log_every and count % int(args.log_every) < len(chunk):
            elapsed = time.time() - started
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "count": count,
                        "elapsed_sec": elapsed,
                        "rows_per_sec": count / max(elapsed, 1e-9),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    embeddings = torch.cat(chunks, dim=0) if chunks else torch.zeros(0, 0)
    export_metadata = {
        "source_checkpoint": str(checkpoint_path),
        "source_index": str(index_path),
        "num_cases": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "case_indices": list(range(int(embeddings.shape[0]))),
        "start_tree_hashes": start_tree_hashes,
        "dataset_ids": dataset_ids,
        "paths": paths,
        "embedding_stats": _bounded_embedding_stats(
            embeddings,
            int(args.stats_max_rows),
        ),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "elapsed_sec": time.time() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"embeddings": embeddings, "metadata": export_metadata}, output_path)
    return {
        "output": str(output_path),
        "num_cases": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "elapsed_sec": export_metadata["elapsed_sec"],
        "embedding_stats": export_metadata["embedding_stats"],
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--metric-encoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=10000)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--stats-max-rows", type=int, default=4096)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    print(json.dumps(export(parse_args(argv)), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
