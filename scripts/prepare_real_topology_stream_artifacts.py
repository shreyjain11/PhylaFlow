#!/usr/bin/env python
"""Prepare extracted real topology-stream cases for metric/relaxer training.

The extracted stream keeps the original worker-local case names, so case IDs
repeat across workers. This script writes normalized start/target JSON copies
with globally unique group keys and emits the tree-list/spec files consumed by
the existing metric encoder and standalone branch-relaxer scripts.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATA_ROOT = Path("/ewsc/yektefai/phylaflow_datasets")
DEFAULT_STREAM_ROOT = DEFAULT_DATA_ROOT / "real_unique_topology_stream"
DEFAULT_NEXUS_ROOT = DEFAULT_DATA_ROOT / "nexus"
DEFAULT_RUNS_ROOT = DEFAULT_DATA_ROOT / "runs"
DEFAULT_PHYLA_EMBEDDING_DIR = (
    DEFAULT_DATA_ROOT / "phyla_embeddings_sitechunk_cpu_20260428"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_DATA_ROOT / "real_topology_stream_metric_relaxer_20260502"
)


def _ensure_semicolon(tree: str) -> str:
    tree = str(tree).strip()
    return tree if tree.endswith(";") else tree + ";"


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _localize_stream_path(raw_path: str, stream_root: Path) -> Path:
    raw = str(raw_path)
    candidate = Path(raw)
    if candidate.exists():
        return candidate
    marker = "/real_unique_topology_stream/"
    if marker in raw:
        suffix = raw.split(marker, 1)[1]
        candidate = stream_root / suffix
        if candidate.exists():
            return candidate
    # Fall back to the final worker_*/... suffix if the absolute source root
    # moved but the stream-relative layout is intact.
    match = re.search(r"(worker_\d{3}/.+)$", raw)
    if match:
        candidate = stream_root / match.group(1)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(raw_path)


def _has_real_support_files(
    dataset_id: str,
    *,
    nexus_root: Path,
    runs_root: Path,
    phyla_embedding_dir: Path,
    require_nexus: bool,
    require_runs: bool,
    require_phyla_embedding: bool,
) -> bool:
    if require_nexus and not (nexus_root / f"{dataset_id}.nex").exists():
        return False
    if require_runs and not (runs_root / dataset_id).is_dir():
        return False
    if require_phyla_embedding:
        candidates = [
            phyla_embedding_dir / f"{dataset_id}_phyla_beta_embeddings.pt",
            phyla_embedding_dir
            / f"{dataset_id}_phyla_beta_sitechunk_w256_s256_embeddings.pt",
        ]
        if not any(path.exists() for path in candidates):
            return False
    return True


def _supported_dataset_ids(
    *,
    nexus_root: Path,
    runs_root: Path,
    phyla_embedding_dir: Path,
    require_nexus: bool,
    require_runs: bool,
    require_phyla_embedding: bool,
) -> set[str] | None:
    supported: set[str] | None = None
    if require_nexus:
        nexus_ids = {
            path.stem
            for pattern in ("*.nex", "*.nexus")
            for path in nexus_root.glob(pattern)
        }
        supported = nexus_ids if supported is None else supported & nexus_ids
    if require_runs:
        run_ids = {path.name for path in runs_root.iterdir() if path.is_dir()}
        supported = run_ids if supported is None else supported & run_ids
    if require_phyla_embedding:
        embedding_ids = set()
        for path in phyla_embedding_dir.glob("*_phyla_beta_embeddings.pt"):
            embedding_ids.add(path.name[: -len("_phyla_beta_embeddings.pt")])
        for path in phyla_embedding_dir.glob(
            "*_phyla_beta_sitechunk_w256_s256_embeddings.pt"
        ):
            embedding_ids.add(
                path.name[: -len("_phyla_beta_sitechunk_w256_s256_embeddings.pt")]
            )
        supported = embedding_ids if supported is None else supported & embedding_ids
    return supported


def _iter_ok_manifest_rows(
    *,
    stream_root: Path,
    supported_dataset_ids: set[str] | None,
) -> Iterable[dict[str, Any]]:
    for manifest_path in sorted(stream_root.glob("worker_*/manifest.jsonl")):
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                if row.get("status") != "ok":
                    continue
                dataset_id = str(row.get("dataset_id") or "").strip()
                if not dataset_id:
                    continue
                if supported_dataset_ids is not None and dataset_id not in supported_dataset_ids:
                    continue
                updated = dict(row)
                updated["manifest_path"] = str(manifest_path)
                updated["manifest_line_number"] = int(line_number)
                yield updated


def _select_rows(
    rows: list[dict[str, Any]],
    *,
    max_cases: int,
    max_datasets: int,
    per_dataset_cap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(int(seed))
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset_id"])].append(row)
    dataset_ids = sorted(by_dataset)
    rng.shuffle(dataset_ids)
    if max_datasets > 0:
        dataset_ids = dataset_ids[: int(max_datasets)]
    for dataset_id in dataset_ids:
        rng.shuffle(by_dataset[dataset_id])
        if per_dataset_cap > 0:
            by_dataset[dataset_id] = by_dataset[dataset_id][: int(per_dataset_cap)]

    selected: list[dict[str, Any]] = []
    while dataset_ids:
        next_dataset_ids = []
        for dataset_id in dataset_ids:
            bucket = by_dataset[dataset_id]
            if not bucket:
                continue
            selected.append(bucket.pop())
            if max_cases > 0 and len(selected) >= int(max_cases):
                selected.sort(
                    key=lambda row: (
                        str(row["dataset_id"]),
                        int(row.get("worker_index", -1)),
                        int(row.get("case_index", -1)),
                    )
                )
                return selected
            if bucket:
                next_dataset_ids.append(dataset_id)
        dataset_ids = next_dataset_ids
    selected.sort(
        key=lambda row: (
            str(row["dataset_id"]),
            int(row.get("worker_index", -1)),
            int(row.get("case_index", -1)),
        )
    )
    return selected


def _load_tree_payload(path: Path, tree_keys: tuple[str, ...]) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    tree = None
    for key in tree_keys:
        if payload.get(key):
            tree = payload[key]
            break
    if not tree:
        raise ValueError(f"No tree key {tree_keys} found in {path}")
    return payload, _ensure_semicolon(str(tree))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(str(line).rstrip("\n") + "\n")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    stream_root = args.stream_root.expanduser().resolve()
    nexus_root = args.nexus_root.expanduser().resolve()
    runs_root = args.runs_root.expanduser().resolve()
    phyla_embedding_dir = args.phyla_embedding_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    supported_ids = _supported_dataset_ids(
        nexus_root=nexus_root,
        runs_root=runs_root,
        phyla_embedding_dir=phyla_embedding_dir,
        require_nexus=not args.allow_missing_nexus,
        require_runs=not args.allow_missing_runs,
        require_phyla_embedding=not args.allow_missing_phyla_embedding,
    )
    all_rows = list(
        _iter_ok_manifest_rows(
            stream_root=stream_root,
            supported_dataset_ids=supported_ids,
        )
    )
    selected_rows = _select_rows(
        all_rows,
        max_cases=int(args.max_cases),
        max_datasets=int(args.max_datasets),
        per_dataset_cap=int(args.per_dataset_cap),
        seed=int(args.seed),
    )
    if not selected_rows:
        raise RuntimeError("No usable topology-stream rows were selected")

    start_json_dir = output_root / "start_json"
    target_json_dir = output_root / "target_json"
    by_dataset_dir = output_root / "branch_relax" / "by_dataset"

    width = max(6, len(str(len(selected_rows) - 1)))
    cases: list[dict[str, Any]] = []
    start_json_paths: list[str] = []
    target_json_paths: list[str] = []
    start_trees: list[str] = []
    target_trees: list[str] = []
    per_dataset: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: {"start_trees": [], "target_trees": []}
    )
    leaf_counts: list[int] = []

    for new_case_index, row in enumerate(selected_rows):
        dataset_id = str(row["dataset_id"])
        case_key = f"{args.case_prefix}_case{new_case_index:0{width}d}"
        start_path = _localize_stream_path(row["start_path"], stream_root)
        target_path = _localize_stream_path(row["target_path"], stream_root)
        row["local_start_path"] = str(start_path)
        row["local_target_path"] = str(target_path)
        start_payload, start_tree = _load_tree_payload(
            start_path,
            ("start_tree", "tree", "final_tree"),
        )
        target_payload, target_tree = _load_tree_payload(
            target_path,
            ("target_tree", "tree", "final_tree"),
        )
        num_leaves = int(row.get("num_leaves") or 0)
        if num_leaves > 0:
            leaf_counts.append(num_leaves)

        start_out = start_json_dir / f"{case_key}_start.json"
        target_out = target_json_dir / f"{case_key}_target.json"
        common_metadata = {
            "group_key": case_key,
            "dataset_id": dataset_id,
            "case_index": int(new_case_index),
            "original_group_key": start_payload.get("group_key")
            or target_payload.get("group_key"),
            "original_start_path": str(row["local_start_path"]),
            "original_target_path": str(row["local_target_path"]),
            "worker_index": row.get("worker_index"),
            "worker_case_index": row.get("case_index"),
            "manifest_path": row.get("manifest_path"),
            "manifest_line_number": row.get("manifest_line_number"),
            "topology_key_hash": row.get("topology_key_hash"),
            "topology_probability": row.get("topology_probability"),
            "topology_count": row.get("topology_count"),
            "posterior_index": row.get("posterior_index"),
            "num_leaves": num_leaves or None,
        }
        normalized_start_payload = dict(start_payload)
        normalized_start_payload.update(common_metadata)
        normalized_start_payload["tree"] = start_tree
        normalized_target_payload = dict(target_payload)
        normalized_target_payload.update(common_metadata)
        normalized_target_payload["tree"] = target_tree
        _write_json(start_out, normalized_start_payload)
        _write_json(target_out, normalized_target_payload)

        start_json_paths.append(str(start_out))
        target_json_paths.append(str(target_out))
        start_trees.append(start_tree)
        target_trees.append(target_tree)
        per_dataset[dataset_id]["start_trees"].append(start_tree)
        per_dataset[dataset_id]["target_trees"].append(target_tree)
        cases.append(
            {
                **common_metadata,
                "start_json": str(start_out),
                "target_json": str(target_out),
            }
        )

    _write_lines(output_root / "start_json_paths.txt", start_json_paths)
    _write_lines(output_root / "target_json_paths.txt", target_json_paths)
    _write_lines(output_root / "start_trees.txt", start_trees)
    _write_lines(output_root / "target_trees.txt", target_trees)

    dataset_spec_rows = []
    dataset_spec_args = []
    for dataset_id in sorted(per_dataset):
        safe_dataset_id = _safe_id(dataset_id)
        start_list = by_dataset_dir / f"{safe_dataset_id}_start_trees.txt"
        target_list = by_dataset_dir / f"{safe_dataset_id}_target_trees.txt"
        _write_lines(start_list, per_dataset[dataset_id]["start_trees"])
        _write_lines(target_list, per_dataset[dataset_id]["target_trees"])
        count = len(per_dataset[dataset_id]["start_trees"])
        dataset_spec_rows.append(
            "\t".join([dataset_id, str(start_list), str(target_list), str(count)])
        )
        dataset_spec_args.append(f"--dataset-spec {dataset_id}:{start_list}:{target_list}")

    _write_lines(output_root / "branch_relax" / "dataset_specs.tsv", dataset_spec_rows)
    _write_lines(output_root / "branch_relax" / "dataset_specs.args", dataset_spec_args)

    size_min = min(leaf_counts) if leaf_counts else None
    size_max = max(leaf_counts) if leaf_counts else None
    metric_sizes = (
        f"{size_min}-{size_max}" if size_min is not None and size_max is not None else None
    )
    manifest = {
        "stream_root": str(stream_root),
        "nexus_root": str(nexus_root),
        "runs_root": str(runs_root),
        "phyla_embedding_dir": str(phyla_embedding_dir),
        "output_root": str(output_root),
        "case_prefix": str(args.case_prefix),
        "seed": int(args.seed),
        "selected_cases": len(cases),
        "available_cases_after_support_filter": len(all_rows),
        "selected_dataset_count": len(per_dataset),
        "selected_dataset_ids": sorted(per_dataset),
        "num_leaves_min": size_min,
        "num_leaves_max": size_max,
        "metric_encoder_sizes": metric_sizes,
        "start_json_glob": str(start_json_dir / "*_start.json"),
        "start_json_paths_file": str(output_root / "start_json_paths.txt"),
        "target_json_paths_file": str(output_root / "target_json_paths.txt"),
        "start_trees_file": str(output_root / "start_trees.txt"),
        "target_trees_file": str(output_root / "target_trees.txt"),
        "branch_relax_dataset_specs_tsv": str(
            output_root / "branch_relax" / "dataset_specs.tsv"
        ),
        "branch_relax_dataset_specs_args": str(
            output_root / "branch_relax" / "dataset_specs.args"
        ),
        "cases": cases,
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-root", type=Path, default=DEFAULT_STREAM_ROOT)
    parser.add_argument("--nexus-root", type=Path, default=DEFAULT_NEXUS_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument(
        "--phyla-embedding-dir",
        type=Path,
        default=DEFAULT_PHYLA_EMBEDDING_DIR,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case-prefix", default="realstream")
    parser.add_argument("--max-cases", type=int, default=4096)
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--per-dataset-cap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--allow-missing-nexus", action="store_true")
    parser.add_argument("--allow-missing-runs", action="store_true")
    parser.add_argument("--allow-missing-phyla-embedding", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    manifest = prepare(parse_args(argv))
    print(
        json.dumps(
            {
                "manifest": str(Path(manifest["output_root"]) / "manifest.json"),
                "selected_cases": manifest["selected_cases"],
                "selected_dataset_count": manifest["selected_dataset_count"],
                "metric_encoder_sizes": manifest["metric_encoder_sizes"],
                "start_json_glob": manifest["start_json_glob"],
                "branch_relax_dataset_specs_tsv": manifest[
                    "branch_relax_dataset_specs_tsv"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
