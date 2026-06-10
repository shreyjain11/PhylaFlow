#!/usr/bin/env python
"""Build a compact JSONL index for real topology-stream training.

This script intentionally does not copy or rewrite per-case JSON payloads. It
only scans worker manifests, filters to supported datasets, localizes the
existing start/target JSON paths, and writes one compact row per selected case.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_real_topology_stream_artifacts import (  # noqa: E402
    DEFAULT_NEXUS_ROOT,
    DEFAULT_PHYLA_EMBEDDING_DIR,
    DEFAULT_RUNS_ROOT,
    DEFAULT_STREAM_ROOT,
    _iter_ok_manifest_rows,
    _localize_stream_path,
    _select_rows,
    _supported_dataset_ids,
)

DEFAULT_OUTPUT = (
    Path("/ewsc/yektefai/phylaflow_datasets")
    / "real_topology_stream_index_20260502.jsonl"
)


def _compact_record(row: dict[str, Any], *, stream_root: Path, case_index: int) -> dict[str, Any]:
    dataset_id = str(row["dataset_id"]).upper()
    start_path = _localize_stream_path(str(row["start_path"]), stream_root)
    target_path = _localize_stream_path(str(row["target_path"]), stream_root)
    record = {
        "case_index": int(case_index),
        "dataset_id": dataset_id,
        "start_path": str(start_path),
        "target_path": str(target_path),
    }
    for key in (
        "num_leaves",
        "worker_index",
        "case_index",
        "posterior_index",
        "topology_key_hash",
        "topology_probability",
        "topology_count",
        "anchors_path",
        "manifest_path",
        "manifest_line_number",
    ):
        if key in row and row.get(key) is not None:
            output_key = "worker_case_index" if key == "case_index" else key
            value = row.get(key)
            if key == "anchors_path":
                try:
                    value = str(_localize_stream_path(str(value), stream_root))
                except FileNotFoundError:
                    value = str(value)
            record[output_key] = value
    record["case_index"] = int(case_index)
    return record


def build_index(args: argparse.Namespace) -> dict[str, Any]:
    stream_root = args.stream_root.expanduser().resolve()
    nexus_root = args.nexus_root.expanduser().resolve()
    runs_root = args.runs_root.expanduser().resolve()
    phyla_embedding_dir = args.phyla_embedding_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()

    started = time.time()
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

    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_ids: set[str] = set()
    leaf_counts: list[int] = []
    with output.open("w", encoding="utf-8") as handle:
        for case_index, row in enumerate(selected_rows):
            record = _compact_record(row, stream_root=stream_root, case_index=case_index)
            dataset_ids.add(str(record["dataset_id"]))
            if record.get("num_leaves") is not None:
                leaf_counts.append(int(record["num_leaves"]))
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    return {
        "output": str(output),
        "selected_cases": len(selected_rows),
        "available_cases_after_support_filter": len(all_rows),
        "selected_dataset_count": len(dataset_ids),
        "num_leaves_min": min(leaf_counts) if leaf_counts else None,
        "num_leaves_max": max(leaf_counts) if leaf_counts else None,
        "elapsed_sec": time.time() - started,
    }


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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--per-dataset-cap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--allow-missing-nexus", action="store_true")
    parser.add_argument("--allow-missing-runs", action="store_true")
    parser.add_argument("--allow-missing-phyla-embedding", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    summary = build_index(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
