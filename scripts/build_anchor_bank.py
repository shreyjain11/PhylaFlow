"""Build direct-set velocity/first-hit anchor samples for fixed path banks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run.TrainingModule import _build_legacy_velocity_oracle_sample
from utils.bhv_utils import (
    return_sampled_tree_orthant_velocity,
    return_tree_boundary_merge_paths,
)
from utils.random_tree import Tree


def _read_pairs_jsonl(path: Path):
    pairs = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        record = json.loads(line)
        if "start_tree" not in record or "target_tree" not in record:
            raise ValueError(f"{path}:{line_number} missing start_tree/target_tree")
        pairs.append(record)
    if not pairs:
        raise ValueError(f"No pairs found in {path}")
    return pairs


def _phase_family_name(phase_idx: int) -> str:
    if phase_idx == 0:
        return "O0"
    if phase_idx == 1:
        return "A1"
    if phase_idx == 2:
        return "O2"
    return f"P{phase_idx}"


def _sample_phase_anchors(
    phase_source_tree: str,
    target_tree: str,
    *,
    bank_group_key: str,
    anchor_family: str,
    phase_idx: int,
    num_leaves: int,
    count: int,
):
    local_paths = return_tree_boundary_merge_paths(phase_source_tree, target_tree)
    if not local_paths:
        return []
    local_time = float(local_paths[0]["global_time"])
    if local_time <= 0.0:
        fractions = [0.0] * int(count)
    elif int(count) == 1:
        fractions = [0.0]
    else:
        fractions = [0.0, 0.25, 0.5, 0.75][: int(count)]
        while len(fractions) < int(count):
            fractions.append(fractions[-1])

    anchors = []
    next_boundary_tree = str(local_paths[0]["start_newick"])
    for idx, frac in enumerate(fractions):
        local_anchor_time = 0.0
        if frac > 0.0 and local_time > 0.0:
            local_anchor_time = min(frac * local_time * 0.95, max(local_time - 1e-8, 0.0))
        sampled_tree, sampled_velocity = return_sampled_tree_orthant_velocity(
            phase_source_tree,
            target_tree,
            local_anchor_time,
            legacy_training_semantics=False,
        )
        anchors.append(
            {
                "anchor_family": str(anchor_family),
                "source_checkpoint": f"{anchor_family.lower()}_{idx}",
                "path_index": int(phase_idx),
                "timepoint": float(phase_idx),
                "num_leaves": int(num_leaves),
                "newick_tree": str(sampled_tree),
                "target_tree": str(target_tree),
                "velocity": {int(k): float(v) for k, v in sampled_velocity.items()},
                "velocity_next_boundary_tree": next_boundary_tree,
                "local_anchor_time": float(local_anchor_time),
                "local_next_boundary_time": float(local_time),
                "bank_group_key": str(bank_group_key),
            }
        )
    return anchors


def build_anchor_payload(record, *, anchors_per_phase: int):
    start_tree = str(record["start_tree"])
    target_tree = str(record["target_tree"])
    bank_group_key = str(
        record.get("bank_group_key")
        or record.get("group_key")
        or record.get("case_id")
        or "case_0"
    )
    boundary_paths = return_tree_boundary_merge_paths(start_tree, target_tree)
    num_leaves = int(Tree(start_tree).n_leaves)
    phase_sources = [start_tree]
    phase_sources.extend(str(path["end_newick"]) for path in boundary_paths[:-1])

    anchors = []
    for phase_idx, phase_source in enumerate(phase_sources):
        family = _phase_family_name(int(phase_idx))
        phase_anchors = _sample_phase_anchors(
            phase_source,
            target_tree,
            bank_group_key=bank_group_key,
            anchor_family=family,
            phase_idx=int(phase_idx),
            num_leaves=num_leaves,
            count=int(anchors_per_phase),
        )
        canonical = _build_legacy_velocity_oracle_sample(
            phase_source,
            target_tree,
            timepoint=float(phase_idx),
            num_leaves=num_leaves,
        )
        if phase_anchors and canonical is not None:
            phase_anchors[0].update(
                {
                    "velocity": dict(canonical.get("velocity", {})),
                    "velocity_next_boundary_tree": canonical.get(
                        "velocity_next_boundary_tree"
                    ),
                    "source_checkpoint": None,
                }
            )
        anchors.extend(phase_anchors)
    return anchors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-jsonl", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--anchors-per-phase", type=int, default=4)
    args = parser.parse_args()

    anchors = []
    for record in _read_pairs_jsonl(args.pairs_jsonl):
        anchors.extend(
            build_anchor_payload(
                record,
                anchors_per_phase=int(args.anchors_per_phase),
            )
        )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(anchors, indent=2), encoding="utf-8")
    print(json.dumps({"anchors": len(anchors), "out_json": str(args.out_json)}))


if __name__ == "__main__":
    main()

