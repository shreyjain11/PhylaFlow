#!/usr/bin/env python3
"""
Utility script to reorganize finished MrBayes runs and their corresponding
Nexus alignments into the directory layout expected by the PhylaFlow data
pipeline.

The PhylaFlow ``TreeDataset`` class expects a ``data_root`` with two
subdirectories:

* ``nexus/`` – containing one nexus file per dataset ID (``<id>.nex`` or
  ``<id>.nexus``).
* ``runs/`` – containing a subdirectory per ID, each holding the MrBayes
  output files for that ID (e.g., ``<id>_DNA.run1.t`` and ``<id>_DNA.run2.t``).

When a run completes, the user's working directory may contain a
``done_files`` folder with the finished ``.nex`` files and an ``output``
folder with subdirectories such as ``11193_NT_AL`` that hold the
corresponding MrBayes output. This script copies (or moves) those files
into a new root (``--output_dir``) arranged according to the layout above.

By default the script copies nexus files and moves the run directories so
that space can be reclaimed. You can customize the source and destination
roots via command‑line options.

Example usage from within the project root (the directory containing
``done_files`` and ``output``):

    python organize_data.py --root_dir=. --output_dir=./organized_data

This will create ``organized_data/nexus`` and ``organized_data/runs``,
copy all ``*.nex``/``*.nexus`` files from ``done_files`` into
``organized_data/nexus``, and move each matching subfolder from
``output`` into ``organized_data/runs``.
"""

import argparse
import os
import shutil
import sys
from typing import List


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {path}: {e}", file=sys.stderr)
        raise


def get_finished_ids(done_dir: str) -> List[str]:
    """
    Scan ``done_dir`` for nexus files and return a list of dataset IDs.

    An ID is defined as the base filename without its extension. Accepted
    extensions are ``.nex`` and ``.nexus`` (case‑insensitive).
    """
    ids: List[str] = []
    for fname in os.listdir(done_dir):
        base, ext = os.path.splitext(fname)
        if ext.lower() in {".nex", ".nexus"}:
            ids.append(base)
    return ids


def organize(root_dir: str, output_dir: str, dry_run: bool = False) -> None:
    """
    Perform the reorganization of finished runs.

    Parameters
    ----------
    root_dir : str
        The path to the directory containing ``done_files`` and ``output``.
    output_dir : str
        The destination root where ``nexus/`` and ``runs/`` will be created.
    dry_run : bool, optional
        If True, only print the planned operations without performing them.
    """
    done_dir = os.path.join(root_dir, "done_files")
    runs_src_root = os.path.join(root_dir, "output")
    if not os.path.isdir(done_dir):
        raise RuntimeError(f"done_files directory not found at {done_dir}")
    if not os.path.isdir(runs_src_root):
        raise RuntimeError(f"output directory not found at {runs_src_root}")

    nexus_dest = os.path.join(output_dir, "nexus")
    runs_dest_root = os.path.join(output_dir, "runs")
    ensure_dir(nexus_dest)
    ensure_dir(runs_dest_root)

    ids = get_finished_ids(done_dir)
    if not ids:
        print("No completed .nex or .nexus files found in done_files; nothing to do.")
        return

    print(f"Found {len(ids)} completed IDs: {', '.join(ids)}")

    for id_ in ids:
        # Copy nexus file
        # Determine whether .nex or .nexus exists
        src_nex = None
        for ext in (".nex", ".nexus", ".NEX", ".NEXUS"):
            candidate = os.path.join(done_dir, id_ + ext)
            if os.path.isfile(candidate):
                src_nex = candidate
                break
        if src_nex is None:
            print(f"Warning: no .nex/.nexus file found for ID {id_} in {done_dir}")
        else:
            dest_nex = os.path.join(nexus_dest, os.path.basename(src_nex))
            if dry_run:
                print(f"Would copy {src_nex} -> {dest_nex}")
            else:
                shutil.copy2(src_nex, dest_nex)

        # Move run directory if it exists
        src_run_dir = os.path.join(runs_src_root, id_)
        dest_run_dir = os.path.join(runs_dest_root, id_)
        if os.path.isdir(src_run_dir):
            if dry_run:
                print(f"Would move {src_run_dir} -> {dest_run_dir}")
            else:
                # Ensure destination does not already exist
                if os.path.exists(dest_run_dir):
                    raise RuntimeError(f"Destination run directory already exists: {dest_run_dir}")
                shutil.move(src_run_dir, dest_run_dir)
        else:
            print(f"Warning: run directory not found for ID {id_} at {src_run_dir}")

    print(f"Finished organizing {len(ids)} dataset(s). Data stored in {output_dir}")


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Organize finished Nexus and MrBayes outputs into the layout expected by PhylaFlow."
        )
    )
    parser.add_argument(
        "--root_dir",
        default=".",
        help="Directory containing 'done_files' and 'output' subdirectories.",
    )
    parser.add_argument(
        "--output_dir",
        default="./organized_data",
        help="Destination directory for organized 'nexus' and 'runs' subdirectories.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, only print what would be done without copying/moving files.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    organize(args.root_dir, args.output_dir, args.dry_run)


if __name__ == "__main__":
    main()