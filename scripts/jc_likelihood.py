"""Small JC likelihood scorer used by branch-length relaxation utilities.

The release code keeps this helper local so branch-length-sensitive evaluation
does not depend on historical experiment scripts. By default the scorer expects:

    ${PHYLAFLOW_DATA_ROOT}/<DATASET_ID>.pickle
    ${PHYLAFLOW_DATA_ROOT}/golden_run_data_DS1-8/<DATASET_ID>/rep_1/<DATASET_ID>.trprobs

The pickle should contain a mapping from translated taxon names to aligned DNA
strings. Trees passed to :meth:`GenericJCLikelihood.log_likelihood` should use
numeric leaf labels matching the MrBayes translate block.
"""

from __future__ import annotations

import math
import os
import pickle
import re
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from ete3 import Tree as EteTree


IUPAC_MASKS = {
    "A": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    "C": np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
    "G": np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float64),
    "T": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
    "U": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
    "R": np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float64),
    "Y": np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float64),
    "S": np.asarray([0.0, 1.0, 1.0, 0.0], dtype=np.float64),
    "W": np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float64),
    "K": np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
    "M": np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float64),
    "B": np.asarray([0.0, 1.0, 1.0, 1.0], dtype=np.float64),
    "D": np.asarray([1.0, 0.0, 1.0, 1.0], dtype=np.float64),
    "H": np.asarray([1.0, 1.0, 0.0, 1.0], dtype=np.float64),
    "V": np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float64),
    "N": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    "?": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
    "-": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
}


def _ensure_semicolon(newick: str) -> str:
    text = str(newick).strip()
    return text if text.endswith(";") else text + ";"


def _parse_translate_block(path: Path) -> Dict[int, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\btranslate\b(.*?);", text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        raise ValueError(f"No translate block found in {path}")
    translate: Dict[int, str] = {}
    for raw in match.group(1).split(","):
        item = raw.strip()
        if not item:
            continue
        fields = item.split(None, 1)
        if len(fields) != 2:
            continue
        key = int(fields[0])
        value = fields[1].strip().strip("'\"")
        translate[key] = value
    if not translate:
        raise ValueError(f"Empty translate block in {path}")
    return translate


def _encode_sequence(sequence: str) -> np.ndarray:
    return np.asarray(
        [IUPAC_MASKS.get(str(base).upper(), IUPAC_MASKS["?"]) for base in sequence],
        dtype=np.float64,
    )


class GenericJCLikelihood:
    """Jukes-Cantor pruning likelihood for a numeric-label Newick tree."""

    def __init__(
        self,
        *,
        dataset_id: str,
        dataset_pickle: str | os.PathLike[str] | None = None,
        golden_root: str | os.PathLike[str] | None = None,
        data_root: str | os.PathLike[str] | None = None,
        branch_length_floor: float = 1e-8,
    ) -> None:
        dataset_id = str(dataset_id).upper()
        root = Path(data_root or os.environ.get("PHYLAFLOW_DATA_ROOT", "."))
        dataset_pickle = Path(dataset_pickle or (root / f"{dataset_id}.pickle"))
        golden_root = Path(
            golden_root or (root / "golden_run_data_DS1-8" / dataset_id)
        )
        translation_path = golden_root / "rep_1" / f"{dataset_id}.trprobs"

        with dataset_pickle.open("rb") as handle:
            sequences: Dict[str, str] = pickle.load(handle)
        translate = _parse_translate_block(translation_path)
        ordered_taxa = [translate[index] for index in sorted(translate)]
        missing = [name for name in ordered_taxa if name not in sequences]
        if missing:
            raise ValueError(f"{dataset_pickle} is missing taxa: {missing}")

        lengths = {len(sequences[name]) for name in ordered_taxa}
        if len(lengths) != 1:
            raise ValueError(f"Sequences are not aligned to one length: {sorted(lengths)}")

        self.n_sites = int(next(iter(lengths)))
        self.branch_length_floor = float(branch_length_floor)
        self.transition_cache: Dict[float, np.ndarray] = {}
        self.leaf_vectors: Dict[str, np.ndarray] = {
            str(index): _encode_sequence(sequences[name])
            for index, name in enumerate(ordered_taxa, start=1)
        }

    def _leaf_vector(self, name: str) -> np.ndarray:
        key = str(name)
        cached = self.leaf_vectors.get(key)
        if cached is not None:
            return cached
        try:
            one_based = str(int(key) + 1)
        except Exception:
            one_based = ""
        cached = self.leaf_vectors.get(one_based)
        if cached is None:
            raise ValueError(f"Unknown leaf label {name!r}; expected numeric labels")
        return cached

    def _transition(self, branch_length: float) -> np.ndarray:
        length = max(float(branch_length), self.branch_length_floor)
        cached = self.transition_cache.get(length)
        if cached is not None:
            return cached
        decay = math.exp(-4.0 * length / 3.0)
        same = 0.25 + 0.75 * decay
        different = 0.25 - 0.25 * decay
        matrix = np.full((4, 4), different, dtype=np.float64)
        np.fill_diagonal(matrix, same)
        self.transition_cache[length] = matrix
        return matrix

    def log_likelihood(self, newick: str) -> float:
        tree = EteTree(_ensure_semicolon(newick), format=1, quoted_node_names=True)
        values: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        zeros = np.zeros(self.n_sites, dtype=np.float64)

        for node in tree.traverse("postorder"):
            if node.is_leaf():
                values[id(node)] = (self._leaf_vector(str(node.name)), zeros)
                continue

            clv = np.ones((self.n_sites, 4), dtype=np.float64)
            log_scale = np.zeros(self.n_sites, dtype=np.float64)
            for child in node.children:
                child_clv, child_log_scale = values[id(child)]
                contribution = child_clv @ self._transition(float(child.dist)).T
                clv *= contribution
                log_scale += child_log_scale

            site_scale = np.maximum(np.max(clv, axis=1), np.finfo(np.float64).tiny)
            clv = clv / site_scale[:, None]
            log_scale += np.log(site_scale)
            values[id(node)] = (clv, log_scale)

        root_clv, root_log_scale = values[id(tree)]
        site_likelihood = np.maximum(
            0.25 * np.sum(root_clv, axis=1),
            np.finfo(np.float64).tiny,
        )
        return float(np.sum(np.log(site_likelihood) + root_log_scale))

