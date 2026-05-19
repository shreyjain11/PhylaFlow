from utils.random_tree import Tree
from utils.bhv_utils import BHVEncoder
from collections import Counter
from typing import Dict, List, Tuple
from scipy.stats import pearsonr
import numpy as np
import math
import random
import subprocess, tempfile, re, pathlib
import logging
from Bio import AlignIO
from utils.utils import (
	jensenshannon_loglh_divergence,
	kl_loglh_divergence,
	return_total_tree_length,
)
from ete3 import Tree as EteTree

_LOGLH_RE = re.compile(r"final logLikelihood:\s*([-0-9.eE]+)")

enc = BHVEncoder()
logger = logging.getLogger(__name__)


def calculate_norm_rf(t1_nw: str, t2_nw: str) -> float:
	try:
		t1 = EteTree(t1_nw, format=1)
		t2 = EteTree(t2_nw, format=1)
		rf, max_rf, _, _, _, _, _ = t1.robinson_foulds(t2, unrooted_trees=True)
		return rf / max_rf if max_rf > 0 else 0.0
	except Exception as e:
		logger.warning("calculate_norm_rf failed: %s", e)
		return float("nan")


def _topology_leaf_sort_key(name: str) -> Tuple[int, str]:
    try:
        return (0, f"{int(name):020d}")
    except Exception:
        return (1, str(name))


def canonicalize_topology_newick(newick: str) -> str:
    """Return a branch-length-free canonical key for an unrooted topology."""
    tree = EteTree(newick, format=1, quoted_node_names=True)
    leaves = list(tree.iter_leaves())
    if not leaves:
        return ";"

    outgroup = min(leaves, key=lambda leaf: _topology_leaf_sort_key(leaf.name))
    tree.set_outgroup(outgroup)

    def emit(node) -> str:
        if node.is_leaf():
            return str(node.name)
        child_strings = sorted(emit(child) for child in node.children)
        return "(" + ",".join(child_strings) + ")"

    return emit(tree) + ";"


def _empirical_tree_topology_counts(trees: List[str]) -> Counter:
    counts = Counter()
    for tree in trees:
        try:
            counts[canonicalize_topology_newick(tree)] += 1
        except Exception as e:
            logger.warning("Skipping tree in topology counting: %s", e)
    return counts


def kl_divergence_tree_topology_distributions(
    posterior_trees: List[str],
    sampled_trees: List[str],
    alpha: float = 1e-6,
) -> Dict[str, float]:
    """KL D(posterior || sampled) over exact whole-tree topology frequencies."""
    posterior_counts = _empirical_tree_topology_counts(posterior_trees)
    sampled_counts = _empirical_tree_topology_counts(sampled_trees)

    support = set(posterior_counts.keys()).union(sampled_counts.keys())
    if not support:
        return {
            "kl_divergence_tree_topology": 0.0,
            "n_unique_posterior_topologies": 0.0,
            "n_unique_sampled_topologies": 0.0,
            "n_shared_topologies": 0.0,
            "posterior_topology_support_recall": 1.0,
        }

    posterior_total = float(sum(posterior_counts.values()))
    sampled_total = float(sum(sampled_counts.values()))
    zp = posterior_total + alpha * len(support)
    zq = sampled_total + alpha * len(support)

    kl = 0.0
    for key in support:
        p = (float(posterior_counts.get(key, 0.0)) + alpha) / zp
        q = (float(sampled_counts.get(key, 0.0)) + alpha) / zq
        kl += p * math.log(p / q)

    shared = len(set(posterior_counts.keys()).intersection(sampled_counts.keys()))
    unique_posterior = len(posterior_counts)
    return {
        "kl_divergence_tree_topology": float(kl),
        "n_unique_posterior_topologies": float(unique_posterior),
        "n_unique_sampled_topologies": float(len(sampled_counts)),
        "n_shared_topologies": float(shared),
        "posterior_topology_support_recall": (
            float(shared) / float(unique_posterior) if unique_posterior else 1.0
        ),
    }


def topk_posterior_tree_recall(
    posterior_trees: List[str],
    sampled_trees: List[str],
    top_ks: Tuple[int, ...] = (1, 5, 10, 20, 50),
) -> Dict[str, float]:
    """Recall of the top-k posterior topologies under sampled support."""
    posterior_counts = _empirical_tree_topology_counts(posterior_trees)
    sampled_support = set(_empirical_tree_topology_counts(sampled_trees).keys())
    if not posterior_counts:
        return {}

    ranked = sorted(
        posterior_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    posterior_total = float(sum(posterior_counts.values()))
    metrics = {}
    for raw_k in top_ks:
        k = max(1, int(raw_k))
        top_items = ranked[: min(k, len(ranked))]
        if not top_items:
            metrics[f"posterior_topology_recall_at_{k}"] = 1.0
            metrics[f"posterior_topology_mass_recall_at_{k}"] = 1.0
            continue
        hits = [key for key, _ in top_items if key in sampled_support]
        top_mass = sum(count for _, count in top_items)
        hit_mass = sum(posterior_counts[key] for key in hits)
        metrics[f"posterior_topology_recall_at_{k}"] = float(len(hits)) / float(
            len(top_items)
        )
        metrics[f"posterior_topology_mass_recall_at_{k}"] = (
            float(hit_mass) / float(top_mass) if top_mass > 0 else 1.0
        )
    metrics["posterior_topology_sample_support_size"] = float(len(sampled_support))
    metrics["posterior_topology_posterior_support_size"] = float(len(posterior_counts))
    metrics["posterior_topology_total_mass"] = float(posterior_total)
    return metrics


def kl_divergence_topological_distributions(
	posterior_trees: List[str],
	sampled_trees: List[str],
	num_leaves: int,
	eps: float = 1e-8,
	alpha: float = 1e-6,
) -> Dict[str, float]:
	"""Compute KL divergence between topological distributions of two sets of trees."""

	full_mask = (1 << num_leaves) - 1

	def return_splits(nw):
		t = Tree(nw)
		enc = BHVEncoder()
		masks, lens = enc.return_BHV_encoding(t)
		return masks

	split_counts_ground_truth = Counter()
	for t in posterior_trees:
		splits = return_splits(t)
		split_counts_ground_truth.update(splits)

	gt_topological_distribution = {
		k: v / sum(split_counts_ground_truth.values())
		for k, v in split_counts_ground_truth.items()
	}

	split_counts_sampled = Counter()
	for t in sampled_trees:
		splits = return_splits(t)
		split_counts_sampled.update(splits)

	sampled_topological_distribution = {
		k: v / sum(split_counts_sampled.values())
		for k, v in split_counts_sampled.items()
	}

	support = set(gt_topological_distribution.keys()).union(
		set(sampled_topological_distribution.keys())
	)
	ZP = 1.0 + alpha * len(support)
	ZQ = 1.0 + alpha * len(support)

	kl = 0.0
	for k in support:
		p = (gt_topological_distribution.get(k, 0.0) + alpha) / ZP
		q = (sampled_topological_distribution.get(k, 0.0) + alpha) / ZQ
		kl += p * (math.log(p / q) / math.log(math.e))
	return {"kl_divergence_topological": kl}


def split_bipartition_frequency_correlation(
	posterior_trees: List[str],
	sampled_trees: List[str],
	num_leaves: int,
	eps: float = 1e-8,
) -> Dict[str, float]:
	"""Compute correlation between split bipartition frequencies of two sets of trees."""

	full_mask = (1 << num_leaves) - 1

	def generate_splits(nw):
		t = Tree(nw)
		enc = BHVEncoder()
		masks, lens = enc.return_BHV_encoding(t)
		return masks

	split_counts_ground_truth = Counter()
	for t in posterior_trees:
		splits = generate_splits(t)
		split_counts_ground_truth.update(splits)

	split_counts_sampled = Counter()
	for t in sampled_trees:
		splits = generate_splits(t)
		split_counts_sampled.update(splits)

	all_splits = set(split_counts_ground_truth.keys()).union(
		set(split_counts_sampled.keys())
	)
	gt_freqs = []
	sampled_freqs = []
	for s in all_splits:
		gt_freqs.append(split_counts_ground_truth.get(s, 0) / len(posterior_trees))
		sampled_freqs.append(split_counts_sampled.get(s, 0) / len(sampled_trees))

	correlation, _ = pearsonr(gt_freqs, sampled_freqs)
	return {"bipartition_frequency_correlation": correlation}


def raxmlng_loglh_batch(
	nexus_path: str,
	newicks: List[str],
	model: str = "JC",
	threads: int = 1,
) -> List[float]:
	"""
	Returns log p(Y | tree, branch_lengths, model) for multiple trees using RAxML-NG --loglh.
	Assumes Newick includes branch lengths.
	"""
	if not newicks:
		return []

	with tempfile.TemporaryDirectory() as td_trees, tempfile.TemporaryDirectory() as td_msa:
		td_trees = pathlib.Path(td_trees)
		td_msa = pathlib.Path(td_msa)

		# ---- Write trees ----
		tree_file = td_trees / "trees.nwk"
		tree_file.write_text("\n".join(t.strip() for t in newicks) + "\n")

		# ---- Convert NEXUS -> FASTA ----
		msa_file = td_msa / "msa.fasta"

		# Option A (recommended): AlignIO.convert
		AlignIO.convert(
			nexus_path,
			"nexus",
			msa_file,
			"fasta",
		)

		cmd = [
			"raxml-ng",
			"--loglh",
			"--msa",
			str(msa_file),
			"--tree",
			str(tree_file),
			"--model",
			model,
			"--threads",
			str(threads),
		]

		p = subprocess.run(cmd, capture_output=True, text=True)
		out = (p.stdout or "") + "\n" + (p.stderr or "")
		if p.returncode != 0:
			raise RuntimeError(f"RAxML-NG failed:\n{out}")

		# Parse all log-likelihoods from output
		loglhs = [float(m.group(1)) for m in _LOGLH_RE.finditer(out)]

		if len(loglhs) != len(newicks):
			raise RuntimeError(
				f"Expected {len(newicks)} loglh values, got {len(loglhs)}:\n{out}"
			)

		return loglhs


def compare_likelihood_distributions(
	nexus_file_path: str,
	true_trees: List[str],
	sampled_trees: List[str],
	threads: int = 1,
) -> Dict[str, float]:
	"""Compare likelihood distributions of true and sampled trees using RAxML-NG."""
	true_loglhs = raxmlng_loglh_batch(
		nexus_path=nexus_file_path, newicks=true_trees, model="JC", threads=threads
	)

	sampled_loglhs = raxmlng_loglh_batch(
		nexus_path=nexus_file_path, newicks=sampled_trees, model="JC", threads=threads
	)

	avg_true_loglh = (
		sum(true_loglhs) / len(true_loglhs) if true_loglhs else float("-inf")
	)
	avg_sampled_loglh = (
		sum(sampled_loglhs) / len(sampled_loglhs) if sampled_loglhs else float("-inf")
	)

	# Difference in average log-likelihoods
	diff_avg_loglh = avg_true_loglh - avg_sampled_loglh
	js_div = jensenshannon_loglh_divergence(true_loglhs, sampled_loglhs, bins=50)
	kl_div = kl_loglh_divergence(true_loglhs, sampled_loglhs, bins=50)

	return {
		"avg_true_loglh": avg_true_loglh,
		"avg_sampled_loglh": avg_sampled_loglh,
		"diff_avg_loglh": diff_avg_loglh,
		"js_divergence_loglh": js_div,
		"kl_divergence_loglh": kl_div,
	}


def compare_branch_length_distributions(
	true_trees: List[str], sampled_trees: List[str]
) -> Dict[str, float]:
	"""Compare branch length distributions of true and sampled trees."""
	true_branch_lengths = []
	for newick in true_trees:
		true_branch_lengths.append(return_total_tree_length(newick))

	sampled_branch_lengths = []
	for newick in sampled_trees:
		sampled_branch_lengths.append(return_total_tree_length(newick))

	js_div = jensenshannon_loglh_divergence(
		true_branch_lengths, sampled_branch_lengths, bins=50
	)
	kl_div = kl_loglh_divergence(true_branch_lengths, sampled_branch_lengths, bins=50)

	return {
		"js_divergence_branch_length": js_div,
		"kl_divergence_branch_length": kl_div,
	}


def load_sample_trprobs(
	path: str, max_trees: int = 1000
) -> Tuple[List[str], Dict[str, str]]:
	"""Load sampled Newick trees and the translation map from a .tprobs file."""
	trees: List[str] = []
	weights: List[float] = []
	translation: Dict[str, str] = {}
	tree_buf: List[str] = []
	in_tree = False
	in_translate = False

	def parse_weight(line: str) -> float | None:
		m = re.search(r"p\s*=\s*([0-9.eE+-]+)", line)
		if m:
			return float(m.group(1))
		m = re.search(r"&W\s*([0-9.eE+-]+)", line)
		if m:
			return float(m.group(1))
		return None

	def finalize_tree(raw: str, weight: float | None) -> None:
		raw = raw.strip()
		while raw.startswith("["):
			end = raw.find("]")
			if end == -1:
				break
			raw = raw[end + 1 :].lstrip()
		raw = raw.rstrip(";").strip()
		if not raw:
			return
		trees.append(raw)
		weights.append(weight if weight is not None else 0.0)

	current_weight = None

	with open(path, "r") as handle:
		for raw_line in handle:
			line = raw_line.strip()
			lower = line.lower()

			if lower.startswith("translate"):
				in_translate = True
				continue
			if in_translate:
				if line:
					match = re.match(r"^(\d+)\s+([^,;]+)", line.rstrip(",;"))
					if match:
						translation[match.group(1)] = match.group(2)
				if ";" in line:
					in_translate = False
				continue

			if lower.startswith("tree "):
				in_tree = True
				current_weight = parse_weight(line)
				after_eq = line.split("=", 1)[-1]
				tree_buf.append(after_eq)
				if ";" in line:
					in_tree = False
			elif in_tree:
				tree_buf.append(line)
				if ";" in line:
					in_tree = False

			if not in_tree and tree_buf:
				tree_str = " ".join(tree_buf)
				tree_buf = []
				finalize_tree(tree_str, current_weight)
				current_weight = None

	if not trees:
		return [], translation

	total = sum(weights)
	if total <= 0.0:
		weights = [1.0 / len(trees)] * len(trees)
	else:
		weights = [w / total for w in weights]

	if max_trees <= 0:
		return [], translation

	sampled_trees = random.choices(trees, weights=weights, k=max_trees)
	result = []
	for t in sampled_trees:
		m = re.search(r"\(.*\)", t.strip())
		if m:
			# Apply translation mapping to convert numeric labels to taxon names
			import pdb

			pdb.set_trace()
			# BELOW will error but not changing cause we may be deleting this anyways
			translated = translate_tree_labels(m.group(0), translation)
			result.append(translated)
	sampled_trees = result

	return sampled_trees, translation


# samples, translation = load_sample_trprobs("./benchmark_data/DS1/rep_1/DS1.trprobs")
# import pdb; pdb.set_trace()
