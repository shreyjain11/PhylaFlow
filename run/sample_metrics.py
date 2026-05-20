import json
import os
import subprocess
import time
from collections import Counter

import numpy as np
import torch
from ete3 import Tree as EteTree

from utils.random_tree import Tree
from utils.bhv_utils import (
    BHVEncoder,
    get_structural_polytomy_groups_from_newick,
    return_sampled_tree_orthant_velocity,
)
from utils.bhv_movie import build_tree_from_splits
from utils.metric_utils import (
    kl_divergence_topological_distributions,
    kl_divergence_tree_topology_distributions,
    topk_posterior_tree_recall,
    split_bipartition_frequency_correlation,
    compare_likelihood_distributions,
    compare_branch_length_distributions,
    calculate_norm_rf,
    canonicalize_topology_newick,
)
from utils.utils import (
    has_polytomy_fast,
    number_to_name_newick,
    resolve_polytomies_random_deterministic,
)
from run.training_helpers import *


class SampleMetricsMixin:
    def _append_sample_metrics_trace(self, metrics):
        if not self.sample_metrics_trace_path:
            return

        payload = {
            "global_step": int(self.global_step),
            "stepper": int(self.stepper),
            "timestamp": time.time(),
        }
        for key, value in metrics.items():
            if not (
                self._metric_key_allowed(key)
                or self._metric_key_allowed(f"sample_metrics/{key}")
            ):
                continue
            if torch.is_tensor(value):
                payload[key] = float(value.detach().cpu().item())
            elif isinstance(value, np.generic):
                payload[key] = float(value)
            elif isinstance(value, (int, float, bool)):
                payload[key] = float(value) if isinstance(value, bool) else value
            else:
                payload[key] = value

        os.makedirs(os.path.dirname(self.sample_metrics_trace_path), exist_ok=True)
        with open(self.sample_metrics_trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _sample_metrics_tree_dump_output_dir(self):
        if self.sample_metrics_tree_dump_dir:
            return self.sample_metrics_tree_dump_dir
        if self.sample_metrics_trace_path:
            return os.path.join(
                os.path.dirname(self.sample_metrics_trace_path),
                "generated_trees",
            )
        return os.path.abspath("sample_metrics_generated_trees")

    def _sample_metrics_can_write_artifacts(self):
        trainer = getattr(self, "trainer", None)
        if trainer is not None and not bool(getattr(trainer, "is_global_zero", True)):
            return False
        try:
            return int(os.environ.get("RANK", "0")) == 0
        except Exception:
            return True

    @staticmethod
    def _sample_metrics_newick_line(tree):
        tree = str(tree).strip()
        return tree if tree.endswith(";") else tree + ";"

    @staticmethod
    def _sample_metrics_json_value(value):
        if torch.is_tensor(value):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return value.detach().cpu().tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _write_sample_metrics_tree_dump(self, rows, relaxed_tree_rows=None, train=True):
        if not self.sample_metrics_tree_dump_enabled:
            return {}
        if not rows or not self._sample_metrics_can_write_artifacts():
            return {}

        relaxed_by_index = {
            int(row.get("sample_index", index)): dict(row)
            for index, row in enumerate(relaxed_tree_rows or [])
        }
        split = "train" if train else "val"
        step = int(self.global_step)
        stepper = int(self.stepper)
        stamp = f"step{step:08d}_stepper{stepper:08d}_{split}"
        out_dir = self._sample_metrics_tree_dump_output_dir()
        os.makedirs(out_dir, exist_ok=True)
        jsonl_path = os.path.join(out_dir, f"{stamp}_trees.jsonl")
        sampled_path = os.path.join(out_dir, f"{stamp}_sampled_trees.txt")
        relaxed_path = os.path.join(out_dir, f"{stamp}_relaxed_trees.txt")

        payloads = []
        sampled_lines = []
        relaxed_lines = []
        base_payload = {
            "global_step": step,
            "stepper": stepper,
            "timestamp": time.time(),
            "split": split,
        }
        for index, row in enumerate(rows):
            row = dict(row)
            payload = dict(base_payload)
            payload["index"] = int(index)
            field_map = {
                "_start_tree": "start_tree",
                "_original_start_tree": "original_start_tree",
                "_target_tree": "target_tree",
                "_sampled_tree": "sampled_tree",
                "_bank_group_key": "bank_group_key",
                "_source_bank_index": "source_bank_index",
                "_n_leaves": "n_leaves",
            }
            for private_key, public_key in field_map.items():
                if private_key in row and row[private_key] is not None:
                    payload[public_key] = self._sample_metrics_json_value(
                        row[private_key]
                    )
            for key, value in row.items():
                if str(key).startswith("_") or key in payload:
                    continue
                payload[key] = self._sample_metrics_json_value(value)
            relaxed_row = relaxed_by_index.get(index)
            if relaxed_row:
                for key, value in relaxed_row.items():
                    if key == "sample_index":
                        continue
                    payload[key] = self._sample_metrics_json_value(value)
            sampled_tree = payload.get("sampled_tree")
            if sampled_tree:
                sampled_lines.append(self._sample_metrics_newick_line(sampled_tree))
            relaxed_tree = payload.get("relaxed_tree")
            if relaxed_tree:
                relaxed_lines.append(self._sample_metrics_newick_line(relaxed_tree))
            payloads.append(payload)

        with open(jsonl_path, "w", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        with open(sampled_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(sampled_lines) + ("\n" if sampled_lines else ""))
        if relaxed_lines:
            with open(relaxed_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(relaxed_lines) + "\n")

        metrics = {
            "tree_dump_written": 1.0,
            "tree_dump_count": float(len(payloads)),
        }
        if relaxed_lines:
            metrics["tree_dump_relaxed_count"] = float(len(relaxed_lines))
        return metrics

    def _get_harness_sampling_pair(self, train=True, frozen_start_bank=False):
        cache_key = "train" if train else "val"

        dataset_split = self.dataset.dataset_train if train else self.dataset.dataset_val
        use_random_fixed_pair_bank = (
            bool(train)
            and getattr(dataset_split, "overfit_fixed_pair", False)
        )
        cached = None if use_random_fixed_pair_bank else self._cached_harness_sampling_pairs.get(cache_key)
        if cached is not None:
            return cached

        if use_random_fixed_pair_bank:
            sampled_pair = None
            sampled_pair = self._sample_overfit_fixed_pair_bank_pair_for_harness(
                dataset_split,
                frozen_start_bank=frozen_start_bank,
            )
            if sampled_pair is not None:
                start_tree = sampled_pair["random_tree"]
                target_tree = sampled_pair["effective_target_tree"]
                return {
                    "start_tree": start_tree,
                    "target_tree": target_tree,
                    "bank_group_key": sampled_pair.get("bank_group_key"),
                    "n_leaves": len(EteTree(start_tree, format=1).get_leaves()),
                    "max_events": int(len(sampled_pair.get("final_labels", []))),
                    "name_mapping": (
                        dataset_split.return_nexus_number_to_name(0)
                        if hasattr(dataset_split, "return_nexus_number_to_name")
                        else None
                    ),
                }

            sampled = dataset_split[0]
            start_tree = sampled.get("start_tree")
            target_tree = sampled.get("target_tree")
            if start_tree and target_tree:
                return {
                    "start_tree": start_tree,
                    "target_tree": target_tree,
                    "bank_group_key": bank_group_key_value,
                    "n_leaves": len(EteTree(start_tree, format=1).get_leaves()),
                    "max_events": int(sampled.get("fixed_pair_num_events", 1024)),
                    "name_mapping": (
                        dataset_split.return_nexus_number_to_name(0)
                        if hasattr(dataset_split, "return_nexus_number_to_name")
                        else None
                    ),
                }

        fixed_pair = None
        if getattr(dataset_split, "overfit_fixed_pair", False):
            fixed_pair = dataset_split.get_overfit_fixed_pair(0)
        if fixed_pair is not None:
            start_tree = fixed_pair["random_tree"]
            target_tree = fixed_pair["effective_target_tree"]
            name_mapping = (
                dataset_split.return_nexus_number_to_name(0)
                if hasattr(dataset_split, "return_nexus_number_to_name")
                else fixed_pair.get("name_mapping")
            )
            pair = {
                "start_tree": start_tree,
                "target_tree": target_tree,
                "n_leaves": len(EteTree(start_tree, format=1).get_leaves()),
                "max_events": int(len(fixed_pair["final_labels"])),
                "name_mapping": name_mapping,
            }
            self._cached_harness_sampling_pairs[cache_key] = pair
            return pair

        random_state = random.getstate()
        try:
            random.seed(13)
            real_tree_raw = dataset_split.return_posterior_trees(0)[0]
            base_random_tree_raw, random_tree_raw = dataset_split.sample_random_tree_with_base(real_tree_raw)
            target_tree_raw = dataset_split.resolve_training_target_tree(
                random_tree_raw,
                real_tree_raw,
                base_start_tree_newick=base_random_tree_raw,
            )
            _, seq_ordering_map = _normalize_tree_like_dataset(real_tree_raw)
        finally:
            random.setstate(random_state)

        start_tree = _remap_tree_with_sequence_ordering(
            random_tree_raw,
            seq_ordering_map,
            offset=0,
            tree_kind="start tree",
        )
        target_tree = _remap_tree_with_sequence_ordering(
            target_tree_raw,
            seq_ordering_map,
            offset=0,
            tree_kind="target tree",
        )
        boundary_paths = return_tree_boundary_merge_paths(start_tree, target_tree)
        max_events = int(
            sum(len(path.get("events", [])) for path in boundary_paths)
        )
        pair = {
            "start_tree": start_tree,
            "target_tree": target_tree,
            "n_leaves": len(EteTree(target_tree, format=1).get_leaves()),
            "max_events": max_events,
            "name_mapping": (
                dataset_split.return_nexus_number_to_name(0)
                if hasattr(dataset_split, "return_nexus_number_to_name")
                else None
            ),
        }
        self._cached_harness_sampling_pairs[cache_key] = pair
        return pair

    def _get_fixed_pair_sampling_details(self, train=True):
        if not hasattr(self, "dataset") or self.dataset is None:
            return None
        dataset_split = self.dataset.dataset_train if train else self.dataset.dataset_val
        if dataset_split is None or not getattr(dataset_split, "overfit_fixed_pair", False):
            return None
        pair = dataset_split.get_overfit_fixed_pair(0)
        if pair is None:
            return None
        if pair.get("name_mapping") is None and hasattr(dataset_split, "return_nexus_number_to_name"):
            try:
                pair = dict(pair)
                pair["name_mapping"] = dataset_split.return_nexus_number_to_name(0)
            except Exception:
                pass
        return pair

    def _evaluate_fixed_pair_velocity_rows(self, fixed_pair):
        boundary_paths = fixed_pair["boundary_paths"]
        effective_target_tree = fixed_pair["effective_target_tree"]
        velocity_trees = [fixed_pair["random_tree"]]
        velocity_trees.extend(path["end_newick"] for path in boundary_paths[:-1])
        timepoints = [0.0]
        timepoints.extend(float(path["global_time"]) for path in boundary_paths[:-1])
        next_boundary_trees = [path["start_newick"] for path in boundary_paths]
        pair = {
            "start_tree": fixed_pair["random_tree"],
            "target_tree": effective_target_tree,
            "bank_group_key": fixed_pair.get("bank_group_key"),
            "n_leaves": len(EteTree(fixed_pair["random_tree"], format=1).get_leaves()),
            "name_mapping": fixed_pair.get("name_mapping"),
            "max_events": len(fixed_pair.get("final_labels", []) or []),
        }
        sample_kwargs = self._build_harness_sample_kwargs(pair, train=True)
        phyla_embeddings = sample_kwargs.get("phyla_embeddings")
        case_indices = sample_kwargs.get("case_indices")
        start_topology_features = sample_kwargs.get(
            "first_hit_start_topology_features"
        )
        start_topology_embeddings = sample_kwargs.get(
            "first_hit_start_topology_embeddings"
        )
        start_topology_pad_mask = sample_kwargs.get(
            "first_hit_start_topology_pad_mask"
        )
        start_tree_graph_context = sample_kwargs.get(
            "first_hit_start_tree_graph_context"
        )

        rows = []
        for idx, (source_tree, next_boundary_tree, model_time) in enumerate(
            zip(velocity_trees, next_boundary_trees, timepoints)
        ):
            input_newick, true_velocity = return_sampled_tree_orthant_velocity(
                source_tree,
                effective_target_tree,
                0.0,
            )
            with torch.no_grad():
                tokenized = _move_tokenized_batch_to_device(
                    self.model.tokenizer([input_newick]),
                    self.device,
                )
                (
                    pred_velocity,
                    edge_splits,
                    _edge_mask,
                    first_hit_logits,
                    _boundary_vanish_logits,
                    edge_features,
                ) = self.forward(
                    tokenized,
                    float(model_time),
                    phyla_embeddings,
                    first_hit_case_indices=case_indices,
                    first_hit_start_topology_features=start_topology_features,
                    first_hit_start_topology_embeddings=start_topology_embeddings,
                    first_hit_start_topology_pad_mask=start_topology_pad_mask,
                    first_hit_start_tree_graph_context=start_tree_graph_context,
                )

            model_masks = [int(mask) for mask in edge_splits[0]]
            mask_to_idx = {mask: i for i, mask in enumerate(model_masks)}
            target_vals = torch.zeros(len(model_masks), dtype=torch.float32)
            for split_mask, value in true_velocity.items():
                idx_match = mask_to_idx.get(int(split_mask))
                if idx_match is None:
                    continue
                target_vals[idx_match] = float(value)

            source_tree_obj = Tree(input_newick)
            source_masks, source_lengths = BHVEncoder().return_BHV_encoding(
                source_tree_obj
            )
            length_map = {
                int(mask): float(length)
                for mask, length in zip(source_masks, source_lengths)
                if length is not None
            }
            split_masks_nonzero = [mask for mask in model_masks if int(mask) != 0]
            real_max_bit = max(
                (int(mask).bit_length() for mask in split_masks_nonzero),
                default=0,
            )
            full_mask = (1 << real_max_bit) - 1 if real_max_bit > 0 else 0
            lengths = torch.zeros(len(model_masks), dtype=torch.float32)
            for idx_mask, mask in enumerate(model_masks):
                edge_len = length_map.get(int(mask))
                if edge_len is None and full_mask:
                    edge_len = length_map.get(full_mask ^ int(mask))
                if edge_len is not None and float(edge_len) > 0.0:
                    lengths[idx_mask] = float(edge_len)

            candidate_mask = _sampling_supervised_candidate_mask(
                model_masks,
                lengths.cpu().numpy(),
                pair["n_leaves"],
            )

            first_hit_stats = {
                "precision": 0.0,
                "recall": 0.0,
            }
            if first_hit_logits is not None:
                pred_first_mask, _raw_first_count, _used_first_fallback = (
                    _predict_first_hit_mask_with_fallback(
                        first_hit_logits[0].squeeze(1).detach().cpu().numpy(),
                        candidate_mask,
                    )
                )
                true_first_mask = _oracle_first_hit_mask_for_sampling(
                    input_newick,
                    effective_target_tree,
                    masks=model_masks,
                    lengths=lengths.cpu().numpy(),
                    n_leaves=pair["n_leaves"],
                    supervised_mask=candidate_mask,
                    velocity_sign_eps=float(self.velocity_sign_eps),
                    dt_eps=float(self.velocity_dt_eps),
                    first_hit_tol=1e-4,
                )
                first_hit_stats = _mask_precision_recall(
                    pred_first_mask,
                    true_first_mask,
                )

            rows.append(
                {
                    "index": int(idx),
                    "timepoint": float(model_time),
                    "first_hit_precision": float(first_hit_stats["precision"]),
                    "first_hit_recall": float(first_hit_stats["recall"]),
                }
            )
        return rows

    def _labels_to_subset_tuples(self, labels):
        subsets = set()
        for label in labels:
            components = [int(component) for component in label["components"]]
            subset = tuple(
                sorted(int(components[idx]) for idx in label["merge_indices"])
            )
            subsets.add(subset)
        return subsets

    def _evaluate_fixed_pair_autoregressive_rows(self, fixed_pair):
        rows = []
        final_labels = fixed_pair["final_labels"]
        max_events = len(final_labels)
        pair = {
            "start_tree": fixed_pair["random_tree"],
            "target_tree": fixed_pair["effective_target_tree"],
            "bank_group_key": fixed_pair.get("bank_group_key"),
            "n_leaves": len(EteTree(fixed_pair["random_tree"], format=1).get_leaves()),
            "name_mapping": fixed_pair.get("name_mapping"),
            "max_events": len(final_labels),
        }
        sample_kwargs = self._build_harness_sample_kwargs(pair, train=True)
        phyla_embeddings = sample_kwargs.get("phyla_embeddings")
        case_indices = sample_kwargs.get("case_indices")
        for event_idx, event in enumerate(final_labels):
            current_newick = event["newick"]
            component_groups = [get_structural_polytomy_groups_from_newick(current_newick)]
            with torch.no_grad():
                tokenized = _move_tokenized_batch_to_device(
                    self.model.tokenizer([current_newick]),
                    self.device,
                )
                outputs = self.forward(
                    tokenized,
                    self._sampling_autoregressive_time_tensor(
                        0.0,
                        event_index=event_idx,
                        max_events=max_events,
                    ),
                    phyla_embeddings,
                    autoregressive=True,
                    autoregressive_component_groups=component_groups,
                    autoregressive_case_indices=case_indices,
                    autoregressive_start_topology_features=sample_kwargs.get(
                        "autoregressive_start_topology_features"
                    ),
                )
            existing_splits = {
                int(mask)
                for mask in self.model.tokenizer([current_newick])[-1][0]
                if int(mask) != 0
            }
            planned = _plan_autoregressive_boundary_merges(outputs, existing_splits)
            pred_subsets = set()
            if planned:
                for subset, _new_split in planned[0]["subsets"]:
                    pred_subsets.add(
                        tuple(sorted(int(component) for component in subset))
                    )
            true_subsets = self._labels_to_subset_tuples(event["labels"])
            rows.append(
                {
                    "event_index": int(event_idx),
                    "exact_match": pred_subsets == true_subsets,
                }
            )
        return rows

    def _evaluate_fixed_pair_path_metrics(self, train=True):
        fixed_pair = self._get_fixed_pair_sampling_details(train=train)
        if fixed_pair is None:
            return {}

        velocity_rows = self._evaluate_fixed_pair_velocity_rows(fixed_pair)
        autoregressive_rows = self._evaluate_fixed_pair_autoregressive_rows(fixed_pair)
        metrics = _summarize_fixed_pair_eval_rows(velocity_rows, autoregressive_rows)

        pair = {
            "start_tree": fixed_pair["random_tree"],
            "target_tree": fixed_pair["effective_target_tree"],
            "bank_group_key": fixed_pair.get("bank_group_key"),
            "n_leaves": len(EteTree(fixed_pair["random_tree"], format=1).get_leaves()),
            "name_mapping": fixed_pair.get("name_mapping"),
            "max_events": len(fixed_pair.get("final_labels", []) or []),
        }
        sample_kwargs = self._build_harness_sample_kwargs(pair, train=train)
        sample_kwargs.pop("return_trace", None)
        with torch.no_grad():
            sampled_trees, _, _, _, _, trace = self.sample(
                [pair["start_tree"]],
                return_trace=True,
                **sample_kwargs,
            )
        sampled_tree = sampled_trees[0]
        sampled_velocity = len(trace.get("velocity", []))
        sampled_ar = len(trace.get("autoregressive", []))
        expected_velocity = len(fixed_pair.get("boundary_paths", []) or [])
        expected_ar = len(fixed_pair.get("final_labels", []) or [])
        metrics.update(
            {
                "fixed_path_sample_rf_norm": float(
                    calculate_norm_rf(sampled_tree, pair["target_tree"])
                ),
                "fixed_path_sampled_num_velocity_states": float(sampled_velocity),
                "fixed_path_sampled_num_autoregressive_events": float(sampled_ar),
                "fixed_path_expected_num_velocity_states": float(expected_velocity),
                "fixed_path_expected_num_autoregressive_events": float(expected_ar),
                "fixed_path_extra_velocity_states": float(
                    max(0, sampled_velocity - expected_velocity)
                ),
                "fixed_path_extra_autoregressive_events": float(
                    max(0, sampled_ar - expected_ar)
                ),
                "fixed_path_missing_velocity_states": float(
                    max(0, expected_velocity - sampled_velocity)
                ),
                "fixed_path_missing_autoregressive_events": float(
                    max(0, expected_ar - sampled_ar)
                ),
                "fixed_path_stopped_for_no_valid_merge": float(
                    1.0 if trace.get("stopped_for_no_valid_merge", False) else 0.0
                ),
                "fixed_path_stopped_for_repeated_topology": float(
                    1.0
                    if trace.get("stopped_for_repeated_topology", False)
                    else 0.0
                ),
            }
        )
        return metrics

    def _build_harness_sample_kwargs(
        self,
        pair,
        train=True,
        rollout_kind: str = "probe",
        **overrides,
    ):
        dataset_id = pair.get("dataset_id") or self._sample_metrics_dataset_id(train=train)
        phyla_embeddings = self._resolve_precomputed_phyla_embeddings_for_tree(
            pair["start_tree"],
            mapping=pair.get("name_mapping"),
            num_leaf=pair.get("n_leaves"),
            device=self.device,
            dataset_id=dataset_id,
        )
        max_steps = self.sampling_max_steps
        uncapped_events = self.sampling_max_events_uncapped
        max_events = None if uncapped_events else self.sampling_max_events
        if max_events is None and not uncapped_events:
            if int(pair.get("max_events", -1)) >= 0:
                max_events = int(pair["max_events"])
            else:
                max_events = 1024
        sample_kwargs = {
            "phyla_embeddings": phyla_embeddings,
            "dt_base": self.training_sampling_dt_base,
            "max_steps": max_steps,
            "max_events": max_events,
            "return_trace": True,
            "trace_state_rf": False,
            "explicit_autoregressive_component_groups": True,
            "target_trees": [pair["target_tree"]],
            "dataset_ids": [dataset_id] if dataset_id else None,
        }
        needs_frozen_ar_case_probe = (
            bool(
                getattr(
                    self.model,
                    "autoregressive_use_start_topology_conditioning",
                    False,
                )
            )
            and getattr(
                self.model,
                "autoregressive_start_topology_conditioning_mode",
                "additive",
            )
            in {"frozen_case_probe", "frozen_case_probe_additive"}
        )
        if (
            getattr(self.model, "first_hit_head_mode", "base")
            in {"case_adapted_mlp", "frozen_start_case_mlp"}
            or getattr(self.model, "autoregressive_use_case_conditioning", False)
            or needs_frozen_ar_case_probe
        ):
            case_index = _extract_case_index_from_group_key(pair.get("bank_group_key"))
            if case_index is not None:
                sample_kwargs["case_indices"] = [int(case_index)]
        needs_start_topology_summary = (
            getattr(self.model, "first_hit_head_mode", "base")
            in {
                "start_topology_adapter_mlp",
                "start_topology_raw_pool_concat_mlp",
            }
            or getattr(
                self.model,
                "autoregressive_use_start_topology_conditioning",
                False,
            )
            and not needs_frozen_ar_case_probe
        )
        if needs_start_topology_summary:
            start_topology_features = _build_start_topology_feature_tensor(
                self,
                [pair["start_tree"]],
                device=self.device,
            )
            if (
                getattr(self.model, "first_hit_head_mode", "base")
                in {
                    "start_topology_adapter_mlp",
                    "start_topology_raw_pool_concat_mlp",
                }
            ):
                sample_kwargs["first_hit_start_topology_features"] = (
                    start_topology_features
                )
            if getattr(
                self.model,
                "autoregressive_use_start_topology_conditioning",
                False,
            ):
                sample_kwargs["autoregressive_start_topology_features"] = (
                    start_topology_features
                )
        if (
            getattr(self.model, "first_hit_head_mode", "base")
            == "start_topology_cross_attn_mlp"
        ):
            embeddings, pad_mask = _build_start_topology_identity_batch(
                self,
                [pair["start_tree"]],
                device=self.device,
            )
            sample_kwargs["first_hit_start_topology_embeddings"] = embeddings
            sample_kwargs["first_hit_start_topology_pad_mask"] = pad_mask
        if (
            getattr(self.model, "first_hit_head_mode", "base")
            == "start_tree_graph_token_mlp"
        ):
            sample_kwargs["first_hit_start_tree_graph_context"] = (
                _build_start_tree_graph_context(
                    self,
                    [pair["start_tree"]],
                    phyla_embeddings,
                    device=self.device,
                    detach=getattr(
                        self.model, "first_hit_start_tree_graph_detach", False
                    ),
                )
            )
        sample_kwargs.update(overrides)
        return sample_kwargs

    def _build_harness_lexicographic_ordering_map(self, reference_tree_newick):
        tree = EteTree(reference_tree_newick, format=1)
        leaves = list(tree.get_leaves())
        leaves.sort(key=lambda leaf: str(leaf.name))
        return {str(leaf.name): str(i) for i, leaf in enumerate(leaves)}

    def _build_numeric_to_harness_lexicographic_ordering_map(
        self,
        reference_tree_newick,
    ):
        tree = EteTree(reference_tree_newick, format=1)
        original_names = [str(leaf.name) for leaf in tree.get_leaves()]
        numeric_sorted = sorted(original_names, key=lambda name: int(str(name)))
        lex_sorted = sorted(original_names, key=lambda name: str(name))
        original_to_numeric = {str(name): str(i) for i, name in enumerate(numeric_sorted)}
        original_to_lex = {str(name): str(i) for i, name in enumerate(lex_sorted)}
        return {
            original_to_numeric[str(name)]: original_to_lex[str(name)]
            for name in original_names
        }

    def _remap_tree_with_ordering_map(self, tree_newick, ordering_map):
        tree = EteTree(tree_newick, format=1)
        for leaf in tree.get_leaves():
            original_name = str(leaf.name)
            mapped_name = ordering_map.get(original_name)
            if mapped_name is None:
                raise KeyError(
                    f"Leaf {original_name!r} is missing from the harness ordering map."
                )
            leaf.name = str(mapped_name)
        return tree.write(format=1)

    def _infer_golden_posterior_root(self, short_root):
        if not short_root:
            return None
        short_root = str(short_root)
        if "short_run_data_DS1-8" in short_root:
            return short_root.replace("short_run_data_DS1-8", "golden_run_data_DS1-8")
        base_name = os.path.basename(short_root.rstrip("/"))
        if base_name.startswith("short_run_data_"):
            return os.path.join(
                os.path.dirname(short_root.rstrip("/")),
                base_name.replace("short_run_data_", "golden_run_data_", 1),
            )
        return None

    def _canonical_tree_topology_counts(self, trees):
        counts = Counter()
        for tree in trees:
            try:
                counts[canonicalize_topology_newick(str(tree))] += 1
            except Exception:
                continue
        return counts

    def _split_topology_counts(self, trees):
        counts = Counter()
        encoder = BHVEncoder()
        for tree in trees:
            try:
                masks, _lengths = encoder.return_BHV_encoding(Tree(str(tree)))
                counts.update(int(mask) for mask in masks)
            except Exception:
                continue
        return counts

    @staticmethod
    def _json_counter(counter):
        return {str(key): int(value) for key, value in Counter(counter).items()}

    @staticmethod
    def _counter_from_json(value, *, integer_keys=False):
        counts = Counter()
        if not isinstance(value, dict):
            return counts
        for key, count in value.items():
            try:
                parsed_key = int(key) if integer_keys else str(key)
                counts[parsed_key] = int(count)
            except Exception:
                continue
        return counts

    def _posterior_reference_cache_dir(self):
        if self.sample_metrics_trace_path:
            base_dir = os.path.dirname(os.path.abspath(self.sample_metrics_trace_path))
        elif self.sample_metrics_tree_dump_dir:
            base_dir = os.path.dirname(os.path.abspath(self.sample_metrics_tree_dump_dir))
        elif self.sample_metrics_mrbayes20k_output_dir:
            base_dir = os.path.dirname(
                os.path.abspath(self.sample_metrics_mrbayes20k_output_dir)
            )
        else:
            base_dir = os.path.abspath("metrics")
        return os.path.join(base_dir, "posterior_reference_cache")

    def _posterior_reference_cache_path(
        self,
        posterior_root,
        golden_root,
        dataset_id,
        trprobs_sample_count,
    ):
        payload = json.dumps(
            {
                "version": 1,
                "posterior_root": str(posterior_root),
                "golden_root": str(golden_root),
                "dataset_id": str(dataset_id),
                "trprobs_sample_count": int(trprobs_sample_count),
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        safe_dataset_id = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_"
            for ch in str(dataset_id)
        )
        return os.path.join(
            self._posterior_reference_cache_dir(),
            f"{safe_dataset_id}_posterior_reference_v1_{digest}.json",
        )

    def _posterior_reference_bundle_from_payload(self, payload):
        if not isinstance(payload, dict) or int(payload.get("version", 0)) != 1:
            return None
        short_block = dict(payload.get("short") or {})
        golden_block = dict(payload.get("golden") or {})
        return {
            "dataset_id": str(payload.get("dataset_id")),
            "short_root": payload.get("short_root"),
            "golden_root": payload.get("golden_root"),
            "posterior_raw_to_lex": {
                str(key): str(value)
                for key, value in dict(
                    payload.get("posterior_raw_to_lex") or {}
                ).items()
            },
            "numeric_to_lex": {
                str(key): str(value)
                for key, value in dict(payload.get("numeric_to_lex") or {}).items()
            },
            "num_leaves": int(payload.get("num_leaves", 0)),
            "short_counts": self._counter_from_json(
                short_block.get("topology_counts")
            ),
            "golden_counts": self._counter_from_json(
                golden_block.get("topology_counts")
            ),
            "short_split_counts": self._counter_from_json(
                short_block.get("split_counts"),
                integer_keys=True,
            ),
            "golden_split_counts": self._counter_from_json(
                golden_block.get("split_counts"),
                integer_keys=True,
            ),
            "short_tree_total": int(short_block.get("tree_total", 0)),
            "golden_tree_total": int(golden_block.get("tree_total", 0)),
            "short_split_total": int(short_block.get("split_total", 0)),
            "golden_split_total": int(golden_block.get("split_total", 0)),
            "cache_path": payload.get("cache_path"),
        }

    def _posterior_reference_bundle_to_payload(
        self,
        bundle,
        *,
        posterior_root,
        golden_root,
        dataset_id,
        trprobs_sample_count,
        cache_path,
    ):
        short_split_counts = Counter(bundle.get("short_split_counts") or {})
        golden_split_counts = Counter(bundle.get("golden_split_counts") or {})
        return {
            "version": 1,
            "dataset_id": str(dataset_id),
            "short_root": str(posterior_root),
            "golden_root": str(golden_root) if golden_root else None,
            "trprobs_sample_count": int(trprobs_sample_count),
            "num_leaves": int(bundle.get("num_leaves", 0)),
            "posterior_raw_to_lex": dict(bundle.get("posterior_raw_to_lex") or {}),
            "numeric_to_lex": dict(bundle.get("numeric_to_lex") or {}),
            "short": {
                "tree_total": int(bundle.get("short_tree_total", 0)),
                "topology_counts": self._json_counter(
                    bundle.get("short_counts") or {}
                ),
                "split_total": int(sum(short_split_counts.values())),
                "split_counts": self._json_counter(short_split_counts),
            },
            "golden": {
                "tree_total": int(bundle.get("golden_tree_total", 0)),
                "topology_counts": self._json_counter(
                    bundle.get("golden_counts") or {}
                ),
                "split_total": int(sum(golden_split_counts.values())),
                "split_counts": self._json_counter(golden_split_counts),
            },
            "cache_path": str(cache_path),
        }

    @staticmethod
    def _kl_from_topology_counts(
        posterior_counts,
        sampled_counts,
        *,
        alpha=1e-6,
    ):
        posterior_counts = Counter(posterior_counts or {})
        sampled_counts = Counter(sampled_counts or {})
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
        zp = posterior_total + float(alpha) * len(support)
        zq = sampled_total + float(alpha) * len(support)
        kl = 0.0
        for key in support:
            p = (float(posterior_counts.get(key, 0.0)) + float(alpha)) / zp
            q = (float(sampled_counts.get(key, 0.0)) + float(alpha)) / zq
            kl += p * math.log(p / q)

        shared = len(set(posterior_counts.keys()).intersection(sampled_counts.keys()))
        unique_posterior = len(posterior_counts)
        return {
            "kl_divergence_tree_topology": float(kl),
            "n_unique_posterior_topologies": float(unique_posterior),
            "n_unique_sampled_topologies": float(len(sampled_counts)),
            "n_shared_topologies": float(shared),
            "posterior_topology_support_recall": (
                float(shared) / float(unique_posterior)
                if unique_posterior
                else 1.0
            ),
        }

    @staticmethod
    def _kl_from_split_counts(
        posterior_split_counts,
        sampled_split_counts,
        *,
        alpha=1e-6,
    ):
        posterior_split_counts = Counter(posterior_split_counts or {})
        sampled_split_counts = Counter(sampled_split_counts or {})
        posterior_total = float(sum(posterior_split_counts.values()))
        sampled_total = float(sum(sampled_split_counts.values()))
        if posterior_total <= 0.0 or sampled_total <= 0.0:
            return {"kl_divergence_topological": 0.0}
        posterior_distribution = {
            key: float(value) / posterior_total
            for key, value in posterior_split_counts.items()
        }
        sampled_distribution = {
            key: float(value) / sampled_total
            for key, value in sampled_split_counts.items()
        }
        support = set(posterior_distribution.keys()).union(
            sampled_distribution.keys()
        )
        if not support:
            return {"kl_divergence_topological": 0.0}
        zp = 1.0 + float(alpha) * len(support)
        zq = 1.0 + float(alpha) * len(support)
        kl = 0.0
        for key in support:
            p = (posterior_distribution.get(key, 0.0) + float(alpha)) / zp
            q = (sampled_distribution.get(key, 0.0) + float(alpha)) / zq
            kl += p * math.log(p / q)
        return {"kl_divergence_topological": float(kl)}

    @staticmethod
    def _topk_recall_from_topology_counts(
        posterior_counts,
        sampled_counts,
        top_ks=(1, 5, 10, 20, 50),
    ):
        posterior_counts = Counter(posterior_counts or {})
        sampled_support = set(Counter(sampled_counts or {}).keys())
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
        metrics["posterior_topology_posterior_support_size"] = float(
            len(posterior_counts)
        )
        metrics["posterior_topology_total_mass"] = float(posterior_total)
        return metrics

    def _support_rate_from_topology_counts(self, sampled_counts, support_counts):
        sampled_counts = Counter(sampled_counts or {})
        if not sampled_counts:
            return float("nan")
        if not support_counts:
            return 0.0
        total = float(sum(sampled_counts.values()))
        if total <= 0.0:
            return float("nan")
        support_keys = set(Counter(support_counts).keys())
        hits = sum(
            int(count)
            for key, count in sampled_counts.items()
            if key in support_keys
        )
        return float(hits) / total

    def _sampled_topology_distribution_metrics_from_counts(
        self,
        counts,
        reference_support_size=None,
    ):
        counts = Counter(counts or {})
        total = float(sum(counts.values()))
        if total <= 0:
            return {
                "sampled_topology_unique_count": 0.0,
                "sampled_topology_mode_mass": float("nan"),
                "sampled_topology_entropy": float("nan"),
                "sampled_topology_entropy_normalized": float("nan"),
            }
        probs = [float(count) / total for count in counts.values()]
        entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
        norm_base = int(reference_support_size or 0)
        if norm_base <= 1:
            entropy_normalized = 0.0
        else:
            entropy_normalized = float(entropy) / float(math.log(norm_base))
        return {
            "sampled_topology_unique_count": float(len(counts)),
            "sampled_topology_mode_mass": float(max(probs)),
            "sampled_topology_entropy": float(entropy),
            "sampled_topology_entropy_normalized": float(entropy_normalized),
        }

    def _support_rate(self, sampled_trees, support_counts):
        if not sampled_trees:
            return float("nan")
        if not support_counts:
            return 0.0
        support_keys = set(support_counts.keys())
        hits = 0
        total = 0
        for tree in sampled_trees:
            try:
                key = canonicalize_topology_newick(str(tree))
            except Exception:
                continue
            total += 1
            if key in support_keys:
                hits += 1
        if total <= 0:
            return float("nan")
        return float(hits) / float(total)

    def _sampled_topology_distribution_metrics(self, sampled_trees, reference_support_size=None):
        counts = self._canonical_tree_topology_counts(sampled_trees)
        total = float(sum(counts.values()))
        if total <= 0:
            return {
                "sampled_topology_unique_count": 0.0,
                "sampled_topology_mode_mass": float("nan"),
                "sampled_topology_entropy": float("nan"),
                "sampled_topology_entropy_normalized": float("nan"),
            }
        probs = [float(count) / total for count in counts.values()]
        entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
        norm_base = int(reference_support_size or 0)
        if norm_base <= 1:
            entropy_normalized = 0.0
        else:
            entropy_normalized = float(entropy) / float(math.log(norm_base))
        return {
            "sampled_topology_unique_count": float(len(counts)),
            "sampled_topology_mode_mass": float(max(probs)),
            "sampled_topology_entropy": float(entropy),
            "sampled_topology_entropy_normalized": float(entropy_normalized),
        }

    def _prefix_metric_block(self, metrics, prefix):
        return {f"{prefix}_{key}": value for key, value in metrics.items()}

    def _load_posterior_reference_bundle(self, train=True):
        dataset_split = self.dataset.dataset_train if train else self.dataset.dataset_val
        posterior_root = getattr(dataset_split, "posterior_trprobs_root", None)
        posterior_ids = list(getattr(dataset_split, "posterior_dataset_ids", []) or [])
        if not posterior_root or len(posterior_ids) != 1:
            return None
        dataset_id = str(posterior_ids[0])
        trprobs_sample_count = int(
            getattr(dataset_split, "trprobs_sample_count_per_file", 1000)
        )
        golden_root = self._infer_golden_posterior_root(posterior_root)
        cache_key = (
            str(posterior_root),
            str(golden_root),
            dataset_id,
            int(trprobs_sample_count),
        )
        cache = getattr(self, "_posterior_reference_bundle_cache", None)
        if cache is None:
            cache = {}
            self._posterior_reference_bundle_cache = cache
        if cache_key in cache:
            return cache[cache_key]

        cache_path = self._posterior_reference_cache_path(
            posterior_root,
            golden_root,
            dataset_id,
            trprobs_sample_count,
        )
        if os.path.exists(cache_path) and os.environ.get(
            "PHYLAFLOW_POSTERIOR_REFERENCE_CACHE_DISABLE", "0"
        ) != "1":
            try:
                with open(cache_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                bundle = self._posterior_reference_bundle_from_payload(payload)
                if bundle is not None:
                    bundle["cache_path"] = cache_path
                    cache[cache_key] = bundle
                    return bundle
            except Exception as exc:
                logger.warning(
                    "Failed to load posterior reference cache %s: %s",
                    cache_path,
                    exc,
                )

        short_dataset = TreeDataset(
            nexus_root="unused",
            mrbayes_root="unused",
            posterior_trprobs_root=str(posterior_root),
            posterior_dataset_id=str(dataset_id),
            trprobs_sample_count_per_file=int(trprobs_sample_count),
        )
        short_raw = list(short_dataset.return_posterior_trees(str(dataset_id)))
        golden_raw = []
        if golden_root and os.path.isdir(str(golden_root)):
            golden_dataset = TreeDataset(
                nexus_root="unused",
                mrbayes_root="unused",
                posterior_trprobs_root=str(golden_root),
                posterior_dataset_id=str(dataset_id),
                trprobs_sample_count_per_file=int(trprobs_sample_count),
            )
            golden_raw = list(golden_dataset.return_posterior_trees(str(dataset_id)))

        reference_tree = short_raw[0] if short_raw else (golden_raw[0] if golden_raw else None)
        if reference_tree is None:
            cache[cache_key] = None
            return None

        posterior_raw_to_lex = self._build_harness_lexicographic_ordering_map(reference_tree)
        numeric_to_lex = self._build_numeric_to_harness_lexicographic_ordering_map(reference_tree)

        short_lex = [
            self._remap_tree_with_ordering_map(tree, posterior_raw_to_lex)
            for tree in short_raw
        ]
        golden_lex = [
            self._remap_tree_with_ordering_map(tree, posterior_raw_to_lex)
            for tree in golden_raw
        ]
        bundle = {
            "dataset_id": str(dataset_id),
            "short_root": str(posterior_root),
            "golden_root": str(golden_root) if golden_root else None,
            "short_counts": self._canonical_tree_topology_counts(short_lex),
            "golden_counts": self._canonical_tree_topology_counts(golden_lex),
            "short_split_counts": self._split_topology_counts(short_lex),
            "golden_split_counts": self._split_topology_counts(golden_lex),
            "short_tree_total": int(len(short_lex)),
            "golden_tree_total": int(len(golden_lex)),
            "posterior_raw_to_lex": posterior_raw_to_lex,
            "numeric_to_lex": numeric_to_lex,
            "num_leaves": len(EteTree(short_lex[0] if short_lex else golden_lex[0], format=1).get_leaves()),
            "cache_path": cache_path,
        }
        bundle["short_split_total"] = int(sum(bundle["short_split_counts"].values()))
        bundle["golden_split_total"] = int(sum(bundle["golden_split_counts"].values()))
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            payload = self._posterior_reference_bundle_to_payload(
                bundle,
                posterior_root=posterior_root,
                golden_root=golden_root,
                dataset_id=dataset_id,
                trprobs_sample_count=trprobs_sample_count,
                cache_path=cache_path,
            )
            tmp_path = f"{cache_path}.tmp.{os.getpid()}"
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_path, cache_path)
        except Exception as exc:
            logger.warning(
                "Failed to write posterior reference cache %s: %s",
                cache_path,
                exc,
            )
        cache[cache_key] = bundle
        return bundle

    def _remap_sampled_tree_to_reference_lex(self, tree_newick, bundle):
        tree_newick = str(tree_newick)
        leaf_names = [str(leaf.name) for leaf in EteTree(tree_newick, format=1).iter_leaves()]
        posterior_raw_to_lex = bundle["posterior_raw_to_lex"]
        numeric_to_lex = bundle["numeric_to_lex"]
        if all(name in posterior_raw_to_lex for name in leaf_names):
            return self._remap_tree_with_ordering_map(tree_newick, posterior_raw_to_lex)
        if all(name in numeric_to_lex for name in leaf_names):
            return self._remap_tree_with_ordering_map(tree_newick, numeric_to_lex)
        return tree_newick

    def _posterior_reference_metrics(self, sampled_trees, train=True):
        bundle = self._load_posterior_reference_bundle(train=train)
        if bundle is None or not sampled_trees:
            return {}

        remapped_sampled = [
            self._remap_sampled_tree_to_reference_lex(tree, bundle)
            for tree in sampled_trees
        ]

        metrics = {}
        sampled_counts = self._canonical_tree_topology_counts(remapped_sampled)
        sampled_split_counts = self._split_topology_counts(remapped_sampled)
        short_counts = Counter(bundle.get("short_counts") or {})
        golden_counts = Counter(bundle.get("golden_counts") or {})
        short_split_counts = Counter(bundle.get("short_split_counts") or {})
        golden_split_counts = Counter(bundle.get("golden_split_counts") or {})
        support_size = len(short_counts) or len(golden_counts)
        metrics.update(
            self._sampled_topology_distribution_metrics_from_counts(
                sampled_counts,
                reference_support_size=support_size,
            )
        )
        metrics["short_support_rate"] = self._support_rate_from_topology_counts(
            sampled_counts,
            short_counts,
        )
        if golden_counts:
            metrics["golden_support_rate"] = self._support_rate_from_topology_counts(
                sampled_counts,
                golden_counts,
            )

        if short_counts:
            short_block = {}
            short_block.update(self._kl_from_split_counts(short_split_counts, sampled_split_counts))
            short_block.update(self._kl_from_topology_counts(short_counts, sampled_counts))
            short_block.update(
                self._topk_recall_from_topology_counts(short_counts, sampled_counts)
            )
            metrics.update(self._prefix_metric_block(short_block, "short"))
        if golden_counts:
            golden_block = {}
            golden_block.update(
                self._kl_from_split_counts(golden_split_counts, sampled_split_counts)
            )
            golden_block.update(self._kl_from_topology_counts(golden_counts, sampled_counts))
            golden_block.update(
                self._topk_recall_from_topology_counts(golden_counts, sampled_counts)
            )
            metrics.update(self._prefix_metric_block(golden_block, "golden"))
        return metrics

    def _get_branch_relax_likelihood_scorer(self):
        if not self.branch_relax_likelihood_metric_enabled:
            return None
        dataset_id = self.branch_relax_likelihood_dataset_id
        if not dataset_id:
            bundle = self._load_posterior_reference_bundle(train=True)
            dataset_id = None if bundle is None else bundle.get("dataset_id")
        if not dataset_id:
            return None
        if self._branch_relax_likelihood_scorer is None:
            from scripts.jc_likelihood import GenericJCLikelihood

            self._branch_relax_likelihood_scorer = GenericJCLikelihood(
                dataset_id=str(dataset_id)
            )
        return self._branch_relax_likelihood_scorer

    def _sample_metrics_dataset_id(self, train=True):
        dataset_id = self.branch_relax_likelihood_dataset_id
        if dataset_id:
            return str(dataset_id)
        bundle = self._load_posterior_reference_bundle(train=train)
        if bundle is not None and bundle.get("dataset_id"):
            return str(bundle["dataset_id"])
        dataset_split = self.dataset.dataset_train if train else self.dataset.dataset_val
        posterior_ids = list(getattr(dataset_split, "posterior_dataset_ids", []) or [])
        if len(posterior_ids) == 1:
            return str(posterior_ids[0])
        return None

    def _get_sample_metrics_likelihood_scorer(self, train=True):
        dataset_id = self._sample_metrics_dataset_id(train=train)
        if not dataset_id:
            return None
        cache = getattr(self, "_sample_metrics_likelihood_scorer_cache", None)
        if cache is None:
            cache = {}
            self._sample_metrics_likelihood_scorer_cache = cache
        key = str(dataset_id).upper()
        scorer = cache.get(key)
        if scorer is None:
            from scripts.jc_likelihood import GenericJCLikelihood

            scorer = GenericJCLikelihood(dataset_id=key)
            cache[key] = scorer
        return scorer

    def _sample_metrics_standalone_relaxer_args(self, checkpoint):
        from types import SimpleNamespace

        raw_args = dict(checkpoint.get("args") or {})
        return SimpleNamespace(
            base_config=raw_args.get(
                "base_config",
                os.path.join(
                    os.environ.get("PHYLAFLOW_RELEASE_ROOT", os.getcwd()),
                    "configs",
                    "final_release.yaml",
                ),
            ),
            embed_dim=int(raw_args.get("embed_dim", 64)),
            n_layers=int(raw_args.get("n_layers", 2)),
            n_heads=int(raw_args.get("n_heads", 4)),
            dropout=float(raw_args.get("dropout", 0.0)),
            head_hidden_dim=int(raw_args.get("head_hidden_dim", 128)),
            case_dim=int(raw_args.get("case_dim", 0)),
            phyla_dim=int(raw_args.get("phyla_dim", 256)),
            phyla_use_leaf_tokens=bool(raw_args.get("phyla_use_leaf_tokens", True)),
            phyla_use_split_tokens=bool(raw_args.get("phyla_use_split_tokens", False)),
            phyla_embedding_dir=raw_args.get(
                "phyla_embedding_dir",
                os.environ.get(
                    "PHYLAFLOW_PHYLA_EMBEDDING_DIR",
                    os.path.join(
                        os.environ.get("PHYLAFLOW_DATA_ROOT", "./artifacts"),
                        "phyla_embeddings",
                    ),
                ),
            ),
        )

    def _get_sample_metrics_standalone_relaxer(self, train=True):
        path = self.sample_metrics_branch_relaxer_checkpoint_path
        if not path:
            return None
        dataset_id = self._sample_metrics_dataset_id(train=train)
        if not dataset_id:
            return None
        device = self.device
        cache = getattr(self, "_sample_metrics_standalone_relaxer_cache", None)
        if cache is None:
            cache = {}
            self._sample_metrics_standalone_relaxer_cache = cache
        cache_key = (str(path), str(device), str(dataset_id).upper())
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        from scripts.train_branch_relaxer import (
            BranchDeltaHead,
            StandaloneRelaxer,
            _load_phyla_embedding_bank,
            _small_model_config,
        )
        from model.model import return_model

        checkpoint = torch.load(str(path), map_location=device)
        ckpt_args = self._sample_metrics_standalone_relaxer_args(checkpoint)
        cfg = _small_model_config(ckpt_args.base_config, ckpt_args)
        model = return_model(cfg).to(device)
        head_state = checkpoint.get("head") or {}
        case_weight = head_state.get("case_embedding.weight")
        num_cases = (
            int(case_weight.shape[0])
            if torch.is_tensor(case_weight)
            else max(1, int(self.sample_metrics_mrbayes20k_num_starts))
        )
        head = BranchDeltaHead(
            int(model.embed_dim),
            hidden_dim=int(ckpt_args.head_hidden_dim),
            case_dim=int(ckpt_args.case_dim),
            num_cases=int(num_cases),
        ).to(device)
        model.load_state_dict(checkpoint["model"])
        head.load_state_dict(head_state)
        relaxer = StandaloneRelaxer(model, head).to(device)
        relaxer.eval()
        phyla_bank = _load_phyla_embedding_bank(
            [str(dataset_id).upper()],
            ckpt_args.phyla_embedding_dir,
            device,
        )
        cached = {
            "relaxer": relaxer,
            "phyla_bank": phyla_bank,
            "dataset_id": str(dataset_id).upper(),
        }
        cache[cache_key] = cached
        return cached

    def _sample_metrics_source_mask_for_node(self, node, n_leaves):
        root_leaf = int(n_leaves) - 1
        biological_bits = max(int(n_leaves) - 1, 0)
        full_mask = (1 << biological_bits) - 1 if biological_bits > 0 else 0
        raw_indices = []
        for leaf in node.iter_leaves():
            value = int(str(leaf.name))
            raw_indices.append(value - 1 if 1 <= value <= int(n_leaves) else value)
        indices = set(raw_indices)
        if root_leaf in indices:
            indices = set(range(int(n_leaves))) - indices
        mask = 0
        for index in indices:
            if 0 <= int(index) < biological_bits:
                mask |= 1 << int(index)
        return int(mask if mask else full_mask)

    def _sample_metrics_apply_standalone_relaxer(self, tree_newick, case_index, train=True):
        bundle = self._get_sample_metrics_standalone_relaxer(train=train)
        if bundle is None:
            return str(tree_newick), {"applied": False}
        relaxer = bundle["relaxer"]
        device = self.device
        dataset_id = str(bundle["dataset_id"]).upper()
        newick = str(tree_newick).strip()
        if not newick.endswith(";"):
            newick += ";"
        tokenized = _move_tokenized_batch_to_device(relaxer.model.tokenizer([newick]), device)
        phyla_embeddings = None
        phyla_bank = bundle.get("phyla_bank") or {}
        if phyla_bank:
            phyla_embeddings = phyla_bank[dataset_id]
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
            entries, _lengths, n_leaves, _mapping = _branch_relax_entries_for_tree(
                relaxer,
                newick,
                edge_split_masks[0],
                labels=None,
            )
            if not entries:
                return newick, {"applied": False}
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
                int(case_index),
                dtype=torch.long,
                device=device,
            )
            deltas = relaxer.head(features, numeric, case_indices).detach().cpu().numpy()

        delta_by_source_mask = {
            int(entry.get("source_mask", entry["mask"])): float(delta)
            for entry, delta in zip(entries, deltas)
        }
        tree = EteTree(newick, format=1, quoted_node_names=True)
        applied = 0
        max_abs_delta = 0.0
        for node in tree.traverse("postorder"):
            if node.is_root():
                continue
            try:
                source_mask = self._sample_metrics_source_mask_for_node(
                    node,
                    int(n_leaves),
                )
            except Exception:
                continue
            delta = delta_by_source_mask.get(int(source_mask))
            if delta is None:
                continue
            before = float(node.dist)
            after = max(before + float(delta), 1e-8)
            node.dist = after
            applied += 1
            max_abs_delta = max(max_abs_delta, abs(after - before))
        return tree.write(format=1), {
            "applied": bool(applied),
            "applied_edge_count": int(applied),
            "max_abs_delta": float(max_abs_delta),
        }

    def _sample_metrics_infer_mrbayes_paths(self, train=True):
        dataset_id = self._sample_metrics_dataset_id(train=train)
        if not dataset_id:
            return None, None, None
        dataset_id = str(dataset_id).upper()
        dataset_pickle = self.sample_metrics_mrbayes20k_dataset_pickle_path
        golden_root = self.sample_metrics_mrbayes20k_golden_root
        bundle = self._load_posterior_reference_bundle(train=train)
        if golden_root is None and bundle is not None and bundle.get("golden_root"):
            golden_root = str(bundle["golden_root"])
        if dataset_pickle is None and golden_root:
            root = os.path.dirname(os.path.dirname(str(golden_root).rstrip("/")))
            candidate = os.path.join(root, f"{dataset_id}.pickle")
            if os.path.exists(candidate):
                dataset_pickle = candidate
        return dataset_id, dataset_pickle, golden_root

    def _sample_metrics_mrbayes20k_output_dir(self):
        if self.sample_metrics_mrbayes20k_output_dir:
            return self.sample_metrics_mrbayes20k_output_dir
        if self.sample_metrics_trace_path:
            return os.path.join(
                os.path.dirname(self.sample_metrics_trace_path),
                "mrbayes20k_sample_metrics",
            )
        return "/tmp/phylaflow_sample_metrics_mrbayes20k_outputs"

    def _sample_metrics_run_mrbayes20k(self, relaxed_trees, train=True):
        if not self.sample_metrics_mrbayes20k_enabled:
            return {}
        dataset_id, dataset_pickle, golden_root = self._sample_metrics_infer_mrbayes_paths(
            train=train
        )
        if not dataset_id or not dataset_pickle or not golden_root:
            return {"mrbayes20k_failed": 1.0}
        selected_trees = list(relaxed_trees)[
            : min(len(relaxed_trees), int(self.sample_metrics_mrbayes20k_num_starts))
        ]
        if not selected_trees:
            return {"mrbayes20k_failed": 1.0}

        step = int(self.global_step)
        stamp = f"pid{os.getpid()}_step{step:08d}_{int(time.time())}"
        work_dir = os.path.join(self.sample_metrics_mrbayes20k_work_root, stamp)
        output_dir = self._sample_metrics_mrbayes20k_output_dir()
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        start_list_path = os.path.join(output_dir, f"{stamp}_relaxed_starts.txt")
        output_json = os.path.join(output_dir, f"{stamp}_mrbayes20k_curve.json")
        log_path = os.path.join(output_dir, f"{stamp}_mrbayes20k.log")
        with open(start_list_path, "w", encoding="utf-8") as handle:
            for tree in selected_trees:
                tree = str(tree).strip()
                handle.write(tree if tree.endswith(";") else tree + ";")
                handle.write("\n")

        benchmark = getattr(
            self,
            "sample_metrics_mrbayes20k_benchmark_script",
            os.environ.get(
                "PHYLAFLOW_MRBAYES_BENCHMARK",
                os.path.join(
                    os.environ.get("PHYLAFLOW_RELEASE_ROOT", os.getcwd()),
                    "scripts",
                    "benchmark_mrbayes_fixed_start.py",
                ),
            ),
        )
        cmd = [
            sys.executable,
            benchmark,
            "--dataset-id",
            str(dataset_id),
            "--dataset-pickle",
            str(dataset_pickle),
            "--golden-root",
            str(golden_root),
            "--label",
            f"sample_metrics_mrbayes20k_step{step}",
            "--num-runs",
            str(len(selected_trees)),
            "--ngen",
            str(int(self.sample_metrics_mrbayes20k_ngen)),
            "--samplefreq",
            str(int(self.sample_metrics_mrbayes20k_samplefreq)),
            "--printfreq",
            str(int(self.sample_metrics_mrbayes20k_printfreq)),
            "--max-workers",
            str(int(self.sample_metrics_mrbayes20k_max_workers)),
            "--curve-interval",
            str(int(self.sample_metrics_mrbayes20k_ngen)),
            "--mrbayes-bin",
            str(self.sample_metrics_mrbayes20k_bin),
            "--work-dir",
            str(work_dir),
            "--output",
            str(output_json),
            "--start-tree-list",
            str(start_list_path),
            "--threshold-check-selected-only",
        ]
        try:
            with open(log_path, "w", encoding="utf-8") as log_file:
                result = subprocess.run(
                    cmd,
                    cwd=os.environ.get("PHYLAFLOW_RELEASE_ROOT", os.getcwd()),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    timeout=int(self.sample_metrics_mrbayes20k_timeout_sec),
                )
        except Exception as exc:
            return {
                "mrbayes20k_failed": 1.0,
                "mrbayes20k_error": str(exc),
            }
        if result.returncode != 0 or not os.path.exists(output_json):
            return {
                "mrbayes20k_failed": 1.0,
                "mrbayes20k_returncode": float(result.returncode),
            }
        with open(output_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        final_block = payload.get("final_cumulative_by_generation") or {}
        tail_block = payload.get("tail_half_samples") or {}
        failures = payload.get("failures") or []
        return {
            "mrbayes20k_failed": 0.0,
            "mrbayes20k_num_starts": float(len(selected_trees)),
            "mrbayes20k_completed_runs": float(payload.get("completed_runs", 0)),
            "mrbayes20k_failure_count": float(len(failures)),
            "mrbayes20k_tree_kl": float(
                final_block.get("kl_divergence_tree_topology", float("nan"))
            ),
            "mrbayes20k_tail_tree_kl": float(
                tail_block.get("kl_divergence_tree_topology", float("nan"))
            ),
        }

    def _sample_metrics_relaxed_downstream_metrics(
        self,
        sampled_trees,
        train=True,
        return_tree_rows=False,
        run_mrbayes=True,
    ):
        if not (
            self.sample_metrics_relaxed_likelihood_enabled
            or self.sample_metrics_mrbayes20k_enabled
        ):
            return ({}, []) if return_tree_rows else {}
        if not sampled_trees:
            return ({}, []) if return_tree_rows else {}
        relaxed_trees = []
        relaxed_tree_rows = []
        likelihoods = []
        applied = 0
        scorer = (
            self._get_sample_metrics_likelihood_scorer(train=train)
            if self.sample_metrics_relaxed_likelihood_enabled
            else None
        )
        for index, tree in enumerate(sampled_trees):
            relaxed_tree, info = self._sample_metrics_apply_standalone_relaxer(
                tree,
                case_index=index,
                train=train,
            )
            relaxed_trees.append(relaxed_tree)
            applied += int(bool(info.get("applied")))
            relaxed_row = {
                "sample_index": int(index),
                "relaxed_tree": str(relaxed_tree),
                "relax_applied": bool(info.get("applied")),
                "relax_applied_edge_count": int(info.get("applied_edge_count", 0)),
                "relax_max_abs_delta": float(info.get("max_abs_delta", 0.0)),
            }
            if scorer is not None:
                likelihood = float(scorer.log_likelihood(str(relaxed_tree)))
                likelihoods.append(likelihood)
                relaxed_row["relaxed_log_likelihood"] = likelihood
            relaxed_tree_rows.append(relaxed_row)

        metrics = {
            "relaxed_start_count": float(len(relaxed_trees)),
            "relaxed_applied_count": float(applied),
        }
        if likelihoods:
            metrics["relaxed_log_likelihood_mean"] = float(
                np.asarray(likelihoods, dtype=np.float64).mean()
            )
        if run_mrbayes:
            metrics.update(
                self._sample_metrics_run_mrbayes20k(relaxed_trees, train=train)
            )
        if return_tree_rows:
            return metrics, relaxed_tree_rows
        return metrics

    def _sample_compare_harness_can_batch_discrete_phase(self, pairs, sample_kwargs):
        if len(pairs) <= 1:
            return False
        if bool(getattr(self, "sample_metrics_trace_topology_repeats_enabled", False)):
            return False
        if bool(getattr(self, "branch_relax_likelihood_metric_enabled", False)):
            return False
        unsupported_keys = {
            "first_hit_start_topology_embeddings",
            "first_hit_start_topology_pad_mask",
            "first_hit_start_tree_graph_context",
        }
        for kwargs in sample_kwargs:
            for key in unsupported_keys:
                if kwargs.get(key) is not None:
                    return False
        return True

    def _sample_compare_harness_batch_tensor(self, sample_kwargs, indices, key):
        values = [sample_kwargs[index].get(key) for index in indices]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError(f"Cannot batch mixed missing/non-missing {key}.")
        tensors = []
        for value in values:
            tensor = torch.as_tensor(value, device=self.device)
            if tensor.ndim == 0:
                tensor = tensor.reshape(1)
            tensors.append(tensor)
        if tensors[0].ndim >= 1 and int(tensors[0].shape[0]) == 1:
            return torch.cat(tensors, dim=0)
        return torch.stack(tensors, dim=0)

    def _sample_compare_harness_batch_case_indices(self, sample_kwargs, indices):
        values = [sample_kwargs[index].get("case_indices") for index in indices]
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise ValueError("Cannot batch mixed missing/non-missing case_indices.")
        case_indices = []
        for value in values:
            case_indices.append(int(torch.as_tensor(value).reshape(-1)[0].item()))
        return torch.tensor(case_indices, dtype=torch.long, device=self.device)

    def _sample_compare_harness_tokenize_with_edge_lengths(self, newicks):
        return self.model.tokenizer(newicks), None

    def _sample_compare_harness_batch_discrete_phase(self, pairs, train=True):
        sample_kwargs = [
            self._build_harness_sample_kwargs(dict(pair), train=train)
            for pair in pairs
        ]
        if not self._sample_compare_harness_can_batch_discrete_phase(
            pairs,
            sample_kwargs,
        ):
            rows = [
                self._sample_compare_harness_once(pair, train=train)
                for pair in pairs
            ]
            return rows, {"batched_discrete_phase_used": 0.0}

        states = []
        for index, pair in enumerate(pairs):
            kwargs = sample_kwargs[index]
            max_events = kwargs.get("max_events")
            max_steps = kwargs.get("max_steps")
            start_tree_obj = Tree(str(pair["start_tree"]))
            states.append(
                {
                    "index": int(index),
                    "current_newick": str(pair["start_tree"]),
                    "target_tree": str(pair["target_tree"]),
                    "n_leaves": int(start_tree_obj.n_leaves),
                    "mapping": dict(start_tree_obj.id_to_name),
                    "phase": 0,
                    "n_events": 0,
                    "n_steps": 0,
                    "effective_max_events": (
                        1000000
                        if max_events is None or int(max_events) < 0
                        else int(max_events)
                    ),
                    "effective_max_steps": (
                        1000000
                        if max_steps is None or int(max_steps) < 0
                        else int(max_steps)
                    ),
                    "done": False,
                    "stopped_for_no_valid_merge": False,
                    "stopped_for_repeated_topology": False,
                    "skipped_no_valid_boundary_revisits": 0.0,
                    "dt_base": float(kwargs.get("dt_base", self.training_sampling_dt_base)),
                }
            )

        eps_len = 1e-8
        autoregressive_birth_length = 1e-3
        max_phases = int(getattr(self, "sampling_discrete_phase_max_phases", 8))
        velocity_forward_batches = 0
        velocity_forward_items = 0
        ar_forward_batches = 0
        ar_forward_items = 0
        tokenizer_edge_length_reuse_items = 0
        tokenizer_edge_length_reuse_fallback_items = 0
        start_time = time.perf_counter()

        for phase in range(max_phases):
            phase_indices = []
            for index, state in enumerate(states):
                if state["done"] or int(state["phase"]) != int(phase):
                    continue
                if (
                    int(state["n_events"]) >= int(state["effective_max_events"])
                    or int(state["n_steps"]) >= int(state["effective_max_steps"])
                ):
                    state["done"] = True
                    continue
                phase_indices.append(index)
            if not phase_indices:
                continue

            current_newicks = [states[index]["current_newick"] for index in phase_indices]
            tokenized, edge_branch_lengths = (
                self._sample_compare_harness_tokenize_with_edge_lengths(current_newicks)
            )
            if edge_branch_lengths is None:
                tokenizer_edge_length_reuse_fallback_items += len(phase_indices)
            else:
                tokenizer_edge_length_reuse_items += len(phase_indices)
            phyla_embeddings = self._sample_compare_harness_batch_tensor(
                sample_kwargs,
                phase_indices,
                "phyla_embeddings",
            )
            first_hit_case_indices = self._sample_compare_harness_batch_case_indices(
                sample_kwargs,
                phase_indices,
            )
            first_hit_start_topology_features = (
                self._sample_compare_harness_batch_tensor(
                    sample_kwargs,
                    phase_indices,
                    "first_hit_start_topology_features",
                )
            )
            with torch.inference_mode():
                (
                    velocity,
                    edge_splits,
                    _edge_split_mask,
                    first_hit_logits,
                    boundary_vanish_logits,
                    edge_features,
                ) = self.forward(
                    tokenized,
                    float(phase),
                    phyla_embeddings,
                    first_hit_case_indices=first_hit_case_indices,
                    first_hit_start_topology_features=first_hit_start_topology_features,
                )
            velocity_forward_batches += 1
            velocity_forward_items += len(phase_indices)

            for local_index, state_index in enumerate(phase_indices):
                state = states[state_index]
                state["n_steps"] += 1
                if edge_branch_lengths is None:
                    td, n_leaves, mapping = _tree_to_model_split_lengths_from_model_masks(
                        state["current_newick"],
                        edge_splits[local_index],
                    )
                else:
                    td, n_leaves, mapping = (
                        _tree_to_model_split_lengths_from_tokenizer_edges(
                            edge_splits[local_index],
                            edge_branch_lengths[local_index],
                            state["n_leaves"],
                            state["mapping"],
                            eps_len=eps_len,
                        )
                    )
                aligned = _align_model_outputs_to_tree_context(
                    self,
                    state["current_newick"],
                    n_leaves,
                    edge_splits[local_index],
                    velocity[local_index, :, 0],
                    first_hit_logits_tree=None
                    if first_hit_logits is None
                    else first_hit_logits[local_index, :, 0],
                    boundary_vanish_logits_tree=None
                    if boundary_vanish_logits is None
                    else boundary_vanish_logits[local_index, :, 0],
                    edge_features_tree=None
                    if edge_features is None
                    else edge_features[local_index],
                    eps_len=eps_len,
                    tree_split_lengths=td,
                )
                aligned_first_hit_logits = self._compute_first_hit_logits(
                    aligned["first_hit_logits"],
                    lengths=aligned["lengths"],
                    velocities=aligned["velocities"],
                    edge_features=aligned["edge_features"],
                    group_sizes=[int(aligned["lengths"].numel())],
                )
                lengths = aligned["lengths"].detach().cpu().numpy().astype(np.float64)
                velocities = (
                    aligned["velocities"].detach().cpu().numpy().astype(np.float64)
                )
                supervised_mask = (
                    aligned["supervised_mask"].detach().cpu().numpy().astype(bool)
                )
                masks = [int(mask) for mask in aligned["aligned_model_masks"]]
                first_logits = (
                    aligned_first_hit_logits.detach().cpu().numpy().astype(np.float64)
                )
                candidate_mask = supervised_mask & (lengths > eps_len)
                predicted_first_mask, _raw_first_count, _used_first_fallback = (
                    _predict_first_hit_mask_with_fallback(
                        first_logits,
                        candidate_mask,
                    )
                )
                pred_neg = (
                    predicted_first_mask
                    & (velocities < 0.0)
                    & (lengths > eps_len)
                )
                if not np.any(pred_neg):
                    state["done"] = True
                    continue
                dt_target = float(
                    np.max(lengths[pred_neg] / np.maximum(-velocities[pred_neg], eps_len))
                )
                dt = float(dt_target)
                L_new = lengths + dt * velocities
                collapse_mask = predicted_first_mask.copy()
                if np.any(collapse_mask):
                    L_new[collapse_mask] = 0.0
                blocked = supervised_mask & (~collapse_mask)
                if np.any(blocked):
                    L_new[blocked] = np.maximum(L_new[blocked], eps_len * 10.0)
                td2 = {
                    int(mask): float(length)
                    for mask, length in zip(masks, L_new)
                    if float(length) > eps_len
                }
                state["current_newick"] = build_tree_from_splits(
                    list(td2.keys()),
                    td2,
                    n_leaves,
                    root_leaf=n_leaves - 1,
                    mapping=mapping,
                )[1]

            while True:
                ar_indices = [
                    index
                    for index in phase_indices
                    if not states[index]["done"]
                    and int(states[index]["phase"]) == int(phase)
                    and int(states[index]["n_events"])
                    < int(states[index]["effective_max_events"])
                    and has_polytomy_fast(
                        states[index]["current_newick"],
                        unrooted_ok=False,
                    )
                ]
                if not ar_indices:
                    break
                ar_newicks = [states[index]["current_newick"] for index in ar_indices]
                tokenized_ar, edge_branch_lengths_ar = (
                    self._sample_compare_harness_tokenize_with_edge_lengths(ar_newicks)
                )
                if edge_branch_lengths_ar is None:
                    tokenizer_edge_length_reuse_fallback_items += len(ar_indices)
                else:
                    tokenizer_edge_length_reuse_items += len(ar_indices)
                use_explicit_component_groups = any(
                    bool(
                        sample_kwargs[index].get(
                            "explicit_autoregressive_component_groups",
                            True,
                        )
                    )
                    for index in ar_indices
                )
                component_groups = (
                    [
                        get_structural_polytomy_groups_from_newick(newick)
                        for newick in ar_newicks
                    ]
                    if use_explicit_component_groups
                    else None
                )
                phyla_embeddings_ar = self._sample_compare_harness_batch_tensor(
                    sample_kwargs,
                    ar_indices,
                    "phyla_embeddings",
                )
                autoregressive_case_indices = (
                    self._sample_compare_harness_batch_case_indices(
                        sample_kwargs,
                        ar_indices,
                    )
                )
                autoregressive_start_topology_features = (
                    self._sample_compare_harness_batch_tensor(
                        sample_kwargs,
                        ar_indices,
                        "autoregressive_start_topology_features",
                    )
                )
                with torch.inference_mode():
                    logit_outputs = self.forward(
                        tokenized_ar,
                        torch.full(
                            (len(ar_indices),),
                            float(phase),
                            dtype=torch.float32,
                            device=self.device,
                        ),
                        phyla_embeddings_ar,
                        autoregressive=True,
                        autoregressive_component_groups=component_groups,
                        autoregressive_case_indices=autoregressive_case_indices,
                        autoregressive_start_topology_features=(
                            autoregressive_start_topology_features
                        ),
                    )
                ar_forward_batches += 1
                ar_forward_items += len(ar_indices)
                outputs_by_batch = {}
                for output in logit_outputs:
                    outputs_by_batch.setdefault(
                        int(output.get("batch_index", 0)),
                        [],
                    ).append(output)

                for local_index, state_index in enumerate(ar_indices):
                    state = states[state_index]
                    if edge_branch_lengths_ar is None:
                        td_ar, n_ar, mapping_ar = (
                            _tree_to_model_split_lengths_from_model_masks(
                                state["current_newick"],
                                tokenized_ar[-1][local_index],
                            )
                        )
                    else:
                        td_ar, n_ar, mapping_ar = (
                            _tree_to_model_split_lengths_from_tokenizer_edges(
                                tokenized_ar[-1][local_index],
                                edge_branch_lengths_ar[local_index],
                                state["n_leaves"],
                                state["mapping"],
                                eps_len=eps_len,
                            )
                        )
                    planned_merges = _plan_autoregressive_boundary_merges(
                        outputs_by_batch.get(local_index, []),
                        td_ar.keys(),
                        top_only=False,
                    )
                    if planned_merges:
                        planned_merges = planned_merges[:1]
                    if not planned_merges:
                        state["stopped_for_no_valid_merge"] = True
                        state["phase"] += 1
                        continue
                    for planned in planned_merges:
                        if (
                            int(state["n_events"])
                            >= int(state["effective_max_events"])
                        ):
                            break
                        if not planned.get("subsets"):
                            continue
                        _subset, new_split = planned["subsets"][0]
                        if int(new_split) in td_ar:
                            continue
                        td_ar[int(new_split)] = float(autoregressive_birth_length)
                        state["n_events"] += 1
                        state["current_newick"] = build_tree_from_splits(
                            list(td_ar.keys()),
                            td_ar,
                            n_ar,
                            root_leaf=n_ar - 1,
                            mapping=mapping_ar,
                        )[1]

            for index in phase_indices:
                state = states[index]
                if not state["done"] and int(state["phase"]) == int(phase):
                    state["phase"] += 1
                if int(state["phase"]) >= int(max_phases):
                    state["done"] = True

        rows = []
        for index, state in enumerate(states):
            pair = pairs[index]
            sampled_tree = str(state["current_newick"])
            metrics = {
                "rf_norm": float(calculate_norm_rf(sampled_tree, pair["target_tree"])),
                "start_rf_norm": float(
                    calculate_norm_rf(pair["start_tree"], pair["target_tree"])
                ),
                "_start_tree": str(pair["start_tree"]),
                "_original_start_tree": (
                    str(pair["original_start_tree"])
                    if pair.get("original_start_tree") is not None
                    else None
                ),
                "_sampled_tree": sampled_tree,
                "_target_tree": str(pair["target_tree"]),
                "_n_leaves": int(pair["n_leaves"]),
                "_bank_group_key": pair.get("bank_group_key"),
                "_source_bank_index": pair.get("source_bank_index"),
                "stopped_for_repeated_topology": float(
                    1.0 if state["stopped_for_repeated_topology"] else 0.0
                ),
                "stopped_for_no_valid_merge": float(
                    1.0 if state["stopped_for_no_valid_merge"] else 0.0
                ),
                "skipped_no_valid_boundary_revisits": float(
                    state["skipped_no_valid_boundary_revisits"]
                ),
            }
            rows.append(metrics)

        stats = {
            "batched_discrete_phase_used": 1.0,
            "batched_discrete_phase_num_pairs": float(len(pairs)),
            "batched_discrete_phase_seconds": float(time.perf_counter() - start_time),
            "batched_discrete_phase_velocity_forward_batches": float(
                velocity_forward_batches
            ),
            "batched_discrete_phase_velocity_forward_items": float(
                velocity_forward_items
            ),
            "batched_discrete_phase_ar_forward_batches": float(ar_forward_batches),
            "batched_discrete_phase_ar_forward_items": float(ar_forward_items),
            "batched_discrete_phase_tokenizer_edge_length_reuse_items": float(
                tokenizer_edge_length_reuse_items
            ),
            "batched_discrete_phase_tokenizer_edge_length_reuse_fallback_items": float(
                tokenizer_edge_length_reuse_fallback_items
            ),
        }
        return rows, stats

    def _sample_compare_harness_once(self, pair, train=True):
        sampled_trees, _, _, _, _, trace = self.sample(
            [pair["start_tree"]],
            **self._build_harness_sample_kwargs(pair, train=train),
        )
        sampled_tree = sampled_trees[0]
        likelihood_scorer = self._get_branch_relax_likelihood_scorer()
        metrics = {
            "rf_norm": float(calculate_norm_rf(sampled_tree, pair["target_tree"])),
            "start_rf_norm": float(
                calculate_norm_rf(pair["start_tree"], pair["target_tree"])
            ),
            "_start_tree": str(pair["start_tree"]),
            "_original_start_tree": (
                str(pair["original_start_tree"])
                if pair.get("original_start_tree") is not None
                else None
            ),
            "_sampled_tree": str(sampled_tree),
            "_target_tree": str(pair["target_tree"]),
            "_n_leaves": int(pair["n_leaves"]),
            "_bank_group_key": pair.get("bank_group_key"),
            "_source_bank_index": pair.get("source_bank_index"),
        }
        if likelihood_scorer is not None:
            metrics["branch_relax_after_log_likelihood"] = float(
                likelihood_scorer.log_likelihood(str(sampled_tree))
            )
        if self.sample_metrics_trace_topology_repeats_enabled:
            metrics.update(_summarize_trace_topology_repeats(trace))
        metrics["stopped_for_repeated_topology"] = float(
            1.0 if trace.get("stopped_for_repeated_topology", False) else 0.0
        )
        metrics["stopped_for_no_valid_merge"] = float(
            1.0 if trace.get("stopped_for_no_valid_merge", False) else 0.0
        )
        metrics["skipped_no_valid_boundary_revisits"] = float(
            trace.get("skipped_no_valid_boundary_revisits", 0.0)
        )
        return metrics

    def _summarize_sample_compare_harness_rows(self, rows, train=True):
        if len(rows) == 1:
            row = dict(rows[0])
            metrics = {"num_pairs": 1}
            for key in ("rf_norm", "start_rf_norm"):
                value = row.get(key)
                if isinstance(value, (int, float, np.generic)):
                    scalar = float(value)
                    metrics[key] = scalar
                    metrics[f"{key}_mean"] = scalar
                    metrics[f"{key}_median"] = scalar
                    metrics[f"{key}_best"] = scalar
                    metrics[f"{key}_worst"] = scalar
                    metrics[f"{key}_min"] = scalar
                    metrics[f"{key}_max"] = scalar
                    metrics[f"{key}_p10"] = scalar
                    metrics[f"{key}_p90"] = scalar
            metrics.update(row)
            sampled_tree = row.get("_sampled_tree")
            relaxed_tree_rows = []
            if sampled_tree:
                downstream_metrics, relaxed_tree_rows = (
                    self._sample_metrics_relaxed_downstream_metrics(
                        [sampled_tree],
                        train=train,
                        return_tree_rows=True,
                        run_mrbayes=False,
                    )
                )
                metrics.update(downstream_metrics)
            metrics.update(
                self._write_sample_metrics_tree_dump(
                    [row],
                    relaxed_tree_rows=relaxed_tree_rows,
                    train=train,
                )
            )
            if relaxed_tree_rows:
                metrics.update(
                    self._sample_metrics_run_mrbayes20k(
                        [row["relaxed_tree"] for row in relaxed_tree_rows],
                        train=train,
                    )
                )
            return metrics

        def _numeric_values(key):
            values = []
            for row in rows:
                value = row.get(key)
                if isinstance(value, (int, float, np.generic)):
                    values.append(float(value))
            return values

        metrics = {"num_pairs": int(len(rows))}
        for key in ("rf_norm", "start_rf_norm"):
            values = _numeric_values(key)
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            metrics[key] = float(arr.mean())
            metrics[f"{key}_mean"] = float(arr.mean())
            metrics[f"{key}_median"] = float(np.median(arr))
            metrics[f"{key}_best"] = float(arr.min())
            metrics[f"{key}_worst"] = float(arr.max())
            metrics[f"{key}_min"] = float(arr.min())
            metrics[f"{key}_max"] = float(arr.max())
            metrics[f"{key}_p10"] = float(np.quantile(arr, 0.10))
            metrics[f"{key}_p90"] = float(np.quantile(arr, 0.90))

        sampled_trees = [row.get("_sampled_tree") for row in rows]
        target_trees = [row.get("_target_tree") for row in rows]
        relaxed_tree_rows = []
        n_leaves_values = [
            int(row.get("_n_leaves"))
            for row in rows
            if row.get("_n_leaves") is not None
        ]
        if (
            len(sampled_trees) == len(rows)
            and len(target_trees) == len(rows)
            and all(sampled_trees)
            and all(target_trees)
        ):
            if n_leaves_values and len(set(n_leaves_values)) == 1:
                metrics.update(
                    kl_divergence_topological_distributions(
                        target_trees,
                        sampled_trees,
                        num_leaves=int(n_leaves_values[0]),
                    )
                )
            metrics.update(
                kl_divergence_tree_topology_distributions(
                    target_trees,
                    sampled_trees,
                )
            )
            metrics.update(
                self._posterior_reference_metrics(sampled_trees, train=train)
            )
            downstream_metrics, relaxed_tree_rows = (
                self._sample_metrics_relaxed_downstream_metrics(
                    sampled_trees,
                    train=train,
                    return_tree_rows=True,
                    run_mrbayes=False,
                )
            )
            metrics.update(downstream_metrics)

        metrics.update(
            self._write_sample_metrics_tree_dump(
                rows,
                relaxed_tree_rows=relaxed_tree_rows,
                train=train,
            )
        )
        if relaxed_tree_rows:
            metrics.update(
                self._sample_metrics_run_mrbayes20k(
                    [row["relaxed_tree"] for row in relaxed_tree_rows],
                    train=train,
                )
            )

        aggregate_keys = sorted(
            {
                key
                for row in rows
                for key in row.keys()
                if key not in {"rf_norm", "start_rf_norm"} and not str(key).startswith("_")
            }
        )
        for key in aggregate_keys:
            values = _numeric_values(key)
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            metrics[f"{key}_mean"] = float(arr.mean())
            metrics[f"{key}_max"] = float(arr.max())
        return metrics

    def _sample_metrics_bank_size(self, dataset_split):
        selections = getattr(dataset_split, "_frozen_full_path_control_selections", None)
        if selections:
            return len(selections)
        start_bank = getattr(dataset_split, "overfit_fixed_pair_start_tree_newick_bank", [])
        target_bank = getattr(dataset_split, "overfit_fixed_pair_target_tree_newick_bank", [])
        if target_bank:
            return len(target_bank)
        if start_bank:
            return len(start_bank)
        try:
            return len(dataset_split)
        except TypeError:
            return 0

    def _sample_metrics_select_bank_indices(self, total_count, num_pairs, train=True):
        total_count = max(0, int(total_count))
        if total_count <= 0:
            return []
        num_pairs = min(max(1, int(num_pairs)), total_count)
        mode = str(
            getattr(self, "sample_metrics_unseen_pair_selection_mode", "random_bank")
        ).strip().lower()
        if mode in {"first", "sequential"}:
            return list(range(num_pairs))
        seed = (
            int(self.sample_metrics_unseen_start_seed)
            + int(self.global_step) * 1009
            + (0 if train else 17)
        )
        rng = random.Random(seed)
        return rng.sample(range(total_count), k=num_pairs)

    def _sample_metrics_build_bank_pair(self, dataset_split, pair_index):
        fixed_pair = None
        sampled = None
        if hasattr(dataset_split, "get_overfit_fixed_pair"):
            try:
                fixed_pair = dataset_split.get_overfit_fixed_pair(int(pair_index))
            except Exception:
                fixed_pair = None
        if fixed_pair is None:
            sampled = dataset_split[int(pair_index)]
            if hasattr(dataset_split, "get_overfit_fixed_pair"):
                try:
                    fixed_pair = dataset_split.get_overfit_fixed_pair(int(pair_index))
                except Exception:
                    fixed_pair = None

        if fixed_pair is not None:
            start_tree = fixed_pair.get("random_tree", fixed_pair.get("start_tree"))
            target_tree = fixed_pair.get(
                "effective_target_tree", fixed_pair.get("target_tree")
            )
            max_events_value = int(
                len(fixed_pair.get("final_labels", []) or [])
                or fixed_pair.get("fixed_pair_num_events", 1024)
            )
            name_mapping_value = fixed_pair.get("name_mapping")
            bank_group_key_value = fixed_pair.get("bank_group_key")
        else:
            start_tree = sampled.get("start_tree")
            target_tree = sampled.get("target_tree")
            max_events_value = int(sampled.get("fixed_pair_num_events", 1024))
            name_mapping_value = sampled.get("name_mapping")
            bank_group_key_value = sampled.get("bank_group_key")

        mapping = (
            name_mapping_value
            if name_mapping_value is not None
            else (
                dataset_split.return_nexus_number_to_name(0)
                if hasattr(dataset_split, "return_nexus_number_to_name")
                else None
            )
        )
        return {
            "start_tree": str(start_tree),
            "target_tree": str(target_tree),
            "bank_group_key": bank_group_key_value,
            "n_leaves": len(EteTree(str(start_tree), format=1).get_leaves()),
            "max_events": int(max_events_value),
            "name_mapping": mapping,
            "source_bank_index": int(pair_index),
        }

    def _sample_metrics_topology_key(self, tree):
        try:
            return canonicalize_topology_newick(str(tree))
        except Exception:
            return str(tree)

    def _sample_metrics_training_start_topology_keys(self, dataset_split):
        bank = getattr(dataset_split, "overfit_fixed_pair_start_tree_newick_bank", [])
        cache = getattr(self, "_sample_metrics_training_start_topology_key_cache", None)
        if cache is None:
            cache = {}
            self._sample_metrics_training_start_topology_key_cache = cache
        cache_key = (id(dataset_split), id(bank), len(bank))
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        keys = {self._sample_metrics_topology_key(tree) for tree in bank}
        cache[cache_key] = keys
        return keys

    def _sample_metrics_generate_unseen_start(
        self,
        dataset_split,
        target_tree,
        seen,
        training_starts=None,
    ):
        if training_starts is None:
            training_starts = self._sample_metrics_training_start_topology_keys(
                dataset_split
            )
        candidate = None
        for _attempt in range(self.sample_metrics_unseen_start_max_duplicate_tries):
            candidate = str(dataset_split.sample_random_tree(str(target_tree)))
            candidate_key = self._sample_metrics_topology_key(candidate)
            if candidate_key not in seen and candidate_key not in training_starts:
                seen.add(candidate_key)
                return candidate
        if candidate is None:
            candidate = str(dataset_split.sample_random_tree(str(target_tree)))
        seen.add(self._sample_metrics_topology_key(candidate))
        return candidate

    def _sample_metrics_unseen_bank_pairs(self, dataset_split, train=True):
        num_pairs = max(1, int(getattr(self, "sample_metrics_num_pairs", 1)))
        bank_size = self._sample_metrics_bank_size(dataset_split)
        indices = self._sample_metrics_select_bank_indices(
            bank_size,
            num_pairs,
            train=train,
        )
        random_state = random.getstate()
        seed = (
            int(self.sample_metrics_unseen_start_seed)
            + int(self.global_step) * 9973
            + (0 if train else 31)
        )
        pairs = []
        seen_starts = set()
        training_start_keys = self._sample_metrics_training_start_topology_keys(
            dataset_split
        )
        try:
            random.seed(seed)
            for eval_index, source_index in enumerate(indices):
                pair = self._sample_metrics_build_bank_pair(dataset_split, source_index)
                original_start_tree = pair["start_tree"]
                unseen_start_tree = self._sample_metrics_generate_unseen_start(
                    dataset_split,
                    pair["target_tree"],
                    seen_starts,
                    training_start_keys,
                )
                pair["original_start_tree"] = original_start_tree
                pair["start_tree"] = unseen_start_tree
                pair["bank_group_key"] = f"sample_metrics_unseen_case{eval_index:05d}"
                pair["n_leaves"] = len(EteTree(unseen_start_tree, format=1).get_leaves())
                pair["source_bank_index"] = int(source_index)
                pairs.append(pair)
        finally:
            random.setstate(random_state)
        return pairs

    def _sample_metrics_model_uses_frozen_start_case_table(self):
        return any(
            hasattr(self.model, name)
            for name in (
                "first_hit_frozen_start_case_embedding",
                "autoregressive_frozen_start_case_embedding",
            )
        )

    def _sample_metrics_model_needs_case_indices(self):
        needs_frozen_ar_case_probe = (
            bool(
                getattr(
                    self.model,
                    "autoregressive_use_start_topology_conditioning",
                    False,
                )
            )
            and getattr(
                self.model,
                "autoregressive_start_topology_conditioning_mode",
                "additive",
            )
            in {"frozen_case_probe", "frozen_case_probe_additive"}
        )
        return (
            getattr(self.model, "first_hit_head_mode", "base")
            in {"case_adapted_mlp", "frozen_start_case_mlp"}
            or getattr(self.model, "autoregressive_use_case_conditioning", False)
            or needs_frozen_ar_case_probe
        )

    def _get_sample_metrics_metric_encoder(self):
        path = self.sample_metrics_unseen_start_metric_encoder_path
        if not path:
            return None
        cached = self._sample_metrics_metric_encoder_cache.get(path)
        if cached is not None:
            return cached

        from scripts.pretrain_start_tree_metric_encoder import SplitSetEncoder

        checkpoint = torch.load(path, map_location="cpu")
        metadata = dict(checkpoint.get("metadata", {}))
        max_bits = int(metadata.get("max_bits", 64))
        hidden_dim = int(metadata.get("hidden_dim", 128))
        embedding_dim = int(metadata.get("embedding_dim", 64))
        encoder = SplitSetEncoder(
            max_bits=max_bits,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            dropout=0.0,
        )
        state_dict = checkpoint.get("encoder_state_dict")
        if state_dict is None and checkpoint.get("model_state_dict") is not None:
            state_dict = {
                key[len("encoder.") :]: value
                for key, value in checkpoint["model_state_dict"].items()
                if str(key).startswith("encoder.")
            }
        if not state_dict:
            raise ValueError(
                "Metric encoder checkpoint must contain encoder_state_dict or "
                f"an encoder.* model_state_dict: {path}"
            )
        encoder.load_state_dict(state_dict)
        encoder.eval()
        cached = {
            "encoder": encoder,
            "max_bits": max_bits,
            "embedding_dim": embedding_dim,
        }
        self._sample_metrics_metric_encoder_cache[path] = cached
        return cached

    def _sample_metrics_encode_metric_starts(self, start_trees):
        encoder_bundle = self._get_sample_metrics_metric_encoder()
        if encoder_bundle is None:
            return None, {}

        from scripts.pretrain_start_tree_metric_encoder import (
            build_tree_tensor_batch,
            canonical_internal_splits,
            embedding_stats,
        )

        split_sets = []
        n_taxa = []
        for tree in start_trees:
            splits, inferred_n_taxa = canonical_internal_splits(str(tree))
            split_sets.append(splits)
            n_taxa.append(int(inferred_n_taxa))

        device = torch.device("cpu")
        encoder = encoder_bundle["encoder"].to(device)
        split_bits, pad_mask, size_features = build_tree_tensor_batch(
            split_sets,
            n_taxa,
            max_bits=int(encoder_bundle["max_bits"]),
            device=device,
        )
        with torch.no_grad():
            embeddings = encoder(split_bits, pad_mask, size_features)
            embeddings = F.normalize(embeddings, dim=-1)
        stats = embedding_stats(embeddings)
        return embeddings.detach().cpu(), stats

    def _sample_metrics_replace_frozen_start_case_tables(self, embeddings):
        replacements = []
        if embeddings is None:
            return replacements
        for name in (
            "first_hit_frozen_start_case_embedding",
            "autoregressive_frozen_start_case_embedding",
        ):
            if not hasattr(self.model, name):
                continue
            current = getattr(self.model, name)
            replacements.append((name, current))
            table = embeddings.to(device=current.device, dtype=current.dtype)
            setattr(self.model, name, table)
        return replacements

    def _sample_metrics_restore_frozen_start_case_tables(self, replacements):
        for name, value in reversed(replacements):
            setattr(self.model, name, value)

    def _sample_compare_harness_unseen_starts(self, train=True):
        dataset_split = self.dataset.dataset_train if train else self.dataset.dataset_val
        pairs = self._sample_metrics_unseen_bank_pairs(dataset_split, train=train)
        if not pairs:
            return {"num_pairs": 0, "unseen_start_eval": 1.0}

        needs_case_indices = self._sample_metrics_model_needs_case_indices()
        uses_frozen_tables = self._sample_metrics_model_uses_frozen_start_case_table()
        if needs_case_indices and not uses_frozen_tables:
            raise RuntimeError(
                "Unseen-start eval does not support trainable case-ID conditioning; "
                "use no conditioning or a frozen start-case table."
            )

        embeddings = None
        embedding_stats_block = {}
        if uses_frozen_tables:
            embeddings, embedding_stats_block = self._sample_metrics_encode_metric_starts(
                [pair["start_tree"] for pair in pairs]
            )
            if embeddings is None:
                raise RuntimeError(
                    "sample_metrics_unseen_start_metric_encoder_path is required "
                    "when unseen-start eval is used with frozen case-table conditioning."
                )

        replacements = self._sample_metrics_replace_frozen_start_case_tables(embeddings)
        try:
            rows, batch_stats = self._sample_compare_harness_batch_discrete_phase(
                pairs,
                train=train,
            )
        finally:
            self._sample_metrics_restore_frozen_start_case_tables(replacements)

        metrics = self._summarize_sample_compare_harness_rows(rows, train=train)
        metrics.update(batch_stats)
        metrics.update(
            {
                "unseen_start_eval": 1.0,
                "unseen_start_count": float(len(pairs)),
                "unseen_start_unique_count": float(
                    len({str(pair["start_tree"]) for pair in pairs})
                ),
                "unseen_start_unique_topology_count": float(
                    len(
                        {
                            canonicalize_topology_newick(str(pair["start_tree"]))
                            for pair in pairs
                        }
                    )
                ),
                "unseen_source_bank_unique_count": float(
                    len({int(pair["source_bank_index"]) for pair in pairs})
                ),
            }
        )
        for key, value in embedding_stats_block.items():
            metrics[f"unseen_start_embedding_{key}"] = float(value)
        return metrics

    def sample_compare_harness(self, train=True):
        if getattr(self, "sample_metrics_unseen_start_eval", False):
            return self._sample_compare_harness_unseen_starts(train=train)
        num_pairs = max(1, int(getattr(self, "sample_metrics_num_pairs", 1)))
        pairs_to_sample = []
        dataset_split = self.dataset.dataset_train if train else self.dataset.dataset_val
        if getattr(dataset_split, "overfit_fixed_pair", False) and int(num_pairs) == 1:
            fixed_pair = self._get_fixed_pair_sampling_details(train=train)
            if fixed_pair is not None:
                pair = {
                    "start_tree": fixed_pair.get("random_tree", fixed_pair.get("start_tree")),
                    "target_tree": fixed_pair.get(
                        "effective_target_tree", fixed_pair.get("target_tree")
                    ),
                    "bank_group_key": fixed_pair.get("bank_group_key"),
                    "n_leaves": len(
                        EteTree(
                            fixed_pair.get("random_tree", fixed_pair.get("start_tree")),
                            format=1,
                        ).get_leaves()
                    ),
                    "max_events": int(
                        len(fixed_pair.get("final_labels", []) or [])
                        or fixed_pair.get("fixed_pair_num_events", 1024)
                    ),
                    "name_mapping": (
                        fixed_pair.get("name_mapping")
                        if fixed_pair.get("name_mapping") is not None
                        else (
                            dataset_split.return_nexus_number_to_name(0)
                            if hasattr(dataset_split, "return_nexus_number_to_name")
                            else None
                        )
                    ),
                }
                rows = [self._sample_compare_harness_once(pair, train=train)]
                metrics = self._summarize_sample_compare_harness_rows(rows, train=train)
                metrics.update(self._evaluate_fixed_pair_path_metrics(train=train))
                return metrics
        if (
            getattr(dataset_split, "_frozen_full_path_control_selections", None)
        ):
            max_pairs = min(
                int(num_pairs),
                len(getattr(dataset_split, "_frozen_full_path_control_selections", [])),
            )
            for pair_index in range(max_pairs):
                fixed_pair = None
                if hasattr(dataset_split, "get_overfit_fixed_pair"):
                    fixed_pair = dataset_split.get_overfit_fixed_pair(pair_index)
                sampled = None
                if fixed_pair is None:
                    sampled = dataset_split[pair_index]
                    if hasattr(dataset_split, "get_overfit_fixed_pair"):
                        fixed_pair = dataset_split.get_overfit_fixed_pair(pair_index)
                if fixed_pair is not None:
                    start_tree = fixed_pair.get("random_tree", fixed_pair.get("start_tree"))
                    target_tree = fixed_pair.get(
                        "effective_target_tree", fixed_pair.get("target_tree")
                    )
                    max_events_value = int(
                        len(fixed_pair.get("final_labels", []) or [])
                        or fixed_pair.get("fixed_pair_num_events", 1024)
                    )
                    name_mapping_value = fixed_pair.get("name_mapping")
                    bank_group_key_value = fixed_pair.get("bank_group_key")
                else:
                    start_tree = sampled.get("start_tree")
                    target_tree = sampled.get("target_tree")
                    max_events_value = int(sampled.get("fixed_pair_num_events", 1024))
                    name_mapping_value = None
                    bank_group_key_value = sampled.get("bank_group_key")
                pair = {
                    "start_tree": start_tree,
                    "target_tree": target_tree,
                    "bank_group_key": bank_group_key_value,
                    "n_leaves": len(EteTree(start_tree, format=1).get_leaves()),
                    "max_events": max_events_value,
                    "name_mapping": (
                        name_mapping_value
                        if name_mapping_value is not None
                        else (
                            dataset_split.return_nexus_number_to_name(0)
                            if hasattr(dataset_split, "return_nexus_number_to_name")
                            else None
                        )
                    ),
                }
                pairs_to_sample.append(pair)
        else:
            for _ in range(num_pairs):
                pair = self._get_harness_sampling_pair(
                    train=train,
                    frozen_start_bank=True,
                )
                pairs_to_sample.append(pair)

        rows, batch_stats = self._sample_compare_harness_batch_discrete_phase(
            pairs_to_sample,
            train=train,
        )
        metrics = self._summarize_sample_compare_harness_rows(rows, train=train)
        metrics.update(batch_stats)
        if num_pairs == 1:
            metrics.update(self._evaluate_fixed_pair_path_metrics(train=train))
        return metrics
