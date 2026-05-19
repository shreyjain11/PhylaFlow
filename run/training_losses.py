import inspect
import logging
import math
import numbers
import random

import numpy as np
import torch
import torch.nn.functional as F
from ete3 import Tree as EteTree

from utils.bhv_utils import (
    BHVEncoder,
    _split_multi_label_training_events,
    get_structural_polytomy_groups_from_newick,
    return_sampled_tree_boundary_decisions,
    return_sampled_tree_orthant_velocity,
)
from utils.random_tree import Tree
from utils.utils import (
    _pick_knn_pair,
    _velocity_diagnostics,
    compute_merge_metrics,
    find_polytomy_nodes,
    has_polytomy_fast,
    pick_group,
    remove_bit,
)
from run.training_helpers import *

logger = logging.getLogger(__name__)


class TrainingLossMixin:
    def _first_hit_bucket_ids(self, values):
        return torch.zeros_like(values, dtype=torch.long)

    def _refine_velocity_predictions(
        self,
        velocity_pred,
        lengths,
        edge_features=None,
        group_sizes=None,
    ):
        return velocity_pred

    def _compute_first_hit_logits(
        self,
        first_hit_logits,
        lengths,
        velocities,
        edge_features=None,
        group_sizes=None,
    ):
        return first_hit_logits

    def _effective_autoregressive_time_value(self, time_value):
        if not self.autoregressive_use_time:
            return 0.0
        return float(time_value)

    def _effective_autoregressive_time_tensor(self, time_value):
        if torch.is_tensor(time_value):
            if self.autoregressive_use_time:
                return time_value
            return torch.zeros_like(time_value, dtype=torch.float32, device=time_value.device)
        return torch.tensor(
            [self._effective_autoregressive_time_value(time_value)],
            dtype=torch.float32,
            device=self.device,
        )

    def _sampling_autoregressive_time_value(
        self,
        current_time,
        event_index=None,
        max_events=None,
    ):
        if not self.autoregressive_use_time:
            return 0.0
        if event_index is None or max_events is None:
            return float(current_time)
        max_events = int(max_events)
        if max_events <= 1:
            return 0.0
        clipped_index = min(max(int(event_index), 0), max_events - 1)
        return float(clipped_index / float(max_events - 1))

    def _sampling_autoregressive_time_tensor(
        self,
        current_time,
        event_index=None,
        max_events=None,
    ):
        return torch.tensor(
            [
                self._sampling_autoregressive_time_value(
                    current_time,
                    event_index=event_index,
                    max_events=max_events,
                )
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def _rollin_single_autoregressive_state(
        self,
        current_newick,
        target_newick,
        current_time,
        phyla_embedding=None,
    ):
        planned_merge = _plan_first_autoregressive_model_merge(
            self,
            current_newick=current_newick,
            current_time=current_time,
            phyla_embedding=phyla_embedding,
        )
        if planned_merge is None:
            return None

        rolled_newick = _apply_merge_subset_to_newick(
            self.model.tokenizer,
            current_newick,
            planned_merge["subset"],
            new_split=planned_merge["new_split"],
        )
        if rolled_newick is None:
            return None

        corrective_events = return_sampled_tree_boundary_decisions(
            rolled_newick,
            target_newick,
        )
        if not corrective_events:
            return None

        next_event = corrective_events[0]
        return {
            "newick": rolled_newick,
            "labels": next_event["labels"],
            "stop_after_merge": bool(next_event.get("stop_after_merge", False)),
            "time": self._effective_autoregressive_time_value(current_time),
        }

    def _dagger_rollin_single_autoregressive_state(
        self,
        current_newick,
        target_newick,
        current_time,
        phyla_embedding=None,
    ):
        oracle_training_topologies = set(
            _oracle_training_topology_keys(current_newick, target_newick)
        )
        state_newick = current_newick
        state_time = float(current_time)

        for rollout_step in range(self.autoregressive_dagger_max_steps):
            planned_merge = _plan_first_autoregressive_model_merge(
                self,
                current_newick=state_newick,
                current_time=state_time,
                phyla_embedding=phyla_embedding,
            )
            if planned_merge is None:
                return None

            next_newick = _apply_merge_subset_to_newick(
                self.model.tokenizer,
                state_newick,
                planned_merge["subset"],
                new_split=planned_merge["new_split"],
            )
            if next_newick is None or next_newick == state_newick:
                return None

            if _topology_key(next_newick) not in oracle_training_topologies:
                corrective_events = return_sampled_tree_boundary_decisions(
                    next_newick,
                    target_newick,
                )
                if not corrective_events:
                    return None

                next_event = corrective_events[0]
                return {
                    "newick": next_newick,
                    "labels": next_event["labels"],
                    "stop_after_merge": bool(next_event.get("stop_after_merge", False)),
                    "time": self._effective_autoregressive_time_value(state_time),
                    "rollout_steps": rollout_step + 1,
                }

            state_newick = next_newick

        return None

    def _perturb_autoregressive_single_state(
        self,
        current_newick,
        target_newick,
        current_time,
        phyla_embedding=None,
    ):
        if self.autoregressive_structure_perturb_mode == "model_wrong_pair":
            chosen_merge = _choose_model_wrong_pair_merge_subset(
                self,
                current_newick=current_newick,
                target_newick=target_newick,
                current_time=current_time,
                phyla_embedding=phyla_embedding,
            )
        else:
            subset = _choose_wrong_pair_merge_subset(
                current_newick,
                target_newick,
                self.model.tokenizer,
            )
            chosen_merge = None if subset is None else {"subset": subset, "new_split": None}

        if chosen_merge is None:
            return None

        perturbed_newick = _apply_merge_subset_to_newick(
            self.model.tokenizer,
            current_newick,
            chosen_merge["subset"],
            new_split=chosen_merge.get("new_split"),
        )
        if perturbed_newick is None:
            return None

        corrective_events = return_sampled_tree_boundary_decisions(
            perturbed_newick,
            target_newick,
        )
        if not corrective_events:
            return None

        next_event = corrective_events[0]
        return {
            "newick": perturbed_newick,
            "labels": next_event["labels"],
            "stop_after_merge": bool(next_event.get("stop_after_merge", False)),
            "time": self._effective_autoregressive_time_value(current_time),
        }

    def _attach_case_indices_to_batch(self, batch):
        if batch is None:
            return batch
        if getattr(self.model, "first_hit_head_mode", "base") not in {
            "case_adapted_mlp",
            "frozen_start_case_mlp",
        }:
            return batch
        if batch.get("_first_hit_case_indices") is not None:
            return batch
        group_keys = batch.get("bank_group_key")
        if group_keys is None:
            group_keys = _infer_case_group_keys_from_batch(self, batch)
        if group_keys is None:
            return batch
        if isinstance(group_keys, str):
            group_keys = [group_keys]
        case_index_tensor = _build_case_index_tensor_from_group_keys(
            group_keys,
            device=self.device,
            module=self,
        )
        if case_index_tensor is None:
            return batch
        updated_batch = dict(batch)
        updated_batch["bank_group_key"] = list(group_keys)
        updated_batch["_first_hit_case_indices"] = case_index_tensor
        return updated_batch

    def _attach_start_topology_features_to_batch(self, batch):
        if batch is None:
            return batch
        mode = getattr(self.model, "first_hit_head_mode", "base")
        first_hit_summary_modes = {
            "start_topology_adapter_mlp",
            "start_topology_raw_pool_concat_mlp",
        }
        needs_first_hit_summary = mode in first_hit_summary_modes
        needs_first_hit_cross_attn = mode == "start_topology_cross_attn_mlp"
        autoregressive_start_topology_mode = getattr(
            self.model,
            "autoregressive_start_topology_conditioning_mode",
            "additive",
        )
        needs_autoregressive_summary = bool(
            getattr(self.model, "autoregressive_use_start_topology_conditioning", False)
            and autoregressive_start_topology_mode
            not in {"frozen_case_probe", "frozen_case_probe_additive"}
        )
        if (
            not needs_first_hit_summary
            and not needs_first_hit_cross_attn
            and not needs_autoregressive_summary
        ):
            return batch
        summary_ready = (
            (
                not needs_first_hit_summary
                or batch.get("_first_hit_start_topology_features") is not None
            )
            and (
                not needs_autoregressive_summary
                or batch.get("_autoregressive_start_topology_features") is not None
            )
        )
        cross_ready = (
            not needs_first_hit_cross_attn
            or (
                batch.get("_first_hit_start_topology_embeddings") is not None
                and batch.get("_first_hit_start_topology_pad_mask") is not None
            )
        )
        if summary_ready and cross_ready:
            return batch
        start_trees = batch.get("start_trees")
        if start_trees is None:
            start_trees = batch.get("original_trees")
        if start_trees is None:
            return batch
        if isinstance(start_trees, str):
            start_trees = [start_trees]
        updated_batch = dict(batch)
        if needs_first_hit_summary or needs_autoregressive_summary:
            feature_tensor = batch.get("_first_hit_start_topology_features")
            if feature_tensor is None:
                feature_tensor = batch.get("_autoregressive_start_topology_features")
            if feature_tensor is None:
                feature_tensor = _build_start_topology_feature_tensor(
                    self,
                    list(start_trees),
                    device=self.device,
                )
            if feature_tensor is None:
                return batch
            if needs_first_hit_summary:
                updated_batch["_first_hit_start_topology_features"] = feature_tensor
            if needs_autoregressive_summary:
                updated_batch["_autoregressive_start_topology_features"] = feature_tensor
        if needs_first_hit_cross_attn:
            embeddings = batch.get("_first_hit_start_topology_embeddings")
            pad_mask = batch.get("_first_hit_start_topology_pad_mask")
            if embeddings is None or pad_mask is None:
                embeddings, pad_mask = _build_start_topology_identity_batch(
                    self,
                    list(start_trees),
                    device=self.device,
                )
            if embeddings is None or pad_mask is None:
                return batch
            updated_batch["_first_hit_start_topology_embeddings"] = embeddings
            updated_batch["_first_hit_start_topology_pad_mask"] = pad_mask
        return updated_batch

    def _attach_start_tree_graph_context_to_batch(self, batch):
        if batch is None:
            return batch
        if getattr(self.model, "first_hit_head_mode", "base") != "start_tree_graph_token_mlp":
            return batch
        if batch.get("_first_hit_start_tree_graph_context") is not None:
            return batch
        start_trees = batch.get("start_trees")
        if start_trees is None:
            start_trees = batch.get("original_trees")
        phyla_embeddings = batch.get("phyla_embeddings")
        if start_trees is None:
            return batch
        if isinstance(start_trees, str):
            start_trees = [start_trees]
        graph_context = _build_start_tree_graph_context(
            self,
            list(start_trees),
            phyla_embeddings,
            device=self.device,
            detach=getattr(self.model, "first_hit_start_tree_graph_detach", False),
        )
        if graph_context is None:
            return batch
        updated_batch = dict(batch)
        updated_batch["_first_hit_start_tree_graph_context"] = graph_context
        return updated_batch

    def _prepare_velocity_training_batch(self, batch):
        batch = self._attach_case_indices_to_batch(batch)
        batch = self._attach_start_topology_features_to_batch(batch)
        batch = self._attach_start_tree_graph_context_to_batch(batch)
        if batch.get("_skip_training_augmentations", False):
            return batch, {"attempted": 0.0, "applied": 0.0}
        if (
            self.velocity_length_jitter_prob <= 0.0
            or self.velocity_length_jitter_scale <= 0.0
            or "original_trees" not in batch
            or "target_trees" not in batch
        ):
            return batch, {"attempted": 0.0, "applied": 0.0}

        newicks = list(batch["original_trees"])
        velocity_labels = list(batch["batched_velocity"])
        batched_time = batch.get("batched_time")
        updated_times = batched_time.clone() if batched_time is not None else None
        attempted = 0
        applied = 0

        for batch_index, (current_newick, target_newick) in enumerate(
            zip(newicks, batch["target_trees"])
        ):
            if random.random() > self.velocity_length_jitter_prob:
                continue
            attempted += 1

            perturbed_newick = _jitter_internal_lengths_newick(
                current_newick,
                self.velocity_length_jitter_scale,
            )
            if perturbed_newick is None:
                continue

            try:
                sampled_newick, perturbed_velocity = return_sampled_tree_orthant_velocity(
                    perturbed_newick,
                    target_newick,
                    0.0,
                )
            except Exception:
                continue

            newicks[batch_index] = sampled_newick
            velocity_labels[batch_index] = perturbed_velocity
            if updated_times is not None:
                updated_times[batch_index] = 0.0
            applied += 1

        if applied == 0:
            return batch, {"attempted": float(attempted), "applied": 0.0}

        updated_batch = dict(batch)
        updated_batch["original_trees"] = newicks
        updated_batch["batched_velocity"] = velocity_labels
        if updated_times is not None:
            updated_batch["batched_time"] = updated_times
        updated_batch["tokenized_trees"] = _move_tokenized_batch_to_device(
            self.model.tokenizer(newicks),
            self.device,
        )
        return updated_batch, {"attempted": float(attempted), "applied": float(applied)}

    def _prepare_autoregressive_training_batch(self, batch):
        batch = self._attach_start_topology_features_to_batch(batch)
        needs_autoregressive_case_indices = bool(
            getattr(self.model, "autoregressive_use_case_conditioning", False)
        ) or (
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
        if needs_autoregressive_case_indices and batch.get("_autoregressive_case_indices") is None:
            group_keys = batch.get("bank_group_key")
            if group_keys is None:
                group_keys = _infer_case_group_keys_from_batch(self, batch)
            if group_keys is not None:
                if isinstance(group_keys, str):
                    group_keys = [group_keys]
                case_index_tensor = _build_case_index_tensor_from_group_keys(
                    group_keys,
                    device=self.device,
                    module=self,
                )
                if case_index_tensor is not None:
                    batch = dict(batch)
                    batch["bank_group_key"] = list(group_keys)
                    batch["_autoregressive_case_indices"] = case_index_tensor
        if batch.get("_skip_training_augmentations", False):
            return batch, {
                "rollin_attempted": 0.0,
                "rollin_applied": 0.0,
                "dagger_attempted": 0.0,
                "dagger_applied": 0.0,
                "dagger_rollout_steps": 0.0,
                "structure_perturb_attempted": 0.0,
                "structure_perturb_applied": 0.0,
            }
        if (
            self.autoregressive_rollin_prob <= 0.0
            and self.autoregressive_dagger_prob <= 0.0
            and self.autoregressive_structure_perturb_prob <= 0.0
            or "target_trees" not in batch
            or "newick_autoregressive_trees" not in batch
        ):
            return batch, {
                "rollin_attempted": 0.0,
                "rollin_applied": 0.0,
                "dagger_attempted": 0.0,
                "dagger_applied": 0.0,
                "dagger_rollout_steps": 0.0,
                "structure_perturb_attempted": 0.0,
                "structure_perturb_applied": 0.0,
            }

        newicks = list(batch["newick_autoregressive_trees"])
        labels = list(batch["batched_autoregressive_labels"])
        times = batch["batched_autoregressive_time"].detach().clone()
        stop_after_merge = None
        if "batched_autoregressive_stop_after_merge" in batch:
            stop_after_merge = (
                batch["batched_autoregressive_stop_after_merge"].detach().clone()
            )

        rollin_attempted = 0
        rollin_applied = 0
        dagger_attempted = 0
        dagger_applied = 0
        dagger_rollout_steps = 0
        structure_attempted = 0
        structure_applied = 0
        for batch_index, (current_newick, target_newick) in enumerate(
            zip(newicks, batch["target_trees"])
        ):
            if self.autoregressive_rollin_prob > 0.0:
                if random.random() <= self.autoregressive_rollin_prob:
                    rollin_attempted += 1

                    phyla_embedding = None
                    if batch["phyla_embeddings"] is not None:
                        phyla_embedding = batch["phyla_embeddings"][batch_index : batch_index + 1]

                    rolled = self._rollin_single_autoregressive_state(
                        current_newick=current_newick,
                        target_newick=target_newick,
                        current_time=float(times[batch_index].item()),
                        phyla_embedding=phyla_embedding,
                    )
                    if rolled is not None:
                        current_newick = rolled["newick"]
                        newicks[batch_index] = rolled["newick"]
                        labels[batch_index] = rolled["labels"]
                        if stop_after_merge is not None:
                            stop_after_merge[batch_index] = (
                                1.0 if rolled.get("stop_after_merge", False) else 0.0
                            )
                        times[batch_index] = float(rolled["time"])
                        rollin_applied += 1

            if self.autoregressive_dagger_prob > 0.0:
                if random.random() <= self.autoregressive_dagger_prob:
                    dagger_attempted += 1

                    phyla_embedding = None
                    if batch["phyla_embeddings"] is not None:
                        phyla_embedding = batch["phyla_embeddings"][
                            batch_index : batch_index + 1
                        ]

                    dagger = self._dagger_rollin_single_autoregressive_state(
                        current_newick=current_newick,
                        target_newick=target_newick,
                        current_time=float(times[batch_index].item()),
                        phyla_embedding=phyla_embedding,
                    )
                    if dagger is not None:
                        current_newick = dagger["newick"]
                        newicks[batch_index] = dagger["newick"]
                        labels[batch_index] = dagger["labels"]
                        if stop_after_merge is not None:
                            stop_after_merge[batch_index] = (
                                1.0 if dagger.get("stop_after_merge", False) else 0.0
                            )
                        times[batch_index] = float(dagger["time"])
                        dagger_applied += 1
                        dagger_rollout_steps += int(dagger["rollout_steps"])

            if self.autoregressive_structure_perturb_prob > 0.0:
                if random.random() <= self.autoregressive_structure_perturb_prob:
                    structure_attempted += 1
                    phyla_embedding = None
                    if batch["phyla_embeddings"] is not None:
                        phyla_embedding = batch["phyla_embeddings"][batch_index : batch_index + 1]
                    perturbed = self._perturb_autoregressive_single_state(
                        current_newick=current_newick,
                        target_newick=target_newick,
                        current_time=float(times[batch_index].item()),
                        phyla_embedding=phyla_embedding,
                    )
                    if perturbed is not None:
                        newicks[batch_index] = perturbed["newick"]
                        labels[batch_index] = perturbed["labels"]
                        if stop_after_merge is not None:
                            stop_after_merge[batch_index] = (
                                1.0 if perturbed.get("stop_after_merge", False) else 0.0
                            )
                        times[batch_index] = float(perturbed["time"])
                        structure_applied += 1

        if rollin_applied == 0 and dagger_applied == 0 and structure_applied == 0:
            return batch, {
                "rollin_attempted": float(rollin_attempted),
                "rollin_applied": 0.0,
                "dagger_attempted": float(dagger_attempted),
                "dagger_applied": 0.0,
                "dagger_rollout_steps": float(dagger_rollout_steps),
                "structure_perturb_attempted": float(structure_attempted),
                "structure_perturb_applied": 0.0,
            }

        updated_batch = dict(batch)
        updated_batch["newick_autoregressive_trees"] = newicks
        updated_batch["batched_autoregressive_labels"] = labels
        updated_batch["batched_autoregressive_time"] = times
        if stop_after_merge is not None:
            updated_batch["batched_autoregressive_stop_after_merge"] = stop_after_merge
        updated_batch["tokenized_autoregressive_trees"] = _move_tokenized_batch_to_device(
            self.model.tokenizer(newicks),
            self.device,
        )
        return updated_batch, {
            "rollin_attempted": float(rollin_attempted),
            "rollin_applied": float(rollin_applied),
            "dagger_attempted": float(dagger_attempted),
            "dagger_applied": float(dagger_applied),
            "dagger_rollout_steps": float(dagger_rollout_steps),
            "structure_perturb_attempted": float(structure_attempted),
            "structure_perturb_applied": float(structure_applied),
        }

    def forward(
        self,
        batched_tokenized_trees,
        t,
        phyla_embeddings,
        autoregressive=False,
        autoregressive_component_groups=None,
        autoregressive_case_indices=None,
        autoregressive_start_topology_features=None,
        first_hit_case_indices=None,
        first_hit_start_topology_features=None,
        first_hit_start_topology_embeddings=None,
        first_hit_start_topology_pad_mask=None,
        first_hit_start_tree_graph_context=None,
    ):
        batched_tokenized_trees = _move_tokenized_batch_to_device(
            batched_tokenized_trees,
            self.device,
        )
        if torch.is_tensor(t):
            t = t.to(self.device)
        if not autoregressive:
            return_first_hit_logits = (
                self.velocity_first_hit_head_weight > 0.0
                or self.velocity_first_hit_head_use_at_sampling
            )
            return_edge_features = (
                self.branch_relax_head_weight > 0.0
                or self.branch_relax_head_use_at_sampling
            )
            edge_outputs = self.model(
                batched_tokenized_trees,
                t,
                phyla_embeddings=phyla_embeddings,
                return_leafs_only=False,
                return_edges_only=True,
                return_edge_features=return_edge_features,
                return_first_hit_logits=return_first_hit_logits,
                first_hit_case_indices=first_hit_case_indices,
                first_hit_start_topology_features=first_hit_start_topology_features,
                first_hit_start_topology_embeddings=first_hit_start_topology_embeddings,
                first_hit_start_topology_pad_mask=first_hit_start_topology_pad_mask,
                first_hit_start_tree_graph_context=first_hit_start_tree_graph_context,
            )
            edge_features = None
            first_hit_logits = None
            if return_first_hit_logits:
                if return_edge_features:
                    velocity, mask, edge_features, first_hit_logits = edge_outputs
                else:
                    velocity, mask, first_hit_logits = edge_outputs
            else:
                if return_edge_features:
                    velocity, mask, edge_features = edge_outputs
                else:
                    velocity, mask = edge_outputs
            edge_split_masks = batched_tokenized_trees[-1]
            edge_mask = batched_tokenized_trees[-2]
            return (
                velocity,
                edge_split_masks,
                edge_mask,
                first_hit_logits,
                None,
                edge_features,
            )
        else:
            model_kwargs = dict(
                phyla_embeddings=phyla_embeddings,
                return_leafs_only=False,
                return_edges_only=True,
                autoregressive=True,
            )
            model_signature = inspect.signature(self.model.forward)
            supports_component_groups = (
                "autoregressive_component_groups" in model_signature.parameters
                or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in model_signature.parameters.values()
                )
            )
            if (
                autoregressive_component_groups is not None
                and supports_component_groups
            ):
                model_kwargs["autoregressive_component_groups"] = (
                    autoregressive_component_groups
                )
            supports_case_indices = (
                "autoregressive_case_indices" in model_signature.parameters
                or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in model_signature.parameters.values()
                )
            )
            if autoregressive_case_indices is not None and supports_case_indices:
                model_kwargs["autoregressive_case_indices"] = autoregressive_case_indices
            supports_start_topology_features = (
                "autoregressive_start_topology_features" in model_signature.parameters
                or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in model_signature.parameters.values()
                )
            )
            if (
                autoregressive_start_topology_features is not None
                and supports_start_topology_features
            ):
                model_kwargs["autoregressive_start_topology_features"] = (
                    autoregressive_start_topology_features
                )
            all_group_logits = self.model(
                batched_tokenized_trees,
                t,
                **model_kwargs,
            )
            return all_group_logits

    def step(self, batch, eval=False, autoregressive=False):
        logs = {}
        is_replay_batch = bool(batch.get("_is_replay_batch", False))
        if not eval and not autoregressive:
            self.current_step_value += 1
        if (
            batch["phyla_embeddings"] is None
            and "ids" in batch
        ):
            phyla_embeddings_list = []
            missing_precomputed = False
            for i in range(len(batch["ids"])):
                mapping = batch["mappings"][i]
                num_leaf = batch["num_leaves"][i]
                ordered_names = self._ordered_leaf_names_from_mapping(
                    mapping,
                    num_leaf=num_leaf,
                )
                embeddings = self._lookup_precomputed_phyla_embeddings(
                    ordered_names or [],
                    device=self.device,
                    dataset_id=batch["ids"][i],
                )
                if embeddings is None:
                    missing_precomputed = True
                    phyla_embeddings_list = []
                    break
                phyla_embeddings_list.append(embeddings)

            if phyla_embeddings_list:
                batch["phyla_embeddings"] = phyla_embeddings_list
            elif self.phyla_model is not None and missing_precomputed:
                phyla_embeddings_list = []
                for i in range(len(batch["ids"])):
                    mapping = batch["mappings"][i]
                    num_leaf = batch["num_leaves"][i]
                    seqs = []
                    names = []
                    for idx in range(num_leaf):
                        idx_str = str(idx)
                        taxon_name = mapping.get(idx_str)
                        if taxon_name:
                            seq = self.dataset.name_to_seq.get(taxon_name, "")
                            seqs.append(seq)
                            names.append(taxon_name)
                        else:
                            seqs.append("")
                            names.append("unknown")

                    embeddings = self.compute_phyla_embeddings(
                        seqs, names, device=str(self.device)
                    )
                    if embeddings.dim() == 3 and embeddings.size(0) == 1:
                        embeddings = embeddings.squeeze(0)
                    phyla_embeddings_list.append(embeddings)

                batch["phyla_embeddings"] = phyla_embeddings_list

        if not autoregressive:
            batch, velocity_perturb_stats = self._prepare_velocity_training_batch(batch)
            (
                v_pred,
                edge_split_masks,
                edge_mask,
                first_hit_logits,
                boundary_vanish_logits,
                edge_features,
            ) = self.forward(
                batch["tokenized_trees"],
                batch["batched_time"],
                batch["phyla_embeddings"],
                first_hit_case_indices=batch.get("_first_hit_case_indices"),
                first_hit_start_topology_features=batch.get(
                    "_first_hit_start_topology_features"
                ),
                first_hit_start_topology_embeddings=batch.get(
                    "_first_hit_start_topology_embeddings"
                ),
                first_hit_start_topology_pad_mask=batch.get(
                    "_first_hit_start_topology_pad_mask"
                ),
                first_hit_start_tree_graph_context=batch.get(
                    "_first_hit_start_tree_graph_context"
                ),
            )

            if self.train_tokenized_trees is None:
                self.train_tokenized_trees = batch["tokenized_trees"]
                self.train_batched_time = batch["batched_time"]
                self.train_tree = batch["original_trees"]
            # else:
            #     if calculate_norm_rf(batch['original_trees'][0], self.train_tree[0]) != 0:
            #         raise Exception("Training tree topology changed during training!")
            #     elif not torch.equal(batch["tokenized_trees"][0], self.train_tokenized_trees[0]):
            #         import pdb; pdb.set_trace()
            #         raise Exception("Training tokenized trees changed during training!")

            direct_set_loss = None
            direct_set_mse_weight = 0.0
            direct_set_loss_weight = float(
                getattr(self, "velocity_probe_direct_set_loss_weight", 1.0)
            )
            if bool(batch.get("_use_probe_parity_direct_set_loss", False)):
                direct_losses = []
                exact_flags = []
                jaccards = []
                direct_set_pos_weights = []
                target_negative_rates = []
                target_negative_losses = []
                nontarget_nonnegative_losses = []
                sample_mask = list(
                    batch.get(
                        "_probe_direct_set_sample_mask",
                        [True for _ in batch.get("original_trees", [])],
                    )
                )
                target_sets = list(
                    batch.get(
                        "_probe_direct_set_targets",
                        [[] for _ in batch.get("original_trees", [])],
                    )
                )
                direct_set_debug_enabled = (
                    os.environ.get("PHYLAFLOW_DEBUG_DIRECT_SET", "0") == "1"
                )
                autoregressive_first_hit_mode = (
                    getattr(self.model, "first_hit_head_mode", "base")
                    == "autoregressive_set"
                )
                for num, current_newick in enumerate(batch.get("original_trees", [])):
                    if num >= len(sample_mask) or not bool(sample_mask[num]):
                        continue
                    if current_newick is None:
                        continue
                    if first_hit_logits is None and not autoregressive_first_hit_mode:
                        continue
                    if autoregressive_first_hit_mode and edge_features is None:
                        continue
                    try:
                        tree_obj = Tree(current_newick)
                        n_leaves = int(tree_obj.n_leaves)
                    except Exception:
                        continue
                    split_masks_num = [int(m) for m in edge_split_masks[num]]
                    split_masks_nonzero = [m for m in split_masks_num if m != 0]
                    if not split_masks_nonzero:
                        continue
                    real_max_bit = max(m.bit_length() for m in split_masks_nonzero)
                    full_mask = (1 << real_max_bit) - 1 if real_max_bit > 0 else 0
                    try:
                        encoder = BHVEncoder()
                        bhv_masks, bhv_lengths = encoder.return_BHV_encoding(tree_obj)
                        bhv_len_map = {
                            int(m): float(l)
                            for m, l in zip(bhv_masks, bhv_lengths)
                            if l is not None
                        }
                    except Exception:
                        continue
                    target_set = {int(x) for x in target_sets[num]}
                    logits = []
                    targets = []
                    velocity_preds = []
                    matched_masks = []
                    edge_feature_rows = []
                    for edge_idx, mask in enumerate(split_masks_num):
                        if mask == 0:
                            continue
                        k_bits = int(mask).bit_count()
                        if min(k_bits, real_max_bit - k_bits) == 1:
                            continue
                        edge_length = bhv_len_map.get(int(mask))
                        if edge_length is None and full_mask:
                            edge_length = bhv_len_map.get(full_mask ^ int(mask))
                        if edge_length is None or float(edge_length) <= 1e-8:
                            continue
                        if not autoregressive_first_hit_mode:
                            logits.append(first_hit_logits[num, edge_idx, 0])
                        velocity_preds.append(v_pred[num, edge_idx, 0])
                        matched_masks.append(int(mask))
                        targets.append(1.0 if int(mask) in target_set else 0.0)
                        if edge_features is not None:
                            edge_feature_rows.append(edge_features[num, edge_idx])
                    if not velocity_preds:
                        continue
                    velocity_tensor = torch.stack(velocity_preds).reshape(-1)
                    target_tensor = torch.tensor(
                        targets,
                        device=velocity_tensor.device,
                        dtype=velocity_tensor.dtype,
                    )
                    target_mask = target_tensor > 0.5
                    if autoregressive_first_hit_mode:
                        if not edge_feature_rows:
                            continue
                        edge_feature_tensor = torch.stack(edge_feature_rows, dim=0).to(
                            device=velocity_tensor.device,
                            dtype=edge_features.dtype,
                        )
                        sample_loss, ar_stats = self.model.first_hit_autoregressive_group_loss(
                            edge_feature_tensor,
                            target_mask,
                        )
                        direct_set_pos_weights.append(1.0)
                    else:
                        logits_tensor = torch.stack(logits).reshape(-1)
                        pos_weight = None
                        if bool(
                            getattr(
                                self,
                                "velocity_probe_direct_set_positive_reweight",
                                False,
                            )
                        ):
                            pos = target_tensor.sum()
                            neg = target_tensor.numel() - pos
                            if (
                                float(pos.item()) > 0.0
                                and float(neg.item()) > 0.0
                            ):
                                pos_weight_value = torch.clamp(
                                    neg / pos, min=1.0
                                )
                                pos_weight_value = torch.pow(
                                    pos_weight_value,
                                    float(
                                        getattr(
                                            self,
                                            "velocity_probe_direct_set_positive_reweight_power",
                                            1.0,
                                        )
                                    ),
                                )
                                pos_weight_max = getattr(
                                    self,
                                    "velocity_probe_direct_set_positive_reweight_max",
                                    None,
                                )
                                if pos_weight_max is not None and pos_weight_max > 0.0:
                                    pos_weight_value = torch.clamp(
                                        pos_weight_value,
                                        min=1.0,
                                        max=float(pos_weight_max),
                                    )
                                pos_weight = pos_weight_value.detach()
                        sample_loss = F.binary_cross_entropy_with_logits(
                            logits_tensor,
                            target_tensor,
                            pos_weight=pos_weight,
                        )
                        direct_set_pos_weights.append(
                            float(pos_weight.detach().item())
                            if pos_weight is not None
                            else 1.0
                        )
                    target_negative_weight = float(
                        getattr(
                            self,
                            "velocity_probe_direct_set_target_negative_weight",
                            1.0,
                        )
                    )
                    if bool(target_mask.any().item()) and target_negative_weight > 0.0:
                        target_negative_loss = F.softplus(
                            velocity_tensor[target_mask]
                        ).mean()
                        sample_loss = sample_loss + (
                            target_negative_weight * target_negative_loss
                        )
                        target_negative_losses.append(target_negative_loss.detach())
                        with torch.no_grad():
                            target_negative_rates.append(
                                float((velocity_tensor[target_mask] < 0.0).float().mean().item())
                            )
                    nontarget_mask = ~target_mask
                    nontarget_nonnegative_weight = 0.0
                    if (
                        bool(nontarget_mask.any().item())
                        and nontarget_nonnegative_weight > 0.0
                    ):
                        nontarget_nonnegative_loss = F.softplus(
                            -velocity_tensor[nontarget_mask]
                        ).mean()
                        sample_loss = sample_loss + (
                            nontarget_nonnegative_weight
                            * nontarget_nonnegative_loss
                        )
                        nontarget_nonnegative_losses.append(
                            nontarget_nonnegative_loss.detach()
                        )
                    direct_losses.append(sample_loss)
                    with torch.no_grad():
                        if autoregressive_first_hit_mode:
                            pred_mask = self.model.predict_first_hit_autoregressive_mask(
                                edge_feature_tensor
                            )
                            pred_set = {
                                matched_masks[i]
                                for i in range(len(matched_masks))
                                if bool(pred_mask[i].item())
                            }
                        else:
                            pred_mask = torch.sigmoid(logits_tensor) > 0.5
                            pred_set = {
                                matched_masks[i]
                                for i in range(len(matched_masks))
                                if bool(pred_mask[i].item())
                            }
                        exact_flags.append(float(pred_set == target_set))
                        union = len(pred_set | target_set)
                        inter = len(pred_set & target_set)
                        jaccards.append(
                            float(inter / union) if union > 0 else 1.0
                        )

                if direct_losses:
                    loss = torch.stack(direct_losses).mean()
                else:
                    loss = torch.tensor(
                        0.0,
                        device=self.device,
                        requires_grad=not eval,
                    )
                zero = torch.zeros((), device=loss.device, dtype=loss.dtype)
                logs = {
                    "loss": loss,
                    "loss_regression": loss,
                    "loss_auxiliary": zero,
                    "velocity/probe_direct_set_exact_rate": torch.tensor(
                        float(sum(exact_flags) / len(exact_flags))
                        if exact_flags
                        else 0.0,
                        device=loss.device,
                        dtype=torch.float32,
                    ),
                    "velocity/probe_direct_set_mean_jaccard": torch.tensor(
                        float(sum(jaccards) / len(jaccards)) if jaccards else 0.0,
                        device=loss.device,
                        dtype=torch.float32,
                    ),
                    "velocity/probe_direct_set_target_negative_rate": torch.tensor(
                        float(sum(target_negative_rates) / len(target_negative_rates))
                        if target_negative_rates
                        else 0.0,
                        device=loss.device,
                        dtype=torch.float32,
                    ),
                    "velocity/probe_direct_set_target_negative_loss": torch.tensor(
                        float(
                            torch.stack(target_negative_losses).mean().item()
                        )
                        if target_negative_losses
                        else 0.0,
                        device=loss.device,
                        dtype=torch.float32,
                    ),
                    "velocity/probe_direct_set_nontarget_nonnegative_loss": torch.tensor(
                        float(
                            torch.stack(nontarget_nonnegative_losses).mean().item()
                        )
                        if nontarget_nonnegative_losses
                        else 0.0,
                        device=loss.device,
                        dtype=torch.float32,
                    ),
                    "velocity/probe_direct_set_pos_weight": torch.tensor(
                        float(sum(direct_set_pos_weights) / len(direct_set_pos_weights))
                        if direct_set_pos_weights
                        else 1.0,
                        device=loss.device,
                        dtype=torch.float32,
                    ),
                    "velocity/full_path_control_mode": torch.tensor(
                        1.0
                        if batch.get("_use_full_path_control_velocity_loss", False)
                        else 0.0,
                        device=loss.device,
                        dtype=torch.float32,
                    ),
                }
                if direct_set_debug_enabled:
                    logging.info(
                        "DIRECT_SET_DEBUG use_probe=%s pos_reweight=%s mean_pos_weight=%.6f samples=%d exact_rate=%.6f mean_jaccard=%.6f",
                        bool(batch.get("_use_probe_parity_direct_set_loss", False)),
                        bool(self.velocity_probe_direct_set_positive_reweight),
                        float(logs["velocity/probe_direct_set_pos_weight"].detach().item()),
                        int(len(direct_losses)),
                        float(logs["velocity/probe_direct_set_exact_rate"].detach().item()),
                        float(logs["velocity/probe_direct_set_mean_jaccard"].detach().item()),
                    )
                if direct_set_mse_weight <= 0.0:
                    return logs
                direct_set_loss = loss

            velocity_labels = batch["batched_velocity"]
            num_leaves = batch["num_leaves"]
            enc = BHVEncoder()
            gathered_velocity_labels = []
            v_pred_indices = []
            gathered_velocity_lengths = []
            gathered_boundary_vanish_targets = []

            for num in range(len(velocity_labels)):
                sub_gathered_velocity_labels = []
                sub_v_pred_indices = []
                sub_gathered_velocity_lengths = []
                sub_boundary_vanish_targets = []

                num_leave = int(num_leaves[num])
                split_masks_num = [int(m) for m in edge_split_masks[num]]
                split_masks_nonzero = [m for m in split_masks_num if m != 0]
                if len(split_masks_nonzero) == 0:
                    gathered_velocity_labels.append(
                        torch.tensor(sub_gathered_velocity_labels)
                    )
                    v_pred_indices.append(torch.tensor(sub_v_pred_indices))
                    gathered_velocity_lengths.append(
                        torch.tensor(sub_gathered_velocity_lengths)
                    )
                    continue

                real_max_bit = max(m.bit_length() for m in split_masks_nonzero)
                full_mask = (1 << real_max_bit) - 1 if real_max_bit > 0 else 0
                mask_to_idx = {m: i for i, m in enumerate(split_masks_num)}
                tree_obj = Tree(batch["original_trees"][num])
                tree_masks, tree_lengths = enc.return_BHV_encoding(tree_obj)
                length_map = {
                    int(m): float(l)
                    for m, l in zip(tree_masks, tree_lengths)
                    if l is not None
                }
                next_boundary_tree = batch.get("velocity_next_boundary_trees", [None])[
                    num
                ]
                next_boundary_active_masks = None
                if next_boundary_tree:
                    boundary_tree_obj = Tree(next_boundary_tree)
                    boundary_masks, boundary_lengths = enc.return_BHV_encoding(
                        boundary_tree_obj
                    )
                    next_boundary_active_masks = set()
                    for boundary_mask, boundary_length in zip(
                        boundary_masks, boundary_lengths
                    ):
                        if boundary_length is None or float(boundary_length) <= 1e-8:
                            continue
                        boundary_mask = int(boundary_mask)
                        if boundary_mask.bit_length() == real_max_bit + 1:
                            boundary_mask = remove_bit(boundary_mask, num_leave - 1)
                        elif boundary_mask.bit_length() > real_max_bit + 1:
                            continue

                        matched_boundary_mask = boundary_mask
                        if (
                            matched_boundary_mask not in mask_to_idx
                            and full_mask
                            and (full_mask ^ matched_boundary_mask) in mask_to_idx
                        ):
                            matched_boundary_mask = full_mask ^ matched_boundary_mask
                        next_boundary_active_masks.add(int(matched_boundary_mask))
                for vel in velocity_labels[num]:
                    original_vel = vel
                    if vel.bit_length() == real_max_bit + 1:
                        vel = remove_bit(vel, num_leave - 1)
                    elif vel.bit_length() > real_max_bit + 1:
                        raise Exception(
                            f"Whoa there is a big problem with this split mask {vel} vs real max {real_max_bit}!"
                        )

                    matched_vel = vel
                    if matched_vel not in mask_to_idx:
                        # Split orientation can flip after dummy-root removal; allow complement match.
                        complement_vel = full_mask ^ vel
                        if complement_vel in mask_to_idx:
                            matched_vel = complement_vel
                        else:
                            print(
                                f"This split {vel} from velocity labels is not in edge splits {split_masks_num}!"
                            )
                            print([i for i in range(vel.bit_length()) if (vel >> i) & 1])
                            raise Exception("Split not found in edge splits")
                    
                    #Ignore leaf edges
                    n_bits = real_max_bit
                    k = int(matched_vel).bit_count()
                    is_pendant = min(k, n_bits - k) == 1
                    if is_pendant:
                        continue

                    edge_len = length_map.get(int(matched_vel))
                    if edge_len is None and full_mask:
                        edge_len = length_map.get(full_mask ^ int(matched_vel))
                    if edge_len is None:
                        print(
                            f"Edge length not found for split {matched_vel} (original {original_vel}) in tree {batch['original_trees'][num]}"
                        )
                        print(f"Available splits: {split_masks_num}")
                        print(f"Length map keys: {list(length_map.keys())}")
                        raise Exception("Edge length not found for matched split")
                    if edge_len is None or float(edge_len) <= 1e-8:
                        continue

                    sub_gathered_velocity_labels.append(velocity_labels[num][original_vel])
                    sub_v_pred_indices.append(mask_to_idx[int(matched_vel)])
                    sub_gathered_velocity_lengths.append(float(edge_len))
                    if next_boundary_active_masks is not None:
                        sub_boundary_vanish_targets.append(
                            0.0 if int(matched_vel) in next_boundary_active_masks else 1.0
                        )


                gathered_velocity_labels.append(
                    torch.tensor(sub_gathered_velocity_labels)
                )
                v_pred_indices.append(torch.tensor(sub_v_pred_indices))
                gathered_velocity_lengths.append(torch.tensor(sub_gathered_velocity_lengths))
                gathered_boundary_vanish_targets.append(
                    torch.tensor(sub_boundary_vanish_targets, dtype=torch.float32)
                )

            # gathered_velocity_labels = torch.stack(gathered_velocity_labels)
            # v_pred_indices = torch.stack(v_pred_indices)

            # Fix: Flatten tensors to handle variable number of edges per tree
            preds_list = []
            first_hit_logits_list = []
            edge_features_list = []
            boundary_vanish_logits_list = []
            for b_idx in range(len(v_pred_indices)):
                indices = v_pred_indices[b_idx].to(v_pred.device)
                if indices.numel() > 0:
                    preds = v_pred[b_idx].index_select(0, indices)
                    preds_list.append(preds)
                    if first_hit_logits is not None:
                        first_hit_logits_list.append(
                            first_hit_logits[b_idx].index_select(0, indices)
                        )
                    if edge_features is not None:
                        edge_features_list.append(
                            edge_features[b_idx].index_select(0, indices)
                        )
                    if boundary_vanish_logits is not None:
                        boundary_vanish_logits_list.append(
                            boundary_vanish_logits[b_idx].index_select(0, indices)
                        )

            if len(preds_list) > 0:
                v_pred_gathered = torch.cat(preds_list).squeeze(-1)
                gathered_velocity_labels_flat = torch.cat(gathered_velocity_labels).to(
                    v_pred_gathered.device
                )
                gathered_velocity_lengths_flat = torch.cat(gathered_velocity_lengths).to(
                    v_pred_gathered.device
                )
                y = gathered_velocity_labels_flat
                p = v_pred_gathered
                lengths = gathered_velocity_lengths_flat
                first_hit_logits_gathered = (
                    torch.cat(first_hit_logits_list).squeeze(-1)
                    if first_hit_logits_list
                    else None
                )
                edge_features_gathered = (
                    torch.cat(edge_features_list, dim=0)
                    if edge_features_list
                    else None
                )
                velocity_group_sizes = [
                    int(indices.numel())
                    for indices in v_pred_indices
                    if int(indices.numel()) > 0
                ]
                if edge_features_gathered is not None:
                    p = self._refine_velocity_predictions(
                        p,
                        lengths=lengths,
                        edge_features=edge_features_gathered,
                        group_sizes=velocity_group_sizes,
                    )
                if first_hit_logits_gathered is not None or edge_features_gathered is not None:
                    first_hit_logits_gathered = self._compute_first_hit_logits(
                        first_hit_logits_gathered,
                        lengths=lengths,
                        velocities=p,
                        edge_features=edge_features_gathered,
                        group_sizes=velocity_group_sizes,
                    )
                boundary_vanish_logits_gathered = (
                    torch.cat(boundary_vanish_logits_list).squeeze(-1)
                    if boundary_vanish_logits_list
                    else None
                )
                boundary_vanish_targets_flat = (
                    torch.cat(gathered_boundary_vanish_targets).to(v_pred_gathered.device)
                    if (
                        boundary_vanish_logits_gathered is not None
                        and gathered_boundary_vanish_targets
                        and sum(int(t.numel()) for t in gathered_boundary_vanish_targets)
                        == int(boundary_vanish_logits_gathered.numel())
                    )
                    else None
                )

                # --- Velocity diagnostics ---
                with torch.no_grad():
                    vel_metrics = _velocity_diagnostics(
                        p,
                        y,
                        topk=3,
                        sign_eps=self.velocity_sign_eps,
                        lengths=gathered_velocity_lengths_flat,
                    )
                if self.verbose:
                    logger.info(
                        f"Velocity metrics: MSE={vel_metrics['mse']:.6f}  "
                        f"Cosine={vel_metrics['cosine']:.4f}  "
                        f"Pearson={vel_metrics['pearson']:.4f}  "
                        f"Spearman={vel_metrics['spearman']:.4f}  "
                        f"SignAcc={vel_metrics['sign_acc']:.4f}  "
                        f"TopK={vel_metrics['topk_overlap']:.4f}  "
                        f"dtTopK={vel_metrics['dt_topk_overlap']:.4f}  "
                        f"dtFirstHitRecall={vel_metrics['dt_first_hit_recall']:.4f}  "
                        f"dtHitRelErr={vel_metrics['dt_hit_rel_err']:.4f}  "
                        f"N={vel_metrics['n_edges']}"
                    )
                logs.update(
                    {
                        "velocity/cosine": torch.tensor(
                            vel_metrics["cosine"], device=v_pred.device
                        ),
                        "velocity/pearson": torch.tensor(
                            vel_metrics["pearson"], device=v_pred.device
                        ),
                        "velocity/spearman": torch.tensor(
                            vel_metrics["spearman"], device=v_pred.device
                        ),
                        "velocity/sign_acc": torch.tensor(
                            vel_metrics["sign_acc"], device=v_pred.device
                        ),
                        "velocity/topk_overlap": torch.tensor(
                            vel_metrics["topk_overlap"], device=v_pred.device
                        ),
                        "velocity/dt_topk_overlap": torch.tensor(
                            vel_metrics["dt_topk_overlap"], device=v_pred.device
                        ),
                        "velocity/dt_first_hit_recall": torch.tensor(
                            vel_metrics["dt_first_hit_recall"], device=v_pred.device
                        ),
                        "velocity/dt_first_hit_precision": torch.tensor(
                            vel_metrics["dt_first_hit_precision"], device=v_pred.device
                        ),
                        "velocity/dt_neg_jaccard": torch.tensor(
                            vel_metrics["dt_neg_jaccard"], device=v_pred.device
                        ),
                        "velocity/length_jitter_attempted": torch.tensor(
                            velocity_perturb_stats["attempted"],
                            device=v_pred.device,
                        ),
                        "velocity/length_jitter_applied": torch.tensor(
                            velocity_perturb_stats["applied"],
                            device=v_pred.device,
                        ),
                    }
                )

                # eps = 1e-6
                # first_hit_tol = 0.01
                # contract = (y < -self.velocity_sign_eps) & (lengths > 1e-8)

                # Lc = lengths[contract].clamp_min(eps)
                # yc = y[contract]
                # pc = p[contract]

                # tau_true = Lc / (-yc).clamp_min(eps)
                # w = (tau_true.median().clamp_min(eps) / tau_true).clamp(max=20.0)

                # tau_min = tau_true.min()
                # first = (tau_true - tau_min).abs() <= 0.01  # true tol (could be 0)

                # boost = 5.0
                # w = w * (1.0 + boost * first.float())

                # loss = (w * (pc - yc)**2).mean()
                # import pdb; pdb.set_trace()

                # tau_true = lengths[contract] / (-y[contract]).clamp_min(eps)
                # tau_min = tau_true.min()
                # first = torch.abs(tau_true - tau_min) <= first_hit_tol
                # w = torch.ones_like(y[contract])
                # alpha = 10
                # w[first] = 1.0 + alpha
                # loss = (w * (p[contract] - y[contract])**2).mean()

                # ####OG LOSS HERE
                residual_sq = (p - y).pow(2)
                plain_mse = residual_sq.mean()

                abs_y = y.abs()
                eps = 1e-6
                scale = abs_y.median().clamp_min(eps)  # robust scale
                w = (abs_y / scale).clamp(min=0.0, max=20.0)
                weighted_mse = (w * residual_sq).sum() / w.sum().clamp_min(eps)
                if self.velocity_loss_mode == "plain":
                    loss = plain_mse
                elif self.velocity_loss_mode == "weighted":
                    loss = weighted_mse
                else:
                    loss = (
                        self.velocity_loss_plain_weight * plain_mse
                        + (1.0 - self.velocity_loss_plain_weight) * weighted_mse
                    )
                regression_loss = loss
                auxiliary_loss = p.new_tensor(0.0)

                # ------------------------------------------------------------------
                # First-hit structured loss on true contracting edges:
                #   1) keep the fastest true contracting edges accurate in rate space
                #   2) ensure true first-hit edges remain earlier than later edges
                #   3) keep the tied first-hit set collapsed together
                # ------------------------------------------------------------------
                # contract_mask = (y < -self.velocity_sign_eps) & (lengths > 1e-8)
                # fast_rate_loss = p.new_tensor(0.0)
                # first_hit_dt_loss = p.new_tensor(0.0)

                # if int(contract_mask.sum()) > 0:
                #     Lc = lengths[contract_mask].clamp_min(eps)
                #     yc = y[contract_mask]
                #     pc = p[contract_mask]

                #     # First-hit ordering is governed by contraction rate (-v / length).
                #     rate_true = (-yc).clamp_min(eps) / Lc
                #     rate_pred = F.softplus(
                #         -pc, beta=self.velocity_event_rate_beta
                #     ) / Lc

                #     # True collapse times for truly contracting edges
                #     tau_true = 1.0 / rate_true.clamp_min(eps)
                #     tau_pred = 1.0 / rate_pred.clamp_min(eps)

                #     # Identify the true first-hit set with tolerance
                #     tau_true_min = tau_true.min()
                #     first_mask = torch.abs(tau_true - tau_true_min) <= 0.01
                #     later_mask = ~first_mask
                #     fast_k = min(8, int(tau_true.numel()))
                #     fast_idx = torch.argsort(tau_true)[:fast_k]
                #     fast_mask = torch.zeros_like(first_mask)
                #     fast_mask[fast_idx] = True
                #     rate_scale = rate_true[fast_mask].median().clamp_min(1.0)
                #     fast_rate_loss = F.smooth_l1_loss(
                #         rate_pred[fast_mask] / rate_scale,
                #         rate_true[fast_mask] / rate_scale,
                #     )

                #     z_pred = torch.log(tau_pred.clamp_min(eps))

                #     # 1) Tie loss: first-hit edges should have similar predicted dt
                #     if int(first_mask.sum()) > 1:
                #         z_first = z_pred[first_mask]
                #         first_hit_tie_loss = ((z_first - z_first.mean()) ** 2).mean()
                #     else:
                #         first_hit_tie_loss = p.new_tensor(0.0)
                #     if int(first_mask.sum()) > 0:
                #         first_hit_dt_loss = F.smooth_l1_loss(
                #             z_pred[first_mask],
                #             torch.log(tau_true[first_mask].clamp_min(eps)),
                #         )

                #     # 2) Rank loss: first-hit edges should be earlier than later contracting edges
                #     if int(first_mask.sum()) > 0 and int(later_mask.sum()) > 0:
                #         z_first = z_pred[first_mask][:, None]   # shape [F, 1]
                #         z_later = z_pred[later_mask][None, :]   # shape [1, L]
                #         first_hit_rank_loss = F.relu(
                #             z_later - z_first + 0.02
                #         ).mean()
                #     else:
                #         first_hit_rank_loss = p.new_tensor(0.0)

                #     first_hit_loss = (
                #         first_hit_tie_loss
                #         + first_hit_rank_loss
                #     )

                #     n_contract = int(contract_mask.sum())
                #     n_first = int(first_mask.sum())
                #     n_later = int(later_mask.sum())
                # else:
                #     first_hit_tie_loss = p.new_tensor(0.0)
                #     first_hit_rank_loss = p.new_tensor(0.0)
                #     first_hit_loss = p.new_tensor(0.0)
                #     n_contract = 0
                #     n_first = 0
                #     n_later = 0

                # first_hit_aux_weight = float(
                #     min(max((self.current_step_value - 100) / 200.0, 0.0), 1.0)
                # )

                first_hit_velocity_loss = p.new_tensor(0.0)
                logtau_all_loss_raw = p.new_tensor(0.0)
                logtau_all_loss = p.new_tensor(0.0)
                logtau_first_over_loss_raw = p.new_tensor(0.0)
                logtau_first_over_loss = p.new_tensor(0.0)
                logtau_first_tie_loss_raw = p.new_tensor(0.0)
                logtau_first_tie_loss = p.new_tensor(0.0)
                logtau_predset_over_loss_raw = p.new_tensor(0.0)
                logtau_predset_over_loss = p.new_tensor(0.0)
                event_loss_raw = p.new_tensor(0.0)
                event_loss = p.new_tensor(0.0)
                event_precision_loss_raw = p.new_tensor(0.0)
                event_precision_loss = p.new_tensor(0.0)
                first_hit_head_loss_raw = p.new_tensor(0.0)
                first_hit_head_loss = p.new_tensor(0.0)
                boundary_vanish_head_loss_raw = p.new_tensor(0.0)
                boundary_vanish_head_loss = p.new_tensor(0.0)
                boundary_time_head_loss_raw = p.new_tensor(0.0)
                boundary_time_head_loss = p.new_tensor(0.0)
                event_stats = {
                    "n_candidates": 0,
                    "target_first_size": 0,
                    "pred_first_mass": 0.0,
                    "top1_hits_first_set": 0.0,
                }
                event_precision_stats = {
                    "margin_gap": 0.0,
                    "n_pos": 0,
                    "n_neg": 0,
                    "violated": 0.0,
                }
                first_hit_head_stats = {
                    "n_candidates": 0,
                    "target_first_size": 0,
                    "pred_first_size": 0,
                    "top1_hits_first_set": 0.0,
                    "recall": 0.0,
                    "precision": 0.0,
                    "jaccard": 0.0,
                }
                boundary_vanish_head_stats = {
                    "n_candidates": 0,
                    "target_size": 0,
                    "pred_size": 0,
                    "top1_hits_target_set": 0.0,
                    "recall": 0.0,
                    "precision": 0.0,
                    "jaccard": 0.0,
                }
                boundary_time_head_stats = {
                    "n_groups": 0,
                    "n_valid": 0,
                    "dt_pred_mean": 0.0,
                    "dt_true_mean": 0.0,
                    "dt_rel_err_mean": 0.0,
                }
                first_hit_extra_penalty_raw = p.new_tensor(0.0)
                first_hit_fp_mass = p.new_tensor(0.0)
                first_hit_fn_mass = p.new_tensor(0.0)
                use_full_path_control_velocity_loss = bool(
                    batch.get("_use_full_path_control_velocity_loss", False)
                )
                if use_full_path_control_velocity_loss:
                    control_loss, control_parts = _full_path_control_velocity_loss(
                        p=p,
                        y=y,
                        lengths=lengths,
                        first_hit_logits=first_hit_logits_gathered,
                        group_sizes=velocity_group_sizes,
                        velocity_sign_eps=self.velocity_sign_eps,
                        dt_eps=self.velocity_dt_eps,
                        first_hit_tol=self.velocity_first_hit_loss_tol,
                        first_hit_head_weight=self.velocity_first_hit_head_weight,
                        logtau_all_weight=self.velocity_logtau_all_weight,
                        logtau_first_over_weight=self.velocity_logtau_first_over_weight,
                        logtau_first_tie_weight=0.0,
                        logtau_predset_over_weight=self.velocity_logtau_predset_over_weight,
                    )
                    loss = control_loss
                    regression_loss = control_parts["mse"]
                    auxiliary_loss = loss - regression_loss
                    logtau_all_loss_raw = control_parts["logtau_all_raw"]
                    logtau_all_loss = control_parts["logtau_all"]
                    logtau_first_over_loss_raw = control_parts["logtau_first_over_raw"]
                    logtau_first_over_loss = control_parts["logtau_first_over"]
                    logtau_first_tie_loss_raw = control_parts["logtau_first_tie_raw"]
                    logtau_first_tie_loss = control_parts["logtau_first_tie"]
                    logtau_predset_over_loss_raw = control_parts[
                        "logtau_predset_over_raw"
                    ]
                    logtau_predset_over_loss = control_parts["logtau_predset_over"]
                    first_hit_head_loss_raw = control_parts["firsthit_bce_raw"]
                    first_hit_head_loss = control_parts["firsthit_bce"]
                    logs["velocity/full_path_control_mode"] = torch.tensor(
                        1.0, dtype=torch.float32, device=v_pred.device
                    )
                contract_mask = (y < -self.velocity_sign_eps) & (lengths > 1e-8)
                if (
                    not use_full_path_control_velocity_loss
                    and (
                    self.velocity_logtau_all_weight > 0.0
                    and int(contract_mask.sum()) > 0
                    )
                ):
                    dt_eps = float(self.velocity_dt_eps)
                    Lc = lengths[contract_mask].clamp_min(dt_eps)
                    yc = y[contract_mask]
                    pc = p[contract_mask]
                    tau_true = Lc / (-yc).clamp_min(dt_eps)
                    tau_pred = Lc / (-pc).clamp_min(dt_eps)
                    logtau_all_loss_raw = F.smooth_l1_loss(
                        torch.log(tau_pred.clamp_min(dt_eps)),
                        torch.log(tau_true.clamp_min(dt_eps)),
                    )
                    logtau_all_loss = (
                        self.velocity_logtau_all_weight * logtau_all_loss_raw
                    )
                    loss = loss + logtau_all_loss
                    auxiliary_loss = auxiliary_loss + logtau_all_loss
                if (
                    not use_full_path_control_velocity_loss
                    and (
                    (
                        self.velocity_logtau_first_over_weight > 0.0
                        or 0.0 > 0.0
                    )
                    and int(contract_mask.sum()) > 0
                    and velocity_group_sizes
                    )
                ):
                    dt_eps = float(self.velocity_dt_eps)
                    over_losses = []
                    tie_losses = []
                    start_idx = 0
                    for group_size in velocity_group_sizes:
                        end_idx = start_idx + int(group_size)
                        Lg = lengths[start_idx:end_idx]
                        yg = y[start_idx:end_idx]
                        pg = p[start_idx:end_idx]
                        start_idx = end_idx
                        group_contract = (yg < -self.velocity_sign_eps) & (Lg > 1e-8)
                        if not bool(group_contract.any().item()):
                            continue
                        tau_true = Lg[group_contract].clamp_min(dt_eps) / (
                            -yg[group_contract]
                        ).clamp_min(dt_eps)
                        tau_min = tau_true.min()
                        group_contract_idx = torch.nonzero(
                            group_contract, as_tuple=False
                        ).reshape(-1)
                        first_idx = group_contract_idx[
                            torch.abs(tau_true - tau_min)
                            <= float(self.velocity_first_hit_loss_tol)
                        ]
                        if int(first_idx.numel()) == 0:
                            continue
                        tau_true_first = Lg[first_idx].clamp_min(dt_eps) / (
                            -yg[first_idx]
                        ).clamp_min(dt_eps)
                        tau_pred_first = Lg[first_idx].clamp_min(dt_eps) / (
                            -pg[first_idx]
                        ).clamp_min(dt_eps)
                        pred_max = torch.log(tau_pred_first.clamp_min(dt_eps)).max()
                        true_max = torch.log(tau_true_first.clamp_min(dt_eps)).max()
                        over_losses.append(F.relu(pred_max - true_max).pow(2))
                        if int(tau_pred_first.numel()) > 1:
                            log_pred_first = torch.log(
                                tau_pred_first.clamp_min(dt_eps)
                            )
                            tie_losses.append(
                                (
                                    log_pred_first - log_pred_first.mean()
                                ).pow(2).mean()
                            )
                    if over_losses and self.velocity_logtau_first_over_weight > 0.0:
                        logtau_first_over_loss_raw = torch.stack(over_losses).mean()
                        logtau_first_over_loss = (
                            self.velocity_logtau_first_over_weight
                            * logtau_first_over_loss_raw
                        )
                        loss = loss + logtau_first_over_loss
                        auxiliary_loss = auxiliary_loss + logtau_first_over_loss
                    if tie_losses and 0.0 > 0.0:
                        logtau_first_tie_loss_raw = torch.stack(tie_losses).mean()
                        logtau_first_tie_loss = (
                            0.0
                            * logtau_first_tie_loss_raw
                        )
                        loss = loss + logtau_first_tie_loss
                        auxiliary_loss = auxiliary_loss + logtau_first_tie_loss
                if (
                    not use_full_path_control_velocity_loss
                    and self.velocity_dt_hit_weight > 0.0
                    and int(contract_mask.sum()) > 0
                ):
                    Lc = lengths[contract_mask].clamp_min(eps)
                    yc = y[contract_mask]
                    pc = p[contract_mask]
                    tau_true = Lc / (-yc).clamp_min(eps)
                    tau_true_min = tau_true.min()
                    first_mask = (
                        torch.abs(tau_true - tau_true_min)
                        <= float(self.velocity_first_hit_loss_tol)
                    )
                    if int(first_mask.sum()) > 0:
                        first_hit_velocity_loss = F.smooth_l1_loss(
                            pc[first_mask], yc[first_mask]
                        )

                # first_hit_aux_weight = float(
                #     min(max((self.current_step_value - 100) / 200.0, 0.0), 1.0)
                # )

                # loss = (
                #     loss
                #     + 1
                #     * (0.5 * self.velocity_dt_hit_weight * first_hit_velocity_loss)
                # )

                # loss = loss

                # loss = (
                #     loss
                #     + self.velocity_dt_hit_weight * first_hit_velocity_loss
                # )

                first_hit_velocity_aux = (
                    self.velocity_dt_hit_weight * first_hit_velocity_loss
                )
                if not use_full_path_control_velocity_loss:
                    loss = loss + first_hit_velocity_aux
                    auxiliary_loss = auxiliary_loss + first_hit_velocity_aux
                if (
                    not use_full_path_control_velocity_loss
                    and self.velocity_event_weight > 0.0
                ):
                    event_loss_raw, event_stats = _boundary_event_distribution_loss(
                        lengths=lengths,
                        y_true=y,
                        y_pred=p,
                        velocity_sign_eps=self.velocity_sign_eps,
                        dt_eps=self.velocity_dt_eps,
                        temp=self.velocity_event_temp,
                        rate_beta=self.velocity_event_rate_beta,
                        normalize_by_log_candidates=self.velocity_event_normalize_by_log_candidates,
                    )
                    event_loss = self.velocity_event_weight * event_loss_raw
                    loss = loss + event_loss
                    auxiliary_loss = auxiliary_loss + event_loss
                if (
                    not use_full_path_control_velocity_loss
                    and self.velocity_event_precision_weight > 0.0
                ):
                    (
                        event_precision_loss_raw,
                        event_precision_stats,
                    ) = _boundary_event_precision_margin_loss(
                        lengths=lengths,
                        y_true=y,
                        y_pred=p,
                        velocity_sign_eps=self.velocity_sign_eps,
                        dt_eps=self.velocity_dt_eps,
                        temp=self.velocity_event_temp,
                        rate_beta=self.velocity_event_rate_beta,
                        margin=self.velocity_event_precision_margin,
                    )
                    event_precision_loss = (
                        self.velocity_event_precision_weight
                        * event_precision_loss_raw
                    )
                    loss = loss + event_precision_loss
                    auxiliary_loss = auxiliary_loss + event_precision_loss
                if (
                    not use_full_path_control_velocity_loss
                    and (
                    self.velocity_first_hit_head_weight > 0.0
                    and first_hit_logits_gathered is not None
                    )
                ):
                    (
                        first_hit_head_loss_raw,
                        first_hit_head_stats,
                    ) = _first_hit_grouped_set_bce_loss(
                        lengths=lengths,
                        y_true=y,
                        first_hit_logits=first_hit_logits_gathered,
                        group_sizes=velocity_group_sizes,
                        velocity_sign_eps=self.velocity_sign_eps,
                        dt_eps=self.velocity_dt_eps,
                        first_hit_tol=self.velocity_first_hit_loss_tol,
                    )
                    first_hit_extra_penalty_raw = p.new_tensor(0.0)
                    first_hit_fp_mass = p.new_tensor(0.0)
                    first_hit_fn_mass = p.new_tensor(0.0)
                    if (
                        self.velocity_first_hit_false_positive_mass_weight > 0.0
                        or 0.0 > 0.0
                    ):
                        first_hit_mass_penalty_raw, first_hit_mass_stats = (
                            _first_hit_grouped_soft_mass_penalty(
                                lengths=lengths,
                                y_true=y,
                                first_hit_logits=first_hit_logits_gathered,
                                group_sizes=velocity_group_sizes,
                                velocity_sign_eps=self.velocity_sign_eps,
                                dt_eps=self.velocity_dt_eps,
                                first_hit_tol=self.velocity_first_hit_loss_tol,
                            )
                        )
                        first_hit_fp_mass = first_hit_mass_stats.get(
                            "fp_mass_tensor", p.new_tensor(0.0)
                        )
                        first_hit_fn_mass = first_hit_mass_stats.get(
                            "fn_mass_tensor", p.new_tensor(0.0)
                        )
                        first_hit_extra_penalty_raw = (
                            self.velocity_first_hit_false_positive_mass_weight
                            * first_hit_fp_mass
                            + 0.0
                            * first_hit_fn_mass
                        )
                    first_hit_head_loss = (
                        self.velocity_first_hit_head_weight
                        * (first_hit_head_loss_raw + first_hit_extra_penalty_raw)
                    )
                    loss = loss + first_hit_head_loss
                    auxiliary_loss = auxiliary_loss + first_hit_head_loss
                    logs["velocity/first_hit_fp_mass_raw"] = first_hit_fp_mass
                    logs["velocity/first_hit_fn_mass_raw"] = first_hit_fn_mass
                    logs["velocity/first_hit_extra_penalty_raw"] = (
                        first_hit_extra_penalty_raw
                    )
                # loss = (
                #     loss
                #     + first_hit_aux_weight
                #     * (
                #         0.02 * fast_rate_loss
                #         + 0.05 * first_hit_dt_loss
                #         + 0.02 * first_hit_loss
                #     )
                # )
    
            else:
                loss = torch.tensor(0.0, device=v_pred.device, requires_grad=True)
                regression_loss = loss.detach() * 0.0
                auxiliary_loss = loss.detach() * 0.0
                plain_mse = loss.detach() * 0.0
                weighted_mse = loss.detach() * 0.0
                first_hit_velocity_loss = loss.detach() * 0.0
                logtau_all_loss_raw = loss.detach() * 0.0
                logtau_all_loss = loss.detach() * 0.0
                logtau_first_over_loss_raw = loss.detach() * 0.0
                logtau_first_over_loss = loss.detach() * 0.0
                logtau_first_tie_loss_raw = loss.detach() * 0.0
                logtau_first_tie_loss = loss.detach() * 0.0
                logtau_predset_over_loss_raw = loss.detach() * 0.0
                logtau_predset_over_loss = loss.detach() * 0.0
                event_loss_raw = loss.detach() * 0.0
                event_loss = loss.detach() * 0.0
                event_precision_loss_raw = loss.detach() * 0.0
                event_precision_loss = loss.detach() * 0.0
                first_hit_head_loss_raw = loss.detach() * 0.0
                first_hit_head_loss = loss.detach() * 0.0
                boundary_vanish_head_loss_raw = loss.detach() * 0.0
                boundary_vanish_head_loss = loss.detach() * 0.0
                boundary_time_head_loss_raw = loss.detach() * 0.0
                boundary_time_head_loss = loss.detach() * 0.0
                event_stats = {
                    "n_candidates": 0,
                    "target_first_size": 0,
                    "pred_first_mass": 0.0,
                    "top1_hits_first_set": 0.0,
                }
                event_precision_stats = {
                    "margin_gap": 0.0,
                    "n_pos": 0,
                    "n_neg": 0,
                    "violated": 0.0,
                }
                first_hit_head_stats = {
                    "n_candidates": 0,
                    "target_first_size": 0,
                    "pred_first_size": 0,
                    "top1_hits_first_set": 0.0,
                    "recall": 0.0,
                    "precision": 0.0,
                    "jaccard": 0.0,
                }
                boundary_vanish_head_stats = {
                    "n_candidates": 0,
                    "target_size": 0,
                    "pred_size": 0,
                    "top1_hits_target_set": 0.0,
                    "recall": 0.0,
                    "precision": 0.0,
                    "jaccard": 0.0,
                }
                boundary_time_head_stats = {
                    "n_groups": 0,
                    "n_valid": 0,
                    "dt_pred_mean": 0.0,
                    "dt_true_mean": 0.0,
                    "dt_rel_err_mean": 0.0,
                }
                n_contract = 0
            mse_branch_loss = loss
            mse_branch_regression_loss = regression_loss
            mse_branch_auxiliary_loss = auxiliary_loss
            if direct_set_loss is not None:
                loss = (
                    direct_set_loss_weight * direct_set_loss
                    + direct_set_mse_weight * mse_branch_loss
                )
                regression_loss = direct_set_mse_weight * mse_branch_regression_loss
                auxiliary_loss = loss - regression_loss
            # print("Wow congrats")
            logs.update(
                {
                    "velocity/loss_plain_mse": plain_mse.detach(),
                    "velocity/loss_weighted_mse": weighted_mse.detach(),
                    "velocity/mse_branch_loss_unscaled": mse_branch_loss.detach(),
                    "velocity/mse_branch_regression_unscaled": mse_branch_regression_loss.detach(),
                    "velocity/mse_branch_auxiliary_unscaled": mse_branch_auxiliary_loss.detach(),
                    "velocity/loss_regression_unscaled": regression_loss.detach(),
                    "velocity/loss_auxiliary_unscaled": auxiliary_loss.detach(),
                    "velocity/probe_direct_set_loss": (
                        torch.zeros((), device=v_pred.device, dtype=v_pred.dtype)
                        if direct_set_loss is None
                        else direct_set_loss.detach()
                    ),
                    "velocity/probe_direct_set_loss_weight": torch.tensor(
                        direct_set_loss_weight, device=v_pred.device
                    ),
                    "velocity/probe_direct_set_mse_weight": torch.tensor(
                        direct_set_mse_weight, device=v_pred.device
                    ),
                    "velocity/first_hit_velocity_loss": first_hit_velocity_loss.detach(),
                    "velocity/logtau_all_loss_raw": logtau_all_loss_raw.detach(),
                    "velocity/logtau_all_loss": logtau_all_loss.detach(),
                    "velocity/logtau_first_over_loss_raw": logtau_first_over_loss_raw.detach(),
                    "velocity/logtau_first_over_loss": logtau_first_over_loss.detach(),
                    "velocity/logtau_first_tie_loss_raw": logtau_first_tie_loss_raw.detach(),
                    "velocity/logtau_first_tie_loss": logtau_first_tie_loss.detach(),
                    "velocity/logtau_predset_over_loss_raw": logtau_predset_over_loss_raw.detach(),
                    "velocity/logtau_predset_over_loss": logtau_predset_over_loss.detach(),
                    "velocity/event_loss_raw": event_loss_raw.detach(),
                    "velocity/event_loss": event_loss.detach(),
                    "velocity/event_precision_loss_raw": event_precision_loss_raw.detach(),
                    "velocity/event_precision_loss": event_precision_loss.detach(),
                    "velocity/first_hit_head_loss_raw": first_hit_head_loss_raw.detach(),
                    "velocity/first_hit_head_loss": first_hit_head_loss.detach(),
                }
            )
            logs["loss_regression"] = regression_loss
            logs["loss_auxiliary"] = auxiliary_loss
            logs["loss"] = loss
            # if len(preds_list) > 0:
            #     logger.info(
            #         f"Velocity loss ({self.velocity_loss_mode}): total={loss.item():.6f} "
            #         f"plain={plain_mse.item():.6f} weighted={weighted_mse.item():.6f} "
            #         # f"dt_gate={dt_gate.item():.4f} dt_candidates={dt_candidates_loss.item():.6f} "
            #         # f"dt_hit={dt_hit_loss.item():.6f}"
            #     )
            # else:

            if self.record and not is_replay_batch:
                dt_hit_pred_log = (
                    vel_metrics["dt_hit_pred"]
                    if np.isfinite(vel_metrics["dt_hit_pred"])
                    else -1.0
                )
                dt_hit_true_log = (
                    vel_metrics["dt_hit_true"]
                    if np.isfinite(vel_metrics["dt_hit_true"])
                    else -1.0
                )
                dt_hit_abs_err_log = (
                    vel_metrics["dt_hit_abs_err"]
                    if np.isfinite(vel_metrics["dt_hit_abs_err"])
                    else 1e6
                )
                dt_hit_rel_err_log = (
                    vel_metrics["dt_hit_rel_err"]
                    if np.isfinite(vel_metrics["dt_hit_rel_err"])
                    else 1e6
                )
                vel_wandb = {"train/velocity_loss": loss.item()}
                if len(preds_list) > 0:
                    vel_wandb.update({
                        "velocity/loss_plain_mse": plain_mse.item(),
                        "velocity/loss_weighted_mse": weighted_mse.item(),
                        "velocity/mse": vel_metrics["mse"],
                        "velocity/mse_vs_zero": vel_metrics["mse_vs_zero"],
                        "velocity/mse_vs_mean": vel_metrics["mse_vs_mean"],
                        "velocity/zero_baseline_mse": vel_metrics["zero_baseline_mse"],
                        "velocity/mean_baseline_mse": vel_metrics["mean_baseline_mse"],
                        "velocity/cosine": vel_metrics["cosine"],
                        "velocity/pearson": vel_metrics["pearson"],
                        "velocity/spearman": vel_metrics["spearman"],
                        "velocity/sign_acc": vel_metrics["sign_acc"],
                        "velocity/topk_overlap": vel_metrics["topk_overlap"],
                        "velocity/dt_hit_pred": dt_hit_pred_log,
                        "velocity/dt_hit_true": dt_hit_true_log,
                        "velocity/dt_hit_abs_err": dt_hit_abs_err_log,
                        "velocity/dt_hit_rel_err": dt_hit_rel_err_log,
                        "velocity/dt_first_hit_match": vel_metrics["dt_first_hit_match"],
                        "velocity/dt_first_hit_recall": vel_metrics["dt_first_hit_recall"],
                        "velocity/dt_first_hit_precision": vel_metrics["dt_first_hit_precision"],
                        "velocity/dt_topk_overlap": vel_metrics["dt_topk_overlap"],
                        "velocity/event_loss_raw": float(event_loss_raw.detach().item()),
                        "velocity/event_loss": float(event_loss.detach().item()),
                        "velocity/event_n_candidates": float(event_stats["n_candidates"]),
                        "velocity/event_target_first_size": float(event_stats["target_first_size"]),
                        "velocity/event_pred_first_mass": float(event_stats["pred_first_mass"]),
                        "velocity/event_top1_hits_first_set": float(event_stats["top1_hits_first_set"]),
                        "velocity/event_precision_loss_raw": float(event_precision_loss_raw.detach().item()),
                        "velocity/event_precision_loss": float(event_precision_loss.detach().item()),
                        "velocity/event_precision_margin_gap": float(event_precision_stats["margin_gap"]),
                        "velocity/event_precision_n_pos": float(event_precision_stats["n_pos"]),
                        "velocity/event_precision_n_neg": float(event_precision_stats["n_neg"]),
                        "velocity/event_precision_violated": float(event_precision_stats["violated"]),
                        "velocity/logtau_all_loss_raw": float(logtau_all_loss_raw.detach().item()),
                        "velocity/logtau_all_loss": float(logtau_all_loss.detach().item()),
                        "velocity/logtau_first_over_loss_raw": float(logtau_first_over_loss_raw.detach().item()),
                        "velocity/logtau_first_over_loss": float(logtau_first_over_loss.detach().item()),
                        "velocity/logtau_first_tie_loss_raw": float(logtau_first_tie_loss_raw.detach().item()),
                        "velocity/logtau_first_tie_loss": float(logtau_first_tie_loss.detach().item()),
                        "velocity/logtau_predset_over_loss_raw": float(logtau_predset_over_loss_raw.detach().item()),
                        "velocity/logtau_predset_over_loss": float(logtau_predset_over_loss.detach().item()),
                        "velocity/first_hit_head_loss_raw": float(first_hit_head_loss_raw.detach().item()),
                        "velocity/first_hit_head_loss": float(first_hit_head_loss.detach().item()),
                        "velocity/first_hit_head_target_size": float(first_hit_head_stats["target_first_size"]),
                        "velocity/first_hit_head_pred_size": float(first_hit_head_stats["pred_first_size"]),
                        "velocity/first_hit_head_top1_hits": float(first_hit_head_stats["top1_hits_first_set"]),
                        "velocity/first_hit_head_recall": float(first_hit_head_stats["recall"]),
                        "velocity/first_hit_head_precision": float(first_hit_head_stats["precision"]),
                        "velocity/first_hit_head_jaccard": float(first_hit_head_stats["jaccard"]),
                        "velocity/first_hit_fp_mass_raw": float(first_hit_fp_mass.detach().item()),
                        "velocity/first_hit_fn_mass_raw": float(first_hit_fn_mass.detach().item()),
                        "velocity/first_hit_extra_penalty_raw": float(first_hit_extra_penalty_raw.detach().item()),
                        "velocity/length_jitter_attempted": velocity_perturb_stats["attempted"],
                        "velocity/length_jitter_applied": velocity_perturb_stats["applied"],
                    })
                self._wandb_log_filtered(vel_wandb, step=self.stepper)
            # import pdb

            # pdb.set_trace()
        else:
            batch, ar_prep_stats = self._prepare_autoregressive_training_batch(batch)
            skip_autoregressive_merge_metrics = bool(
                batch.get("_skip_autoregressive_merge_metrics", False)
            )
            cached_component_groups = batch.get(
                "_cached_autoregressive_component_groups"
            )
            if cached_component_groups is not None:
                autoregressive_component_groups = cached_component_groups
            elif "newick_autoregressive_trees" in batch:
                autoregressive_component_groups = [
                    get_structural_polytomy_groups_from_newick(newick_tree)
                    for newick_tree in batch["newick_autoregressive_trees"]
                ]
            else:
                autoregressive_component_groups = []
                for labeled_merge_cluster in batch["batched_autoregressive_labels"]:
                    seen_groups = set()
                    groups = []
                    for label in labeled_merge_cluster:
                        components = tuple(int(component) for component in label["components"])
                        if components in seen_groups:
                            continue
                        seen_groups.add(components)
                        groups.append(list(components))
                    autoregressive_component_groups.append(groups)

            autoregressive_times = self._effective_autoregressive_time_tensor(
                batch["batched_autoregressive_time"]
            )
            all_group_logits = self.forward(
                batch["tokenized_autoregressive_trees"],
                autoregressive_times,
                batch["phyla_embeddings"],
                autoregressive=True,
                autoregressive_component_groups=autoregressive_component_groups,
                autoregressive_case_indices=batch.get("_autoregressive_case_indices"),
                autoregressive_start_topology_features=batch.get(
                    "_autoregressive_start_topology_features"
                ),
            )

            found = {}
            label_targets_by_batch = []
            for batch_index, labeled_merge_cluster in enumerate(batch["batched_autoregressive_labels"]):
                group_targets = {}
                for label in labeled_merge_cluster:
                    result_split = int(label["result_split"])
                    components = tuple(int(component) for component in label["components"])
                    merge_indices = [int(idx) for idx in label["merge_indices"]]
                    found[(batch_index, result_split)] = False
                    group_targets.setdefault(components, []).append(
                        (result_split, merge_indices)
                    )
                label_targets_by_batch.append(group_targets)

            losses = []

            total_metrics = []
            alternative_target_counts = []
            stop_after_merge_losses = []
            stop_after_merge_accuracies = []
            stop_after_merge_targets = []
            stop_after_merge_predictions = []
            subset_size_losses = []
            subset_size_accuracies = []
            subset_size_target_means = []
            subset_size_prediction_means = []

            chosen_polytomies = []
            polytomy_logits = []
            polytomy_sizes = []  # Track size of each polytomy encountered

            for group in all_group_logits:
                logits = group["logits"]
                splits_in_polytomy = tuple(int(split) for split in group["splits_represented"])
                batch_index = int(group["batch_index"])
                decoder_mode = str(group.get("decoder_mode", "pairwise_threshold"))
                
                # Track polytomy size (number of splits in the polytomy)
                polytomy_sizes.append(len(splits_in_polytomy))

                explicit_subsets = []
                for resulting_split, idxs in label_targets_by_batch[batch_index].get(
                    splits_in_polytomy,
                    [],
                ):
                    found[(batch_index, resulting_split)] = True
                    explicit_subsets.append(
                        tuple(sorted(int(splits_in_polytomy[i]) for i in idxs))
                    )
                explicit_subsets = list(dict.fromkeys(explicit_subsets))

                candidate_subsets = list(dict.fromkeys(explicit_subsets))
                if (
                    self.autoregressive_target_mode == "ready_alternatives"
                    and "target_trees" in batch
                ):
                    ready_subsets = _ready_target_merge_subsets_for_group(
                        splits_in_polytomy,
                        batch["target_trees"][batch_index],
                        Tree(batch["newick_autoregressive_trees"][batch_index]).n_leaves,
                    )
                    for subset in ready_subsets:
                        subset = tuple(sorted(int(split) for split in subset))
                        if subset not in candidate_subsets:
                            candidate_subsets.append(subset)

                alternative_target_counts.append(float(len(candidate_subsets)))

                if not candidate_subsets:
                    chosen_polytomies.append(torch.tensor(0.0))
                else:
                    chosen_polytomies.append(torch.tensor(1.0))

                polytomy_logits.append(group["polytomy_pred"])

                size_info = None
                if decoder_mode == "structured_subset" and len(explicit_subsets) <= 1:
                    size_targets = (
                        [len(subset) for subset in candidate_subsets]
                        if candidate_subsets
                        else [0]
                    )
                    size_info = _structured_size_loss_and_prediction(
                        group.get("subset_size_logits"),
                        target_sizes=size_targets,
                        max_group_size=len(splits_in_polytomy),
                    )
                    if size_info is not None:
                        subset_size_losses.append(size_info["loss"].detach())
                        subset_size_target_means.append(
                            float(np.mean(size_info["target_sizes"]))
                        )
                        subset_size_prediction_means.append(
                            float(size_info["predicted_size"])
                        )
                        subset_size_accuracies.append(
                            1.0
                            if int(size_info["predicted_size"])
                            in {int(size) for size in size_info["target_sizes"]}
                            else 0.0
                        )

                if candidate_subsets:
                    candidate_losses = []
                    candidate_targets = []
                    candidate_pred_logits = []
                    for subset in candidate_subsets:
                        if decoder_mode == "structured_subset":
                            structured = _structured_subset_loss_and_prediction(
                                group,
                                splits_in_polytomy,
                                subset,
                                include_metric_logits=(
                                    not skip_autoregressive_merge_metrics
                                ),
                            )
                            if structured is None:
                                continue
                            candidate_losses.append(structured["loss"])
                            candidate_targets.append(structured["target_logits"])
                            candidate_pred_logits.append(structured["predicted_logits"])
                        else:
                            G = logits.size(0)
                            mask = ~torch.eye(
                                G, dtype=torch.bool, device=logits.device
                            )
                            tri = torch.triu(mask, diagonal=1)

                            y = _subset_target_matrix(
                                splits_in_polytomy,
                                subset,
                                logits.device,
                            )
                            y_vec = y[tri]
                            candidate_targets.append(y)

                            logits_vec = logits[tri]
                            finite = torch.isfinite(logits_vec)
                            logits_vec_f = logits_vec[finite]
                            y_vec_f = y_vec[finite]

                            pos = y_vec_f.sum().clamp(min=1.0)
                            neg = (y_vec_f.numel() - y_vec_f.sum()).clamp(min=1.0)
                            pos_weight = (neg / pos).detach()

                            candidate_losses.append(
                                F.binary_cross_entropy_with_logits(
                                    logits_vec_f,
                                    y_vec_f,
                                    pos_weight=pos_weight,
                                    reduction="mean",
                                )
                            )
                            candidate_pred_logits.append(logits)

                    if candidate_losses:
                        loss_stack = torch.stack(candidate_losses)
                        best_candidate_index = int(torch.argmin(loss_stack).item())
                        loss = loss_stack[best_candidate_index]
                        best_target = candidate_targets[best_candidate_index]
                        best_pred_logits = candidate_pred_logits[best_candidate_index]

                        if (
                            decoder_mode == "structured_subset"
                            and 0.0 > 0.0
                            and "batched_autoregressive_stop_after_merge" in batch
                            and group.get("stop_after_merge_logit") is not None
                        ):
                            stop_target = batch[
                                "batched_autoregressive_stop_after_merge"
                            ][batch_index].to(group["stop_after_merge_logit"].device)
                            stop_loss = F.binary_cross_entropy_with_logits(
                                group["stop_after_merge_logit"].view(()),
                                stop_target.view(()),
                            )
                            loss = (
                                loss
                                + 0.0
                                * stop_loss
                            )
                            stop_after_merge_losses.append(stop_loss.detach())
                            stop_prob = torch.sigmoid(
                                group["stop_after_merge_logit"].detach()
                            )
                            stop_after_merge_targets.append(float(stop_target.item()))
                            stop_after_merge_predictions.append(float(stop_prob.item()))
                            stop_after_merge_accuracies.append(
                                1.0
                                if ((stop_prob > 0.5).float() == stop_target).item()
                                else 0.0
                            )

                        if best_pred_logits is not None and best_target is not None:
                            metrics = compute_merge_metrics(
                                best_pred_logits,
                                best_target,
                                threshold_logit=0.0,
                            )
                            total_metrics.append(metrics)

                        losses.append(loss)
                elif size_info is not None:
                    losses.append(size_info["loss"])

            loss_device = (
                all_group_logits[0]["logits"].device if all_group_logits else self.device
            )

            missing_explicit_targets = sum(
                1 for was_found in found.values() if not was_found
            )
            if missing_explicit_targets > 0:
                if self.autoregressive_target_mode == "ready_alternatives":
                    if self.verbose:
                        logger.info(
                            "Autoregressive explicit-target misses under ready-alternatives: %s",
                            missing_explicit_targets,
                        )
                    logs["autoregressive_stats/missing_explicit_targets"] = torch.tensor(
                        float(missing_explicit_targets),
                        device=loss_device,
                    )
                else:
                    for (batch_index, split_mask), was_found in found.items():
                        if not was_found:
                            print(
                                "Missing split: ",
                                [
                                    j
                                    for j in range(int(split_mask).bit_length())
                                    if (int(split_mask) >> j) & 1
                                ],
                            )
                            raise Exception(
                                f"Did not find merge for split {split_mask} in batch element {batch_index}!"
                            )

            L_polytomy_choosing = None

            if len(chosen_polytomies) > 1:
                polytomy_logits_tensor = torch.stack(polytomy_logits).squeeze(1)
                chosen_polytomies_tensor = torch.stack(chosen_polytomies).to(polytomy_logits_tensor.device)
                L_polytomy_choosing = F.binary_cross_entropy_with_logits(
                    polytomy_logits_tensor,
                    chosen_polytomies_tensor,
                ) 

                if self.training_step_verbose_logging_enabled:
                    logger.info(f"Polytomy choosing loss: {L_polytomy_choosing.item()}")
                if self.record:
                    self._wandb_log_filtered(
                        {
                            "train/polytomy_choosing_loss": L_polytomy_choosing.item(),
                            "train/polytomy_choosing_loss_weighted": (
                                self.autoregressive_polytomy_choosing_weight
                                * L_polytomy_choosing.item()
                            ),
                        },
                        step=self.stepper,
                    )

            if losses:
                L_merging = torch.stack(losses).mean()
            else:
                anchor_param = next(self.model.parameters())
                L_merging = anchor_param.sum() * 0.0
                if self.training_step_verbose_logging_enabled:
                    logger.info(
                        "Autoregressive loss skipped because no candidate merge targets were available."
                    )
                logs["autoregressive_stats/no_candidate_merge_loss"] = torch.tensor(
                    1.0,
                    device=loss_device,
                )
            logs["loss"] = _combine_autoregressive_losses(
                L_merging,
                L_polytomy_choosing,
                self.autoregressive_polytomy_choosing_weight,
            )
            if self.training_step_verbose_logging_enabled:
                logger.info(f"Autoregressive loss: {L_merging.item()}")

            aggregated_metrics = {}
            if len(total_metrics) > 0:
                for key in total_metrics[0]:
                    aggregated_metrics[key] = sum(
                        m[key] for m in total_metrics
                    ) / len(total_metrics)

                if self.training_step_verbose_logging_enabled:
                    for key in aggregated_metrics:
                        logger.info(f"{key}: {aggregated_metrics[key]}")

            if L_polytomy_choosing is not None:
                logs["autoregressive_stats/polytomy_choosing_weight"] = torch.tensor(
                    float(self.autoregressive_polytomy_choosing_weight),
                    device=loss_device,
                )

            if stop_after_merge_losses:
                logs["autoregressive_stats/stop_after_merge_loss"] = torch.stack(
                    stop_after_merge_losses
                ).mean().to(loss_device)
                logs["autoregressive_stats/stop_after_merge_accuracy"] = torch.tensor(
                    float(np.mean(stop_after_merge_accuracies)),
                    device=loss_device,
                )
                logs["autoregressive_stats/stop_after_merge_target_rate"] = torch.tensor(
                    float(np.mean(stop_after_merge_targets)),
                    device=loss_device,
                )
                logs["autoregressive_stats/stop_after_merge_pred_rate"] = torch.tensor(
                    float(np.mean(stop_after_merge_predictions)),
                    device=loss_device,
                )
            if subset_size_losses:
                logs["autoregressive_stats/subset_size_loss"] = torch.stack(
                    subset_size_losses
                ).mean().to(loss_device)
                logs["autoregressive_stats/subset_size_accuracy"] = torch.tensor(
                    float(np.mean(subset_size_accuracies)),
                    device=loss_device,
                )
                logs["autoregressive_stats/subset_size_target_mean"] = torch.tensor(
                    float(np.mean(subset_size_target_means)),
                    device=loss_device,
                )
                logs["autoregressive_stats/subset_size_pred_mean"] = torch.tensor(
                    float(np.mean(subset_size_prediction_means)),
                    device=loss_device,
                )

            # Calculate average polytomy size
            avg_polytomy_size = np.mean(polytomy_sizes) if polytomy_sizes else 0.0
            num_polytomies = len(polytomy_sizes)
            avg_alternative_targets = (
                float(np.mean(alternative_target_counts))
                if alternative_target_counts
                else 0.0
            )
            if self.training_step_verbose_logging_enabled:
                logger.info(f"Average polytomy size: {avg_polytomy_size}")
                logger.info(
                    f"Average alternative autoregressive targets: {avg_alternative_targets}"
                )
            logs["autoregressive_stats/avg_candidate_targets"] = torch.tensor(
                avg_alternative_targets,
                device=loss_device,
            )
            logs["autoregressive_stats/rollin_attempted"] = torch.tensor(
                ar_prep_stats["rollin_attempted"],
                device=loss_device,
            )
            logs["autoregressive_stats/rollin_applied"] = torch.tensor(
                ar_prep_stats["rollin_applied"],
                device=loss_device,
            )
            logs["autoregressive_stats/dagger_attempted"] = torch.tensor(
                ar_prep_stats["dagger_attempted"],
                device=loss_device,
            )
            logs["autoregressive_stats/dagger_applied"] = torch.tensor(
                ar_prep_stats["dagger_applied"],
                device=loss_device,
            )
            dagger_avg_steps = (
                ar_prep_stats["dagger_rollout_steps"] / ar_prep_stats["dagger_applied"]
                if ar_prep_stats["dagger_applied"] > 0.0
                else 0.0
            )
            logs["autoregressive_stats/dagger_avg_rollout_steps"] = torch.tensor(
                dagger_avg_steps,
                device=loss_device,
            )
            logs["autoregressive_stats/structure_perturb_attempted"] = torch.tensor(
                ar_prep_stats["structure_perturb_attempted"],
                device=loss_device,
            )
            logs["autoregressive_stats/structure_perturb_applied"] = torch.tensor(
                ar_prep_stats["structure_perturb_applied"],
                device=loss_device,
            )

            if self.record and not is_replay_batch:
                # Batch all metrics into a single wandb.log call to avoid step conflicts
                wandb_metrics = {
                    "train/autoregressive_loss": L_merging.item(),
                    "autoregressive_stats/avg_polytomy_size": avg_polytomy_size,
                    "autoregressive_stats/num_polytomies": num_polytomies,
                    "autoregressive_stats/avg_candidate_targets": avg_alternative_targets,
                    "autoregressive_stats/rollin_attempted": ar_prep_stats["rollin_attempted"],
                    "autoregressive_stats/rollin_applied": ar_prep_stats["rollin_applied"],
                    "autoregressive_stats/dagger_attempted": ar_prep_stats["dagger_attempted"],
                    "autoregressive_stats/dagger_applied": ar_prep_stats["dagger_applied"],
                    "autoregressive_stats/dagger_avg_rollout_steps": dagger_avg_steps,
                    "autoregressive_stats/structure_perturb_attempted": ar_prep_stats["structure_perturb_attempted"],
                    "autoregressive_stats/structure_perturb_applied": ar_prep_stats["structure_perturb_applied"],
                }
                wandb_metrics.update(
                    {f"{key}": aggregated_metrics[key] for key in aggregated_metrics}
                )
                self._wandb_log_filtered(wandb_metrics, step=self.stepper)

        return logs
