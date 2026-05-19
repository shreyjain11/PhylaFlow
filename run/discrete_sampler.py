import torch
from ete3 import Tree as EteTree

from run.training_helpers import *


class DiscreteSamplerMixin:
    def sample(
        self,
        newick_starting_trees: list[str],
        phyla_embeddings,
        case_indices=None,
        num_samples=None,
        mapping=None,
        dataset_ids=None,
        T=1.0,
        dt_base=0.02,
        eps_len=1e-8,
        hit_tol=1e-10,
        first_hit_tol=1e-4,
        autoregressive_birth_length=1e-3,
        stop_on_no_valid_merge=False,
        max_events=1000,
        max_steps=20000,
        topology_repeat_cap=0,
        KNN_TOPM=32,
        KNN_TAU=0.05,
        KNN_STOCHASTIC=False,
        debug_real_tree=None,
        return_trace: bool = False,
        target_trees: list[str] | None = None,
        first_hit_start_topology_features=None,
        autoregressive_start_topology_features=None,
        first_hit_start_topology_embeddings=None,
        first_hit_start_topology_pad_mask=None,
        first_hit_start_tree_graph_context=None,
        split_multi_label_events: bool = False,
        max_allowed_polytomy_size: int = -1,
        oversize_polytomy_policy: str = "none",
        oversize_polytomy_blacklist_revisits: bool = False,
        oversize_polytomy_min_dt_escape: float = 0.0,
        fixed_dt_sampling: bool = False,
        max_autoregressive_merges_per_boundary: int = -1,
        prefix_replay_velocity_quota: int = 0,
        prefix_replay_autoregressive_quota: int = 0,
        prefix_replay_split_multi_label_events: bool = False,
        oracle_first_hit_use_at_sampling: bool = False,
        oracle_gate_first_hit_use_at_sampling: bool = False,
        oracle_boundary_vanish_use_at_sampling: bool = False,
        trace_state_rf: bool = True,
        explicit_autoregressive_component_groups: bool = True,
    ):
        if target_trees is None:
            raise ValueError("sample() now requires target_trees for the final discrete-phase sampler.")
        if len(newick_starting_trees) != len(target_trees):
            raise ValueError("sample() requires one target tree per starting tree.")

        shared_start_topology_features = first_hit_start_topology_features
        if shared_start_topology_features is None:
            shared_start_topology_features = autoregressive_start_topology_features

        if (
            phyla_embeddings is None
            and (
                self.phyla_precomputed_name_to_embedding is not None
                or bool(getattr(self, "phyla_precomputed_by_dataset_id", None))
            )
        ):
            batch_embeddings = []
            for tree_idx, tree_newick in enumerate(newick_starting_trees):
                tree_mapping = None
                if isinstance(mapping, list):
                    tree_mapping = mapping[tree_idx]
                elif isinstance(mapping, dict):
                    tree_mapping = mapping
                dataset_id = None
                if isinstance(dataset_ids, list):
                    dataset_id = dataset_ids[tree_idx]
                elif isinstance(dataset_ids, str):
                    dataset_id = dataset_ids
                resolved = self._resolve_precomputed_phyla_embeddings_for_tree(
                    tree_newick,
                    mapping=tree_mapping,
                    device=self.device,
                    dataset_id=dataset_id,
                )
                if resolved is None:
                    batch_embeddings = []
                    break
                batch_embeddings.append(resolved.squeeze(0))
            if batch_embeddings:
                phyla_embeddings = torch.stack(batch_embeddings, dim=0)

        self.model.eval()
        final_trees = []
        traces = []
        num_ar_states_total = 0

        for idx, (start_tree, target_tree) in enumerate(zip(newick_starting_trees, target_trees)):
            case_index = None
            if case_indices is not None:
                case_index = int(torch.as_tensor(case_indices).reshape(-1)[idx].item())

            tree_phyla_embeddings = phyla_embeddings
            if torch.is_tensor(phyla_embeddings) and phyla_embeddings.ndim >= 3:
                tree_phyla_embeddings = phyla_embeddings[idx : idx + 1]

            start_features = shared_start_topology_features
            if torch.is_tensor(shared_start_topology_features) and shared_start_topology_features.ndim >= 2:
                start_features = shared_start_topology_features[idx : idx + 1]

            start_embeddings = first_hit_start_topology_embeddings
            if torch.is_tensor(first_hit_start_topology_embeddings) and first_hit_start_topology_embeddings.ndim >= 3:
                start_embeddings = first_hit_start_topology_embeddings[idx : idx + 1]

            start_pad_mask = first_hit_start_topology_pad_mask
            if torch.is_tensor(first_hit_start_topology_pad_mask) and first_hit_start_topology_pad_mask.ndim >= 2:
                start_pad_mask = first_hit_start_topology_pad_mask[idx : idx + 1]

            rollout = _discrete_phase_rollout(
                self,
                start_tree,
                target_tree,
                tree_phyla_embeddings,
                case_index=case_index,
                start_topology_features=start_features,
                start_topology_embeddings=start_embeddings,
                start_topology_pad_mask=start_pad_mask,
                dt_base=float(dt_base),
                eps_len=float(eps_len),
                autoregressive_birth_length=float(autoregressive_birth_length),
                max_events=max_events,
                max_steps=max_steps,
                max_phases=int(getattr(self, "sampling_discrete_phase_max_phases", 8)),
                return_trace=return_trace,
                trace_state_rf=bool(trace_state_rf),
                explicit_autoregressive_component_groups=bool(
                    explicit_autoregressive_component_groups
                ),
            )
            final_trees.append(rollout["final_tree"])
            num_ar_states_total += int(rollout.get("num_ar_states", 0))
            if return_trace:
                traces.append(rollout["trace"])

        result = (
            final_trees,
            num_ar_states_total,
            0.0,
            0.0,
            num_ar_states_total,
        )
        if return_trace:
            return result + (traces[0] if len(traces) == 1 else traces,)
        return result

