import numbers
import logging
import os
import sys

import torch
import torch.optim as optim
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

try:
    import wandb
except ImportError:  # pragma: no cover - optional experiment logger
    wandb = None

# Ensure the current directory is in sys.path to import 'phyla'
sys.path.append(os.getcwd())

from data.dataset import PhylaDataModule
from model.model import TreeDenoiserTokenGT

logger = logging.getLogger(__name__)


from run.training_helpers import *
from run.sample_metrics import SampleMetricsMixin
from run.branch_relaxer import BranchRelaxerMixin, BranchRelaxHead
from run.phyla_embeddings import PhylaEmbeddingMixin, _load_phyla_runtime
from run.training_losses import TrainingLossMixin
from run.discrete_sampler import DiscreteSamplerMixin


class TrainingModule(
    SampleMetricsMixin,
    BranchRelaxerMixin,
    PhylaEmbeddingMixin,
    TrainingLossMixin,
    DiscreteSamplerMixin,
    LightningModule,
):
    def __init__(
        self,
        model: TreeDenoiserTokenGT,
        dataset: PhylaDataModule,
        lr: float = 1e-4,
        optimizer_name: str = "adamw",
        record=False,
        epochs: int = 5000,
        lr_scheduler: str = "default",
        num_annealing_steps: int = 10000,
        num_warmup_steps: int = 1000,
        logger=None,
        max_num_timesteps: int = 20,
        training_sampling_frequency: int = 200,
        training_sampling_start: int = 500,
        training_sampling_dt_base: float = 0.02,
        sampling_fixed_dt_base: float | None = None,
        sampling_max_steps: int | None = 256,
        sampling_max_events: int | None = None,
        sampling_max_autoregressive_merges_per_boundary: int = -1,
        training_sampling_stop_on_zero_rf: bool = False,
        training_sampling_stop_rf_threshold: float | None = None,
        num_samples: int = 10,
        dt: float = 0.1,
        # Figure out how to do typing here
        global_splits=None,
        random_trees=None,
        verbose: bool = False,
        phyla_checkpoint_path=None,
        phyla_precomputed_embeddings_path: str | None = None,
        velocity_loss_mode: str = "weighted",
        velocity_loss_plain_weight: float = 0.5,
        velocity_sign_eps: float = 1e-3,
        training_step_velocity_weight: float = 1.0,
        training_step_autoregressive_weight: float = 1.0,
        training_step_gradient_clip_val: float = 1.0,
        training_step_separate_optimizer_steps: bool = False,
        training_step_verbose_logging_enabled: bool = False,
        autoregressive_use_time: bool = False,
        autoregressive_target_mode: str = "scheduled",
        autoregressive_polytomy_choosing_weight: float = 1.0,
        autoregressive_rollin_prob: float = 0.0,
        autoregressive_dagger_prob: float = 0.0,
        autoregressive_dagger_max_steps: int = 4,
        autoregressive_structure_perturb_prob: float = 0.0,
        autoregressive_structure_perturb_mode: str = "random_wrong_pair",
        velocity_length_jitter_prob: float = 0.0,
        velocity_length_jitter_scale: float = 0.0,
        velocity_dt_candidate_weight: float = 0.0,
        velocity_dt_hit_weight: float = 0.0,
        velocity_logtau_all_weight: float = 0.0,
        velocity_logtau_first_over_weight: float = 0.0,
        velocity_logtau_predset_over_weight: float = 0.0,
        velocity_dt_eps: float = 1e-6,
        velocity_event_weight: float = 0.5,
        velocity_event_temp: float = 0.5,
        velocity_event_rate_beta: float = 5.0,
        velocity_event_normalize_by_log_candidates: bool = True,
        velocity_event_precision_weight: float = 0.0,
        velocity_event_precision_margin: float = 0.0,
        velocity_first_hit_head_weight: float = 0.0,
        velocity_first_hit_loss_tol: float = 0.01,
        velocity_first_hit_false_positive_mass_weight: float = 0.0,
        velocity_first_hit_head_use_at_sampling: bool = False,
        velocity_probe_direct_set_loss: bool = False,
        velocity_probe_direct_set_anchor_only: bool = False,
        velocity_probe_direct_set_target_negative_weight: float = 1.0,
        velocity_probe_direct_set_positive_reweight: bool = False,
        velocity_probe_direct_set_include_base_samples: bool = False,
        velocity_probe_direct_set_positive_reweight_power: float = 1.0,
        velocity_probe_direct_set_positive_reweight_max: float | None = None,
        velocity_probe_direct_set_loss_weight: float = 1.0,
        training_step_probe_parity_joint_update: bool = False,
        sampling_discrete_phase_rollout_use_at_sampling: bool = False,
        sampling_discrete_phase_exact_boundary_step_use_at_sampling: bool = False,
        sampling_discrete_phase_max_phases: int = 8,
        sample_metrics_trace_path: str | None = None,
        sample_metrics_num_pairs: int = 1,
        sample_metrics_trace_topology_repeats_enabled: bool = False,
        sample_metrics_unseen_start_eval: bool = False,
        sample_metrics_unseen_start_seed: int = 20260430,
        sample_metrics_unseen_start_metric_encoder_path: str | None = None,
        sample_metrics_unseen_pair_selection_mode: str = "random_bank",
        sample_metrics_unseen_start_max_duplicate_tries: int = 100,
        sample_metrics_batched_discrete_phase_enabled: bool = True,
        sample_metrics_reuse_tokenizer_edge_lengths_enabled: bool = False,
        sample_metrics_relaxed_likelihood_enabled: bool = False,
        sample_metrics_branch_relaxer_checkpoint_path: str | None = None,
        sample_metrics_mrbayes20k_enabled: bool = False,
        sample_metrics_mrbayes20k_num_starts: int = 64,
        sample_metrics_mrbayes20k_ngen: int = 20000,
        sample_metrics_mrbayes20k_samplefreq: int = 200,
        sample_metrics_mrbayes20k_printfreq: int = 5000,
        sample_metrics_mrbayes20k_max_workers: int = 12,
        sample_metrics_mrbayes20k_timeout_sec: int = 1800,
        sample_metrics_mrbayes20k_dataset_pickle_path: str | None = None,
        sample_metrics_mrbayes20k_golden_root: str | None = None,
        sample_metrics_mrbayes20k_work_root: str = "/tmp/phylaflow_sample_metrics_mrbayes20k",
        sample_metrics_mrbayes20k_output_dir: str | None = None,
        sample_metrics_mrbayes20k_bin: str = "/opt/conda/envs/phylaflow-mrbayes/bin/mb",
        sample_metrics_tree_dump_enabled: bool = False,
        sample_metrics_tree_dump_dir: str | None = None,
        metric_log_exact_keys=None,
        metric_log_prefixes=None,
        branch_relax_head_weight: float = 0.0,
        branch_relax_head_use_at_sampling: bool = False,
        branch_relax_start_tree_list_path: str | None = None,
        branch_relax_target_tree_list_path: str | None = None,
        branch_relax_detach_trunk: bool = True,
        branch_relax_batch_size: int = 1,
        branch_relax_case_dim: int = 64,
        branch_relax_hidden_dim: int = 256,
        branch_relax_likelihood_dataset_id: str | None = None,
        branch_relax_likelihood_metric_enabled: bool = False,
        sampling_disable_inner_logging: bool = True,
        sampling_blocked_edge_floor: float | None = None,
        sampling_random_fixed_pair_bank_use_at_sampling: bool = False,
        velocity_first_hit_sampling_max_edges: int = -1,
        velocity_first_hit_sampling_fallback_threshold: int = -1,
        velocity_first_hit_sampling_fallback_top_k: int = -1,
        sampling_use_inference_mode: bool = False,
        **_removed_options,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.optimizer_name = str(optimizer_name).strip().lower()
        self.record = record
        self.epochs = epochs
        self.warmup_steps = 400
        self.current_step_value = 0
        self.lr_scheduler = lr_scheduler
        self.num_annealing_steps = num_annealing_steps
        self.num_warmup_steps = num_warmup_steps
        self.dataset = dataset
        self.max_num_timesteps = max_num_timesteps
        self.global_splits = global_splits
        self.random_trees = random_trees
        self.verbose = verbose
        self.training_sampling_frequency = training_sampling_frequency
        self.training_sampling_start = training_sampling_start
        self._next_training_sample_step = None
        self.training_sampling_dt_base = float(training_sampling_dt_base)
        self.sampling_fixed_dt_base = (
            None
            if sampling_fixed_dt_base is None
            else float(sampling_fixed_dt_base)
        )
        self.sampling_max_steps = (
            None
            if sampling_max_steps is None or int(sampling_max_steps) < 0
            else int(sampling_max_steps)
        )
        self.sampling_max_events_uncapped = bool(
            sampling_max_events is not None and int(sampling_max_events) < 0
        )
        self.sampling_max_events = (
            None
            if sampling_max_events is None or int(sampling_max_events) < 0
            else int(sampling_max_events)
        )
        self.sampling_max_autoregressive_merges_per_boundary = int(
            sampling_max_autoregressive_merges_per_boundary
        )
        self.training_sampling_stop_on_zero_rf = bool(training_sampling_stop_on_zero_rf)
        self.training_sampling_stop_rf_threshold = (
            None
            if training_sampling_stop_rf_threshold is None
            else float(training_sampling_stop_rf_threshold)
        )
        self.num_samples = num_samples
        self.dt = dt
        self.training_step_gradient_clip_val = float(training_step_gradient_clip_val)
        self.train_tokenized_trees = None
        self.train_batched_time = None
        self.train_tree = None
        self._cached_harness_sampling_pairs = {}
        self.sampling_random_fixed_pair_bank_use_at_sampling = bool(
            sampling_random_fixed_pair_bank_use_at_sampling
        )

        self.automatic_optimization = False
        self.logger_ = logger
        if verbose:
            logging.getLogger("filelock").setLevel(logging.WARNING)
            logging.getLogger("fsspec").setLevel(logging.WARNING)
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        self.phyla_checkpoint_path = phyla_checkpoint_path
        self.phyla_precomputed_embeddings_path = phyla_precomputed_embeddings_path
        self.phyla_model = None
        self.phyla_precomputed_name_to_embedding = None
        self.phyla_precomputed_by_dataset_id = {}
        self.stepper = 1

        phyla_config_path = "configs/sample_eval_config.yaml"
        if self.phyla_checkpoint_path is not None:
            original_argv = sys.argv
            sys.argv = ["script", phyla_config_path]
            try:
                if not os.path.exists(phyla_config_path):
                    logging.warning(
                        f"Phyla configuration file not found at {phyla_config_path}"
                    )

                load_config, Config, load_model, _ = _load_phyla_runtime()
                config = load_config(Config)
                config.trainer.checkpoint_path = self.phyla_checkpoint_path
                config.eval.device = "cuda" if torch.cuda.is_available() else "cpu"
                loaded = load_model(config=config, random_model=False)
                self.phyla_model = loaded["model"]
                self.phyla_model.eval()
                if verbose:
                    logging.info("Phyla model loaded successfully.")
            except Exception as e:
                logging.warning(f"Failed to load Phyla model: {e}")
            finally:
                sys.argv = original_argv

        if self.phyla_precomputed_embeddings_path is not None:
            try:
                self._load_precomputed_phyla_embeddings(
                    self.phyla_precomputed_embeddings_path
                )
                if verbose:
                    logging.info(
                        "Loaded precomputed Phyla embeddings from %s",
                        self.phyla_precomputed_embeddings_path,
                    )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load precomputed Phyla embeddings from "
                    f"{self.phyla_precomputed_embeddings_path}: {e}"
                ) from e

        if self.optimizer_name not in {"adam", "adamw"}:
            raise ValueError(
                "optimizer_name must be one of ['adam', 'adamw'], "
                f"got {optimizer_name!r}."
            )

        valid_velocity_loss_modes = {"plain", "weighted", "blended"}
        if velocity_loss_mode not in valid_velocity_loss_modes:
            raise ValueError(
                f"Invalid velocity_loss_mode={velocity_loss_mode!r}. "
                f"Expected one of {sorted(valid_velocity_loss_modes)}."
            )
        if not (0.0 <= float(velocity_loss_plain_weight) <= 1.0):
            raise ValueError(
                "velocity_loss_plain_weight must be in [0, 1], "
                f"got {velocity_loss_plain_weight}."
            )
        if float(velocity_sign_eps) < 0.0:
            raise ValueError(
                f"velocity_sign_eps must be non-negative, got {velocity_sign_eps}."
            )
        if float(training_step_velocity_weight) < 0.0:
            raise ValueError(
                "training_step_velocity_weight must be non-negative, "
                f"got {training_step_velocity_weight}."
            )
        if float(training_step_autoregressive_weight) < 0.0:
            raise ValueError(
                "training_step_autoregressive_weight must be non-negative, "
                f"got {training_step_autoregressive_weight}."
            )
        valid_autoregressive_target_modes = {"scheduled", "ready_alternatives"}
        if autoregressive_target_mode not in valid_autoregressive_target_modes:
            raise ValueError(
                f"Invalid autoregressive_target_mode={autoregressive_target_mode!r}. "
                f"Expected one of {sorted(valid_autoregressive_target_modes)}."
            )
        if self.training_sampling_dt_base <= 0.0:
            raise ValueError(
                "training_sampling_dt_base must be > 0, "
                f"got {training_sampling_dt_base}."
            )
        if (
            self.sampling_fixed_dt_base is not None
            and self.sampling_fixed_dt_base <= 0.0
        ):
            raise ValueError(
                "sampling_fixed_dt_base must be > 0 when provided, "
                f"got {sampling_fixed_dt_base}."
            )
        if sampling_max_steps is not None and int(sampling_max_steps) == 0:
            raise ValueError(
                "sampling_max_steps must be >= 1 or < 0 for uncapped, "
                f"got {sampling_max_steps}."
            )
        if sampling_max_events is not None and int(sampling_max_events) == 0:
            raise ValueError(
                "sampling_max_events must be >= 1 or < 0 for uncapped, "
                f"got {sampling_max_events}."
            )
        if int(sampling_max_autoregressive_merges_per_boundary) == 0:
            raise ValueError(
                "sampling_max_autoregressive_merges_per_boundary must be >= 1 or < 0 "
                f"for uncapped, got {sampling_max_autoregressive_merges_per_boundary}."
            )
        if self.training_step_gradient_clip_val < 0.0:
            raise ValueError(
                "training_step_gradient_clip_val must be >= 0, "
                f"got {training_step_gradient_clip_val}."
            )
        valid_structure_perturb_modes = {"random_wrong_pair", "model_wrong_pair"}
        if (
            autoregressive_structure_perturb_mode
            not in valid_structure_perturb_modes
        ):
            raise ValueError(
                "Invalid autoregressive_structure_perturb_mode="
                f"{autoregressive_structure_perturb_mode!r}. Expected one of "
                f"{sorted(valid_structure_perturb_modes)}."
            )
        if not (0.0 <= float(autoregressive_rollin_prob) <= 1.0):
            raise ValueError(
                "autoregressive_rollin_prob must be in [0, 1], "
                f"got {autoregressive_rollin_prob}."
            )
        if not (0.0 <= float(autoregressive_dagger_prob) <= 1.0):
            raise ValueError(
                "autoregressive_dagger_prob must be in [0, 1], "
                f"got {autoregressive_dagger_prob}."
            )
        if int(autoregressive_dagger_max_steps) < 1:
            raise ValueError(
                "autoregressive_dagger_max_steps must be >= 1, "
                f"got {autoregressive_dagger_max_steps}."
            )
        if not (0.0 <= float(autoregressive_structure_perturb_prob) <= 1.0):
            raise ValueError(
                "autoregressive_structure_perturb_prob must be in [0, 1], "
                f"got {autoregressive_structure_perturb_prob}."
            )
        if not (0.0 <= float(velocity_length_jitter_prob) <= 1.0):
            raise ValueError(
                "velocity_length_jitter_prob must be in [0, 1], "
                f"got {velocity_length_jitter_prob}."
            )
        if float(velocity_length_jitter_scale) < 0.0:
            raise ValueError(
                "velocity_length_jitter_scale must be non-negative, "
                f"got {velocity_length_jitter_scale}."
            )
        if float(velocity_dt_candidate_weight) < 0.0:
            raise ValueError(
                "velocity_dt_candidate_weight must be non-negative, "
                f"got {velocity_dt_candidate_weight}."
            )
        if float(velocity_dt_hit_weight) < 0.0:
            raise ValueError(
                "velocity_dt_hit_weight must be non-negative, "
                f"got {velocity_dt_hit_weight}."
            )
        if float(velocity_logtau_all_weight) < 0.0:
            raise ValueError(
                "velocity_logtau_all_weight must be non-negative, "
                f"got {velocity_logtau_all_weight}."
            )
        if float(velocity_logtau_first_over_weight) < 0.0:
            raise ValueError(
                "velocity_logtau_first_over_weight must be non-negative, "
                f"got {velocity_logtau_first_over_weight}."
            )
        if float(velocity_logtau_predset_over_weight) < 0.0:
            raise ValueError(
                "velocity_logtau_predset_over_weight must be non-negative, "
                f"got {velocity_logtau_predset_over_weight}."
            )
        if float(velocity_dt_eps) <= 0.0:
            raise ValueError(
                f"velocity_dt_eps must be > 0, got {velocity_dt_eps}."
            )
        if float(velocity_event_weight) < 0.0:
            raise ValueError(
                "velocity_event_weight must be non-negative, "
                f"got {velocity_event_weight}."
            )
        if float(velocity_event_temp) <= 0.0:
            raise ValueError(
                f"velocity_event_temp must be > 0, got {velocity_event_temp}."
            )
        if float(velocity_event_rate_beta) <= 0.0:
            raise ValueError(
                f"velocity_event_rate_beta must be > 0, got {velocity_event_rate_beta}."
            )
        if float(velocity_event_precision_weight) < 0.0:
            raise ValueError(
                "velocity_event_precision_weight must be non-negative, "
                f"got {velocity_event_precision_weight}."
            )
        if float(velocity_event_precision_margin) < 0.0:
            raise ValueError(
                "velocity_event_precision_margin must be non-negative, "
                f"got {velocity_event_precision_margin}."
            )
        if float(velocity_first_hit_head_weight) < 0.0:
            raise ValueError(
                "velocity_first_hit_head_weight must be non-negative, "
                f"got {velocity_first_hit_head_weight}."
            )
        if float(velocity_first_hit_loss_tol) < 0.0:
            raise ValueError(
                "velocity_first_hit_loss_tol must be non-negative, "
                f"got {velocity_first_hit_loss_tol}."
            )
        if float(velocity_first_hit_false_positive_mass_weight) < 0.0:
            raise ValueError(
                "velocity_first_hit_false_positive_mass_weight must be non-negative, "
                f"got {velocity_first_hit_false_positive_mass_weight}."
            )
        if int(sampling_discrete_phase_max_phases) < 1:
            raise ValueError(
                "sampling_discrete_phase_max_phases must be >= 1, "
                f"got {sampling_discrete_phase_max_phases}."
            )
        self.velocity_loss_mode = velocity_loss_mode
        self.velocity_loss_plain_weight = float(velocity_loss_plain_weight)
        self.velocity_sign_eps = float(velocity_sign_eps)
        self.training_step_velocity_weight = float(training_step_velocity_weight)
        self.training_step_autoregressive_weight = float(
            training_step_autoregressive_weight
        )
        self.training_step_separate_optimizer_steps = bool(
            training_step_separate_optimizer_steps
        )
        self.training_step_verbose_logging_enabled = bool(
            training_step_verbose_logging_enabled
        )
        self.autoregressive_use_time = bool(autoregressive_use_time)
        self.autoregressive_target_mode = str(autoregressive_target_mode)
        self.autoregressive_polytomy_choosing_weight = float(
            autoregressive_polytomy_choosing_weight
        )
        self.autoregressive_rollin_prob = float(autoregressive_rollin_prob)
        self.autoregressive_dagger_prob = float(autoregressive_dagger_prob)
        self.autoregressive_dagger_max_steps = int(autoregressive_dagger_max_steps)
        self.autoregressive_structure_perturb_prob = float(
            autoregressive_structure_perturb_prob
        )
        self.autoregressive_structure_perturb_mode = str(
            autoregressive_structure_perturb_mode
        )
        self.velocity_length_jitter_prob = float(velocity_length_jitter_prob)
        self.velocity_length_jitter_scale = float(velocity_length_jitter_scale)
        self.velocity_dt_candidate_weight = float(velocity_dt_candidate_weight)
        self.velocity_dt_hit_weight = float(velocity_dt_hit_weight)
        self.velocity_logtau_all_weight = float(velocity_logtau_all_weight)
        self.velocity_logtau_first_over_weight = float(
            velocity_logtau_first_over_weight
        )
        self.velocity_logtau_predset_over_weight = float(
            velocity_logtau_predset_over_weight
        )
        self.velocity_dt_eps = float(velocity_dt_eps)
        self.velocity_event_weight = float(velocity_event_weight)
        self.velocity_event_temp = float(velocity_event_temp)
        self.velocity_event_rate_beta = float(velocity_event_rate_beta)
        self.velocity_event_normalize_by_log_candidates = bool(
            velocity_event_normalize_by_log_candidates
        )
        self.velocity_event_precision_weight = float(velocity_event_precision_weight)
        self.velocity_event_precision_margin = float(velocity_event_precision_margin)
        self.velocity_first_hit_head_weight = float(velocity_first_hit_head_weight)
        self.velocity_first_hit_loss_tol = float(velocity_first_hit_loss_tol)
        self.velocity_first_hit_false_positive_mass_weight = float(
            velocity_first_hit_false_positive_mass_weight
        )
        self.velocity_first_hit_head_use_at_sampling = bool(
            velocity_first_hit_head_use_at_sampling
        )
        self.sampling_blocked_edge_floor = (
            None
            if sampling_blocked_edge_floor is None
            else float(sampling_blocked_edge_floor)
        )
        self.velocity_first_hit_sampling_max_edges = int(
            velocity_first_hit_sampling_max_edges
        )
        self.velocity_first_hit_sampling_fallback_threshold = int(
            velocity_first_hit_sampling_fallback_threshold
        )
        self.velocity_first_hit_sampling_fallback_top_k = int(
            velocity_first_hit_sampling_fallback_top_k
        )
        self.velocity_first_hit_predictor_mode = "base"
        self.velocity_probe_direct_set_loss = bool(
            velocity_probe_direct_set_loss
        )
        self.velocity_probe_direct_set_anchor_only = bool(
            velocity_probe_direct_set_anchor_only
        )
        self.velocity_probe_direct_set_target_negative_weight = float(
            velocity_probe_direct_set_target_negative_weight
        )
        self.velocity_probe_direct_set_positive_reweight = bool(
            velocity_probe_direct_set_positive_reweight
        )
        self.velocity_probe_direct_set_include_base_samples = bool(
            velocity_probe_direct_set_include_base_samples
        )
        self.velocity_probe_direct_set_positive_reweight_power = float(
            velocity_probe_direct_set_positive_reweight_power
        )
        self.velocity_probe_direct_set_positive_reweight_max = (
            None
            if velocity_probe_direct_set_positive_reweight_max is None
            else float(velocity_probe_direct_set_positive_reweight_max)
        )
        self.velocity_probe_direct_set_loss_weight = float(
            velocity_probe_direct_set_loss_weight
        )
        self.training_step_probe_parity_joint_update = bool(
            training_step_probe_parity_joint_update
        )
        self.sampling_discrete_phase_rollout_use_at_sampling = bool(
            sampling_discrete_phase_rollout_use_at_sampling
        )
        self.sampling_discrete_phase_exact_boundary_step_use_at_sampling = bool(
            sampling_discrete_phase_exact_boundary_step_use_at_sampling
        )
        self.sampling_discrete_phase_max_phases = int(
            sampling_discrete_phase_max_phases
        )
        self.sample_metrics_trace_path = (
            str(sample_metrics_trace_path).strip()
            if sample_metrics_trace_path
            else None
        )
        self.sample_metrics_num_pairs = max(1, int(sample_metrics_num_pairs))
        self.sample_metrics_trace_topology_repeats_enabled = bool(
            sample_metrics_trace_topology_repeats_enabled
        )
        self.sample_metrics_unseen_start_eval = bool(sample_metrics_unseen_start_eval)
        self.sample_metrics_unseen_start_seed = int(sample_metrics_unseen_start_seed)
        self.sample_metrics_unseen_start_metric_encoder_path = (
            os.path.abspath(str(sample_metrics_unseen_start_metric_encoder_path))
            if sample_metrics_unseen_start_metric_encoder_path
            else None
        )
        self.sample_metrics_unseen_pair_selection_mode = str(
            sample_metrics_unseen_pair_selection_mode
        ).strip().lower()
        self.sample_metrics_unseen_start_max_duplicate_tries = max(
            1, int(sample_metrics_unseen_start_max_duplicate_tries)
        )
        self.sample_metrics_batched_discrete_phase_enabled = bool(
            sample_metrics_batched_discrete_phase_enabled
        )
        self.sample_metrics_reuse_tokenizer_edge_lengths_enabled = bool(
            sample_metrics_reuse_tokenizer_edge_lengths_enabled
        )
        self.sample_metrics_relaxed_likelihood_enabled = bool(
            sample_metrics_relaxed_likelihood_enabled
        )
        self.sample_metrics_branch_relaxer_checkpoint_path = (
            os.path.abspath(str(sample_metrics_branch_relaxer_checkpoint_path))
            if sample_metrics_branch_relaxer_checkpoint_path
            else None
        )
        self.sample_metrics_mrbayes20k_enabled = bool(
            sample_metrics_mrbayes20k_enabled
        )
        self.sample_metrics_mrbayes20k_num_starts = max(
            1, int(sample_metrics_mrbayes20k_num_starts)
        )
        self.sample_metrics_mrbayes20k_ngen = max(
            1, int(sample_metrics_mrbayes20k_ngen)
        )
        self.sample_metrics_mrbayes20k_samplefreq = max(
            1, int(sample_metrics_mrbayes20k_samplefreq)
        )
        self.sample_metrics_mrbayes20k_printfreq = max(
            1, int(sample_metrics_mrbayes20k_printfreq)
        )
        self.sample_metrics_mrbayes20k_max_workers = max(
            1, int(sample_metrics_mrbayes20k_max_workers)
        )
        self.sample_metrics_mrbayes20k_timeout_sec = max(
            1, int(sample_metrics_mrbayes20k_timeout_sec)
        )
        self.sample_metrics_mrbayes20k_dataset_pickle_path = (
            os.path.abspath(str(sample_metrics_mrbayes20k_dataset_pickle_path))
            if sample_metrics_mrbayes20k_dataset_pickle_path
            else None
        )
        self.sample_metrics_mrbayes20k_golden_root = (
            os.path.abspath(str(sample_metrics_mrbayes20k_golden_root))
            if sample_metrics_mrbayes20k_golden_root
            else None
        )
        self.sample_metrics_mrbayes20k_work_root = os.path.abspath(
            str(sample_metrics_mrbayes20k_work_root)
        )
        self.sample_metrics_mrbayes20k_output_dir = (
            os.path.abspath(str(sample_metrics_mrbayes20k_output_dir))
            if sample_metrics_mrbayes20k_output_dir
            else None
        )
        self.sample_metrics_mrbayes20k_bin = str(sample_metrics_mrbayes20k_bin)
        self.sample_metrics_tree_dump_enabled = bool(
            sample_metrics_tree_dump_enabled
        )
        self.sample_metrics_tree_dump_dir = (
            os.path.abspath(str(sample_metrics_tree_dump_dir))
            if sample_metrics_tree_dump_dir
            else None
        )
        self._sample_metrics_standalone_relaxer_cache = {}
        self._sample_metrics_likelihood_scorer_cache = {}
        self._sample_metrics_metric_encoder_cache = {}
        self.metric_log_exact_keys = (
            {
                str(key).strip()
                for key in metric_log_exact_keys
                if str(key).strip()
            }
            if metric_log_exact_keys
            else None
        )
        self.metric_log_prefixes = tuple(
            str(prefix).strip()
            for prefix in (metric_log_prefixes or [])
            if str(prefix).strip()
        )
        self.branch_relax_head_weight = float(branch_relax_head_weight)
        self.branch_relax_head_use_at_sampling = bool(branch_relax_head_use_at_sampling)
        self.branch_relax_detach_trunk = bool(branch_relax_detach_trunk)
        self.branch_relax_batch_size = max(1, int(branch_relax_batch_size))
        self.branch_relax_likelihood_dataset_id = (
            str(branch_relax_likelihood_dataset_id).strip()
            if branch_relax_likelihood_dataset_id
            else None
        )
        self.branch_relax_likelihood_metric_enabled = bool(
            branch_relax_likelihood_metric_enabled
        )
        self._branch_relax_likelihood_scorer = None
        self.branch_relax_samples = []
        if branch_relax_start_tree_list_path and branch_relax_target_tree_list_path:
            self.branch_relax_samples = _build_branch_relax_samples_for_module(
                self,
                str(branch_relax_start_tree_list_path),
                str(branch_relax_target_tree_list_path),
            )
        branch_relax_num_cases = (
            int(getattr(self.model, "first_hit_head_num_cases", 0) or 0)
            or len(self.branch_relax_samples)
            or 1
        )
        self.branch_relax_head = None
        if (
            self.branch_relax_head_weight > 0.0
            or self.branch_relax_head_use_at_sampling
        ):
            self.branch_relax_head = BranchRelaxHead(
                int(self.model.embed_dim),
                int(branch_relax_num_cases),
                case_dim=int(branch_relax_case_dim),
                hidden_dim=int(branch_relax_hidden_dim),
            )
        self._harness_sampling_frozen_start_bank_base = None
        self.sampling_disable_inner_logging = bool(sampling_disable_inner_logging)
        self.sampling_use_inference_mode = bool(sampling_use_inference_mode)
        self._posterior_reference_bundle_cache = {}

    def on_train_start(self):
        super().on_train_start()
        self._reset_training_sampling_schedule()

    def _reset_training_sampling_schedule(self):
        frequency = int(self.training_sampling_frequency)
        if frequency <= 0:
            self._next_training_sample_step = None
            return

        next_step = int(self.training_sampling_start)
        current_step = int(self.global_step)
        if current_step >= next_step:
            missed_intervals = ((current_step - next_step) // frequency) + 1
            next_step += missed_intervals * frequency
        self._next_training_sample_step = int(next_step)

    def _training_sample_due(self):
        frequency = int(self.training_sampling_frequency)
        if frequency <= 0:
            return False
        if self._next_training_sample_step is None:
            self._reset_training_sampling_schedule()
        return int(self.global_step) >= int(self._next_training_sample_step)

    def _advance_training_sampling_schedule(self):
        frequency = int(self.training_sampling_frequency)
        if frequency <= 0 or self._next_training_sample_step is None:
            return
        current_step = int(self.global_step)
        while int(self._next_training_sample_step) <= current_step:
            self._next_training_sample_step += frequency

    def _metric_key_allowed(self, key):
        key = str(key)
        if self.metric_log_exact_keys is None and not self.metric_log_prefixes:
            return True
        if self.metric_log_exact_keys is not None and key in self.metric_log_exact_keys:
            return True
        return any(key.startswith(prefix) for prefix in self.metric_log_prefixes)

    def _filter_metric_dict(self, metrics):
        return {
            key: value
            for key, value in metrics.items()
            if self._metric_key_allowed(key)
        }

    def _log_scalar_filtered(self, key, value, **kwargs):
        if not self._metric_key_allowed(key):
            return
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return
        elif not isinstance(value, numbers.Number):
            return
        self.log(key, value, **kwargs)

    def _wandb_log_filtered(self, metrics, step=None):
        if not self.record or wandb is None:
            return
        filtered = self._filter_metric_dict(metrics)
        if filtered:
            wandb.log(filtered, step=self.stepper if step is None else step)

    def _copy_bank_items(self, items):
        return [
            dict(item) if isinstance(item, dict) else item
            for item in (items or [])
        ]

    def _get_frozen_harness_start_bank_base(self, dataset_split):
        if self._harness_sampling_frozen_start_bank_base is not None:
            return self._copy_bank_items(self._harness_sampling_frozen_start_bank_base)

        base = self._copy_bank_items(
            getattr(dataset_split, "overfit_fixed_pair_start_tree_bank_items", [])
        )
        self._harness_sampling_frozen_start_bank_base = self._copy_bank_items(base)
        return base

    def _sample_overfit_fixed_pair_bank_pair_for_harness(
        self,
        dataset_split,
        *,
        frozen_start_bank=False,
    ):
        if not hasattr(dataset_split, "sample_overfit_fixed_pair_bank_pair"):
            return None
        if not frozen_start_bank:
            return dataset_split.sample_overfit_fixed_pair_bank_pair()

        frozen_bank = self._get_frozen_harness_start_bank_base(dataset_split)
        if not frozen_bank:
            return dataset_split.sample_overfit_fixed_pair_bank_pair()

        current_bank = self._copy_bank_items(
            getattr(dataset_split, "overfit_fixed_pair_start_tree_bank_items", [])
        )
        try:
            dataset_split.set_overfit_fixed_pair_start_tree_bank(frozen_bank)
            return dataset_split.sample_overfit_fixed_pair_bank_pair()
        finally:
            dataset_split.set_overfit_fixed_pair_start_tree_bank(current_bank)

    def on_train_end(self):
        if self.record:
            wandb.finish()

    def training_step(self, batch, _):
        if batch is None:
            logging.warning(
                "Skipping training step: batch is None (tokenization failed for all items)"
            )
            return None

        self.stepper += 1
        opt = self.optimizers()
        opt.zero_grad()

        velocity_training_batch = batch
        autoregressive_training_batch = batch
        control_mode = bool(batch.get("_full_path_control_mode", False))
        probe_parity_joint = bool(
            control_mode and self.training_step_probe_parity_joint_update
        )

        if control_mode:
            full_path_velocity_samples = batch.get("full_path_velocity_samples") or []
            full_path_autoregressive_samples = (
                batch.get("full_path_autoregressive_samples") or []
            )
            if full_path_velocity_samples:
                velocity_training_batch = _build_velocity_replay_batch(
                    self,
                    full_path_velocity_samples,
                )
                velocity_training_batch["_use_full_path_control_velocity_loss"] = True
            if full_path_autoregressive_samples:
                autoregressive_training_batch = _build_autoregressive_replay_batch(
                    self,
                    full_path_autoregressive_samples,
                )

        def _step_optimizer():
            if self.training_step_gradient_clip_val > 0.0:
                self.clip_gradients(
                    opt,
                    gradient_clip_val=self.training_step_gradient_clip_val,
                    gradient_clip_algorithm="norm",
                )
            opt.step()
            opt.zero_grad()

        logs_vel = self.step(
            velocity_training_batch,
            autoregressive=False,
        )
        loss_vel_unscaled = logs_vel["loss"]
        loss_vel_regression_unscaled = logs_vel.get(
            "loss_regression", loss_vel_unscaled
        )
        loss_vel_auxiliary_unscaled = logs_vel.get(
            "loss_auxiliary",
            loss_vel_unscaled - loss_vel_regression_unscaled,
        )
        loss_vel = loss_vel_unscaled * self.training_step_velocity_weight
        velocity_metric_logs = {
            key: value for key, value in logs_vel.items() if key.startswith("velocity/")
        }

        logs = {
            "train/velocity_loss_unscaled": loss_vel_unscaled.detach(),
            "train/velocity_loss_regression_unscaled": (
                loss_vel_regression_unscaled.detach()
            ),
            "train/velocity_loss_auxiliary_unscaled": (
                loss_vel_auxiliary_unscaled.detach()
            ),
            "train/velocity_loss_scaled": loss_vel.detach(),
        }
        logs.update(velocity_metric_logs)

        if self.training_step_separate_optimizer_steps and not probe_parity_joint:
            self.manual_backward(loss_vel)
            _step_optimizer()

        logs_ar = self.step(
            autoregressive_training_batch,
            eval=control_mode,
            autoregressive=True,
        )
        if "loss" not in logs_ar:
            raise RuntimeError("Loss not found in logs for autoregressive head.")

        loss_ar_unscaled = logs_ar["loss"]
        loss_ar = loss_ar_unscaled * self.training_step_autoregressive_weight
        logs_ar["train/autoregressive_loss_unscaled"] = loss_ar_unscaled.detach()
        logs_ar["train/autoregressive_loss_scaled"] = loss_ar.detach()
        logs_ar.update(logs)
        logs = logs_ar

        branch_relax_loss_unscaled, branch_relax_logs = (
            self._branch_relax_training_loss()
        )
        if branch_relax_loss_unscaled is not None:
            branch_relax_loss = branch_relax_loss_unscaled * self.branch_relax_head_weight
            loss_ar = loss_ar + branch_relax_loss
            logs["train/branch_relax_loss_unscaled"] = (
                branch_relax_loss_unscaled.detach()
            )
            logs["train/branch_relax_loss_scaled"] = branch_relax_loss.detach()
            logs.update(branch_relax_logs)

        if probe_parity_joint:
            total_loss = loss_vel + loss_ar
            logs["train/probe_parity_joint_loss_scaled"] = total_loss.detach()
            self.manual_backward(total_loss)
            _step_optimizer()
        elif self.training_step_separate_optimizer_steps:
            total_loss = loss_ar
            self.manual_backward(loss_ar)
            _step_optimizer()
        else:
            total_loss = loss_vel + loss_ar
            self.manual_backward(total_loss)
            _step_optimizer()

        logs["loss"] = total_loss.detach()

        for key, value in logs.items():
            log_value = value
            if torch.is_tensor(log_value):
                log_value = log_value.detach()
            self._log_scalar_filtered(
                key,
                log_value,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )

        chosen_tree = getattr(self.dataset, "chosen_tree", None)
        if chosen_tree is not None:
            _index, sub_tree_size, num_subtrees = chosen_tree
            self._log_scalar_filtered("num_seq_per_subtree", sub_tree_size)
            self._log_scalar_filtered("num_subtrees", num_subtrees)
            logs["num_seq_per_subtree"] = sub_tree_size
            logs["num_subtrees"] = num_subtrees

        base_optimizer = getattr(opt, "optimizer", opt)
        if getattr(base_optimizer, "param_groups", None):
            lr = base_optimizer.param_groups[0]["lr"]
            self._log_scalar_filtered("lr", lr)
            logs["lr"] = lr

        if self.logger_ is not None:
            self.logger_.log(logs, level=logging.INFO)

        if self.record:
            self._wandb_log_filtered(logs, step=self.stepper)
        if (
            not getattr(self.dataset, "msa_distance", False)
            and "norm_rf_distance" in logs
        ):
            self.dataset.update_normrf(logs["norm_rf_distance"])

        self.current_step_value += 1

        if self.lr_scheduler == "cosine":
            sch1 = self.lr_schedulers()
            sch1.step()
        elif self.lr_scheduler == "cosine_warmup":
            sch1, sch2 = self.lr_schedulers()
            if self.num_warmup_steps > 0:
                sch1.step()
                self.num_warmup_steps -= 1
            else:
                sch2.step()
        elif self.lr_scheduler == "warmup":
            sch1 = self.lr_schedulers()
            if self.num_warmup_steps > 0:
                sch1.step()
                self.num_warmup_steps -= 1

        if self._training_sample_due():
            metrics = self.sample_compare_harness(train=True)
            for key, value in metrics.items():
                self._log_scalar_filtered(
                    f"sample_metrics/{key}",
                    value,
                    on_step=True,
                    logger=True,
                )
            self._append_sample_metrics_trace(metrics)
            self._advance_training_sampling_schedule()
            if self.record:
                self._wandb_log_filtered(
                    {f"sample_metrics/{key}": value for key, value in metrics.items()},
                    step=self.stepper,
                )

            rf_norm = metrics.get("rf_norm")
            stop_threshold = self.training_sampling_stop_rf_threshold
            if stop_threshold is None and self.training_sampling_stop_on_zero_rf:
                stop_threshold = 0.0
            if (
                stop_threshold is not None
                and rf_norm is not None
                and float(rf_norm) <= float(stop_threshold)
                and self.trainer is not None
            ):
                logging.info(
                    "Stopping early because sampled rf_norm reached %.6f "
                    "(threshold=%.6f) at global_step=%s",
                    float(rf_norm),
                    float(stop_threshold),
                    self.global_step,
                )
                self.trainer.should_stop = True

        return total_loss

    def validation_step(self, batch, batch_idx):
        pass

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        norms = grad_norm(self, norm_type=2)
        if "grad_2.0_norm_total" in norms:
            total = norms["grad_2.0_norm_total"]
        else:
            total = norms.get("total_grad_norm", 0.0)  # hypothetical fallback
            if total == 0.0:
                # Just take the first key that looks like total if exists
                keys = [k for k in norms.keys() if "total" in k]
                if keys:
                    total = norms[keys[0]]

        # total = norms.get("grad_2.0_norm_total", 0.0)

        layer_norms = {k: v for k, v in norms.items() if "total" not in k}
        if layer_norms:
            max_grad = max(layer_norms.values())
            mean_grad = torch.mean(torch.stack(list(layer_norms.values())))
        else:
            max_grad = 0.0
            mean_grad = 0.0

        self._log_scalar_filtered(
            "grad_norm_max", max_grad, prog_bar=True, on_step=True
        )
        self._log_scalar_filtered(
            "grad_norm_mean", mean_grad, prog_bar=False, on_step=True
        )

        # Print a warning if exploding
        if self.training_step_verbose_logging_enabled and max_grad > 1:
            print(
                f"[Warning] Gradient norm unusually high: max={max_grad:.2e}, mean={mean_grad:.2e}"
            )

        self._log_scalar_filtered("grad_norm_total", total)
        if self.training_step_verbose_logging_enabled:
            print(
                f"step {self.global_step:4d}  total_grad_norm = {total:.2f} mean is {mean_grad:.2f} max is {max_grad:.2f}"
            )
        if self.record:
            self._wandb_log_filtered(
                {
                    "grad/grad_norm_total": total,
                    "grad/grad_norm_max": max_grad,
                    "grad/grad_norm_mean": mean_grad,
                },
                step=self.stepper,
            )

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = optim.AdamW(self.parameters(), lr=self.lr)

        if self.lr_scheduler == "cosine":
            sch1 = CosineAnnealingLR(
                optimizer, T_max=self.num_annealing_steps
            )  # Set to current number of steps for training 7 days
            return [optimizer], [sch1]
        elif self.lr_scheduler == "cosine_warmup":
            sch1 = LinearLR(
                optimizer, start_factor=self.lr, total_iters=self.num_warmup_steps
            )
            sch2 = CosineAnnealingLR(optimizer, T_max=self.num_annealing_steps)
            return [optimizer], [sch1, sch2]
        elif self.lr_scheduler == "warmup":
            sch1 = LinearLR(
                optimizer, start_factor=self.lr, total_iters=self.num_warmup_steps
            )
            return [optimizer], [sch1]
        else:
            scheduler = []
            return optimizer
