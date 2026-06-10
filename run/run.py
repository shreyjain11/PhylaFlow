from model.model import return_model
from data.dataset import PhylaDataModule
import yaml
import sys
from utils.utils import get_possible_ids
from run.TrainingModule import TrainingModule
from utils.random_tree import Tree
import random
import wandb
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning import Trainer
import multiprocessing
import os
import torch
from datetime import datetime

def _expand_config_env_vars(obj):
    """Recursively expand ${ENV_VAR} / $ENV_VAR in all string values of a loaded config.
    Enables portable configs via PHYLAFLOW_DATA_ROOT / PHYLAFLOW_ARTIFACT_ROOT /
    PHYLAFLOW_OUTPUT_ROOT instead of hardcoded absolute paths."""
    if isinstance(obj, dict):
        return {k: _expand_config_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_config_env_vars(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj

import logging

# Global variables to hold the model in worker processes
worker_model = None


class ThresholdStepCheckpoint(Callback):
    """Save once a completed train batch has crossed the next due step.

    Lightning's every_n_train_steps callback can miss all future checkpoints if
    manual optimization/OOM retry logic shifts batch-end global_step off the
    exact modulo. This callback keeps the same cadence but uses >= thresholding.
    """

    def __init__(self, dirpath, every_n_train_steps):
        super().__init__()
        self.dirpath = str(dirpath)
        self.every_n_train_steps = int(every_n_train_steps)
        self._next_step = None

    def _reset_next_step(self, current_step):
        if self.every_n_train_steps <= 0:
            self._next_step = None
            return
        next_step = int(self.every_n_train_steps)
        current_step = int(current_step)
        if current_step >= next_step:
            missed = ((current_step - next_step) // self.every_n_train_steps) + 1
            next_step += missed * self.every_n_train_steps
        self._next_step = int(next_step)

    def on_train_start(self, trainer, pl_module):
        self._reset_next_step(trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n_train_steps <= 0:
            return
        if self._next_step is None:
            self._reset_next_step(trainer.global_step)

        current_step = int(trainer.global_step)
        if current_step < int(self._next_step):
            return

        os.makedirs(self.dirpath, exist_ok=True)
        ckpt_path = os.path.join(
            self.dirpath,
            f"epoch={int(trainer.current_epoch)}-step={current_step:06d}.ckpt",
        )
        if getattr(trainer, "is_global_zero", True) and not os.path.exists(ckpt_path):
            trainer.save_checkpoint(ckpt_path)
            logging.info(
                "Saved threshold checkpoint at global_step=%s to %s",
                current_step,
                ckpt_path,
            )

        while self._next_step is not None and int(self._next_step) <= current_step:
            self._next_step += self.every_n_train_steps


def _birthset_trainer_kwargs(config):
    trainer_cfg = config.get("trainer") or {}
    nested_cfg = {}
    nested_cfg.update(config.get("birthset") or {})
    nested_cfg.update(trainer_cfg.get("birthset") or {})

    def pick(name, nested_name=None, default=None):
        if name in trainer_cfg:
            return trainer_cfg[name]
        if name in config:
            return config[name]
        if nested_name is not None and nested_name in nested_cfg:
            return nested_cfg[nested_name]
        if name in nested_cfg:
            return nested_cfg[name]
        return default

    return {
        "topology_decoder": pick("topology_decoder", default="ar"),
        "birthset_birth_length": pick("birthset_birth_length", "birth_length", 1e-3),
        "birthset_lambda_birth": pick("birthset_lambda_birth", "lambda_birth", 0.2),
        "birthset_lambda_rank": pick("birthset_lambda_rank", "lambda_rank", 0.1),
        "birthset_lambda_proposal": pick(
            "birthset_lambda_proposal",
            "lambda_proposal",
            0.0,
        ),
        "birthset_rank_margin": pick("birthset_rank_margin", "rank_margin", 1.0),
        "birthset_pos_weight": pick("birthset_pos_weight", "pos_weight", "auto"),
        "birthset_use_train_birth_split_bank": pick(
            "birthset_use_train_birth_split_bank",
            "use_train_birth_split_bank",
            True,
        ),
        "birthset_use_small_polytomy_enumeration": pick(
            "birthset_use_small_polytomy_enumeration",
            "use_small_polytomy_enumeration",
            True,
        ),
        "birthset_use_pair_prefix_candidates": pick(
            "birthset_use_pair_prefix_candidates",
            "use_pair_prefix_candidates",
            False,
        ),
        "birthset_use_component_phyla_conditioning": pick(
            "birthset_use_component_phyla_conditioning",
            "use_component_phyla_conditioning",
            False,
        ),
        "birthset_pair_prefix_top_pairs": pick(
            "birthset_pair_prefix_top_pairs",
            "top_pairs",
            64,
        ),
        "birthset_proposal_pair_target_mode": pick(
            "birthset_proposal_pair_target_mode",
            "proposal_pair_target_mode",
            "contained",
        ),
        "birthset_proposal_max_expansion_examples": pick(
            "birthset_proposal_max_expansion_examples",
            "proposal_max_expansion_examples",
            4096,
        ),
        "birthset_proposal_max_order_seed_pairs": pick(
            "birthset_proposal_max_order_seed_pairs",
            "proposal_max_order_seed_pairs",
            128,
        ),
        "birthset_proposal_train_topk": pick(
            "birthset_proposal_train_topk",
            "proposal_train_topk",
            False,
        ),
        "birthset_max_enum_components": pick(
            "birthset_max_enum_components",
            "max_enum_components",
            12,
        ),
        "birthset_max_candidates_per_polytomy": pick(
            "birthset_max_candidates_per_polytomy",
            "max_candidates_per_polytomy",
            2048,
        ),
        "birthset_negatives_per_positive": pick(
            "birthset_negatives_per_positive",
            "negatives_per_positive",
            64,
        ),
        "birthset_decoder": pick("birthset_decoder", "decoder", "greedy"),
        "birthset_beam_width": pick("birthset_beam_width", "beam_width", 8),
        "birthset_fallback": pick("birthset_fallback", "fallback", "ar"),
    }


def _set_global_seed(seed):
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _coerce_config_ids(raw_ids):
    if raw_ids is None:
        return []
    if isinstance(raw_ids, str):
        stripped = raw_ids.strip()
        if not stripped:
            return []
        import re

        range_match = re.fullmatch(r"DS(\d+)\s*-\s*(?:DS)?(\d+)", stripped, re.IGNORECASE)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            step = 1 if end >= start else -1
            return [f"DS{i}" for i in range(start, end + step, step)]
        return [part.strip() for part in stripped.split(",") if part.strip()]
    if isinstance(raw_ids, (list, tuple, set)):
        return [str(item).strip() for item in raw_ids if str(item).strip()]
    return [str(raw_ids).strip()]


def _get_dataset_ids_from_config(config):
    data_cfg = config.get("data", {})
    posterior_ids = _coerce_config_ids(
        data_cfg.get("posterior_dataset_ids", data_cfg.get("short_run_dataset_ids"))
    )
    if not posterior_ids:
        posterior_ids = _coerce_config_ids(
            data_cfg.get("posterior_dataset_id", data_cfg.get("short_run_dataset_id"))
        )
    if posterior_ids:
        return posterior_ids
    return get_possible_ids(data_cfg["nexus_root"])


def _configure_torch_runtime():
    if not torch.cuda.is_available():
        return
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True


def _is_orthomam_config(config):
    data_cfg = config.get("data", {}) or {}
    trainer_cfg = config.get("trainer", {}) or {}
    markers = [
        data_cfg.get("topology_stream_index_jsonl_path"),
        data_cfg.get("nexus_root"),
        data_cfg.get("mrbayes_root"),
        trainer_cfg.get("phyla_precomputed_embeddings_path"),
    ]
    if any("phylaflow_datasets" in str(value) for value in markers if value):
        return True
    ids = _coerce_config_ids(
        data_cfg.get("posterior_dataset_ids", data_cfg.get("short_run_dataset_ids"))
    )
    ids.extend(
        _coerce_config_ids(
            data_cfg.get("posterior_dataset_id", data_cfg.get("short_run_dataset_id"))
        )
    )
    return any("_NT_AL" in dataset_id.upper() for dataset_id in ids)


def _is_ds_dataset_wandb_tag(tag):
    value = str(tag).strip().lower()
    if value == "project_ds":
        return True
    return len(value) > 2 and value.startswith("ds") and value[2].isdigit()


def _normalized_wandb_tags(config, raw_tags):
    tags = [str(tag) for tag in (raw_tags or []) if str(tag).strip()]
    if not _is_orthomam_config(config):
        return tags

    normalized = []
    seen = set()
    for tag in ["orthomam", *tags]:
        if _is_ds_dataset_wandb_tag(tag):
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(tag)
    return normalized


def _init_wandb_run(config, default_project):
    trainer_cfg = config.get("trainer", {})
    wandb_kwargs = {
        "project": trainer_cfg.get("wandb_project", default_project),
        "name": trainer_cfg.get("wandb_name"),
        "group": trainer_cfg.get("wandb_group"),
        "job_type": trainer_cfg.get("wandb_job_type"),
        "notes": trainer_cfg.get("wandb_notes"),
        "tags": _normalized_wandb_tags(config, trainer_cfg.get("wandb_tags")),
        "config": {
            "seed": trainer_cfg.get("seed"),
            "epochs": trainer_cfg.get("epochs"),
            "autoregressive_head_mode": config.get("model", {}).get(
                "autoregressive_head_mode"
            ),
            "autoregressive_group_refinement_layers": config.get(
                "model", {}
            ).get("autoregressive_group_refinement_layers"),
            "autoregressive_max_subset_size": config.get("model", {}).get(
                "autoregressive_max_subset_size"
            ),
            "velocity_first_hit_head_weight": trainer_cfg.get(
                "velocity_first_hit_head_weight"
            ),
            "velocity_first_hit_head_use_at_sampling": trainer_cfg.get(
                "velocity_first_hit_head_use_at_sampling"
            ),
            "velocity_first_hit_predictor_mode": trainer_cfg.get(
                "velocity_first_hit_predictor_mode"
            ),
            "velocity_first_hit_false_positive_mass_weight": trainer_cfg.get(
                "velocity_first_hit_false_positive_mass_weight"
            ),
            "velocity_first_hit_false_negative_mass_weight": trainer_cfg.get(
                "velocity_first_hit_false_negative_mass_weight"
            ),
            "velocity_first_hit_use_geometry_features": trainer_cfg.get(
                "velocity_first_hit_use_geometry_features"
            ),
            "velocity_first_hit_geometry_hidden_dim": trainer_cfg.get(
                "velocity_first_hit_geometry_hidden_dim"
            ),
            "velocity_first_hit_edge_length_hidden_dim": trainer_cfg.get(
                "velocity_first_hit_edge_length_hidden_dim"
            ),
            "velocity_first_hit_attention_layers": trainer_cfg.get(
                "velocity_first_hit_attention_layers"
            ),
            "velocity_first_hit_attention_heads": trainer_cfg.get(
                "velocity_first_hit_attention_heads"
            ),
            "velocity_first_hit_bucket_count": trainer_cfg.get(
                "velocity_first_hit_bucket_count"
            ),
            "velocity_first_hit_bucket_log_min": trainer_cfg.get(
                "velocity_first_hit_bucket_log_min"
            ),
            "velocity_first_hit_bucket_log_max": trainer_cfg.get(
                "velocity_first_hit_bucket_log_max"
            ),
            "velocity_refiner_mode": trainer_cfg.get("velocity_refiner_mode"),
            "velocity_refiner_attention_layers": trainer_cfg.get(
                "velocity_refiner_attention_layers"
            ),
            "velocity_refiner_attention_heads": trainer_cfg.get(
                "velocity_refiner_attention_heads"
            ),
            "velocity_refiner_bucket_count": trainer_cfg.get(
                "velocity_refiner_bucket_count"
            ),
            "velocity_refiner_bucket_log_min": trainer_cfg.get(
                "velocity_refiner_bucket_log_min"
            ),
            "velocity_refiner_bucket_log_max": trainer_cfg.get(
                "velocity_refiner_bucket_log_max"
            ),
            "velocity_boundary_vanish_head_weight": trainer_cfg.get(
                "velocity_boundary_vanish_head_weight"
            ),
            "velocity_boundary_vanish_head_use_at_sampling": trainer_cfg.get(
                "velocity_boundary_vanish_head_use_at_sampling"
            ),
            "velocity_boundary_vanish_one_step_use_at_sampling": trainer_cfg.get(
                "velocity_boundary_vanish_one_step_use_at_sampling"
            ),
            "rollout_replay_velocity_weight": trainer_cfg.get(
                "rollout_replay_velocity_weight"
            ),
            "rollout_replay_autoregressive_weight": trainer_cfg.get(
                "rollout_replay_autoregressive_weight"
            ),
            "rollout_replay_start_step": trainer_cfg.get(
                "rollout_replay_start_step"
            ),
            "rollout_replay_frequency": trainer_cfg.get(
                "rollout_replay_frequency"
            ),
            "rollout_replay_max_velocity_states": trainer_cfg.get(
                "rollout_replay_max_velocity_states"
            ),
            "rollout_replay_max_autoregressive_states": trainer_cfg.get(
                "rollout_replay_max_autoregressive_states"
            ),
            "rollout_replay_max_steps": trainer_cfg.get(
                "rollout_replay_max_steps"
            ),
            "rollout_replay_anchor_states": trainer_cfg.get(
                "rollout_replay_anchor_states"
            ),
            "rollout_replay_oracle_horizon": trainer_cfg.get(
                "rollout_replay_oracle_horizon"
            ),
            "rollout_replay_mode": trainer_cfg.get("rollout_replay_mode"),
            "rollout_replay_bank_max_polytomy_size": trainer_cfg.get(
                "rollout_replay_bank_max_polytomy_size"
            ),
            "sampling_fixed_dt_base": trainer_cfg.get("sampling_fixed_dt_base"),
            "sampling_max_steps": trainer_cfg.get("sampling_max_steps"),
            "sampling_max_events": trainer_cfg.get("sampling_max_events"),
            "sampling_max_autoregressive_merges_per_boundary": trainer_cfg.get(
                "sampling_max_autoregressive_merges_per_boundary"
            ),
            "rollout_replay_fixed_dt_base": trainer_cfg.get(
                "rollout_replay_fixed_dt_base"
            ),
            "rollout_replay_max_events": trainer_cfg.get(
                "rollout_replay_max_events"
            ),
            "rollout_replay_prefix_stop_early": trainer_cfg.get(
                "rollout_replay_prefix_stop_early"
            ),
            "rollout_replay_cache_reuse_every_step": trainer_cfg.get(
                "rollout_replay_cache_reuse_every_step"
            ),
            "rollout_replay_refresh_only_if_better_rf": trainer_cfg.get(
                "rollout_replay_refresh_only_if_better_rf"
            ),
            "rollout_replay_legacy_loss_structure": trainer_cfg.get(
                "rollout_replay_legacy_loss_structure"
            ),
            "dynamic_start_bank_mode": trainer_cfg.get(
                "dynamic_start_bank_mode",
                (
                    "soft_hybrid"
                    if (
                        trainer_cfg.get("analysis_soft_hybrid_best_rf_repeat")
                        is not None
                        or trainer_cfg.get("analysis_soft_hybrid_best_multivel_repeat")
                        is not None
                    )
                    else None
                ),
            ),
            "dynamic_start_bank_min_velocity_states": trainer_cfg.get(
                "dynamic_start_bank_min_velocity_states",
                trainer_cfg.get("analysis_dynamic_start_bank_min_velocity_states"),
            ),
            "dynamic_start_bank_best_rf_repeat": trainer_cfg.get(
                "dynamic_start_bank_best_rf_repeat",
                trainer_cfg.get("analysis_soft_hybrid_best_rf_repeat"),
            ),
            "dynamic_start_bank_best_multivel_repeat": trainer_cfg.get(
                "dynamic_start_bank_best_multivel_repeat",
                trainer_cfg.get("analysis_soft_hybrid_best_multivel_repeat"),
            ),
            "overfit_oracle_prefix_start_prob": config.get("data", {}).get(
                "overfit_oracle_prefix_start_prob",
                config.get("data", {}).get("analysis_oracle_prefix_start_prob"),
            ),
            "overfit_oracle_prefix_max_fraction": config.get("data", {}).get(
                "overfit_oracle_prefix_max_fraction",
                config.get("data", {}).get("analysis_oracle_prefix_max_fraction"),
            ),
            "sampling_disable_inner_logging": trainer_cfg.get(
                "sampling_disable_inner_logging"
            ),
            "sampling_only_first_hit_collapse": trainer_cfg.get(
                "sampling_only_first_hit_collapse"
            ),
            "sampling_actual_event_boundary_use_at_sampling": trainer_cfg.get(
                "sampling_actual_event_boundary_use_at_sampling"
            ),
            "sampling_actual_event_boundary_include_predicted_first_hit": trainer_cfg.get(
                "sampling_actual_event_boundary_include_predicted_first_hit"
            ),
            "velocity_first_hit_sampling_max_edges": trainer_cfg.get(
                "velocity_first_hit_sampling_max_edges"
            ),
            "velocity_first_hit_sampling_fallback_threshold": trainer_cfg.get(
                "velocity_first_hit_sampling_fallback_threshold"
            ),
            "velocity_first_hit_sampling_fallback_top_k": trainer_cfg.get(
                "velocity_first_hit_sampling_fallback_top_k"
            ),
            "legacy_first_hit_gather_only": trainer_cfg.get(
                "legacy_first_hit_gather_only"
            ),
            "sampling_use_top_merge_planner": trainer_cfg.get(
                "sampling_use_top_merge_planner"
            ),
            "sampling_use_inference_mode": trainer_cfg.get(
                "sampling_use_inference_mode"
            ),
            "sampling_cache_tri_mask": trainer_cfg.get(
                "sampling_cache_tri_mask"
            ),
            "sampling_cache_polytomy_groups": trainer_cfg.get(
                "sampling_cache_polytomy_groups"
            ),
            "sampling_cache_autoregressive_state": trainer_cfg.get(
                "sampling_cache_autoregressive_state"
            ),
            "training_step_autoregressive_weight": trainer_cfg.get(
                "training_step_autoregressive_weight"
            ),
            "training_step_velocity_weight": trainer_cfg.get(
                "training_step_velocity_weight"
            ),
            "training_step_autoregressive_grad_ratio": trainer_cfg.get(
                "training_step_autoregressive_grad_ratio"
            ),
            "training_step_separate_optimizer_steps": trainer_cfg.get(
                "training_step_separate_optimizer_steps"
            ),
            "training_step_verbose_logging_enabled": trainer_cfg.get(
                "training_step_verbose_logging_enabled"
            ),
            "optimizer_name": trainer_cfg.get("optimizer_name"),
            "autoregressive_use_time": trainer_cfg.get("autoregressive_use_time"),
            "autoregressive_target_mode": trainer_cfg.get(
                "autoregressive_target_mode"
            ),
            "autoregressive_rollin_prob": trainer_cfg.get(
                "autoregressive_rollin_prob"
            ),
            "autoregressive_dagger_prob": trainer_cfg.get(
                "autoregressive_dagger_prob"
            ),
            "autoregressive_dagger_max_steps": trainer_cfg.get(
                "autoregressive_dagger_max_steps"
            ),
            "autoregressive_structure_perturb_prob": trainer_cfg.get(
                "autoregressive_structure_perturb_prob"
            ),
            "autoregressive_structure_perturb_mode": trainer_cfg.get(
                "autoregressive_structure_perturb_mode"
            ),
            "velocity_length_jitter_prob": trainer_cfg.get(
                "velocity_length_jitter_prob"
            ),
            "velocity_length_jitter_scale": trainer_cfg.get(
                "velocity_length_jitter_scale"
            ),
            "training_sampling_start": trainer_cfg.get("training_sampling_start"),
            "training_sampling_frequency": trainer_cfg.get(
                "training_sampling_frequency"
            ),
            "init_checkpoint_path": trainer_cfg.get("init_checkpoint_path"),
            "resume_ckpt_path": trainer_cfg.get("resume_ckpt_path"),
            "overfit_event_horizon": config.get("data", {}).get(
                "overfit_event_horizon"
            ),
            "overfit_velocity_event_states": config.get("data", {}).get(
                "overfit_velocity_event_states"
            ),
            "overfit_velocity_orthant_start_states": config.get("data", {}).get(
                "overfit_velocity_orthant_start_states"
            ),
            "overfit_velocity_explicit_boundary_end_states": config.get(
                "data", {}
            ).get("overfit_velocity_explicit_boundary_end_states"),
            "overfit_velocity_fixed_timepoints": config.get("data", {}).get(
                "overfit_velocity_fixed_timepoints"
            ),
            "overfit_fixed_pair": config.get("data", {}).get("overfit_fixed_pair"),
            "overfit_split_multi_subset_events": config.get("data", {}).get(
                "overfit_split_multi_subset_events"
            ),
        },
    }
    wandb_dir = trainer_cfg.get("wandb_dir")
    if wandb_dir:
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_kwargs["dir"] = wandb_dir
    wandb_kwargs = {k: v for k, v in wandb_kwargs.items() if v is not None}
    return wandb.init(**wandb_kwargs)


def _load_model_init_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = {
        k[len("model.") :]: v for k, v in state_dict.items() if k.startswith("model.")
    }
    if not model_state:
        model_state = state_dict
    model.load_state_dict(model_state, strict=True)
    return model


def _load_training_module_init_checkpoint(module, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    current_state = module.state_dict()
    loadable = {}
    skipped = []
    for key, value in state_dict.items():
        current_value = current_state.get(key)
        if current_value is None:
            skipped.append(key)
            continue
        if hasattr(value, "shape") and hasattr(current_value, "shape"):
            if tuple(value.shape) != tuple(current_value.shape):
                skipped.append(key)
                continue
        loadable[key] = value
    missing, unexpected = module.load_state_dict(loadable, strict=False)
    print(
        "Initialized TrainingModule weights from checkpoint: "
        f"{checkpoint_path} "
        f"(loaded={len(loadable)}, skipped={len(skipped)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)})"
    )
    if skipped:
        print(f"Skipped checkpoint tensors: {skipped[:8]}")
    return module


def _resolve_checkpoint_paths(trainer_cfg):
    init_checkpoint_path = trainer_cfg.get("init_checkpoint_path")
    resume_ckpt_path = trainer_cfg.get("resume_ckpt_path")
    if init_checkpoint_path and resume_ckpt_path:
        raise ValueError(
            "Use either trainer.init_checkpoint_path or trainer.resume_ckpt_path, not both."
        )
    if init_checkpoint_path:
        init_checkpoint_path = os.path.abspath(init_checkpoint_path)
    if resume_ckpt_path:
        resume_ckpt_path = os.path.abspath(resume_ckpt_path)
    return init_checkpoint_path, resume_ckpt_path


def init_worker(config_file, device_id):
    """
    Initializer for worker processes. Loads the model once.
    """
    global worker_model

    # Silence detailed logs in workers
    logging.getLogger("run.TrainingModule").setLevel(logging.WARNING)
    logging.getLogger("phyla").setLevel(logging.WARNING)

    # Load config
    with open(config_file, "r") as f:
        config = _expand_config_env_vars(yaml.safe_load(f))

    _configure_torch_runtime()
    _set_global_seed(config["trainer"].get("seed"))

    # Initialize Dataset (needed for embeddings calculation in sample)
    ids = _get_dataset_ids_from_config(config)
    ran = random.Random(42)
    ran.shuffle(ids)
    train_ids = ids[: int(0.8 * len(ids))]
    test_ids = ids[int(0.8 * len(ids)) :]

    dataset = PhylaDataModule(config, train_ids=train_ids, test_ids=test_ids)

    # Initialize Model
    phyla_flow = return_model(config)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    phyla_flow.to(device)

    model = TrainingModule(
        model=phyla_flow,
        lr=config["trainer"]["lr"],
        optimizer_name=config["trainer"].get("optimizer_name", "adamw"),
        record=config["trainer"]["record"],
        epochs=config["trainer"]["epochs"],
        dataset=dataset,
        lr_scheduler="default",
        num_annealing_steps=10000,
        num_warmup_steps=1000,
        deepspeed=False,
        logger=None,
        phyla_checkpoint_path=config["trainer"].get("phyla_checkpoint_path"),
        phyla_precomputed_embeddings_path=config["trainer"].get(
            "phyla_precomputed_embeddings_path"
        ),
        live_phyla_checkpoint_path=config["trainer"].get(
            "live_phyla_checkpoint_path"
        ),
        live_phyla_unfreeze=config["trainer"].get("live_phyla_unfreeze", True),
        live_phyla_lr=config["trainer"].get("live_phyla_lr"),
        live_phyla_input_mode=config["trainer"].get(
            "live_phyla_input_mode",
            "raw-full",
        ),
        live_phyla_max_input_tokens=config["trainer"].get(
            "live_phyla_max_input_tokens",
            0,
        ),
        live_phyla_device=config["trainer"].get("live_phyla_device"),
        velocity_loss_mode=config["trainer"].get("velocity_loss_mode", "weighted"),
        velocity_loss_plain_weight=config["trainer"].get(
            "velocity_loss_plain_weight", 0.5
        ),
        velocity_sign_eps=config["trainer"].get("velocity_sign_eps", 1e-3),
        training_step_velocity_weight=config["trainer"].get(
            "training_step_velocity_weight", 1.0
        ),
        training_step_autoregressive_weight=config["trainer"].get(
            "training_step_autoregressive_weight", 1.0
        ),
        training_step_gradient_clip_val=config["trainer"].get(
            "training_step_gradient_clip_val", 1.0
        ),
        grad_norm_log_frequency=config["trainer"].get("grad_norm_log_frequency", 1),
        training_step_profile_frequency=config["trainer"].get(
            "training_step_profile_frequency", 0
        ),
        training_step_profile_warmup_steps=config["trainer"].get(
            "training_step_profile_warmup_steps", 0
        ),
        training_step_profile_sync_cuda=config["trainer"].get(
            "training_step_profile_sync_cuda", True
        ),
        training_step_autoregressive_grad_ratio=config["trainer"].get(
            "training_step_autoregressive_grad_ratio"
        ),
        training_step_separate_optimizer_steps=config["trainer"].get(
            "training_step_separate_optimizer_steps", False
        ),
        training_step_verbose_logging_enabled=config["trainer"].get(
            "training_step_verbose_logging_enabled", False
        ),
        autoregressive_use_time=config["trainer"].get(
            "autoregressive_use_time", False
        ),
        autoregressive_target_mode=config["trainer"].get(
            "autoregressive_target_mode", "scheduled"
        ),
        autoregressive_polytomy_choosing_weight=config["trainer"].get(
            "autoregressive_polytomy_choosing_weight", 1.0
        ),
        autoregressive_stop_after_merge_weight=config["trainer"].get(
            "autoregressive_stop_after_merge_weight", 0.0
        ),
        autoregressive_stop_after_merge_use_at_sampling=config["trainer"].get(
            "autoregressive_stop_after_merge_use_at_sampling", False
        ),
        autoregressive_rollin_prob=config["trainer"].get(
            "autoregressive_rollin_prob", 0.0
        ),
        autoregressive_dagger_prob=config["trainer"].get(
            "autoregressive_dagger_prob", 0.0
        ),
        autoregressive_dagger_max_steps=config["trainer"].get(
            "autoregressive_dagger_max_steps", 4
        ),
        autoregressive_structure_perturb_prob=config["trainer"].get(
            "autoregressive_structure_perturb_prob", 0.0
        ),
        autoregressive_structure_perturb_mode=config["trainer"].get(
            "autoregressive_structure_perturb_mode", "random_wrong_pair"
        ),
        **_birthset_trainer_kwargs(config),
        velocity_length_jitter_prob=config["trainer"].get(
            "velocity_length_jitter_prob", 0.0
        ),
        velocity_length_jitter_scale=config["trainer"].get(
            "velocity_length_jitter_scale", 0.0
        ),
        velocity_dt_candidate_weight=config["trainer"].get(
            "velocity_dt_candidate_weight", 0.0
        ),
        velocity_dt_hit_weight=config["trainer"].get("velocity_dt_hit_weight", 0.0),
        velocity_logtau_all_weight=config["trainer"].get(
            "velocity_logtau_all_weight", 0.0
        ),
        velocity_logtau_first_over_weight=config["trainer"].get(
            "velocity_logtau_first_over_weight", 0.0
        ),
        velocity_logtau_first_tie_weight=config["trainer"].get(
            "velocity_logtau_first_tie_weight", 0.0
        ),
        velocity_dt_eps=config["trainer"].get("velocity_dt_eps", 1e-6),
        velocity_event_weight=config["trainer"].get("velocity_event_weight", 0.5),
        velocity_event_temp=config["trainer"].get("velocity_event_temp", 0.5),
        velocity_event_rate_beta=config["trainer"].get("velocity_event_rate_beta", 5.0),
        velocity_event_normalize_by_log_candidates=config["trainer"].get(
            "velocity_event_normalize_by_log_candidates", True
        ),
        velocity_event_precision_weight=config["trainer"].get(
            "velocity_event_precision_weight", 0.0
        ),
        velocity_event_precision_margin=config["trainer"].get(
            "velocity_event_precision_margin", 0.0
        ),
        velocity_first_hit_head_weight=config["trainer"].get(
            "velocity_first_hit_head_weight", 0.0
        ),
        velocity_first_hit_loss_tol=config["trainer"].get(
            "velocity_first_hit_loss_tol", 0.01
        ),
        velocity_first_hit_head_use_at_sampling=config["trainer"].get(
            "velocity_first_hit_head_use_at_sampling", False
        ),
        velocity_first_hit_predictor_mode=config["trainer"].get(
            "velocity_first_hit_predictor_mode", "base"
        ),
        velocity_first_hit_false_positive_mass_weight=config["trainer"].get(
            "velocity_first_hit_false_positive_mass_weight", 0.0
        ),
        velocity_first_hit_false_negative_mass_weight=config["trainer"].get(
            "velocity_first_hit_false_negative_mass_weight", 0.0
        ),
        velocity_first_hit_use_geometry_features=config["trainer"].get(
            "velocity_first_hit_use_geometry_features", False
        ),
        velocity_first_hit_geometry_hidden_dim=config["trainer"].get(
            "velocity_first_hit_geometry_hidden_dim", 32
        ),
        velocity_first_hit_edge_length_hidden_dim=config["trainer"].get(
            "velocity_first_hit_edge_length_hidden_dim", 64
        ),
        velocity_first_hit_attention_layers=config["trainer"].get(
            "velocity_first_hit_attention_layers", 1
        ),
        velocity_first_hit_attention_heads=config["trainer"].get(
            "velocity_first_hit_attention_heads", 4
        ),
        velocity_first_hit_bucket_count=config["trainer"].get(
            "velocity_first_hit_bucket_count", 32
        ),
        velocity_first_hit_bucket_log_min=config["trainer"].get(
            "velocity_first_hit_bucket_log_min", -8.0
        ),
        velocity_first_hit_bucket_log_max=config["trainer"].get(
            "velocity_first_hit_bucket_log_max", 1.0
        ),
        velocity_refiner_mode=config["trainer"].get(
            "velocity_refiner_mode", "base"
        ),
        velocity_refiner_attention_layers=config["trainer"].get(
            "velocity_refiner_attention_layers", 1
        ),
        velocity_refiner_attention_heads=config["trainer"].get(
            "velocity_refiner_attention_heads", 4
        ),
        velocity_refiner_bucket_count=config["trainer"].get(
            "velocity_refiner_bucket_count", 32
        ),
        velocity_refiner_bucket_log_min=config["trainer"].get(
            "velocity_refiner_bucket_log_min", -8.0
        ),
        velocity_refiner_bucket_log_max=config["trainer"].get(
            "velocity_refiner_bucket_log_max", 1.0
        ),
        velocity_boundary_vanish_head_weight=config["trainer"].get(
            "velocity_boundary_vanish_head_weight", 0.0
        ),
        velocity_boundary_vanish_head_use_at_sampling=config["trainer"].get(
            "velocity_boundary_vanish_head_use_at_sampling", False
        ),
        velocity_boundary_vanish_one_step_use_at_sampling=config["trainer"].get(
            "velocity_boundary_vanish_one_step_use_at_sampling", False
        ),
        velocity_boundary_time_head_weight=config["trainer"].get(
            "velocity_boundary_time_head_weight", 0.0
        ),
        velocity_boundary_time_head_use_at_sampling=config["trainer"].get(
            "velocity_boundary_time_head_use_at_sampling", False
        ),
        velocity_boundary_time_hidden_dim=config["trainer"].get(
            "velocity_boundary_time_hidden_dim", 64
        ),
        velocity_terminal_head_weight=config["trainer"].get(
            "velocity_terminal_head_weight", 0.0
        ),
        velocity_terminal_head_use_at_sampling=config["trainer"].get(
            "velocity_terminal_head_use_at_sampling", False
        ),
        velocity_terminal_head_sampling_action=config["trainer"].get(
            "velocity_terminal_head_sampling_action", "after_phase"
        ),
        velocity_terminal_head_hidden_dim=config["trainer"].get(
            "velocity_terminal_head_hidden_dim", 64
        ),
        velocity_terminal_head_probe_features=config["trainer"].get(
            "velocity_terminal_head_probe_features", False
        ),
        velocity_terminal_head_input_mode=config["trainer"].get(
            "velocity_terminal_head_input_mode"
        ),
        velocity_terminal_head_use_case_adapt=config["trainer"].get(
            "velocity_terminal_head_use_case_adapt", False
        ),
        velocity_terminal_head_balance_loss=config["trainer"].get(
            "velocity_terminal_head_balance_loss", False
        ),
        velocity_terminal_head_topology_pool=config["trainer"].get(
            "velocity_terminal_head_topology_pool", "mean"
        ),
        velocity_probe_direct_set_loss=config["trainer"].get(
            "velocity_probe_direct_set_loss", False
        ),
        velocity_probe_direct_set_anchor_only=config["trainer"].get(
            "velocity_probe_direct_set_anchor_only", False
        ),
        velocity_probe_direct_set_include_base_samples=config["trainer"].get(
            "velocity_probe_direct_set_include_base_samples", False
        ),
        velocity_probe_direct_set_target_negative_weight=config["trainer"].get(
            "velocity_probe_direct_set_target_negative_weight", 1.0
        ),
        velocity_probe_direct_set_nontarget_nonnegative_weight=config["trainer"].get(
            "velocity_probe_direct_set_nontarget_nonnegative_weight", 0.0
        ),
        velocity_probe_direct_set_positive_reweight=config["trainer"].get(
            "velocity_probe_direct_set_positive_reweight", False
        ),
        velocity_probe_direct_set_positive_reweight_power=config["trainer"].get(
            "velocity_probe_direct_set_positive_reweight_power", 1.0
        ),
        velocity_probe_direct_set_positive_reweight_max=config["trainer"].get(
            "velocity_probe_direct_set_positive_reweight_max"
        ),
        velocity_probe_direct_set_bce_weight=config["trainer"].get(
            "velocity_probe_direct_set_bce_weight", 1.0
        ),
        velocity_probe_direct_set_loss_weight=config["trainer"].get(
            "velocity_probe_direct_set_loss_weight", 1.0
        ),
        velocity_probe_direct_set_mse_weight=config["trainer"].get(
            "velocity_probe_direct_set_mse_weight", 0.0
        ),
        training_step_probe_parity_joint_update=config["trainer"].get(
            "training_step_probe_parity_joint_update", False
        ),
        training_step_joint_tokenize_velocity_ar=config["trainer"].get(
            "training_step_joint_tokenize_velocity_ar", False
        ),
        training_step_full_path_replay_initial_retry_attempt=config["trainer"].get(
            "training_step_full_path_replay_initial_retry_attempt", 0
        ),
        skip_repeated_no_valid_boundary_use_at_sampling=config["trainer"].get(
            "skip_repeated_no_valid_boundary_use_at_sampling", False
        ),
        sampling_discrete_phase_rollout_use_at_sampling=config["trainer"].get(
            "sampling_discrete_phase_rollout_use_at_sampling", False
        ),
        sampling_discrete_phase_exact_boundary_step_use_at_sampling=config["trainer"].get(
            "sampling_discrete_phase_exact_boundary_step_use_at_sampling", False
        ),
        sampling_discrete_phase_max_phases=config["trainer"].get(
            "sampling_discrete_phase_max_phases", 8
        ),
        sampling_final_orthant_relax_use_at_sampling=config["trainer"].get(
            "sampling_final_orthant_relax_use_at_sampling", False
        ),
        sampling_final_orthant_relax_steps=config["trainer"].get(
            "sampling_final_orthant_relax_steps", 0
        ),
        sampling_final_orthant_relax_total_time=config["trainer"].get(
            "sampling_final_orthant_relax_total_time", 1.0
        ),
        sampling_final_orthant_relax_time_mode=config["trainer"].get(
            "sampling_final_orthant_relax_time_mode", "local"
        ),
        sampling_final_orthant_relax_edge_floor=config["trainer"].get(
            "sampling_final_orthant_relax_edge_floor"
        ),
        training_sampling_mode=config["trainer"].get(
            "training_sampling_mode", "batch_compare"
        ),
        training_sampling_dt_base=config["trainer"].get(
            "training_sampling_dt_base", 0.02
        ),
        sampling_fixed_dt_base=config["trainer"].get("sampling_fixed_dt_base"),
        sampling_max_steps=config["trainer"].get("sampling_max_steps", 256),
        sampling_max_events=config["trainer"].get("sampling_max_events"),
        sampling_max_autoregressive_merges_per_boundary=config["trainer"].get(
            "sampling_max_autoregressive_merges_per_boundary", -1
        ),
        sampling_disable_inner_logging=config["trainer"].get(
            "sampling_disable_inner_logging", True
        ),
        sampling_only_first_hit_collapse=config["trainer"].get(
            "sampling_only_first_hit_collapse", False
        ),
        sampling_actual_event_boundary_use_at_sampling=config["trainer"].get(
            "sampling_actual_event_boundary_use_at_sampling", False
        ),
        sampling_actual_event_boundary_include_predicted_first_hit=config["trainer"].get(
            "sampling_actual_event_boundary_include_predicted_first_hit", False
        ),
        sampling_predsim_overrun_use_at_sampling=config["trainer"].get(
            "sampling_predsim_overrun_use_at_sampling", False
        ),
        sampling_random_fixed_pair_bank_use_at_sampling=config["trainer"].get(
            "sampling_random_fixed_pair_bank_use_at_sampling", False
        ),
        velocity_first_hit_sampling_max_edges=config["trainer"].get(
            "velocity_first_hit_sampling_max_edges", -1
        ),
        velocity_first_hit_sampling_fallback_threshold=config["trainer"].get(
            "velocity_first_hit_sampling_fallback_threshold", -1
        ),
        velocity_first_hit_sampling_fallback_top_k=config["trainer"].get(
            "velocity_first_hit_sampling_fallback_top_k", -1
        ),
        sampling_use_top_merge_planner=config["trainer"].get(
            "sampling_use_top_merge_planner", False
        ),
        sampling_use_inference_mode=config["trainer"].get(
            "sampling_use_inference_mode", False
        ),
        sampling_cache_tri_mask=config["trainer"].get(
            "sampling_cache_tri_mask", False
        ),
        sampling_cache_polytomy_groups=config["trainer"].get(
            "sampling_cache_polytomy_groups", False
        ),
        sampling_cache_autoregressive_state=config["trainer"].get(
            "sampling_cache_autoregressive_state", False
        ),
        training_sampling_stop_on_zero_rf=config["trainer"].get(
            "training_sampling_stop_on_zero_rf", False
        ),
        training_sampling_stop_rf_threshold=config["trainer"].get(
            "training_sampling_stop_rf_threshold"
        ),
        sample_metrics_trace_path=config["trainer"].get("sample_metrics_trace_path"),
        sample_metrics_num_pairs=config["trainer"].get("sample_metrics_num_pairs", 1),
        sample_metrics_trace_topology_repeats_enabled=config["trainer"].get(
            "sample_metrics_trace_topology_repeats_enabled", False
        ),
        sample_metrics_unseen_start_eval=config["trainer"].get(
            "sample_metrics_unseen_start_eval", False
        ),
        sample_metrics_zero_shot_random_start_eval=config["trainer"].get(
            "sample_metrics_zero_shot_random_start_eval", False
        ),
        sample_metrics_unseen_start_seed=config["trainer"].get(
            "sample_metrics_unseen_start_seed", 20260430
        ),
        sample_metrics_unseen_start_metric_encoder_path=config["trainer"].get(
            "sample_metrics_unseen_start_metric_encoder_path"
        ),
        sample_metrics_unseen_pair_selection_mode=config["trainer"].get(
            "sample_metrics_unseen_pair_selection_mode", "random_bank"
        ),
        sample_metrics_unseen_start_max_duplicate_tries=config["trainer"].get(
            "sample_metrics_unseen_start_max_duplicate_tries", 100
        ),
        sample_metrics_relaxed_likelihood_enabled=config["trainer"].get(
            "sample_metrics_relaxed_likelihood_enabled", False
        ),
        sample_metrics_branch_relaxer_checkpoint_path=config["trainer"].get(
            "sample_metrics_branch_relaxer_checkpoint_path"
        ),
        sample_metrics_mrbayes20k_enabled=config["trainer"].get(
            "sample_metrics_mrbayes20k_enabled", False
        ),
        sample_metrics_mrbayes20k_num_starts=config["trainer"].get(
            "sample_metrics_mrbayes20k_num_starts", 64
        ),
        sample_metrics_mrbayes20k_ngen=config["trainer"].get(
            "sample_metrics_mrbayes20k_ngen", 20000
        ),
        sample_metrics_mrbayes20k_samplefreq=config["trainer"].get(
            "sample_metrics_mrbayes20k_samplefreq", 200
        ),
        sample_metrics_mrbayes20k_printfreq=config["trainer"].get(
            "sample_metrics_mrbayes20k_printfreq", 5000
        ),
        sample_metrics_mrbayes20k_max_workers=config["trainer"].get(
            "sample_metrics_mrbayes20k_max_workers", 12
        ),
        sample_metrics_mrbayes20k_timeout_sec=config["trainer"].get(
            "sample_metrics_mrbayes20k_timeout_sec", 1800
        ),
        sample_metrics_mrbayes20k_dataset_pickle_path=config["trainer"].get(
            "sample_metrics_mrbayes20k_dataset_pickle_path"
        ),
        sample_metrics_mrbayes20k_golden_root=config["trainer"].get(
            "sample_metrics_mrbayes20k_golden_root"
        ),
        sample_metrics_mrbayes20k_work_root=config["trainer"].get(
            "sample_metrics_mrbayes20k_work_root",
            "/tmp/phylaflow_sample_metrics_mrbayes20k",
        ),
        sample_metrics_mrbayes20k_output_dir=config["trainer"].get(
            "sample_metrics_mrbayes20k_output_dir"
        ),
        sample_metrics_mrbayes20k_bin=config["trainer"].get(
            "sample_metrics_mrbayes20k_bin",
            "/opt/conda/envs/phylaflow-mrbayes/bin/mb",
        ),
        sample_metrics_tree_dump_enabled=config["trainer"].get(
            "sample_metrics_tree_dump_enabled", False
        ),
        sample_metrics_tree_dump_dir=config["trainer"].get(
            "sample_metrics_tree_dump_dir"
        ),
        sample_metrics_checkpoint_enabled=config["trainer"].get(
            "sample_metrics_checkpoint_enabled", True
        ),
        sample_metrics_checkpoint_dir=config["trainer"].get(
            "sample_metrics_checkpoint_dir"
        ),
        metric_log_exact_keys=config["trainer"].get("metric_log_exact_keys"),
        metric_log_prefixes=config["trainer"].get("metric_log_prefixes"),
        branch_relax_head_weight=config["trainer"].get("branch_relax_head_weight", 0.0),
        branch_relax_head_use_at_sampling=config["trainer"].get(
            "branch_relax_head_use_at_sampling", False
        ),
        branch_relax_start_tree_list_path=config["trainer"].get(
            "branch_relax_start_tree_list_path"
        ),
        branch_relax_target_tree_list_path=config["trainer"].get(
            "branch_relax_target_tree_list_path"
        ),
        branch_relax_detach_trunk=config["trainer"].get(
            "branch_relax_detach_trunk", True
        ),
        branch_relax_batch_size=config["trainer"].get("branch_relax_batch_size", 1),
        branch_relax_case_dim=config["trainer"].get("branch_relax_case_dim", 64),
        branch_relax_hidden_dim=config["trainer"].get("branch_relax_hidden_dim", 256),
        branch_relax_likelihood_dataset_id=config["trainer"].get(
            "branch_relax_likelihood_dataset_id"
        ),
        branch_relax_likelihood_metric_enabled=config["trainer"].get(
            "branch_relax_likelihood_metric_enabled", False
        ),
        rollout_replay_velocity_weight=config["trainer"].get(
            "rollout_replay_velocity_weight", 0.0
        ),
        rollout_replay_autoregressive_weight=config["trainer"].get(
            "rollout_replay_autoregressive_weight", 0.0
        ),
        rollout_replay_start_step=config["trainer"].get(
            "rollout_replay_start_step", 0
        ),
        rollout_replay_frequency=config["trainer"].get(
            "rollout_replay_frequency", 1
        ),
        rollout_replay_max_velocity_states=config["trainer"].get(
            "rollout_replay_max_velocity_states", 0
        ),
        rollout_replay_max_autoregressive_states=config["trainer"].get(
            "rollout_replay_max_autoregressive_states", 0
        ),
        rollout_replay_max_steps=config["trainer"].get(
            "rollout_replay_max_steps", 256
        ),
        rollout_replay_max_events=config["trainer"].get(
            "rollout_replay_max_events"
        ),
        rollout_replay_anchor_states=config["trainer"].get(
            "rollout_replay_anchor_states", 4
        ),
        rollout_replay_oracle_horizon=config["trainer"].get(
            "rollout_replay_oracle_horizon", 2
        ),
        rollout_replay_mode=config["trainer"].get(
            "rollout_replay_mode", "anchor_oracle"
        ),
        rollout_replay_pairwise_max_group_size=config["trainer"].get(
            "rollout_replay_pairwise_max_group_size", 0
        ),
        rollout_replay_bank_max_polytomy_size=config["trainer"].get(
            "rollout_replay_bank_max_polytomy_size", -1
        ),
        rollout_replay_topology_repeat_cap=config["trainer"].get(
            "rollout_replay_topology_repeat_cap", 0
        ),
        rollout_replay_dump_refreshes=config["trainer"].get(
            "rollout_replay_dump_refreshes", False
        ),
        rollout_replay_dump_dir=config["trainer"].get(
            "rollout_replay_dump_dir"
        ),
        rollout_replay_fixed_dt_base=config["trainer"].get(
            "rollout_replay_fixed_dt_base"
        ),
        rollout_replay_prefix_stop_early=config["trainer"].get(
            "rollout_replay_prefix_stop_early", False
        ),
        rollout_replay_cache_reuse_every_step=config["trainer"].get(
            "rollout_replay_cache_reuse_every_step", True
        ),
        rollout_replay_refresh_only_if_better_rf=config["trainer"].get(
            "rollout_replay_refresh_only_if_better_rf", False
        ),
        rollout_replay_legacy_loss_structure=config["trainer"].get(
            "rollout_replay_legacy_loss_structure", False
        ),
        rollout_replay_velocity_use_pair_oracle_orthant_labels=config["trainer"].get(
            "rollout_replay_velocity_use_pair_oracle_orthant_labels", False
        ),
        dynamic_start_bank_enabled=config["trainer"].get(
            "dynamic_start_bank_enabled", False
        ),
        dynamic_start_bank_start_step=config["trainer"].get(
            "dynamic_start_bank_start_step", 0
        ),
        dynamic_start_bank_max_entries=config["trainer"].get(
            "dynamic_start_bank_max_entries", 2
        ),
        dynamic_start_bank_min_rf_improvement=config["trainer"].get(
            "dynamic_start_bank_min_rf_improvement", 0.0
        ),
        dynamic_start_bank_max_polytomy_size=config["trainer"].get(
            "dynamic_start_bank_max_polytomy_size", -1
        ),
        dynamic_start_bank_mode=config["trainer"].get(
            "dynamic_start_bank_mode",
            (
                "soft_hybrid"
                if (
                    config["trainer"].get("analysis_soft_hybrid_best_rf_repeat")
                    is not None
                    or config["trainer"].get(
                        "analysis_soft_hybrid_best_multivel_repeat"
                    )
                    is not None
                )
                else "best_start"
            ),
        ),
        dynamic_start_bank_min_velocity_states=config["trainer"].get(
            "dynamic_start_bank_min_velocity_states",
            config["trainer"].get(
                "analysis_dynamic_start_bank_min_velocity_states", 2
            ),
        ),
        dynamic_start_bank_best_rf_repeat=config["trainer"].get(
            "dynamic_start_bank_best_rf_repeat",
            config["trainer"].get("analysis_soft_hybrid_best_rf_repeat", 18),
        ),
        dynamic_start_bank_best_multivel_repeat=config["trainer"].get(
            "dynamic_start_bank_best_multivel_repeat",
            config["trainer"].get(
                "analysis_soft_hybrid_best_multivel_repeat", 9
            ),
        ),
        dynamic_start_bank_trace_path=config["trainer"].get(
            "dynamic_start_bank_trace_path"
        ),
        dynamic_start_bank_artifact_dir=config["trainer"].get(
            "dynamic_start_bank_artifact_dir"
        ),
        dynamic_start_bank_save_improved_checkpoint=config["trainer"].get(
            "dynamic_start_bank_save_improved_checkpoint", False
        ),
    )
    model.legacy_first_hit_gather_only = bool(
        config["trainer"].get("legacy_first_hit_gather_only", False)
    )
    model.to(device)
    model.eval()

    worker_model = model


def sample_worker_task(tree_list):
    """
    Task function called by pool.map. Uses the global worker_model.
    """
    global worker_model
    # dt_base set to 0.01 to force approx 100 steps (T=1.0)
    return worker_model.sample(tree_list, None, num_samples=1, dt_base=0.01)


def run_test():
    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        config = _expand_config_env_vars(yaml.safe_load(f))

    _configure_torch_runtime()
    _set_global_seed(config["trainer"].get("seed"))

    ids = _get_dataset_ids_from_config(config)
    # Random 80-20 train-test split for now
    ran = random.Random(42)
    ran.shuffle(ids)
    train_ids = ids[: int(0.8 * len(ids))]
    test_ids = ids[int(0.8 * len(ids)) :]
    ###TEMPORARY FOR DEBUGGING
    train_ids = test_ids

    dataset = PhylaDataModule(config, train_ids=train_ids, test_ids=test_ids)
    one = dataset.dataset_train[0]
    two = dataset.dataset_train[0]
    batch = dataset.collate_fn([one, two])

    phyla_flow = return_model(config)

    model = TrainingModule(
        model=phyla_flow,
        lr=config["trainer"]["lr"],
        optimizer_name=config["trainer"].get("optimizer_name", "adamw"),
        record=config["trainer"]["record"],
        epochs=config["trainer"]["epochs"],
        dataset=dataset,
        lr_scheduler="default",
        num_annealing_steps=10000,
        num_warmup_steps=1000,
        deepspeed=False,
        logger=None,
        phyla_checkpoint_path=config["trainer"].get("phyla_checkpoint_path"),
        phyla_precomputed_embeddings_path=config["trainer"].get(
            "phyla_precomputed_embeddings_path"
        ),
        live_phyla_checkpoint_path=config["trainer"].get(
            "live_phyla_checkpoint_path"
        ),
        live_phyla_unfreeze=config["trainer"].get("live_phyla_unfreeze", True),
        live_phyla_lr=config["trainer"].get("live_phyla_lr"),
        live_phyla_input_mode=config["trainer"].get(
            "live_phyla_input_mode",
            "raw-full",
        ),
        live_phyla_max_input_tokens=config["trainer"].get(
            "live_phyla_max_input_tokens",
            0,
        ),
        live_phyla_device=config["trainer"].get("live_phyla_device"),
        velocity_loss_mode=config["trainer"].get("velocity_loss_mode", "weighted"),
        velocity_loss_plain_weight=config["trainer"].get(
            "velocity_loss_plain_weight", 0.5
        ),
        velocity_sign_eps=config["trainer"].get("velocity_sign_eps", 1e-3),
        training_step_velocity_weight=config["trainer"].get(
            "training_step_velocity_weight", 1.0
        ),
        training_step_autoregressive_weight=config["trainer"].get(
            "training_step_autoregressive_weight", 1.0
        ),
        training_step_gradient_clip_val=config["trainer"].get(
            "training_step_gradient_clip_val", 1.0
        ),
        grad_norm_log_frequency=config["trainer"].get("grad_norm_log_frequency", 1),
        training_step_profile_frequency=config["trainer"].get(
            "training_step_profile_frequency", 0
        ),
        training_step_profile_warmup_steps=config["trainer"].get(
            "training_step_profile_warmup_steps", 0
        ),
        training_step_profile_sync_cuda=config["trainer"].get(
            "training_step_profile_sync_cuda", True
        ),
        training_step_autoregressive_grad_ratio=config["trainer"].get(
            "training_step_autoregressive_grad_ratio"
        ),
        training_step_separate_optimizer_steps=config["trainer"].get(
            "training_step_separate_optimizer_steps", False
        ),
        training_step_verbose_logging_enabled=config["trainer"].get(
            "training_step_verbose_logging_enabled", False
        ),
        autoregressive_use_time=config["trainer"].get(
            "autoregressive_use_time", False
        ),
        autoregressive_target_mode=config["trainer"].get(
            "autoregressive_target_mode", "scheduled"
        ),
        autoregressive_polytomy_choosing_weight=config["trainer"].get(
            "autoregressive_polytomy_choosing_weight", 1.0
        ),
        autoregressive_stop_after_merge_weight=config["trainer"].get(
            "autoregressive_stop_after_merge_weight", 0.0
        ),
        autoregressive_stop_after_merge_use_at_sampling=config["trainer"].get(
            "autoregressive_stop_after_merge_use_at_sampling", False
        ),
        autoregressive_rollin_prob=config["trainer"].get(
            "autoregressive_rollin_prob", 0.0
        ),
        autoregressive_dagger_prob=config["trainer"].get(
            "autoregressive_dagger_prob", 0.0
        ),
        autoregressive_dagger_max_steps=config["trainer"].get(
            "autoregressive_dagger_max_steps", 4
        ),
        autoregressive_structure_perturb_prob=config["trainer"].get(
            "autoregressive_structure_perturb_prob", 0.0
        ),
        autoregressive_structure_perturb_mode=config["trainer"].get(
            "autoregressive_structure_perturb_mode", "random_wrong_pair"
        ),
        **_birthset_trainer_kwargs(config),
        velocity_length_jitter_prob=config["trainer"].get(
            "velocity_length_jitter_prob", 0.0
        ),
        velocity_length_jitter_scale=config["trainer"].get(
            "velocity_length_jitter_scale", 0.0
        ),
        velocity_dt_candidate_weight=config["trainer"].get(
            "velocity_dt_candidate_weight", 0.0
        ),
        velocity_dt_hit_weight=config["trainer"].get("velocity_dt_hit_weight", 0.0),
        velocity_logtau_all_weight=config["trainer"].get(
            "velocity_logtau_all_weight", 0.0
        ),
        velocity_logtau_first_over_weight=config["trainer"].get(
            "velocity_logtau_first_over_weight", 0.0
        ),
        velocity_logtau_first_tie_weight=config["trainer"].get(
            "velocity_logtau_first_tie_weight", 0.0
        ),
        velocity_dt_eps=config["trainer"].get("velocity_dt_eps", 1e-6),
        velocity_event_weight=config["trainer"].get("velocity_event_weight", 0.5),
        velocity_event_temp=config["trainer"].get("velocity_event_temp", 0.5),
        velocity_event_rate_beta=config["trainer"].get("velocity_event_rate_beta", 5.0),
        velocity_event_normalize_by_log_candidates=config["trainer"].get(
            "velocity_event_normalize_by_log_candidates", True
        ),
        velocity_event_precision_weight=config["trainer"].get(
            "velocity_event_precision_weight", 0.0
        ),
        velocity_event_precision_margin=config["trainer"].get(
            "velocity_event_precision_margin", 0.0
        ),
        velocity_first_hit_head_weight=config["trainer"].get(
            "velocity_first_hit_head_weight", 0.0
        ),
        velocity_first_hit_loss_tol=config["trainer"].get(
            "velocity_first_hit_loss_tol", 0.01
        ),
        velocity_first_hit_head_use_at_sampling=config["trainer"].get(
            "velocity_first_hit_head_use_at_sampling", False
        ),
        velocity_first_hit_predictor_mode=config["trainer"].get(
            "velocity_first_hit_predictor_mode", "base"
        ),
        velocity_first_hit_false_positive_mass_weight=config["trainer"].get(
            "velocity_first_hit_false_positive_mass_weight", 0.0
        ),
        velocity_first_hit_false_negative_mass_weight=config["trainer"].get(
            "velocity_first_hit_false_negative_mass_weight", 0.0
        ),
        velocity_refiner_mode=config["trainer"].get(
            "velocity_refiner_mode", "base"
        ),
        velocity_first_hit_use_geometry_features=config["trainer"].get(
            "velocity_first_hit_use_geometry_features", False
        ),
        velocity_first_hit_geometry_hidden_dim=config["trainer"].get(
            "velocity_first_hit_geometry_hidden_dim", 32
        ),
        velocity_first_hit_edge_length_hidden_dim=config["trainer"].get(
            "velocity_first_hit_edge_length_hidden_dim", 64
        ),
        velocity_refiner_attention_layers=config["trainer"].get(
            "velocity_refiner_attention_layers", 1
        ),
        velocity_refiner_attention_heads=config["trainer"].get(
            "velocity_refiner_attention_heads", 4
        ),
        velocity_refiner_bucket_count=config["trainer"].get(
            "velocity_refiner_bucket_count", 32
        ),
        velocity_refiner_bucket_log_min=config["trainer"].get(
            "velocity_refiner_bucket_log_min", -8.0
        ),
        velocity_refiner_bucket_log_max=config["trainer"].get(
            "velocity_refiner_bucket_log_max", 1.0
        ),
        velocity_boundary_vanish_head_weight=config["trainer"].get(
            "velocity_boundary_vanish_head_weight", 0.0
        ),
        velocity_boundary_vanish_head_use_at_sampling=config["trainer"].get(
            "velocity_boundary_vanish_head_use_at_sampling", False
        ),
        velocity_boundary_vanish_one_step_use_at_sampling=config["trainer"].get(
            "velocity_boundary_vanish_one_step_use_at_sampling", False
        ),
        velocity_boundary_time_head_weight=config["trainer"].get(
            "velocity_boundary_time_head_weight", 0.0
        ),
        velocity_boundary_time_head_use_at_sampling=config["trainer"].get(
            "velocity_boundary_time_head_use_at_sampling", False
        ),
        velocity_boundary_time_hidden_dim=config["trainer"].get(
            "velocity_boundary_time_hidden_dim", 64
        ),
        velocity_terminal_head_weight=config["trainer"].get(
            "velocity_terminal_head_weight", 0.0
        ),
        velocity_terminal_head_use_at_sampling=config["trainer"].get(
            "velocity_terminal_head_use_at_sampling", False
        ),
        velocity_terminal_head_sampling_action=config["trainer"].get(
            "velocity_terminal_head_sampling_action", "after_phase"
        ),
        velocity_terminal_head_hidden_dim=config["trainer"].get(
            "velocity_terminal_head_hidden_dim", 64
        ),
        velocity_terminal_head_probe_features=config["trainer"].get(
            "velocity_terminal_head_probe_features", False
        ),
        velocity_terminal_head_input_mode=config["trainer"].get(
            "velocity_terminal_head_input_mode"
        ),
        velocity_terminal_head_use_case_adapt=config["trainer"].get(
            "velocity_terminal_head_use_case_adapt", False
        ),
        velocity_terminal_head_balance_loss=config["trainer"].get(
            "velocity_terminal_head_balance_loss", False
        ),
        velocity_terminal_head_topology_pool=config["trainer"].get(
            "velocity_terminal_head_topology_pool", "mean"
        ),
        velocity_probe_direct_set_loss=config["trainer"].get(
            "velocity_probe_direct_set_loss", False
        ),
        velocity_probe_direct_set_anchor_only=config["trainer"].get(
            "velocity_probe_direct_set_anchor_only", False
        ),
        velocity_probe_direct_set_include_base_samples=config["trainer"].get(
            "velocity_probe_direct_set_include_base_samples", False
        ),
        velocity_probe_direct_set_target_negative_weight=config["trainer"].get(
            "velocity_probe_direct_set_target_negative_weight", 1.0
        ),
        velocity_probe_direct_set_nontarget_nonnegative_weight=config["trainer"].get(
            "velocity_probe_direct_set_nontarget_nonnegative_weight", 0.0
        ),
        velocity_probe_direct_set_positive_reweight=config["trainer"].get(
            "velocity_probe_direct_set_positive_reweight", False
        ),
        velocity_probe_direct_set_positive_reweight_power=config["trainer"].get(
            "velocity_probe_direct_set_positive_reweight_power", 1.0
        ),
        velocity_probe_direct_set_positive_reweight_max=config["trainer"].get(
            "velocity_probe_direct_set_positive_reweight_max"
        ),
        velocity_probe_direct_set_bce_weight=config["trainer"].get(
            "velocity_probe_direct_set_bce_weight", 1.0
        ),
        velocity_probe_direct_set_loss_weight=config["trainer"].get(
            "velocity_probe_direct_set_loss_weight", 1.0
        ),
        velocity_probe_direct_set_mse_weight=config["trainer"].get(
            "velocity_probe_direct_set_mse_weight", 0.0
        ),
        training_step_probe_parity_joint_update=config["trainer"].get(
            "training_step_probe_parity_joint_update", False
        ),
        training_step_joint_tokenize_velocity_ar=config["trainer"].get(
            "training_step_joint_tokenize_velocity_ar", False
        ),
        training_step_full_path_replay_initial_retry_attempt=config["trainer"].get(
            "training_step_full_path_replay_initial_retry_attempt", 0
        ),
        skip_repeated_no_valid_boundary_use_at_sampling=config["trainer"].get(
            "skip_repeated_no_valid_boundary_use_at_sampling", False
        ),
        sampling_discrete_phase_rollout_use_at_sampling=config["trainer"].get(
            "sampling_discrete_phase_rollout_use_at_sampling", False
        ),
        sampling_discrete_phase_exact_boundary_step_use_at_sampling=config["trainer"].get(
            "sampling_discrete_phase_exact_boundary_step_use_at_sampling", False
        ),
        sampling_discrete_phase_max_phases=config["trainer"].get(
            "sampling_discrete_phase_max_phases", 8
        ),
        sampling_final_orthant_relax_use_at_sampling=config["trainer"].get(
            "sampling_final_orthant_relax_use_at_sampling", False
        ),
        sampling_final_orthant_relax_steps=config["trainer"].get(
            "sampling_final_orthant_relax_steps", 0
        ),
        sampling_final_orthant_relax_total_time=config["trainer"].get(
            "sampling_final_orthant_relax_total_time", 1.0
        ),
        sampling_final_orthant_relax_time_mode=config["trainer"].get(
            "sampling_final_orthant_relax_time_mode", "local"
        ),
        sampling_final_orthant_relax_edge_floor=config["trainer"].get(
            "sampling_final_orthant_relax_edge_floor"
        ),
        training_sampling_mode=config["trainer"].get(
            "training_sampling_mode", "batch_compare"
        ),
        training_sampling_dt_base=config["trainer"].get(
            "training_sampling_dt_base", 0.02
        ),
        sampling_fixed_dt_base=config["trainer"].get("sampling_fixed_dt_base"),
        sampling_max_steps=config["trainer"].get("sampling_max_steps", 256),
        sampling_max_events=config["trainer"].get("sampling_max_events"),
        sampling_max_autoregressive_merges_per_boundary=config["trainer"].get(
            "sampling_max_autoregressive_merges_per_boundary", -1
        ),
        sampling_disable_inner_logging=config["trainer"].get(
            "sampling_disable_inner_logging", True
        ),
        sampling_only_first_hit_collapse=config["trainer"].get(
            "sampling_only_first_hit_collapse", False
        ),
        sampling_actual_event_boundary_use_at_sampling=config["trainer"].get(
            "sampling_actual_event_boundary_use_at_sampling", False
        ),
        sampling_actual_event_boundary_include_predicted_first_hit=config["trainer"].get(
            "sampling_actual_event_boundary_include_predicted_first_hit", False
        ),
        sampling_predsim_overrun_use_at_sampling=config["trainer"].get(
            "sampling_predsim_overrun_use_at_sampling", False
        ),
        sampling_random_fixed_pair_bank_use_at_sampling=config["trainer"].get(
            "sampling_random_fixed_pair_bank_use_at_sampling", False
        ),
        velocity_first_hit_sampling_max_edges=config["trainer"].get(
            "velocity_first_hit_sampling_max_edges", -1
        ),
        velocity_first_hit_sampling_fallback_threshold=config["trainer"].get(
            "velocity_first_hit_sampling_fallback_threshold", -1
        ),
        velocity_first_hit_sampling_fallback_top_k=config["trainer"].get(
            "velocity_first_hit_sampling_fallback_top_k", -1
        ),
        sampling_use_top_merge_planner=config["trainer"].get(
            "sampling_use_top_merge_planner", False
        ),
        sampling_use_inference_mode=config["trainer"].get(
            "sampling_use_inference_mode", False
        ),
        sampling_cache_tri_mask=config["trainer"].get(
            "sampling_cache_tri_mask", False
        ),
        sampling_cache_polytomy_groups=config["trainer"].get(
            "sampling_cache_polytomy_groups", False
        ),
        sampling_cache_autoregressive_state=config["trainer"].get(
            "sampling_cache_autoregressive_state", False
        ),
        training_sampling_stop_on_zero_rf=config["trainer"].get(
            "training_sampling_stop_on_zero_rf", False
        ),
        training_sampling_stop_rf_threshold=config["trainer"].get(
            "training_sampling_stop_rf_threshold"
        ),
        sample_metrics_trace_path=config["trainer"].get("sample_metrics_trace_path"),
        sample_metrics_num_pairs=config["trainer"].get("sample_metrics_num_pairs", 1),
        sample_metrics_trace_topology_repeats_enabled=config["trainer"].get(
            "sample_metrics_trace_topology_repeats_enabled", False
        ),
        sample_metrics_unseen_start_eval=config["trainer"].get(
            "sample_metrics_unseen_start_eval", False
        ),
        sample_metrics_zero_shot_random_start_eval=config["trainer"].get(
            "sample_metrics_zero_shot_random_start_eval", False
        ),
        sample_metrics_unseen_start_seed=config["trainer"].get(
            "sample_metrics_unseen_start_seed", 20260430
        ),
        sample_metrics_unseen_start_metric_encoder_path=config["trainer"].get(
            "sample_metrics_unseen_start_metric_encoder_path"
        ),
        sample_metrics_unseen_pair_selection_mode=config["trainer"].get(
            "sample_metrics_unseen_pair_selection_mode", "random_bank"
        ),
        sample_metrics_unseen_start_max_duplicate_tries=config["trainer"].get(
            "sample_metrics_unseen_start_max_duplicate_tries", 100
        ),
        sample_metrics_relaxed_likelihood_enabled=config["trainer"].get(
            "sample_metrics_relaxed_likelihood_enabled", False
        ),
        sample_metrics_branch_relaxer_checkpoint_path=config["trainer"].get(
            "sample_metrics_branch_relaxer_checkpoint_path"
        ),
        sample_metrics_mrbayes20k_enabled=config["trainer"].get(
            "sample_metrics_mrbayes20k_enabled", False
        ),
        sample_metrics_mrbayes20k_num_starts=config["trainer"].get(
            "sample_metrics_mrbayes20k_num_starts", 64
        ),
        sample_metrics_mrbayes20k_ngen=config["trainer"].get(
            "sample_metrics_mrbayes20k_ngen", 20000
        ),
        sample_metrics_mrbayes20k_samplefreq=config["trainer"].get(
            "sample_metrics_mrbayes20k_samplefreq", 200
        ),
        sample_metrics_mrbayes20k_printfreq=config["trainer"].get(
            "sample_metrics_mrbayes20k_printfreq", 5000
        ),
        sample_metrics_mrbayes20k_max_workers=config["trainer"].get(
            "sample_metrics_mrbayes20k_max_workers", 12
        ),
        sample_metrics_mrbayes20k_timeout_sec=config["trainer"].get(
            "sample_metrics_mrbayes20k_timeout_sec", 1800
        ),
        sample_metrics_mrbayes20k_dataset_pickle_path=config["trainer"].get(
            "sample_metrics_mrbayes20k_dataset_pickle_path"
        ),
        sample_metrics_mrbayes20k_golden_root=config["trainer"].get(
            "sample_metrics_mrbayes20k_golden_root"
        ),
        sample_metrics_mrbayes20k_work_root=config["trainer"].get(
            "sample_metrics_mrbayes20k_work_root",
            "/tmp/phylaflow_sample_metrics_mrbayes20k",
        ),
        sample_metrics_mrbayes20k_output_dir=config["trainer"].get(
            "sample_metrics_mrbayes20k_output_dir"
        ),
        sample_metrics_mrbayes20k_bin=config["trainer"].get(
            "sample_metrics_mrbayes20k_bin",
            "/opt/conda/envs/phylaflow-mrbayes/bin/mb",
        ),
        sample_metrics_tree_dump_enabled=config["trainer"].get(
            "sample_metrics_tree_dump_enabled", False
        ),
        sample_metrics_tree_dump_dir=config["trainer"].get(
            "sample_metrics_tree_dump_dir"
        ),
        sample_metrics_checkpoint_enabled=config["trainer"].get(
            "sample_metrics_checkpoint_enabled", True
        ),
        sample_metrics_checkpoint_dir=config["trainer"].get(
            "sample_metrics_checkpoint_dir"
        ),
        metric_log_exact_keys=config["trainer"].get("metric_log_exact_keys"),
        metric_log_prefixes=config["trainer"].get("metric_log_prefixes"),
        branch_relax_head_weight=config["trainer"].get("branch_relax_head_weight", 0.0),
        branch_relax_head_use_at_sampling=config["trainer"].get(
            "branch_relax_head_use_at_sampling", False
        ),
        branch_relax_start_tree_list_path=config["trainer"].get(
            "branch_relax_start_tree_list_path"
        ),
        branch_relax_target_tree_list_path=config["trainer"].get(
            "branch_relax_target_tree_list_path"
        ),
        branch_relax_detach_trunk=config["trainer"].get(
            "branch_relax_detach_trunk", True
        ),
        branch_relax_batch_size=config["trainer"].get("branch_relax_batch_size", 1),
        branch_relax_case_dim=config["trainer"].get("branch_relax_case_dim", 64),
        branch_relax_hidden_dim=config["trainer"].get("branch_relax_hidden_dim", 256),
        branch_relax_likelihood_dataset_id=config["trainer"].get(
            "branch_relax_likelihood_dataset_id"
        ),
        branch_relax_likelihood_metric_enabled=config["trainer"].get(
            "branch_relax_likelihood_metric_enabled", False
        ),
        rollout_replay_velocity_weight=config["trainer"].get(
            "rollout_replay_velocity_weight", 0.0
        ),
        rollout_replay_autoregressive_weight=config["trainer"].get(
            "rollout_replay_autoregressive_weight", 0.0
        ),
        rollout_replay_start_step=config["trainer"].get(
            "rollout_replay_start_step", 0
        ),
        rollout_replay_frequency=config["trainer"].get(
            "rollout_replay_frequency", 1
        ),
        rollout_replay_max_velocity_states=config["trainer"].get(
            "rollout_replay_max_velocity_states", 0
        ),
        rollout_replay_max_autoregressive_states=config["trainer"].get(
            "rollout_replay_max_autoregressive_states", 0
        ),
        rollout_replay_max_steps=config["trainer"].get(
            "rollout_replay_max_steps", 256
        ),
        rollout_replay_max_events=config["trainer"].get(
            "rollout_replay_max_events"
        ),
        rollout_replay_anchor_states=config["trainer"].get(
            "rollout_replay_anchor_states", 4
        ),
        rollout_replay_oracle_horizon=config["trainer"].get(
            "rollout_replay_oracle_horizon", 2
        ),
        rollout_replay_mode=config["trainer"].get(
            "rollout_replay_mode", "anchor_oracle"
        ),
        rollout_replay_pairwise_max_group_size=config["trainer"].get(
            "rollout_replay_pairwise_max_group_size", 0
        ),
        rollout_replay_bank_max_polytomy_size=config["trainer"].get(
            "rollout_replay_bank_max_polytomy_size", -1
        ),
        rollout_replay_topology_repeat_cap=config["trainer"].get(
            "rollout_replay_topology_repeat_cap", 0
        ),
        rollout_replay_dump_refreshes=config["trainer"].get(
            "rollout_replay_dump_refreshes", False
        ),
        rollout_replay_dump_dir=config["trainer"].get(
            "rollout_replay_dump_dir"
        ),
        rollout_replay_fixed_dt_base=config["trainer"].get(
            "rollout_replay_fixed_dt_base"
        ),
        rollout_replay_prefix_stop_early=config["trainer"].get(
            "rollout_replay_prefix_stop_early", False
        ),
        rollout_replay_cache_reuse_every_step=config["trainer"].get(
            "rollout_replay_cache_reuse_every_step", True
        ),
        rollout_replay_refresh_only_if_better_rf=config["trainer"].get(
            "rollout_replay_refresh_only_if_better_rf", False
        ),
        rollout_replay_legacy_loss_structure=config["trainer"].get(
            "rollout_replay_legacy_loss_structure", False
        ),
        rollout_replay_velocity_use_pair_oracle_orthant_labels=config["trainer"].get(
            "rollout_replay_velocity_use_pair_oracle_orthant_labels", False
        ),
        dynamic_start_bank_enabled=config["trainer"].get(
            "dynamic_start_bank_enabled", False
        ),
        dynamic_start_bank_start_step=config["trainer"].get(
            "dynamic_start_bank_start_step", 0
        ),
        dynamic_start_bank_max_entries=config["trainer"].get(
            "dynamic_start_bank_max_entries", 2
        ),
        dynamic_start_bank_min_rf_improvement=config["trainer"].get(
            "dynamic_start_bank_min_rf_improvement", 0.0
        ),
        dynamic_start_bank_max_polytomy_size=config["trainer"].get(
            "dynamic_start_bank_max_polytomy_size", -1
        ),
        dynamic_start_bank_mode=config["trainer"].get(
            "dynamic_start_bank_mode",
            (
                "soft_hybrid"
                if (
                    config["trainer"].get("analysis_soft_hybrid_best_rf_repeat")
                    is not None
                    or config["trainer"].get(
                        "analysis_soft_hybrid_best_multivel_repeat"
                    )
                    is not None
                )
                else "best_start"
            ),
        ),
        dynamic_start_bank_min_velocity_states=config["trainer"].get(
            "dynamic_start_bank_min_velocity_states",
            config["trainer"].get(
                "analysis_dynamic_start_bank_min_velocity_states", 2
            ),
        ),
        dynamic_start_bank_best_rf_repeat=config["trainer"].get(
            "dynamic_start_bank_best_rf_repeat",
            config["trainer"].get("analysis_soft_hybrid_best_rf_repeat", 18),
        ),
        dynamic_start_bank_best_multivel_repeat=config["trainer"].get(
            "dynamic_start_bank_best_multivel_repeat",
            config["trainer"].get(
                "analysis_soft_hybrid_best_multivel_repeat", 9
            ),
        ),
        dynamic_start_bank_trace_path=config["trainer"].get(
            "dynamic_start_bank_trace_path"
        ),
        dynamic_start_bank_artifact_dir=config["trainer"].get(
            "dynamic_start_bank_artifact_dir"
        ),
        dynamic_start_bank_save_improved_checkpoint=config["trainer"].get(
            "dynamic_start_bank_save_improved_checkpoint", False
        ),
    )
    model.legacy_first_hit_gather_only = bool(
        config["trainer"].get("legacy_first_hit_gather_only", False)
    )
    # res = model(batch['tokenized_trees'], batch['batched_time'], batch['phyla_embeddings'])
    # This fails now btw non-autoregressive LOL NEED TO FIX!

    # Initialize wandb (mock) or disable logging in step for test
    import wandb

    wandb.init(mode="disabled")

    res = model.step(batch)

    # This works below
    # res = model.step(batch, autoregressive=True)
    rt = Tree(num_leaves=50, random=True)
    import time

    start = time.time()
    num_trees = 100
    trees_to_sample = []
    print(f"Generating {num_trees} trees...")
    for _ in range(num_trees):
        trees_to_sample.append(str(Tree(num_leaves=50, random=True)))

    print("Sampling with Multiprocessing (Initialized Workers)...")

    # Split into chunks of size 2
    batch_size_per_worker = 2
    chunks = [
        trees_to_sample[i : i + batch_size_per_worker]
        for i in range(0, len(trees_to_sample), batch_size_per_worker)
    ]

    # We use 0 as device_id assuming single GPU
    # Use 'spawn' context usually safer for CUDA
    ctx = multiprocessing.get_context("spawn")
    num_workers = min(4, os.cpu_count())

    pool_start = time.time()
    try:
        # Pass initializer to set up model ONCE per worker
        with ctx.Pool(
            num_workers, initializer=init_worker, initargs=(config_file, 0)
        ) as pool:
            results = pool.map(sample_worker_task, chunks)
    except Exception as e:
        print(f"Multiprocessing failed: {e}. Falling back to sequential.")
        final_tree = model.sample(trees_to_sample, None, num_samples=1, dt_base=0.1)
        results = [final_tree]
    pool_end = time.time()
    print(f"Pool execution time: {pool_end - pool_start}")

    # Flatten results
    final_tree = []
    for r in results:
        final_tree.extend(r)

    # final_tree = model.sample(trees_to_sample, None, num_samples=1, dt_base=0.1)
    res = time.time() - start

    print("Sampling time:", res)
    print(
        f"Sampling time for a million trees in seconds:",
        res * 1e6 / num_trees,
        " in days:",
        res * 1e6 / num_trees / 86400,
    )
    print(
        f"Sampling time for a thousand trees in seconds:",
        res * 1e3 / num_trees,
        " in minutes:",
        res * 1e3 / num_trees / 60,
    )


def run_overfit():
    # Get first command line argument as config file
    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        config = _expand_config_env_vars(yaml.safe_load(f))

    _configure_torch_runtime()
    _set_global_seed(config["trainer"].get("seed"))

    # --- Start Overrides ---
    print("Running OVERFIT mode (Single Tree)")
    config["data"]["batch_size"] = 1
    config["data"]["num_workers"] = 0
    # config["trainer"]["epochs"] = 10000  # Train for a long time
    config["trainer"]["val_callback_freq"] = 0
    # config["trainer"]["record"] = True # Optional: force recording
    # --- End Overrides ---

    ids = _get_dataset_ids_from_config(config)

    if not ids:
        print("No IDs found!")
        return

    # Override IDs to just one (deterministic)
    single_id = sorted(ids)[0]
    train_ids = [single_id]
    test_ids = [single_id]
    print(f"Overfitting on ID: {single_id}")

    dataset = PhylaDataModule(config, train_ids=train_ids, test_ids=test_ids)

    phyla_flow = return_model(config)
    init_checkpoint_path, resume_ckpt_path = _resolve_checkpoint_paths(
        config["trainer"]
    )
    if init_checkpoint_path:
        print(f"Initializing model weights from checkpoint: {init_checkpoint_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        phyla_flow = _load_model_init_checkpoint(
            phyla_flow, init_checkpoint_path, device
        )
    if resume_ckpt_path:
        print(f"Resuming full trainer state from checkpoint: {resume_ckpt_path}")

    model = TrainingModule(
        model=phyla_flow,
        lr=config["trainer"]["lr"],
        optimizer_name=config["trainer"].get("optimizer_name", "adamw"),
        record=config["trainer"]["record"],
        epochs=config["trainer"]["epochs"],
        dataset=dataset,
        lr_scheduler="default",
        num_annealing_steps=10000,
        num_warmup_steps=config["trainer"].get("num_warmup_steps", 1000),
        num_samples=1,
        deepspeed=False,
        logger=None,
        phyla_checkpoint_path=config["trainer"].get("phyla_checkpoint_path"),
        phyla_precomputed_embeddings_path=config["trainer"].get(
            "phyla_precomputed_embeddings_path"
        ),
        live_phyla_checkpoint_path=config["trainer"].get(
            "live_phyla_checkpoint_path"
        ),
        live_phyla_unfreeze=config["trainer"].get("live_phyla_unfreeze", True),
        live_phyla_lr=config["trainer"].get("live_phyla_lr"),
        live_phyla_input_mode=config["trainer"].get(
            "live_phyla_input_mode",
            "raw-full",
        ),
        live_phyla_max_input_tokens=config["trainer"].get(
            "live_phyla_max_input_tokens",
            0,
        ),
        live_phyla_device=config["trainer"].get("live_phyla_device"),
        velocity_loss_mode=config["trainer"].get("velocity_loss_mode", "weighted"),
        velocity_loss_plain_weight=config["trainer"].get(
            "velocity_loss_plain_weight", 0.5
        ),
        velocity_sign_eps=config["trainer"].get("velocity_sign_eps", 1e-3),
        training_step_velocity_weight=config["trainer"].get(
            "training_step_velocity_weight", 1.0
        ),
        training_step_autoregressive_weight=config["trainer"].get(
            "training_step_autoregressive_weight", 1.0
        ),
        training_step_gradient_clip_val=config["trainer"].get(
            "training_step_gradient_clip_val", 1.0
        ),
        grad_norm_log_frequency=config["trainer"].get("grad_norm_log_frequency", 1),
        training_step_profile_frequency=config["trainer"].get(
            "training_step_profile_frequency", 0
        ),
        training_step_profile_warmup_steps=config["trainer"].get(
            "training_step_profile_warmup_steps", 0
        ),
        training_step_profile_sync_cuda=config["trainer"].get(
            "training_step_profile_sync_cuda", True
        ),
        training_step_autoregressive_grad_ratio=config["trainer"].get(
            "training_step_autoregressive_grad_ratio"
        ),
        training_step_separate_optimizer_steps=config["trainer"].get(
            "training_step_separate_optimizer_steps", False
        ),
        training_step_verbose_logging_enabled=config["trainer"].get(
            "training_step_verbose_logging_enabled", False
        ),
        autoregressive_use_time=config["trainer"].get(
            "autoregressive_use_time", False
        ),
        autoregressive_target_mode=config["trainer"].get(
            "autoregressive_target_mode", "scheduled"
        ),
        autoregressive_polytomy_choosing_weight=config["trainer"].get(
            "autoregressive_polytomy_choosing_weight", 1.0
        ),
        autoregressive_stop_after_merge_weight=config["trainer"].get(
            "autoregressive_stop_after_merge_weight", 0.0
        ),
        autoregressive_stop_after_merge_use_at_sampling=config["trainer"].get(
            "autoregressive_stop_after_merge_use_at_sampling", False
        ),
        autoregressive_rollin_prob=config["trainer"].get(
            "autoregressive_rollin_prob", 0.0
        ),
        autoregressive_dagger_prob=config["trainer"].get(
            "autoregressive_dagger_prob", 0.0
        ),
        autoregressive_dagger_max_steps=config["trainer"].get(
            "autoregressive_dagger_max_steps", 4
        ),
        autoregressive_structure_perturb_prob=config["trainer"].get(
            "autoregressive_structure_perturb_prob", 0.0
        ),
        autoregressive_structure_perturb_mode=config["trainer"].get(
            "autoregressive_structure_perturb_mode", "random_wrong_pair"
        ),
        **_birthset_trainer_kwargs(config),
        velocity_length_jitter_prob=config["trainer"].get(
            "velocity_length_jitter_prob", 0.0
        ),
        velocity_length_jitter_scale=config["trainer"].get(
            "velocity_length_jitter_scale", 0.0
        ),
        velocity_dt_candidate_weight=config["trainer"].get(
            "velocity_dt_candidate_weight", 0.0
        ),
        velocity_dt_hit_weight=config["trainer"].get("velocity_dt_hit_weight", 0.0),
        velocity_logtau_all_weight=config["trainer"].get(
            "velocity_logtau_all_weight", 0.0
        ),
        velocity_logtau_first_over_weight=config["trainer"].get(
            "velocity_logtau_first_over_weight", 0.0
        ),
        velocity_logtau_first_tie_weight=config["trainer"].get(
            "velocity_logtau_first_tie_weight", 0.0
        ),
        velocity_dt_eps=config["trainer"].get("velocity_dt_eps", 1e-6),
        velocity_event_weight=config["trainer"].get("velocity_event_weight", 0.5),
        velocity_event_temp=config["trainer"].get("velocity_event_temp", 0.5),
        velocity_event_rate_beta=config["trainer"].get("velocity_event_rate_beta", 5.0),
        velocity_event_normalize_by_log_candidates=config["trainer"].get(
            "velocity_event_normalize_by_log_candidates", True
        ),
        velocity_event_precision_weight=config["trainer"].get(
            "velocity_event_precision_weight", 0.0
        ),
        velocity_event_precision_margin=config["trainer"].get(
            "velocity_event_precision_margin", 0.0
        ),
        velocity_first_hit_head_weight=config["trainer"].get(
            "velocity_first_hit_head_weight", 0.0
        ),
        velocity_first_hit_loss_tol=config["trainer"].get(
            "velocity_first_hit_loss_tol", 0.01
        ),
        velocity_first_hit_head_use_at_sampling=config["trainer"].get(
            "velocity_first_hit_head_use_at_sampling", False
        ),
        velocity_first_hit_predictor_mode=config["trainer"].get(
            "velocity_first_hit_predictor_mode", "base"
        ),
        velocity_first_hit_false_positive_mass_weight=config["trainer"].get(
            "velocity_first_hit_false_positive_mass_weight", 0.0
        ),
        velocity_first_hit_false_negative_mass_weight=config["trainer"].get(
            "velocity_first_hit_false_negative_mass_weight", 0.0
        ),
        velocity_refiner_mode=config["trainer"].get(
            "velocity_refiner_mode", "base"
        ),
        velocity_first_hit_use_geometry_features=config["trainer"].get(
            "velocity_first_hit_use_geometry_features", False
        ),
        velocity_first_hit_geometry_hidden_dim=config["trainer"].get(
            "velocity_first_hit_geometry_hidden_dim", 32
        ),
        velocity_first_hit_edge_length_hidden_dim=config["trainer"].get(
            "velocity_first_hit_edge_length_hidden_dim", 64
        ),
        velocity_refiner_attention_layers=config["trainer"].get(
            "velocity_refiner_attention_layers", 1
        ),
        velocity_refiner_attention_heads=config["trainer"].get(
            "velocity_refiner_attention_heads", 4
        ),
        velocity_refiner_bucket_count=config["trainer"].get(
            "velocity_refiner_bucket_count", 32
        ),
        velocity_refiner_bucket_log_min=config["trainer"].get(
            "velocity_refiner_bucket_log_min", -8.0
        ),
        velocity_refiner_bucket_log_max=config["trainer"].get(
            "velocity_refiner_bucket_log_max", 1.0
        ),
        velocity_boundary_vanish_head_weight=config["trainer"].get(
            "velocity_boundary_vanish_head_weight", 0.0
        ),
        velocity_boundary_vanish_head_use_at_sampling=config["trainer"].get(
            "velocity_boundary_vanish_head_use_at_sampling", False
        ),
        velocity_boundary_vanish_one_step_use_at_sampling=config["trainer"].get(
            "velocity_boundary_vanish_one_step_use_at_sampling", False
        ),
        velocity_boundary_time_head_weight=config["trainer"].get(
            "velocity_boundary_time_head_weight", 0.0
        ),
        velocity_boundary_time_head_use_at_sampling=config["trainer"].get(
            "velocity_boundary_time_head_use_at_sampling", False
        ),
        velocity_boundary_time_hidden_dim=config["trainer"].get(
            "velocity_boundary_time_hidden_dim", 64
        ),
        velocity_terminal_head_weight=config["trainer"].get(
            "velocity_terminal_head_weight", 0.0
        ),
        velocity_terminal_head_use_at_sampling=config["trainer"].get(
            "velocity_terminal_head_use_at_sampling", False
        ),
        velocity_terminal_head_sampling_action=config["trainer"].get(
            "velocity_terminal_head_sampling_action", "after_phase"
        ),
        velocity_terminal_head_hidden_dim=config["trainer"].get(
            "velocity_terminal_head_hidden_dim", 64
        ),
        velocity_terminal_head_probe_features=config["trainer"].get(
            "velocity_terminal_head_probe_features", False
        ),
        velocity_terminal_head_input_mode=config["trainer"].get(
            "velocity_terminal_head_input_mode"
        ),
        velocity_terminal_head_use_case_adapt=config["trainer"].get(
            "velocity_terminal_head_use_case_adapt", False
        ),
        velocity_terminal_head_balance_loss=config["trainer"].get(
            "velocity_terminal_head_balance_loss", False
        ),
        velocity_terminal_head_topology_pool=config["trainer"].get(
            "velocity_terminal_head_topology_pool", "mean"
        ),
        velocity_probe_direct_set_loss=config["trainer"].get(
            "velocity_probe_direct_set_loss", False
        ),
        velocity_probe_direct_set_anchor_only=config["trainer"].get(
            "velocity_probe_direct_set_anchor_only", False
        ),
        velocity_probe_direct_set_include_base_samples=config["trainer"].get(
            "velocity_probe_direct_set_include_base_samples", False
        ),
        velocity_probe_direct_set_target_negative_weight=config["trainer"].get(
            "velocity_probe_direct_set_target_negative_weight", 1.0
        ),
        velocity_probe_direct_set_nontarget_nonnegative_weight=config["trainer"].get(
            "velocity_probe_direct_set_nontarget_nonnegative_weight", 0.0
        ),
        velocity_probe_direct_set_positive_reweight=config["trainer"].get(
            "velocity_probe_direct_set_positive_reweight", False
        ),
        velocity_probe_direct_set_positive_reweight_power=config["trainer"].get(
            "velocity_probe_direct_set_positive_reweight_power", 1.0
        ),
        velocity_probe_direct_set_positive_reweight_max=config["trainer"].get(
            "velocity_probe_direct_set_positive_reweight_max"
        ),
        velocity_probe_direct_set_bce_weight=config["trainer"].get(
            "velocity_probe_direct_set_bce_weight", 1.0
        ),
        velocity_probe_direct_set_loss_weight=config["trainer"].get(
            "velocity_probe_direct_set_loss_weight", 1.0
        ),
        velocity_probe_direct_set_mse_weight=config["trainer"].get(
            "velocity_probe_direct_set_mse_weight", 0.0
        ),
        training_step_probe_parity_joint_update=config["trainer"].get(
            "training_step_probe_parity_joint_update", False
        ),
        training_step_joint_tokenize_velocity_ar=config["trainer"].get(
            "training_step_joint_tokenize_velocity_ar", False
        ),
        training_step_full_path_replay_initial_retry_attempt=config["trainer"].get(
            "training_step_full_path_replay_initial_retry_attempt", 0
        ),
        skip_repeated_no_valid_boundary_use_at_sampling=config["trainer"].get(
            "skip_repeated_no_valid_boundary_use_at_sampling", False
        ),
        sampling_discrete_phase_rollout_use_at_sampling=config["trainer"].get(
            "sampling_discrete_phase_rollout_use_at_sampling", False
        ),
        sampling_discrete_phase_exact_boundary_step_use_at_sampling=config["trainer"].get(
            "sampling_discrete_phase_exact_boundary_step_use_at_sampling", False
        ),
        sampling_discrete_phase_max_phases=config["trainer"].get(
            "sampling_discrete_phase_max_phases", 8
        ),
        sampling_final_orthant_relax_use_at_sampling=config["trainer"].get(
            "sampling_final_orthant_relax_use_at_sampling", False
        ),
        sampling_final_orthant_relax_steps=config["trainer"].get(
            "sampling_final_orthant_relax_steps", 0
        ),
        sampling_final_orthant_relax_total_time=config["trainer"].get(
            "sampling_final_orthant_relax_total_time", 1.0
        ),
        sampling_final_orthant_relax_time_mode=config["trainer"].get(
            "sampling_final_orthant_relax_time_mode", "local"
        ),
        sampling_final_orthant_relax_edge_floor=config["trainer"].get(
            "sampling_final_orthant_relax_edge_floor"
        ),
        training_sampling_frequency=config["trainer"].get(
            "training_sampling_frequency", 200
        ),
        training_sampling_start=config["trainer"].get(
            "training_sampling_start", 500
        ),
        training_sampling_mode=config["trainer"].get(
            "training_sampling_mode", "harness_sanity"
        ),
        training_sampling_dt_base=config["trainer"].get(
            "training_sampling_dt_base", 0.02
        ),
        sampling_fixed_dt_base=config["trainer"].get("sampling_fixed_dt_base"),
        sampling_max_steps=config["trainer"].get("sampling_max_steps", 256),
        sampling_max_events=config["trainer"].get("sampling_max_events"),
        sampling_max_autoregressive_merges_per_boundary=config["trainer"].get(
            "sampling_max_autoregressive_merges_per_boundary", -1
        ),
        sampling_disable_inner_logging=config["trainer"].get(
            "sampling_disable_inner_logging", True
        ),
        sampling_only_first_hit_collapse=config["trainer"].get(
            "sampling_only_first_hit_collapse", False
        ),
        sampling_actual_event_boundary_use_at_sampling=config["trainer"].get(
            "sampling_actual_event_boundary_use_at_sampling", False
        ),
        sampling_actual_event_boundary_include_predicted_first_hit=config["trainer"].get(
            "sampling_actual_event_boundary_include_predicted_first_hit", False
        ),
        sampling_predsim_overrun_use_at_sampling=config["trainer"].get(
            "sampling_predsim_overrun_use_at_sampling", False
        ),
        sampling_random_fixed_pair_bank_use_at_sampling=config["trainer"].get(
            "sampling_random_fixed_pair_bank_use_at_sampling", False
        ),
        velocity_first_hit_sampling_max_edges=config["trainer"].get(
            "velocity_first_hit_sampling_max_edges", -1
        ),
        velocity_first_hit_sampling_fallback_threshold=config["trainer"].get(
            "velocity_first_hit_sampling_fallback_threshold", -1
        ),
        velocity_first_hit_sampling_fallback_top_k=config["trainer"].get(
            "velocity_first_hit_sampling_fallback_top_k", -1
        ),
        sampling_use_top_merge_planner=config["trainer"].get(
            "sampling_use_top_merge_planner", False
        ),
        sampling_use_inference_mode=config["trainer"].get(
            "sampling_use_inference_mode", False
        ),
        sampling_cache_tri_mask=config["trainer"].get(
            "sampling_cache_tri_mask", False
        ),
        sampling_cache_polytomy_groups=config["trainer"].get(
            "sampling_cache_polytomy_groups", False
        ),
        sampling_cache_autoregressive_state=config["trainer"].get(
            "sampling_cache_autoregressive_state", False
        ),
        training_sampling_stop_on_zero_rf=config["trainer"].get(
            "training_sampling_stop_on_zero_rf", False
        ),
        training_sampling_stop_rf_threshold=config["trainer"].get(
            "training_sampling_stop_rf_threshold"
        ),
        dt=config["trainer"].get("dt", 0.1),
        sample_metrics_trace_path=config["trainer"].get("sample_metrics_trace_path"),
        sample_metrics_num_pairs=config["trainer"].get("sample_metrics_num_pairs", 1),
        sample_metrics_trace_topology_repeats_enabled=config["trainer"].get(
            "sample_metrics_trace_topology_repeats_enabled", False
        ),
        sample_metrics_unseen_start_eval=config["trainer"].get(
            "sample_metrics_unseen_start_eval", False
        ),
        sample_metrics_zero_shot_random_start_eval=config["trainer"].get(
            "sample_metrics_zero_shot_random_start_eval", False
        ),
        sample_metrics_unseen_start_seed=config["trainer"].get(
            "sample_metrics_unseen_start_seed", 20260430
        ),
        sample_metrics_unseen_start_metric_encoder_path=config["trainer"].get(
            "sample_metrics_unseen_start_metric_encoder_path"
        ),
        sample_metrics_unseen_pair_selection_mode=config["trainer"].get(
            "sample_metrics_unseen_pair_selection_mode", "random_bank"
        ),
        sample_metrics_unseen_start_max_duplicate_tries=config["trainer"].get(
            "sample_metrics_unseen_start_max_duplicate_tries", 100
        ),
        sample_metrics_relaxed_likelihood_enabled=config["trainer"].get(
            "sample_metrics_relaxed_likelihood_enabled", False
        ),
        sample_metrics_branch_relaxer_checkpoint_path=config["trainer"].get(
            "sample_metrics_branch_relaxer_checkpoint_path"
        ),
        sample_metrics_mrbayes20k_enabled=config["trainer"].get(
            "sample_metrics_mrbayes20k_enabled", False
        ),
        sample_metrics_mrbayes20k_num_starts=config["trainer"].get(
            "sample_metrics_mrbayes20k_num_starts", 64
        ),
        sample_metrics_mrbayes20k_ngen=config["trainer"].get(
            "sample_metrics_mrbayes20k_ngen", 20000
        ),
        sample_metrics_mrbayes20k_samplefreq=config["trainer"].get(
            "sample_metrics_mrbayes20k_samplefreq", 200
        ),
        sample_metrics_mrbayes20k_printfreq=config["trainer"].get(
            "sample_metrics_mrbayes20k_printfreq", 5000
        ),
        sample_metrics_mrbayes20k_max_workers=config["trainer"].get(
            "sample_metrics_mrbayes20k_max_workers", 12
        ),
        sample_metrics_mrbayes20k_timeout_sec=config["trainer"].get(
            "sample_metrics_mrbayes20k_timeout_sec", 1800
        ),
        sample_metrics_mrbayes20k_dataset_pickle_path=config["trainer"].get(
            "sample_metrics_mrbayes20k_dataset_pickle_path"
        ),
        sample_metrics_mrbayes20k_golden_root=config["trainer"].get(
            "sample_metrics_mrbayes20k_golden_root"
        ),
        sample_metrics_mrbayes20k_work_root=config["trainer"].get(
            "sample_metrics_mrbayes20k_work_root",
            "/tmp/phylaflow_sample_metrics_mrbayes20k",
        ),
        sample_metrics_mrbayes20k_output_dir=config["trainer"].get(
            "sample_metrics_mrbayes20k_output_dir"
        ),
        sample_metrics_mrbayes20k_bin=config["trainer"].get(
            "sample_metrics_mrbayes20k_bin",
            "/opt/conda/envs/phylaflow-mrbayes/bin/mb",
        ),
        sample_metrics_tree_dump_enabled=config["trainer"].get(
            "sample_metrics_tree_dump_enabled", False
        ),
        sample_metrics_tree_dump_dir=config["trainer"].get(
            "sample_metrics_tree_dump_dir"
        ),
        sample_metrics_checkpoint_enabled=config["trainer"].get(
            "sample_metrics_checkpoint_enabled", True
        ),
        sample_metrics_checkpoint_dir=config["trainer"].get(
            "sample_metrics_checkpoint_dir"
        ),
        metric_log_exact_keys=config["trainer"].get("metric_log_exact_keys"),
        metric_log_prefixes=config["trainer"].get("metric_log_prefixes"),
        branch_relax_head_weight=config["trainer"].get("branch_relax_head_weight", 0.0),
        branch_relax_head_use_at_sampling=config["trainer"].get(
            "branch_relax_head_use_at_sampling", False
        ),
        branch_relax_start_tree_list_path=config["trainer"].get(
            "branch_relax_start_tree_list_path"
        ),
        branch_relax_target_tree_list_path=config["trainer"].get(
            "branch_relax_target_tree_list_path"
        ),
        branch_relax_detach_trunk=config["trainer"].get(
            "branch_relax_detach_trunk", True
        ),
        branch_relax_batch_size=config["trainer"].get("branch_relax_batch_size", 1),
        branch_relax_case_dim=config["trainer"].get("branch_relax_case_dim", 64),
        branch_relax_hidden_dim=config["trainer"].get("branch_relax_hidden_dim", 256),
        branch_relax_likelihood_dataset_id=config["trainer"].get(
            "branch_relax_likelihood_dataset_id"
        ),
        branch_relax_likelihood_metric_enabled=config["trainer"].get(
            "branch_relax_likelihood_metric_enabled", False
        ),
        rollout_replay_velocity_weight=config["trainer"].get(
            "rollout_replay_velocity_weight", 0.0
        ),
        rollout_replay_autoregressive_weight=config["trainer"].get(
            "rollout_replay_autoregressive_weight", 0.0
        ),
        rollout_replay_start_step=config["trainer"].get(
            "rollout_replay_start_step", 0
        ),
        rollout_replay_frequency=config["trainer"].get(
            "rollout_replay_frequency", 1
        ),
        rollout_replay_max_velocity_states=config["trainer"].get(
            "rollout_replay_max_velocity_states", 0
        ),
        rollout_replay_max_autoregressive_states=config["trainer"].get(
            "rollout_replay_max_autoregressive_states", 0
        ),
        rollout_replay_max_steps=config["trainer"].get(
            "rollout_replay_max_steps", 256
        ),
        rollout_replay_max_events=config["trainer"].get(
            "rollout_replay_max_events"
        ),
        rollout_replay_anchor_states=config["trainer"].get(
            "rollout_replay_anchor_states", 4
        ),
        rollout_replay_oracle_horizon=config["trainer"].get(
            "rollout_replay_oracle_horizon", 2
        ),
        rollout_replay_mode=config["trainer"].get(
            "rollout_replay_mode", "anchor_oracle"
        ),
        rollout_replay_pairwise_max_group_size=config["trainer"].get(
            "rollout_replay_pairwise_max_group_size", 0
        ),
        rollout_replay_bank_max_polytomy_size=config["trainer"].get(
            "rollout_replay_bank_max_polytomy_size", -1
        ),
        rollout_replay_topology_repeat_cap=config["trainer"].get(
            "rollout_replay_topology_repeat_cap", 0
        ),
        rollout_replay_dump_refreshes=config["trainer"].get(
            "rollout_replay_dump_refreshes", False
        ),
        rollout_replay_dump_dir=config["trainer"].get(
            "rollout_replay_dump_dir"
        ),
        rollout_replay_fixed_dt_base=config["trainer"].get(
            "rollout_replay_fixed_dt_base"
        ),
        rollout_replay_prefix_stop_early=config["trainer"].get(
            "rollout_replay_prefix_stop_early", False
        ),
        rollout_replay_cache_reuse_every_step=config["trainer"].get(
            "rollout_replay_cache_reuse_every_step", True
        ),
        rollout_replay_refresh_only_if_better_rf=config["trainer"].get(
            "rollout_replay_refresh_only_if_better_rf", False
        ),
        rollout_replay_legacy_loss_structure=config["trainer"].get(
            "rollout_replay_legacy_loss_structure", False
        ),
        dynamic_start_bank_enabled=config["trainer"].get(
            "dynamic_start_bank_enabled", False
        ),
        dynamic_start_bank_start_step=config["trainer"].get(
            "dynamic_start_bank_start_step", 0
        ),
        dynamic_start_bank_max_entries=config["trainer"].get(
            "dynamic_start_bank_max_entries", 2
        ),
        dynamic_start_bank_min_rf_improvement=config["trainer"].get(
            "dynamic_start_bank_min_rf_improvement", 0.0
        ),
        dynamic_start_bank_max_polytomy_size=config["trainer"].get(
            "dynamic_start_bank_max_polytomy_size", -1
        ),
        dynamic_start_bank_mode=config["trainer"].get(
            "dynamic_start_bank_mode",
            (
                "soft_hybrid"
                if (
                    config["trainer"].get("analysis_soft_hybrid_best_rf_repeat")
                    is not None
                    or config["trainer"].get(
                        "analysis_soft_hybrid_best_multivel_repeat"
                    )
                    is not None
                )
                else "best_start"
            ),
        ),
        dynamic_start_bank_min_velocity_states=config["trainer"].get(
            "dynamic_start_bank_min_velocity_states",
            config["trainer"].get(
                "analysis_dynamic_start_bank_min_velocity_states", 2
            ),
        ),
        dynamic_start_bank_best_rf_repeat=config["trainer"].get(
            "dynamic_start_bank_best_rf_repeat",
            config["trainer"].get("analysis_soft_hybrid_best_rf_repeat", 18),
        ),
        dynamic_start_bank_best_multivel_repeat=config["trainer"].get(
            "dynamic_start_bank_best_multivel_repeat",
            config["trainer"].get(
                "analysis_soft_hybrid_best_multivel_repeat", 9
            ),
        ),
        dynamic_start_bank_trace_path=config["trainer"].get(
            "dynamic_start_bank_trace_path"
        ),
        dynamic_start_bank_artifact_dir=config["trainer"].get(
            "dynamic_start_bank_artifact_dir"
        ),
        dynamic_start_bank_save_improved_checkpoint=config["trainer"].get(
            "dynamic_start_bank_save_improved_checkpoint", False
        ),
        verbose=True,  # Enable verbose logging for overfitting
    )
    init_training_module_checkpoint_path = config["trainer"].get(
        "init_training_module_checkpoint_path"
    )
    if init_training_module_checkpoint_path:
        if resume_ckpt_path:
            raise ValueError(
                "Use either trainer.init_training_module_checkpoint_path or "
                "trainer.resume_ckpt_path, not both."
            )
        init_training_module_checkpoint_path = os.path.abspath(
            init_training_module_checkpoint_path
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _load_training_module_init_checkpoint(
            model,
            init_training_module_checkpoint_path,
            device,
        )
    model.legacy_first_hit_gather_only = bool(
        config["trainer"].get("legacy_first_hit_gather_only", False)
    )

    checkpoint_base = config["trainer"]["checkpoint_dir"]
    checkpoint_dir = os.path.join(checkpoint_base, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Saving checkpoints to: {checkpoint_dir}")

    checkpoint_every_n_train_steps = int(
        config["trainer"].get("steps_callback", 0) or 0
    )
    if checkpoint_every_n_train_steps > 0:
        save_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="overfit-{epoch:02d}-step={step:06d}",
            every_n_train_steps=checkpoint_every_n_train_steps,
            save_top_k=-1,
            save_last=True,
        )
    else:
        save_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="overfit-{epoch:02d}",
            every_n_epochs=50,
            save_top_k=-1,
            save_last=True,
        )

    trainer_args = {}
    if config["trainer"]["record"]:
        _init_wandb_run(config, default_project="phylaflow_overfit")
    else:
        trainer_args["logger"] = False

    if config["trainer"].get("default_root_dir"):
        trainer_args["default_root_dir"] = config["trainer"]["default_root_dir"]

    trainer_args["max_epochs"] = config["trainer"]["epochs"]
    trainer_args["callbacks"] = [save_callback]

    # Log frequently for overfitting
    trainer_args["log_every_n_steps"] = 1

    if config["trainer"]["val_callback_freq"] != 0:
        trainer_args["val_check_interval"] = config["trainer"]["val_callback_freq"]

    trainer_args["accelerator"] = "gpu"
    trainer_args["devices"] = config["trainer"].get("devices", 1)

    trainer = Trainer(**trainer_args)
    trainer.fit(
        model,
        train_dataloaders=dataset.train_dataloader(),
        val_dataloaders=dataset.val_dataloader(),
        ckpt_path=resume_ckpt_path,
    )


def main():
    # Get first command line argument as config file
    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        config = _expand_config_env_vars(yaml.safe_load(f))

    _configure_torch_runtime()
    _set_global_seed(config["trainer"].get("seed"))

    ids = _get_dataset_ids_from_config(config)
    # Random 80-20 train-test split for now
    ran = random.Random(42)
    ran.shuffle(ids)
    if len(ids) < 2:
        train_ids = ids
        test_ids = ids
    else:
        train_ids = ids[: int(0.8 * len(ids))]
        test_ids = ids[int(0.8 * len(ids)) :]

    dataset = PhylaDataModule(config, train_ids=train_ids, test_ids=test_ids)

    phyla_flow = return_model(config)
    init_checkpoint_path, resume_ckpt_path = _resolve_checkpoint_paths(
        config["trainer"]
    )
    if init_checkpoint_path:
        print(f"Initializing model weights from checkpoint: {init_checkpoint_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        phyla_flow = _load_model_init_checkpoint(
            phyla_flow, init_checkpoint_path, device
        )
    if resume_ckpt_path:
        print(f"Resuming full trainer state from checkpoint: {resume_ckpt_path}")

    model = TrainingModule(
        model=phyla_flow,
        lr=config["trainer"]["lr"],
        optimizer_name=config["trainer"].get("optimizer_name", "adamw"),
        record=config["trainer"]["record"],
        epochs=config["trainer"]["epochs"],
        dataset=dataset,
        lr_scheduler="default",
        num_annealing_steps=10000,
        num_warmup_steps=1000,
        deepspeed=False,
        logger=None,
        phyla_checkpoint_path=config["trainer"].get("phyla_checkpoint_path"),
        phyla_precomputed_embeddings_path=config["trainer"].get(
            "phyla_precomputed_embeddings_path"
        ),
        live_phyla_checkpoint_path=config["trainer"].get(
            "live_phyla_checkpoint_path"
        ),
        live_phyla_unfreeze=config["trainer"].get("live_phyla_unfreeze", True),
        live_phyla_lr=config["trainer"].get("live_phyla_lr"),
        live_phyla_input_mode=config["trainer"].get(
            "live_phyla_input_mode",
            "raw-full",
        ),
        live_phyla_max_input_tokens=config["trainer"].get(
            "live_phyla_max_input_tokens",
            0,
        ),
        live_phyla_device=config["trainer"].get("live_phyla_device"),
        velocity_loss_mode=config["trainer"].get("velocity_loss_mode", "weighted"),
        velocity_loss_plain_weight=config["trainer"].get(
            "velocity_loss_plain_weight", 0.5
        ),
        velocity_sign_eps=config["trainer"].get("velocity_sign_eps", 1e-3),
        training_step_velocity_weight=config["trainer"].get(
            "training_step_velocity_weight", 1.0
        ),
        training_step_autoregressive_weight=config["trainer"].get(
            "training_step_autoregressive_weight", 1.0
        ),
        training_step_gradient_clip_val=config["trainer"].get(
            "training_step_gradient_clip_val", 1.0
        ),
        grad_norm_log_frequency=config["trainer"].get("grad_norm_log_frequency", 1),
        training_step_profile_frequency=config["trainer"].get(
            "training_step_profile_frequency", 0
        ),
        training_step_profile_warmup_steps=config["trainer"].get(
            "training_step_profile_warmup_steps", 0
        ),
        training_step_profile_sync_cuda=config["trainer"].get(
            "training_step_profile_sync_cuda", True
        ),
        training_step_autoregressive_grad_ratio=config["trainer"].get(
            "training_step_autoregressive_grad_ratio"
        ),
        training_step_separate_optimizer_steps=config["trainer"].get(
            "training_step_separate_optimizer_steps", False
        ),
        training_step_verbose_logging_enabled=config["trainer"].get(
            "training_step_verbose_logging_enabled", False
        ),
        autoregressive_use_time=config["trainer"].get(
            "autoregressive_use_time", False
        ),
        autoregressive_target_mode=config["trainer"].get(
            "autoregressive_target_mode", "scheduled"
        ),
        autoregressive_polytomy_choosing_weight=config["trainer"].get(
            "autoregressive_polytomy_choosing_weight", 1.0
        ),
        autoregressive_stop_after_merge_weight=config["trainer"].get(
            "autoregressive_stop_after_merge_weight", 0.0
        ),
        autoregressive_stop_after_merge_use_at_sampling=config["trainer"].get(
            "autoregressive_stop_after_merge_use_at_sampling", False
        ),
        autoregressive_rollin_prob=config["trainer"].get(
            "autoregressive_rollin_prob", 0.0
        ),
        autoregressive_dagger_prob=config["trainer"].get(
            "autoregressive_dagger_prob", 0.0
        ),
        autoregressive_dagger_max_steps=config["trainer"].get(
            "autoregressive_dagger_max_steps", 4
        ),
        autoregressive_structure_perturb_prob=config["trainer"].get(
            "autoregressive_structure_perturb_prob", 0.0
        ),
        autoregressive_structure_perturb_mode=config["trainer"].get(
            "autoregressive_structure_perturb_mode", "random_wrong_pair"
        ),
        **_birthset_trainer_kwargs(config),
        velocity_length_jitter_prob=config["trainer"].get(
            "velocity_length_jitter_prob", 0.0
        ),
        velocity_length_jitter_scale=config["trainer"].get(
            "velocity_length_jitter_scale", 0.0
        ),
        velocity_dt_candidate_weight=config["trainer"].get(
            "velocity_dt_candidate_weight", 0.0
        ),
        velocity_dt_hit_weight=config["trainer"].get("velocity_dt_hit_weight", 0.0),
        velocity_logtau_all_weight=config["trainer"].get(
            "velocity_logtau_all_weight", 0.0
        ),
        velocity_logtau_first_over_weight=config["trainer"].get(
            "velocity_logtau_first_over_weight", 0.0
        ),
        velocity_logtau_first_tie_weight=config["trainer"].get(
            "velocity_logtau_first_tie_weight", 0.0
        ),
        velocity_dt_eps=config["trainer"].get("velocity_dt_eps", 1e-6),
        velocity_event_weight=config["trainer"].get("velocity_event_weight", 0.5),
        velocity_event_temp=config["trainer"].get("velocity_event_temp", 0.5),
        velocity_event_rate_beta=config["trainer"].get("velocity_event_rate_beta", 5.0),
        velocity_event_normalize_by_log_candidates=config["trainer"].get(
            "velocity_event_normalize_by_log_candidates", True
        ),
        velocity_event_precision_weight=config["trainer"].get(
            "velocity_event_precision_weight", 0.0
        ),
        velocity_event_precision_margin=config["trainer"].get(
            "velocity_event_precision_margin", 0.0
        ),
        velocity_first_hit_head_weight=config["trainer"].get(
            "velocity_first_hit_head_weight", 0.0
        ),
        velocity_first_hit_loss_tol=config["trainer"].get(
            "velocity_first_hit_loss_tol", 0.01
        ),
        velocity_first_hit_head_use_at_sampling=config["trainer"].get(
            "velocity_first_hit_head_use_at_sampling", False
        ),
        velocity_first_hit_predictor_mode=config["trainer"].get(
            "velocity_first_hit_predictor_mode", "base"
        ),
        velocity_first_hit_false_positive_mass_weight=config["trainer"].get(
            "velocity_first_hit_false_positive_mass_weight", 0.0
        ),
        velocity_first_hit_false_negative_mass_weight=config["trainer"].get(
            "velocity_first_hit_false_negative_mass_weight", 0.0
        ),
        velocity_refiner_mode=config["trainer"].get(
            "velocity_refiner_mode", "base"
        ),
        velocity_first_hit_use_geometry_features=config["trainer"].get(
            "velocity_first_hit_use_geometry_features", False
        ),
        velocity_first_hit_geometry_hidden_dim=config["trainer"].get(
            "velocity_first_hit_geometry_hidden_dim", 32
        ),
        velocity_first_hit_edge_length_hidden_dim=config["trainer"].get(
            "velocity_first_hit_edge_length_hidden_dim", 64
        ),
        velocity_refiner_attention_layers=config["trainer"].get(
            "velocity_refiner_attention_layers", 1
        ),
        velocity_refiner_attention_heads=config["trainer"].get(
            "velocity_refiner_attention_heads", 4
        ),
        velocity_refiner_bucket_count=config["trainer"].get(
            "velocity_refiner_bucket_count", 32
        ),
        velocity_refiner_bucket_log_min=config["trainer"].get(
            "velocity_refiner_bucket_log_min", -8.0
        ),
        velocity_refiner_bucket_log_max=config["trainer"].get(
            "velocity_refiner_bucket_log_max", 1.0
        ),
        velocity_boundary_vanish_head_weight=config["trainer"].get(
            "velocity_boundary_vanish_head_weight", 0.0
        ),
        velocity_boundary_vanish_head_use_at_sampling=config["trainer"].get(
            "velocity_boundary_vanish_head_use_at_sampling", False
        ),
        velocity_boundary_vanish_one_step_use_at_sampling=config["trainer"].get(
            "velocity_boundary_vanish_one_step_use_at_sampling", False
        ),
        velocity_boundary_time_head_weight=config["trainer"].get(
            "velocity_boundary_time_head_weight", 0.0
        ),
        velocity_boundary_time_head_use_at_sampling=config["trainer"].get(
            "velocity_boundary_time_head_use_at_sampling", False
        ),
        velocity_boundary_time_hidden_dim=config["trainer"].get(
            "velocity_boundary_time_hidden_dim", 64
        ),
        velocity_terminal_head_weight=config["trainer"].get(
            "velocity_terminal_head_weight", 0.0
        ),
        velocity_terminal_head_use_at_sampling=config["trainer"].get(
            "velocity_terminal_head_use_at_sampling", False
        ),
        velocity_terminal_head_sampling_action=config["trainer"].get(
            "velocity_terminal_head_sampling_action", "after_phase"
        ),
        velocity_terminal_head_hidden_dim=config["trainer"].get(
            "velocity_terminal_head_hidden_dim", 64
        ),
        velocity_terminal_head_probe_features=config["trainer"].get(
            "velocity_terminal_head_probe_features", False
        ),
        velocity_terminal_head_input_mode=config["trainer"].get(
            "velocity_terminal_head_input_mode"
        ),
        velocity_terminal_head_use_case_adapt=config["trainer"].get(
            "velocity_terminal_head_use_case_adapt", False
        ),
        velocity_terminal_head_balance_loss=config["trainer"].get(
            "velocity_terminal_head_balance_loss", False
        ),
        velocity_terminal_head_topology_pool=config["trainer"].get(
            "velocity_terminal_head_topology_pool", "mean"
        ),
        velocity_probe_direct_set_loss=config["trainer"].get(
            "velocity_probe_direct_set_loss", False
        ),
        velocity_probe_direct_set_anchor_only=config["trainer"].get(
            "velocity_probe_direct_set_anchor_only", False
        ),
        velocity_probe_direct_set_include_base_samples=config["trainer"].get(
            "velocity_probe_direct_set_include_base_samples", False
        ),
        velocity_probe_direct_set_target_negative_weight=config["trainer"].get(
            "velocity_probe_direct_set_target_negative_weight", 1.0
        ),
        velocity_probe_direct_set_nontarget_nonnegative_weight=config["trainer"].get(
            "velocity_probe_direct_set_nontarget_nonnegative_weight", 0.0
        ),
        velocity_probe_direct_set_positive_reweight=config["trainer"].get(
            "velocity_probe_direct_set_positive_reweight", False
        ),
        velocity_probe_direct_set_positive_reweight_power=config["trainer"].get(
            "velocity_probe_direct_set_positive_reweight_power", 1.0
        ),
        velocity_probe_direct_set_positive_reweight_max=config["trainer"].get(
            "velocity_probe_direct_set_positive_reweight_max"
        ),
        velocity_probe_direct_set_bce_weight=config["trainer"].get(
            "velocity_probe_direct_set_bce_weight", 1.0
        ),
        velocity_probe_direct_set_loss_weight=config["trainer"].get(
            "velocity_probe_direct_set_loss_weight", 1.0
        ),
        velocity_probe_direct_set_mse_weight=config["trainer"].get(
            "velocity_probe_direct_set_mse_weight", 0.0
        ),
        training_step_probe_parity_joint_update=config["trainer"].get(
            "training_step_probe_parity_joint_update", False
        ),
        training_step_joint_tokenize_velocity_ar=config["trainer"].get(
            "training_step_joint_tokenize_velocity_ar", False
        ),
        training_step_full_path_replay_initial_retry_attempt=config["trainer"].get(
            "training_step_full_path_replay_initial_retry_attempt", 0
        ),
        skip_repeated_no_valid_boundary_use_at_sampling=config["trainer"].get(
            "skip_repeated_no_valid_boundary_use_at_sampling", False
        ),
        sampling_discrete_phase_rollout_use_at_sampling=config["trainer"].get(
            "sampling_discrete_phase_rollout_use_at_sampling", False
        ),
        sampling_discrete_phase_exact_boundary_step_use_at_sampling=config["trainer"].get(
            "sampling_discrete_phase_exact_boundary_step_use_at_sampling", False
        ),
        sampling_discrete_phase_max_phases=config["trainer"].get(
            "sampling_discrete_phase_max_phases", 8
        ),
        sampling_final_orthant_relax_use_at_sampling=config["trainer"].get(
            "sampling_final_orthant_relax_use_at_sampling", False
        ),
        sampling_final_orthant_relax_steps=config["trainer"].get(
            "sampling_final_orthant_relax_steps", 0
        ),
        sampling_final_orthant_relax_total_time=config["trainer"].get(
            "sampling_final_orthant_relax_total_time", 1.0
        ),
        sampling_final_orthant_relax_time_mode=config["trainer"].get(
            "sampling_final_orthant_relax_time_mode", "local"
        ),
        sampling_final_orthant_relax_edge_floor=config["trainer"].get(
            "sampling_final_orthant_relax_edge_floor"
        ),
        training_sampling_frequency=config["trainer"].get(
            "training_sampling_frequency", 200
        ),
        training_sampling_start=config["trainer"].get(
            "training_sampling_start", 500
        ),
        training_sampling_mode=config["trainer"].get(
            "training_sampling_mode", "batch_compare"
        ),
        training_sampling_dt_base=config["trainer"].get(
            "training_sampling_dt_base", 0.02
        ),
        sampling_fixed_dt_base=config["trainer"].get("sampling_fixed_dt_base"),
        sampling_max_steps=config["trainer"].get("sampling_max_steps", 256),
        sampling_max_events=config["trainer"].get("sampling_max_events"),
        sampling_max_autoregressive_merges_per_boundary=config["trainer"].get(
            "sampling_max_autoregressive_merges_per_boundary", -1
        ),
        sampling_disable_inner_logging=config["trainer"].get(
            "sampling_disable_inner_logging", True
        ),
        sampling_only_first_hit_collapse=config["trainer"].get(
            "sampling_only_first_hit_collapse", False
        ),
        sampling_actual_event_boundary_use_at_sampling=config["trainer"].get(
            "sampling_actual_event_boundary_use_at_sampling", False
        ),
        sampling_actual_event_boundary_include_predicted_first_hit=config["trainer"].get(
            "sampling_actual_event_boundary_include_predicted_first_hit", False
        ),
        sampling_predsim_overrun_use_at_sampling=config["trainer"].get(
            "sampling_predsim_overrun_use_at_sampling", False
        ),
        sampling_random_fixed_pair_bank_use_at_sampling=config["trainer"].get(
            "sampling_random_fixed_pair_bank_use_at_sampling", False
        ),
        velocity_first_hit_sampling_max_edges=config["trainer"].get(
            "velocity_first_hit_sampling_max_edges", -1
        ),
        velocity_first_hit_sampling_fallback_threshold=config["trainer"].get(
            "velocity_first_hit_sampling_fallback_threshold", -1
        ),
        velocity_first_hit_sampling_fallback_top_k=config["trainer"].get(
            "velocity_first_hit_sampling_fallback_top_k", -1
        ),
        sampling_use_top_merge_planner=config["trainer"].get(
            "sampling_use_top_merge_planner", False
        ),
        sampling_use_inference_mode=config["trainer"].get(
            "sampling_use_inference_mode", False
        ),
        sampling_cache_tri_mask=config["trainer"].get(
            "sampling_cache_tri_mask", False
        ),
        sampling_cache_polytomy_groups=config["trainer"].get(
            "sampling_cache_polytomy_groups", False
        ),
        sampling_cache_autoregressive_state=config["trainer"].get(
            "sampling_cache_autoregressive_state", False
        ),
        training_sampling_stop_on_zero_rf=config["trainer"].get(
            "training_sampling_stop_on_zero_rf", False
        ),
        training_sampling_stop_rf_threshold=config["trainer"].get(
            "training_sampling_stop_rf_threshold"
        ),
        dt=config["trainer"].get("dt", 0.1),
        sample_metrics_trace_path=config["trainer"].get("sample_metrics_trace_path"),
        sample_metrics_num_pairs=config["trainer"].get("sample_metrics_num_pairs", 1),
        sample_metrics_trace_topology_repeats_enabled=config["trainer"].get(
            "sample_metrics_trace_topology_repeats_enabled", False
        ),
        sample_metrics_unseen_start_eval=config["trainer"].get(
            "sample_metrics_unseen_start_eval", False
        ),
        sample_metrics_zero_shot_random_start_eval=config["trainer"].get(
            "sample_metrics_zero_shot_random_start_eval", False
        ),
        sample_metrics_unseen_start_seed=config["trainer"].get(
            "sample_metrics_unseen_start_seed", 20260430
        ),
        sample_metrics_unseen_start_metric_encoder_path=config["trainer"].get(
            "sample_metrics_unseen_start_metric_encoder_path"
        ),
        sample_metrics_unseen_pair_selection_mode=config["trainer"].get(
            "sample_metrics_unseen_pair_selection_mode", "random_bank"
        ),
        sample_metrics_unseen_start_max_duplicate_tries=config["trainer"].get(
            "sample_metrics_unseen_start_max_duplicate_tries", 100
        ),
        sample_metrics_relaxed_likelihood_enabled=config["trainer"].get(
            "sample_metrics_relaxed_likelihood_enabled", False
        ),
        sample_metrics_branch_relaxer_checkpoint_path=config["trainer"].get(
            "sample_metrics_branch_relaxer_checkpoint_path"
        ),
        sample_metrics_mrbayes20k_enabled=config["trainer"].get(
            "sample_metrics_mrbayes20k_enabled", False
        ),
        sample_metrics_mrbayes20k_num_starts=config["trainer"].get(
            "sample_metrics_mrbayes20k_num_starts", 64
        ),
        sample_metrics_mrbayes20k_ngen=config["trainer"].get(
            "sample_metrics_mrbayes20k_ngen", 20000
        ),
        sample_metrics_mrbayes20k_samplefreq=config["trainer"].get(
            "sample_metrics_mrbayes20k_samplefreq", 200
        ),
        sample_metrics_mrbayes20k_printfreq=config["trainer"].get(
            "sample_metrics_mrbayes20k_printfreq", 5000
        ),
        sample_metrics_mrbayes20k_max_workers=config["trainer"].get(
            "sample_metrics_mrbayes20k_max_workers", 12
        ),
        sample_metrics_mrbayes20k_timeout_sec=config["trainer"].get(
            "sample_metrics_mrbayes20k_timeout_sec", 1800
        ),
        sample_metrics_mrbayes20k_dataset_pickle_path=config["trainer"].get(
            "sample_metrics_mrbayes20k_dataset_pickle_path"
        ),
        sample_metrics_mrbayes20k_golden_root=config["trainer"].get(
            "sample_metrics_mrbayes20k_golden_root"
        ),
        sample_metrics_mrbayes20k_work_root=config["trainer"].get(
            "sample_metrics_mrbayes20k_work_root",
            "/tmp/phylaflow_sample_metrics_mrbayes20k",
        ),
        sample_metrics_mrbayes20k_output_dir=config["trainer"].get(
            "sample_metrics_mrbayes20k_output_dir"
        ),
        sample_metrics_mrbayes20k_bin=config["trainer"].get(
            "sample_metrics_mrbayes20k_bin",
            "/opt/conda/envs/phylaflow-mrbayes/bin/mb",
        ),
        sample_metrics_tree_dump_enabled=config["trainer"].get(
            "sample_metrics_tree_dump_enabled", False
        ),
        sample_metrics_tree_dump_dir=config["trainer"].get(
            "sample_metrics_tree_dump_dir"
        ),
        sample_metrics_checkpoint_enabled=config["trainer"].get(
            "sample_metrics_checkpoint_enabled", True
        ),
        sample_metrics_checkpoint_dir=config["trainer"].get(
            "sample_metrics_checkpoint_dir"
        ),
        metric_log_exact_keys=config["trainer"].get("metric_log_exact_keys"),
        metric_log_prefixes=config["trainer"].get("metric_log_prefixes"),
        branch_relax_head_weight=config["trainer"].get("branch_relax_head_weight", 0.0),
        branch_relax_head_use_at_sampling=config["trainer"].get(
            "branch_relax_head_use_at_sampling", False
        ),
        branch_relax_start_tree_list_path=config["trainer"].get(
            "branch_relax_start_tree_list_path"
        ),
        branch_relax_target_tree_list_path=config["trainer"].get(
            "branch_relax_target_tree_list_path"
        ),
        branch_relax_detach_trunk=config["trainer"].get(
            "branch_relax_detach_trunk", True
        ),
        branch_relax_batch_size=config["trainer"].get("branch_relax_batch_size", 1),
        branch_relax_case_dim=config["trainer"].get("branch_relax_case_dim", 64),
        branch_relax_hidden_dim=config["trainer"].get("branch_relax_hidden_dim", 256),
        branch_relax_likelihood_dataset_id=config["trainer"].get(
            "branch_relax_likelihood_dataset_id"
        ),
        branch_relax_likelihood_metric_enabled=config["trainer"].get(
            "branch_relax_likelihood_metric_enabled", False
        ),
        rollout_replay_velocity_weight=config["trainer"].get(
            "rollout_replay_velocity_weight", 0.0
        ),
        rollout_replay_autoregressive_weight=config["trainer"].get(
            "rollout_replay_autoregressive_weight", 0.0
        ),
        rollout_replay_start_step=config["trainer"].get(
            "rollout_replay_start_step", 0
        ),
        rollout_replay_frequency=config["trainer"].get(
            "rollout_replay_frequency", 1
        ),
        rollout_replay_max_velocity_states=config["trainer"].get(
            "rollout_replay_max_velocity_states", 0
        ),
        rollout_replay_max_autoregressive_states=config["trainer"].get(
            "rollout_replay_max_autoregressive_states", 0
        ),
        rollout_replay_max_steps=config["trainer"].get(
            "rollout_replay_max_steps", 256
        ),
        rollout_replay_max_events=config["trainer"].get(
            "rollout_replay_max_events"
        ),
        rollout_replay_anchor_states=config["trainer"].get(
            "rollout_replay_anchor_states", 4
        ),
        rollout_replay_oracle_horizon=config["trainer"].get(
            "rollout_replay_oracle_horizon", 2
        ),
        rollout_replay_mode=config["trainer"].get(
            "rollout_replay_mode", "anchor_oracle"
        ),
        rollout_replay_pairwise_max_group_size=config["trainer"].get(
            "rollout_replay_pairwise_max_group_size", 0
        ),
        rollout_replay_bank_max_polytomy_size=config["trainer"].get(
            "rollout_replay_bank_max_polytomy_size", -1
        ),
        rollout_replay_topology_repeat_cap=config["trainer"].get(
            "rollout_replay_topology_repeat_cap", 0
        ),
        rollout_replay_dump_refreshes=config["trainer"].get(
            "rollout_replay_dump_refreshes", False
        ),
        rollout_replay_dump_dir=config["trainer"].get(
            "rollout_replay_dump_dir"
        ),
        rollout_replay_fixed_dt_base=config["trainer"].get(
            "rollout_replay_fixed_dt_base"
        ),
        rollout_replay_prefix_stop_early=config["trainer"].get(
            "rollout_replay_prefix_stop_early", False
        ),
        rollout_replay_cache_reuse_every_step=config["trainer"].get(
            "rollout_replay_cache_reuse_every_step", True
        ),
        rollout_replay_refresh_only_if_better_rf=config["trainer"].get(
            "rollout_replay_refresh_only_if_better_rf", False
        ),
        rollout_replay_legacy_loss_structure=config["trainer"].get(
            "rollout_replay_legacy_loss_structure", False
        ),
        dynamic_start_bank_enabled=config["trainer"].get(
            "dynamic_start_bank_enabled", False
        ),
        dynamic_start_bank_start_step=config["trainer"].get(
            "dynamic_start_bank_start_step", 0
        ),
        dynamic_start_bank_max_entries=config["trainer"].get(
            "dynamic_start_bank_max_entries", 2
        ),
        dynamic_start_bank_min_rf_improvement=config["trainer"].get(
            "dynamic_start_bank_min_rf_improvement", 0.0
        ),
        dynamic_start_bank_max_polytomy_size=config["trainer"].get(
            "dynamic_start_bank_max_polytomy_size", -1
        ),
        dynamic_start_bank_mode=config["trainer"].get(
            "dynamic_start_bank_mode",
            (
                "soft_hybrid"
                if (
                    config["trainer"].get("analysis_soft_hybrid_best_rf_repeat")
                    is not None
                    or config["trainer"].get(
                        "analysis_soft_hybrid_best_multivel_repeat"
                    )
                    is not None
                )
                else "best_start"
            ),
        ),
        dynamic_start_bank_min_velocity_states=config["trainer"].get(
            "dynamic_start_bank_min_velocity_states",
            config["trainer"].get(
                "analysis_dynamic_start_bank_min_velocity_states", 2
            ),
        ),
        dynamic_start_bank_best_rf_repeat=config["trainer"].get(
            "dynamic_start_bank_best_rf_repeat",
            config["trainer"].get("analysis_soft_hybrid_best_rf_repeat", 18),
        ),
        dynamic_start_bank_best_multivel_repeat=config["trainer"].get(
            "dynamic_start_bank_best_multivel_repeat",
            config["trainer"].get(
                "analysis_soft_hybrid_best_multivel_repeat", 9
            ),
        ),
        dynamic_start_bank_trace_path=config["trainer"].get(
            "dynamic_start_bank_trace_path"
        ),
        dynamic_start_bank_artifact_dir=config["trainer"].get(
            "dynamic_start_bank_artifact_dir"
        ),
        dynamic_start_bank_save_improved_checkpoint=config["trainer"].get(
            "dynamic_start_bank_save_improved_checkpoint", False
        ),
    )
    init_training_module_checkpoint_path = config["trainer"].get(
        "init_training_module_checkpoint_path"
    )
    if init_training_module_checkpoint_path:
        if resume_ckpt_path:
            raise ValueError(
                "Use either trainer.init_training_module_checkpoint_path or "
                "trainer.resume_ckpt_path, not both."
            )
        init_training_module_checkpoint_path = os.path.abspath(
            init_training_module_checkpoint_path
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = _load_training_module_init_checkpoint(
            model,
            init_training_module_checkpoint_path,
            device,
        )
    model.legacy_first_hit_gather_only = bool(
        config["trainer"].get("legacy_first_hit_gather_only", False)
    )

    checkpoint_base = config["trainer"]["checkpoint_dir"]
    checkpoint_dir = os.path.join(checkpoint_base, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Saving checkpoints to: {checkpoint_dir}")

    save_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch:02d}-{step:06d}",  # Include metric value in the filename
        every_n_train_steps=config["trainer"]["steps_callback"],  # Save every N steps
        save_top_k=-1,  # Save all checkpoints
    )
    threshold_save_callback = ThresholdStepCheckpoint(
        dirpath=checkpoint_dir,
        every_n_train_steps=config["trainer"]["steps_callback"],
    )

    trainer_args = {}
    if config["trainer"]["record"]:
        _init_wandb_run(config, default_project="phylaflow")
    else:
        trainer_args["logger"] = False
    if config["trainer"].get("disable_lightning_logger", False):
        trainer_args["logger"] = False

    if config["trainer"].get("default_root_dir"):
        trainer_args["default_root_dir"] = config["trainer"]["default_root_dir"]

    trainer_args["max_epochs"] = config["trainer"]["epochs"]
    trainer_args["callbacks"] = [
        save_callback,
        threshold_save_callback,
    ]  # For validation callback runs
    if config["trainer"]["val_callback_freq"] != 0:
        trainer_args["val_check_interval"] = config["trainer"]["val_callback_freq"]
    if config["trainer"]["limit_val_batches"] == 0:
        trainer_args["limit_val_batches"] = 0.0  # Disable validation
    if config["trainer"].get("limit_train_batches") is not None:
        trainer_args["limit_train_batches"] = config["trainer"][
            "limit_train_batches"
        ]

    trainer_args["accelerator"] = "gpu"
    trainer_args["devices"] = config["trainer"].get("devices", 1)
    trainer = Trainer(**trainer_args)
    trainer.fit(
        model,
        train_dataloaders=dataset.train_dataloader(),
        val_dataloaders=dataset.val_dataloader(),
        ckpt_path=resume_ckpt_path,
    )


if __name__ == "__main__":
    main()
