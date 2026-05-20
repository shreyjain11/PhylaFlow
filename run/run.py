from model.model import return_model
from data.dataset import PhylaDataModule
import yaml
import sys
from utils.utils import get_possible_ids
from run.TrainingModule import TrainingModule
import random
try:
    import wandb
except ImportError:  # pragma: no cover - optional experiment logger
    wandb = None
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning import Trainer
import os
import torch
from datetime import datetime

import logging

def _expand_env_vars(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_env_vars(item) for item in value)
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    return value


def _load_config_file(config_file):
    with open(config_file, "r", encoding="utf-8") as handle:
        return _expand_env_vars(yaml.safe_load(handle))


class ThresholdStepCheckpoint(Callback):
    """Save once a completed train batch has crossed the next due step.

    Lightning's every_n_train_steps callback can miss future checkpoints when
    manual optimization leaves batch-end global_step off the exact modulo. This
    callback keeps the same cadence but uses >= thresholding.
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


def _init_wandb_run(config, default_project):
    if wandb is None:
        raise ImportError("Weights & Biases logging requested, but wandb is not installed.")
    trainer_cfg = config.get("trainer", {})
    wandb_kwargs = {
        "project": trainer_cfg.get("wandb_project", default_project),
        "name": trainer_cfg.get("wandb_name"),
        "group": trainer_cfg.get("wandb_group"),
        "job_type": trainer_cfg.get("wandb_job_type"),
        "notes": trainer_cfg.get("wandb_notes"),
        "tags": trainer_cfg.get("wandb_tags"),
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
            "velocity_first_hit_false_positive_mass_weight": trainer_cfg.get(
                "velocity_first_hit_false_positive_mass_weight"
            ),
            "sampling_max_steps": trainer_cfg.get("sampling_max_steps"),
            "sampling_max_events": trainer_cfg.get("sampling_max_events"),
            "training_step_autoregressive_weight": trainer_cfg.get(
                "training_step_autoregressive_weight"
            ),
            "training_step_velocity_weight": trainer_cfg.get(
                "training_step_velocity_weight"
            ),
            "optimizer_name": trainer_cfg.get("optimizer_name"),
            "training_sampling_start": trainer_cfg.get("training_sampling_start"),
            "training_sampling_frequency": trainer_cfg.get(
                "training_sampling_frequency"
            ),
            "init_checkpoint_path": trainer_cfg.get("init_checkpoint_path"),
            "resume_ckpt_path": trainer_cfg.get("resume_ckpt_path"),
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


def main():
    # Get first command line argument as config file
    config_file = sys.argv[1]

    config = _load_config_file(config_file)

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
        record=config["trainer"].get("record", False),
        epochs=config["trainer"]["epochs"],
        dataset=dataset,
        lr_scheduler="default",
        num_annealing_steps=10000,
        num_warmup_steps=config["trainer"].get("num_warmup_steps", 1000),
        logger=None,
        phyla_checkpoint_path=config["trainer"].get("phyla_checkpoint_path"),
        phyla_precomputed_embeddings_path=config["trainer"].get(
            "phyla_precomputed_embeddings_path"
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
        autoregressive_polytomy_choosing_weight=config["trainer"].get(
            "autoregressive_polytomy_choosing_weight", 1.0
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
        velocity_first_hit_false_positive_mass_weight=config["trainer"].get(
            "velocity_first_hit_false_positive_mass_weight", 0.0
        ),
        sampling_discrete_phase_max_phases=config["trainer"].get(
            "sampling_discrete_phase_max_phases", 8
        ),
        training_sampling_frequency=config["trainer"].get(
            "training_sampling_frequency", 200
        ),
        training_sampling_start=config["trainer"].get(
            "training_sampling_start", 500
        ),
        training_sampling_dt_base=config["trainer"].get(
            "training_sampling_dt_base", 0.02
        ),
        sampling_max_steps=config["trainer"].get("sampling_max_steps", 256),
        sampling_max_events=config["trainer"].get("sampling_max_events"),
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
    if config["trainer"].get("record", False):
        _init_wandb_run(config, default_project="phylaflow")
    else:
        trainer_args["logger"] = False

    trainer_args["max_epochs"] = config["trainer"]["epochs"]
    trainer_args["callbacks"] = [
        save_callback,
        threshold_save_callback,
    ]  # For validation callback runs
    val_callback_freq = config["trainer"].get("val_callback_freq", 0)
    if val_callback_freq != 0:
        trainer_args["val_check_interval"] = val_callback_freq
    if config["trainer"].get("limit_val_batches", 0) == 0:
        trainer_args["limit_val_batches"] = 0.0  # Disable validation
    if config["trainer"].get("limit_train_batches") is not None:
        trainer_args["limit_train_batches"] = config["trainer"][
            "limit_train_batches"
        ]
    if config["trainer"].get("max_steps") is not None:
        trainer_args["max_steps"] = int(config["trainer"]["max_steps"])

    trainer_args["accelerator"] = "gpu"
    trainer = Trainer(**trainer_args)
    trainer.fit(
        model,
        train_dataloaders=dataset.train_dataloader(),
        val_dataloaders=dataset.val_dataloader(),
        ckpt_path=resume_ckpt_path,
    )


if __name__ == "__main__":
    main()
