# PhylaFlow

Code and reproduction recipes for **PhylaFlow**, hybrid flow matching for
phylogenetic inference in Billera–Holmes–Vogtmann (BHV) tree space.

This repository contains the full training/eval code and the canonical configs
used to produce the paper's per-dataset (Table 1) and joint DS1–DS8 (Table 12)
results.

## Setup

```bash
pip install -r requirements.txt
```

All configs reference three environment variables, expanded at config-load time
by `run.run`:

| Variable | Contents |
|----------|----------|
| `PHYLAFLOW_DATA_ROOT` | posterior references + fixed-path artifacts (see below) |
| `PHYLAFLOW_ARTIFACT_ROOT` | phyla embeddings, metric start-tables, encoders |
| `PHYLAFLOW_OUTPUT_ROOT` | writable dir for checkpoints, metrics |

```bash
export PHYLAFLOW_DATA_ROOT=/path/to/data
export PHYLAFLOW_ARTIFACT_ROOT=/path/to/artifacts
export PHYLAFLOW_OUTPUT_ROOT=/path/to/outputs
```

### Required data layout

`PHYLAFLOW_DATA_ROOT` must contain:
- `short_run_data_DS1-8/`   — short-run MrBayes posterior references
- `golden_run_data_DS1-8/`  — long-run (golden) posterior references
- `fixed_path_artifacts/DS{1..8}/` — per-case start/target trees and
  `*_fullpathanchors4_*_velocity_anchors.json` boundary-supervision anchors

`PHYLAFLOW_ARTIFACT_ROOT` must contain:
- `phyla_embeddings/DS{1..8}_phyla_beta_embeddings.pt`
- `start_tree_metric_encoder/` with the joint metric start-table and DS2 encoder
  used by the joint config.

## Reproducing Table 1 (per-dataset)

Each dataset trains an independent model. The launcher maps short names to the
canonical `currentrecipe` configs:

```bash
./launch_ds_local.sh ds1     # ... through ds8
```

Convergence metrics (golden/short tree-KL and split-KL) are written to
`$PHYLAFLOW_OUTPUT_ROOT/metrics/` during training.

## Reproducing Table 12 (joint DS1–DS8)

The joint model trains on a single bank spanning all eight datasets. Build the
joint bank once (its case ordering is keyed to the frozen metric start-table),
then launch:

```bash
python scripts/build_joint_bank.py \
  --metric-table $PHYLAFLOW_ARTIFACT_ROOT/start_tree_metric_encoder/ds1ds8_smallbank_joint_metric_start_table_100step_20260505.pt \
  --fixed-path-root $PHYLAFLOW_DATA_ROOT/fixed_path_artifacts \
  --output $PHYLAFLOW_DATA_ROOT/ds1ds8_smallbank_joint_bank.jsonl

./launch_ds_local.sh joint
```

The joint model is evaluated zero-shot on held-out DS2 during training (the
`ds2eval` setting); its best checkpoint is selected on DS2 split-KL.

## Layout

- `run/`     — training entry point (`run.run`) and the training module
- `data/`    — dataset / fixed-pair bank / BHV path construction
- `model/`   — tree transformer, tokenizer, heads
- `phyla/`   — Phyla sequence-embedding model
- `configs/` — the 8 per-dataset configs + the joint config
- `scripts/` — metric-encoder pretraining, start-case probe, joint-bank build, eval
- `utils/`, `tests/`
