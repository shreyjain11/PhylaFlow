# PhylaFlow 

## Quick Start

Create an environment with Python 3.10+ and install the dependencies:

```bash
pip install -r requirements.txt
```

Download the required release dataset bundle from Zenodo:

- Record: https://zenodo.org/records/20297912
- DOI: `10.5281/zenodo.20297912`
- File: `phylaflow_data.tar.gz`
- MD5: `2acd0a99c2bc9182e5318b0c3fa41659`
- Extracted size: about `3.3G`

Training and reproduction require this bundle. It contains the short-run
posterior references, fixed start/target path artifacts, precomputed Phyla
embeddings, frozen start-tree tables, and evaluation artifacts used by the
release configs.

Example install under `/ewsc`:

```bash
mkdir -p /ewsc/$USER/phylaflow_zenodo_20297912
cd /ewsc/$USER/phylaflow_zenodo_20297912

curl -L \
  -o phylaflow_data.tar.gz \
  https://zenodo.org/api/records/20297912/files/phylaflow_data.tar.gz/content

printf '2acd0a99c2bc9182e5318b0c3fa41659  phylaflow_data.tar.gz\n' | md5sum -c -
tar -xzf phylaflow_data.tar.gz

# Optional cleanup for archives extracted on Linux from macOS-created tarballs.
find neurips_phylaflow_datasets -name '._*' -delete
find neurips_phylaflow_datasets -name '.DS_Store' -delete
```

Set the release roots:

```bash
export PHYLAFLOW_RELEASE_ROOT=/ewsc/$USER/phylaflow_zenodo_20297912/neurips_phylaflow_datasets
export PHYLAFLOW_DATA_ROOT=$PHYLAFLOW_RELEASE_ROOT/data
export PHYLAFLOW_ARTIFACT_ROOT=$PHYLAFLOW_RELEASE_ROOT/artifacts
export PHYLAFLOW_OUTPUT_ROOT=/path/to/write_outputs
```

Run the verified DS2 one-step smoke training:

```bash
CUDA_VISIBLE_DEVICES=0 python -m run.run configs/ds2_zenodo_smoke.yaml
```

The smoke config uses the final cleaned model components, loads DS2 data and
Phyla embeddings from the Zenodo bundle, writes to
`$PHYLAFLOW_OUTPUT_ROOT/checkpoints/ds2_zenodo_smoke`, and stops after one
training step.

The final cleaned DS1-DS8 recipe is:

```bash
configs/final_release.yaml
```

To launch the full recipe:

```bash
python -m run.run configs/final_release.yaml
```

The config uses environment-variable expansion, so artifact and data roots are
resolved at load time.

## Initial arXiv Release Scope

This repository is intended to accompany the first arXiv preprint. The release
path is the cleaned DS1-DS8 training and sampling code in
`configs/final_release.yaml`, plus the DS2 smoke check above. The code has been
trimmed to the components used by that path: fixed DS start/target banks,
full-path velocity and autoregressive supervision, Phyla conditioning, the
first-hit head, structured-subset topology resolution, optional branch-length
relaxation utilities, and the metric code used for release checks.

This initial release is not a full paper-table reproduction harness. The main
follow-up items are:

- Guided-refinement kernels: the split-guided PhylaFlow-MCMC NNI/MH sampler
  reported in the paper, plus the matched PhyloGFN-MCMC comparison.
- Paper-table orchestration: external baseline wrappers, long MrBayes
  refinement runs, IQ-TREE likelihood diagnostics, DS1 ablations, and the
  joint sequence-conditioning / cross-conditioning experiment drivers.
- Convenience interfaces: a checkpoint-only sampling/evaluation CLI,
  one-command table reproduction, broader custom-dataset recipes, and Lightning
  `test_dataloader()` / `predict_dataloader()` entry points.

## Expected Artifacts

`PHYLAFLOW_DATA_ROOT` is expected to contain:

- `short_run_data_DS1-8/`: posterior topology reference files used by the data
  loader and metric code.
- `golden_run_data_DS1-8/`: longer posterior references used by the reported
  comparison metrics.
- `fixed_path_artifacts/`: fixed start/target banks used by final-model
  training and sampling checks.
- `mrbayes20k_pickles/`: dataset pickles used by MrBayes 20k evaluation.

`PHYLAFLOW_ARTIFACT_ROOT` is expected to contain:

- `start_tree_metric_encoder/ds1ds8_joint_metric_start_table.pt`: frozen
  start-tree embedding table used during final-model training.
- `start_tree_metric_encoder/start_tree_metric_encoder.pt`: frozen encoder
  checkpoint used for unseen-start evaluation.
- `phyla_embeddings/`: precomputed per-dataset Phyla embedding banks.
- `branch_relaxer/standalone_branch_relaxer.pt`: optional branch-length
  relaxer checkpoint for likelihood-sensitive evaluation.
- `best_model/`: released model checkpoint artifacts.

## Code Map

For a paper-method crosswalk, see `docs/method_to_code_map.md`.

- `configs/final_release.yaml`: final DS1-DS8 model recipe with anonymized paths.
- `model/treeTokenizer.py`: TokenGT-style tree tokenization into node and edge
  tokens, including branch-length encodings and Laplacian positional features.
- `model/model.py`: graph transformer, Phyla conditioning, velocity head,
  first-hit head, structured-subset autoregressive topology head, and frozen
  start-tree adapters.
- `data/dataset.py`: random start-tree generation, posterior/reference loading,
  fixed-pair sampling, and full-path oracle samples.
- `run/TrainingModule.py`: training losses, first-hit set supervision,
  deterministic event-based sampler, sample metrics, unseen-start evaluation,
  and branch-relaxer application.
- `utils/bhv_utils.py`: BHV geodesic boundary paths, sampled orthant velocities,
  and oracle topology-transition targets.
- `utils/bhv_movie.py`: split-mask/Newick reconstruction helpers.
- `utils/metric_utils.py`: Tree-KL, Split-KL, RF, support recall, and branch
  length/likelihood diagnostics.
- `utils/random_tree.py`: random unrooted start topology generator and branch
  length initialization.
- `scripts/pretrain_start_tree_metric_encoder.py`: frozen start-tree metric
  encoder pretraining and table export.
- `scripts/train_branch_relaxer.py`: standalone branch-length relaxer training.
- `scripts/jc_likelihood.py`: small JC pruning likelihood used by relaxer
  evaluation utilities.
- `tests/`: focused tests for metrics, tokenization, Phyla fusion, BHV sampling,
  encoding consistency, and structured-subset decoding.

## Final Model Components

The final model uses a 4-layer, 8-head, 128-dimensional graph transformer. Each
tree state is represented as node and edge tokens, with a learned graph token
prepended before the transformer.

Phyla conditioning is added through leaf-token embeddings and through global
and clade-level context features used by the transition heads. Split-token
Phyla additions are disabled in the final configuration.

The velocity head predicts within-orthant branch-length velocities. The
first-hit head predicts the active internal split coordinates that collapse at
the next BHV boundary. The autoregressive head is invoked after a collapse
creates a polytomy; it predicts a structured subset of incident components to
merge, inserts the corresponding split, and repeats until the topology is fully
resolved or the event budget is exhausted.

The frozen start-tree metric encoder is not a graph-token input. Its exported
64-dimensional code conditions the first-hit and autoregressive heads through
separate adapter MLPs. During unseen-start evaluation, the same frozen encoder
generates embeddings on the fly for starts not present in the training table.

## Sampling

Sampling is deterministic in the reported configuration. At each discrete phase,
the sampler predicts velocities and first-hit logits, selects the positive-logit
first-hit set with argmax fallback, advances exactly to the predicted boundary,
clamps selected internal edges to zero, and resolves any resulting polytomy with
the autoregressive head. The default budgets are 8 phases, 128 continuous rollout
steps, and 500 autoregressive split-insertion events. New splits inserted by the
topology head receive birth length `1e-3`.

The optional branch-length relaxer is applied only after topology generation.
It preserves the split set exactly and changes only branch lengths, so topology
metrics such as Tree-KL, Split-KL, and RF are unaffected by relaxation.

## Tests

Run the lightweight source tests with:

```bash
python -m pytest tests -q
```

Some likelihood fixture tests are skipped unless `PHYLAFLOW_EXAMPLE_DATA_ROOT`
is set to a small local Nexus/MrBayes fixture directory.
