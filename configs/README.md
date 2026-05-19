# Configs

`final_release.yaml` is the cleaned DS1-DS8 recipe corresponding to the final
reported PhylaFlow model.

`ds2_zenodo_smoke.yaml` is a one-step DS2 launch check for the Zenodo release
bundle. It uses the same final model components but caps training at one step
and points the fixed-path bank to `data/fixed_path_artifacts/DS2`.

The config keeps only the final active components:

- TokenGT-style graph-state encoder, 128 hidden dimension, 4 layers, 8 heads.
- Phyla conditioning from leaf tokens plus global and clade contexts; split-token
  Phyla additions are disabled in the final model.
- Frozen start-tree metric table used by the first-hit and autoregressive heads.
- Structured-subset autoregressive topology head.
- Direct-set first-hit supervision.
- Deterministic discrete-phase exact-boundary sampling.

Before running, set:

```bash
export PHYLAFLOW_RELEASE_ROOT=/path/to/neurips_phylaflow_datasets
export PHYLAFLOW_DATA_ROOT=$PHYLAFLOW_RELEASE_ROOT/data
export PHYLAFLOW_ARTIFACT_ROOT=$PHYLAFLOW_RELEASE_ROOT/artifacts
export PHYLAFLOW_OUTPUT_ROOT=/path/to/write_outputs
```

The expected artifact paths are documented in the top-level README.
