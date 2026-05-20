# Method To Code Map

This document links the paper-level method components to the release source.

## Tree And Split Metrics

`utils/metric_utils.py` implements topology metrics. Tree-KL and Split-KL are
computed over smoothed finite supports using the same canonical topology and
split encodings used by the sampler. A split is an unordered bipartition of the
leaf set induced by an internal edge; the split and its complement are
canonicalized to the same key.

## Random Start Trees

`data/dataset.py` calls the random tree generator in `utils/random_tree.py`.
The generator builds an unrooted topology by starting from a three-leaf tree and
then repeatedly selecting an existing edge uniformly, subdividing it, and
attaching the next leaf. Internal branch lengths are sampled in the generator;
the rooted encoding includes zero-length root scaffold edges as implementation
bookkeeping.

## Tokenization And Encoder

`model/treeTokenizer.py` converts a Newick state into node tokens and edge
tokens. `model/model.py` adds the graph token, time/phase input, Phyla
conditioning, and transformer trunk.

The final config uses Phyla leaf-token additions and global/clade context for
heads, but disables split-token Phyla additions.

## Transition Heads

`model/model.py` contains:

- `StructuredSubsetMergeHead`: autoregressive topology head. It scores starter
  pairs, predicts merge cardinality, and scores component membership.
- First-hit modules in `TreeDenoiserTokenGT`: per-active-internal-edge logits
  for the next collapsing split set, with phase and frozen start-tree context.
- Velocity head in `TreeDenoiserTokenGT`: per-edge branch-length derivative.

The frozen start-tree code is consumed by the first-hit and autoregressive
heads. It is not prepended as a token to the graph-transformer input.

## Losses

`run/TrainingModule.py` implements:

- velocity regression in fixed BHV orthants,
- first-hit set BCE and false-positive mass penalties,
- log hitting-time penalties,
- structured-subset autoregressive loss,
- separate optimizer steps for velocity and autoregressive components.

Full-path control samples are generated from the fixed start/target path banks
inside `data/dataset.py`. The final configuration trains on those velocity
states and the corresponding structured-subset autoregressive merge targets.

## Sampling

`run/TrainingModule.py::sample` is the event-based sampler. With the final
settings it uses discrete phase input, exact-boundary stepping, deterministic
first-hit decoding, and structured-subset topology resolution. New splits
inserted by the topology head are assigned birth length `1e-3`.

## PhylaFlow-MCMC

The split-guided PhylaFlow-MCMC refinement kernel described in the paper is not
implemented in this initial release. The current repository includes direct
PhylaFlow generation and MrBayes comparison/evaluation utilities, but not the
proposal-guided NNI Metropolis-Hastings sampler used for those refinement
experiments. The matched PhyloGFN-MCMC comparison is also planned as follow-up
release code.

## Paper Baselines And Table Harnesses

The initial release does not include the full paper-table orchestration layer:
external baseline wrappers, long MrBayes refinement runs, IQ-TREE likelihood
diagnostics, DS1 ablations, and joint sequence-conditioning / cross-conditioning
drivers are planned follow-up items. The current code focuses on the cleaned
PhylaFlow DS1-DS8 training/sampling path and the release metric utilities.

## Start-Tree Metric Encoder

`scripts/pretrain_start_tree_metric_encoder.py` trains the frozen split-set
metric encoder. The encoder represents a tree by its internal split bitmasks,
pools per-split features, and trains against normalized Robinson-Foulds
distance using similarity, distance-regression, bin-classification, and VICReg
regularization terms.

## Branch-Length Relaxer

`scripts/train_branch_relaxer.py` trains the standalone branch-length relaxer.
It reuses a small graph-transformer trunk and predicts additive branch-length
deltas toward topology-frozen MrBayes warmup branch lengths. `scripts/jc_likelihood.py`
provides the JC likelihood scorer used for heldout relaxed likelihood.

At evaluation time, `run/TrainingModule.py` applies the relaxer once after
topology generation. The relaxer changes only branch lengths and never changes
the generated split set.
