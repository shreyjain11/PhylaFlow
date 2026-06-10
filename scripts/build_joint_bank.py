#!/usr/bin/env python3
"""Build the DS1-DS8 joint fixed-pair bank (JSONL) used for joint training.

The joint training config references a single JSONL bank that lists, for every
case across DS1-DS8, the (start_tree, target_tree) pair plus its dataset id and
group key. The case ordering must match the frozen metric start-table so that
first-hit / autoregressive case-index conditioning lines up.

This script reconstructs that bank deterministically from two inputs that ship
with the data/artifact release:
  1. the frozen joint metric start-table (.pt), whose metadata.source_group_keys
     defines the canonical case ordering, and
  2. the per-dataset fixed-path artifacts directory, which contains the
     '<group_key>_start.json' / '<group_key>_target.json' files.

Usage:
  python scripts/build_joint_bank.py \
      --metric-table $PHYLAFLOW_ARTIFACT_ROOT/start_tree_metric_encoder/ds1ds8_smallbank_joint_metric_start_table_100step_20260505.pt \
      --fixed-path-root $PHYLAFLOW_DATA_ROOT/fixed_path_artifacts \
      --output $PHYLAFLOW_DATA_ROOT/ds1ds8_smallbank_joint_bank.jsonl
"""
import argparse, json, os, sys
import torch


def _load_newick(path):
    data = json.load(open(path))
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for k in ('newick', 'tree', 'newick_tree', 'start_tree', 'target_tree'):
            if k in data:
                return data[k]
        if len(data) == 1:
            return next(iter(data.values()))
    raise ValueError('cannot parse newick from %s' % path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--metric-table', required=True,
                    help='frozen joint metric start-table .pt (defines case ordering)')
    ap.add_argument('--fixed-path-root', required=True,
                    help='dir containing per-dataset DS*/<group_key>_{start,target}.json')
    ap.add_argument('--output', required=True, help='output JSONL path')
    args = ap.parse_args()

    table = torch.load(args.metric_table, map_location='cpu', weights_only=False)
    group_keys = table['metadata']['source_group_keys']
    print('frozen table cases: %d' % len(group_keys))

    n_written, missing = 0, 0
    with open(args.output, 'w') as out:
        for gk in group_keys:
            ds = gk.split('_')[0].upper()  # 'ds1...' -> 'DS1'
            sp = os.path.join(args.fixed_path_root, ds, gk + '_start.json')
            tp = os.path.join(args.fixed_path_root, ds, gk + '_target.json')
            if not (os.path.exists(sp) and os.path.exists(tp)):
                missing += 1
                continue
            out.write(json.dumps({
                'dataset_id': ds,
                'group_key': gk,
                'source_start_json': sp,
                'source_target_json': tp,
                'start_tree': _load_newick(sp),
                'target_tree': _load_newick(tp),
            }) + '\n')
            n_written += 1

    print('wrote %d records to %s' % (n_written, args.output))
    if missing:
        print('WARNING: %d cases missing start/target files' % missing, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
