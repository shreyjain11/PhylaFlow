#!/usr/bin/env bash
# Launch PhylaFlow training for a per-dataset (Table 1) or the joint DS1-DS8
# (Table 12) model. Configs use ${PHYLAFLOW_DATA_ROOT} / ${PHYLAFLOW_ARTIFACT_ROOT}
# / ${PHYLAFLOW_OUTPUT_ROOT}, which run.py expands at load time. Set those env
# vars before launching.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="$ROOT/configs"
R='currentrecipe_20260425'

declare -A CONFIGS=(
  [ds1]="$CFG/ds1_short_multipair234_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptboth_6000_${R}.yaml"
  [ds2]="$CFG/ds2_short_multipair42_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_${R}.yaml"
  [ds3]="$CFG/ds3_short_multipair243_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_${R}.yaml"
  [ds4]="$CFG/ds4_short_multipair573_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_${R}.yaml"
  [ds5]="$CFG/ds5_short_multipair525_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_${R}.yaml"
  [ds6]="$CFG/ds6_short_multipair219_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_${R}.yaml"
  [ds7]="$CFG/ds7_short_multipair1344_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_${R}.yaml"
  [ds8]="$CFG/ds8_short_multipair1122_topofreqcover_discretephase_terminal_probeparity_wandbclean_termw1_fullpathanchors4_sample1000_edgetopologyterm_caseadaptfhonly_6000_${R}.yaml"
  [joint]="$CFG/local_ds1ds8_smallbank_exactanchors_phy256_leafglobal_cladehead_metricprobe64_fh64_aradd_mlp2cap_s128_lr2e3_ds2eval_mrbayes20k_20260505.yaml"
)

usage() { echo "Usage: ./launch_ds_local.sh {ds1..ds8|joint} [--print-only]"; echo "Targets: ${!CONFIGS[*]}"; }

[[ $# -ge 1 ]] || { usage; exit 1; }
case "${1,,}" in list|--list|-l) printf '%s\n' "${!CONFIGS[@]}"; exit 0;; help|--help|-h) usage; exit 0;; esac

config="${CONFIGS[${1,,}]:-}"
[[ -n "$config" ]] || { echo "Unknown target: $1" >&2; usage >&2; exit 1; }
[[ -f "$config" ]] || { echo "Config not found: $config" >&2; exit 1; }

if [[ "${2:-}" == "--print-only" ]]; then printf 'python -m run.run %q\n' "$config"; exit 0; fi
exec python -m run.run "$config"
