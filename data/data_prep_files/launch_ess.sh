#!/bin/bash
#SBATCH -c 1
#SBATCH -t 24:00:00
#SBATCH --mem=50G

#SBATCH -p kempner
#SBATCH --account kempner_mzitnik_lab
#SBATCH --gres=gpu:1

#SBATCH --array=0-10                    # 8 parallel shards: IDs 0..7
#SBATCH -o logs/11_17/%A_%a.out
#SBATCH -e logs/11_17/%A_%a.err

module load cuda/12.2
#bash run_until_ess.sh 1803_NT_AL.nex
# bash run_until_ess.sh 2653_NT_AL.nex 

SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}

# Infer total shard count
if [[ -n "${SLURM_ARRAY_TASK_MAX:-}" && -n "${SLURM_ARRAY_TASK_MIN:-}" ]]; then
  STEP=${SLURM_ARRAY_TASK_STEP:-1}
  TOTAL_SHARDS=$(( (SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN) / STEP + 1 ))
else
  TOTAL_SHARDS=1
fi

# ESS_THRESH=${3:-100}
# CHUNK=${4:-50000}

if (( SHARD_INDEX < 0 || SHARD_INDEX >= TOTAL_SHARDS )); then
  echo "SHARD_INDEX must be in [0, $((TOTAL_SHARDS-1))]" >&2
  exit 1
fi


i=0
sharded_any=false

# Use LC_ALL=C to get a deterministic sort order
for nex in $(ls -1 input_files/*.nex 2>/dev/null | LC_ALL=C sort); do
  # select only files for this shard
  if (( (i % TOTAL_SHARDS) == SHARD_INDEX )); then
    sharded_any=true
    echo "=== [shard ${SHARD_INDEX}/${TOTAL_SHARDS}] starting: ${nex} (i=${i}) ==="
    if bash run_until_ess.sh "${nex}"; then
      echo "=== finished: ${nex}; moving to done_files/ ==="
      mv -f "${nex}" done_files/
    else
      echo "!!! failed: ${nex}; leaving in input_files/ for inspection" >&2
    fi
  fi
  ((i++))
done

if ! $sharded_any; then
  echo "[shard ${SHARD_INDEX}/${TOTAL_SHARDS}] No matching .nex files to process."
fi
