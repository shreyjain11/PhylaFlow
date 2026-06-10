#!/usr/bin/env bash
set -euo pipefail

NEX=${1:? "Give .nex file"}        # e.g., 1803_NT_AL_DNA.nex
# BASE=${NEX%.nex}                   # base name
BASE="$(basename "$NEX" .nex)"
ESS_THRESH=${2:-100}               # default 100
CHUNK=${3:-50000}                  # gens per chunk (tweak to 50k–100k)

OUTDIR="output/${BASE}"
PREFIX="${OUTDIR}/${BASE}"         # matches your mcmcp filename=output/<BASE>/<BASE>
mkdir -p "${OUTDIR}"

# GPU-friendly env
export BEAGLE_OPENCL_DISABLED=1
export BEAGLE_RESOURCE_ORDER=CUDA,CPU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

COUNTER_FILE="${PREFIX}.chunks"

# Initialize counter if missing
if [[ ! -f "$COUNTER_FILE" ]]; then
  echo 0 > "$COUNTER_FILE"
fi

# One MrBayes chunk as a here-doc (append results)
run_first_chunk() {
  local target_ngen="$1"
  mb <<EOF
execute ${NEX};
mcmc ngen=${target_ngen};
quit
EOF
}

run_last_chunk() {
  local target_ngen="$1"
  mb <<EOF
execute ${NEX};
mcmcp append=yes;
mcmc ngen=${target_ngen};
quit
EOF
}

run_summary() {
  rm -f "${PREFIX}_DNA".{pstat,tstat,vstat,con.tre,trprobs} 2>/dev/null || true

  mb <<EOF
execute ${NEX};
sump filename=${PREFIX}_DNA relburnin=yes burninfrac=0.25 all=yes;    
sumt filename=${PREFIX}_DNA relburnin=yes burninfrac=0.25; 
quit
EOF
}


# First chunk (creates .p/.t/.pstat/.lstat)
chunks=$(<"$COUNTER_FILE")
target_ngen=$(( (chunks + 1) * CHUNK ))
run_first_chunk "$target_ngen"
run_summary
echo $((chunks + 1)) > "$COUNTER_FILE"

# Loop: check ESS from .pstat and keep appending until threshold met
while true; do
  PSTAT="${PREFIX}_DNA.pstat"
  if [[ ! -f "$PSTAT" ]]; then
    echo "No ${PSTAT} yet; running another chunk..."
    chunks=$(<"$COUNTER_FILE")
    target_ngen=$(( (chunks + 1) * CHUNK ))
    run_last_chunk "$target_ngen"
    run_summary
    echo $((chunks + 1)) > "$COUNTER_FILE"
    continue
  fi

  if python3 check_ess.py "$PSTAT" "$ESS_THRESH"; then
    echo "✅ ESS met: $(python3 check_ess.py "$PSTAT" "$ESS_THRESH" | sed -E 's/.*MIN_ESS=([0-9.]+).*AVG_ESS=([0-9.]+).*/min=\1 avg=\2/')"
    break
  fi

  echo "ESS below threshold; appending ${CHUNK} more generations…"
  chunks=$(<"$COUNTER_FILE")
  target_ngen=$(( (chunks + 1) * CHUNK ))
  run_last_chunk "$target_ngen"
  run_summary
  echo $((chunks + 1)) > "$COUNTER_FILE"
done

echo "Done. Results in ${OUTDIR}/ (prefix ${PREFIX})"
