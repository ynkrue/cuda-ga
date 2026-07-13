#!/bin/bash
# =============================================================================
# cuSA Sweep — Baseline
# Solution quality vs. cluster size N, at fixed walkers/iterations.
# =============================================================================
#SBATCH --job-name=cusa_baseline
#SBATCH --output=logs/cusa_%j.out
#SBATCH --nodes=1
#SBATCH --nodelist=piora1,piora2
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

set -u

SA="${SA:-./bin/sa}"
WALKERS="${WALKERS:-1024}"
ITERS="${ITERS:-250}"
N_MIN="${N_MIN:-2}"
N_MAX="${N_MAX:-150}"

RESULTS="sweep/results_baseline.txt"
LOGDIR="sweep/logs_baseline"
mkdir -p "$LOGDIR"

echo "# N,BestEnergy,KnownMin,Gap,GapPercent,Hit,Time_s" > "$RESULTS"

run_one() {
    local N=$1
    local LOGFILE="$LOGDIR/N${N}.log"

    "$SA" --n_atoms "$N" --n_walkers "$WALKERS" --iterations "$ITERS" --seed 42 \
        > "$LOGFILE" 2>&1

    # GapPercent = 100 * gap / |known|, so quality is comparable across N even
    # though the raw energy scale grows with cluster size.
    awk -v n="$N" '
        /^Best energy:/          { best = $3 }
        /^Known minimum:/        { known = $3; hit = ($0 ~ /\[HIT\]/) ? 1 : 0 }
        /^Optimization finished/ { time_s = $4 / 1000.0 }
        END {
            if (known == "") { printf "%d,%.6f,NA,NA,NA,NA,%.3f\n", n, best, time_s; exit }
            gap = best - known
            gap_pct = 100.0 * gap / (known < 0 ? -known : known)
            printf "%d,%.6f,%.6f,%.6f,%.4f,%d,%.3f\n", n, best, known, gap, gap_pct, hit, time_s
        }
    ' "$LOGFILE" >> "$RESULTS"
}

echo "=== Baseline sweep: N=${N_MIN}..${N_MAX}, ${WALKERS} walkers, ${ITERS} iterations ==="
start=$(date +%s)
for N in $(seq "$N_MIN" "$N_MAX"); do
    run_one "$N"
done
end=$(date +%s)

hits=$(awk -F, 'NR>1 && $6==1' "$RESULTS" | wc -l)
total=$(($(wc -l < "$RESULTS") - 1))
echo "Done: ${hits}/${total} hit the known minimum. Wall time: $((end - start))s"
