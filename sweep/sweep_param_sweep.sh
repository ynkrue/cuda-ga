#!/bin/bash
# =============================================================================
# cuSA Sweep — Parameter sweep (more walkers x more iterations)
# For each N that missed in the baseline sweep, a 2x2 cartesian grid over
# n_walkers and iterations (1024w/250i is already covered by the baseline).
# TAG suffixes output files so multiple chunks (different N_LIST) can run as
# separate parallel jobs without clobbering each other's results.
# =============================================================================
#SBATCH --job-name=cusa_sweep
#SBATCH --output=logs/cusa_%j.out
#SBATCH --nodes=1
#SBATCH --nodelist=piora1,piora2
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1

set -u

SA="${SA:-./bin/sa}"
TAG="${TAG:-}"

N_LIST="${N_LIST:-69 75 76 77 78 82 85 86 88 89 90 92 93 94 95 96 98 99 100}"

WALKERS_GRID="${WALKERS_GRID:-2048 8192}"
ITERS_GRID="${ITERS_GRID:-500 2000}"

# Only RESULTS needs a per-job suffix (each job truncates its own file at
# start); LOGDIR is shared since log filenames are already unique per N/combo.
RESULTS="sweep/results_param_sweep${TAG:+_$TAG}.txt"
LOGDIR="sweep/logs_param_sweep"
mkdir -p "$LOGDIR"

echo "# N,Walkers,NTemps,NEnsembles,Iterations,BestEnergy,KnownMin,Gap,GapPercent,Hit,Time_s" > "$RESULTS"

run_one() {
    local n=$1 walkers=$2 iters=$3
    local logfile="$LOGDIR/N${n}_w${walkers}_i${iters}.log"

    "$SA" --n_atoms "$n" --n_walkers "$walkers" --iterations "$iters" --seed 42 \
        > "$logfile" 2>&1

    awk -v n="$n" -v iters="$iters" '
        /Walkers.*ensembles/ {
            walkers = $3
            gsub(/[()]/, "", $4); ensembles = $4
            temps = $7
        }
        /^Best energy:/          { best = $3 }
        /^Known minimum:/        { known = $3; hit = ($0 ~ /\[HIT\]/) ? 1 : 0 }
        /^Optimization finished/ { time_s = $4 / 1000.0 }
        END {
            if (known == "") {
                printf "%d,%d,%d,%d,%s,%.6f,NA,NA,NA,NA,%.3f\n", \
                    n, walkers, temps, ensembles, iters, best, time_s
                exit
            }
            gap = best - known
            gap_pct = 100.0 * gap / (known < 0 ? -known : known)
            printf "%d,%d,%d,%d,%s,%.6f,%.6f,%.6f,%.4f,%d,%.3f\n", \
                n, walkers, temps, ensembles, iters, best, known, gap, gap_pct, hit, time_s
        }
    ' "$logfile" >> "$RESULTS"
}

echo "=== Parameter sweep${TAG:+ [$TAG]}: N in {${N_LIST}}, walkers in {${WALKERS_GRID}}, iters in {${ITERS_GRID}} ==="
start=$(date +%s)
for n in $N_LIST; do
    for w in $WALKERS_GRID; do
        for it in $ITERS_GRID; do
            run_one "$n" "$w" "$it"
        done
    done
done
end=$(date +%s)

echo "Done. Wall time: $((end - start))s"
