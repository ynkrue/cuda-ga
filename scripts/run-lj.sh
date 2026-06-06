#!/bin/bash
#SBATCH --job-name=lj
#SBATCH --output=logs/lj_%j.out
#SBATCH --ntasks=1
#SBATCH --gpus=1

echo "Atoms,BestFitness" > output/lj_summary.log

for atom_count in {2..60}; do
    echo "Running GA for ${atom_count} atoms..."
    srun bin/ga --n_atoms $atom_count --population 10000 --generations 500 --tournament_k 2 --crossover_rate 0.8 --mutation_rate 0.3 --seed 42 | tee >(grep "Best fitness" | sed "s/^/[atoms=$atom_count] /" >> output/lj_summary.log)
done