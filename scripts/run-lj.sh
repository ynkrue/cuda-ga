#!/bin/bash
#SBATCH --job-name=lj
#SBATCH --output=logs/lj_%j.out
#SBATCH --ntasks=1
#SBATCH --gpus=1

echo "Atoms,BestEnergy" > output/lj_summary.log

for atom_count in {2..60}; do
    echo "Running SA for ${atom_count} atoms..."
    srun bin/sa --n_atoms $atom_count --n_walkers 1024 --iterations 1000 --T_init 1.0 --cooling_rate 0.999 --step_size 0.5 --seed 42 | tee >(grep "Best energy" | sed "s/^/[atoms=$atom_count] /" >> output/lj_summary.log)
done