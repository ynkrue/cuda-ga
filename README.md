# cuda-sa-optimizer

GPU-accelerated parallel-tempering basin hopping for Lennard-Jones cluster
optimization, written in CUDA C++.

## Problem

Find the minimum-energy arrangement of N atoms interacting via the Lennard-Jones
potential. The number of local minima grows exponentially with N, so a single
search is easily trapped far from the global minimum. We run a large population of
independent walkers in parallel on the GPU and keep the best structure any of them
finds.

## Method

Walkers are split into independent ensembles of `n_temps` replicas held at constant
temperatures on a geometric ladder from `T_min` to `T_max`. Each step, every replica:

1. **Perturb.** Displace every atom by a uniform random vector in `[-step, step]³`,
   with `step = step_size * T` so hot replicas take larger moves.
2. **Relax.** Locally minimize the trial with FIRE (damped MD with an adaptive
   timestep), run to a force tolerance.
3. **Accept.** Metropolis test `ΔE ≤ 0 or rand < exp(-ΔE/T)` at the replica's own
   temperature; track each walker's best structure.

Then adjacent rungs attempt to **swap** configurations, accepting with
`exp((1/T_lo - 1/T_hi)(E_lo - E_hi))`. Hot replicas cross funnel barriers and the
swaps pull the good structures they find down to the cold rungs.

## Implementation

- **Layout:** each walker's coordinates are stored contiguously
  (`coords[walker * 3N + 3a + c]`), so a block can stream its cluster into shared memory.
- **Launch regimes** (`src/kernels.cu`):
  - *Cooperative* (`energy`, `relaxation`): one block per walker, threads cooperate
    over atoms for the O(N²) energy/forces and block-reduce. Threads scale with N
    (`coop_threads`, a power of two in [32, 256]).
  - *Flat* (`init`, `perturb`, `accept`): one thread per walker.
  - *Swap* (`swap_kernel`): one thread per ensemble.
- **RNG:** one cuRAND state per walker.

The run logs `Best`, `Basins` (distinct energy levels in the population), `InBest`
(walkers in the lowest basin), and `Swap%` (replica-exchange acceptance).

## Build & run

```sh
make                       # needs nvcc; built for sm_80 (A100)
./bin/sa --config configs/lj.ini
./bin/sa --n_atoms 38 --n_walkers 2048 --iterations 5000 --seed 42
```

`n_temps`, `T_min`, `T_max`, `step_size` and `fire_max_steps` auto-derive in
`Config::finalize()` (see `include/utils.hpp`) — the only knobs you normally need
are `n_atoms`, `n_walkers`, `iterations` and `seed`. Pass any of the derived ones
explicitly (config file or CLI) to override.

`sweep/sweep_baseline.sh` sweeps N=2..150 under SLURM.

## Validation

The best energy is compared at the end against the putative global minima in
`reference/lj_minima.data` (N = 2 to 150, from the
[Cambridge Cluster Database](https://doye.chem.ox.ac.uk/jon/structures/LJ/tables.150.html)).
The search itself never sees these values.

## Possible improvements

- **Smarter moves:** angular/surface moves or moving only high-energy atoms may
  cross funnels better than uniform displacement, for the double-funnel cases
  (LJ75-77, 98) that stay hard even with correctly-tuned search parameters.
- **Local minimizer:** L-BFGS reaches a minimum in fewer force evaluations than FIRE.
- **Scaling to large N:** replace the O(N²) all-pairs energy with cutoff and
  cell/neighbor lists.

## Status

Work in progress.
