# cuda-sa-optimizer

GPU-accelerated parallel basin hopping with simulated annealing for Lennard-Jones
cluster optimization, implemented in CUDA.

## What this is

Many independent **walkers** explore the energy landscape in parallel on the GPU.
Each walker runs a basin-hopping/simulated-annealing trajectory: perturb the
current structure, **locally minimize** it, then accept or reject the result with
a Metropolis criterion whose temperature is gradually cooled. Running thousands of
walkers at once dramatically raises the chance that at least one reaches the global
minimum of a rugged landscape.

## Problem

**Lennard-Jones clusters** — find the minimum-energy arrangement of N atoms
interacting via the LJ potential. Results are validated against the
[Cambridge Cluster Database](http://www-wales.ch.cam.ac.uk/CCD.html). The code
reproduces the known global minima for the easy/magic cases (e.g. LJ13 −44.3268,
LJ38 −173.9284, LJ55 −279.2485) and is being pushed toward the harder cases.

## Algorithm

One basin-hopping step, applied to every walker:

1. **Perturb** — displace every atom of the current structure by a uniform random
   vector in `[-step, step]³`.
2. **Relax** — locally minimize the trial structure with **FIRE** (Fast Inertial
   Relaxation Engine): damped molecular dynamics with an adaptive timestep, run to
   a force tolerance. This turns the energy landscape into its staircase of local
   minima, which is what makes basin hopping effective.
3. **Accept** — Metropolis test `ΔE ≤ 0 or rand < exp(−ΔE/T)`; track each walker's
   best structure so far.
4. **Cool** — geometric schedule `T ← T · cooling_rate`.

At the end the globally best structure across all walkers is written to
`output/output.csv`.

## Implementation notes

- **Layout:** each walker's coordinates are stored contiguously
  (`coords[walker * 3N + 3a + c]`), so a block can stream its cluster into shared
  memory.
- **Two launch regimes** (see `src/kernels.cu`):
  - *Cooperative* (`energy`, `relaxation`): **one block per walker**; the block's
    threads cooperate over atoms to evaluate the O(N²) energy/forces and
    block-reduce the result. Thread count scales with N (`coop_threads`, a power
    of two in [32, 256]); reduction scratch lives in dynamic shared memory.
  - *Flat* (`init`, `perturb`, `accept`): **one thread per walker** — the work is
    O(N) bookkeeping that needs no cooperation.
- **RNG:** one cuRAND state per walker.

## Build & run

```sh
make                       # needs nvcc; built for sm_80 (A100)
./bin/sa --config configs/lj.ini
./bin/sa --n_atoms 38 --n_walkers 1024 --iterations 1000 --T_init 1.0 \
         --cooling_rate 0.999 --step_size 0.5 --seed 42
```

`scripts/run-lj.sh` sweeps a range of cluster sizes under SLURM.

## Possible improvements (from the literature)

The current method is textbook basin hopping with annealing. Known refinements,
roughly in order of expected payoff for LJ clusters:

- **Acceptance / temperature.** Canonical basin hopping (Wales & Doye, *J. Phys.
  Chem. A* 1997) uses a *constant* temperature (kT ≈ 0.8 ε) rather than cooling to
  zero, which avoids freezing walkers in the first basin they find. Adapting the
  **step size** to hold the acceptance ratio near ~50% is standard and cheap.
- **Local minimizer.** **L-BFGS** typically reaches a minimum in far fewer force
  evaluations than FIRE/GD; since relaxation dominates the runtime, this is the
  biggest single-walker speedup.
- **Smarter moves.** Angular/surface ("shell") moves and moving only
  high-energy/misplaced atoms cross funnels better than uniform displacement —
  important for the double-funnel cases (LJ38, LJ75–77, LJ98).
- **Population coupling.** Replica exchange / **parallel tempering** lets cold and
  hot walkers swap configurations; **minima hopping** (Goedecker 2004) uses MD
  escape with history feedback to avoid revisiting basins; cut-and-splice genetic
  operators (Deaven & Ho 1995) recombine good fragments.
- **Seeding.** Biasing initial structures toward known motifs (icosahedral, Marks
  decahedral, fcc) accelerates the magic-number sizes.
- **Scaling to large N.** Replace the O(N²) all-pairs energy/force with a **cutoff
  + cell/neighbor lists** (O(N)); this matters most for clusters of hundreds of
  atoms, where shared-memory tiling of the pair loop also helps.

The famously hard sizes (LJ38 fcc truncated octahedron, LJ75–77 Marks decahedra,
LJ98 "Leary tetrahedron") are where the move set and population coupling above make
the difference between occasionally and reliably finding the global minimum.

## Status

Work in progress.
