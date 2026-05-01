# CUDA Genetic Algorithm — Project Sketch

## 1. Motivation

Modern optimization problems in physics and engineering often involve
high-dimensional, non-convex fitness landscapes with an exponential number
of local minima. Classical gradient-based methods get trapped; population-
based methods like the Genetic Algorithm (GA) fare better, but only if the
population is large enough to explore the landscape broadly.

The bottleneck is fitness evaluation. Running a GA with thousands of
individuals on a CPU is prohibitively slow because evaluations are done
serially. A GPU offers thousands of parallel cores, each capable of
evaluating one individual independently and simultaneously. This makes
population sizes that were previously infeasible entirely practical.

**Core claim:** GPU parallelism does not just speed up the GA — it changes
what population sizes are accessible, and population size directly affects
solution quality on hard landscapes.

---

## 2. The Method — Genetic Algorithm

A GA maintains a population of candidate solutions (individuals). Each
individual encodes a point in the search space as a genome. The algorithm
evolves the population over generations:

1. **Evaluate** fitness of every individual
2. **Select** parents preferentially from fitter individuals
3. **Crossover** pairs of parents to produce children
4. **Mutate** children with small random perturbations
5. **Replace** the old population with the new generation (keep the best via elitism)
6. Repeat until convergence

For this project the genome is a real-valued vector — a flat array of
floating-point numbers representing the parameters to be optimized.

### Key operators for real-valued genomes

- **Selection:** tournament selection — each parent is chosen as the best
  among a small random subset of the population. No global communication
  needed; each selection is independent.
- **Crossover:** blend crossover (BLX) — children inherit coordinates
  interpolated between the two parents, with slight extrapolation allowed.
- **Mutation:** Gaussian perturbation — a small random offset drawn from
  a normal distribution is added to each coordinate with some probability.
- **Elitism:** the single best individual is always copied unchanged into
  the next generation, preventing loss of the best found solution.

---

## 3. GPU Parallelisation Strategy

The GA maps naturally onto the GPU because fitness evaluations are
completely independent across individuals — no individual needs to know
about any other during evaluation.

**Mapping:** one CUDA thread evaluates one individual. A kernel launch over
the entire population evaluates all individuals in parallel in a single
pass.

**Pipeline:** each GA phase (evaluate, select, crossover, mutate) is a
separate kernel. The CPU orchestrates the generation loop, launching
kernels in sequence. This keeps concerns cleanly separated and allows the
fitness kernel to be swapped without touching the GA operators.

**Memory layout:** genomes are stored in Structure-of-Arrays (SoA) format
on the GPU — all coordinates of the same dimension across all individuals
are contiguous in memory. This ensures coalesced memory access when threads
in a warp read their genome data simultaneously.

**Scalability:** because each thread is independent, the framework scales
directly with population size. Doubling the population doubles the number
of active threads and barely changes wall-clock time per generation — the
GPU absorbs the extra work through parallelism.

---

## 4. Case Study 1 — Rosenbrock Function (Development Benchmark)

### Problem

The Rosenbrock function is a standard mathematical benchmark for
optimization algorithms:

```
F(x) = sum_i [ 100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2 ]
```

The global minimum is at x = (1, 1, ..., 1) with F = 0. The landscape
features a narrow curved valley — easy to find, hard to follow — that
reliably traps gradient methods and tests an optimizer's ability to
navigate non-trivial curvature.

### Role in the project

Rosenbrock serves as the development and validation benchmark:

- **Known ground truth** — the exact global minimum is known analytically,
  so convergence quality is directly measurable
- **Scalable dimension** — running at N=2 allows the landscape to be
  visualized; N=10, N=50, N=100 stress-tests scaling behavior
- **No domain knowledge required** — clean separation between optimizer
  development and physics problem setup
- **Hyperparameter tuning** — population size, mutation rate, crossover
  probability are tuned here before applying the framework to LJ

### Expected results

Convergence plots (fitness vs. number of evaluations) for several
population sizes. Demonstration that larger populations, made accessible
by the GPU, find the global minimum more reliably and in fewer generations.

---

## 5. Case Study 2 — Lennard-Jones Cluster (Physics Problem)

### Background

The Lennard-Jones (LJ) potential describes the interaction between a pair
of neutral atoms separated by distance r:

```
V(r) = 4ε [ (σ/r)^12 - (σ/r)^6 ]
```

The repulsive term (r^-12) dominates at short range; the attractive term
(r^-6) dominates at long range. The two balance at an equilibrium distance
where energy is minimized at -ε.

### Problem

Given N atoms in three-dimensional space, find the arrangement of atomic
positions that minimizes the total pairwise potential energy:

```
E_total = sum_{i<j} V(r_ij)
```

This is the **cluster ground state problem**. The genome encodes the (x,y,z)
coordinates of all N atoms as a flat real-valued vector of length 3N.
The fitness is -E_total (negated because the GA maximizes).

### Why it is hard

For N atoms there are 3N continuous degrees of freedom. The energy
landscape has an exponential number of local minima — metastable
configurations that look optimal locally but are not the true ground state.
For N=13, the true ground state is a perfect icosahedron; thousands of
near-optimal traps exist that gradient methods and small-population
algorithms fall into.

### Validation

The Cambridge Cluster Database publishes exact known minimum energies for
LJ clusters up to N > 1000. Results can be directly compared against these
reference values. Selected landmark cluster sizes:

| N  | Known ground state geometry | Reference energy (units of ε) |
|----|----------------------------|-------------------------------|
| 7  | Pentagonal bipyramid       | −16.505                       |
| 13 | Icosahedron                | −44.327                       |
| 19 | Double icosahedron         | −72.660                       |
| 38 | Truncated octahedron       | −173.928                      |

Primary target: **N=13** — small enough for large populations, hard enough
for small ones, and the icosahedron result is physically meaningful and
visually recognizable.

### Role in the project

The LJ cluster problem demonstrates the real-world value of the GPU-GA
framework. The core argument is:

> The LJ landscape has sufficiently many local minima that only a
> population large enough to simultaneously explore many basins can
> reliably find the global minimum. The GPU makes such populations
> practical.

---

## 6. Experiments and Evaluation

For both case studies the same set of experiments is performed:

- **Convergence study:** fitness vs. number of evaluations for multiple
  independent runs; plot best-so-far and population mean
- **Population size study:** vary population size over a wide range and
  record final solution quality; this is where the GPU advantage is
  directly visible
- **Hyperparameter sensitivity:** mutation rate, crossover probability,
  tournament size — reproduce the fitness-vs-parameter analysis from
  lecture
- **Validation:** compare best found solution against known reference
  (Rosenbrock: F=0 at (1,...,1); LJ: Cambridge Cluster Database energies)
- **Visualisation:** 2D Rosenbrock landscape with optimization trajectory;
  3D cluster geometry for LJ ground states

---

## 7. Discussion

- What does the GPU concretely buy in terms of solution quality?
- At what population size does the GA begin to reliably find the global
  minimum for LJ N=13?
- How do elitism, mutation rate, and diversity interact on a rugged
  landscape?
- Limitations: the GA makes no use of gradient information; a hybrid
  approach (local gradient polish after crossover) could improve results
  significantly — this is noted as a natural extension
- Further extensions: island model for additional diversity, adaptive
  mutation rate, scaling to larger cluster sizes

---

## Summary

| | Rosenbrock | Lennard-Jones |
|---|---|---|
| **Type** | Mathematical benchmark | Physics / molecular science |
| **Genome** | Real vector, length N | Real vector, length 3N |
| **Fitness** | Analytical formula | Pairwise LJ sum |
| **Validation** | Known exact minimum | Cambridge Cluster Database |
| **Purpose** | Develop and tune framework | Demonstrate real-world value |

One framework. One genome encoding. Two problems of increasing physical
relevance.
