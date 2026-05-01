# cuda-ga-optimizer

GPU-accelerated genetic algorithm for real-valued optimization, implemented in CUDA.

## What this is

A parallel genetic algorithm running on the GPU, where each CUDA thread
evaluates one individual in the population. The goal is to make population
sizes practical that would be prohibitively slow on a CPU, and to study
what that buys on hard optimization landscapes.

## Case studies

**Rosenbrock function** — standard benchmark used to develop and validate
the framework. Known global minimum provides a clean measure of convergence
quality.

**Lennard-Jones clusters** — find the minimum-energy arrangement of N atoms
interacting via the Lennard-Jones potential. The primary target is N=13,
whose ground state is a perfect icosahedron. Results are validated against
the Cambridge Cluster Database.

## Status

Work in progress.
