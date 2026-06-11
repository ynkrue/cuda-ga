/**
 * @file kernels.cuh
 * @brief Definitions related to CUDA kernels for simulated annealing.
 *
 * @author Yannik Rüfenacht
 *
 * Parallelism model: one CUDA block drives one independent walker (an SA
 * trajectory over a Lennard-Jones cluster). The threads of a block cooperate
 * to evaluate the O(N^2) energy and forces of that walker's cluster, which is
 * what makes large clusters (hundreds of atoms) tractable.
 *
 * Memory layout: each walker's coordinates are stored contiguously,
 * coords[walker * dimension + (3*a + c)], so a block can stream its cluster
 * into shared memory.
 */

#pragma once

#include "utils.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace cusa {

// Threads per block for the cooperative (block-per-walker) kernels: energy and
// relaxation. One block drives one walker and its threads cooperate over the
// cluster's atoms, so we want roughly one thread per atom. Snapped to a power of
// two (required by the block reductions) and clamped to [32, 256].
inline int coop_threads(int n_atoms) {
    int t = 32;
    while (t < n_atoms && t < 256) t <<= 1;
    return t;
}

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

namespace kernels {
    /**
     * @brief Initializes each walker's cluster with random atom positions.
     * @param coords The device array of walker configurations.
     * @param states The device array of per-walker random states.
     * @param config The configuration struct containing initialization parameters.
     */
    __global__ void init_walkers(double* coords, curandState* states, Config config);

    /**
     * @brief Evaluates the Lennard-Jones energy of each walker's cluster.
     * @param coords The device array of walker configurations.
     * @param energy The array to store the computed energy per walker.
     * @param config The configuration struct.
     * Requires dynamic shared memory of (dimension) doubles.
     */
    __global__ void energy_kernel(const double* coords, double* energy, Config config);

    /**
     * @brief Local minimization of each walker's cluster via gradient descent on the
     * Lennard-Jones potential. This is the local-relaxation step of basin hopping.
     * @param coords The device array of walker configurations.
     * @param config The configuration struct.
     * Requires dynamic shared memory of (2 * dimension) doubles.
     */
    __global__ void relaxation_kernel(double* coords, Config config);

    /**
     * @brief Proposes a trial configuration by randomly displacing every atom of the
     * current configuration. The displacement magnitude is step_size * the walker's
     * temperature, so hot replicas take larger moves.
     * @param current The current accepted configurations.
     * @param trial The output trial configurations.
     * @param states The device array of per-walker random states.
     * @param temps The per-walker temperatures.
     * @param config The configuration struct.
     */
    __global__ void perturb_kernel(const double* current, double* trial, curandState* states,
                                   const double* temps, Config config);

    /**
     * @brief Metropolis acceptance step. Accepts or rejects each walker's trial
     * configuration against its current configuration at the given temperature,
     * and tracks the best configuration found so far per walker.
     * @param current The current accepted configurations.
     * @param trial The trial configurations.
     * @param current_energy The current energies.
     * @param trial_energy The trial energies.
     * @param best The best configurations found so far per walker.
     * @param best_energy The best energies found so far per walker.
     * @param states The device array of per-walker random states.
     * @param temps The per-walker temperatures.
     * @param config The configuration struct.
     */
    __global__ void accept_kernel(double* current, const double* trial,
                                  double* current_energy, const double* trial_energy,
                                  double* best, double* best_energy,
                                  curandState* states, const double* temps, Config config);

    /**
     * @brief Replica-exchange sweep: one thread per ensemble attempts to swap the
     * configurations of each adjacent pair of rungs, accepting with
     * exp((1/T_lo - 1/T_hi)(E_lo - E_hi)).
     * @param current The current configurations (swapped in place on acceptance).
     * @param current_energy The current energies (swapped alongside).
     * @param best The best configurations found so far per walker.
     * @param best_energy The best energies found so far per walker.
     * @param states The device array of per-walker random states.
     * @param temps The per-walker temperatures.
     * @param swap_accepts Device counter incremented once per accepted swap.
     * @param config The configuration struct.
     */
    __global__ void swap_kernel(double* current, double* current_energy,
                                double* best, double* best_energy,
                                curandState* states, const double* temps,
                                unsigned long long* swap_accepts, Config config);

} // namespace kernels

} // namespace cusa