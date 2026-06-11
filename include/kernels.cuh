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

// Threads per block for the flat (thread-per-walker) kernels, where each thread
// owns one walker: init, perturb, accept.
constexpr int WALKER_THREADS = 256;

// Threads per block for the cooperative (block-per-walker) kernels: energy and
// relaxation. One block drives one walker and its threads cooperate over the
// cluster's atoms, so we want roughly one thread per atom. Snapped to a power of
// two (required by the block reductions) and clamped to [32, 256].
inline int coop_threads(int n_atoms) {
    int t = 32;
    while (t < n_atoms && t < 256) t <<= 1;
    return t;
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
     * @brief Proposes a trial configuration by randomly displacing the atoms of the
     * current configuration (perturbation move).
     * @param current The current accepted configurations.
     * @param trial The output trial configurations.
     * @param states The device array of per-walker random states.
     * @param config The configuration struct.
     */
    __global__ void perturb_kernel(const double* current, double* trial, curandState* states, Config config);

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
     * @param temperature The current annealing temperature.
     * @param config The configuration struct.
     */
    __global__ void accept_kernel(double* current, const double* trial,
                                  double* current_energy, const double* trial_energy,
                                  double* best, double* best_energy,
                                  curandState* states, double temperature, Config config);

    /**
     * @brief Computes aggregate statistics (best, worst, average) of the per-walker
     * best energies, for logging. Launched as a single block.
     * @param best_energy The array of best energies per walker.
     * @param config The configuration struct.
     * @param stats Output array of 3 doubles: best, worst, average.
     */
    __global__ void statistics_kernel(const double* best_energy, Config config, double* stats);

} // namespace kernels

} // namespace cusa