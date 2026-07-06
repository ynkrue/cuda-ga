/**
 * @file kernels.cuh
 * @brief CUDA kernel declarations for parallel-tempering basin hopping.
 *
 * @author Yannik Rüfenacht
 *
 * One block per walker; threads cooperate on the O(N^2) energy/forces.
 * Layout: coords[walker * dimension + (3*a + c)].
 */

#pragma once

#include "utils.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace cusa {

// Threads per block for cooperative kernels: ~1 per atom, power of two.
inline int coop_threads(int n_atoms) {
    int t = 32;
    while (t < n_atoms && t < 256) t <<= 1;
    return t;
}

inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

namespace kernels {
    /// Randomizes each walker's initial cluster geometry.
    __global__ void init_walkers(double* coords, curandState* states, Config config);

    /// Lennard-Jones energy per walker. Shared mem: (dimension + threads) doubles.
    __global__ void energy_kernel(const double* coords, double* energy, Config config);

    /// FIRE local minimization per walker. Shared mem: (3*dimension + 4*threads) doubles.
    __global__ void relaxation_kernel(double* coords, Config config);

    /// Displaces every atom by step_size * temps[w] (uniform, per axis).
    __global__ void perturb_kernel(const double* current, double* trial, curandState* states,
                                   const double* temps, Config config);

    /// Metropolis accept/reject at temps[w]; tracks each walker's best.
    __global__ void accept_kernel(double* current, const double* trial,
                                  double* current_energy, const double* trial_energy,
                                  double* best, double* best_energy,
                                  curandState* states, const double* temps, Config config);

    /// Replica exchange between adjacent rungs: accept with exp((1/T_lo - 1/T_hi)(E_lo - E_hi)).
    __global__ void swap_kernel(double* current, double* current_energy,
                                double* best, double* best_energy,
                                curandState* states, const double* temps,
                                unsigned long long* swap_accepts, Config config);

} // namespace kernels

} // namespace cusa