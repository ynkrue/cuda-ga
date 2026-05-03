/**
 * @file kernels.cuh
 * @brief Definitions related to CUDA kernels for genetic algorithm optimization.
 *
 * @author Yannik Rüfenacht
 */

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace cuga {

enum class Mode { Rosenbrock, LennardJones };

struct Config {
    Mode   mode           = Mode::Rosenbrock;
    int    population     = 4096;
    int    n_atoms        = 13;
    int    dimensionality = 2;
    int    generations    = 500;
    int    seed           = 42;

    double mutation_rate  = 0.1;
    double mutation_sigma = 0.1;
    double crossover_rate = 0.7;
    double init_low       = -2.0;
    double init_high      =  2.0;
};

namespace kernels {
    /**
     * @brief CUDA kernel for initializing the population with random values.
     * @param pop The device array to store the initialized population.
     * @param d_states The device array to store the random states.
     * @param config The configuration struct containing initialization parameters.
     */
    __global__ void init_population(double* pop, curandState* d_states, Config config) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= config.population) return;
        // Initialize the random state for each thread
        curandState state;
        curand_init(config.seed, idx, 0, &state);
        d_states[idx] = state;
        // Initialize the population with random values
        for (int j = 0; j < config.dimensionality; ++j) {
            pop[j * config.population + idx] = config.init_low + (config.init_high - config.init_low) * curand_uniform_double(&state);
        }
    }
}

/**
 * @brief CUDA kernel for evaluating the fitness of a population.
 * @param pop The population of candidate solutions.
 * @param fitness The array to store the computed fitness values.
 * @param p The number of individuals in the population.
 * @param n The dimensionality of each individual.
 */
template <typename Fitness>
__global__ void fitness_kernel(const double* pop, double* fitness, int P, int D, Fitness f) {
    // Kernel implementation goes here
}

__global__ void scm_kernel(const double* pop, const double* fitness, double* new_pop, Config config) {
    // Kernel implementation goes here
}

} // namespace cuga
