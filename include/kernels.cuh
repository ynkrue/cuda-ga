/**
 * @file kernels.cuh
 * @brief Definitions related to CUDA kernels for genetic algorithm optimization.
 *
 * @author Yannik Rüfenacht
 */

#pragma once

#include "utils.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace cuga {

namespace kernels {
    /**
     * @brief CUDA kernel for initializing the population with random values.
     * @param pop The device array to store the initialized population.
     * @param states The device array to store the random states.
     * @param config The configuration struct containing initialization parameters.
     */
    __global__ void init_population(double* pop, curandState* states, Config config);

    /**
    * @brief CUDA kernel for evaluating the fitness of a population.
    * @param pop The population of candidate solutions.
    * @param fitness The array to store the computed fitness values.
    * @param config The configuration struct containing parameters for fitness evaluation.
    */
    __global__ void fitness_kernel(const double* pop, double* fitness, Config config);

    /**
     * @brief CUDA kernel for the selection, crossover, and mutation operations.
     * @param pop The current population of candidate solutions.
     * @param new_pop The new population generated through selection, crossover, and mutation.
     * @param fitness The array of fitness values for the current population.
     * @param config The configuration struct containing parameters for the genetic operations.
     */
    __global__ void selection_kernel(const double* pop, double* mating_pool, const double* fitness, curandState* states, Config config);

    /**
     * @brief CUDA kernel for the crossover operation.
     * @param mating_pool The pool of selected parents for crossover.
     * @param new_pop The new population generated through crossover.
     * @param states The device array to store the random states.
     * @param config The configuration struct containing parameters for the crossover operation.
     */
    __global__ void crossover_kernel(const double* mating_pool, double* new_pop, curandState* states, Config config);

    /**
     * @brief CUDA kernel for the mutation operation.
     * @param pop The population of candidate solutions to be mutated.
     * @param states The device array to store the random states.
     * @param config The configuration struct containing parameters for the mutation operation.
     */
    __global__ void mutation_kernel(double* pop, curandState* states, Config config);

    /**
     * @brief CUDA kernel for the cataclysm operation.
     * @param pop The population of candidate solutions to be subjected to cataclysm.
     * @param states The device array to store the random states.
     * @param config The configuration struct containing parameters for the cataclysm operation.
     */
    __global__ void cataclysm_kernel(double* pop, curandState* states, Config config);

    /**
     * @brief CUDA kernel for relaxing the positions of atoms.
     * @param pop The population of candidate solutions.
     * @param config The configuration struct containing parameters for relaxation.
     * This function performs a simple relaxation step by using Gradient Descent to move
     * the atoms in the direction of the negative gradient of the energy. This can help to
     * improve the fitness of the solutions by finding local minima in the energy landscape
     * before the next generation of the genetic algorithm is created.
     */
    __global__ void relaxation_kernel(double* pop, Config config, int start_idx);

    /**
     * @brief CUDA kernel for the elitism operation.
     * @param pop The current population of candidate solutions.
     * @param new_pop The new population to which the elite individuals will be copied.
     * @param fitness The array of fitness values for the current population.
     * @param config The configuration struct containing parameters for elitism.
     */
    __global__ void elitism_kernel(const double* pop, double* new_pop, const double* fitness, Config config);

    /**
     * @brief CUDA kernel for population statistics.
     * @param pop The current population of candidate solutions.
     * @param fitness The array of fitness values for the current population.
     * @param config The configuration struct containing parameters for statistics calculation.
     * @param stats The struct to store the computed statistics.
     */
     __global__ void statistics_kernel(const double* pop, const double* fitness, Config config, double* stats);

} // namespace kernels

} // namespace cuga
