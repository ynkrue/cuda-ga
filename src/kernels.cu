/**
 * @file kernels.cu
 * @brief Implementation of CUDA kernels for genetic algorithm optimization.
 *
 * @author Yannik Rüfenacht
 */

#include "kernels.cuh"
#include "fitness.cuh"

namespace cuga::kernels {

__global__ void init_population(double* pop, curandState* states, const Config config) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= config.population) return;
    // Initialize the random state for each thread
    curandState state;
    curand_init(config.seed, idx, 0, &state);
    states[idx] = state;
    // Initialize the population with random values
    for (int j = 0; j < config.dimension; ++j) {
        pop[j * config.population + idx] = config.init_low + (config.init_high - config.init_low) * curand_uniform_double(&state);
    }
}

__global__ void fitness_kernel(const double* pop, double* fitness, Config config) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= config.population) return;
    if (config.mode == Mode::Rosenbrock) {
        fitness[idx] = rosenbrock(pop, idx, config.population, config.dimension);
    } else {
        // Implement other fitness functions as needed
        fitness[idx] = 0.0; // Placeholder
    }
}

__global__ void selection_kernel(const double* pop, double* mating_pool, const double* fitness, curandState* states, Config config) {
    /// Selection
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= config.parents) return;

    // Tournament selection
    if (config.selection == Selection::Tournament) {
        // loop over k competitors and select the best one
        int best_idx = curand(&states[idx]) % config.population;
        double best_fit = fitness[best_idx];
        for (int i = 1; i < config.tournament_k; ++i) {
            int competitor_idx = curand(&states[idx]) % config.population;
            if (fitness[competitor_idx] < best_fit) {
                best_idx = competitor_idx;
                best_fit = fitness[competitor_idx];
            }
        }
        // copy the selected parent to new population
        for (int j = 0; j < config.dimension; ++j) {
            mating_pool[j * config.parents + idx] = pop[j * config.population + best_idx];
        }

    // Other selection
    } else {
        for (int j = 0; j < config.dimension; ++j) {
            mating_pool[j * config.parents + idx] = pop[j * config.population + idx];
        }
    }
}

__global__ void crossover_kernel(const double* mating_pool, double* new_pop, curandState* states, Config config) {
    /// Crossover
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // each thread produces two children
    int child_a_idx = 2 * idx;
    int child_b_idx = 2 * idx + 1;
    if (child_b_idx >= config.population) return;

    // select two parents randomly
    int parent_a_idx = curand(&states[idx]) % config.parents;
    int parent_b_idx = curand(&states[idx]) % config.parents;

    // loop over dimensions and perform blend crossover
    for (int j = 0; j < config.dimension; ++j) {
        double gene_a = mating_pool[j * config.parents + parent_a_idx];
        double gene_b = mating_pool[j * config.parents + parent_b_idx];
        
        double alpha = curand_uniform_double(&states[idx]);
        new_pop[j * config.population + child_a_idx] = alpha * gene_a + (1.0 - alpha) * gene_b;
        new_pop[j * config.population + child_b_idx] = alpha * gene_b + (1.0 - alpha) * gene_a;
    }
}

__global__ void mutation_kernel(double* pop, curandState* states, Config config) {
    /// Mutation
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= config.population) return;

    for (int j = 0; j < config.dimension; ++j) {
        if (curand_uniform_double(&states[idx]) < config.mutation_rate) {
            // add small random value to gene
            double mutation = (curand_uniform_double(&states[idx]) - 0.5) * 0.2; // mutation in range [-0.1, 0.1]
            pop[j * config.population + idx] += mutation;
        }
    }
}

void elitism_kernel(const double* pop, double* new_pop, const double* fitness, Config config) {
    (void)pop; // Unused parameter
    (void)new_pop; // Unused parameter
    (void)fitness; // Unused parameter
    (void)config; // Unused parameter
}

} // namespace kernels
