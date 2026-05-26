/**
 * @file kernels.cu
 * @brief Implementation of CUDA kernels for genetic algorithm optimization.
 *
 * @author Yannik Rüfenacht
 */

#include "kernels.cuh"
#include "fitness.cuh"
#include "crossover.cuh"

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
        fitness[idx] = lennard_jones(pop, idx, config.population, config.dimension);
    }
}

__global__ void selection_kernel(const double* pop, double* mating_pool, const double* fitness, curandState* states, Config config) {
    /// Selection
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= config.parents) return;

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
}

__global__ void crossover_kernel(const double* mating_pool, double* new_pop, curandState* states, Config config) {
    /// Crossover
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (config.mode == Mode::Rosenbrock) {
        blending(mating_pool, new_pop, idx, states, config);
    } else {
        cut_and_splice(mating_pool, new_pop, idx, states, config);
    }
}

__global__ void mutation_kernel(double* pop, curandState* states, Config config) {
    // Mutation
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= config.population) return;

    if (config.mode == Mode::Rosenbrock) {
        for (int j = 0; j < config.dimension; ++j) {
            if (curand_uniform_double(&states[idx]) < config.mutation_rate) {
                double m = (curand_uniform_double(&states[idx]) - 0.5) * 0.2;
                pop[j * config.population + idx] += m;
            }
        }
    } else {
        // displace one random atom
        if (curand_uniform_double(&states[idx]) < config.mutation_rate) {
            int a = curand(&states[idx]) % config.n_atoms;
            double span = config.init_high - config.init_low;
            for (int c = 0; c < 3; ++c) {
                pop[(3*a + c) * config.population + idx]
                    = config.init_low + span * curand_uniform_double(&states[idx]);
            }
        }
    }
}

__global__ void elitism_kernel(const double* pop, double* new_pop, const double* fitness, Config config) {
    /// Elitism
    int best_idx = threadIdx.x;
    double best_fit = 1e9;
    
    // thread search best with stride of blockDim.x
    for (int i = threadIdx.x; i < config.population; i += blockDim.x) {
        if (fitness[i] < best_fit) {
            best_fit = fitness[i];
            best_idx = i;
        }
    }

    // Block reduction: reduce from 1024 to 32 threads
    __shared__ double shared_fit[1024];
    __shared__ int shared_idx[1024];
    shared_fit[threadIdx.x] = best_fit;
    shared_idx[threadIdx.x] = best_idx;
    __syncthreads();
    
    // Parallel block reduction step
    for (int s = blockDim.x / 2; s >= 32; s /= 2) {
        if (threadIdx.x < s) {
            if (shared_fit[threadIdx.x + s] < shared_fit[threadIdx.x]) {
                shared_fit[threadIdx.x] = shared_fit[threadIdx.x + s];
                shared_idx[threadIdx.x] = shared_idx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    
    // Warp reduction for the last 32 threads
    if (threadIdx.x < 32) {
        double warp_fit = shared_fit[threadIdx.x];
        int warp_idx = shared_idx[threadIdx.x];
        
        for (int offset = 16; offset > 0; offset /= 2) {
            double other_fit = __shfl_down_sync(0xFFFFFFFF, warp_fit, offset);
            int other_idx = __shfl_down_sync(0xFFFFFFFF, warp_idx, offset);
            if (other_fit < warp_fit) {
                warp_fit = other_fit;
                warp_idx = other_idx;
            }
        }
        
        // Thread 0 of warp stores result back
        if (threadIdx.x == 0) {
            shared_fit[0] = warp_fit;
            shared_idx[0] = warp_idx;
        }
    }
    __syncthreads();
    
    // Thread 0 copies the elite individual to new_pop[0]
    if (threadIdx.x == 0) {
        int best = shared_idx[0];
        for (int j = 0; j < config.dimension; ++j) {
            new_pop[j * config.population + 0] = pop[j * config.population + best];
        }
    }
}

__global__ void statistics_kernel(const double* pop, const double* fitness, Config config, double* stats) {
    /// Statistics
    double sum = 0.0;
    double sum_sq = 0.0;
    double best_fit = 1e9;
    double worst_fit = -1e9;
    
    for (int i = threadIdx.x; i < config.population; i += blockDim.x) {
        sum += fitness[i];
        sum_sq += fitness[i] * fitness[i];
        if (fitness[i] < best_fit) {
            best_fit = fitness[i];
        }
        if (fitness[i] > worst_fit) {
            worst_fit = fitness[i];
        }
    }

    // Block reduction: reduce from 1024 to 32 threads
    __shared__ double shared_sum[1024];
    __shared__ double shared_sum_sq[1024];
    __shared__ double shared_best[1024];
    __shared__ double shared_worst[1024];
    shared_sum[threadIdx.x] = sum;
    shared_sum_sq[threadIdx.x] = sum_sq;
    shared_best[threadIdx.x] = best_fit;
    shared_worst[threadIdx.x] = worst_fit;
    __syncthreads();

    // Parallel block reduction step
    for (int s = blockDim.x / 2; s >= 32; s /= 2) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
            shared_sum_sq[threadIdx.x] += shared_sum_sq[threadIdx.x + s];
            if (shared_best[threadIdx.x + s] < shared_best[threadIdx.x]) {
                shared_best[threadIdx.x] = shared_best[threadIdx.x + s];
            }
            if (shared_worst[threadIdx.x + s] > shared_worst[threadIdx.x]) {
                shared_worst[threadIdx.x] = shared_worst[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    // Warp reduction for the last 32 threads
    if (threadIdx.x < 32) {
        double warp_sum = shared_sum[threadIdx.x];
        double warp_sum_sq = shared_sum_sq[threadIdx.x];
        double warp_best = shared_best[threadIdx.x];
        double warp_worst = shared_worst[threadIdx.x];

        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
            warp_sum_sq += __shfl_down_sync(0xFFFFFFFF, warp_sum_sq, offset);
            double other_best = __shfl_down_sync(0xFFFFFFFF, warp_best, offset);
            double other_worst = __shfl_down_sync(0xFFFFFFFF, warp_worst, offset);
            if (other_best < warp_best) {
                warp_best = other_best;
            }
            if (other_worst > warp_worst) {
                warp_worst = other_worst;
            }
        }

        if (threadIdx.x == 0) {
            double mean = warp_sum / config.population;
            double stddev = sqrt(warp_sum_sq / config.population - mean * mean);
            stats[0] = warp_best; // best
            stats[1] = warp_worst; // worst
            stats[2] = mean; // average
            stats[3] = stddev; // stddev
        }
    }
}

} // namespace kernels
