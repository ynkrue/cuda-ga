/**
 * @file kernels.cu
 * @brief Implementation of CUDA kernels for genetic algorithm optimization.
 *
 * @author Yannik Rüfenacht
 */

#include "utils.cuh"
#include "kernels.cuh"

namespace cuga::kernels {

#define PI 3.14159265358979323846

/// Initialization ================================================================= ///
__global__ void init_population(double* pop, curandState* states, const Config config) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= config.population) return;

    // Initialize the random state for each thread
    curandState state;
    curand_init(config.seed, idx, 0, &state);
    states[idx] = state;

    // Loop over atoms
    for (int a = 0; a < config.n_atoms; ++a) {
        double u_r = curand_uniform_double(&state);
        double r = config.init_radius * cbrt(u_r);

        double z = 2.0 * curand_uniform_double(&state) - 1.0; // z in [-1, 1]
        double theta = 2.0 * PI * curand_uniform_double(&state); // theta in [0, 2pi]

        double r_xy = sqrt(1.0 - z*z);
        pop[(3*a + 0) * config.population + idx] = r * r_xy * cos(theta);;
        pop[(3*a + 1) * config.population + idx] = r * r_xy * sin(theta);;
        pop[(3*a + 2) * config.population + idx] = r * z;
    }
}


/// Fitness evaluation ============================================================= ///
__global__ void fitness_kernel(const double* pop, double* fitness, Config config) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= config.population) return;
    
    constexpr double R2_FLOOR = 0.5;     // singularity guard, in sigma^2 units
    double energy = 0.0;
    int n_atoms = config.dimension / 3;

    // loop over atoms
    for (int a = 0; a < n_atoms; ++a) {
        double xa = pop[(3*a + 0) * config.population + idx];
        double ya = pop[(3*a + 1) * config.population + idx];
        double za = pop[(3*a + 2) * config.population + idx];

        // loop over other atoms
        for (int b = a + 1; b < n_atoms; ++b) {
            double dx = xa - pop[(3*b + 0) * config.population + idx];
            double dy = ya - pop[(3*b + 1) * config.population + idx];
            double dz = za - pop[(3*b + 2) * config.population + idx];

            double r2 = dx*dx + dy*dy + dz*dz;
            r2 = fmax(r2, R2_FLOOR);

            double r2_inv = 1.0 / r2;
            double r6_inv = r2_inv * r2_inv * r2_inv;
            double r12_inv = r6_inv * r6_inv;

            energy += 4.0 * (r12_inv - r6_inv);
        }
    }
    fitness[idx] = energy;
}


/// Selection ========================================================================= ///
__global__ void selection_kernel(const double* pop, double* mating_pool, const double* fitness, curandState* states, Config config) {
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


/// Crossover ========================================================================= ///
__global__ void crossover_kernel(const double* mating_pool, double* new_pop, curandState* states, Config config) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // each thread produces two children
    int child_a_idx = 2 * idx;
    int child_b_idx = 2 * idx + 1;
    if (child_b_idx >= config.population) return;
    
    // select two parents randomly
    int parent_a_idx = curand(&states[idx]) % config.parents;
    int parent_b_idx = curand(&states[idx]) % config.parents;
    
    int N = config.n_atoms;
    int k = curand(&states[idx]) % (N-1) + 1;
    bool do_crossover = curand_uniform_double(&states[idx]) < config.crossover_rate;

    if (do_crossover) {
        
        double r_z = 2.0 * curand_uniform_double(&states[idx]) - 1.0; // z in [-1, 1]
        double r_theta = 2.0 * PI * curand_uniform_double(&states[idx]); // theta in [0, 2pi]
        double r_xy = sqrt(1.0 - r_z*r_z);
        
        double normal[3] = {r_xy * cos(r_theta), r_xy * sin(r_theta), r_z};
        double comA[3], comB[3];
        center_of_mass(mating_pool, parent_a_idx, config.parents, N, &comA[0], &comA[1], &comA[2]);
        center_of_mass(mating_pool, parent_b_idx, config.parents, N, &comB[0], &comB[1], &comB[2]);

        // child A
        int wa = 0;
        for (int a = 0; a < N; ++a) {
            if (rank_by_projection(mating_pool, parent_a_idx, config.parents, N, a, normal) < k) {
                new_pop[(3*wa + 0) * config.population + child_a_idx] = mating_pool[(3*a + 0) * config.parents + parent_a_idx] - comA[0];
                new_pop[(3*wa + 1) * config.population + child_a_idx] = mating_pool[(3*a + 1) * config.parents + parent_a_idx] - comA[1];
                new_pop[(3*wa + 2) * config.population + child_a_idx] = mating_pool[(3*a + 2) * config.parents + parent_a_idx] - comA[2];
                ++wa;
            }
        }
        for (int b = 0; b < N; ++b) {
            if (rank_by_projection(mating_pool, parent_b_idx, config.parents, N, b, normal) >= k) {
                new_pop[(3*wa + 0) * config.population + child_a_idx] = mating_pool[(3*b + 0) * config.parents + parent_b_idx] - comB[0];
                new_pop[(3*wa + 1) * config.population + child_a_idx] = mating_pool[(3*b + 1) * config.parents + parent_b_idx] - comB[1];
                new_pop[(3*wa + 2) * config.population + child_a_idx] = mating_pool[(3*b + 2) * config.parents + parent_b_idx] - comB[2];
                ++wa;
            }
        }

        // child B
        int wb = 0;
        for (int a = 0; a < N; ++a) {
            if (rank_by_projection(mating_pool, parent_a_idx, config.parents, N, a, normal) >= k) {
                new_pop[(3*wb + 0) * config.population + child_b_idx] = mating_pool[(3*a + 0) * config.parents + parent_a_idx] - comA[0];
                new_pop[(3*wb + 1) * config.population + child_b_idx] = mating_pool[(3*a + 1) * config.parents + parent_a_idx] - comA[1];
                new_pop[(3*wb + 2) * config.population + child_b_idx] = mating_pool[(3*a + 2) * config.parents + parent_a_idx] - comA[2];
                ++wb;
            }
        }
        for (int b = 0; b < N; ++b) {
            if (rank_by_projection(mating_pool, parent_b_idx, config.parents, N, b, normal) < k) {
                new_pop[(3*wb + 0) * config.population + child_b_idx] = mating_pool[(3*b + 0) * config.parents + parent_b_idx] - comB[0];
                new_pop[(3*wb + 1) * config.population + child_b_idx] = mating_pool[(3*b + 1) * config.parents + parent_b_idx] - comB[1];
                new_pop[(3*wb + 2) * config.population + child_b_idx] = mating_pool[(3*b + 2) * config.parents + parent_b_idx] - comB[2];
                ++wb;
            }
        }

        // re-center both children on their OWN COM (fixes the seam)
        recenter_child(new_pop, child_a_idx, config.population, N);
        recenter_child(new_pop, child_b_idx, config.population, N);
    } else {
        // no crossover, just copy parents
        for (int j = 0; j < config.dimension; ++j) {
            new_pop[j * config.population + child_a_idx] = mating_pool[j * config.parents + parent_a_idx];
            new_pop[j * config.population + child_b_idx] = mating_pool[j * config.parents + parent_b_idx];
        }
    }
}


/// Mutation ========================================================================== ///
__global__ void mutation_kernel(double* pop, curandState* states, Config config) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= config.population) return;

    if (curand_uniform_double(&states[idx]) < config.mutation_rate) {
        int a = curand(&states[idx]) % config.n_atoms;

        double z = 2.0 * curand_uniform_double(&states[idx]) - 1.0; // z in [-1, 1]
        double theta = 2.0 * PI * curand_uniform_double(&states[idx]); // theta in [0, 2pi]
        double r_xy = sqrt(1.0 - z*z);

        double shove_dist = 1.5;

        double dx = shove_dist * r_xy * cos(theta);
        double dy = shove_dist * r_xy * sin(theta);
        double dz = shove_dist * z;

        pop[(3*a + 0) * config.population + idx] += dx;
        pop[(3*a + 1) * config.population + idx] += dy;
        pop[(3*a + 2) * config.population + idx] += dz;
    }
}

/// Cataclysm / Hypermutation ============================================================== ///
__global__ void cataclysm_kernel(double* pop, curandState* states, Config config) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx == 0 || idx >= config.population) return;

    // copy best solution and shuffle
    if (idx < 10) {
        for (int j = 0; j < config.dimension; ++j) {
            pop[j * config.population + idx] = pop[j * config.population + 0];
        }

    }

    // scramble 30% of the atoms to escape local minima
    int num_to_shove = config.n_atoms / 2; 
    if (num_to_shove < 1) num_to_shove = 1;

    for (int i = 0; i < num_to_shove; ++i) {
        int a = curand(&states[idx]) % config.n_atoms;
        
        // bigger shove distance
        double shove_dist = 4.0; 
        double z = 2.0 * curand_uniform_double(&states[idx]) - 1.0;
        double theta = 2.0 * PI * curand_uniform_double(&states[idx]);
        double r_xy = sqrt(1.0 - z*z);
        
        pop[(3*a + 0) * config.population + idx] += shove_dist * r_xy * cos(theta);
        pop[(3*a + 1) * config.population + idx] += shove_dist * r_xy * sin(theta);
        pop[(3*a + 2) * config.population + idx] += shove_dist * z;
    }
}

/// Local relaxation ================================================================== ///
__global__ void relaxation_kernel(double* pop, Config config, int start_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < start_idx || idx >= config.population) return;

    constexpr double R2_FLOOR = 0.5;
    constexpr int MAX_STEPS = 100;
    constexpr double ALPHA = 0.001;
    constexpr double MAX_MOVE = 0.05;

    for (int step = 0; step < MAX_STEPS; ++step) {

        // loop over every atom
        for (int a = 0; a < config.n_atoms; ++a) {
            double fx = 0.0, fy = 0.0, fz = 0.0;
            
            double xa = pop[(3*a + 0) * config.population + idx];
            double ya = pop[(3*a + 1) * config.population + idx];
            double za = pop[(3*a + 2) * config.population + idx];

            // compute forces from other atoms
            for (int b = 0; b < config.n_atoms; ++b) {
                if (a == b) continue;

                double dx = xa - pop[(3*b + 0) * config.population + idx];
                double dy = ya - pop[(3*b + 1) * config.population + idx];
                double dz = za - pop[(3*b + 2) * config.population + idx];

                double r2 = dx*dx + dy*dy + dz*dz;

                if (r2 < R2_FLOOR) r2 = R2_FLOOR; // singularity guard

                double r2_inv = 1.0 / r2;
                double r6_inv = r2_inv * r2_inv * r2_inv;

                // LJ force derivative
                double fm = 24.0 * r2_inv * r6_inv * (2.0 * r6_inv - 1.0);

                fx += fm * dx;
                fy += fm * dy;
                fz += fm * dz;
            }
            
            // calculate move with simple gradient descent
            double move_x = ALPHA * fx;
            double move_y = ALPHA * fy;
            double move_z = ALPHA * fz;
            
            // gradient clipping
            double move_mag = sqrt(move_x*move_x + move_y*move_y + move_z*move_z);
            if (move_mag > MAX_MOVE) {
                move_x = (move_x / move_mag) * MAX_MOVE;
                move_y = (move_y / move_mag) * MAX_MOVE;
                move_z = (move_z / move_mag) * MAX_MOVE;
            }
            
            // apply move
            pop[(3*a + 0) * config.population + idx] += move_x;
            pop[(3*a + 1) * config.population + idx] += move_y;
            pop[(3*a + 2) * config.population + idx] += move_z;
        }
    }
}


/// Elitism ========================================================================== ///
__global__ void elitism_kernel(const double* pop, double* new_pop, const double* fitness, Config config) {
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


/// Statistics ========================================================================= ///
__global__ void statistics_kernel(const double* pop, const double* fitness, Config config, double* stats) {
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

} // namespace cuga::kernels
