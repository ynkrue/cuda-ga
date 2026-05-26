/**
 * @file geometry.cuh
 * @brief Definitions related to geometric calculations for genetic algorithm optimization.
 *
 * @author Yannik Rüfenacht
 */

#pragma once

#include "utils.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace cuga {

/**
 * @brief Performs blending crossover (BLX-alpha) between two parents to produce two children.
 * @param mating_pool The array containing the selected parents.
 * @param new_pop The array to store the new population after crossover.
 * @param idx The index of the thread.
 * @param states The array of random states for each thread.
 * @param config The configuration parameters for the genetic algorithm.
 */
__device__ __inline__ void blending(const double* mating_pool, double* new_pop, int idx, curandState* states, Config config) {
    // each thread produces two children
    int child_a_idx = 2 * idx;
    int child_b_idx = 2 * idx + 1;
    if (child_b_idx >= config.population) return;

    // select two parents randomly
    int parent_a_idx = curand(&states[idx]) % config.parents;
    int parent_b_idx = curand(&states[idx]) % config.parents;

    bool do_crossover = curand_uniform_double(&states[idx]) < config.crossover_rate;

    // loop over dimensions and perform blx alpha crossover
    for (int j = 0; j < config.dimension; ++j) {
        double gene_a = mating_pool[j * config.parents + parent_a_idx];
        double gene_b = mating_pool[j * config.parents + parent_b_idx];
        if (do_crossover) {
            double low = fmin(gene_a, gene_b) - config.crossover_alpha * fabs(gene_a - gene_b);
            double high = fmax(gene_a, gene_b) + config.crossover_alpha * fabs(gene_a - gene_b);
            new_pop[j * config.population + child_a_idx] = low + (high - low) * curand_uniform_double(&states[idx]);
            new_pop[j * config.population + child_b_idx] = low + (high - low) * curand_uniform_double(&states[idx]);
        } else {
            // no crossover, just copy parents
            new_pop[j * config.population + child_a_idx] = gene_a;
            new_pop[j * config.population + child_b_idx] = gene_b;
        }
    }
}

/**
 * @brief Computes the center of mass for a given set of atoms.
 * @param population The array containing the population.
 * @param idx The index of the individual in the population.
 * @param pop_size The total size of the population.
 * @param n_atmos The number of atoms.
 * @param com_x Pointer to store the x-coordinate of the center of mass.
 * @param com_y Pointer to store the y-coordinate of the center of mass.
 * @param com_z Pointer to store the z-coordinate of the center of mass.
 */
__device__ __inline__ void center_of_mass(const double* population, int idx, int pop_size, int n_atmos, double* com_x, double* com_y, double* com_z) {
    *com_x = 0.0;
    *com_y = 0.0;
    *com_z = 0.0;

    for (int i = 0; i < n_atmos; ++i) {
        *com_x += population[(3 * i + 0) * pop_size + idx];
        *com_y += population[(3 * i + 1) * pop_size + idx];
        *com_z += population[(3 * i + 2) * pop_size + idx];
    }

    *com_x /= n_atmos;
    *com_y /= n_atmos;
    *com_z /= n_atmos;
}

/**
 * @brief Ranks atoms based on their z-coordinate for cut-and-splice crossover.
 * @param pop The array containing the population.
 * @param idx The index of the individual in the population.
 * @param pop_size The total size of the population.
 * @param n_atoms The number of atoms.
 * @param a The index of the atom to rank.
 * @return The rank of the atom based on its z-coordinate.
 *         Atoms with higher z-coordinate have higher rank. In case of ties, the atom with the smaller index has higher rank.
 *         The rank is in the range [0, n_atoms-1], where 0 is the highest rank (highest z) and n_atoms-1 is the lowest rank (lowest z).
 */
__device__ __inline__ int rank_by_z(const double* pop, int idx, int pop_size, int n_atoms, int a) {
    double az = pop[(3*a + 2) * pop_size + idx];
    int rank = 0;

    for (int b = 0; b < n_atoms; ++b) {
        if (b == a) continue;
        double bz = pop[(3*b + 2) * pop_size + idx];
        if (bz > az || (bz == az && b < a)) {
            rank += 1;
        }
    }
    return rank;
}

/**
 * @brief Recenters a child solution on its own center of mass after cut-and-splice crossover.
 * @param new_pop The array containing the new population after crossover.
 * @param child_idx The index of the child in the new population.
 * @param pop_size The total size of the population.
 * @param n_atoms The number of atoms.
 * This function computes the center of mass of the child solution and shifts all atoms so that
 * the center of mass is at the origin. This helps to fix the "seam" that can occur in cut-and-splice
 * crossover when combining two parents with different centers of mass.
 */
__device__ __inline__ void recenter_child(double* new_pop, int child_idx, int pop_size, int n_atoms) {
    double cx, cy, cz;
    center_of_mass(new_pop, child_idx, pop_size, n_atoms, &cx, &cy, &cz);
    for (int a = 0; a < n_atoms; ++a) {
        new_pop[(3*a + 0) * pop_size + child_idx] -= cx;
        new_pop[(3*a + 1) * pop_size + child_idx] -= cy;
        new_pop[(3*a + 2) * pop_size + child_idx] -= cz;
    }
}

/**
 * @brief Performs cut and splice crossover between two parents to produce two children.
 * @param mating_pool The array containing the selected parents.
 * @param new_pop The array to store the new population after crossover.
 * @param idx The index of the thread.
 * @param states The array of random states for each thread.
 * @param config The configuration parameters for the genetic algorithm.
 */
__device__ __inline__ void cut_and_splice(const double* mating_pool, double* new_pop, int idx, curandState* states, Config config) {

    
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
        // compute center of mass for both parents
        double comA[3], comB[3];
        center_of_mass(mating_pool, parent_a_idx, config.parents, N, &comA[0], &comA[1], &comA[2]);
        center_of_mass(mating_pool, parent_b_idx, config.parents, N, &comB[0], &comB[1], &comB[2]);

        // child A: select k points of parent A and N-k points of parent B
        int wa = 0;
        for (int a = 0; a < N; ++a) {
            if (rank_by_z(mating_pool, parent_a_idx, config.parents, N, a) < k) {
                for (int c = 0; c < 3; ++c)
                    new_pop[(3*wa + c) * config.population + child_a_idx]
                        = mating_pool[(3*a + c) * config.parents + parent_a_idx] - comA[c];
                ++wa;
            }
        }
        for (int b = 0; b < N; ++b) {
            if (rank_by_z(mating_pool, parent_b_idx, config.parents, N, b) >= k) {
                for (int c = 0; c < 3; ++c)
                    new_pop[(3*wa + c) * config.population + child_a_idx]
                        = mating_pool[(3*b + c) * config.parents + parent_b_idx] - comB[c];
                ++wa;
            }
        }

        // child B: select k points of parent B and N-k points of parent A
        int wb = 0;
        for (int a = 0; a < N; ++a) {
            if (rank_by_z(mating_pool, parent_a_idx, config.parents, N, a) >= k) {
                for (int c = 0; c < 3; ++c)
                    new_pop[(3*wb + c) * config.population + child_b_idx]
                        = mating_pool[(3*a + c) * config.parents + parent_a_idx] - comA[c];
                ++wb;
            }
        }
        for (int b = 0; b < N; ++b) {
            if (rank_by_z(mating_pool, parent_b_idx, config.parents, N, b) < k) {
                for (int c = 0; c < 3; ++c)
                    new_pop[(3*wb + c) * config.population + child_b_idx]
                        = mating_pool[(3*b + c) * config.parents + parent_b_idx] - comB[c];
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

} // namespace cuga