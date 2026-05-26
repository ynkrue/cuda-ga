/**
 * @file fitness.cuh
 * @brief Definitions related to fitness functions for genetic algorithm optimization.
 *
 * @author Yannik Rüfenacht
 */

#pragma once

#include <cuda_runtime.h>

namespace cuga {

/**
 * @brief Computes the Rosenbrock function value for a given point.
 * @param pop The population array.
 * @param idx The index of the individual in the population.
 * @param pop_size The population size.
 * @param dim The dimension of each individual.
 * @return The function value at the given point.
 */
__device__ __inline__ double rosenbrock(const double* pop, int idx, int pop_size, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim - 1; ++i) {
        double x_i = pop[i * pop_size + idx];
        double x_ip1 = pop[(i + 1) * pop_size + idx];
        sum += 100.0 * (x_ip1 - x_i * x_i) * (x_ip1 - x_i * x_i) + (1.0 - x_i) * (1.0 - x_i);
    }
    return sum;
}

/**
 * @brief Computes the Lennard-Jones function value for a given point.
 * @param pop The population array.
 * @param idx The index of the individual in the population.
 * @param pop_size The population size.
 * @param dim The dimension of each individual.
 * @return The function value at the given point.
 */
__device__ __inline__ double lennard_jones(const double* pop, int idx, int pop_size, int dim) {
    const double R2_FLOOR = 0.5;     // singularity guard, in sigma^2 units
    double energy = 0.0;
    int n_atoms = dim / 3;

    // loop over atoms
    for (int a = 0; a < n_atoms; ++a) {
        double xa = pop[(3*a + 0) * pop_size + idx];
        double ya = pop[(3*a + 1) * pop_size + idx];
        double za = pop[(3*a + 2) * pop_size + idx];

        // loop over other atoms
        for (int b = a + 1; b < n_atoms; ++b) {
            double dx = xa - pop[(3*b + 0) * pop_size + idx];
            double dy = ya - pop[(3*b + 1) * pop_size + idx];
            double dz = za - pop[(3*b + 2) * pop_size + idx];

            double r2 = dx*dx + dy*dy + dz*dz;
            r2 = fmax(r2, R2_FLOOR);

            double r2_inv = 1.0 / r2;
            double r6_inv = r2_inv * r2_inv * r2_inv;
            double r12_inv = r6_inv * r6_inv;

            energy += 4.0 * (r12_inv - r6_inv);
        }
    }

    return energy;
}

} // namespace cuga