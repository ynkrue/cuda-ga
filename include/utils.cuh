/**
 * @file crossover.cuh
 * @brief Crossover operators for the Lennard-Jones GA.
 *
 * @author Yannik Rüfenacht
 */

#pragma once

#include "utils.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace cuga {

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
 * @brief Ranks atoms based on their projection onto a random 3D normal vector.
 */
__device__ __inline__ int rank_by_projection(const double* pop, int idx, int pop_size, int n_atoms, int a, const double* normal) {
    double ax = pop[(3*a + 0) * pop_size + idx];
    double ay = pop[(3*a + 1) * pop_size + idx];
    double az = pop[(3*a + 2) * pop_size + idx];

    // project atom a into random normal vector
    double a_proj = ax * normal[0] + ay * normal[1] + az * normal[2];
    int rank = 0;

    for (int b = 0; b < n_atoms; ++b) {
        if (b == a) continue;
        double bx = pop[(3*b + 0) * pop_size + idx];
        double by = pop[(3*b + 1) * pop_size + idx];
        double bz = pop[(3*b + 2) * pop_size + idx];

        // project atom b into random normal vector
        double b_proj = bx * normal[0] + by * normal[1] + bz * normal[2];

        if (b_proj > a_proj || (b_proj == a_proj && b < a)) {
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

} // namespace cuga