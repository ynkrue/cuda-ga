/**
 * @file fitness.cuh
 * @brief Definitions related to fitness functions for genetic algorithm optimization.
 *
 * @author Yannik Rüfenacht
 */

#pragma once

#include <cuda_runtime.h>

/**
 * @brief Computes the Rosenbrock function value for a given point.
 * @param x The input point.
 * @param P population size (number of individuals).
 * @param D dimension of each individual.
 * @return The function value at the given point.
 */
__device__ __inline__ double rosenbrock(const double* pop, int idx, int P, int D) {
    double sum = 0.0;
    for (int i = 0; i < D - 1; ++i) {
        double x_i = pop[i * P + idx];
        double x_ip1 = pop[(i + 1) * P + idx];
        sum += 100.0 * (x_ip1 - x_i * x_i) * (x_ip1 - x_i * x_i) + (1.0 - x_i) * (1.0 - x_i);
    }
    return sum;
}