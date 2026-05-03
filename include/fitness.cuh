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
 * @param dim The dimensionality of the input point.
 * @return The function value at the given point.
 */
__device__ __inline__ void rosenbrock(const double* pop, int p, int n) {
    
}