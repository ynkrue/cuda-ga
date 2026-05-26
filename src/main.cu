/**
 * @file main.cpp
 * @brief Main entry point for the cuda genetic algorithm optimizer.
 *
 * @author Yannik Rüfenacht
 */

#include "kernels.cuh"
#include "utils.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace cuga;

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "usage: ga <config_file> \n";
        return 1;
    }
    
    // Parse configuration from INI file
    Config config;
    config.parse(argv[1]);
    
    // welcome message
    config.print();

    /// ========== Initialization ============================================================ ///
    std::cout << "Initializing population..." << std::endl;
    const auto init_start = std::chrono::steady_clock::now();

    // allocate memory on host and device
    double *h_fitness = new double[config.population];
    double *h_stats = new double[4]; // best, worst, average, stddev
    double *d_pop, *d_pop_new, *d_mating_pool, *d_fitness, *d_stats;
    curandState* d_states;
    cudaMalloc(&d_pop,     config.population * config.dimension * sizeof(double));
    cudaMalloc(&d_pop_new, config.population * config.dimension * sizeof(double));
    cudaMalloc(&d_mating_pool, config.parents * config.dimension * sizeof(double));
    cudaMalloc(&d_fitness, config.population * sizeof(double));
    cudaMalloc(&d_stats, 4 * sizeof(double));
    cudaMalloc(&d_states,  config.population * sizeof(curandState));

    // initialize population
    int numThreads = 256;
    int numBlocks = (config.population + numThreads - 1) / numThreads;
    kernels::init_population<<<numBlocks, numThreads>>>(d_pop, d_states, config);
    cudaDeviceSynchronize();
    
    const auto init_end = std::chrono::steady_clock::now();
    const auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start).count();
    std::cout << "Population initialized with " << config.population << " individuals and " << config.dimension << " dimensions." << std::endl;
    std::cout << "Initialization finished in " << init_ms << " ms" << std::endl;
    
    /// ========== Optimization loop ========================================================= ///
    std::cout << "Starting optimization..." << std::endl;
    log_header(config);

    // generation loop
    const auto optimization_start = std::chrono::steady_clock::now();
    for (int gen = 1; gen < config.generations+1; ++gen) {
        
        // evaluate fitness
        kernels::fitness_kernel<<<numBlocks, numThreads>>>(d_pop, d_fitness, config);
        
        // selection, crossover, mutation and elitism
        numBlocks = (config.parents + numThreads - 1) / numThreads;
        kernels::selection_kernel<<<numBlocks, numThreads>>>(d_pop, d_mating_pool, d_fitness, d_states, config);
        
        numBlocks = (config.population + numThreads - 1) / numThreads;
        kernels::crossover_kernel<<<numBlocks, numThreads>>>(d_mating_pool, d_pop_new, d_states, config);
        
        kernels::mutation_kernel<<<numBlocks, numThreads>>>(d_pop_new, d_states, config);
        
        kernels::elitism_kernel<<<1, 1024>>>(d_pop, d_pop_new, d_fitness, config);
        
        // swap populations
        std::swap(d_pop, d_pop_new);
        
        // status update
        if (gen % config.logging_interval == 0) {
            kernels::statistics_kernel<<<1, 1024>>>(d_pop, d_fitness, config, d_stats);
            cudaMemcpy(h_stats, d_stats, 4 * sizeof(double), cudaMemcpyDeviceToHost);
        }
        log_stats(config, h_stats, gen);
    }
    std::cout << std::endl;
    
    // timing
    cudaDeviceSynchronize();
    const auto optimization_end = std::chrono::steady_clock::now();
    const auto optimization_ms = std::chrono::duration_cast<std::chrono::milliseconds>(optimization_end - optimization_start).count();
    std::cout << "Optimization loop finished in " << optimization_ms << " ms" << std::endl;

    // retrieve best solution
    cudaMemcpy(h_fitness, d_fitness, config.population * sizeof(double), cudaMemcpyDeviceToHost);
    int best_idx = 0;
    for (int i = 1; i < config.population; ++i) {
        if (h_fitness[i] < h_fitness[best_idx]) {
            best_idx = i;
        }
    }
    double* h_pop = new double[config.population * config.dimension];
    cudaMemcpy(h_pop, d_pop, config.population * config.dimension * sizeof(double), cudaMemcpyDeviceToHost);
    for (int j = 0; j < config.dimension; ++j) {
        std::cout << "  x[" << j << "] = " << h_pop[j * config.population + best_idx] << "\n";
    }
    std::cout << "Best fitness: " << h_fitness[best_idx] << std::endl;

    /// ========== Cleanup ==================================================================== ///
    delete[] h_fitness;
    delete[] h_pop;
    delete[] h_stats;
    cudaFree(d_pop);
    cudaFree(d_pop_new);
    cudaFree(d_mating_pool);
    cudaFree(d_fitness);
    cudaFree(d_states);
    cudaFree(d_stats);

    return 0;
}