/**
 * @file main.cpp
 * @brief Main entry point for the cuda genetic algorithm optimizer.
 *
 * @author Yannik Rüfenacht
 */

#include "kernels.cuh"
#include "config.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace cuga;

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "usage: ga config_file \n";
        return 1;
    }
    
    // Parse configuration from INI file
    Config config;
    config.parse(argv[1]);
    
    // welcome message
    std::cout << std::string(80, '=') << std::endl << std::endl;
    std::cout << "              Welcome to the CUDA Genetic Algorithm Optimizer!" << std::endl << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    config.print();
    std::cout << std::string(80, '=') << std::endl << std::endl;

    /// ========== Initialization ============================================================ ///
    std::cout << "Initializing population..." << std::endl;
    double *h_fitness = new double[config.population];
    double *d_pop, *d_pop_new, *d_mating_pool, *d_fitness;
    curandState* d_states;
    std::ofstream log_file;
    if (config.file_logging != "") {
        log_file.open(config.file_logging);
        if (!log_file.is_open()) {
            std::cerr << "Error: Could not open log file '" << config.file_logging << "'" << std::endl;
            return 1;
        }
        log_file << "generation,individual";
        for (int j = 0; j < config.dimension; ++j) {
            log_file << ",x" << j;
        }
        log_file << "\n";
    }
    cudaMalloc(&d_pop,     config.population * config.dimension * sizeof(double));
    cudaMalloc(&d_pop_new, config.population * config.dimension * sizeof(double));
    cudaMalloc(&d_mating_pool, config.parents * config.dimension * sizeof(double));
    cudaMalloc(&d_fitness, config.population * sizeof(double));
    cudaMalloc(&d_states,  config.population * sizeof(curandState));

    int blockSize = 256;
    int numBlocks = (config.population + blockSize - 1) / blockSize;
    kernels::init_population<<<numBlocks, blockSize>>>(d_pop, d_states, config);
    
    /// ========== Optimization loop ========================================================= ///
    std::cout << "Starting optimization..." << std::endl;
    const auto optimization_start = std::chrono::steady_clock::now();
    for (int gen = 0; gen < config.generations; ++gen) {
        // evaluate fitness
        kernels::fitness_kernel<<<numBlocks, blockSize>>>(d_pop, d_fitness, config);

        // selection, crossover, mutation
        numBlocks = (config.parents + blockSize - 1) / blockSize;
        kernels::selection_kernel<<<numBlocks, blockSize>>>(d_pop, d_mating_pool, d_fitness, d_states, config);

        numBlocks = (config.population + blockSize - 1) / blockSize;
        kernels::crossover_kernel<<<numBlocks, blockSize>>>(d_mating_pool, d_pop_new, d_states, config);

        kernels::mutation_kernel<<<numBlocks, blockSize>>>(d_pop_new, d_states, config);

        // elitism
        kernels::elitism_kernel<<<1, 1024>>>(d_pop, d_pop_new, d_fitness, config);


        if (config.file_logging != "") {
            double* h_pop = new double[config.population * config.dimension];
            cudaMemcpy(h_pop, d_pop, config.population * config.dimension * sizeof(double), cudaMemcpyDeviceToHost);
            for (int i = 0; i < config.population; ++i) {
                log_file << gen << "," << i;
                for (int j = 0; j < config.dimension; ++j) {
                    log_file << "," << h_pop[j * config.population + i];
                }
                log_file << "\n";
            }
            delete[] h_pop;
        }

        // swap populations
        std::swap(d_pop, d_pop_new);
    }
    
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
    std::cout << "Best solution found: \n";
    double* h_pop = new double[config.population * config.dimension];
    cudaMemcpy(h_pop, d_pop, config.population * config.dimension * sizeof(double), cudaMemcpyDeviceToHost);
    for (int j = 0; j < config.dimension; ++j) {
        std::cout << "  x[" << j << "] = " << h_pop[j * config.population + best_idx] << "\n";
    }
    std::cout << "Best fitness: " << h_fitness[best_idx] << std::endl;


    return 0;
}