/**
 * @file main.cu
 * @brief Main entry point for the cuda genetic algorithm optimizer.
 *
 * @author Yannik Rüfenacht
 */

#include "utils.hpp"
#include "kernels.cuh"

#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace cuga;

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "usage: ga --config <config_file> | ga --[options] \n";
        return 1;
    }
    
    // Parse configuration
    Config config;
    if (std::string(argv[1]) == "--config") {
        config.parse(argv[2]);
    } else {
        config.parse_cmd(argc, argv);
    }
    config.print();
    
    // cuda launch parameters
    int numThreads = 256;
    int numThreads_reduction = 1024;
    int numBlocks_population = (config.population + numThreads - 1) / numThreads;
    int numBlocks_mating = (config.parents + numThreads - 1) / numThreads;
    int numBlocks_reduction = 1;


    /// ========== Initialization ============================================================ ///
    std::cout << "Initializing population..." << std::endl;
    const auto init_start = std::chrono::steady_clock::now();

    // allocate memory on host and device
    double* h_pop = new double[config.population * config.dimension];
    double *h_fitness = new double[config.population];
    double *h_stats = new double[4]; // best, worst, average, stddev
    double *d_pop, *d_pop_new, *d_mating_pool, *d_fitness, *d_stats;
    curandState* d_states;
    // double prev_best = 1e9;
    // data stored as: [ind0_a0_x, ind1_a0_x, ..., ind0_a0_y, ind1_a0_y, ..., ind0_a0_z, ind1_a0_z, ..., ind0_a1_x, ...]
    cudaMalloc(&d_pop,     config.population * config.dimension * sizeof(double));
    cudaMalloc(&d_pop_new, config.population * config.dimension * sizeof(double));
    cudaMalloc(&d_mating_pool, config.parents * config.dimension * sizeof(double));
    cudaMalloc(&d_fitness, config.population * sizeof(double));
    cudaMalloc(&d_stats, 4 * sizeof(double));
    cudaMalloc(&d_states,  config.population * sizeof(curandState));

    // initialize population
    kernels::init_population<<<numBlocks_population, numThreads>>>(d_pop, d_states, config);
    kernels::fitness_kernel<<<numBlocks_population, numThreads>>>(d_pop, d_fitness, config);
    kernels::statistics_kernel<<<numBlocks_reduction, numThreads_reduction>>>(d_pop, d_fitness, config, d_stats);
    cudaMemcpy(h_stats, d_stats, 4 * sizeof(double), cudaMemcpyDeviceToHost);
    
    const auto init_end = std::chrono::steady_clock::now();
    const auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start).count();
    std::cout << "Population initialized with " << config.population << " individuals and " << config.dimension << " dimensions." << std::endl;
    std::cout << "Initialization finished in " << init_ms << " ms" << std::endl;

    /// ========== Optimization loop ========================================================= ///
    std::cout << "Starting optimization..." << std::endl;
    log_header(config);
    log_stats(config, h_stats, 0);


    // generation loop
    const auto optimization_start = std::chrono::steady_clock::now();
    for (int gen = 1; gen < config.generations+1; ++gen) {
        
        // evaluate fitness
        kernels::fitness_kernel<<<numBlocks_population, numThreads>>>(d_pop, d_fitness, config);
        
        // selection, crossover, mutation, local relaxation and elitism
        kernels::selection_kernel<<<numBlocks_mating, numThreads>>>(d_pop, d_mating_pool, d_fitness, d_states, config);
        kernels::crossover_kernel<<<numBlocks_population, numThreads>>>(d_mating_pool, d_pop_new, d_states, config);        
        kernels::mutation_kernel<<<numBlocks_population, numThreads>>>(d_pop_new, d_states, config);
        kernels::relaxation_kernel<<<numBlocks_population, numThreads>>>(d_pop_new, config, 0);
        kernels::elitism_kernel<<<numBlocks_reduction, numThreads_reduction>>>(d_pop, d_pop_new, d_fitness, config);
        
        // swap populations
        cudaDeviceSynchronize();
        std::swap(d_pop, d_pop_new);
        
        // status update
        if (gen % config.logging_interval == 0) {
            kernels::statistics_kernel<<<numBlocks_reduction, numThreads_reduction>>>(d_pop, d_fitness, config, d_stats);
            cudaMemcpy(h_stats, d_stats, 4 * sizeof(double), cudaMemcpyDeviceToHost);
            // if (std::abs(h_stats[0] - prev_best) < 1e-5) {
            //     kernels::cataclysm_kernel<<<numBlocks_population, numThreads>>>(d_pop, d_states, config);
            //     kernels::relaxation_kernel<<<numBlocks_population, numThreads>>>(d_pop, config, 1);
            //     prev_best = 1e9;
            // } else {
            //     prev_best = h_stats[0];
            // }
        }
        log_stats(config, h_stats, gen);
    }
    
    cudaDeviceSynchronize();
    const auto optimization_end = std::chrono::steady_clock::now();
    const auto optimization_ms = std::chrono::duration_cast<std::chrono::milliseconds>(optimization_end - optimization_start).count();
    std::cout << "Optimization loop finished in " << optimization_ms << " ms" << std::endl;


    /// ========== Final results ============================================================ ///
    // best solution in first position of population due to elitism
    std::ofstream result_file("output/output.csv");
    result_file << "N Atoms" << "," << "Best Fitness" << "," << "Solution" << std::endl;
    result_file << config.n_atoms << "," << std::setprecision(10) << h_stats[0] << ",";
    cudaMemcpy(h_pop, d_pop, config.population * config.dimension * sizeof(double), cudaMemcpyDeviceToHost);
    for (int j = 0; j < config.dimension; ++j) {
        result_file << std::setprecision(5) << h_pop[j * config.population + 0];
        if (j < config.dimension - 1) {
            result_file << ",";
        }
    }
    result_file << std::endl;
    cudaMemcpy(h_fitness, d_fitness, sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "Best fitness: " << std::setprecision(10) << h_fitness[0] << std::endl;


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