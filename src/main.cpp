/**
 * @file main.cpp
 * @brief Main entry point for the cuda genetic algorithm optimizer.
 *
 * @author Yannik Rüfenacht
 */

#include "kernels.cuh"

#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace cuga;

int main(int argc, char** argv) {

    if (argc < 4) {
        std::cerr << "usage: ga <rb|lj> <population> <generations> [seed]\n";
        return 1;
    }

    Config config;

    std::string mode_str = argv[1];
    if      (mode_str == "rb") config.mode = Mode::Rosenbrock;
    // else if (mode_str == "lj") config.mode = Mode::LennardJones;
    else { std::cerr << "unknown mode: " << mode_str << "\n"; return 1; }

    config.population  = std::stoi(argv[2]);
    config.generations = std::stoi(argv[3]);
    if (argc > 4) config.seed = std::stoi(argv[4]);

    // config.dimensionality = (config.mode == Mode::LennardJones) ? 3 * config.n_atoms : config.dimensionality;

    // welcome message
    std::cout << std::string(80, '=') << "\n"
              << "                     Welcome to cuda genetic algorithm!" << std::endl
              << "  cuda-ga  |  mode: " << mode_str
              << "  pop: "  << config.population
              << "  gen: "  << config.generations
              << "  seed: " << config.seed << "\n"
              << std::string(80, '=') << "\n";

    /// ========== Initialization ============================================================ ///
    std::cout << "Initializing population...\n";
    double* d_pop, d_pop_new, d_fitness;
    curandState* d_states;
    cudaMalloc(&d_pop,     config.population * config.dimensionality * sizeof(double));
    cudaMalloc(&d_pop_new, config.population * config.dimensionality * sizeof(double));
    cudaMalloc(&d_fitness, config.population * sizeof(double));
    cudaMalloc(&d_states,  config.population * sizeof(curandState));

    dim3 dimBlock(256);
    dim3 dimGrid((config.population + dimBlock.x - 1) / dimBlock.x);
    kernels::init_population<<<dimGrid, dimBlock>>>(d_pop, d_states, config);
    
    /// ========== Optimization loop ========================================================= ///
    std::cout << "Starting optimization..." << std::endl;
    for (int gen = 0; gen < config.generations; ++gen) {

    }


    return 0;
}