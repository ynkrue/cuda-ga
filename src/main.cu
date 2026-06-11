/**
 * @file main.cu
 * @brief Main entry point for the CUDA simulated annealing / basin hopping optimizer.
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

using namespace cusa;

int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "usage: sa --config <config_file> | sa --[options] \n";
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

    // CUDA launch parameters.
    // Cooperative kernels (energy, relaxation): one block per walker, ~one thread
    // per atom, with reduction scratch in dynamic shared memory.
    int coopBlocks  = config.n_walkers;
    int coopThreads = coop_threads(config.n_atoms);
    size_t energy_shmem = (size_t)(config.dimension + coopThreads) * sizeof(double);
    size_t relax_shmem  = (size_t)(3 * config.dimension + 4 * coopThreads) * sizeof(double);
    // Flat kernels (init, perturb, accept): one thread per walker.
    int flatThreads = WALKER_THREADS;
    int flatBlocks  = (config.n_walkers + flatThreads - 1) / flatThreads;
    int statsThreads = 1024;


    /// ========== Initialization ============================================================ ///
    std::cout << "Initializing walkers..." << std::endl;
    const auto init_start = std::chrono::steady_clock::now();

    // Per-walker configurations are stored contiguously: [walker * dimension + coord].
    double* h_best = new double[(size_t)config.n_walkers * config.dimension];
    double* h_best_energy = new double[config.n_walkers];
    double* h_stats = new double[3]; // best, worst, average

    double *d_current, *d_trial, *d_best;
    double *d_current_energy, *d_trial_energy, *d_best_energy, *d_stats;
    curandState* d_states;
    size_t coords_bytes = (size_t)config.n_walkers * config.dimension * sizeof(double);
    cudaMalloc(&d_current, coords_bytes);
    cudaMalloc(&d_trial,   coords_bytes);
    cudaMalloc(&d_best,    coords_bytes);
    cudaMalloc(&d_current_energy, config.n_walkers * sizeof(double));
    cudaMalloc(&d_trial_energy,   config.n_walkers * sizeof(double));
    cudaMalloc(&d_best_energy,    config.n_walkers * sizeof(double));
    cudaMalloc(&d_stats, 3 * sizeof(double));
    cudaMalloc(&d_states, config.n_walkers * sizeof(curandState));

    // Initialize the starting configurations.
    kernels::init_walkers<<<flatBlocks, flatThreads>>>(d_current, d_states, config);
    kernels::relaxation_kernel<<<coopBlocks, coopThreads, relax_shmem>>>(d_current, config);
    kernels::energy_kernel<<<coopBlocks, coopThreads, energy_shmem>>>(d_current, d_current_energy, config);

    // The starting configuration is the best so far
    cudaMemcpy(d_best, d_current, coords_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_best_energy, d_current_energy, config.n_walkers * sizeof(double), cudaMemcpyDeviceToDevice);

    kernels::statistics_kernel<<<1, statsThreads>>>(d_best_energy, config, d_stats);
    cudaMemcpy(h_stats, d_stats, 3 * sizeof(double), cudaMemcpyDeviceToHost);

    const auto init_end = std::chrono::steady_clock::now();
    const auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start).count();
    std::cout << "Initialized " << config.n_walkers << " walkers of " << config.n_atoms << " atoms." << std::endl;
    std::cout << "Initialization finished in " << init_ms << " ms" << std::endl;

    /// ========== Optimization loop ========================================================= ///
    std::cout << "Starting optimization..." << std::endl;
    log_header(config);
    log_stats(config, h_stats, 0, config.T_init);

    double temperature = config.T_init;
    const auto optimization_start = std::chrono::steady_clock::now();
    for (int step = 1; step < config.iterations + 1; ++step) {

        // Propose, locally minimize, and evaluate a trial configuration.
        kernels::perturb_kernel<<<flatBlocks, flatThreads>>>(d_current, d_trial, d_states, config);
        kernels::relaxation_kernel<<<coopBlocks, coopThreads, relax_shmem>>>(d_trial, config);
        kernels::energy_kernel<<<coopBlocks, coopThreads, energy_shmem>>>(d_trial, d_trial_energy, config);

        // Metropolis acceptance + best tracking.
        kernels::accept_kernel<<<flatBlocks, flatThreads>>>(
            d_current, d_trial, d_current_energy, d_trial_energy,
            d_best, d_best_energy, d_states, temperature, config);

        // Geometric cooling schedule.
        temperature *= config.cooling_rate;

        // Status update
        if (step % config.logging_interval == 0) {
            kernels::statistics_kernel<<<1, statsThreads>>>(d_best_energy, config, d_stats);
            cudaMemcpy(h_stats, d_stats, 3 * sizeof(double), cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();
        log_stats(config, h_stats, step, temperature);
    }

    cudaDeviceSynchronize();
    const auto optimization_end = std::chrono::steady_clock::now();
    const auto optimization_ms = std::chrono::duration_cast<std::chrono::milliseconds>(optimization_end - optimization_start).count();
    std::cout << "Optimization loop finished in " << optimization_ms << " ms" << std::endl;


    /// ========== Final results ============================================================ ///
    // Find the globally best walker on the host.
    cudaMemcpy(h_best_energy, d_best_energy, config.n_walkers * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best, d_best, coords_bytes, cudaMemcpyDeviceToHost);

    int best_walker = 0;
    for (int w = 1; w < config.n_walkers; ++w) {
        if (h_best_energy[w] < h_best_energy[best_walker]) {
            best_walker = w;
        }
    }
    double best_energy = h_best_energy[best_walker];
    const double* best_coords = h_best + (size_t)best_walker * config.dimension;

    std::ofstream result_file("output/output.csv");
    result_file << "N Atoms" << "," << "Best Energy" << "," << "Solution" << std::endl;
    result_file << config.n_atoms << "," << std::setprecision(10) << best_energy << ",";
    for (int j = 0; j < config.dimension; ++j) {
        result_file << std::setprecision(5) << best_coords[j];
        if (j < config.dimension - 1) {
            result_file << ",";
        }
    }
    result_file << std::endl;
    std::cout << "Best energy: " << std::setprecision(10) << best_energy << std::endl;


    /// ========== Cleanup ==================================================================== ///
    delete[] h_best;
    delete[] h_best_energy;
    delete[] h_stats;
    cudaFree(d_current);
    cudaFree(d_trial);
    cudaFree(d_best);
    cudaFree(d_current_energy);
    cudaFree(d_trial_energy);
    cudaFree(d_best_energy);
    cudaFree(d_states);
    cudaFree(d_stats);

    return 0;
}
