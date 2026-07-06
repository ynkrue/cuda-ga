/**
 * @file main.cu
 * @brief Main entry point for the CUDA parallel-tempering basin-hopping optimizer.
 *
 * @author Yannik Rüfenacht
 */

#include "utils.hpp"
#include "kernels.cuh"

#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cuda_runtime.h>
#include <curand_kernel.h>

using namespace cusa;

int main(int argc, char** argv) {
    Config config;
    if (argc > 2 && std::string(argv[1]) == "--config") {
        config.parse(argv[2]);
    } else if (argc > 1) {
        config.parse_cmd(argc, argv);
    }
    config.print();

    /// CUDA launch parameters
    // Cooperative kernels (energy, relaxation)
    int coopBlocks  = config.n_walkers;
    int coopThreads = coop_threads(config.n_atoms);
    size_t energy_shmem = (size_t)(config.dimension + coopThreads) * sizeof(double);
    size_t relax_shmem  = (size_t)(3 * config.dimension + 4 * coopThreads) * sizeof(double);
    // Flat kernels (init, perturb, accept)
    int flatThreads = 256;
    int flatBlocks  = ceil_div(config.n_walkers, flatThreads);
    int swapBlocks  = ceil_div(config.n_ensembles, flatThreads);

    constexpr double BASIN_TOL = 1e-2;


    /// ========== Initialization ============================================================ ///
    std::cout << "Initializing walkers..." << std::endl;
    const auto init_start = std::chrono::steady_clock::now();

    double *d_current, *d_trial, *d_best;
    double *d_current_energy, *d_trial_energy, *d_best_energy, *d_temps;
    unsigned long long* d_swap_accepts;
    curandState* d_states;
    size_t coords_bytes = (size_t)config.n_walkers * config.dimension * sizeof(double);
    cudaMalloc(&d_current, coords_bytes);
    cudaMalloc(&d_trial,   coords_bytes);
    cudaMalloc(&d_best,    coords_bytes);
    cudaMalloc(&d_current_energy, config.n_walkers * sizeof(double));
    cudaMalloc(&d_trial_energy,   config.n_walkers * sizeof(double));
    cudaMalloc(&d_best_energy,    config.n_walkers * sizeof(double));
    cudaMalloc(&d_temps, config.n_walkers * sizeof(double));
    cudaMalloc(&d_swap_accepts, sizeof(unsigned long long));
    cudaMalloc(&d_states, config.n_walkers * sizeof(curandState));
    std::cout << "  Allocated device memory." << std::endl;

    double* h_best_energy = new double[config.n_walkers];

    double* h_temps = new double[config.n_walkers];
    for (int e = 0; e < config.n_ensembles; ++e) {
        for (int r = 0; r < config.n_temps; ++r) {
            double frac = (config.n_temps > 1) ? (double)r / (config.n_temps - 1) : 0.0;
            h_temps[e * config.n_temps + r] = config.T_min * std::pow(config.T_max / config.T_min, frac);
        }
    }
    cudaMemcpy(d_temps, h_temps, config.n_walkers * sizeof(double), cudaMemcpyHostToDevice);

    kernels::init_walkers<<<flatBlocks, flatThreads>>>(d_current, d_states, config);
    kernels::relaxation_kernel<<<coopBlocks, coopThreads, relax_shmem>>>(d_current, config);
    kernels::energy_kernel<<<coopBlocks, coopThreads, energy_shmem>>>(d_current, d_current_energy, config);

    cudaMemcpy(d_best, d_current, coords_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_best_energy, d_current_energy, config.n_walkers * sizeof(double), cudaMemcpyDeviceToDevice);

    const auto init_end = std::chrono::steady_clock::now();
    const auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start).count();
    std::cout << "  " << config.n_walkers << " walkers (" << config.n_atoms << " atoms). Init: " << init_ms << " ms" << std::endl;
    cudaMemset(d_swap_accepts, 0, sizeof(unsigned long long));

    /// ========== Optimization loop ========================================================= ///
    std::cout << "Starting optimization..." << std::endl;
    log_header(config);
    cudaMemcpy(h_best_energy, d_best_energy, config.n_walkers * sizeof(double), cudaMemcpyDeviceToHost);
    log_stats(config, 0, population_stats(h_best_energy, config.n_walkers, BASIN_TOL), 0.0);

    const auto optimization_start = std::chrono::steady_clock::now();
    for (int step = 1; step < config.iterations + 1; ++step) {
        kernels::perturb_kernel<<<flatBlocks, flatThreads>>>(d_current, d_trial, d_states, d_temps, config);
        kernels::relaxation_kernel<<<coopBlocks, coopThreads, relax_shmem>>>(d_trial, config);
        kernels::energy_kernel<<<coopBlocks, coopThreads, energy_shmem>>>(d_trial, d_trial_energy, config);
        kernels::accept_kernel<<<flatBlocks, flatThreads>>>(
            d_current, d_trial, d_current_energy, d_trial_energy,
            d_best, d_best_energy, d_states, d_temps, config);

        kernels::swap_kernel<<<swapBlocks, flatThreads>>>(
            d_current, d_current_energy, d_best, d_best_energy,
            d_states, d_temps, d_swap_accepts, config);

        if (step % config.logging_interval == 0) {
            cudaMemcpy(h_best_energy, d_best_energy, config.n_walkers * sizeof(double), cudaMemcpyDeviceToHost);

            unsigned long long h_swaps = 0;
            cudaMemcpy(&h_swaps, d_swap_accepts, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            double attempts = (double)config.n_ensembles * (config.n_temps - 1) * config.logging_interval;
            double swap_rate = (attempts > 0.0) ? (double)h_swaps / attempts : 0.0;
            cudaMemset(d_swap_accepts, 0, sizeof(unsigned long long));

            log_stats(config, step, population_stats(h_best_energy, config.n_walkers, BASIN_TOL), swap_rate);
        }
    }

    cudaDeviceSynchronize();
    const auto optimization_end = std::chrono::steady_clock::now();
    const auto optimization_ms = std::chrono::duration_cast<std::chrono::milliseconds>(optimization_end - optimization_start).count();
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Optimization finished in " << optimization_ms << " ms" << std::endl << std::endl;

    /// ========== Final results ============================================================ ///
    double* h_best = new double[(size_t)config.n_walkers * config.dimension];
    cudaMemcpy(h_best, d_best, coords_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_energy, d_best_energy, config.n_walkers * sizeof(double), cudaMemcpyDeviceToHost);

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
        result_file << std::setprecision(10) << best_coords[j];
        if (j < config.dimension - 1) {
            result_file << ",";
        }
    }
    result_file << std::endl;
    std::cout << "Best energy: " << std::setprecision(10) << best_energy << std::endl;

    double known = known_lj_minimum(config.n_atoms);
    if (!std::isnan(known)) {
        double gap = best_energy - known;
        std::cout << "Known minimum: " << std::setprecision(10) << known
                  << "  |  gap: " << std::setprecision(10) << gap
                  << (std::abs(gap) < 1e-6 ? "  [HIT]" : "") << std::endl;
    } else {
        std::cout << "No tabulated minimum for N=" << config.n_atoms << std::endl;
    }
    /// ========== Cleanup ==================================================================== ///
    delete[] h_best_energy;
    delete[] h_best;
    delete[] h_temps;
    cudaFree(d_current);
    cudaFree(d_trial);
    cudaFree(d_best);
    cudaFree(d_current_energy);
    cudaFree(d_trial_energy);
    cudaFree(d_best_energy);
    cudaFree(d_states);
    cudaFree(d_temps);
    cudaFree(d_swap_accepts);

    return 0;
}
