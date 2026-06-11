/**
 * @file main.cu
 * @brief Main entry point for the CUDA simulated annealing
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
    // Parse configuration
    Config config;
    if (argc > 2 && std::string(argv[1]) == "--config") {
        config.parse(argv[2]);
    } else if (argc > 1) {
        config.parse_cmd(argc, argv);
    }
    config.print();

    /// CUDA launch parameters
    // Cooperative kernels (energy, relaxation)
    int coopBlocks  = config.n_walkers;             // one block per walker
    int coopThreads = coop_threads(config.n_atoms); // threads per block scale with the number of atoms
    size_t energy_shmem = (size_t)(config.dimension + coopThreads) * sizeof(double);
    size_t relax_shmem  = (size_t)(3 * config.dimension + 4 * coopThreads) * sizeof(double);
    // Flat kernels (init, perturb, accept)
    int flatThreads = 256;                                          // one thread per walker
    int flatBlocks  = ceil_div(config.n_walkers, flatThreads);      // enough blocks to cover walkers
    int swapBlocks  = ceil_div(config.n_ensembles, flatThreads);    // enough blocks to cover ensembles

    // Energy gap for which two walkers count as the same basin.
    constexpr double BASIN_TOL = 1e-2;


    /// ========== Initialization ============================================================ ///
    std::cout << "Initializing walkers..." << std::endl;
    const auto init_start = std::chrono::steady_clock::now();

    /// Per-walker configurations are stored contiguously [walker * dimension + coord].

    // host arrays
    double* h_best = new double[(size_t)config.n_walkers * config.dimension];
    double* h_best_energy = new double[config.n_walkers];

    // device arrays
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
    std::cout << "  " << "Allocated device memory for optmization." << std::endl;

    // Per-walker temperature ladder: every ensemble holds the same geometric
    // ladder from T_min (rung 0) to T_max (top rung).
    double* h_temps = new double[config.n_walkers];
    for (int e = 0; e < config.n_ensembles; ++e) {
        for (int r = 0; r < config.n_temps; ++r) {
            double frac = (config.n_temps > 1) ? (double)r / (config.n_temps - 1) : 0.0;
            h_temps[e * config.n_temps + r] = config.T_min * std::pow(config.T_max / config.T_min, frac);
        }
    }
    cudaMemcpy(d_temps, h_temps, config.n_walkers * sizeof(double), cudaMemcpyHostToDevice);

    // Initialize the starting configurations.
    kernels::init_walkers<<<flatBlocks, flatThreads>>>(d_current, d_states, config);
    kernels::relaxation_kernel<<<coopBlocks, coopThreads, relax_shmem>>>(d_current, config);
    kernels::energy_kernel<<<coopBlocks, coopThreads, energy_shmem>>>(d_current, d_current_energy, config);

    // The starting configuration is the best so far
    cudaMemcpy(d_best, d_current, coords_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_best_energy, d_current_energy, config.n_walkers * sizeof(double), cudaMemcpyDeviceToDevice);

    const auto init_end = std::chrono::steady_clock::now();
    const auto init_ms = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start).count();
    std::cout << "  " << "Initialized " << config.n_walkers << " walkers of " << config.n_atoms << " atoms." << std::endl;
    std::cout << "Initialization finished in " << init_ms << " ms" << std::endl << std::endl;

    /// ========== Optimization loop ========================================================= ///
    std::cout << "Starting optimization..." << std::endl;
    log_header(config);
    cudaMemcpy(h_best_energy, d_best_energy, config.n_walkers * sizeof(double), cudaMemcpyDeviceToHost);
    log_stats(config, 0, population_stats(h_best_energy, config.n_walkers, BASIN_TOL), 0.0);

    cudaMemset(d_swap_accepts, 0, sizeof(unsigned long long));
    const auto optimization_start = std::chrono::steady_clock::now();
    for (int step = 1; step < config.iterations + 1; ++step) {

        // One basin-hopping move per replica at its own temperature.
        kernels::perturb_kernel<<<flatBlocks, flatThreads>>>(d_current, d_trial, d_states, d_temps, config);
        kernels::relaxation_kernel<<<coopBlocks, coopThreads, relax_shmem>>>(d_trial, config);
        kernels::energy_kernel<<<coopBlocks, coopThreads, energy_shmem>>>(d_trial, d_trial_energy, config);
        kernels::accept_kernel<<<flatBlocks, flatThreads>>>(
            d_current, d_trial, d_current_energy, d_trial_energy,
            d_best, d_best_energy, d_states, d_temps, config);

        // Replica exchange between adjacent rungs.
        kernels::swap_kernel<<<swapBlocks, flatThreads>>>(
            d_current, d_current_energy, d_best, d_best_energy,
            d_states, d_temps, d_swap_accepts, config);

        // Log a line every interval
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
    std::cout << "Optimization loop finished in " << optimization_ms << " ms" << std::endl << std::endl;


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
        result_file << std::setprecision(10) << best_coords[j];
        if (j < config.dimension - 1) {
            result_file << ",";
        }
    }
    result_file << std::endl;
    std::cout << "Best energy: " << std::setprecision(10) << best_energy << std::endl;

    // Gap to the known putative global minimum, when tabulated (reported only here;
    // the search itself never sees it).
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
    delete[] h_best;
    delete[] h_best_energy;
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
