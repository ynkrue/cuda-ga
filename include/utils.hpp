/**
 * @file utils.hpp
 * @brief Utility functions for the CUDA simulated annealing optimizer.
 *
 * @author Yannik Rüfenacht
 */
#pragma once

#include <string>
#include <vector>

namespace cusa {

/**
 * @brief Simulated-annealing / basin-hopping search parameters.
 *
 * n_temps, T_min, T_max, step_size, fire_max_steps default to AUTO and are
 * resolved in finalize(): T_min/T_max/step_size to fixed constants, n_temps
 * from n_walkers, fire_max_steps from n_atoms. Override any of them via
 * config file or CLI.
 */
struct Config {
    static constexpr double AUTO = -1.0;

    int    n_atoms        = 13;
    int    dimension      = 39;      // 3 * n_atoms

    int    n_walkers      = 1024;
    int    iterations     = 250;
    int    n_temps        = (int)AUTO;
    double T_min          = AUTO;
    double T_max          = AUTO;
    double step_size      = AUTO;
    int    fire_max_steps = (int)AUTO;

    int    seed           = 42;

    int    n_ensembles     = 64;     // n_walkers / n_temps
    double init_radius     = 10.0;
    int   logging_interval = 25;

    /// Parse config file; missing keys keep their default.
    void parse(std::string config_file);

    /// Parse CLI args; overrides config file values.
    void parse_cmd(int argc, char* argv[]);

    /// Resolve AUTO fields and derived fields (dimension, n_ensembles, ...).
    void finalize();

    /// Print the configuration.
    void print() const;
};

/// Putative global-minimum LJ energy for n_atoms (Cambridge Cluster Database), or NaN if untabulated.
double known_lj_minimum(int n_atoms);

/// Population statistics at a step, within an energy tolerance.
struct PopStats {
    double best;      ///< lowest energy in the population
    int    basins;    ///< distinct energy levels occupied
    int    best_occ;  ///< walkers in the lowest basin
};

/// Computes PopStats for a population's energies, within tol.
PopStats population_stats(const double* energy, int n, double tol);

/// Prints the log table header.
void log_header(const Config& config);

/// Logs one row: step, best energy, basin count, best-basin occupancy, swap rate.
void log_stats(const Config& config, int step, const PopStats& ps, double swap_rate);

} // namespace cusa