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
 * @class Config
 * @brief Configuration parameters for the simulated annealing optimizer.
 * Fields:
 *  - n_atoms: Number of atoms in the cluster.
 *  - dimension: Dimensionality of the problem (3 * n_atoms).
 *  - n_walkers: Total number of walkers (replicas) in the population.
 *  - iterations: Total number of iterations to run the optimization.
 *  - n_temps: Number of temperatures in the replica exchange ladder.
 *  - T_min: Minimum temperature in the ladder.
 *  - T_max: Maximum temperature in the ladder.
 *  - step_size: Base step size for perturbations (scaled by temperature).
 *  - seed: Random seed for reproducibility.
 *  - n_ensembles: Number of ensembles (n_walkers / n_temps).
 *  - init_radius: Initial radius for random walker initialization.
 *  - logging_interval: Number of stats logged (log every iterations / logging_interval).
 * Methods:
 *  - parse(config_file): Parse configuration from a file.
 *  - parse_cmd(argc, argv): Parse command-line arguments.
 *  - finalize(): Finalize the configuration.
 *  - print(): Print the configuration.
 */
struct Config {
    // problem definition
    int    n_atoms        = 13;
    int    dimension      = 39;

    int    n_walkers      = 1024;
    int    iterations     = 250;
    int    n_temps        = 16;
    double T_min          = 0.5;
    double T_max          = 1.5;
    double step_size      = 0.5;

    int    seed           = 42;

    int    n_ensembles     = 64;
    double init_radius     = 10.0;
    int   logging_interval = 25;

    /**
     * @brief Parse configuration from a file.
     * @param config_file Path to the configuration file.
     * @result Populates the Config object with values from
     *   the config file, kepping defaults for missing keys.
     */
    void parse(std::string config_file);

    /**
     * @brief Parse command-line arguments.
     * @param argc Number of command-line arguments.
     * @param argv Array of command-line arguments.
     * @result Populates the Config object with values from
     *  the command line, overriding any config values.
     */
    void parse_cmd(int argc, char* argv[]);

    /// Finalize config by computing derived parameters.
    void finalize();

    /// Print the configuration to the console.
    void print() const;
};

/**
 * @brief Get the known minimum LJ energy for a given number of atoms.
 * @param n_atoms The number of atoms.
 * @result The known minimum LJ energy in reduced units obtained
 *   from the Cambridge Cluster Database, or NaN if not tabulated.
 *   Used for confirmation of results at the end of the search.
 */
double known_lj_minimum(int n_atoms);

/**
 * @class PopStats
 * @brief Statistics about the population of walkers at a given step.
 * Fields:
 *  - best: lowest energy in the population
 *  - basins: number of distinct energy levels occupied (within tol)
 *  - best_occ: number of walkers sitting in the lowest basin (within tol)
 */
struct PopStats {
    double best;
    int    basins;
    int    best_occ;
};

/**
 * @brief Compute population statistics for a given energy array.
 * @param energy Array of energies for the population.
 * @param n Number of walkers.
 * @param tol Energy tolerance for distinguishing basins.
 * @result A PopStats struct containing the best energy, number of basins,
 *   and occupancy of the best basin. Used for logging the search progress.
 */
PopStats population_stats(const double* energy, int n, double tol);

/// Print the header for the log output.
void log_header(const Config& config);

/**
 * @brief Log the statistics for a given step of the simulation.
 * @param config The configuration object.
 * @param step The current iteration step.
 * @param ps The population statistics to log.
 * @param swap_rate The acceptance rate of replica exchanges in this interval.
 */
void log_stats(const Config& config, int step, const PopStats& ps, double swap_rate);

} // namespace cusa