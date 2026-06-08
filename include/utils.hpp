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

struct Config {
    // problem definition
    int    n_atoms        = 13;
    int    dimension      = 39;   // 3 * n_atoms

    // parallel sa: one block drives one independent walker
    int    n_walkers      = 1024; // number of parallel SA trajectories
    int    iterations     = 1000; // basin-hopping steps per walker

    int    seed           = 42;

    // simulated annealing schedule
    double T_init         = 1.0;   // initial acceptance temperature
    double cooling_rate   = 0.999; // geometric cooling
    double step_size      = 0.5;   // perturbation displacement magnitude

    // initialization
    double init_radius     = 10.0; // derived from n_atoms
    int   logging_interval = 100;
    void parse(std::string config_file);
    void parse_cmd(int argc, char* argv[]);
    void print() const;
};

void log_header(const Config& config);
void log_stats(const Config& config, double* stats, int step, double temperature);

} // namespace cusa