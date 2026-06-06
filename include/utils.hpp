/**
 * @file utils.hpp
 * @brief Utility functions for the CUDA genetic algorithm.
 *
 * @author Yannik Rüfenacht
 */
#pragma once

#include <string>
#include <vector>

namespace cuga {

struct Config {
    // general configuration
    int    n_atoms        = 7;
    int    dimension      = 21;
    int    population     = 300;
    int    generations    = 50;

    int    seed           = 42;

    // selection, crossover, mutation and elitism parameters
    int    parents         = -1;
    int    tournament_k    = 2;
    double crossover_rate  = 0.8;
    double mutation_rate   = 0.3;

    // parameter space parameters
    double init_radius     = 10.0;
    int   logging_interval = 5;
    void parse(std::string config_file);
    void parse_cmd(int argc, char* argv[]);
    void print() const;
};

void log_header(const Config& config);
void log_stats(const Config& config, double* stats, int gen);

} // namespace cuga
