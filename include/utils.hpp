/**
 * @file config.hpp
 * @brief Configuration struct for the CUDA genetic algorithm.
 *
 * @author Yannik Rüfenacht
 */
#pragma once

#include <string>

namespace cuga {

enum class Mode { Rosenbrock, LennardJones };

struct Config {
    // general configuration
    Mode   mode           = Mode::Rosenbrock;
    int    n_atoms        = 0;
    int    dimension      = 2;
    int    population     = 300;
    int    generations    = 50;

    int    seed           = 42;

    // selection, crossover, mutation and elitism parameters
    int    parents        = -1;
    int    tournament_k   = 5;
    double crossover_rate  = 0.8;
    double crossover_alpha = 0.5;
    double mutation_rate  = 0.3;
    bool elitism          = false;

    // parameter space parameters
    double init_low       = -2.0;
    double init_high      =  2.0;

    int logging_interval  = 100;
    // std::string file_logging = "";

    void parse(std::string config_file);
    void print() const;
};

void log_header(const Config& config);
void log_stats(const Config& config, double* stats, int gen);

} // namespace cuga