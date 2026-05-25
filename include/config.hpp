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
    Mode   mode           = Mode::Rosenbrock;
    int    population     = 200;
    int    n_atoms        = 13;
    int    dimension = 2;
    int    generations    = 500;
    int    seed           = 42;

    int    parents        = -1;
    int    tournament_k   = 5;

    double crossover_rate  = 0.8;
    double crossover_alpha = 0.5;

    double mutation_rate  = 0.3;

    bool elitism          = false;

    double init_low       = -2.0;
    double init_high      =  2.0;

    std::string file_logging = "";

    void parse(std::string config_file);
    void print() const;
};

} // namespace cuga