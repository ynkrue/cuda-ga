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

enum class Selection { Tournament, Roulette, Truncation };

struct Config {
    Mode   mode           = Mode::Rosenbrock;
    int    population     = 4096;
    int    n_atoms        = 13;
    int    dimension = 2;
    int    generations    = 500;
    int    seed           = 42;

    Selection selection   = Selection::Tournament;
    int    parents        = 4096;
    int    tournament_k   = 2;

    double crossover_rate  = 0.4;
    double mutation_rate  = 0.01;

    int elite_size        = 0;

    double init_low       = -2.0;
    double init_high      =  2.0;

    void parse(std::string config_file);
    void print() const;
};

} // namespace cuga