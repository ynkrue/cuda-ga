/**
 * @file utils.cpp
 * @brief Implementation of utility functions for the CUDA genetic algorithm.
 *
 * @author Yannik Rüfenacht
 */

#include "utils.hpp"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cctype>
#include <algorithm>
#include <cmath>

namespace cuga {

// Helper: trim whitespace from both ends
static std::string trim(const std::string& str) {
    auto start = str.begin();
    while (start != str.end() && std::isspace(*start)) {
        ++start;
    }

    auto end = str.end();
    do {
        --end;
    } while (std::distance(start, end) > 0 && std::isspace(*end));

    return std::string(start, end + 1);
}

void Config::parse(std::string config_file) {
    std::ifstream file(config_file);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open config file '" << config_file << "'" << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Find '=' separator
        size_t pos = line.find('=');
        if (pos == std::string::npos) {
            continue; // skip lines without '='
        }

        // Split into key and value
        std::string key = trim(line.substr(0, pos));
        std::string val = trim(line.substr(pos + 1));

        // Parse key-value pairs
        if (key == "population") {
            population = std::stoi(val);
        } else if (key == "n_atoms") {
            n_atoms = std::stoi(val);
        } else if (key == "generations") {
            generations = std::stoi(val);
        } else if (key == "parents") {
            parents = std::stoi(val);
        } else if (key == "tournament_k") {
            tournament_k = std::stoi(val);
        } else if (key == "crossover_rate") {
            crossover_rate = std::stod(val);
        } else if (key == "mutation_rate") {
            mutation_rate = std::stod(val);
        } else if (key == "seed") {
            seed = std::stoi(val);
        } else {
            std::cerr << "Warning: Unrecognized config key '" << key << "' in file '" << config_file << "'" << std::endl;
        }
    }

    if (parents < 0) {
        parents = population;
    }

    logging_interval = std::max(1, generations / 10);
    dimension = 3 * n_atoms;
    double half_width = 0.7 * std::cbrt((double)n_atoms);
    init_low  = -half_width;
    init_high =  half_width;

    file.close();
}

void Config::print() const {
    std::cout << std::string(80, '=') << std::endl << std::endl;
    std::cout << "              Welcome to the CUDA Genetic Algorithm Optimizer!" << std::endl << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  N Atoms          :: " << n_atoms << std::endl;
    std::cout << "  Dimension        :: " << dimension << std::endl;
    std::cout << "  Population       :: " << population << std::endl;
    std::cout << "  Generations      :: " << generations << std::endl;

    std::cout << "  Seed             :: " << seed << std::endl;

    std::cout << "  Parents          :: " << parents << std::endl;
    std::cout << "  Selection        :: Tournament" << std::endl;
    std::cout << "  Tournament K     :: " << tournament_k << std::endl;
    std::cout << "  Crossover Rate   :: " << crossover_rate << std::endl;
    std::cout << "  Crossover Type   :: Cut and Splice" << std::endl;
    std::cout << "  Mutation Rate    :: " << mutation_rate << std::endl;
    std::cout << "  Init Low         :: " << init_low << std::endl;
    std::cout << "  Init High        :: " << init_high << std::endl;

    std::cout << "  Logging Interval :: " << logging_interval << " generations" << std::endl;

    std::cout << std::string(80, '=') << std::endl << std::endl;
}

void log_header(const Config& config) {
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(17) << "Gen" << " | "
              << std::right << std::setw(12) << "Best" << " | "
              << std::setw(12) << "Worst" << " | "
              << std::setw(12) << "Avg" << " | "
              << std::setw(11) << "StdDev" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
}

void log_stats(const Config& config, double* stats, int gen) {
    auto format_stat = [](double value) {
        std::ostringstream oss;
        oss << std::setw(12) << std::right << std::fixed << std::setprecision(6) << value;
        return oss.str();
    };

    if (gen % config.logging_interval == 0) {
        std::cout << "\r[" << std::setw(11) << std::right << gen << "/" << config.generations << "]"
                    << "  |" << format_stat(stats[0]) << "  |" << format_stat(stats[1])
                    << "  |" << format_stat(stats[2]) << "  |" << format_stat(stats[3])
                    << std::endl << std::flush;
    } else {
        std::cout << "\r[" << std::setw(12) << std::right << gen << "/" << config.generations << "]" << std::flush;
    }
}

} // namespace cuga
