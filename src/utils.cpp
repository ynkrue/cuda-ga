/**
 * @file utils.cpp
 * @brief Implementation of utility functions for the CUDA simulated annealing optimizer.
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

namespace cusa {

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
        if (key == "n_atoms") {
            n_atoms = std::stoi(val);
        } else if (key == "n_walkers") {
            n_walkers = std::stoi(val);
        } else if (key == "iterations") {
            iterations = std::stoi(val);
        } else if (key == "T_init") {
            T_init = std::stod(val);
        } else if (key == "cooling_rate") {
            cooling_rate = std::stod(val);
        } else if (key == "step_size") {
            step_size = std::stod(val);
        } else if (key == "seed") {
            seed = std::stoi(val);
        } else {
            std::cerr << "Warning: Unrecognized config key '" << key << "' in file '" << config_file << "'" << std::endl;
        }
    }

    logging_interval = std::max(1, iterations / 10);
    dimension = 3 * n_atoms;
    init_radius = 0.7 * std::cbrt((double)n_atoms);

    file.close();
}

void Config::parse_cmd(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            std::cerr << "Error: Either use a config file or command line arguments, not both!" << std::endl;
            exit(1);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [--config <file>] [--n_atoms <int>] [--n_walkers <int>] [--iterations <int>] [--T_init <float>] [--cooling_rate <float>] [--step_size <float>] [--seed <int>]" << std::endl;
            exit(0);
        } else if (arg == "--n_atoms" && i + 1 < argc) {
            n_atoms = std::stoi(argv[++i]);
        } else if (arg == "--n_walkers" && i + 1 < argc) {
            n_walkers = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--T_init" && i + 1 < argc) {
            T_init = std::stod(argv[++i]);
        } else if (arg == "--cooling_rate" && i + 1 < argc) {
            cooling_rate = std::stod(argv[++i]);
        } else if (arg == "--step_size" && i + 1 < argc) {
            step_size = std::stod(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
        } else {
            std::cerr << "Warning: Unrecognized command line argument '" << arg << "'" << std::endl;
            exit(1);
        }
    }

    logging_interval = std::max(1, iterations / 10);
    dimension = 3 * n_atoms;
    init_radius = 0.7 * std::cbrt((double)n_atoms);
}

void Config::print() const {
    std::cout << std::string(80, '=') << std::endl << std::endl;
    std::cout << "      Welcome to the CUDA Simulated Annealing Optimizer!" << std::endl << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  N Atoms          :: " << n_atoms << std::endl;
    std::cout << "  Dimension        :: " << dimension << std::endl;
    std::cout << "  Walkers          :: " << n_walkers << std::endl;
    std::cout << "  Iterations       :: " << iterations << std::endl;

    std::cout << "  Seed             :: " << seed << std::endl;

    std::cout << "  Initial Temp     :: " << T_init << std::endl;
    std::cout << "  Cooling Rate     :: " << cooling_rate << std::endl;
    std::cout << "  Step Size        :: " << step_size << std::endl;
    std::cout << "  Init Radius      :: " << init_radius << std::endl;

    std::cout << "  Logging Interval :: " << logging_interval << " iterations" << std::endl;

    std::cout << std::string(80, '=') << std::endl << std::endl;
}

void log_header(const Config& config) {
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::left << std::setw(17) << "Step" << "  |"
              << std::right << std::setw(14) << "Best" << "  |"
              << std::setw(14) << "Worst" << "  |"
              << std::setw(14) << "Avg" << "  |"
              << std::setw(14) << "Temp" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
}

void log_stats(const Config& config, double* stats, int step, double temperature) {
    auto format_stat = [](double value) {
        std::ostringstream oss;
        oss << std::setw(14) << std::right << std::fixed << std::setprecision(6) << value;
        return oss.str();
    };

    if (step % config.logging_interval == 0) {
        std::cout << "\r[" << std::setw(10) << std::right << step << "/" << config.iterations << "]"
                    << "  |" << format_stat(stats[0]) << "  |" << format_stat(stats[1])
                    << "  |" << format_stat(stats[2]) << "  |" << format_stat(temperature)
                    << std::endl << std::flush;
    } else {
        std::cout << "\r[" << std::setw(10) << std::right << step << "/" << config.iterations << "]" << std::flush;
    }
}

} // namespace cusa
