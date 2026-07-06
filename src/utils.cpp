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
#include <map>
#include <vector>

namespace cusa {

/// Helper to trim whitespace from both ends
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
        } else if (key == "n_temps") {
            n_temps = std::stoi(val);
        } else if (key == "T_min") {
            T_min = std::stod(val);
        } else if (key == "T_max") {
            T_max = std::stod(val);
        } else if (key == "step_size") {
            step_size = std::stod(val);
        } else if (key == "fire_max_steps") {
            fire_max_steps = std::stoi(val);
        } else if (key == "seed") {
            seed = std::stoi(val);
        } else {
            std::cerr << "Warning: Unrecognized config key '" << key << "' in file '" << config_file << "'" << std::endl;
        }
    }

    file.close();
    finalize();
}

void Config::parse_cmd(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            std::cerr << "Error: Either use a config file or command line arguments, not both!" << std::endl;
            exit(1);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [--config <file>] [--n_atoms <int>] [--n_walkers <int>] [--iterations <int>] [--n_temps <int>] [--T_min <float>] [--T_max <float>] [--step_size <float>] [--fire_max_steps <int>] [--seed <int>]" << std::endl;
            std::cout << "  n_temps, T_min, T_max, step_size and fire_max_steps auto-derive from n_atoms if omitted." << std::endl;
            exit(0);
        } else if (arg == "--n_atoms" && i + 1 < argc) {
            n_atoms = std::stoi(argv[++i]);
        } else if (arg == "--n_walkers" && i + 1 < argc) {
            n_walkers = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--n_temps" && i + 1 < argc) {
            n_temps = std::stoi(argv[++i]);
        } else if (arg == "--T_min" && i + 1 < argc) {
            T_min = std::stod(argv[++i]);
        } else if (arg == "--T_max" && i + 1 < argc) {
            T_max = std::stod(argv[++i]);
        } else if (arg == "--step_size" && i + 1 < argc) {
            step_size = std::stod(argv[++i]);
        } else if (arg == "--fire_max_steps" && i + 1 < argc) {
            fire_max_steps = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
        } else {
            std::cerr << "Warning: Unrecognized command line argument '" << arg << "'" << std::endl;
            exit(1);
        }
    }

    finalize();
}

void Config::finalize() {
    // n_temps: log2(n_walkers), clamped [8,32] -- grows ensembles, not ladder length.
    if (n_temps < 1) {
        int derived = (int)std::lround(std::log2((double)std::max(2, n_walkers)));
        n_temps = std::max(8, std::min(32, derived));
    }
    if (T_min == AUTO)      T_min      = 0.5;
    if (T_max == AUTO)      T_max      = 1.5;
    if (step_size == AUTO)  step_size  = 0.5;

    if (fire_max_steps == (int)AUTO) {
        fire_max_steps = std::max(1000, 20 * n_atoms);
    }

    // Round n_walkers down to a whole number of ensembles.
    n_ensembles = std::max(1, n_walkers / n_temps);
    n_walkers   = n_ensembles * n_temps;

    dimension        = 3 * n_atoms;
    init_radius      = 0.7 * std::cbrt((double)n_atoms);
    logging_interval = std::max(1, iterations / logging_interval);
}

double known_lj_minimum(int n_atoms) {
    // Putative global minima, loaded once from reference/lj_minima.data.
    static const std::map<int, double> table = [] {
        std::map<int, double> t;
        std::ifstream file("reference/lj_minima.data");
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            int n; double e;
            if (ss >> n >> e) t[n] = e;
        }
        return t;
    }();

    auto it = table.find(n_atoms);
    return it != table.end() ? it->second : std::nan("");
}

void Config::print() const {
    std::cout << std::string(80, '=') << std::endl << std::endl;
    std::cout << "      Welcome to the CUDA Simulated Annealing Optimizer!" << std::endl << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  N Atoms          :: " << n_atoms << std::endl;
    std::cout << "  Dimension        :: " << dimension << std::endl;
    std::cout << "  Walkers          :: " << n_walkers
              << " (" << n_ensembles << " ensembles x " << n_temps << " temps)" << std::endl;
    std::cout << "  Iterations       :: " << iterations << std::endl;

    std::cout << "  Seed             :: " << seed << std::endl;

    std::cout << "  Temp Ladder      :: [" << T_min << ", " << T_max << "]" << std::endl;
    std::cout << "  Step Size        :: " << step_size << " (per replica: step_size * T)" << std::endl;
    std::cout << "  FIRE Max Steps   :: " << fire_max_steps << std::endl;
    std::cout << "  Init Radius      :: " << init_radius << std::endl;

    std::cout << "  Logging Interval :: " << logging_interval << " iterations" << std::endl;

    std::cout << std::string(80, '=') << std::endl << std::endl;
}

PopStats population_stats(const double* energy, int n, double tol) {
    std::vector<double> e(energy, energy + n);
    std::sort(e.begin(), e.end());

    PopStats ps{e.front(), 1, 0};
    for (int i = 0; i < n; ++i) {
        if (i > 0 && e[i] - e[i - 1] > tol) ++ps.basins;     // new distinct level
        if (e[i] - ps.best <= tol) ++ps.best_occ;            // still the lowest basin
    }
    return ps;
}

void log_header(const Config& config) {
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::setw(20) << "Step   |"
              << std::setw(22) << "Best"
              << std::setw(12) << "Basins"
              << std::setw(12) << "InBest"
              << std::setw(12) << "Swap%" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
}

void log_stats(const Config& config, int step, const PopStats& ps, double swap_rate) {
    std::string step_str = "[" + std::to_string(step) + "/" + std::to_string(config.iterations) + "]   |";
    std::cout << std::setw(20) << step_str
              << std::setw(22) << std::setprecision(10) << ps.best
              << std::setw(12) << ps.basins
              << std::setw(12) << ps.best_occ
              << std::setw(12) << int(100.0 * swap_rate) << std::endl;
}

} // namespace cusa
