/**
 * @file config.cpp
 * @brief Implementation of the configuration struct parser for the CUDA genetic algorithm.
 *
 * @author Yannik Rüfenacht
 */

#include "config.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cctype>
#include <algorithm>

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

// Helper: convert string to Mode enum
static Mode parse_mode(const std::string& val) {
    std::string v = trim(val);
    if (v == "rb" || v == "Rosenbrock") return Mode::Rosenbrock;
    if (v == "lj" || v == "LennardJones") return Mode::LennardJones;
    return Mode::Rosenbrock; // default
}

void Config::parse(std::string config_file) {
    std::ifstream file(config_file);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open config file '" << config_file << "'" << std::endl;
        return; // return defaults if file cannot be opened
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
        if (key == "mode") {
            mode = parse_mode(val);
        } else if (key == "population") {
            population = std::stoi(val);
        } else if (key == "n_atoms") {
            n_atoms = std::stoi(val);
        } else if (key == "generations") {
            generations = std::stoi(val);
        } else if (key == "parents") {
            parents = std::stoi(val);
        } else if (key == "tournament_k") {
            tournament_k = std::stoi(val);
        } else if (key == "elitism") {
            elitism = (val == "true");
        } else if (key == "crossover_rate") {
            crossover_rate = std::stod(val);
        } else if (key == "crossover_alpha") {
            crossover_alpha = std::stod(val);
        } else if (key == "mutation_rate") {
            mutation_rate = std::stod(val);
        } else if (key == "seed") {
            seed = std::stoi(val);
        } else if (key == "init_low") {
            init_low = std::stod(val);
        } else if (key == "init_high") {
            init_high = std::stod(val);
        } else if (key == "dimension") {
            dimension = std::stoi(val);
        } else if (key == "file_logging") {
            file_logging = val;
        } else {
            std::cerr << "Warning: Unrecognized config key '" << key << "' in file '" << config_file << "'" << std::endl;
        }
    }

    if (parents < 0) {
        parents = population;
    }
    
    file.close();
}

void Config::print() const {
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Mode            :: " << (mode == Mode::Rosenbrock ? "Rosenbrock" : "LennardJones") << std::endl;
    std::cout << "  Population      :: " << population << std::endl;
    if (mode == Mode::LennardJones) {
        std::cout << "  N Atoms     :: " << n_atoms << std::endl;
    } else {
        std::cout << "  Dimension       :: " << dimension << std::endl;
    }
    std::cout << "  Generations     :: " << generations << std::endl;
    std::cout << "  Seed            :: " << seed << std::endl;
    
    std::cout << "  Selection       :: Tournament" << std::endl;
    std::cout << "  Tournament K    :: " << tournament_k << std::endl;
    std::cout << std::endl;
    
    std::cout << "  Parents         :: " << parents << std::endl;
    if (elitism) {
        std::cout << "  Elitism         :: enabled" << std::endl;
    }
    std::cout << "  Crossover Rate  :: " << crossover_rate << std::endl;
    std::cout << "  Crossover Alpha :: " << crossover_alpha << std::endl;
    std::cout << "  Mutation Rate   :: " << mutation_rate << std::endl;
    std::cout << "  Init Low        :: " << init_low << std::endl;
    std::cout << "  Init High       :: " << init_high << std::endl;

    std::cout << "  File Logging    :: " << (file_logging.empty() ? "None" : file_logging) << std::endl;
}

} // namespace cuga