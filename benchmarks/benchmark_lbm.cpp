#include "atmospheric_lbm.hpp"
#include <chrono>
#include <iostream>

int main() {
    AtmosphericLBM lbm(false);
    auto start = std::chrono::high_resolution_clock::now();
    lbm.run();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Benchmark took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms" << std::endl;
    return 0;
}
