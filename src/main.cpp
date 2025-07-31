#include "atmospheric_lbm.hpp"
#include <iostream>

int main() {
    AtmosphericLBM lbm(false);
    lbm.run();
    std::cout << "Atmospheric LBM simulation executed." << std::endl;
    return 0;
}
