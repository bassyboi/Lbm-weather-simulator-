#include "atmospheric_lbm.hpp"
#include <iostream>

int main() {
    AtmosphericLBM lbm(true);
    lbm.run();
    return 0;
}
