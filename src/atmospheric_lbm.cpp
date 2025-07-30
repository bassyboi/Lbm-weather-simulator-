#include "atmospheric_lbm.hpp"
#include <iostream>

AtmosphericLBM::AtmosphericLBM(bool use_opencl)
    : use_opencl_(use_opencl)
{
#ifdef HAVE_OPENCL
    if (use_opencl_) {
        Grid3D grid{32, 32, 16};
        cl_sim_ = std::make_unique<OpenCLWeatherSim>(grid);
        if (!cl_sim_->initialize()) {
            std::cerr << "Failed to initialize OpenCL weather sim, falling back to CPU" << std::endl;
            cl_sim_.reset();
            use_opencl_ = false;
        }
    }
#endif
}

void AtmosphericLBM::run() {
#ifdef HAVE_OPENCL
    if (use_opencl_ && cl_sim_) {
        for (int i = 0; i < 10; ++i) {
            cl_sim_->step();
        }
        std::cout << "OpenCL temperature[0]: " << cl_sim_->temperature()[0] << std::endl;
        return;
    }
#endif
    std::cout << "Running LBM core..." << std::endl;
}
