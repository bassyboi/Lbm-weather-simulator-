#ifndef ATMOSPHERIC_LBM_HPP
#define ATMOSPHERIC_LBM_HPP

#ifdef HAVE_OPENCL
#include "opencl_weather.hpp"
#endif
#include <memory>

class AtmosphericLBM {
public:
    explicit AtmosphericLBM(bool use_opencl = false);
    void run();

private:
    bool use_opencl_;
#ifdef HAVE_OPENCL
    std::unique_ptr<OpenCLWeatherSim> cl_sim_;
#endif
};

#endif
