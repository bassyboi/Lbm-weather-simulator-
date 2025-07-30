#ifndef OPENCL_WEATHER_HPP
#define OPENCL_WEATHER_HPP

#include <CL/cl.h>
#include <vector>
#include <string>

struct Grid3D {
    size_t nx, ny, nz;
};

class OpenCLWeatherSim {
public:
    OpenCLWeatherSim(const Grid3D& grid);
    bool initialize();
    void step();
    const std::vector<float>& temperature() const { return h_temperature; }
    const std::vector<float>& pressure() const { return h_pressure; }

private:
    bool init_opencl();
    bool build_kernels(const std::string& source);
    void upload();
    void download();

    Grid3D grid;
    size_t cell_count;

    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel update_kernel = nullptr;

    cl_mem d_temperature = nullptr;
    cl_mem d_pressure = nullptr;

    std::vector<float> h_temperature;
    std::vector<float> h_pressure;
};

#endif // OPENCL_WEATHER_HPP
