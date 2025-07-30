#include "opencl_weather.hpp"
#include <iostream>
#include <fstream>

OpenCLWeatherSim::OpenCLWeatherSim(const Grid3D& g) : grid(g) {
    cell_count = grid.nx * grid.ny * grid.nz;
    h_temperature.resize(cell_count, 290.0f);
    h_pressure.resize(cell_count, 100000.0f);
}

bool OpenCLWeatherSim::initialize() {
    if (!init_opencl()) return false;

    d_temperature = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(float)*cell_count, nullptr, nullptr);
    d_pressure = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                 sizeof(float)*cell_count, nullptr, nullptr);
    upload();

    std::ifstream file("kernels/weather_kernels.cl");
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open kernel file 'kernels/weather_kernels.cl'" << std::endl;
        return false;
    }
    std::string source((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    if (!build_kernels(source)) return false;
    return true;
}

bool OpenCLWeatherSim::init_opencl() {
    cl_uint num;
    clGetPlatformIDs(1, &platform, &num);
    if (!num) return false;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num);
    if (!num) return false;
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    queue = clCreateCommandQueueWithProperties(context, device, nullptr, nullptr);
    return true;
}

bool OpenCLWeatherSim::build_kernels(const std::string& source) {
    const char* src = source.c_str();
    size_t len = source.size();
    program = clCreateProgramWithSource(context, 1, &src, &len, nullptr);
    if (!program) return false;
    if (clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr) != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "Kernel build error:\n" << log << std::endl;
        return false;
    }
    update_kernel = clCreateKernel(program, "update_fields", nullptr);
    return update_kernel != nullptr;
}

void OpenCLWeatherSim::upload() {
    clEnqueueWriteBuffer(queue, d_temperature, CL_TRUE, 0,
                         sizeof(float)*cell_count, h_temperature.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, d_pressure, CL_TRUE, 0,
                         sizeof(float)*cell_count, h_pressure.data(), 0, nullptr, nullptr);
}

void OpenCLWeatherSim::download() {
    clEnqueueReadBuffer(queue, d_temperature, CL_TRUE, 0,
                        sizeof(float)*cell_count, h_temperature.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, d_pressure, CL_TRUE, 0,
                        sizeof(float)*cell_count, h_pressure.data(), 0, nullptr, nullptr);
}

void OpenCLWeatherSim::step() {
    size_t global[3] = {grid.nx, grid.ny, grid.nz};
    clSetKernelArg(update_kernel, 0, sizeof(cl_mem), &d_temperature);
    clSetKernelArg(update_kernel, 1, sizeof(cl_mem), &d_pressure);
    clSetKernelArg(update_kernel, 2, sizeof(int), &grid.nx);
    clSetKernelArg(update_kernel, 3, sizeof(int), &grid.ny);
    clSetKernelArg(update_kernel, 4, sizeof(int), &grid.nz);
    clEnqueueNDRangeKernel(queue, update_kernel, 3, nullptr, global, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    download();
}
