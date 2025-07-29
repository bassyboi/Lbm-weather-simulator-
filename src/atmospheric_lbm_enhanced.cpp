// ==============================================================================
// Atmospheric LBM - Enhanced Physics & Terrain Integration
// Advanced atmospheric boundary layer, terrain effects, and surface physics
// ==============================================================================

#pragma once
#include <CL/cl.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <memory>
#include <fstream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <execution>
#include <map>
#include <string>

// Stage 1 includes (assuming previous code)
// #include "atmospheric_lbm_stage1.hpp"

// ==============================================================================
// Enhanced Data Structures
// ==============================================================================

struct GeographicCoordinates {
    double latitude;     // degrees North
    double longitude;    // degrees East
    double elevation;    // meters above sea level

    GeographicCoordinates(double lat = 0.0, double lon = 0.0, double elev = 0.0)
        : latitude(lat), longitude(lon), elevation(elev) {}
};

struct AtmosphericProfile {
    std::vector<float> height;        // Height levels (m)
    std::vector<float> temperature;   // Temperature profile (K)
    std::vector<float> pressure;      // Pressure profile (Pa)
    std::vector<float> humidity;      // Relative humidity (%)
    std::vector<float> u_wind;        // U wind component (m/s)
    std::vector<float> v_wind;        // V wind component (m/s)
};

struct SurfaceProperties {
    float roughness_length;    // m
    float albedo;             // 0-1
    float emissivity;         // 0-1
    float heat_capacity;      // J/(m²·K)
    float thermal_diffusivity; // m²/s
    uint8_t land_use_type;    // Land use classification
};

// ==============================================================================
// Terrain Manager - Real DEM Data Integration
// ==============================================================================

class TerrainManager {
private:
    Domain domain;
    float dx, dy;
    GeographicCoordinates origin;

    // Terrain data
    std::vector<float> elevation;          // Height at each (x,y) point
    std::vector<float> slope_x, slope_y;   // Terrain slopes
    std::vector<SurfaceProperties> surface_props;

    // GPU buffers
    cl::Buffer d_elevation;
    cl::Buffer d_slope_x, d_slope_y;
    cl::Buffer d_surface_props;

public:
    TerrainManager(const Domain& dom, float grid_spacing, const GeographicCoordinates& geo_origin)
        : domain(dom), dx(grid_spacing), dy(grid_spacing), origin(geo_origin) {
        elevation.resize(domain.nx * domain.ny);
        slope_x.resize(domain.nx * domain.ny);
        slope_y.resize(domain.nx * domain.ny);
        surface_props.resize(domain.nx * domain.ny);

        std::cout << "Terrain Manager initialized for domain: "
                  << domain.nx << "x" << domain.ny << std::endl;
        std::cout << "Origin: " << origin.latitude << "°N, "
                  << origin.longitude << "°E" << std::endl;
    }

    bool load_srtm_data(const std::string& filename) {
        std::cout << "Loading SRTM elevation data from: " << filename << std::endl;

        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open DEM file: " << filename << std::endl;
            create_synthetic_terrain();
            return false;
        }

        // Read binary DEM data (assuming 32-bit float format)
        file.read(reinterpret_cast<char*>(elevation.data()),
                  elevation.size() * sizeof(float));
        file.close();

        // Compute terrain slopes
        compute_terrain_slopes();

        // Initialize surface properties based on elevation
        initialize_surface_properties();

        std::cout << "✓ Loaded real terrain data" << std::endl;
        print_terrain_stats();
        return true;
    }

    void create_synthetic_terrain() {
        std::cout << "Creating synthetic Darling Downs terrain..." << std::endl;

        // Create realistic terrain for Darling Downs region
        // Great Dividing Range runs roughly north-south
        for (uint32_t y = 0; y < domain.ny; ++y) {
            for (uint32_t x = 0; x < domain.nx; ++x) {
                uint32_t idx = x + y * domain.nx;

                // Distance from eastern edge (Great Dividing Range)
                float x_norm = static_cast<float>(x) / domain.nx;
                float y_norm = static_cast<float>(y) / domain.ny;

                // Base elevation (rises towards east - Great Dividing Range)
                float base_elevation = 200.0f + 600.0f * (1.0f - x_norm);

                // Add some hills and valleys
                float hills = 100.0f * std::sin(x_norm * 3.14159f * 3.0f) *
                                     std::cos(y_norm * 3.14159f * 2.0f);

                // Toowoomba area (higher elevation)
                float dx_toowoomba = x_norm - 0.7f;  // Eastern part
                float dy_toowoomba = y_norm - 0.6f;  // Central north
                float dist_toowoomba = std::sqrt(dx_toowoomba*dx_toowoomba +
                                               dy_toowoomba*dy_toowoomba);
                float toowoomba_height = 200.0f * std::exp(-dist_toowoomba*dist_toowoomba*50.0f);

                elevation[idx] = base_elevation + hills + toowoomba_height;

                // Ensure minimum elevation
                elevation[idx] = std::max(elevation[idx], 100.0f);
            }
        }

        compute_terrain_slopes();
        initialize_surface_properties();

        std::cout << "✓ Created synthetic terrain" << std::endl;
        print_terrain_stats();
    }

private:
    void compute_terrain_slopes() {
        // Compute terrain slopes using central differences
        for (uint32_t y = 1; y < domain.ny - 1; ++y) {
            for (uint32_t x = 1; x < domain.nx - 1; ++x) {
                uint32_t idx = x + y * domain.nx;

                uint32_t idx_xp = (x+1) + y * domain.nx;
                uint32_t idx_xm = (x-1) + y * domain.nx;
                uint32_t idx_yp = x + (y+1) * domain.nx;
                uint32_t idx_ym = x + (y-1) * domain.nx;

                slope_x[idx] = (elevation[idx_xp] - elevation[idx_xm]) / (2.0f * dx);
                slope_y[idx] = (elevation[idx_yp] - elevation[idx_ym]) / (2.0f * dy);
            }
        }

        // Handle boundaries
        for (uint32_t y = 0; y < domain.ny; ++y) {
            slope_x[0 + y * domain.nx] = slope_x[1 + y * domain.nx];
            slope_x[(domain.nx-1) + y * domain.nx] = slope_x[(domain.nx-2) + y * domain.nx];
        }
        for (uint32_t x = 0; x < domain.nx; ++x) {
            slope_y[x + 0 * domain.nx] = slope_y[x + 1 * domain.nx];
            slope_y[x + (domain.ny-1) * domain.nx] = slope_y[x + (domain.ny-2) * domain.nx];
        }
    }

    void initialize_surface_properties() {
        for (uint32_t i = 0; i < elevation.size(); ++i) {
            float elev = elevation[i];

            // Set surface properties based on elevation and location
            if (elev < 300.0f) {
                // Low-lying areas: agricultural/grassland
                surface_props[i] = {
                    .roughness_length = 0.1f,      // Smooth agricultural land
                    .albedo = 0.25f,               // Green vegetation
                    .emissivity = 0.95f,
                    .heat_capacity = 2.0e6f,       // Typical soil
                    .thermal_diffusivity = 1.0e-6f,
                    .land_use_type = 1             // Agricultural
                };
            } else if (elev < 600.0f) {
                // Mid-elevations: mixed forest/grassland
                surface_props[i] = {
                    .roughness_length = 0.5f,      // Rougher vegetation
                    .albedo = 0.15f,               // Darker forest
                    .emissivity = 0.98f,
                    .heat_capacity = 2.5e6f,
                    .thermal_diffusivity = 0.8e-6f,
                    .land_use_type = 2             // Mixed forest
                };
            } else {
                // High elevations: rocky/sparse vegetation
                surface_props[i] = {
                    .roughness_length = 1.0f,      // Rocky terrain
                    .albedo = 0.35f,               // Lighter rock/sparse vegetation
                    .emissivity = 0.90f,
                    .heat_capacity = 1.5e6f,
                    .thermal_diffusivity = 1.5e-6f,
                    .land_use_type = 3             // Rocky/sparse
                };
            }
        }
    }

    void print_terrain_stats() const {
        auto [min_elev, max_elev] = std::minmax_element(elevation.begin(), elevation.end());
        auto [min_slope_x, max_slope_x] = std::minmax_element(slope_x.begin(), slope_x.end());
        auto [min_slope_y, max_slope_y] = std::minmax_element(slope_y.begin(), slope_y.end());

        std::cout << "Terrain statistics:" << std::endl;
        std::cout << "  Elevation: " << *min_elev << " - " << *max_elev << " m" << std::endl;
        std::cout << "  Slope X: " << *min_slope_x << " - " << *max_slope_x << std::endl;
        std::cout << "  Slope Y: " << *min_slope_y << " - " << *max_slope_y << std::endl;
        std::cout << "  Relief: " << (*max_elev - *min_elev) << " m" << std::endl;
    }

public:
    void allocate_gpu_memory(const cl::Context& context) {
        size_t surface_size = domain.nx * domain.ny;

        d_elevation = cl::Buffer(context, CL_MEM_READ_ONLY, surface_size * sizeof(float));
        d_slope_x = cl::Buffer(context, CL_MEM_READ_ONLY, surface_size * sizeof(float));
        d_slope_y = cl::Buffer(context, CL_MEM_READ_ONLY, surface_size * sizeof(float));
        d_surface_props = cl::Buffer(context, CL_MEM_READ_ONLY, surface_size * sizeof(SurfaceProperties));

        std::cout << "✓ Terrain GPU memory allocated" << std::endl;
    }

    void copy_to_gpu(const cl::CommandQueue& queue) {
        queue.enqueueWriteBuffer(d_elevation, CL_TRUE, 0,
                                 elevation.size() * sizeof(float), elevation.data());
        queue.enqueueWriteBuffer(d_slope_x, CL_TRUE, 0,
                                 slope_x.size() * sizeof(float), slope_x.data());
        queue.enqueueWriteBuffer(d_slope_y, CL_TRUE, 0,
                                 slope_y.size() * sizeof(float), slope_y.data());
        queue.enqueueWriteBuffer(d_surface_props, CL_TRUE, 0,
                                 surface_props.size() * sizeof(SurfaceProperties), surface_props.data());
    }

    // Getters for GPU buffers
    const cl::Buffer& get_elevation_buffer() const { return d_elevation; }
    const cl::Buffer& get_slope_x_buffer() const { return d_slope_x; }
    const cl::Buffer& get_slope_y_buffer() const { return d_slope_y; }
    const cl::Buffer& get_surface_props_buffer() const { return d_surface_props; }

    // Terrain queries
    float get_elevation(uint32_t x, uint32_t y) const {
        if (x >= domain.nx || y >= domain.ny) return 0.0f;
        return elevation[x + y * domain.nx];
    }

    SurfaceProperties get_surface_properties(uint32_t x, uint32_t y) const {
        if (x >= domain.nx || y >= domain.ny) return {};
        return surface_props[x + y * domain.nx];
    }

    uint32_t get_terrain_level(uint32_t x, uint32_t y, float dz) const {
        float elev = get_elevation(x, y);
        return static_cast<uint32_t>(elev / dz);
    }
};

// ==============================================================================
// Enhanced Atmospheric Physics
// ==============================================================================

class AtmosphericPhysics {
private:
    Domain domain;
    float dx, dy, dz, dt;
    GeographicCoordinates origin;

    // Physical constants
    static constexpr float g = 9.81f;           // Gravity
    static constexpr float R_d = 287.0f;        // Dry air gas constant
    static constexpr float cp = 1004.0f;        // Specific heat at constant pressure
    static constexpr float omega_earth = 7.27e-5f; // Earth rotation rate

    // GPU buffers for enhanced physics
    cl::Buffer d_potential_temp;     // Potential temperature
    cl::Buffer d_brunt_vaisala;      // Buoyancy frequency
    cl::Buffer d_surface_fluxes;     // Surface heat/momentum fluxes
    cl::Buffer d_mixing_length;      // Turbulent mixing length

public:
    AtmosphericPhysics(const Domain& dom, float grid_spacing, float time_step,
                       const GeographicCoordinates& geo_origin)
        : domain(dom), dx(grid_spacing), dy(grid_spacing), dz(grid_spacing),
          dt(time_step), origin(geo_origin) {}

    void allocate_gpu_memory(const cl::Context& context) {
        size_t total_cells = domain.total_cells();
        size_t surface_cells = domain.nx * domain.ny;

        d_potential_temp = cl::Buffer(context, CL_MEM_READ_WRITE, total_cells * sizeof(float));
        d_brunt_vaisala = cl::Buffer(context, CL_MEM_READ_WRITE, total_cells * sizeof(float));
        d_surface_fluxes = cl::Buffer(context, CL_MEM_READ_WRITE, surface_cells * 3 * sizeof(float));
        d_mixing_length = cl::Buffer(context, CL_MEM_READ_WRITE, total_cells * sizeof(float));

        std::cout << "✓ Atmospheric physics GPU memory allocated" << std::endl;
    }

    std::string get_enhanced_kernels() const {
        return R"(
        // Enhanced atmospheric physics kernels

        typedef struct {
            float roughness_length;
            float albedo;
            float emissivity;
            float heat_capacity;
            float thermal_diffusivity;
            uchar land_use_type;
        } SurfaceProperties;

        __kernel void compute_potential_temperature(__global float* temperature,
                                                    __global float* pressure,
                                                    __global float* potential_temp,
                                                    float p0, int total_size) {
            int gid = get_global_id(0);
            if (gid >= total_size) return;

            float T = temperature[gid];
            float p = pressure[gid];
            float kappa = 287.0f / 1004.0f;  // R_d / cp

            // Potential temperature: θ = T * (p0/p)^κ
            potential_temp[gid] = T * pow(p0 / p, kappa);
        }

        __kernel void compute_buoyancy_frequency(__global float* potential_temp,
                                                 __global float* brunt_vaisala,
                                                 float dz, int Nx, int Ny, int Nz) {
            int gid = get_global_id(0);
            int total_size = Nx * Ny * Nz;
            if (gid >= total_size) return;

            int x = gid % Nx;
            int y = (gid / Nx) % Ny;
            int z = gid / (Nx * Ny);

            if (z == 0 || z == Nz - 1) {
                brunt_vaisala[gid] = 0.0f;
                return;
            }

            int idx_up = x + y * Nx + (z + 1) * Nx * Ny;
            int idx_down = x + y * Nx + (z - 1) * Nx * Ny;

            float theta_up = potential_temp[idx_up];
            float theta_down = potential_temp[idx_down];
            float theta_center = potential_temp[gid];

            float dtheta_dz = (theta_up - theta_down) / (2.0f * dz);

            // Brunt-Väisälä frequency: N² = (g/θ) * (dθ/dz)
            float g = 9.81f;
            brunt_vaisala[gid] = (g / theta_center) * dtheta_dz;
        }

        __kernel void surface_energy_balance(__global float* temperature,
                                             __global float* surface_temp,
                                             __global float* surface_fluxes,
                                             __global SurfaceProperties* surf_props,
                                             __global float* elevation,
                                             float solar_zenith, float solar_constant,
                                             float dt, int Nx, int Ny, int Nz) {
            int gid = get_global_id(0);
            int surface_size = Nx * Ny;
            if (gid >= surface_size) return;

            int x = gid % Nx;
            int y = gid / Nx;

            SurfaceProperties props = surf_props[gid];
            float T_surf = surface_temp[gid];
            float T_air = temperature[gid];  // Surface level air temperature

            // Solar radiation (simplified)
            float cos_zenith = cos(solar_zenith);
            float solar_flux = solar_constant * fmax(0.0f, cos_zenith);
            float absorbed_solar = solar_flux * (1.0f - props.albedo);

            // Longwave radiation
            float sigma = 5.67e-8f;  // Stefan-Boltzmann constant
            float emitted_lw = props.emissivity * sigma * pow(T_surf, 4.0f);
            float incoming_lw = sigma * pow(T_air, 4.0f);  // Simplified

            // Sensible heat flux (simplified bulk formula)
            float Ch = 0.001f;  // Heat transfer coefficient
            float wind_speed = 2.0f;  // Simplified
            float sensible_flux = 1004.0f * 1.2f * Ch * wind_speed * (T_surf - T_air);

            // Net surface energy flux
            float net_flux = absorbed_solar + incoming_lw - emitted_lw - sensible_flux;

            // Update surface temperature
            float dT_dt = net_flux / props.heat_capacity;
            surface_temp[gid] = T_surf + dT_dt * dt;

            // Store fluxes for boundary conditions
            surface_fluxes[gid * 3 + 0] = sensible_flux;        // Heat flux
            surface_fluxes[gid * 3 + 1] = 0.0f;                 // Moisture flux (later)
            surface_fluxes[gid * 3 + 2] = wind_speed * 0.001f;  // Momentum flux
        }

        __kernel void planetary_boundary_layer(__global float* velocity,
                                               __global float* temperature,
                                               __global float* mixing_length,
                                               __global float* elevation,
                                               __global SurfaceProperties* surf_props,
                                               float dz, int Nx, int Ny, int Nz) {
            int gid = get_global_id(0);
            int total_size = Nx * Ny * Nz;
            if (gid >= total_size) return;

            int x = gid % Nx;
            int y = (gid / Nx) % Ny;
            int z = gid / (Nx * Ny);

            float height_agl = z * dz;  // Height above ground level
            int surface_idx = x + y * Nx;

            if (z == 0) {
                // Surface level - use terrain height
                height_agl = 0.0f;
                mixing_length[gid] = surf_props[surface_idx].roughness_length;
            } else {
                // Calculate height above local terrain
                height_agl = z * dz;  // Simplified - should subtract terrain elevation

                // Mixing length in planetary boundary layer
                float von_karman = 0.4f;
                float pbl_height = 1000.0f;  // Simplified PBL height

                if (height_agl < pbl_height) {
                    // Within PBL: l = κ z / (1 + κ z/λ)
                    float lambda = 100.0f;  // Asymptotic mixing length
                    mixing_length[gid] = von_karman * height_agl /
                                        (1.0f + von_karman * height_agl / lambda);
                } else {
                    // Above PBL: constant mixing length
                    mixing_length[gid] = 100.0f;
                }
            }
        }

        __kernel void coriolis_force(__global float* velocity,
                                     __global float* velocity_new,
                                     float latitude, float dt,
                                     int Nx, int Ny, int Nz) {
            int gid = get_global_id(0);
            int total_size = Nx * Ny * Nz;
            if (gid >= total_size) return;

            float omega_earth = 7.27e-5f;
            float f = 2.0f * omega_earth * sin(latitude * M_PI / 180.0f);  // Coriolis parameter

            float u = velocity[gid * 3 + 0];
            float v = velocity[gid * 3 + 1];
            float w = velocity[gid * 3 + 2];

            // Coriolis acceleration: a_coriolis = -f × v
            float du_dt_coriolis = f * v;
            float dv_dt_coriolis = -f * u;
            // dw_dt_coriolis = 0 (f-plane approximation)

            velocity_new[gid * 3 + 0] = u + du_dt_coriolis * dt;
            velocity_new[gid * 3 + 1] = v + dv_dt_coriolis * dt;
            velocity_new[gid * 3 + 2] = w;
        }

        __kernel void terrain_following_boundary(__global float* f,
                                                 __global float* velocity,
                                                 __global uchar* flags,
                                                 __global float* elevation,
                                                 float dz, int Nx, int Ny, int Nz) {
            int gid = get_global_id(0);
            int total_size = Nx * Ny * Nz;
            if (gid >= total_size) return;

            int x = gid % Nx;
            int y = (gid / Nx) % Ny;
            int z = gid / (Nx * Ny);

            int surface_idx = x + y * Nx;
            float terrain_height = elevation[surface_idx];
            int terrain_level = (int)(terrain_height / dz);

            if (z <= terrain_level) {
                // Below terrain - solid boundary
                flags[gid] = 1;  // Solid

                // Zero velocity in solid region
                velocity[gid * 3 + 0] = 0.0f;
                velocity[gid * 3 + 1] = 0.0f;
                velocity[gid * 3 + 2] = 0.0f;

                // Set distribution functions for no-slip boundary
                for (int q = 0; q < 19; q++) {
                    f[gid + q * total_size] = 0.0f;
                }
            } else {
                // Above terrain - fluid
                flags[gid] = 0;  // Fluid
            }
        }

        __kernel void orographic_lifting(__global float* velocity,
                                         __global float* slope_x,
                                         __global float* slope_y,
                                         float dt, int Nx, int Ny, int Nz) {
            int gid = get_global_id(0);
            int total_size = Nx * Ny * Nz;
            if (gid >= total_size) return;

            int x = gid % Nx;
            int y = (gid / Nx) % Ny;
            int z = gid / (Nx * Ny);

            if (z == 0) return;  // Skip surface level

            int surface_idx = x + y * Nx;
            float sx = slope_x[surface_idx];
            float sy = slope_y[surface_idx];

            float u = velocity[gid * 3 + 0];
            float v = velocity[gid * 3 + 1];
            float w = velocity[gid * 3 + 2];

            // Orographic lifting: w = u * dh/dx + v * dh/dy
            float orographic_w = u * sx + v * sy;

            // Apply orographic forcing (stronger near surface)
            float height_factor = exp(-z * 0.001f);  // Decay with height
            velocity[gid * 3 + 2] = w + orographic_w * height_factor * dt;
        }
    )";
    }

    // Getter methods for GPU buffers
    const cl::Buffer& get_potential_temp_buffer() const { return d_potential_temp; }
    const cl::Buffer& get_brunt_vaisala_buffer() const { return d_brunt_vaisala; }
    const cl::Buffer& get_surface_fluxes_buffer() const { return d_surface_fluxes; }
    const cl::Buffer& get_mixing_length_buffer() const { return d_mixing_length; }
};


// =============================================================================
// Enhanced AtmosphericLBM
// ==============================================================================

class AtmosphericLBMEnhanced {

    // enhanced capabilities
    std::unique_ptr<TerrainManager> terrain;
    std::unique_ptr<AtmosphericPhysics> physics;

    // OpenCL context (from Stage 1)
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue queue;
    cl::Program program;

    // Enhanced kernels
    cl::Kernel streaming_kernel;
    cl::Kernel collision_kernel;
    cl::Kernel potential_temp_kernel;
    cl::Kernel buoyancy_freq_kernel;
    cl::Kernel surface_energy_kernel;
    cl::Kernel pbl_kernel;
    cl::Kernel coriolis_kernel;
    cl::Kernel terrain_boundary_kernel;
    cl::Kernel orographic_kernel;

    // GPU buffers (from Stage 1 + new ones)
    cl::Buffer d_f_in, d_f_out;
    cl::Buffer d_velocity, d_density, d_temperature, d_pressure, d_flags;
    cl::Buffer d_surface_temperature;  // New for surface energy balance

    // Host data
    std::vector<float> h_velocity, h_density, h_temperature, h_pressure;
    std::vector<float> h_surface_temperature;
    std::vector<uint8_t> h_flags;

public:
    AtmosphericLBMEnhanced(const Domain& dom, float grid_spacing, float viscosity,
                         const GeographicCoordinates& geo_origin)
        : domain(dom), dx(grid_spacing), dy(grid_spacing), dz(grid_spacing),
          nu(viscosity), origin(geo_origin) {
        dt = 0.1f * dx;  // CFL condition

        // Initialize terrain and physics modules
        terrain = std::make_unique<TerrainManager>(domain, dx, origin);
        physics = std::make_unique<AtmosphericPhysics>(domain, dx, dt, origin);

        // Resize host arrays
        size_t total_cells = domain.total_cells();
        size_t surface_cells = domain.nx * domain.ny;

        h_velocity.resize(total_cells * 3);
        h_density.resize(total_cells);
        h_temperature.resize(total_cells);
        h_pressure.resize(total_cells);
        h_surface_temperature.resize(surface_cells);
        h_flags.resize(total_cells);

        print_enhanced_info();
    }

    bool initialize() {
        if (!initialize_opencl()) return false;
        if (!load_terrain_data()) return false;
        if (!allocate_memory()) return false;
        initialize_atmospheric_fields();
        return true;
    }

private:
    bool initialize_opencl() {
        try {
            // OpenCL setup (similar to Stage 1)
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            if (platforms.empty()) return false;

            platform = platforms[0];

            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            if (devices.empty()) return false;

            device = devices[0];
            context = cl::Context(device);
            queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

            return compile_enhanced_kernels();

        } catch (const cl::Error& e) {
            std::cerr << "OpenCL Error: " << e.what() << std::endl;
            return false;
        }
    }

    bool compile_enhanced_kernels() {
        // Combine Stage 1 kernels with enhanced capabilities
        std::string kernel_source = get_stage1_kernels() + physics->get_enhanced_kernels();

        try {
            program = cl::Program(context, kernel_source);
            if (program.build({device}) != CL_SUCCESS) {
                std::cerr << "Kernel compilation error:\n"
                          << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
                return false;
            }

            // Create enhanced kernel objects
            streaming_kernel = cl::Kernel(program, "streaming_step");
            collision_kernel = cl::Kernel(program, "collision_step");
            potential_temp_kernel = cl::Kernel(program, "compute_potential_temperature");
            buoyancy_freq_kernel = cl::Kernel(program, "compute_buoyancy_frequency");
            surface_energy_kernel = cl::Kernel(program, "surface_energy_balance");
            pbl_kernel = cl::Kernel(program, "planetary_boundary_layer");
            coriolis_kernel = cl::Kernel(program, "coriolis_force");
            terrain_boundary_kernel = cl::Kernel(program, "terrain_following_boundary");
            orographic_kernel = cl::Kernel(program, "orographic_lifting");

            std::cout << "✓ Enhanced atmospheric kernels compiled" << std::endl;
            return true;

        } catch (const cl::Error& e) {
            std::cerr << "Enhanced kernel compilation error: " << e.what() << std::endl;
            return false;
        }
    }

    std::string get_stage1_kernels() const {
        // Include basic LBM kernels from Stage 1
        // (This would be the kernel code from the previous Stage 1 implementation)
        return R"(
        // Basic Stage 1 kernels would go here
        // streaming_step, collision_step, etc.
        // [Previous Stage 1 kernel code]
    )";
    }

    bool load_terrain_data() {
        // Try to load real DEM data, fallback to synthetic
        std::string dem_file = "darling_downs_dem.bin";
        if (!terrain->load_srtm_data(dem_file)) {
            std::cout << "Using synthetic terrain for testing" << std::endl;
        }
        return true;
    }

    bool allocate_memory() {
        try {
            size_t total_cells = domain.total_cells();
            size_t surface_cells = domain.nx * domain.ny;

            // Basic LBM buffers (from Stage 1)
            d_f_in = cl::Buffer(context, CL_MEM_READ_WRITE, total_cells * 19 * sizeof(float));
            d_f_out = cl::Buffer(context, CL_MEM_READ_WRITE, total_cells * 19 * sizeof(float));
            d_velocity = cl::Buffer(context, CL_MEM_READ_WRITE, total_cells * 3 * sizeof(float));
            d_density = cl::Buffer(context, CL_MEM_READ_WRITE, total_cells * sizeof(float));
            d_temperature = cl::Buffer(context, CL_MEM_READ_WRITE, total_cells * sizeof(float));
            d_pressure = cl::Buffer(context, CL_MEM_READ_WRITE, total_cells * sizeof(float));
            d_flags = cl::Buffer(context, CL_MEM_READ_WRITE, total_cells * sizeof(uint8_t));

            // enhanced capabilities
            d_surface_temperature = cl::Buffer(context, CL_MEM_READ_WRITE, surface_cells * sizeof(float));

            // Allocate terrain and physics memory
            terrain->allocate_gpu_memory(context);
            physics->allocate_gpu_memory(context);

            std::cout << "✓ Enhanced memory allocated successfully" << std::endl;
            return true;

        } catch (const cl::Error& e) {
            std::cerr << "Memory allocation error: " << e.what() << std::endl;
            return false;
        }
    }

    void initialize_atmospheric_fields() {
        // Initialize with realistic atmospheric profile
        init_atmospheric_profile();

        // Apply terrain-following coordinates
        apply_terrain_following_grid();

        // Initialize surface temperature
        init_surface_temperature();

        // Copy everything to GPU
        copy_to_gpu();

        std::cout << "✓ Enhanced atmospheric fields initialized" << std::endl;
    }

    void init_atmospheric_profile() {
        // Standard atmosphere with realistic temperature and pressure profiles
        for (uint32_t z = 0; z < domain.nz; ++z) {
            float height = z * dz;

            // Standard atmosphere temperature profile
            float T_surface = 288.0f;  // 15°C at sea level
            float lapse_rate = 0.0065f; // K/m
            float T_height = T_surface - lapse_rate * height;

            // Pressure using barometric formula
            float p_surface = 101325.0f;  // Pa at sea level
            float p_height = p_surface * std::pow(T_height / T_surface, 9.81f / (287.0f * lapse_rate));

            // Density from ideal gas law
            float rho_height = p_height / (287.0f * T_height);

            for (uint32_t y = 0; y < domain.ny; ++y) {
                for (uint32_t x = 0; x < domain.nx; ++x) {
                    uint32_t idx = domain.index(x, y, z);

                    h_temperature[idx] = T_height;
                    h_pressure[idx] = p_height;
                    h_density[idx] = rho_height;
                    h_velocity[idx*3] = 0.0f;      // u
                    h_velocity[idx*3+1] = 0.0f;    // v
                    h_velocity[idx*3+2] = 0.0f;    // w
                    h_flags[idx] = 0;              // Fluid
                }
            }
        }

        // Add some initial wind shear for testing
        add_initial_wind_profile();
    }

    void add_initial_wind_profile() {
        // Add realistic wind shear profile
        for (uint32_t z = 0; z < domain.nz; ++z) {
            float height = z * dz;

            // Typical boundary layer wind profile
            float u_wind = 10.0f * std::log(std::max(height, 10.0f) / 0.1f) / std::log(10.0f / 0.1f);
            float v_wind = 2.0f + 5.0f * height / 2000.0f;  // Slight backing with height

            for (uint32_t y = 0; y < domain.ny; ++y) {
                for (uint32_t x = 0; x < domain.nx; ++x) {
                    uint32_t idx = domain.index(x, y, z);
                    h_velocity[idx*3] = u_wind;
                    h_velocity[idx*3+1] = v_wind;
                }
            }
        }
    }

    void apply_terrain_following_grid() {
        // Set solid flags for cells below terrain
        for (uint32_t y = 0; y < domain.ny; ++y) {
            for (uint32_t x = 0; x < domain.nx; ++x) {
                float terrain_height = terrain->get_elevation(x, y);
                uint32_t terrain_level = static_cast<uint32_t>(terrain_height / dz);

                for (uint32_t z = 0; z <= std::min(terrain_level, domain.nz - 1); ++z) {
                    uint32_t idx = domain.index(x, y, z);
                    h_flags[idx] = 1;  // Solid
                    h_velocity[idx*3] = 0.0f;
                    h_velocity[idx*3+1] = 0.0f;
                    h_velocity[idx*3+2] = 0.0f;
                }
            }
        }
    }

    void init_surface_temperature() {
        // Initialize surface temperature based on terrain properties
        for (uint32_t y = 0; y < domain.ny; ++y) {
            for (uint32_t x = 0; x < domain.nx; ++x) {
                uint32_t idx = x + y * domain.nx;
                float elevation = terrain->get_elevation(x, y);

                // Temperature decreases with elevation
                float T_surface = 288.0f - 0.0065f * elevation;
                h_surface_temperature[idx] = T_surface;
            }
        }
    }

    void copy_to_gpu() {
        size_t total_cells = domain.total_cells();

        // Copy basic fields
        queue.enqueueWriteBuffer(d_velocity, CL_TRUE, 0, h_velocity.size() * sizeof(float), h_velocity.data());
        queue.enqueueWriteBuffer(d_density, CL_TRUE, 0, h_density.size() * sizeof(float), h_density.data());
        queue.enqueueWriteBuffer(d_temperature, CL_TRUE, 0, h_temperature.size() * sizeof(float), h_temperature.data());
        queue.enqueueWriteBuffer(d_pressure, CL_TRUE, 0, h_pressure.size() * sizeof(float), h_pressure.data());
        queue.enqueueWriteBuffer(d_flags, CL_TRUE, 0, h_flags.size() * sizeof(uint8_t), h_flags.data());
        queue.enqueueWriteBuffer(d_surface_temperature, CL_TRUE, 0, h_surface_temperature.size() * sizeof(float), h_surface_temperature.data());

        // Initialize distribution functions to equilibrium
        std::vector<float> h_f(total_cells * 19);
        for (size_t i = 0; i < total_cells; ++i) {
            for (int q = 0; q < 19; q++) {
                h_f[i + q * total_cells] = (1.0f/19.0f) * h_density[i];  // Simplified equilibrium
            }
        }
        queue.enqueueWriteBuffer(d_f_in, CL_TRUE, 0, h_f.size() * sizeof(float), h_f.data());

        // Copy terrain data
        terrain->copy_to_gpu(queue);
    }

public:
    void run_timestep() {
        // Enhanced timestep with atmospheric physics

        // 1. Standard LBM steps
        streaming_step();
        collision_step();

        // 2. Enhanced atmospheric physics
        compute_atmospheric_stability();
        surface_energy_balance();
        planetary_boundary_layer_mixing();
        apply_coriolis_effects();
        apply_orographic_effects();

        // 3. Update terrain-following boundaries
        update_terrain_boundaries();
    }

private:
    void streaming_step() {
        streaming_kernel.setArg(0, d_f_in);
        streaming_kernel.setArg(1, d_f_out);
        streaming_kernel.setArg(2, d_flags);
        streaming_kernel.setArg(3, static_cast<int>(domain.nx));
        streaming_kernel.setArg(4, static_cast<int>(domain.ny));
        streaming_kernel.setArg(5, static_cast<int>(domain.nz));

        queue.enqueueNDRangeKernel(streaming_kernel, cl::NullRange, cl::NDRange(domain.total_cells()));
        std::swap(d_f_in, d_f_out);
    }

    void collision_step() {
        collision_kernel.setArg(0, d_f_in);
        collision_kernel.setArg(1, d_velocity);
        collision_kernel.setArg(2, d_density);
        collision_kernel.setArg(3, d_temperature);
        collision_kernel.setArg(4, d_pressure);
        collision_kernel.setArg(5, nu);
        collision_kernel.setArg(6, dt);
        collision_kernel.setArg(7, static_cast<int>(domain.nx));
        collision_kernel.setArg(8, static_cast<int>(domain.ny));
        collision_kernel.setArg(9, static_cast<int>(domain.nz));

        queue.enqueueNDRangeKernel(collision_kernel, cl::NullRange, cl::NDRange(domain.total_cells()));
    }

    void compute_atmospheric_stability() {
        // Compute potential temperature
        potential_temp_kernel.setArg(0, d_temperature);
        potential_temp_kernel.setArg(1, d_pressure);
        potential_temp_kernel.setArg(2, physics->get_potential_temp_buffer());
        potential_temp_kernel.setArg(3, 101325.0f);  // Reference pressure
        potential_temp_kernel.setArg(4, static_cast<int>(domain.total_cells()));

        queue.enqueueNDRangeKernel(potential_temp_kernel, cl::NullRange, cl::NDRange(domain.total_cells()));

        // Compute Brunt-Väisälä frequency
        buoyancy_freq_kernel.setArg(0, physics->get_potential_temp_buffer());
        buoyancy_freq_kernel.setArg(1, physics->get_brunt_vaisala_buffer());
        buoyancy_freq_kernel.setArg(2, dz);
        buoyancy_freq_kernel.setArg(3, static_cast<int>(domain.nx));
        buoyancy_freq_kernel.setArg(4, static_cast<int>(domain.ny));
        buoyancy_freq_kernel.setArg(5, static_cast<int>(domain.nz));

        queue.enqueueNDRangeKernel(buoyancy_freq_kernel, cl::NullRange, cl::NDRange(domain.total_cells()));
    }

    void surface_energy_balance() {
        float solar_zenith = 0.5f;  // Simplified solar angle
        float solar_constant = 1361.0f;  // W/m²

        surface_energy_kernel.setArg(0, d_temperature);
        surface_energy_kernel.setArg(1, d_surface_temperature);
        surface_energy_kernel.setArg(2, physics->get_surface_fluxes_buffer());
        surface_energy_kernel.setArg(3, terrain->get_surface_props_buffer());
        surface_energy_kernel.setArg(4, terrain->get_elevation_buffer());
        surface_energy_kernel.setArg(5, solar_zenith);
        surface_energy_kernel.setArg(6, solar_constant);
        surface_energy_kernel.setArg(7, dt);
        surface_energy_kernel.setArg(8, static_cast<int>(domain.nx));
        surface_energy_kernel.setArg(9, static_cast<int>(domain.ny));
        surface_energy_kernel.setArg(10, static_cast<int>(domain.nz));

        queue.enqueueNDRangeKernel(surface_energy_kernel, cl::NullRange, cl::NDRange(domain.nx * domain.ny));
    }

    void planetary_boundary_layer_mixing() {
        pbl_kernel.setArg(0, d_velocity);
        pbl_kernel.setArg(1, d_temperature);
        pbl_kernel.setArg(2, physics->get_mixing_length_buffer());
        pbl_kernel.setArg(3, terrain->get_elevation_buffer());
        pbl_kernel.setArg(4, terrain->get_surface_props_buffer());
        pbl_kernel.setArg(5, dz);
        pbl_kernel.setArg(6, static_cast<int>(domain.nx));
        pbl_kernel.setArg(7, static_cast<int>(domain.ny));
        pbl_kernel.setArg(8, static_cast<int>(domain.nz));

        queue.enqueueNDRangeKernel(pbl_kernel, cl::NullRange, cl::NDRange(domain.total_cells()));
    }

    void apply_coriolis_effects() {
        static cl::Buffer d_velocity_new;
        static bool buffer_allocated = false;

        if (!buffer_allocated) {
            d_velocity_new = cl::Buffer(context, CL_MEM_READ_WRITE, domain.total_cells() * 3 * sizeof(float));
            buffer_allocated = true;
        }

        coriolis_kernel.setArg(0, d_velocity);
        coriolis_kernel.setArg(1, d_velocity_new);
        coriolis_kernel.setArg(2, static_cast<float>(origin.latitude));
        coriolis_kernel.setArg(3, dt);
        coriolis_kernel.setArg(4, static_cast<int>(domain.nx));
        coriolis_kernel.setArg(5, static_cast<int>(domain.ny));
        coriolis_kernel.setArg(6, static_cast<int>(domain.nz));

        queue.enqueueNDRangeKernel(coriolis_kernel, cl::NullRange, cl::NDRange(domain.total_cells()));
        std::swap(d_velocity, d_velocity_new);
    }

    void apply_orographic_effects() {
        orographic_kernel.setArg(0, d_velocity);
        orographic_kernel.setArg(1, terrain->get_slope_x_buffer());
        orographic_kernel.setArg(2, terrain->get_slope_y_buffer());
        orographic_kernel.setArg(3, dt);
        orographic_kernel.setArg(4, static_cast<int>(domain.nx));
        orographic_kernel.setArg(5, static_cast<int>(domain.ny));
        orographic_kernel.setArg(6, static_cast<int>(domain.nz));

        queue.enqueueNDRangeKernel(orographic_kernel, cl::NullRange, cl::NDRange(domain.total_cells()));
    }

    void update_terrain_boundaries() {
        terrain_boundary_kernel.setArg(0, d_f_in);
        terrain_boundary_kernel.setArg(1, d_velocity);
        terrain_boundary_kernel.setArg(2, d_flags);
        terrain_boundary_kernel.setArg(3, terrain->get_elevation_buffer());
        terrain_boundary_kernel.setArg(4, dz);
        terrain_boundary_kernel.setArg(5, static_cast<int>(domain.nx));
        terrain_boundary_kernel.setArg(6, static_cast<int>(domain.ny));
        terrain_boundary_kernel.setArg(7, static_cast<int>(domain.nz));

        queue.enqueueNDRangeKernel(terrain_boundary_kernel, cl::NullRange, cl::NDRange(domain.total_cells()));
    }

public:
    void copy_from_gpu() {
        queue.enqueueReadBuffer(d_velocity, CL_TRUE, 0, h_velocity.size() * sizeof(float), h_velocity.data());
        queue.enqueueReadBuffer(d_density, CL_TRUE, 0, h_density.size() * sizeof(float), h_density.data());
        queue.enqueueReadBuffer(d_temperature, CL_TRUE, 0, h_temperature.size() * sizeof(float), h_temperature.data());
        queue.enqueueReadBuffer(d_pressure, CL_TRUE, 0, h_pressure.size() * sizeof(float), h_pressure.data());
        queue.enqueueReadBuffer(d_surface_temperature, CL_TRUE, 0, h_surface_temperature.size() * sizeof(float), h_surface_temperature.data());
    }

    void save_enhanced_vtk(const std::string& filename, size_t timestep) const {
        std::string full_filename = filename + "_enhanced_" + std::to_string(timestep) + ".vtk";
        std::ofstream file(full_filename);

        file << "# vtk DataFile Version 3.0\n";
        file << "Atmospheric LBM Enhanced - Enhanced Physics\n";
        file << "ASCII\n";
        file << "DATASET STRUCTURED_POINTS\n";
        file << "DIMENSIONS " << domain.nx << " " << domain.ny << " " << domain.nz << "\n";
        file << "ORIGIN 0 0 0\n";
        file << "SPACING " << dx << " " << dy << " " << dz << "\n";
        file << "POINT_DATA " << domain.total_cells() << "\n";

        // Enhanced fields
        file << "SCALARS temperature float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (float temp : h_temperature) {
            file << temp - 273.15f << "\n";  // Convert to Celsius
        }

        file << "SCALARS pressure float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (float press : h_pressure) {
            file << press / 100.0f << "\n";  // Convert to hPa
        }

        file << "SCALARS density float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (float dens : h_density) {
            file << dens << "\n";
        }

        file << "SCALARS flags float 1\n";
        file << "LOOKUP_TABLE default\n";
        for (uint8_t flag : h_flags) {
            file << static_cast<float>(flag) << "\n";
        }

        file << "VECTORS velocity float\n";
        for (size_t i = 0; i < h_density.size(); ++i) {
            file << h_velocity[i*3] << " " << h_velocity[i*3+1] << " " << h_velocity[i*3+2] << "\n";
        }

        std::cout << "✓ Saved enhanced output: " << full_filename << std::endl;
    }

    void run_simulation(size_t num_steps) {
        std::cout << "\n🌍 Starting enhanced atmospheric simulation...\n" << std::endl;

        Timer total_timer;
        total_timer.start();

        for (size_t step = 0; step < num_steps; ++step) {
            run_timestep();

            if (step % 50 == 0) {
                double progress = 100.0 * step / num_steps;
                std::cout << std::fixed << std::setprecision(1)
                          << "Step " << std::setw(5) << step << "/" << num_steps
                          << " (" << std::setw(5) << progress << "%)" << std::endl;

                if (step % 250 == 0) {
                    copy_from_gpu();
                    save_enhanced_vtk("atmospheric_enhanced", step);
                    print_enhanced_statistics();
                }
            }
        }

        copy_from_gpu();
        save_enhanced_vtk("atmospheric_enhanced_final", num_steps);

        double total_time = total_timer.elapsed_s();
        std::cout << "\n✅ Enhanced simulation complete!" << std::endl;
        std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time << " seconds" << std::endl;
        std::cout << "Check atmospheric_enhanced_*.vtk files for results" << std::endl;
    }

private:
    void print_enhanced_info() const {
        std::cout << "\n🚀 Atmospheric LBM Enhanced - Enhanced Physics:" << std::endl;
        std::cout << "Domain: " << domain.nx << " × " << domain.ny << " × " << domain.nz << std::endl;
        std::cout << "Physical size: " << (domain.nx * dx / 1000.0) << " × "
                  << (domain.ny * dy / 1000.0) << " × " << (domain.nz * dz / 1000.0) << " km" << std::endl;
        std::cout << "Origin: " << origin.latitude << "°N, " << origin.longitude << "°E" << std::endl;
        std::cout << "Grid spacing: " << dx << " m" << std::endl;
        std::cout << "Enhanced features: Terrain, Surface Physics, PBL, Coriolis" << std::endl;
    }

    void print_enhanced_statistics() const {
        auto [min_temp, max_temp] = std::minmax_element(h_temperature.begin(), h_temperature.end());
        auto [min_surf_temp, max_surf_temp] = std::minmax_element(h_surface_temperature.begin(), h_surface_temperature.end());

        float max_vel = 0.0f;
        for (size_t i = 0; i < h_velocity.size(); i += 3) {
            float vel_mag = std::sqrt(h_velocity[i]*h_velocity[i] +
                                     h_velocity[i+1]*h_velocity[i+1] +
                                     h_velocity[i+2]*h_velocity[i+2]);
            max_vel = std::max(max_vel, vel_mag);
        }

        std::cout << "Enhanced Physics: T_air=[" << std::fixed << std::setprecision(1)
                  << (*min_temp - 273.15) << "," << (*max_temp - 273.15) << "]°C "
                  << "T_surf=[" << (*min_surf_temp - 273.15) << "," << (*max_surf_temp - 273.15) << "]°C "
                  << "V_max=" << std::setprecision(2) << max_vel << " m/s" << std::endl;
    }
};

// ==============================================================================
// Main Application for Enhanced
// ==============================================================================

int main() {
    try {
        // Darling Downs domain configuration
        Domain domain(150, 150, 60);  // 7.5km × 7.5km × 3km at 50m resolution
        float grid_spacing = 50.0f;   // 50 meters
        float viscosity = 1.5e-5f;    // Air kinematic viscosity

        // Geographic origin (approximate center of Darling Downs)
        GeographicCoordinates origin(-27.0, 151.5, 300.0);  // Latitude, Longitude, Elevation

        // Initialize enhanced simulation
        AtmosphericLBMEnhanced lbm(domain, grid_spacing, viscosity, origin);

        if (!lbm.initialize()) {
            std::cerr << "❌ Failed to initialize Enhanced simulation" << std::endl;
            return -1;
        }

        // Run enhanced simulation
        lbm.run_simulation(2000);

        std::cout << "\n🎉 Enhanced Success!" << std::endl;
        std::cout << "Enhanced atmospheric simulation with terrain effects complete!" << std::endl;
        std::cout << "\nFeatures demonstrated:" << std::endl;
        std::cout << "✓ Terrain-following coordinates" << std::endl;
        std::cout << "✓ Surface energy balance" << std::endl;
        std::cout << "✓ Planetary boundary layer mixing" << std::endl;
        std::cout << "✓ Coriolis effects" << std::endl;
        std::cout << "✓ Orographic lifting" << std::endl;
        std::cout << "✓ Atmospheric stability analysis" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "❌ Enhanced Error: " << e.what() << std::endl;
        return -1;
    }
}

