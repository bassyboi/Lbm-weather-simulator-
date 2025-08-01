cmake_minimum_required(VERSION 3.15)
project(AtmosphericLBM VERSION 1.0.0 LANGUAGES CXX)

# Modern C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -ffast-math")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /arch:AVX2")
  set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi")
endif()

# Optionally enable OpenCL
option(ENABLE_OPENCL "Build with OpenCL support" ON)
if(ENABLE_OPENCL)
  find_package(OpenCL QUIET)
  if(OpenCL_FOUND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_OPENCL)
  else()
    message(WARNING "OpenCL not found - continuing without GPU acceleration")
  endif()
endif()

# Find optional packages
find_package(PkgConfig)

# NetCDF for weather data (optional)
if(PkgConfig_FOUND)
  pkg_check_modules(NETCDF netcdf)
  if(NETCDF_FOUND)
    add_definitions(-DHAVE_NETCDF)
  endif()
endif()

# HDF5 for weather data (optional)
find_package(HDF5 COMPONENTS CXX)
if(HDF5_FOUND)
  add_definitions(-DHAVE_HDF5)
endif()

# Threading support
find_package(Threads REQUIRED)

# OpenMP for CPU parallelization (optional)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  add_definitions(-DHAVE_OPENMP)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
if(OpenCL_FOUND)
  include_directories(${OpenCL_INCLUDE_DIRS})
endif()

# Source files shared by all executables
set(COMMON_SOURCES
  src/atmospheric_lbm.cpp
  src/tornado_simulator.cpp
  src/weather_data.cpp
  src/terrain_manager.cpp
  src/boundary_conditions.cpp
  src/visualization.cpp
)

# Header files
set(HEADERS
  include/atmospheric_lbm.hpp
  include/tornado_simulator.hpp
  include/weather_data.hpp
  include/terrain_manager.hpp
  include/boundary_conditions.hpp
  include/visualization.hpp
  include/common.hpp
)

# Main executable
add_executable(atmospheric_lbm src/main.cpp ${COMMON_SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(atmospheric_lbm Threads::Threads)
if(OpenCL_FOUND)
  target_link_libraries(atmospheric_lbm ${OpenCL_LIBRARIES})
endif()

# Optional library linking
if(NETCDF_FOUND)
  target_link_libraries(atmospheric_lbm ${NETCDF_LIBRARIES})
  target_include_directories(atmospheric_lbm PRIVATE ${NETCDF_INCLUDE_DIRS})
endif()

if(HDF5_FOUND)
  target_link_libraries(atmospheric_lbm ${HDF5_CXX_LIBRARIES})
  target_include_directories(atmospheric_lbm PRIVATE ${HDF5_CXX_INCLUDE_DIRS})
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(atmospheric_lbm OpenMP::OpenMP_CXX)
endif()

# Additional executables for different simulation modes
add_executable(tornado_sim src/tornado_main.cpp ${COMMON_SOURCES})
target_link_libraries(tornado_sim Threads::Threads)
if(OpenCL_FOUND)
  target_link_libraries(tornado_sim ${OpenCL_LIBRARIES})
endif()

add_executable(weather_forecast src/forecast_main.cpp ${COMMON_SOURCES})
target_link_libraries(weather_forecast Threads::Threads)
if(OpenCL_FOUND)
  target_link_libraries(weather_forecast ${OpenCL_LIBRARIES})
endif()

# Copy OpenCL kernels to build directory
if(OpenCL_FOUND)
  file(COPY ${CMAKE_SOURCE_DIR}/kernels/ DESTINATION ${CMAKE_BINARY_DIR}/kernels/)
endif()

# Installation
install(TARGETS atmospheric_lbm tornado_sim weather_forecast
        RUNTIME DESTINATION bin)

install(DIRECTORY kernels/ DESTINATION share/atmospheric_lbm/kernels/)
install(DIRECTORY config/ DESTINATION share/atmospheric_lbm/config/)
install(FILES README.md LICENSE DESTINATION share/atmospheric_lbm/)

# Testing (optional)
enable_testing()

add_executable(test_lbm tests/test_lbm.cpp ${COMMON_SOURCES})
if(OpenCL_FOUND)
  target_link_libraries(test_lbm ${OpenCL_LIBRARIES})
endif()
add_test(NAME LBMTest COMMAND test_lbm)

add_executable(test_tornado tests/test_tornado.cpp ${COMMON_SOURCES})
if(OpenCL_FOUND)
  target_link_libraries(test_tornado ${OpenCL_LIBRARIES})
endif()
add_test(NAME TornadoTest COMMAND test_tornado)

# Benchmarks
add_executable(benchmark_lbm benchmarks/benchmark_lbm.cpp ${COMMON_SOURCES})
if(OpenCL_FOUND)
  target_link_libraries(benchmark_lbm ${OpenCL_LIBRARIES})
endif()

# Documentation (requires Doxygen)
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_SOURCE_DIR}/docs/Doxyfile.in
                 ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)
  add_custom_target(docs
    ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen"
    VERBATIM)
endif()

# Package configuration
set(CPACK_PACKAGE_NAME "AtmosphericLBM")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-resolution atmospheric modeling using Lattice Boltzmann Method")
set(CPACK_PACKAGE_VENDOR "Your Organization")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")

include(CPack)

# Print configuration summary
message(STATUS "=== AtmosphericLBM Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "OpenCL found: ${OpenCL_FOUND}")
message(STATUS "NetCDF found: ${NETCDF_FOUND}")
message(STATUS "HDF5 found: ${HDF5_FOUND}")
message(STATUS "OpenMP found: ${OpenMP_CXX_FOUND}")
message(STATUS "===================================")
