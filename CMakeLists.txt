cmake_minimum_required(VERSION 3.17)

project(QUAGO
    VERSION 1.0.0
    DESCRIPTION "QUAGO: query-agnostic graph enhancement for approximate nearest neighbor search"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Use Boost's own CMake config (CMake >= 3.30 removed the legacy FindBoost module).
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()
option(QUAGO_NATIVE_BUILD "Compile with -march=native -mtune=native (disable for portable binaries)" ON)

# ---------------------------------------------------------------------------
# Dependencies: OpenMP, Boost.program_options, Intel MKL (BLAS for the kmeans tool).
# ---------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)

# --- Intel oneAPI MKL (ILP64 interface, OpenMP threading) ------------------
# Search hints, in order:
#   - MKL_ROOT cache variable (default /opt/intel/oneapi/mkl/latest)
#   - $ENV{MKLROOT}
#   - common distro locations (/usr/include/mkl, /usr/lib/x86_64-linux-gnu/mkl)
# Defines the imported target MKL::MKL plus MKL_FOUND / MKL_INCLUDE_DIRS / MKL_LIBRARIES.
set(MKL_ROOT "/opt/intel/oneapi/mkl/latest" CACHE PATH "Intel oneAPI MKL root directory")

find_path(MKL_INCLUDE_DIR NAMES mkl.h HINTS
    ${MKL_ROOT}/include
    $ENV{MKLROOT}/include
    /usr/include/mkl
    /usr/include)

set(_mkl_lib_hints
    ${MKL_ROOT}/lib/intel64
    $ENV{MKLROOT}/lib/intel64
    /usr/lib/x86_64-linux-gnu/mkl
    /usr/lib/x86_64-linux-gnu
    /usr/lib64)

# Link order matters for MKL's static/dynamic layers: interface, threading, core, then iomp5.
find_library(MKL_INTERFACE_LIB NAMES mkl_intel_ilp64 HINTS ${_mkl_lib_hints})
find_library(MKL_THREADING_LIB NAMES mkl_intel_thread HINTS ${_mkl_lib_hints})
find_library(MKL_CORE_LIB      NAMES mkl_core         HINTS ${_mkl_lib_hints})
find_library(MKL_IOMP5_LIB     NAMES iomp5            HINTS ${_mkl_lib_hints})

include(FindPackageHandleStandardArgs)
set(MKL_FIND_REQUIRED TRUE)   # reproduces the hard failure of the former find_package(MKL REQUIRED)
find_package_handle_standard_args(MKL
    REQUIRED_VARS MKL_INCLUDE_DIR MKL_INTERFACE_LIB MKL_THREADING_LIB MKL_CORE_LIB MKL_IOMP5_LIB)

if(MKL_FOUND AND NOT TARGET MKL::MKL)
    set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
    set(MKL_LIBRARIES ${MKL_INTERFACE_LIB} ${MKL_THREADING_LIB} ${MKL_CORE_LIB} ${MKL_IOMP5_LIB})

    add_library(MKL::MKL INTERFACE IMPORTED)
    set_target_properties(MKL::MKL PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${MKL_INTERFACE_LIB};${MKL_THREADING_LIB};${MKL_CORE_LIB};${MKL_IOMP5_LIB};pthread;dl;m")
endif()

mark_as_advanced(MKL_INCLUDE_DIR MKL_INTERFACE_LIB MKL_THREADING_LIB MKL_CORE_LIB MKL_IOMP5_LIB)

# ---------------------------------------------------------------------------
# Compile flags.
# ---------------------------------------------------------------------------
add_compile_options(-Wall -Wextra)
if(QUAGO_NATIVE_BUILD)
    add_compile_options(-march=native -mtune=native)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -ftree-vectorize")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")

add_subdirectory(src)
add_subdirectory(apps)
