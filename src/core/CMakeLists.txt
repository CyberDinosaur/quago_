# quago_core: shared infrastructure (Index base, distance functors, parameters,
# neighbor queue, file IO, and the MKL-backed kmeans / recall utilities).
add_library(quago_core STATIC
    index.cpp
    math_utils.cpp
)
add_library(quago::core ALIAS quago_core)

target_include_directories(quago_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(quago_core PUBLIC
    MKL::MKL
    OpenMP::OpenMP_CXX
)
