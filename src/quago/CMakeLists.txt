# quago: the IndexQuago graph index and the QUAGO region-enhancement methods.
add_library(quago STATIC
    index_quago.cpp
    index_quago_enhance.cpp
    quago_enhancement.cpp
)
add_library(quago::quago ALIAS quago)

target_link_libraries(quago PUBLIC
    quago_core
    OpenMP::OpenMP_CXX
)
