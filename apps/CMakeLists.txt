# CLI drivers. Binaries land in build/apps/ so the experiment scripts can find them.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/apps)

function(add_quago_app target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE
        quago
        Boost::program_options
        OpenMP::OpenMP_CXX
    )
endfunction()

add_quago_app(quago_build   quago_build.cpp)
add_quago_app(quago_search  quago_search.cpp)
add_quago_app(quago_search_fast      quago_search_fast.cpp)
add_quago_app(file_io_check tools/file_io_check.cpp)
add_quago_app(kmeans_run    tools/kmeans_run.cpp)
