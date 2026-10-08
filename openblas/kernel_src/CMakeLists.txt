cmake_minimum_required(VERSION 3.16)

project(
    openblas_kernel
    LANGUAGES C CXX
)

find_package(OpenMP REQUIRED)

# Call the fetch_openblas.cmake script to download and configure OpenBLAS
include(extras/fetch_openblas.cmake)

add_executable(kernel
    main.cpp
)

target_link_libraries(kernel
    PRIVATE
        OpenBLAS::OpenBLAS
        OpenMP::OpenMP_CXX
)

option(OPENBLAS_KERNEL_NATIVE
    "Enable -march=native"
    ON
)

if(OPENBLAS_KERNEL_NATIVE)
    target_compile_options(kernel
        PRIVATE
            -march=native
    )
endif()