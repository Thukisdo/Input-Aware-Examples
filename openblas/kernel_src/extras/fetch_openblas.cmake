# Fedora installs OpenBLAS headers under /usr/include/openblas.
find_path(OPENBLAS_INCLUDE_DIR
    NAMES cblas.h
    PATH_SUFFIXES openblas
    REQUIRED
)

message(STATUS "Attempting to find libopenblaso.so in the system library paths...")
# On many systems, the OpenBLAS + OpenMP library is named libopenblaso.so. Try to find it first.
find_library(OPENBLAS_LIBRARY
    NAMES openblaso
)

# If we did not find libopenblaso.so, try to find libopenblas.so instead.
if (NOT OPENBLAS_LIBRARY)
    find_library(OPENBLAS_LIBRARY
        NAMES openblas
    )
    if (NOT OPENBLAS_LIBRARY)
        message(FATAL_ERROR "Could not find OpenBLAS library. Please install OpenBLAS or specify the path to the library.")
    endif()
endif()


add_library(OpenBLAS::OpenBLAS UNKNOWN IMPORTED)

set_target_properties(OpenBLAS::OpenBLAS PROPERTIES
    IMPORTED_LOCATION "${OPENBLAS_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${OPENBLAS_INCLUDE_DIR}"
)

set(OPENBLAS_CHECK_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/openblas_check.c")

file(WRITE "${OPENBLAS_CHECK_SOURCE}" [=[
#include <stdio.h>
#include <cblas.h>

int main(void)
{
    int model = openblas_get_parallel();

    printf("OpenBLAS parallel model: %d\n", model);

    if (model != OPENBLAS_OPENMP)
        return 1;

    return 0;
}
]=]
)

try_run(
    OPENBLAS_CHECK_RUN_RESULT
    OPENBLAS_CHECK_COMPILE_RESULT
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${OPENBLAS_CHECK_SOURCE}"
    LINK_LIBRARIES
        OpenBLAS::OpenBLAS
    CMAKE_FLAGS
        "-DINCLUDE_DIRECTORIES=${OPENBLAS_INCLUDE_DIR}"
)

if(NOT OPENBLAS_CHECK_COMPILE_RESULT)
    message(FATAL_ERROR
        "Could not compile the OpenBLAS backend check."
    )
endif()

if(NOT OPENBLAS_CHECK_RUN_RESULT EQUAL 0)
    message(FATAL_ERROR
        "The selected OpenBLAS library is not using the OpenMP backend.\n"
        "Library: ${OPENBLAS_LIBRARY}"
    )
endif()

message(STATUS "OpenBLAS backend: OpenMP")
