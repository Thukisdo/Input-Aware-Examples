
# Input Aware examples

This repository contains two simple examples of input-aware kernels. We provide both benchmarking harness, 
a few example exploration scripts, and a starting point for curious readers.

- `oneDAL`: Batch Size tuning for LinearRegression of the Intel oneDAL library. The library uses a default configuration of 8,192.
    You can attempt to find a better input-aware configuration.

    The proposed scripts use optuna to identify inputs (n, ndim) where we can maximize the speedup compared to the default config.

- `openblas`: Tuning of the number of threads for OpenBLAS + OpenMP backend `DGER` kernel (rank-one matrix update).

    The proposed scripts perform grid/exhaustive search over the number of threads, for n = 512, but also for a grid of 64 < n < 6000.

For all kernels, we provide a `setup_env.sh` script **that must be sourced** to setup all required Python packages.
For the OpenBLAS kernel, you will need an OpenBLAS installation compiled with the OpenMP blackend, often listed as `openblaso` 
or `openblas-openmp-devel` in your favorite package manager.

You will also need cmake.
