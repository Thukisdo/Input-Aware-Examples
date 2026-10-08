
# How to run:

First, setup all required Python packages:

```bash
source ./setup_env.sh
```

Then build the kernel:

```bash
./build_kernel.sh
```

You will need CMake and an OpenMP install of OpenBLAS. The CMakeFile we provide will perform automatic check to ensure
you have the correct OpenBLAS version, and we attempt multiple default install path for the library.

However, the exact library name can change depending on your distribution, so you might need to adapt the build system to your need.


You can then manually run the kernel:

```bash
OMP_PROC_BIND=close OMP_PLACES=cores ./build/kernel <vecsize> <nthreads>
```

# What to do next

You can either investigate the provided `scripts/` for inspiration, or start working in `start_here.py` which provides basic
functionality to implement your own search space exploration. You can use Optuna, grid search, or any other method to find the 
best configuration for the kernel. 

Good luck !