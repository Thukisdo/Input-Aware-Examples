import psutil

# We want small problem sizes to have more variety in DT
n_range = [64, 6_000]

# Always use the process affinity rather than the number of physical / logical cores
# As the user may have set a CPU affinity for the process, which would limit the number of threads that can be used.
nthreads_range = [1, len(psutil.Process().cpu_affinity())]
