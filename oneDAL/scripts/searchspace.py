
import psutil

# n dictates the number of samples in the training dataset
n_range = [1000, 1e6]
# ndim dictates the number of features in the training dataset
ndim_range = [2, 25]

# Design parameter, we will search for the optimal batch size in this range
bsz_range = [100, 20_000]

# Retrieve CPU affinity
nthreads = len(psutil.Process().cpu_affinity())