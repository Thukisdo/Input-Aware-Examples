#!/usr/bin/env python3
# This file is a slight rework of Anatoly Volkov's original oneDAL kernel harness
# Which was used internally at intel to experiment with input-awareness.

import numpy as np
import time

# Import the Intel oneDAL LinearRegression kernel
# We are tuning the batch size hyperparameter for this kernel
# Which we can access through the cpu_macro_block hyperparameter of the fit method.
# This is a class-level hyperparameter, so we can set it once and it will be used for all subsequent calls to fit.
#
# LinearRegression.get_hyperparameters("fit").cpu_macro_block = ...
#
# The default value is 8,192 at the time of writing, which does not appear to be retrievable after the value has been changed
from sklearnex.linear_model import LinearRegression

# Default values for meta-repetitions.
# To ensure robust results, we perform the following:
# 1. We generate a random dataset of size n x ndim.
# 2. We fit the model on this dataset meta_fit times and record the timing results
# 3. We repeat steps 1 and 2 meta_regen times, generating a new random dataset each time.
# 4. We aggregate the timing results across all meta_regen
#
# Below are the default values for meta_regen and meta_fit that are used through the onedal_linreg_harness function.
# We purposefully set them relatively high to ensure robust results for our initial experiments.
# Use linreg_with_meta directly if you want to change these values.
DEFAULT_META_REGEN = 3
DEFAULT_META_FIT = 10

# Number of warmup fit to perform to stabilize the timing results.
NWARMUP = 1

_DEFAULT_BLOCKSIZE = LinearRegression.get_hyperparameters("fit").cpu_macro_block


# Helper function to generate random uniform data of size n rows x ndims features.
# The random seed can be set for reproducibility.
def _gen_random_uniform(n: int, ndim: int, seed: int = 123) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=seed)
    X = rng.random(size=(n, ndim))
    y = rng.random(size=n)
    return X, y


# Retrieves the current block size used by the Intel oneDAL LinearRegression kernel.
# This is a hyperparameter that can be tuned for performance.
def get_blocksize() -> int:
    return LinearRegression.get_hyperparameters("fit").cpu_macro_block


def get_default_blocksize() -> int:
    return _DEFAULT_BLOCKSIZE


# Run the Intel oneDAL LinearRegression with meta-repetitions.
def linreg_with_meta(
    meta_regen: int, meta_fit: int, n: int, ndim: int, batch_size: int, nthreads: int, same_seed: bool = True, seed: int = 42
) -> list[int]:
    """Execute the Intel oneDAL LinearRegression kernel with meta-repetitions and return the timing results.

    :param meta_regen: The number of datasets to run the kernel on.
        Each dataset will be run meta_fit times and the results will be aggregated.
        This is useful to avoid the effects of caching and other system-level optimizations that can affect the timing results.
    :type meta_regen: int
    :param meta_fit: The number of time to refit the model on the same dataset. (This is timed function)
    :type meta_fit: int
    :param n: The number of rows in the dataset used for training.
    :type n: int
    :param ndim: The number of features (columns) in the dataset used for training.
    :type ndim: int
    :param batch_size: The hyperparameter that controls the number of rows processed in a single batch during training.
        This is the main hyperparameter that we are tuning for performance.
    :type batch_size: int
    :param nthreads: The number of threads to use for training.
        For simplicity, this should be the number of physical cores on the machine.
    :type nthreads: int
    :param same_seed: Use the same seed to generate all datasets (meta_regen), defaults to True
    :type same_seed: bool, optional
    :param seed: The seed to use to generate the starting dataset
        If same_seed is True, this will be used for all datasets. If same_seed is False, this will be incremented for each dataset.
        defaults to 42
    :type seed: int, optional
    :return: A list of timing results in nanoseconds for each fit operation across all datasets and meta-repetitions.
    :rtype: list[int]
    """

    LinearRegression.get_hyperparameters("fit").cpu_macro_block = min(batch_size, n)

    # Warmup
    # We found a single iteration to be enough to ensure stable timing results on our setup
    X, y = _gen_random_uniform(n, ndim, seed)
    for _ in range(NWARMUP):
        LinearRegression(n_jobs=nthreads).fit(X, y)

    res = []
    for _ in range(meta_regen):
        X, y = _gen_random_uniform(n, ndim, seed)

        for _ in range(meta_fit):
            begin = time.perf_counter_ns()
            LinearRegression(n_jobs=nthreads).fit(X, y)
            end = time.perf_counter_ns()
            res.append(end - begin)

        if not same_seed:
            seed += 1

    return res


# Helper function to run the Intel oneDAL LinearRegression kernel with meta-repetitions and return statistics.
def onedal_linreg_harness(
    n: int, ndim: int, batch_size: int, nthreads: int, same_seed: bool = True, seed: int = 42
) -> dict[str, float]:
    """Run the Intel oneDAL LinearRegression kernel with the default meta-repetitions and return the median, min, max,
        and std of the timing results. To access the raw timing results, use linreg_with_meta directly.

    :param n: The number of rows in the dataset used for training.
    :type n: int
    :param ndim: The number of features (columns) in the dataset used for training.
    :type ndim: int
    :param batch_size: The hyperparameter that controls the number of rows processed in a single batch during training.
        This is the main hyperparameter that we are tuning for performance.
    :type batch_size: int
    :param nthreads: The number of threads to use for training.
        For simplicity, this should be the number of physical cores on the machine.
    :type nthreads: int
    :param same_seed: Whether to use the same seed for all datasets. If True, the seed will be used for all datasets. If False, the seed will be incremented for each dataset.
        defaults to True
    :type same_seed: bool, optional
    :param seed: The seed to use to generate the starting dataset.
        If same_seed is True, this will be used for all datasets. If same_seed is False, this will be incremented for each dataset.
        defaults to 42
    :type seed: int, optional
    :return: A dictionary containing the median, min, max, and std of the timing results in nanoseconds for each fit operation across all datasets and meta-repetitions.
    :rtype: dict[str, float]
    """

    res = linreg_with_meta(DEFAULT_META_REGEN, DEFAULT_META_FIT, n, ndim, batch_size, nthreads, same_seed=same_seed, seed=seed)

    median, rmin, rmax, std = np.median(res), np.min(res), np.max(res), np.std(res)

    return {
        "median": median,
        "min": rmin,
        "max": rmax,
        "std": std,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Execute the Intel oneDAL LinearRegression kernel with meta-repetitions and return statistics."
    )
    parser.add_argument("-n", type=int, default=10000, help="Number of rows in the dataset used for training.")
    parser.add_argument("-ndim", type=int, default=20, help="Number of features (columns) in the dataset used for training.")
    parser.add_argument(
        "-bsz",
        type=int,
        default=2500,
        help="Batch size hyperparameter that controls the number of rows processed in a single batch during training.",
    )
    parser.add_argument(
        "-nthreads",
        type=int,
        default=8,
        help="Number of threads to use for training. For simplicity, this should be the number of physical cores on the machine.",
    )

    args = parser.parse_args()

    res = onedal_linreg_harness(args.n, args.ndim, args.bsz, args.nthreads)
    print(f"Result: {res}")
