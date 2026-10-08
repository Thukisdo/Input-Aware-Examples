#!/usr/bin/env python3

# We import the search space here to ensure that the ranges are computed 
# at the beginning of the experiment, before any other code is executed.
from scripts.searchspace import n_range, nthreads_range
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns

from scripts.wrapper import execute_openblas
import tqdm

# Make sure progress bars work well with logging
from tqdm.contrib.logging import logging_redirect_tqdm
import os
import time
import logging

_logger = None


def _check_environment():
    env = os.environ
    warn = False
    warn_str = "== ENVIRONMENT WARNINGS ==\n"

    if "OMP_PROC_BIND" not in env:
        warn_str += "- OMP_PROC_BIND is not set. Use 'close' (or 'spread')\n"
        warn = True

    if "OMP_PLACES" not in env:
        warn_str += "- OMP_PLACES is not set. Use 'cores' (or 'sockets')\n"
        warn = True

    if warn:
        # Trust me, this is a good idea.
        # People have thanked me for this in the past.
        warn_str += "\nPlease set these environment variables for optimal performance. Pausing for 10 seconds...\n\n"
        _logger.warning(warn_str)
        time.sleep(10)


def _do_preflight_checks(args):
    _check_environment()
    # Add your checks here


def run_df(df: pd.DataFrame) -> pd.DataFrame:
    # Helper function if you want to execute a DataFrame
    # of configurations (vecsize, nthreads) and get the results back in a DataFrame.
    if not {"vecsize", "nthreads"}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'vecsize' and 'nthreads' columns.")

    res = []
    for _, row in df.iterrows():
        result = execute_openblas(row["vecsize"], row["nthreads"])
        res.append(result)

    res = pd.DataFrame(res, columns=["time"])
    res = pd.concat([df, res], axis=1, ignore_index=True)

    return res


def main(args):
    _do_preflight_checks(args)
    # Size of x, y, A (n x n) matrices
    n = 10000

    # Number of openblas/openMP threads to use
    nthreads = 8

    # The search space is defined in scripts/searchspace.py as ranges [low, high] for n and nthreads.
    _logger.info(f"Admissible range for n: {n_range}")
    _logger.info(f"Admissible range for nthreads: {nthreads_range}")

    # Run the kernel with meta-repetitions
    # Returns the minimum execution time in seconds
    res = execute_openblas(nthreads=nthreads, vecsize=n)

    print(f"Result: n={n} nthreads={nthreads} time={res:.6f}s")

    # Code your own search space exploration here !
    # For example, you could perform a grid search over n and nthreads, use Optuna, ...
    # Enjoy !


def setup_logger(args):
    global _logger
    logging.basicConfig(
        level=logging.INFO,
        format="{%(asctime)s} | [%(levelname)s]: %(message)s",
        handlers=[logging.FileHandler(args.output / "experiment.log"), logging.StreamHandler()],
    )
    _logger = logging.getLogger(__name__)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run a simple experiment for oneDAL DGER kernel.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Directory to save the results.",
        required=True,
    )

    # Add arguments here

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    return args


if __name__ == "__main__":
    args = parse_args()
    setup_logger(args)
    with logging_redirect_tqdm():
        main(args)
