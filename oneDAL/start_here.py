#!/usr/bin/env python3
# Each parameter is defined as a range [low, high] of admissible values
from scripts.searchspace import bsz_range, n_range, ndim_range

import pandas as pd
import numpy as np

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tqdm

# Make sure progress bars work well with logging
from tqdm.contrib.logging import logging_redirect_tqdm

import kernel
import logging

from pathlib import Path

_logger = None

def run_df(df: pd.DataFrame) -> pd.DataFrame:
    # Helper function if you want to execute a DataFrame
    # of configurations (vecsize, nthreads) and get the results back in a DataFrame.
    res = []
    for _, row in df.iterrows():
        result = kernel.onedal_linreg_harness(
            batch_size=row["bsz"],
            n=row["n"],
            ndim=row["ndim"],
            nthreads=row["nthreads"],
        )
        res.append(result)
    res = pd.DataFrame(res)
    res = pd.concat([df, res], axis=1, ignore_index=True)
    return res


def main(args):
    bsz = 2500
    n = 10000
    ndim = 20
    nthreads = 8

    # The search space is defined in scripts/searchspace.py as ranges [low, high] for bsz, n, ndim.
    _logger.info(f"Admissible range for bsz: {bsz_range}")
    _logger.info(f"Admissible range for n: {n_range}")
    _logger.info(f"Admissible range for ndim: {ndim_range}")

    # Run the kernel with meta-repetitions
    res = kernel.onedal_linreg_harness(batch_size=bsz, n=n, ndim=ndim, nthreads=nthreads)
    # All values are returned in nanoseconds
    _logger.info(f"Result: {res}ns")

    # Code your own search space exploration here !
    # For example, you could perform a grid search over n and nthreads, use Optuna, ...
    # Enjoy !


def setup_logger(args):
    global _logger
    logging.basicConfig(
        level=logging.INFO,
        format="{%(asctime)s} | [%(levelname)s]: %(message)s",
        handlers=[logging.FileHandler(args.output / "experiment.log"), logging.StreamHandler()],
        force=True,
    )
    _logger = logging.getLogger(__name__)

    # Reduce sklearn logging level to WARNING to avoid cluttering the output
    logging.getLogger("sklearnex").setLevel(logging.WARNING)


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
