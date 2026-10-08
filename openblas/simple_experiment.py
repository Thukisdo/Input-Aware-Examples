#!/usr/bin/env python3

# We import the search space here to ensure that the ranges are computed 
# at the beginning of the experiment, before any other code is executed.
import scripts.searchspace
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")

from scripts.grid_search import grid_search
from scripts.explore_vecsize import do_gs_exhaustive_on_vecsize, do_optuna_study_on_vecsize
import logging
import time

logger = None



def main(args):
    start = time.time()
    logger.info("Starting the experiment.")
    do_optuna_study_on_vecsize(args)
    do_gs_exhaustive_on_vecsize(args)
    grid_search(args)

    end = time.time()
    logger.info(f"Experiment completed in {end - start:.2f} seconds.")


def setup_logger(args):
    global logger
    logging.basicConfig(
        level=logging.INFO,
        format="{%(asctime)s} | [%(levelname)s]: %(message)s",
        handlers=[
            logging.FileHandler(args.output / "experiment.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run a simple experiment for OpenBLAS DGER kernel.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Directory to save the results.",
        required=True,
    )

    parser.add_argument("-n", "--nvec-grid", type=int, default=32, help="Number of points in the vector size grid.")
    parser.add_argument("-nt", "--nthreads-grid", type=int, default=32, help="Number of points in the thread count grid.")
    parser.add_argument("-no", "--nsample-optuna", type=int, default=100, help="Number of samples to evaluate during the Optuna search.")
    parser.add_argument("-f", "--force", action="store_true", help="Force re-run of the experiment even if results exist.")
    parser.add_argument("-t", "--target-vecsize", type=int, default=512, help="Target vector size for the experiment.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    setup_logger(args)
    with mpl.rc_context({"font.size": 8}):
        main(args)

