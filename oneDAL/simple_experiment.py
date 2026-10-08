#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

import kernel
from scripts.searchspace import bsz_range, n_range, ndim_range, nthreads
from scripts.optuna_search import optuna_multi_search
from scripts.grid_search import grid_search_block_size

def plot(args, df_optuna: pd.DataFrame, optuna_bounds: list, df_gs: pd.DataFrame, gs_config: dict):
    # Plot size
    width = 5.75
    height = 5.75

    vmin = df_optuna["ratio"].min()
    vmax = df_optuna["ratio"].max()
    vcenter = 1.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    fig, axs = plt.subplots(2, 1, figsize=(width, height), height_ratios=[0.7, 0.3], gridspec_kw={"hspace": 0.25})

    ax = axs[0]
    ax.hexbin(
        x=df_optuna["n"],
        y=df_optuna["ndim"],
        C=df_optuna["ratio"],
        gridsize=32,
        cmap="RdBu",
        edgecolors="none",
        reduce_C_function=np.max,
        linewidths=0,
        rasterized=True,
        norm=norm,
    )

    sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
    ticks = np.unique(np.concatenate([np.linspace(vmin, vcenter, 5), np.linspace(vcenter, vmax, 5)]))
    cbar = fig.colorbar(sm, ax=ax, label="Speedup (default / new batch size)", orientation="horizontal", aspect=40, pad=0.2)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    ax.set_xlabel("Number of samples (n)")
    ax.set_ylabel("Number of features (ndim)")
    ax.set_title("a) Speedup ratio of new batch size vs default batch size")

    for bound in optuna_bounds:
        nlo, nhi, ndimlo, ndimhi = bound
        rect = Rectangle(
            (nlo, ndimlo), nhi - nlo, ndimhi - ndimlo, linewidth=1, edgecolor="black", facecolor="none", alpha=0.2
        )
        ax.add_patch(rect)

    ax.set_xlim(n_range[0], n_range[1])
    ax.set_ylim(ndim_range[0], ndim_range[1])

    ax = axs[1]

    df_gs["value"] /= 1e6  # Convert to milliseconds

    sns.lineplot(data=df_gs, x="batch_size", y="value", ax=ax, errorbar="sd", estimator="mean", color="gray")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Fit time (ms)")
    ax.set_title(f"b) n={gs_config['n']}, ndim={gs_config['ndim']}")

    ax.axvline(gs_config["batch_size"], color="red", linestyle="--", label="Optuna best batch size")
    ax.axvline(kernel.get_default_blocksize(), color="blue", linestyle="--", label="Default batch size")
    ax.legend(bbox_to_anchor=(0.5, -0.4), loc="upper center", ncol=2)
    ax.margins(x=0)

    fig.savefig(args.output / "speedup_ratio_plot.pdf", bbox_inches="tight", dpi=300, pad_inches=0.1)
    fig.savefig(args.output / "speedup_ratio_plot.png", bbox_inches="tight", dpi=300, pad_inches=0.1)
    plt.close(fig)


def main(args):
    df_optuna, optuna_bounds = optuna_multi_search(args, n_trials=args.nruns)
    df_gs, gs_config = grid_search_block_size(args, df_optuna)
    plot(args, df_optuna, optuna_bounds, df_gs, gs_config)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run Optuna search for optimal batch size.")
    parser.add_argument("-o", "--output", type=Path, help="Output directory for results.")
    parser.add_argument("-n", "--nruns", type=int, default=100, help="Number of trials for the Optuna search.")
    parser.add_argument("--force", action="store_true", help="Force re-run even if results exist.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with mpl.rc_context({"font.size": 8}):
        main(args)
