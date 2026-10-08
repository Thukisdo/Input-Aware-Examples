from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns

from .searchspace import n_range, nthreads_range
from .wrapper import execute_openblas
import tqdm
import logging

_logger = logging.getLogger(__name__)


def reload_results(output_path, df):

    if not output_path.exists():
        return None, df

    existing_results = pd.read_csv(output_path)

    # We merge the existing results with the new grid to avoid re-running configurations that have already been evaluated.
    df = df.merge(
        existing_results[["vecsize", "nthreads"]],
        on=["vecsize", "nthreads"],
        how="left",
        indicator=True,
    )
    df = df[df["_merge"] == "left_only"].drop(columns="_merge").reset_index(drop=True)

    # The first return value is the existing results,
    # the second is the remaining configurations to evaluate (i.e., those that are not already in the existing results).
    return existing_results, df


def _safe_dump_results(output_path, tmp_path, df, new_measures):
    # No need to dump if there are no new measures
    if not new_measures:
        return df, []

    # Concat existing results with new results and save to temporary file
    # Note that we are fully rewriting the file each time
    # But for the low number of measures we have, this shouldn't be a problem
    tmp = pd.DataFrame(new_measures)
    tmp = pd.concat([df, tmp], ignore_index=True) if df is not None else tmp
    tmp.to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)

    return tmp, []


def _do_grid_search(args, output_path):
    # Tmp place to ensure atomic replace
    # This avoid partial writes in case of interruption, which would corrupt the results file.
    tmp_path = output_path.with_suffix(".tmp")

    _logger.info(
        f"Starting input-aware grid search with {args.nvec_grid} vector sizes and {args.nthreads_grid} thread counts..."
    )

    # Build a 2d regular grid over n * nthreads
    Gn = np.linspace(n_range[0], n_range[1], args.nvec_grid, dtype=int)
    Gt = np.linspace(nthreads_range[0], nthreads_range[1], args.nthreads_grid, dtype=int)
    df = pd.DataFrame(np.array(np.meshgrid(Gn, Gt)).T.reshape(-1, 2), columns=["vecsize", "nthreads"])
    df.drop_duplicates(inplace=True)

    res, df = reload_results(output_path, df)

    measures = []
    i = 0

    try:
        for row in tqdm.tqdm(df.to_dict(orient="records"), total=len(df), desc="Grid Search"):
            result = execute_openblas(row["vecsize"], row["nthreads"])

            row["time"] = result
            measures.append(row)

            i += 1
            # Dump intermediate results every 200 runs to avoid losing data in case of interruption.
            # 200 might seems high, but the runs can be very small and fast
            if i % 200 == 0:
                res, measures = _safe_dump_results(output_path, tmp_path, res, measures)
    finally:
        res, _ = _safe_dump_results(output_path, tmp_path, res, measures)

    return res


def _plot(args, df_gs):

    # At maximum, we take ~16 different unique vecsize
    # unique_vecsizes = np.unique(df_gs["vecsize"])
    # vecsizes_to_plot = unique_vecsizes[:: max(1, len(unique_vecsizes) // 16)]
    # df_gs = df_gs[df_gs["vecsize"].isin(vecsizes_to_plot)]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), layout="constrained")

    sns.lineplot(
        data=df_gs,
        x="nthreads",
        y="time",
        hue="vecsize",
        ax=ax,
        palette="viridis",
        legend=False,
    )

    ax.set_yscale("log")

    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Execution Time (s)")

    scatter = []

    for _, df in df_gs.groupby("vecsize"):

        idxmin = df["time"].idxmin()
        row = df.loc[idxmin]
        scatter.append(row)

    scatter = pd.DataFrame(scatter)
    sns.scatterplot(
        data=scatter,
        x="nthreads",
        y="time",
        color="red",
        ax=ax,
        legend=False,
        s=30,
        edgecolor="black",
    )

    fig.savefig(args.output / "grid_search.png", bbox_inches="tight", pad_inches=0.1, dpi=300)


def grid_search(args):
    output_path = args.output / "grid_search_multiple.csv"

    if args.force and output_path.exists():
        output_path.unlink()

    samples = _do_grid_search(args, output_path)

    _plot(args, samples)
