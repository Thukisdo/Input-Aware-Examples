#!/usr/bin/env python3
import pandas as pd
import optuna
from .searchspace import nthreads_range
from .wrapper import execute_openblas
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

_logger = logging.getLogger(__name__)


def objective(n, trial):
    nthreads = trial.suggest_int("nthreads", *nthreads_range)
    return execute_openblas(n, nthreads)


def _run_optuna_on_vecsize(args):
    output_path = args.output / f"optuna_{args.target_vecsize}.csv"
    target_n = args.target_vecsize

    if not args.force and output_path.exists():
        _logger.info(f"Optuna results already exist at `{output_path}`. Reloading...")
        res = pd.read_csv(output_path)
        return res

    # Bind the objective function with the target vector size
    objective_func = lambda trial: objective(target_n, trial)
    n_trials = args.nsample_optuna


    _logger.info(f"Starting Optuna study with {n_trials} trials...")

    study = optuna.create_study(direction="minimize")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective_func, n_trials=n_trials)

    _logger.info(f"Best configuration: {study.best_trial.params}\n" f"Best performance: {study.best_trial.value}")

    # Recreate a DataFrame from Optuna's results
    # Optuna exposes a .to_dataframe() method
    # But we find it has poor formatting and prefer to create our own DataFrame
    # for better control
    res = []
    for trial in study.trials:
        _logger.info(f"Trial {trial.number}: Params: {trial.params}, Performance: {trial.value}")
        res.append({**trial.params, "time": trial.value})

    res = pd.DataFrame(res)
    res.to_csv(output_path, index=False)

    return res


def do_optuna_study_on_vecsize(args):
    df_o = _run_optuna_on_vecsize(args)

    df_o["time"] *= 1000  # Convert to milliseconds

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")

    sns.scatterplot(data=df_o, x="nthreads", y="time", ax=ax, s=10, alpha=0.5)
    ax.set_title(rf"Optuna Study - DGER $n = {args.target_vecsize}$")

    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Execution Time (ms)")

    fig.savefig(args.output / "optuna_study.png", dpi=300, bbox_inches="tight", pad_inches=0.05)


def _evaluate_df(df, nmeta=1):

    if nmeta > 1:
        df = pd.concat([df] * nmeta, ignore_index=True)
        df.sort_values(by=["n", "nthreads"], inplace=True)

    results = []
    for row in df.itertuples(index=False):
        performance = execute_openblas(row.n, row.nthreads)
        if performance is not None:
            results.append(performance)

    df["time"] = results
    return df


def do_grid_on_vecsize(args):
    output_path = args.output / f"grid_search_results_{args.target_vecsize}.csv"
    target_n = args.target_vecsize

    if not args.force and output_path.exists():
        _logger.info(f"Grid search results already exist at `{output_path}`. Reloading...")
        res = pd.read_csv(output_path)
        return res

    _logger.info(f"Starting grid search on vector size {target_n}...")

    targets = np.arange(nthreads_range[0], nthreads_range[1] + 1, step=3)
    targets = pd.DataFrame(targets, columns=["nthreads"])
    targets["n"] = target_n

    res = _evaluate_df(targets, nmeta=10)
    res.to_csv(output_path, index=False)

    return res


def do_exhaustive_on_vecsize(args):
    output_path = args.output / f"exhaustive_search_results_{args.target_vecsize}.csv"
    target_n = args.target_vecsize

    if not args.force and output_path.exists():
        _logger.info(f"Exhaustive search results already exist at `{output_path}`. Reloading...")
        res = pd.read_csv(output_path)
        return res

    _logger.info(f"Starting exhaustive search on vector size {target_n}...")

    targets = np.arange(nthreads_range[0], nthreads_range[1] + 1)
    targets = pd.DataFrame(targets, columns=["nthreads"])
    targets["n"] = target_n

    res = _evaluate_df(targets, nmeta=10)
    res.to_csv(output_path, index=False)

    return res


def do_gs_exhaustive_on_vecsize(args):

    df_ex = do_exhaustive_on_vecsize(args)
    df_gs = do_grid_on_vecsize(args)

    df_ex["time"] *= 1000  # Convert to milliseconds
    df_gs["time"] *= 1000  # Convert to milliseconds

    fig, axs = plt.subplots(1, 2, figsize=(5.75, 2.5), layout="constrained")

    fig.suptitle(rf"DGER $n = {args.target_vecsize}$")

    ax = axs[0]
    sns.lineplot(data=df_ex, x="nthreads", y="time", ax=ax, errorbar="sd")

    ax.set_title("Exhaustive Search")
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Execution Time (ms)")

    ax.margins(x=0)

    min_row = df_ex.loc[df_ex["time"].idxmin()]
    min_nthreads = min_row["nthreads"]
    ax.axvline(min_nthreads, color="red", linestyle="--", label=f"Best: {min_nthreads} threads")

    ax = axs[1]

    sns.lineplot(data=df_gs, x="nthreads", y="time", ax=ax, errorbar="sd", color="purple")

    ax.set_title("Grid Search")
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Execution Time (ms)")
    ax.margins(x=0)

    min_row = df_gs.loc[df_gs["time"].idxmin()]
    min_nthreads = min_row["nthreads"]
    ax.axvline(min_nthreads, color="red", linestyle="--", label=f"Best: {min_nthreads} threads")

    fig.savefig(args.output / "search_comparison.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
