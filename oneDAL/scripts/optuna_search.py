
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
import tqdm
optuna.logging.set_verbosity(optuna.logging.WARNING)

import kernel
from .searchspace import bsz_range, n_range, ndim_range, nthreads

# Constants for grid search, we run one Optuna study per grid cell
GS_N = 4
GS_NDIM = 4

def optuna_objective(nrange: list[int], ndimrange: list[int], trial: optuna.Trial):
    n = trial.suggest_int("n", nrange[0], nrange[1])
    ndim = trial.suggest_int("ndim", ndimrange[0], ndimrange[1])
    bsz = trial.suggest_int("batch_size", bsz_range[0], bsz_range[1])

    new = kernel.onedal_linreg_harness(
        n=n,
        ndim=ndim,
        batch_size=bsz,
        nthreads=nthreads,
    )

    default = kernel.onedal_linreg_harness(
        n=n,
        ndim=ndim,
        batch_size=kernel.get_default_blocksize(),
        nthreads=nthreads,
    )

    # We also want to store the statistics of the new and baseline runs in the trial's user attributes for later analysis
    trial.set_user_attr("median", new["median"])
    trial.set_user_attr("min", new["min"])
    trial.set_user_attr("max", new["max"])
    trial.set_user_attr("std", new["std"])

    trial.set_user_attr("def_median", default["median"])
    trial.set_user_attr("def_min", default["min"])
    trial.set_user_attr("def_max", default["max"])
    trial.set_user_attr("def_std", default["std"])

    ratio = default["median"] / new["median"]
    return ratio


def optuna_search(args, nrange: list[int], ndimrange: list[int], id: str, n_trials: int):

    outpath = args.output / f"optuna_partial/{id}.csv"
    outpath.parent.mkdir(parents=True, exist_ok=True)

    if outpath.exists() and not args.force:
        print(f"Found existing results in {outpath}, loading...")
        return pd.read_csv(outpath)

    # Increase the number of startup trials to ensure a better exploration of the search space
    nbootstrap = int(0.2 * n_trials)
    sampler = TPESampler(n_startup_trials=nbootstrap)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(lambda trial: optuna_objective(nrange, ndimrange, trial), n_trials=n_trials, show_progress_bar=True)

    # We aggregate all samples into a single DataFrame for easier analysis and visualization
    # While Optuna provide a trials_dataframe() method, it has extraneous rows
    res = []
    for s in study.trials:
        res.append({**s.params, **s.user_attrs, "ratio": s.value})
    res = pd.DataFrame(res)

    # Remove the "params_" and "user_attrs_" prefixes from the column names
    res.columns = [col.replace("params_", "").replace("user_attrs_", "") for col in res.columns]

    res.to_csv(outpath, index=False)

    return res

def generate_bounds(gs_n: int, gs_dim: int):
    n_bounds = np.linspace(n_range[0], n_range[1], gs_n + 1, dtype=int)
    ndim_bounds = np.linspace(ndim_range[0], ndim_range[1], gs_dim + 1, dtype=int)

    bounds = []
    for i in range(gs_n):
        for j in range(gs_dim):
            bounds.append((n_bounds[i], n_bounds[i + 1], ndim_bounds[j], ndim_bounds[j + 1]))

    return bounds


def optuna_multi_search(args, n_trials=200):

    outpath = args.output / "optuna_merged.csv"
    opt_grid = generate_bounds(gs_n=GS_N, gs_dim=GS_NDIM)

    if outpath.exists() and not args.force:
        print(f"Found existing results in {outpath}, reloading...")
        return pd.read_csv(outpath), opt_grid

    res = []
    for bounds in tqdm.tqdm(opt_grid, desc="Optuna Multi-Search"):
        nlo, nhi, ndimlo, ndimhi = bounds

        tmp_nrange = [nlo, nhi]
        tmp_ndimrange = [ndimlo, ndimhi]

        res.append(
            optuna_search(
                args,
                tmp_nrange,
                tmp_ndimrange,
                id=f"{nlo}_{nhi}_{ndimlo}_{ndimhi}",
                n_trials=n_trials,
            )
        )

    res = pd.concat(res, ignore_index=True)
    res.to_csv(outpath, index=False)

    return res, opt_grid