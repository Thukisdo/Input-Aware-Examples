import pandas as pd
import tqdm
import numpy as np

from .searchspace import bsz_range, n_range, ndim_range, nthreads
import kernel


# Helper function to run the Intel oneDAL LinearRegression kernel
# with high meta-repetitions (10x10) for a given row of parameters (n, ndim, batch_size)
def run_kernel_10_10_metas(cdict):
    res = kernel.linreg_with_meta(
        10, 10, int(cdict["n"]), int(cdict["ndim"]), int(cdict["batch_size"]), nthreads, same_seed=False, seed=42
    )
    return res


def grid_search_block_size(args, samples: pd.DataFrame):

    outpath = args.output / "grid_search_block_size.csv"

    # First, we retrieve the input (n, ndim) with maximum speedup from the samples
    idxmax = samples["ratio"].idxmax()
    sample_max = samples.loc[idxmax].to_dict()

    if outpath.exists() and not args.force:
        print(f"Found existing results in {outpath}, loading...")
        return pd.read_csv(outpath), sample_max

    values = np.linspace(bsz_range[0], bsz_range[1], 128, dtype=int)

    df = pd.DataFrame({"batch_size": values})
    columns = ["n", "ndim"]
    for col in columns:
        df[col] = sample_max[col]

    res = []
    for _, row in tqdm.tqdm(
        df.iterrows(),
        total=df.shape[0],
        desc=f"Grid search (n={sample_max['n']}, ndim={sample_max['ndim']})",
    ):
        raw_measures = run_kernel_10_10_metas(row)

        tmp = pd.DataFrame({"value": raw_measures})
        for k, v in row.items():
            tmp[k] = v
        res.append(tmp)

    res = pd.concat(res, axis=0)
    res.to_csv(outpath, index=False)

    return res, sample_max
