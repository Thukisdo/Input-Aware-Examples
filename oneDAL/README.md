# How to run:

First, setup all required Python packages:

```bash
source ./setup_env.sh
```

You can then manually run the kernel:

```bash
# /python3 kernel.py -n 10000 -ndim 20 -bsz 8192 -nthreads 8
/python3 kernel.py -n <n> -ndim <ndim> -bsz <batch_size> -nthreads <number of threads>
```

Where:
- `<n>` is the number of samples in the training dataset.
- `<ndim>` is the number of features in the training dataset.
- `<batch_size>` is the batch size used for the kernel (default is 8192) at the time of writing.
- `<number of threads>` is the number of threads used for the kernel

Our experiments focus on the `batch_size` parameter, but you can also explore threading.

# What to do next

You can either investigate the provided `scripts/` for inspiration, or start working in `start_here.py` which provides basic
functionality to implement your own search space exploration. You can use Optuna, grid search, or any other method to find the 
best configuration for the kernel. 

Good luck !