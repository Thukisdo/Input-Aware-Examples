import subprocess
from pathlib import Path
import os
import logging

_logger = logging.getLogger(__name__)

KERNEL_PATH = Path("build/kernel")

if not KERNEL_PATH.exists():
    raise FileNotFoundError(f"Kernel path {KERNEL_PATH} does not exist. Please build the kernel first." f"./build_kernel.sh")

error_count = 0


def execute_openblas(vecsize: int, nthreads: int):
    """
    Execute the OpenBLAS kernel with the given parameters.

    Args:
        vecsize (int): The vector size to use.
        nthreads (int): The number of threads to use.

    Returns:
        float: The execution time in seconds.
    """
    global error_count

    command = [
        str(KERNEL_PATH),
        str(vecsize),
        str(nthreads),
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    if result.returncode != 0:
        error_count += 1
        _logger.error(
            f"Kernel execution failed for vecsize={vecsize}, nthreads={nthreads}. Return code: {result.returncode}. Output:\n{result.stdout}"
        )
        return None

    # Assuming the kernel prints the execution time in seconds
    try:
        execution_time = float(result.stdout.strip())
    except ValueError:
        error_count += 1
        _logger.error(
            f"Failed to parse execution time from kernel output for vecsize={vecsize}, nthreads={nthreads}. Output:\n{result.stdout}"
        )
        return None

    return execution_time
