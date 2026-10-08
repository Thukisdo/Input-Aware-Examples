#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "Usage: $0 <output_dir> <nruns>"
    exit 1
fi

output_dir="$1"
nruns="$2"

mkdir -p "$output_dir"


export OMP_PLACES=cores
export OMP_PROC_BIND=close

lscpu > "$output_dir/lscpu.txt"
uname -a > "$output_dir/uname.txt"
date > "$output_dir/timestamp.txt"

{
    ./simple_experiment.py -o "$output_dir" -n "$nruns"
} | tee "$output_dir/run.log"