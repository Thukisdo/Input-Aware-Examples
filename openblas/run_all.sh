#!/usr/bin/bash
set -eo pipefail

if [ -z "$1" ]; then
    echo "Usage: $0 <run_label>"
    exit 1
fi

RUN_LABEL=$1
OUTPUT_DIR="runs/$RUN_LABEL"

mkdir -p $OUTPUT_DIR

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_SCHEDULE=static
export OMP_WAIT_POLICY=active # For the very small size we are testing, we don't want threads to sleep and wake up
                              # as that would be a significant noise source            

{
    ./simple_experiment.py -o "$OUTPUT_DIR" -n 64 -nt 52 -no 100

} 2>&1 | tee -a $OUTPUT_DIR/log.txt
