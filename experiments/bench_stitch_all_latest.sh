#!/bin/bash
set -e

if [ -z $1 ]
then
    echo "Usage: ./bench_stitch_all_latest.sh benches/"
    exit 1
fi

if [ -z $STITCH_DIR ]
then
    echo "[bench_stitch_all_latest.sh] Please set environment variable \$STITCH_DIR to the stitch/ directory"
    exit 1
fi

BENCHES=$1

for COMPARE_TO in $(python3 analyze.py latest dreamcoder $BENCHES); do
    ./bench_stitch.sh compare $COMPARE_TO
done
