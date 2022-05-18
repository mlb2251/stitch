#!/bin/bash
set -e
#ulimit -v 50000000  # attempt to limit children to 50GB=50,000,000KB of virtual memory

if [ -z $STITCH_DIR ]
then
    echo "[cs2_ex3.sh] Please set environment variable \$STITCH_DIR to the stitch/ directory"
    exit 1
fi

OUT_DIR="cs2/ex3/$(TZ='America/New_York' date '+%Y-%m-%d_%H-%M-%S')"
mkdir -p $OUT_DIR

# Save some info about what state of the repo this experiment was run in
# to aid reporducibility
echo -n "Current git commit: " > $OUT_DIR/readme.md
git log -n 1 >> $OUT_DIR/readme.md
echo -n "Current git branch: " >> $OUT_DIR/readme.md
git rev-parse --abbrev-ref HEAD >> $OUT_DIR/readme.md

# compile Stitch
pushd $STITCH_DIR
cargo build --release --bin=compress
popd


for WL_PATH in $STITCH_DIR/data/cogsci/*.json; do
    WL=$(basename -s .json $WL_PATH)
    OUTF=$OUT_DIR/$WL/
    mkdir -p $OUTF
    echo "[cs2_ex3.sh] Starting workload $WL"
    /usr/bin/time -v $STITCH_DIR/target/release/compress $WL_PATH --verbose-best --hole-choice=last --heap-choice=max-bound --fmt=programs-list --max-arity=3 --iterations=1 --no-mismatch-check --out=$OUTF/1.json > $OUTF/1.stderrandout 2>&1 &
done
wait  # move this up/down between loops to change how many jobs to run at once


echo "Done: $OUT_DIR"
