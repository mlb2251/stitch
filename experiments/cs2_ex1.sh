#!/bin/bash
set -e
#ulimit -v 50000000  # attempt to limit children to 50GB=50,000,000KB of virtual memory

if [ -z $STITCH_DIR ]
then
    echo "[cs2_ex1.sh] Please set environment variable \$STITCH_DIR to the stitch/ directory"
    exit 1
fi

echo "WARNING: The results of this experiment are sensitive to noise, e.g. caused by other processes using up your CPU."

OUT_DIR="$STITCH_DIR/experiments/cs2/ex1/$(TZ='America/New_York' date '+%Y-%m-%d_%H-%M-%S')"
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
    # first chunk the dataset
    pushd $STITCH_DIR
    cargo run --bin=chunk-dataset -- -n 5 -o $OUT_DIR/$WL $WL_PATH
    popd

    # then do 50 stitch runs over each chunk
    for MODE in "length" "depth" ; do
    for CHUNK in {0..4} ; do
    for RUN in {1..10} ; do
    OUTF=$OUT_DIR/$WL/$MODE/runs/$RUN
    mkdir -p $OUTF
    $STITCH_DIR/target/release/compress $OUT_DIR/$WL/$MODE/$CHUNK.json --hole-choice=last --heap-choice=max-bound --fmt=programs-list --max-arity=3 --iterations=10 --out=$OUT_DIR/$WL/$MODE/$CHUNK-out.json | grep -F "Time: " >> $OUTF/times
    done
    done
    done
done


# The below is the old cs2-ex1, where we varied the arity
#for WL_PATH in $STITCH_DIR/data/cogsci/*.json; do
#    WL=$(basename -s .json $WL_PATH)
#    for ARITY in {1..10} ; do
#    for SEED in {1..50} ; do
#    OUTF=$OUT_DIR/$WL/$ARITY
#    mkdir -p $OUTF
#    echo "[case_study_2.sh] Starting workload $WL, arity $ARITY, seed $SEED"
#    /usr/bin/time -v $STITCH_DIR/target/release/compress $WL_PATH --hole-choice=last --heap-choice=max-bound --fmt=programs-list --max-arity=$ARITY --iterations=1 --no-mismatch-check --out=$OUTF/$SEED.json > $OUTF/$SEED.stderrandout 2>&1 &
#    done
#    wait  # move this up/down between loops to change how many jobs to run at once
#    done
#done

echo "Done: $OUT_DIR"
