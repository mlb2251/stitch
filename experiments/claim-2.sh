#!/bin/bash
set -e
#ulimit -v 50000000  # attempt to limit children to 50GB=50,000,000KB of virtual memory

if [ -z $STITCH_DIR ]
then
    echo "[claim-2.sh] Please set environment variable \$STITCH_DIR to the stitch/ directory"
    exit 1
fi

if [ -z $1 ]
then
    echo "[claim-2.sh] Usage: ./claim-2.sh NUM_REPETITIONS [OUTDIR]"
    exit 1
fi
SEEDS=$1

if [ -z $2 ]
then
    OUT_DIR="claim-2/$(TZ='America/New_York' date '+%Y-%m-%d_%H-%M-%S')"
else
    OUT_DIR=$2
fi

mkdir -p $OUT_DIR

# Save some info about what state of the repo this experiment was run in
# to aid reporducibility
echo -n "Current git commit: " > $OUT_DIR/readme.md
git log -n 1 >> $OUT_DIR/readme.md
echo -n "Current git branch: " >> $OUT_DIR/readme.md
git rev-parse --abbrev-ref HEAD >> $OUT_DIR/readme.md
STITCH_FLAGS="--hole-choice=depth-first --fmt=split-programs-list --max-arity=3 --iterations=10"
echo -n "Stitch hyperparameter settings:" >> $OUT_DIR/readme.md
echo $STITCH_FLAGS >> $OUT_DIR/readme.md

# compile Stitch
pushd $STITCH_DIR
cargo build --release --bin=compress
popd


for WL_PATH in $STITCH_DIR/data/cogsci/*.json; do
    WL=$(basename -s .json $WL_PATH)
    mkdir -p $OUT_DIR/$WL
    for SEED in {1..$SEEDS} ; do
    echo "[claim-2.sh] Starting workload $WL, seed $SEED"
    python3 split_data.py $SEED $WL_PATH "$WL-$SEED-split.json"
    echo "[claim-2.sh] Split data; seed used was $SEED, train test % was 80%"
    echo "Running Stitch"
    /usr/bin/time -v $STITCH_DIR/target/release/compress "$WL-$SEED-split.json" $STITCH_FLAGS --out=$OUT_DIR/$WL/$SEED.json > $OUT_DIR/$WL/$SEED.stderrandout 2>&1
    done
    rm -v *-split.json
done


echo "Done: $OUT_DIR"
