#!/bin/bash
set -e

if [ -z $STITCH_DIR ]
then
    echo "[claim-3.sh] Please set environment variable \$STITCH_DIR to the stitch/ directory"
    exit 1
fi

if [ -z $1 ]
then
    OUT_DIR="out/claim-2/$(TZ='America/New_York' date '+%Y-%m-%d_%H-%M-%S')"
else
    OUT_DIR=$1
fi

if [[ $OSTYPE == 'darwin'* ]]
then
    GTIME="gtime"
else
    GTIME="/usr/bin/time"
fi

mkdir -p $OUT_DIR

# Save some info about what state of the repo this experiment was run in
# to aid reporducibility
echo -n "Current git commit: " > $OUT_DIR/readme.md
git log -n 1 >> $OUT_DIR/readme.md
echo -n "Current git branch: " >> $OUT_DIR/readme.md
git rev-parse --abbrev-ref HEAD >> $OUT_DIR/readme.md
STITCH_FLAGS="--verbose-best --fmt=programs-list --max-arity=3 --iterations=1"
echo -n "Stitch hyperparameter settings:" >> $OUT_DIR/readme.md
echo $STITCH_FLAGS >> $OUT_DIR/readme.md

# compile Stitch
pushd $STITCH_DIR
cargo build --release --bin=compress
popd


for WL_PATH in $STITCH_DIR/data/cogsci/*.json; do
    WL=$(basename -s .json $WL_PATH)
    OUTF=$OUT_DIR/$WL/
    mkdir -p $OUTF
    echo "[claim-3.sh] Starting workload $WL with stderr: $OUTF/1.stderrandout"
    echo "$STITCH_DIR/target/release/compress $WL_PATH $STITCH_FLAGS"
    $GTIME -v $STITCH_DIR/target/release/compress $WL_PATH $STITCH_FLAGS --out=$OUTF/1.json > $OUTF/1.stderrandout 2>&1
done


echo "Done: $OUT_DIR"