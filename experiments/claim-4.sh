#!/bin/bash
set -e

if [ -z $STITCH_DIR ]
then
    echo "[claim-4.sh] Please set environment variable \$STITCH_DIR to the stitch/ directory"
    exit 1
fi

if [ -z $1 ]
then
    OUT_DIR="out/ablation-results/$(TZ='America/New_York' date '+%Y-%m-%d_%H-%M-%S')"
else
    OUT_DIR=$1
fi

if [[ $OSTYPE == 'darwin'* ]]
then
    GTIME="gtime"
    GTIMEOUT="gtimeout"
    echo "WARNING: On OS X `ulimit` doesn't work so you can't run these experiments with a memory limit. You could run them with a short time limit instead if desired using TIMEOUT=30m or something similar"
else
    GTIME="/usr/bin/time"
    GTIMEOUT="timeout"
fi

ulimit -v 50000000  # limit children to 50GB=50,000,000KB of virtual memory


if [ -z $NO_NO_OPT ]
then
    echo "[claim-4.sh] including --no-opt"
    NO_OPT="--no-opt"
else
    echo "[claim-4.sh] NOT including --no-opt"
    NO_OPT=""  
fi

if [ -z $TIMEOUT ]
then
   TIMEOUT="90m"
fi

echo "[claim-4.sh] using a timeout of $TIMEOUT"

# compile Stitch
pushd $STITCH_DIR
cargo build --release --bin=compress
popd

mkdir -p $OUT_DIR/raw
mkdir -p $OUT_DIR/stdout

# Save some info about what state of the repo this experiment was run in
# to aid reporducibility
echo -n "Current git commit: " > $OUT_DIR/readme.md
git log -n 1 >> $OUT_DIR/readme.md
echo -n "Current git branch: " >> $OUT_DIR/readme.md
git rev-parse --abbrev-ref HEAD >> $OUT_DIR/readme.md
STITCH_FLAGS="--max-arity=3 --iterations=1"
echo -n "Stitch hyperparameter settings: " >> $OUT_DIR/readme.md
echo $STITCH_FLAGS >> $OUT_DIR/readme.md

for WL_PATH in $STITCH_DIR/data/cogsci/*.json; do
    WL=$(basename -s .json $WL_PATH)
    echo "[claim-4.sh] Starting workload $WL"
    # In the lingo of the paper, "--no-opt-useless-abstract" is "no-arg-capture";
    # "--no-opt-force-multiuse" is "no-redundant-args".
    for OPTIM in "" "--no-opt-upper-bound" "--no-opt-force-multiuse" "--no-opt-useless-abstract" "$NO_OPT" ; do
        echo "Running with OPTIM=$OPTIM "
        echo "$STITCH_DIR/target/release/compress $WL_PATH $OPTIM $STITCH_FLAGS"
        $GTIMEOUT $TIMEOUT $GTIME -v $STITCH_DIR/target/release/compress $WL_PATH $OPTIM $STITCH_FLAGS --out=$OUT_DIR/raw/${WL}${OPTIM}.json > $OUT_DIR/stdout/${WL}${OPTIM}.stdout 2>&1 || true #&
    done
done
echo "Done, wrote results to $OUT_DIR"
