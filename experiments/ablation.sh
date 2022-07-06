#!/bin/bash
set -e
ulimit -v 50000000  # limit children to 50GB=50,000,000KB of virtual memory

if [ -z $STITCH_DIR ]
then
    echo "[ablation.sh] Please set environment variable \$STITCH_DIR to the stitch/ directory"
    exit 1
fi

OUT_DIR="cogsci/out/stitch/$(TZ='America/New_York' date '+%Y-%m-%d_%H-%M-%S')"

# compile Stitch
pushd $STITCH_DIR
cargo build --release --bin=compress
popd

mkdir -p $OUT_DIR/raw
mkdir -p $OUT_DIR/stderr

# run Stitch on all the input files from the run
for WL_PATH in $STITCH_DIR/data/cogsci/*.json; do
    WL=$(basename -s .json $WL_PATH)
    echo "[ablation.sh] Starting workload $WL"
    for FVO in "--no-opt-free-vars" "" ; do
    for STO in "--no-opt-single-task" "" ; do
    for UBO in "--no-opt-upper-bound" "" ; do
    for MUO in "--no-opt-force-multiuse" "" ; do
    #for UAO in "--no-opt-useless-abstract" "" ; do
    for UAO in "" ; do
    for AZO in "--no-opt-arity-zero" "" ; do
        echo "Running with FVO=$FVO, STO=$STO, UBO=$UBO, MUO=$MUO, UAO=$UAO, AZO=$AZO"
        $STITCH_DIR/target/release/compress $WL_PATH $FVO $STO $UBO $MUO $UAO $AZO --max-arity=3 --iterations=10 --out=$OUT_DIR/raw/$WL-$FVO-$STO-$UBO-$MUO-$UAO-$AZO.json > $OUT_DIR/stderr/$WL-$FVO-$STO-$UBO-$MUO-$UAO-$AZO.stderr2>&1 &
    done
    wait $(jobs -rp)
    done
    done
    done
    done
    done
done


echo "Done: $OUT_DIR"
