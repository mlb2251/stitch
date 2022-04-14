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

NON_TOWER_DOMAINS=( "dials" "furniture" "nuts-bolts" "wheels" )
# single ablation experiments
#for WL_PATH in $STITCH_DIR/data/cogsci/*.json; do
for WL in "castle" "city" "house" ; do
    #WL=$(basename -s .json $WL_PATH)
    WL_PATH=${STITCH_DIR}/data/cogsci/${WL}.json
    echo "[ablation.sh] Starting workload $WL"
    #for FVO in "--no-opt-free-vars" "" ; do
    #OPTIMS=( "" "--no-opt-single-use" "--no-opt-upper-bound" "--no-opt-force-multiuse" "--no-opt-arity-zero")
    #if [[ "${NON_TOWER_DOMAINS[*]}" =~ "${WL}" ]]; then
    #    # workload is not a tower, add UAO as optim to consider
    #    OPTIMS+=( "--no-opt-useless-abstract" )
    #fi

    for OPTIM in "--no-opt-useless-abstract" ; do
        echo "Running with OPTIM=$OPTIM"
        $STITCH_DIR/target/release/compress $WL_PATH $OPTIM --verbose-best --max-arity=3 --iterations=1 --no-mismatch-check --out=$OUT_DIR/raw/${WL}${OPTIM}.json > $OUT_DIR/stderr/${WL}${OPTIM}.stderr 2>&1 &
    done
done
wait $(jobs -rp)

# ablation combination experiments
#for WL_PATH in $STITCH_DIR/data/cogsci/*.json; do
#    WL=$(basename -s .json $WL_PATH)
#    echo "[ablation.sh] Starting workload $WL"
#    #for FVO in "--no-opt-free-vars" "" ; do
#    for FVO in "" ; do
#    for SUO in "--no-opt-single-use" "" ; do
#    for STO in "--no-opt-single-task" ; do
#    for UBO in "--no-opt-upper-bound" "" ; do
#    for MUO in "--no-opt-force-multiuse" "" ; do
#    #for UAO in "--no-opt-useless-abstract" "" ; do
#    for UAO in "" ; do
#    for AZO in "--no-opt-arity-zero" "" ; do
#        echo "Running with FVO=$FVO, SUO=$SUO, UBO=$UBO, MUO=$MUO, UAO=$UAO, AZO=$AZO"
#        $STITCH_DIR/target/release/compress $WL_PATH $FVO $SUO $UBO $MUO $UAO $AZO --verbose-best --max-arity=3 --iterations=1 --no-mismatch-check --out=$OUT_DIR/raw/$WL-$FVO-$SUO-$UBO-$MUO-$UAO-$AZO.json > $OUT_DIR/stderr/$WL-$FVO-$SUO-$UBO-$MUO-$UAO-$AZO.stderr2>&1 &
#    done
#    wait $(jobs -rp)
#    done
#    done
#    done
#    done
#    done
#    done
#done


echo "Done: $OUT_DIR"
