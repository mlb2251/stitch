#!/bin/bash
set -e


# for BENCHGROUP in trash/benches/*; do
#     for RUN in $BENCHGROUP/out/dc/*; do
#         # echo "Restoring run: $RUN"
#         TO="$BENCHGROUP/out/dc"
#         mv $RUN ${TO#trash/}
#         echo mv $RUN ${TO#trash/}
#     done
# done


for BENCHGROUP in benches/*; do
    if [ -d $BENCHGROUP/out/dc ]; then
        for RUN in $BENCHGROUP/out/dc/*; do
            if ! [ -s $RUN/raw/bench000* ]; then
                echo "Trashing empty run: $RUN"
                mkdir -p trash/$BENCHGROUP/out/dc
                mv $RUN trash/$BENCHGROUP/out/dc
            fi
        done
    fi
done