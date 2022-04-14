#!/bin/bash

# This script should be run in <EXPS>/stderr/,
# where <EXPS> is the directory containing your ablation
# study results (i.e. the outputs of ablation.sh)
echo "workload,UAO,UBO,MUO,AZO,FVO,STO,number of nodes" > results.csv
LINES=()
for F in *.stderr2 ; do
    FVO="on"
    STO="on"
    UBO="on"
    MUO="on"
    UAO="on"
    AZO="on"
    if [[ $F =~ "free-vars" ]]; then
        FVO="off"
    fi
    if [[ $F =~ "single-task" ]]; then
        STO="off"
    fi
    if [[ $F =~ "upper-bound" ]]; then
        UBO="off"
    fi
    if [[ $F =~ "force-multiuse" ]]; then
        MUO="off"
    fi
    if [[ $F =~ "useless-abstract" ]]; then
        UAO="off"
    fi
    if [[ $F =~ "arity-zero" ]]; then
        AZO="off"
    fi
    WL=$(echo $F | sed 's/\([^-]*\).*/\1/')
    NUM_NODES=$(grep -F "worklist_steps" $F | sed 's/[^w]*worklist_steps: \([^,]*\).*/\1/')
    LINES+=(`echo "$WL,$UAO,$UBO,$MUO,$AZO,$FVO,$STO,$NUM_NODES"`)
done

# Sort the results
IFS=$'\n'
LINES=( $(printf "%s\n" ${LINES[@]} | sort) )

for LINE in ${LINES[@]} ; do
    echo $LINE >> results.csv
done

