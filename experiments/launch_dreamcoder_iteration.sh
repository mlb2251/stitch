#!/bin/bash
set -e

if [ -z $COMPRESSION_BIN ]
then
    echo "Please set environment variable \$COMPRESSION_BIN to the ocaml compression binary" 
    exit 1
fi

echo "[launch_dreamcoder_iteration.sh] NEVER RUN MULTIPLE COPIES OF THIS SCRIPT AT ONCE"
echo "[launch_dreamcoder_iteration.sh] (it relies on unique access to compressionMessages/)"

if [ -z $3 ]
then
    echo "[launch_dreamcoder_iteration.sh] Usage: ./launch_dreamcoder_iteration.sh DOMAIN RUN ITERATION"
    echo "[launch_dreamcoder_iteration.sh] For example: ./launch_dreamcoder_iteration.sh logo 2019-03-23T18:06:23.106382 2"
    echo "[launch_dreamcoder_iteration.sh] Or here's a simple example: ./launch_dreamcoder_iteration.sh example example 1"
    exit 1
fi


rm -rf compressionMessages
mkdir compressionMessages # the dreamcoder compression binary will use this folder
echo "[launch_dreamcoder_iteration.sh] mkdir compressionMessages"

DOMAIN=$1
RUN=$2
ITERATION=$3

INPUT="data/$DOMAIN/$RUN/iteration_${ITERATION}.json"

if ! [ -f $INPUT ]
then
    echo "[launch_dreamcoder_iteration.sh] input file not found: $INPUT"
    exit 1
fi

LOG_STDERR="out/$DOMAIN/$RUN/iteration_${ITERATION}_rerun_stderr.log"
LOG_STDOUT="out/$DOMAIN/$RUN/iteration_${ITERATION}_rerun_dsl.json"
LOG_COMPRESSION_MESSAGES="out/$DOMAIN/$RUN/iteration_${ITERATION}_rerun_compressionMessages"
mkdir -p out/$DOMAIN/$RUN
mkdir -p $LOG_COMPRESSION_MESSAGES
rm -f $LOG_COMPRESSION_MESSAGES/*

echo "[launch_dreamcoder_iteration.sh] *** Starting ***"
echo "[launch_dreamcoder_iteration.sh] DOMAIN: $DOMAIN"
echo "[launch_dreamcoder_iteration.sh] RUN: $RUN"
echo "[launch_dreamcoder_iteration.sh] ITERATION: $ITERATION"
echo "[launch_dreamcoder_iteration.sh] INPUT: $INPUT"
echo "[launch_dreamcoder_iteration.sh] LOG_STDERR: $LOG_STDERR"
echo "[launch_dreamcoder_iteration.sh] LOG_STDOUT: $LOG_STDOUT"
echo "[launch_dreamcoder_iteration.sh] LOG_COMPRESSION_MESSAGES: $LOG_COMPRESSION_MESSAGES"


# time + measure memory for the compression binary running on the input
CMD="/usr/bin/time -v $COMPRESSION_BIN $INPUT"

echo "[launch_dreamcoder_iteration.sh] Launching Compression Binary at $(date '+%Y-%m-%d_%H-%M-%S')"
echo "[launch_dreamcoder_iteration.sh] $CMD"
# remove lines that start with "0.000000" or "Frontier" which are spam and fill up memory to log
($CMD 1>$LOG_STDOUT) 2>&1 | awk "\$1 != \"0.000000\" && \$1 != \"Frontier\" {print}" | tee $LOG_STDERR 1>&2

echo "[launch_dreamcoder_iteration.sh] Compression Binary Exited at $(date '+%Y-%m-%d_%H-%M-%S')"

echo "[launch_dreamcoder_iteration.sh] saving compressionMessages/ to $LOG_COMPRESSION_MESSAGES"
cd compressionMessages
for f in * ; do
    mv $f ../$LOG_COMPRESSION_MESSAGES/$f.json
done
cd ..


echo "[launch_dreamcoder_iteration.sh] *** Done ***"
echo "[launch_dreamcoder_iteration.sh] DOMAIN: $DOMAIN"
echo "[launch_dreamcoder_iteration.sh] RUN: $RUN"
echo "[launch_dreamcoder_iteration.sh] ITERATION: $ITERATION"
echo "[launch_dreamcoder_iteration.sh] INPUT: $INPUT"
echo "[launch_dreamcoder_iteration.sh] LOG_STDERR: $LOG_STDERR"
echo "[launch_dreamcoder_iteration.sh] LOG_STDOUT: $LOG_STDOUT"
echo "[launch_dreamcoder_iteration.sh] LOG_COMPRESSION_MESSAGES: $LOG_COMPRESSION_MESSAGES"
