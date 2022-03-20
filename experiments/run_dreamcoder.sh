#!/bin/bash
set -e

if [ -z $COMPRESSION_BIN ]
then
    echo "Please set environment variable \$COMPRESSION_BIN to the ocaml compression binary" 
    exit 1
fi

echo "NEVER RUN MULTIPLE COPIES OF THIS SCRIPT AT ONCE"
echo "(it relies on unique access to compressionMessages/)"

# if [ -d "compressionMessages" ]
# then
#     echo "Please delete compressionMessages/ before launching this script."
#     echo "Note that only one copy of this script can be running at a time because"
#     echo "we need to use the output in the fixed output directory compressionMessages/"
#     echo "after the binary stops running. Also because for performance tests it makes"
#     echo "sense to only run one copy of this at a time."
#     exit 1
# fi

if [ -z $3 ]
then
    echo "Usage: ./run_dreamcoder.sh DOMAIN RUN ITERATION"
    echo "For example: ./run_dreamcoder.sh logo 2019-03-23T18:06:23.106382 2"
    echo "Or here's a simple example: ./run_dreamcoder.sh example example 1"
    exit 1
fi


rm -rf compressionMessages
mkdir compressionMessages # the dreamcoder compression binary will use this folder
echo "mkdir compressionMessages"

DOMAIN=$1
RUN=$2
ITERATION=$3

INPUT="data/$DOMAIN/$RUN/iteration_${ITERATION}.json"

if ! [ -f $INPUT ]
then
    echo "input file not found: $INPUT"
    exit 1
fi

LOG_STDERR="out/$DOMAIN/$RUN/iteration_${ITERATION}_stderr.log"
LOG_STDOUT="out/$DOMAIN/$RUN/iteration_${ITERATION}_stdout.log"
LOG_COMPRESSION_MESSAGES="out/$DOMAIN/$RUN/iteration_${ITERATION}_compressionMessages"
mkdir -p out/$DOMAIN/$RUN
mkdir -p $LOG_COMPRESSION_MESSAGES
rm -f $LOG_COMPRESSION_MESSAGES/*

echo "*** Starting ***"
echo "DOMAIN: $DOMAIN"
echo "RUN: $RUN"
echo "ITERATION: $ITERATION"
echo "INPUT: $INPUT"
echo "LOG_STDERR: $LOG_STDERR"
echo "LOG_STDOUT: $LOG_STDOUT"
echo "LOG_COMPRESSION_MESSAGES: $LOG_COMPRESSION_MESSAGES"

CMD="/usr/bin/time -v $COMPRESSION_BIN $INPUT"

echo "Launching Compression Binary"
$CMD > >(tee -a $LOG_STDOUT) 2> >(tee -a $LOG_STDERR >&2)
echo "Compression Binary Exited"

echo "saving compressionMessages/ to $LOG_COMPRESSION_MESSAGES"
for f in compressionMessages/* ; do
    mv compressionMessages/$f $LOG_COMPRESSION_MESSAGES/$f.json
done


echo "*** Done ***"
echo "DOMAIN: $DOMAIN"
echo "RUN: $RUN"
echo "ITERATION: $ITERATION"
echo "INPUT: $INPUT"
echo "LOG_STDERR: $LOG_STDERR"
echo "LOG_STDOUT: $LOG_STDOUT"
echo "LOG_COMPRESSION_MESSAGES: $LOG_COMPRESSION_MESSAGES"
