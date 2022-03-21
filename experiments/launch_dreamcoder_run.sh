#!/bin/bash

if [ -z $2 ]
then
    echo "[launch_dreamcoder_run.sh] Usage: ./launch_dreamcoder_run.sh DOMAIN RUN"
    echo "[launch_dreamcoder_run.sh] For example: ./launch_dreamcoder_run.sh logo 2019-03-23T18:06:23.106382"
    echo "[launch_dreamcoder_run.sh] Or here's a simple example: ./launch_dreamcoder_run.sh example example"
    exit 1
fi

DOMAIN=$1
RUN=$2

mkdir -p "trash/$DOMAIN/$RUN"

if [ -d "out/$DOMAIN/$RUN" ]; then
    TRASH="trash/$DOMAIN/$RUN/$(date '+%Y-%m-%d_%H-%M-%S')"
    echo "[launch_dreamcoder_run.sh] output file out/$DOMAIN/$RUN, moving to $TRASH"
    mv out/$DOMAIN/$RUN $TRASH
    continue
fi


mkdir -p "out/$DOMAIN/$RUN"


echo "[launch_dreamcoder_run.sh] Starting run at $(date '+%Y-%m-%d_%H-%M-%S')"


for i in `seq 0 19`; do
    INPUT="data/$DOMAIN/$RUN/iteration_${i}.json"
    LOG="out/$DOMAIN/$RUN/iteration_${i}_launch_dreamcoder_iteration.log"
    
    if ! [ -f $INPUT ]
    then
        echo "[launch_dreamcoder_run.sh] input file for iteration $i not found, skipping: $INPUT"
        continue
    fi

    echo "[launch_dreamcoder_run.sh] launching launch_dreamcoder_iteration.sh and logging to $LOG"
    echo "[launch_dreamcoder_run.sh] ./launch_dreamcoder_iteration.sh $DOMAIN $RUN $i"
    ./launch_dreamcoder_iteration.sh $DOMAIN $RUN $i 2>&1 | tee $LOG
    echo "[launch_dreamcoder_run.sh] done, logged to $LOG"
done

echo "[launch_dreamcoder_run.sh] Run finished at $(date '+%Y-%m-%d_%H-%M-%S')"


# all experiments:
# list 2019-02-15T16:31:58.353555 
# list 2019-02-15T16:36:06.861462 
# list 2019-02-15T16:39:37.056521 
# list 2019-02-15T16:43:47.195104 
# list 2019-02-15T16:47:54.339779 
# logo 2019-03-23T18:06:23.106382 
# logo 2019-03-23T18:10:01.834307 
# logo 2019-03-23T18:13:43.818837 
# logo 2019-03-23T18:17:35.090237 
# logo 2019-03-23T18:21:31.435612 
# logo 2019-11-29T16:33:12.597455 
# rational 2019-02-15T11:28:40.635813 
# rational 2019-02-19T18:04:34.743890 
# rational 2019-02-15T11:28:27.615614 
# rational 2019-02-15T11:28:17.171165 
# rational 2019-11-29T16:56:44.307505 
# rational 2019-02-15T11:28:20.454707 
# text 2019-01-25T02:53:59.182941 
# text 2019-01-25T02:58:17.082195 
# text 2019-01-25T03:02:10.385246 
# text 2019-01-25T03:06:06.834290 
# text 2019-01-25T03:09:45.988462 
# text 2019-01-26T01:23:35.287813 
# regex 2019-03-04T19:13:10.698186 
# regex 2019-03-04T19:17:09.170192 
# regex 2019-03-04T19:21:20.153727 
# regex 2019-03-04T19:25:56.137631 
# regex 2019-03-04T19:30:01.252339 