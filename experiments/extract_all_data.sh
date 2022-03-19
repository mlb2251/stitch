#!/bin/bash

# The below must match the path to the bin/ directory of the DreamCoder
# PLDI artifact, and must not include a trailing /
ARTIFACT_BIN_PATH="/scratch/theoxo/artifact/bin"
# The below must match the path to the experimentOutputs/ directory
# of the DreamCoder PLDI artifact, and must not include a trailing /
EXP_OUTS_PATH="/scratch/theoxo/artifact/experimentOutputs"

# These are the logfiles mentioned by K.E. in their email,
# minus the 'recursive functional programming ones'.
declare -a LOG_FILES=(
     "list/2019-02-15T16:31:58.353555/list_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=720_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=10_TRR=randomShuffle_K=2_topkNotMAP=False_graph=True.pickle"
     "list/2019-02-15T16:36:06.861462/list_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=720_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=10_TRR=randomShuffle_K=2_topkNotMAP=False_graph=True.pickle"
     "list/2019-02-15T16:39:37.056521/list_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=720_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=10_TRR=randomShuffle_K=2_topkNotMAP=False_graph=True.pickle"
     "list/2019-02-15T16:43:47.195104/list_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=720_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=10_TRR=randomShuffle_K=2_topkNotMAP=False_graph=True.pickle"
     "list/2019-02-15T16:47:54.339779/list_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=720_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=10_TRR=randomShuffle_K=2_topkNotMAP=False_graph=True.pickle"
     "logo/2019-03-23T18:06:23.106382/logo_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=3600_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=50_TRR=randomShuffle_K=2_topkNotMAP=False_graph=True.pickle"
     "logo/2019-03-23T18:10:01.834307/logo_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=3600_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=50_TRR=randomShuffle_K=2_topkNotMAP=False_graph=True.pickle"
     "logo/2019-03-23T18:13:43.818837/logo_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=3600_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=50_TRR=randomShuffle_K=2_topkNotMAP=False_graph=True.pickle"
     "logo/2019-03-23T18:17:35.090237/logo_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=3600_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=50_TRR=randomShuffle_K=2_topkNotMAP=False_graph=True.pickle"
     "logo/2019-03-23T18:21:31.435612/logo_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=3600_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=50_TRR=randomShuffle_K=2_topkNotMAP=False_graph=True.pickle"
     "logo/2019-11-29T16:33:12.597455/logo_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=3600_HR=0.5_it=20_MF=5_noConsolidation=False_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_L=1.5_batch=50_TRR=randomShuffle_K=2_topkNotMAP=False_graph=True.pickle"
     "rational/2019-02-15T11:28:40.635813/rational_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=120_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.0_batch=10_TRR=randomShuffle_K=2_topkNotMAP=False_graph=True.pickle"
     "rational/2019-02-19T18:04:34.743890/rational_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=120_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.0_batch=10_TRR=randomShuffle_K=2_topkNotMAP=False_graph=True.pickle"
     "rational/2019-02-15T11:28:27.615614/rational_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=120_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.0_batch=10_TRR=randomShuffle_K=2_topkNotMAP=False_graph=True.pickle"
     "rational/2019-02-15T11:28:17.171165/rational_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=120_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.0_batch=10_TRR=randomShuffle_K=2_topkNotMAP=False_graph=True.pickle"
     "rational/2019-11-29T16:56:44.307505/rational_aic=1.0_arity=3_BO=True_CO=True_ES=1_ET=120_HR=0.5_it=20_MF=5_noConsolidation=False_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_solver=ocaml_STM=True_L=1.0_batch=10_TRR=randomShuffle_K=2_topkNotMAP=False_graph=True.pickle"
     "rational/2019-02-15T11:28:20.454707/rational_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=120_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.0_batch=10_TRR=randomShuffle_K=2_topkNotMAP=False_graph=True.pickle"
     "text/2019-01-25T02:53:59.182941/text_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=720_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=10_TRR=randomShuffle_K=2_topkNotMAP=False_rec=True_vec=True_graph=True.pickle"
     "text/2019-01-25T02:58:17.082195/text_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=720_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=10_TRR=randomShuffle_K=2_topkNotMAP=False_rec=True_vec=True_graph=True.pickle"
     "text/2019-01-25T03:02:10.385246/text_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=720_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=10_TRR=randomShuffle_K=2_topkNotMAP=False_rec=True_vec=True_graph=True.pickle"
     "text/2019-01-25T03:06:06.834290/text_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=720_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=10_TRR=randomShuffle_K=2_topkNotMAP=False_rec=True_vec=True_graph=True.pickle"
     "text/2019-01-25T03:09:45.988462/text_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=720_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=10_TRR=randomShuffle_K=2_topkNotMAP=False_rec=True_vec=True_graph=True.pickle"
     "text/2019-01-26T01:23:35.287813/text_aic=1.0_arity=3_aux=True_BO=True_CO=True_ES=1_ET=720_HR=0.5_it=20_MF=5_pc=30.0_RS=5000_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=10_TRR=randomShuffle_K=2_topkNotMAP=False_rec=True_vec=True_graph=True.pickle"
     "regex/2019-03-04T19:13:10.698186/regex_aic=1.0_arity=3_BO=True_CO=False_ES=1_ET=720_HR=0.5_it=23_MF=10_pc=30.0_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=10_TRR=randomShuffle_K=2_topkNotMAP=True_graph=True.pickle"
     "regex/2019-03-04T19:17:09.170192/regex_aic=1.0_arity=3_BO=True_CO=False_ES=1_ET=720_HR=0.5_it=23_MF=10_pc=30.0_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=10_TRR=randomShuffle_K=2_topkNotMAP=True_graph=True.pickle"
     "regex/2019-03-04T19:21:20.153727/regex_aic=1.0_arity=3_BO=True_CO=False_ES=1_ET=720_HR=0.5_it=23_MF=10_pc=30.0_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=10_TRR=randomShuffle_K=2_topkNotMAP=True_graph=True.pickle"
     "regex/2019-03-04T19:25:56.137631/regex_aic=1.0_arity=3_BO=True_CO=False_ES=1_ET=720_HR=0.5_it=23_MF=10_pc=30.0_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=10_TRR=randomShuffle_K=2_topkNotMAP=True_graph=True.pickle"
     "regex/2019-03-04T19:30:01.252339/regex_aic=1.0_arity=3_BO=True_CO=False_ES=1_ET=720_HR=0.5_it=23_MF=10_pc=30.0_RT=3600_RR=False_RW=False_STM=True_L=1.5_batch=10_TRR=randomShuffle_K=2_topkNotMAP=True_graph=True.pickle"
     )

for LOG_FILE in ${LOG_FILES[@]} ; do
    echo "Started extracting data from $LOG_FILE"
    DOMAIN=$(echo "$LOG_FILE" | awk -F "/" '{print $1}')
    RUN=$(echo "$LOG_FILE" | awk -F "/" '{print $2}')
    echo "Domain=$DOMAIN, Run=$RUN"
    echo "Logging extraction process to $DOMAIN/$RUN/.log"
    mkdir -p "$DOMAIN/$RUN"  # this is just making a folder called e.g. regex/2019...252339/
    echo "$LOG_FILE" > "$DOMAIN/$RUN/.log"
    python3 "$ARTIFACT_BIN_PATH/data_extractor.py" "$EXP_OUTS_PATH/$LOG_FILE" "$DOMAIN/$RUN" >> "$DOMAIN/$RUN/.log" 2>&1
    echo "Finished extraction from $LOG_FILE"
done


# for the recursive functional programming domain, the domain name is listed as "list" and K.E.
# found pickles for each of the 10 iterations. Saving them each separately and setting the domain name
# to "rec-fp"
# TODO(theoxo): Is this the most sane behaviour?
for ITERATION in {1..10} ; do
    LOG_FILE="list/2019-07-11T19:49:10.899159/list_aic=1.0_arity=4_ET=57600_it={$ITERATION}_MF=5_noConsolidation=False_pc=30.0_RW=False_solver=ocaml_STM=True_L=1.0_TRR=unsolved_K=5_topkNotMAP=False_rec=False.pickle"
    echo "Started extracting data from $LOG_FILE"
    DOMAIN="rec-fp"
    RUN=$(echo "$LOG_FILE" | awk -F "/" '{print $2}')
    RUN="iteration=${ITERATION}_$RUN"
    echo "Domain=$DOMAIN, Run=$RUN"
    echo "Logging extraction process to $DOMAIN/$RUN/.log"
    mkdir -p "$DOMAIN/$RUN"  # this is just making a folder called e.g. regex/2019...252339/
    echo "$LOG_FILE" > "$DOMAIN/$RUN/.log"
    python3 "$ARTIFACT_BIN_PATH/data_extractor.py" "$EXP_OUTS_PATH/$LOG_FILE" "$DOMAIN/$RUN" >> "$DOMAIN/$RUN/.log" 2>&1
    echo "Finished extraction from $LOG_FILE"
done
