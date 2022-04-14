

if [ -z $1 ]
then
    echo "Missing argument to launch script" ;
    exit 1 ;
fi

LOG="launchers/2022-03-12_$1.log"
echo "" > $LOG

if [ $1 -eq 0 ]; then
    ./bench_dreamcoder.sh benches/text_text_ellisk_2019-01-25T20.19.06
    echo ./bench_dreamcoder.sh benches/text_text_ellisk_2019-01-25T20.19.06 >> $LOG
    ./bench_dreamcoder.sh benches/towers_tower_batch_50_3600_ellisk_2019-03-26T10.58.24
    echo ./bench_dreamcoder.sh benches/towers_tower_batch_50_3600_ellisk_2019-03-26T10.58.24 >> $LOG

    ./bench_dreamcoder.sh benches/physics_scientific_unsolved_4h_ellisk_2019-07-20T18.16.45
    echo ./bench_dreamcoder.sh benches/physics_scientific_unsolved_4h_ellisk_2019-07-20T18.16.45 >> $LOG
    ./bench_dreamcoder.sh benches/towers_tower_batch_50_3600_ellisk_2019-03-26T10.51.16
    echo ./bench_dreamcoder.sh benches/towers_tower_batch_50_3600_ellisk_2019-03-26T10.51.16 >> $LOG
    ./bench_dreamcoder.sh benches/logo_logo_batch_50_1h_ellisk_2019-03-23T14.02.03
    echo ./bench_dreamcoder.sh benches/logo_logo_batch_50_1h_ellisk_2019-03-23T14.02.03 >> $LOG
    ./bench_dreamcoder.sh benches/text_text_ellisk_2019-01-24T21.53.45
    echo ./bench_dreamcoder.sh benches/text_text_ellisk_2019-01-24T21.53.45 >> $LOG
    ./bench_dreamcoder.sh benches/physics_scientific_unsolved_4h_ellisk_2019-07-20T18.20.13
    echo ./bench_dreamcoder.sh benches/physics_scientific_unsolved_4h_ellisk_2019-07-20T18.20.13 >> $LOG
    ./bench_dreamcoder.sh benches/list_list_hard_test_ellisk_2019-02-15T11.39.19
    echo ./bench_dreamcoder.sh benches/list_list_hard_test_ellisk_2019-02-15T11.39.19 >> $LOG
    ./bench_dreamcoder.sh benches/text_text_ellisk_2019-01-24T22.05.53
    echo ./bench_dreamcoder.sh benches/text_text_ellisk_2019-01-24T22.05.53 >> $LOG
    ./bench_dreamcoder.sh benches/physics_scientific_unsolved_4h_ellisk_2019-07-20T18.13.12
    echo ./bench_dreamcoder.sh benches/physics_scientific_unsolved_4h_ellisk_2019-07-20T18.13.12 >> $LOG
    ./bench_dreamcoder.sh benches/list_list_hard_test_ellisk_2019-02-15T11.35.48
    echo ./bench_dreamcoder.sh benches/list_list_hard_test_ellisk_2019-02-15T11.35.48 >> $LOG
    ./bench_dreamcoder.sh benches/physics_scientific_unsolved_4h_ellisk_2019-07-20T18.09.34
    echo ./bench_dreamcoder.sh benches/physics_scientific_unsolved_4h_ellisk_2019-07-20T18.09.34 >> $LOG
    ./bench_dreamcoder.sh benches/list_list_hard_test_ellisk_2019-02-15T11.31.39    
    echo ./bench_dreamcoder.sh benches/list_list_hard_test_ellisk_2019-02-15T11.31.39 >> $LOG
    


fi

if [ $1 -eq 1 ]; then
    ./bench_dreamcoder.sh benches/list_list_hard_test_ellisk_2019-02-15T11.26.41
    echo ./bench_dreamcoder.sh benches/list_list_hard_test_ellisk_2019-02-15T11.26.41 >> $LOG
    ./bench_dreamcoder.sh benches/logo_logo_batch_50_1h_ellisk_2019-03-23T14.16.56
    echo ./bench_dreamcoder.sh benches/logo_logo_batch_50_1h_ellisk_2019-03-23T14.16.56 >> $LOG
    ./bench_dreamcoder.sh benches/physics_scientific_unsolved_4h_ellisk_2019-07-20T18.05.46
    echo ./bench_dreamcoder.sh benches/physics_scientific_unsolved_4h_ellisk_2019-07-20T18.05.46 >> $LOG

    ./bench_dreamcoder.sh benches/towers_tower_batch_50_3600_ellisk_2019-03-26T11.01.50
    echo ./bench_dreamcoder.sh benches/towers_tower_batch_50_3600_ellisk_2019-03-26T11.01.50 >> $LOG
    ./bench_dreamcoder.sh benches/text_text_ellisk_2019-01-24T21.49.39
    echo ./bench_dreamcoder.sh benches/text_text_ellisk_2019-01-24T21.49.39 >> $LOG
    ./bench_dreamcoder.sh benches/list_list_hard_test_ellisk_2019-02-15T11.43.28
    echo ./bench_dreamcoder.sh benches/list_list_hard_test_ellisk_2019-02-15T11.43.28 >> $LOG
    ./bench_dreamcoder.sh benches/towers_tower_batch_50_3600_ellisk_2019-03-26T11.05.16
    echo ./bench_dreamcoder.sh benches/towers_tower_batch_50_3600_ellisk_2019-03-26T11.05.16 >> $LOG
    ./bench_dreamcoder.sh benches/logo_logo_batch_50_1h_ellisk_2019-03-23T14.05.43
    echo ./bench_dreamcoder.sh benches/logo_logo_batch_50_1h_ellisk_2019-03-23T14.05.43 >> $LOG
    ./bench_dreamcoder.sh benches/text_text_ellisk_2019-01-24T21.58.02
    echo ./bench_dreamcoder.sh benches/text_text_ellisk_2019-01-24T21.58.02 >> $LOG
    ./bench_dreamcoder.sh benches/logo_logo_batch_50_1h_ellisk_2019-03-23T14.09.23
    echo ./bench_dreamcoder.sh benches/logo_logo_batch_50_1h_ellisk_2019-03-23T14.09.23 >> $LOG
    ./bench_dreamcoder.sh benches/towers_tower_batch_50_3600_ellisk_2019-03-26T10.54.52
    echo ./bench_dreamcoder.sh benches/towers_tower_batch_50_3600_ellisk_2019-03-26T10.54.52 >> $LOG
    ./bench_dreamcoder.sh benches/logo_logo_batch_50_1h_ellisk_2019-03-23T14.13.04
    echo ./bench_dreamcoder.sh benches/logo_logo_batch_50_1h_ellisk_2019-03-23T14.13.04 >> $LOG
    ./bench_dreamcoder.sh benches/text_text_ellisk_2019-01-24T22.01.57
    echo ./bench_dreamcoder.sh benches/text_text_ellisk_2019-01-24T22.01.57 >> $LOG
fi

# shuffled commands minus mccarthy and rational:










# text_text_ellisk_2019-01-25T20.19.06
# towers_tower_batch_50_3600_ellisk_2019-03-26T10.58.24
# physics_scientific_unsolved_4h_ellisk_2019-07-20T18.16.45
# towers_tower_batch_50_3600_ellisk_2019-03-26T10.51.16
# logo_logo_batch_50_1h_ellisk_2019-03-23T14.02.03
# text_text_ellisk_2019-01-24T21.53.45
# physics_scientific_unsolved_4h_ellisk_2019-07-20T18.20.13
# list_list_hard_test_ellisk_2019-02-15T11.39.19
# text_text_ellisk_2019-01-24T22.05.53
# physics_scientific_unsolved_4h_ellisk_2019-07-20T18.13.12
# list_list_hard_test_ellisk_2019-02-15T11.35.48
# physics_scientific_unsolved_4h_ellisk_2019-07-20T18.09.34
# list_list_hard_test_ellisk_2019-02-15T11.31.39
# towers_tower_batch_50_3600_ellisk_2019-03-26T11.01.50
# text_text_ellisk_2019-01-24T21.49.39
# list_list_hard_test_ellisk_2019-02-15T11.26.41
# list_list_hard_test_ellisk_2019-02-15T11.43.28
# towers_tower_batch_50_3600_ellisk_2019-03-26T11.05.16
# logo_logo_batch_50_1h_ellisk_2019-03-23T14.16.56
# logo_logo_batch_50_1h_ellisk_2019-03-23T14.05.43
# text_text_ellisk_2019-01-24T21.58.02
# logo_logo_batch_50_1h_ellisk_2019-03-23T14.09.23
# physics_scientific_unsolved_4h_ellisk_2019-07-20T18.05.46
# origami_McCarthy_unsolved_16h_tk5_ellisk_2019-07-11T15.44.44
# towers_tower_batch_50_3600_ellisk_2019-03-26T10.54.52
# logo_logo_batch_50_1h_ellisk_2019-03-23T14.13.04
# text_text_ellisk_2019-01-24T22.01.57


# sorted order:

# list_list_hard_test_ellisk_2019-02-15T11.26.41
# list_list_hard_test_ellisk_2019-02-15T11.31.39
# list_list_hard_test_ellisk_2019-02-15T11.35.48
# list_list_hard_test_ellisk_2019-02-15T11.39.19
# list_list_hard_test_ellisk_2019-02-15T11.43.28
# logo_logo_batch_50_1h_ellisk_2019-03-23T14.02.03
# logo_logo_batch_50_1h_ellisk_2019-03-23T14.05.43
# logo_logo_batch_50_1h_ellisk_2019-03-23T14.09.23
# logo_logo_batch_50_1h_ellisk_2019-03-23T14.13.04
# logo_logo_batch_50_1h_ellisk_2019-03-23T14.16.56
# origami_McCarthy_unsolved_16h_tk5_ellisk_2019-07-11T15.44.44
# physics_scientific_unsolved_4h_ellisk_2019-07-20T18.05.46
# physics_scientific_unsolved_4h_ellisk_2019-07-20T18.09.34
# physics_scientific_unsolved_4h_ellisk_2019-07-20T18.13.12
# physics_scientific_unsolved_4h_ellisk_2019-07-20T18.16.45
# physics_scientific_unsolved_4h_ellisk_2019-07-20T18.20.13
# text_text_ellisk_2019-01-24T21.49.39
# text_text_ellisk_2019-01-24T21.53.45
# text_text_ellisk_2019-01-24T21.58.02
# text_text_ellisk_2019-01-24T22.01.57
# text_text_ellisk_2019-01-24T22.05.53
# text_text_ellisk_2019-01-25T20.19.06
# towers_tower_batch_50_3600_ellisk_2019-03-26T10.51.16
# towers_tower_batch_50_3600_ellisk_2019-03-26T10.54.52
# towers_tower_batch_50_3600_ellisk_2019-03-26T10.58.24
# towers_tower_batch_50_3600_ellisk_2019-03-26T11.01.50
# towers_tower_batch_50_3600_ellisk_2019-03-26T11.05.16