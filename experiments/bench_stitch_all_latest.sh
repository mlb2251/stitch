

if [ -z $1 ]
then
    echo "Usage: ./bench_stitch_all_latest.sh benches/"
    exit 1
fi

BENCHES=$1

for COMPARE_TO in $(python3 analyze.py latest dreamcoder $BENCHES); do
    ./bench_stitch.sh compare $COMPARE_TO
done


    # python3 analyze.py process dreamcoder $f
# for f in benches/*/out/dc/* ; do
#     echo $f
#     # python3 analyze.py process dreamcoder $f
# done