
for COMPARE_TO in $(python analyze.py latest dreamcoder); do
    ./bench_stitch.sh compare $COMPARE_TO
done


    # python3 analyze.py process dreamcoder $f
# for f in benches/*/out/dc/* ; do
#     echo $f
#     # python3 analyze.py process dreamcoder $f
# done