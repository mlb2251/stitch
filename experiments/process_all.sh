if [ -z $1 ]
then
    echo "Usage: ./process_all.sh benches"
    exit 1
fi

BENCH_DIR=$1

for f in $BENCH_DIR/*/out/dc/* ; do
    # echo $f
    python3 analyze.py process dreamcoder $f
done

for f in $BENCH_DIR/*/out/stitch/* ; do
    # echo $f
    python3 analyze.py process stitch $f
done