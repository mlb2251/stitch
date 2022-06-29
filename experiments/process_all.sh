

for f in benches/*/out/dc/* ; do
    # echo $f
    python3 analyze.py process dreamcoder $f
done

for f in benches/*/out/stitch/* ; do
    # echo $f
    python3 analyze.py process stitch $f
done