

for f in benches/*/out/dc/* ; do
    # echo $f
    python3 analyze.py process dreamcoder $f
done