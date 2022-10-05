
for domain in "list" "logo" "physics" "text" "towers"
do
    echo "starting domain $domain"
    for bench in compression_benchmark/benches/$domain*
    do
        echo "starting bench $bench"
        COMPRESSION_BIN=./compression ./bench_dreamcoder.sh $bench
    done
done