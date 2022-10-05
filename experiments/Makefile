claim-1: clean
	STITCH_DIR=.. ./bench_stitch_all_latest.sh compression_benchmark/benches
	./process_all.sh compression_benchmark/benches
	python3 analyze.py graph_all compression_benchmark/benches

claim-1-viz:
	python3 analyze.py graph_all compression_benchmark/benches

clean:
	echo "Cleaning outputs... Do NOT run this while either claim-1 or claim-2 is running as it will corrupt their outputs"
	rm -rf plots/*
	rm -rf compression_benchmark/benches/*/out/stitch

.PHONY: claim-1 claim-1-viz clean