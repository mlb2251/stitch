.PHONY: claim-2 claim-2-viz claim-3 claim-3-viz clean claim-1 claim-1-viz clean-1 clean-2 clean-3 claim-4 claim-4-viz clean-4

# CLAIM 1

claim-1: compression_benchmark clean-1
	STITCH_DIR=.. ./bench_stitch_all_latest.sh compression_benchmark/benches
	./process_all.sh compression_benchmark/benches
	python3 analyze.py graph_all compression_benchmark/benches

claim-1-viz: compression_benchmark
	python3 analyze.py graph_all compression_benchmark/benches

clean-1:
	echo "Cleaning outputs... Do NOT run this while either claim-1 or claim-2 is running as it will corrupt their outputs"
	rm -rf plots/*
	rm -rf compression_benchmark/benches/*/out/stitch

compression_benchmark:
	git clone -b popl23_camera_ready https://github.com/mlb2251/compression_benchmark.git compression_benchmark

clean-benchmark:
	rm -rf compression_benchmark

dreamcoder: compression_benchmark clean-dreamcoder
	./bench_dreamcoder_all.sh

clean-dreamcoder:
	rm -rf compression_benchmark/benches/*/out/dc

# CLAIM 2

claim-2: def-num-seeds
	rm -rf claim-2-out-${NUM_SEEDS}
	rm -rf claim-2-data
	STITCH_DIR=.. ./claim-2.sh ${NUM_SEEDS} claim-2-out-${NUM_SEEDS}
	python3 plot.py claim-2 claim-2-out-${NUM_SEEDS} ${NUM_SEEDS}

claim-2-viz: def-num-seeds
	python3 plot.py claim-2 claim-2-out-${NUM_SEEDS} ${NUM_SEEDS}

clean-2:
	echo "Cleaning outputs..."
	rm -rf claim-2-out-*

def-num-seeds:
ifndef NUM_SEEDS
	$(error usage: set the environment variable NUM_SEEDS=3 or NUM_SEEDS=50 etc)
endif

# CLAIM 3

claim-3: clean-3
	STITCH_DIR=.. ./claim-3.sh claim-3-out
	python3 plot.py claim-3 claim-3-out

claim-3-viz:
	python3 plot.py claim-3 claim-3-out

clean-3:
	echo "Cleaning outputs..."
	rm -rf claim-3-out

# CLAIM 4

claim-4: clean-4
	STITCH_DIR=.. ./claim-4.sh claim-4-out
	python3 plot.py ablation claim-4-out

claim-4-quick: clean-4
	STITCH_DIR=.. TIMEOUT=60m NO_NO_OPT=1 ./claim-4.sh claim-4-out
	NO_NO_OPT=1 python3 plot.py ablation claim-4-out

claim-4-viz:
	python3 plot.py ablation claim-4-out

clean-4:
	echo "Cleaning outputs..."
	rm -rf claim-4-out
