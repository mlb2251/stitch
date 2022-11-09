claim-1: clean
	STITCH_DIR=.. ./bench_stitch_all_latest.sh compression_benchmark/benches
	./process_all.sh compression_benchmark/benches
	python3 analyze.py graph_all compression_benchmark/benches

claim-1-viz:
	python3 analyze.py graph_all compression_benchmark/benches

clean-1:
	echo "Cleaning outputs... Do NOT run this while either claim-1 or claim-2 is running as it will corrupt their outputs"
	rm -rf plots/*
	rm -rf compression_benchmark/benches/*/out/stitch

clean-dreamcoder:
	rm -rf compression_benchmark/benches/*/out/dc

dreamcoder: clean-dreamcoder
	./bench_dreamcoder_all.sh

claim-2:
	rm -rf claim-2-out-${NUM_SEEDS}
	STITCH_DIR=.. ./claim-2.sh ${NUM_SEEDS} claim-2-out-${NUM_SEEDS}
	python3 plot.py claim-2 claim-2-out-${NUM_SEEDS} ${NUM_SEEDS}

claim-2-viz:
	python3 plot.py claim-2 claim-2-out-${NUM_SEEDS} ${NUM_SEEDS}

claim-3:
	rm -rf claim-3-out
	STITCH_DIR=.. ./claim-3.sh claim-3-out
	python3 plot.py claim-3 claim-3-out

claim-3-viz:
	python3 plot.py claim-3 claim-3-out

clean-2:
	echo "Cleaning outputs..."
	rm -rf claim-2-out-*

clean-3:
	echo "Cleaning outputs..."
	rm -rf claim-3-out

claim-4: clean-4
	STITCH_DIR=.. ./claim-4.sh claim-4-out
	python3 plot.py ablation claim-4-out

claim-4-quick: clean-4
	STITCH_DIR=.. TIMEOUT=60m NO_NO_OPT=1 ./claim-4.sh claim-4-out
	python3 plot.py ablation claim-4-out


claim-4-viz:
	python3 plot.py ablation claim-4-out

clean-4:
	echo "Cleaning outputs..."
	rm -rf claim-4-out

.PHONY: claim-2 claim-2-viz claim-3 claim-3-viz clean claim-1 claim-1-viz clean-1 clean-2 clean-3 claim-4 claim-4-viz clean-4