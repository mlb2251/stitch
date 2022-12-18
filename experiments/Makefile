.PHONY: claim-2 claim-2-3 claim-2-50 claim-2-viz claim-3 claim-3-viz clean claim-1 claim-1-viz clean-1 clean-2 clean-3 claim-4 claim-4-viz clean-4

# CLAIM 1

clean: clean-1 clean-2 clean-3 clean-4 clean-benchmark
	rm -f out/out.json
	echo "cleaned all outputs"

claim-1: compression_benchmark clean-1
	STITCH_DIR=.. ./bench_stitch_all_latest.sh compression_benchmark/benches
	./process_all.sh compression_benchmark/benches
	make claim-1-viz

claim-1-viz: compression_benchmark clean-1-viz
	python3 analyze.py graph_all compression_benchmark/benches

clean-1: clean-1-viz
	rm -rf compression_benchmark/benches/*/out/stitch
clean-1-viz:
	rm -rf plots/benches_*

compression_benchmark:
	git clone -b popl23_camera_ready https://github.com/mlb2251/compression_benchmark.git compression_benchmark

clean-benchmark:
	rm -rf compression_benchmark

dreamcoder: compression_benchmark clean-dreamcoder
	./bench_dreamcoder_all.sh

clean-dreamcoder:
	rm -rf compression_benchmark/benches/*/out/dc

# CLAIM 2

SEEDS := 3

claim-2:
	python3 analyze.py claim-2 ${SEEDS}

# CLAIM 3

claim-3: clean-3
	STITCH_DIR=.. ./claim-3.sh out/claim-3-out
	make claim-3-viz

claim-3-viz: clean-3-viz
	rm -f plots/claim-3.pdf
	python3 plot.py claim-3 out/claim-3-out

clean-3: clean-3-viz
	rm -rf out/claim-3-out
clean-3-viz:
	rm -f plots/claim-3.pdf
	rm -f plots/claim-3.png

# CLAIM 4

claim-4: clean-4
	STITCH_DIR=.. ./claim-4.sh out/claim-4-out
	python3 plot.py ablation out/claim-4-out

claim-4-quick: clean-4
	STITCH_DIR=.. TIMEOUT=30m NO_NO_OPT=1 ./claim-4.sh out/claim-4-out
	NO_NO_OPT=1 python3 plot.py ablation out/claim-4-out

claim-4-viz:
	python3 plot.py ablation out/claim-4-out

clean-4:
	rm -rf out/claim-4-out

# it can be finicky to kill the ablation study - this does it for you
claim-4-kill:
	pkill -u ${USER} --full "./claim-4.sh"
	pkill -u ${USER} --full "../target/release/compress"