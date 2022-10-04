claim-4:
	export STITCH_DIR=..
	./ablation.sh claim-4-out
	python3 plot.py ablation claim-4-out

claim-4-viz:
	python3 plot.py ablation claim-4-out

clean:
	cargo clean
	rm -rf claim-4-out

.PHONY: claim-4 claim-4-viz clean