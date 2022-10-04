

# List of Claims
- Claim 1: Stitch learns libraries of comparable quality to those found by existing deductive library learning algorithms in prior work, while requiring significantly less resources
- Claim 2: Stitch scales to corpora of programs that contain more and longer programs than would be tractable with prior work.
- 

# Download, installation, and sanity-testing instructions


## Prerequisites

Install rust (tested on 1.64.0; default installation settings) https://www.rust-lang.org/tools/install which currently requires running:

`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

Ensure that `python3` points to python (tested on Python 3.8.10), and ensure that the following dependencies are installed:
`python3 -m pip install numpy matplotlib seaborn`

`python3` will be used by analysis scripts.

## Download stitch and benchmarks
```
git clone https://github.com/mlb2251/stitch.git
cd stitch
git checkout artifact-0
export STITCH_DIR=$PWD
git clone https://github.com/mlb2251/compression_benchmark.git
```
The benchmark repo should be cloned *inside* of the stitch directory, as the instructions above imply.

You may want to add `export STITCH_DIR=path/to/stitch` to your `.bashrc` as this is used by many analysis scripts.

# Evaluation instructions

## Kick the Tires
Expected time: 10 min.

Build and test `stitch` with the commands below. Downloading packages and compilation will each take a handful of minutes and will be the bulk of the runtime, while the tests themselves will likely take less than a minute (Our test runtimes: 24s in VM; 7.4s outside of VM).

```
make build
make test
```



## Claim 1
Expected time: 


From the root of the stitch repo, run:
```
git checkout artifact-main
cd experiments
make claim-1
```

View the relevant plots in `stitch/experiments/plots`:
- Fig 1: `benches_compression_ratio_min.pdf`
- Fig 2: `benches_mem_peak_kb.pdf`
- Fig 3: `benches_time_per_inv_with_rewrite.pdf`

## Claim 2
From the root of the stitch repo, run:
```
git checkout artifact-experiments
cd experiments
make claim-2
```


## Claim 3

```
git checkout artifact-experiments
cd experiments
make claim-3
```

## Claim 4

```
git checkout artifact-ablation_experiments
cd experiments
make claim-4
```


## Claim 5

```
git checkout artifact-1
cd experiments
make claim-2
```




# Additional artifact description







# For reference, we provide the steps it took for us to get the Ubuntu 20.04 LTS running Stitch

Install git with
`sudo apt install git`
`sudo apt install curl`
`sudo apt install python3-pip`


