

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
export STITCH_DIR=$PWD
git clone https://github.com/mlb2251/compression_benchmark.git
```
The benchmark repo should be cloned *inside* of the stitch directory, as the instructions above imply.

You may want to add `export STITCH_DIR=path/to/stitch` to your `.bashrc` as this is used by many analysis scripts.

# Evaluation instructions

## Claim 1


./bench_stitch_all_latest.sh compression_benchmark/benches







## Kick the Tires

`make build`

`make test`



# Additional artifact description







# For reference, we provide the steps it took for us to get the Ubuntu 20.04 LTS running Stitch

Install git with
`sudo apt install git`
`sudo apt install curl`
`sudo apt install python3-pip`


