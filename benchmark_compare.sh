#!/bin/bash

# Check if we have the right number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <old_commit_hash> <new_commit_hash>"
    echo "Example: $0 abcd1234 wxyz5678"
    exit 1
fi

OLD_COMMIT=$1
NEW_COMMIT=$2

# Store the absolute paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BINDINGS_DIR="$SCRIPT_DIR/../stitch_bindings"

# Check if stitch_bindings directory exists
if [ ! -d "$BINDINGS_DIR" ]; then
    echo "Error: stitch_bindings directory not found at $BINDINGS_DIR"
    exit 1
fi

# Function to update Cargo.toml and run benchmark
run_benchmark() {
    local commit=$1
    echo "Running benchmark for commit: $commit"
    
    # Update Cargo.toml to use the specified commit
    sed -i "s/stitch_core = { git = \"https:\/\/github.com\/mlb2251\/stitch\"[^}]*}/stitch_core = { git = \"https:\/\/github.com\/mlb2251\/stitch\", rev = \"$commit\" }/" Cargo.toml
    
    # Run the benchmark
    make benchmark-minimal
}

# Navigate to stitch_bindings directory
cd "$BINDINGS_DIR"
mkdir -p experiments/plots
pip install prettytable
echo "Changed to directory: $BINDINGS_DIR"

# First run with old commit
echo "Starting benchmark for old commit ($OLD_COMMIT)..."
run_benchmark $OLD_COMMIT
make plots-old

# Then run with new commit
echo "Starting benchmark for new commit ($NEW_COMMIT)..."
run_benchmark $NEW_COMMIT

# Print the results location
RESULTS_PATH="$BINDINGS_DIR/experiments/results.html"
echo "Benchmark comparison completed!"
echo "View results at: $RESULTS_PATH" 