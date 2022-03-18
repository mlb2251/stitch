# Extracting compression messages from the DreamCoder PLDI artifact
See `extract_data.py` in this directory.
Simply pass the log file (from `experimentOutputs/` in the artifact) that you want to extract the compression messages from, as well as a directory to write the results to (one file per compression iteration).
Note that it must be placed within the `bin/` directory of the artifact before being run, so that it can find the relevant DreamCoder modules.
Note also that this script relies on the `ECResult.frontiersOverTime` field, which is not present in some of the older log files.
TODO(theoxo): List precisely which log files this is a problem in, and see if there is a way around it.

# Running DreamCoder compression on Linux (e.g. sketch4)

Clone the LAPS repo. Then:
```
cd ocaml/linux_bin
./compression < message.json
```
Where `message.json` is in the format constructed on line 34 in [DreamCoder's compression.py](https://github.com/CatherineWong/laps_dreamcoder/blob/d4e7cece89d71c5d1d25a972116b7aec2d16f1e6/compression.py#L53).

For example, `message.json` can be `data/dc/dc_log_log.json` if you first make the following modifications to the latter:
1. `pseudoCounts`, `aic`, `structurePenalty` should be of type `float`, not `int`
2. `lc_score` needs to be added to the JSON. Setting it to `0.0` should be safe if you're only running compression in isolation.

TODO(theoxo): Double check with Cathy/Gabe that it's OK to set `lc_score` to `0.0` if running compression outside of the LAPS/DreamCoder loop. 

# Evaluate the likelihood of programs under a new Grammar
TODO