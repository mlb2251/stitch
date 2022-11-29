
This is the official repo for the tool `stitch` presented in the POPL 2023 paper "Top-Down Synthesis For Library Learning". A pre-print of Stitch is available [here](https://mlb2251.github.io/stitch.pdf). The artifact for reproducing results from the paper can be found [here](https://github.com/mlb2251/stitch-artifact).

# Stitch

## Installation & Testing

- Install `rust` from [here](https://www.rust-lang.org/tools/install)
- Clone this repo
- ensure that `cargo run --release --bin=compress -- data/cogsci/nuts-bolts.json` runs without crashing
- For a more thorough test, run `make test`

## Quickstart

Lets take a look at some simple examples of the `stitch` input format. Put the following in a new file `data/basic/ex1.json`:
```json
[
    "(foo (a a a))",
    "(bar (b b b))"
]
```
As above, stitch input format is a json file containing a list of input programs, where each program is a string written in a lisp-like lambda calculus syntax. The first program in this example corresponds to the curried lambda calculus expression `(app foo (app (app a a) a))`.

The clear structure in these examples is that they all contain a subterm of the form `\x. (x x x)`. Lets see if stitch can pull that out:

```
cargo run --release --bin=compress -- data/basic/ex1.json --max-arity=3 --iterations=1
```

(If you're having any trouble, check out other examples like `data/basic/simple1.json` to make sure you have the right format.)

The output should look like:
```
=======Compression Summary=======
Found 1 inventions
Cost Improvement: (1.33x better) 806 -> 604
fn_0 (1.33x wrt orig): utility: 200 | final_cost: 604 | 1.33x | uses: 2 | body: [fn_0 arity=1: (#0 #0 #0)]
Time: 0ms
Wrote to "out/out.json"
```

Primer on the output format:
- `Cost Improvement: (1.33x better) 806 -> 604`
  - here we see that by the cost metric (which values terminals like `foo` and `a` as `100` and nonterminals like `app` and `lam` as `1`) our programs were rewritten to be 1.33x smaller. To see the actual rewritten programs you can include `--show-rewritten` in the command and the rewritten programs will appear a few lines above the compression summary:
    - `(foo (fn_0 a))` and `(bar (fn_0 b))`
- `fn_0`
  - this is the name the new primitive was assigned
- `(1.33x wrt orig)`
  - this is a *cumulative* measure of compression (ie "with respect to original"), so if we were learning more than one abstraction it would represent the accumulated compression over all previous abstractions
- `utility: 200`
  - This is the utility, which corresponds to the difference in program cost before and after rewriting.
- `final_cost: 604`
  - final cost of programs after rewriting
- `1.33x`
  - compression gained from this abstraction - again, only relevant when learning more than one abstraction
- `uses: 2`
  - the abstraction is used twice in the set of programs
- `body: [fn_0 arity=1: (#0 #0 #0)]`
  - this is the abstraction itself! `(#0 #0 #0)` is equivalent to `\x. (x x x)` - the first abstraction variable is always `#0`, the second is `#1`, etc.

Theres also a more complete output that is sent to `out/out.json` by default and can be consumed by other programs that are using stitch as a subroutine (if they arent using the Rust/Python bindings for it). A very important flag is `--rewritten-intermediates`, which includes the rewritten version in the output after *each* abstraction is found - this can be very helpful for understanding the abstractions you're learning.

Now let's take a look at the output of one of the benchmarks from the paper. This will be the `data/cogsci/nuts-bolts.json` file from the [Wong et al. 2022](https://arxiv.org/abs/2205.05666) dataset, feel free to open the file and take a look.

Run it, using `--iterations=3` to get 3 abstractions:
```
cargo run --release --bin=compress -- data/cogsci/nuts-bolts.json --max-arity=3 --iterations=3
```

The output should end with the following compression summary:
```
=======Compression Summary=======
Found 3 inventions
Cost Improvement: (6.06x better) 1919558 -> 316890
fn_0 (1.78x wrt orig): utility: 837792 | final_cost: 1079238 | 1.78x | uses: 320 | body: [fn_0 arity=2: (T (repeat (T l (M 1 0 -0.5 (/ 0.5 (tan (/ pi #1))))) #1 (M 1 (/ (* 2 pi) #1) 0 0)) (M #0 0 0 0))]
fn_1 (3.81x wrt orig): utility: 572767 | final_cost: 503538 | 2.14x | uses: 190 | body: [fn_1 arity=3: (repeat (T (T #2 (M 0.5 0 0 0)) (M 1 0 (* #1 (cos (/ pi 4))) (* #1 (sin (/ pi 4))))) #0 (M 1 (/ (* 2 pi) #0) 0 0))]
fn_2 (6.06x wrt orig): utility: 185436 | final_cost: 316890 | 1.59x | uses: 168 | body: [fn_2 arity=1: (T (T c (M 2 0 0 0)) (M #0 0 0 0))]
Time: 120ms
Wrote to "out/out.json"
```

These are written in a low-level graphics DSL, and the first function (which yields 1.78x compression) is the function for rendering a scaled n-sided polygon, which is used 320 times in the dataset.

Primer on input format:
- Variables should be written as *de Bruijn* indices (i.e. `$i` refers to the variable bound by the `i`th lambda above it) so `\x. \y. x y` is written `(lam (lam ($1 $0)))`
- Lambdas need explicit parentheses around their body so `(lam + 3 2)` should instead be written `(lam (+ 3 2))`. The parser outputs an error message explaining this if you make this mistake. Lambdas can also be written with `lambda` instead of `lam` but the output of stitch will always be normalized to use `lam`.
- Be sure to balance your parentheses.
- You don't need to pre-define a DSL or anything to work with `stitch`. Any space-separated series of tokens that isn't reserved for something else is treated as a DSL primitive, like `foo` and `a` in the earlier example or any of the primitive likes `T` or `-0.5` in the second example.
- check out other examples in `data/basic/` and `data/cogsci/`

## Common command-line arguments

- `--max-arity=2` or `-a2` controls max arity of abstraction found (default is 2). Try to keep the arity relatively low if you don't need high arity abstractions, as it can significantly increase runtime.
- `--iterations=10` or `-i10` controls how many iterations of compression to run. Each iteration produces one abstraction (which can build on the previous ones)
- `--threads=10` or `-t10` is a quick way to boost performance by multithreading (default is 1)

## All command-line arguments
From `cargo run --release --bin=compress -- --help`
```
USAGE:
    compress [OPTIONS] <FILE>

ARGS:
    <FILE>    json file to read compression input programs from

OPTIONS:
    -a, --max-arity <MAX_ARITY>
            max arity of abstractions to find (will find all from 0 to this number inclusive)
            [default: 2]

        --allow-single-task
            allow for abstractions that are only useful in a single task

        --args-from-json
            extracts argument values from the json; specifically assumes a key value pair like
            "stitch_args": "data/dc/logo_iteration_1_stitchargs.json -a3 -t8 --fmt=dreamcoder
            --dreamcoder-drop-last --no-mismatch-check", in the toplevel dictionary of the json. All
            other commandline args get discarded when you specify this option

    -b, --batch <BATCH>
            how many worklist items a thread will take at once [default: 1]

        --cost <COST>
            Cost function to use [default: dreamcoder] [possible values: dreamcoder]

        --cost-app <COST_APP>
            Override `cost` with a custom application cost

        --cost-ivar <COST_IVAR>
            Override `cost` with a custom abstraction variable cost

        --cost-lam <COST_LAM>
            Override `cost` with a custom lambda cost

        --cost-prim-default <COST_PRIM_DEFAULT>
            Override `cost` with a custom default primitive cost

        --cost-var <COST_VAR>
            Override `cost` with a custom $i variable cost

        --dreamcoder-comparison
            anything related to running a dreamcoder comparison

        --dynamic-batch
            threads will autoadjust how large their batches are based on the worklist size

        --fmt <FMT>
            the format of the input file, e.g. 'programs-list' for a simple JSON array of programs
            or 'dreamcoder' for a JSON in the style expected by the original dreamcoder codebase.
            See [formats.rs] for options or to add new ones [default: programs-list] [possible
            values: dreamcoder, programs-list, split-programs-list]

        --follow <FOLLOW>
            pattern or abstraction to follow. if `follow_prune=True` we will aggressively prune to
            only follow this pattern, otherwise we will just verbosely print when ancestors of this
            pattern are encountered

        --follow-prune
            for use with `--follow`, enables aggressive pruning

    -h, --help
            Print help information

        --hole-choice <HOLE_CHOICE>
            Method for choosing hole to expand at each step, doesn't have a huge effect [default:
            depth-first] [possible values: random, breadth-first, depth-first, max-largest-subset,
            high-entropy, low-entropy, max-cost, min-cost, many-groups, few-groups, few-apps]

    -i, --iterations <ITERATIONS>
            Number of iterations to run compression for (number of inventions to find) [default: 3]

        --inv-arg-cap
            disables the edge case handling where argument capture needs to be inverted for
            optimality

    -n, --inv-candidates <INV_CANDIDATES>
            Number of invention candidates compression_step should return in a *single* step. Note
            that these will be the top n optimal candidates modulo subsumption pruning (and the
            top-1  is guaranteed to be globally optimal) [default: 1]

        --no-mismatch-check
            disables the safety check for the utility being correct; you only want to do this if you
            truly dont mind unsoundness for a minute

        --no-opt
            disable all optimizations

        --no-opt-arity-zero
            disable the arity zero priming optimization

        --no-opt-force-multiuse
            disable the force multiuse pruning optimization

        --no-opt-single-use
            disable the single structurally hashed subtree match pruning

        --no-opt-upper-bound
            disable the upper bound pruning optimization

        --no-opt-useless-abstract
            disable the useless abstraction pruning optimization

        --no-other-util
            makes it so utility is based purely on corpus size without adding in the abstraction
            size

        --no-stats
            Disable stat logging - note that stat logging in multithreading requires taking a mutex
            so it can be a source of slowdown in the massively multithreaded case, hence this flag
            to disable it

        --no-top-lambda
            makes it so inventions cant start with a lambda at the top

    -o, --out <OUT>
            json output file [default: out/out.json]

        --print-stats <PRINT_STATS>
            print stats this often (0 means never) [default: 0]

    -r, --show-rewritten
            print out programs rewritten under abstraction

        --rewrite-check
            whenever you finish an invention do a full rewrite to check that rewriting doesnt raise
            a cost mismatch exception

        --rewritten-dreamcoder
            include `rewritten_dreamcoder` in the output json

        --rewritten-intermediates
            include `rewritten` from each intermediate rewritten result in the output json after
            each invention

        --save-rewritten <SAVE_REWRITTEN>
            saves the rewritten frontiers in an input-readable format

        --shuffle
            shuffle order of set of inventions

        --silent
            silence all printing

    -t, --threads <THREADS>
            number of threads (no parallelism if set to 1) [default: 1]

        --truncate <TRUNCATE>
            truncate set of inventions to include only this many (happens after shuffle if shuffle
            is also specified)

        --utility-by-rewrite
            calculate utility exhaustively by performing a full rewrite; mainly used when cost
            mismatches are happening and we need something slow but accurate

        --verbose-best
            prints whenever a new best abstraction is found

        --verbose-worklist
            prints every worklist item as it is processed (will slow things down a ton due to
            rendering out expressins)
```

## Python Bindings

Currently initial Python bindings are offered in the [stitch_bindings repo](https://github.com/mlb2251/stitch_bindings).

## Generating a flamegraph

Installation:

```
cargo install flamegraph
```

Running:
```
cargo flamegraph --root --open --deterministic --output=out/flamegraph.svg --bin=compress -- data/cogsci/nuts-bolts.json
```

## Acknowledgement

This work is supported by the National Science Foundation under Grant No. 1918839 *Understanding the World Through Code* http://www.neurosymbolic.org/ 

This work is supported in part by the Defense Advanced Research Projects Agency (DARPA) under the program Symbiotic Design for Cyber Physical Systems (SDCPS) Contract FA8750-20-C-0542 (Systemic Generative Engineering). The views, opinions, and/or findings expressed are those of the author(s) and do not necessarily reflect the view of DARPA.
