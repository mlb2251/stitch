# <img src="dream_egg.png" alt="egg of dreams" height="40" align="left"> DreamEgg

# Quickstart

Run `cargo run --bin=compress --release -- -f data/train_19.json --max-arity=2 --iterations=3`

This will run compression on 19 programs from the dreamcoder logo graphics domain. The largest is depth ~11 and has ~10 leaf nodes, and there is a LOT of obvious shared structure if you look at the file. The command should have pulled out the n-sided polygon function first:

`Chose Invention inv0: ([arity=2]: (lam (logo_forLoop #0 (lam (lam (logo_FWRT (logo_MULL logo_UL #1) (logo_DIVA logo_UA #0) $0))) $0)))`

Note that `#i` is used for invention args and `$i` for original program args (this avoids many index shifting woes!).

Change `--iterations` to alter how many inventions will be greedily found. Change `--max-arity` to alter the maximum allowed arity of inventions. Arity scaling is very exponential. It should be roughly linear with number of programs but exponential with number of leaf nodes. There are many other data files in the repo, including for example `data/train_200.json` which has 200 programs from the logo domain, the largest being depth ~23 and ~50 leaf nodes.

Some relevant files in `src/`
* `bin/compress.rs` the executable that runs when you do `cargo run --bin=compress`
* `domains/` domain semantics implementations go here! You don't need this if you are just doing compression without execution/semantics.
* `compression.rs` the core code for compression
* `domain.rs` all about giving semantics to your programs!
* `lib.rs` the parent dreamegg library file
* `expr.rs` all about expressions / nodes
* `macros.rs` some macros to make life easier
* `run_with_timeout.rs` lets you run a closure with a timeout in a separate process
* `old/old_egg.rs` the old egg based implementation if you're curious

