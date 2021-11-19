# <img src="dream_egg.png" alt="egg of dreams" height="40" align="left"> DreamEgg


A smaller example - 19 programs - largest one being depth ~11 and ~10 leaf nodes. There is a LOT of obvious shared structure if you look at the file.

`cargo run --release -- -f "data/train_19.json" --max-arity=2 --iterations=1`

It should extract the n-gon drawing invention:
`***Found Invention inv0: [arity 2]: (lam (lam (lam (app (app (app logo_forLoop #0) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) #1)) (app (app logo_DIVA logo_UA) #0)) $0)))) $0))))` where `#i` is used for invention args and `$i` for original program args (this avoids many index shifting woes!).


A larger example - 200 programs - largest one being depth ~23 and ~50 leaf nodes. Dreamcoder would probably deal only with partially rewritten versions of programs this big (using whatever inventions were found) so you wouldn't actually have to scale to this.

`cargo run --release -- -f "data/train_200.json" --max-arity=2 --iterations=1`

We greedily extract the single best invention and rewrite and repeat. Use `--inventions=` to set how many inventions to extract in this way.

Note that dreamcoder generally wants arity=3 which will be much much much slower.

Some relevant files
* `src/old_egg.rs` has the old egg based implementation of beta inversion
* `src/main.rs` has the current version that doesnt do rewrites


