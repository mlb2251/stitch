use crate::*;
use serde::{Deserialize, Serialize};
use serde_json::Result;

type Env<D> = Vec<Val<D>>; // env[i] is the binding for $i
type Envs<D> = Vec<Env<D>>;
type EvalResults<D> = Vec<(Env<D>,Val<D>)>;


#[derive(Serialize, Deserialize)]
struct Task<D: Domain> {
    program: String,
    inputs: Envs<D>,
}



fn execution_guided_compression(
    programs_expr: &Expr,
    args: &CompressionArgs,
    out_dir: &str,
) {

    // execute the programs on their inputs to build up a semantic set at each node.

    /*
        1. execute the programs on their inputs to build up a semantic set at each node. We'll store everything in a hashmap keyed by Id (in one big Programs Expr).
        2. filter down these semantic sets to throw out any variables in the contexts that arent actually FVs in the expr itself (bc we're clearly invariant to these). Freeze these semantic sets.
        3. Now pairwise (using full N^2 not just handshake) pick i,j ids and decide if `i <- j` as follows:
            - if i.fvs is not a subset of j.fvs, abort
            - (in the future possibly ensure ctx match up by types too)
            - execute `j` in the ctxlist of the semantic set of `i`. This can be done by a map() of the eval function over the ctxlist. Notably we've frozen our semantic sets so we wont be directly trying to access the Executables caches, and theyll just be used simply as caches which should make everything go fast.
            - if the resulting semantic list is equal to the semantic list of `i`, then `i <- j` is valid and we can add `j` to the Vec in the hashmap keyed by `i`.
    */
}