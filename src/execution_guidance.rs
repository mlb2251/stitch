use crate::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Result;

pub type Env<D> = Vec<Val<D>>; // env[i] is the binding for $i
pub type Envs<D> = Vec<Env<D>>;
pub type EvalResults<D> = Vec<(Env<D>,Val<D>)>;


#[derive(Serialize, Deserialize, Clone)]
pub struct Task<D: Domain> {
    pub program: Expr,
    pub inputs: Envs<D>,
}

pub fn execution_guided_compression<D: Domain>(
    tasks: Vec<Task<D>>,
    args: &CompressionArgs,
    out_dir: &str,
) {
    let programs: Executable<D> = Expr::programs(tasks.iter().map(|t| t.program.clone()).collect()).into();
    let inputs: Vec<Envs<D>> = tasks.iter().map(|t| t.inputs.clone()).collect();

    let roots: Vec<Id> = programs.expr.nodes[usize::from(programs.expr.root())].children().iter().copied().collect();


    // execute the programs on their inputs to build up a semantic set at each node.
    for (root_id,envs) in roots.iter().zip(inputs.iter()) {
        for env in envs.iter() {
            programs.eval_child_safe(*root_id, env).unwrap();
        }
    }

    let cvecs: Vec<EvalResults<D>> = (0..programs.expr.root().into()).map(|i| programs.evals_of_node(i.into())).collect();


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