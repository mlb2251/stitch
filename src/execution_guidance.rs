use crate::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Result;
use std::collections::{HashMap, HashSet};

pub type Env<D> = Vec<Val<D>>; // env[i] is the binding for $i
pub type Envs<D> = Vec<Env<D>>;
pub type EvalResults<D> = Vec<(Env<D>,Val<D>)>;


#[derive(Serialize, Deserialize, Clone)]
pub struct Task<D: Domain> {
    // #[serde(deserialize_with = "deserialize_expr")]
    pub program: Expr,
    // #[serde(deserialize_with = "deserialize_envs")]
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
    let N: usize = usize::from(programs.expr.root()) - 1;


    // execute the programs on their inputs to build up evalresults at each node.
    for (root_id,envs) in roots.iter().zip(inputs.iter()) {
        for env in envs.iter() {
            programs.eval_child_safe(*root_id, env).unwrap();
        }
    }

    let evalresults: Vec<EvalResults<D>> = (0..N).map(|i| programs.evals_of_node(i.into())).collect();
    let free_vars: Vec<HashSet<i32>> = programs.expr.free_vars(false);

    assert_eq!(N,evalresults.len());
    assert_eq!(N,free_vars.len());

    let mut rewrites: Vec<Vec<Id>> = vec![vec![]; N];

    // Now lets go full N^2 (not just N*(N-1)) and pick i,j to see if `i <- j` rewrite is valid
    for i in 0..N {
        for j in 0..N {
            // check if i <- j is valid, so i.fvs must be a subset of j.fvs. Note that
            // it's more precise to use free_vars here than to use the env vecs because much
            // of the env might never be used.
            if !free_vars[i].is_subset(&free_vars[j]) {
                continue;
            }
            // execute j in each env of i, exiting early if you ever come across a crash or a differing result
            for (env,ires) in evalresults[i].iter() {
                match programs.eval_child_safe(j.into(), env) {
                    Ok(jres) => if jres != *ires { continue } // if result differs i <- j is not valid
                    Err(_) => continue, // if error then i <- j is not valid
                }
            }
            rewrites[i].push(j.into()); // successfully found a rewrite that works!
        }
    }

    println!("done! found {} rewrites over {} nodes", rewrites.iter().map(|r| r.len()).sum::<usize>(), N);

}