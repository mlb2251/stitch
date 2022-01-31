use crate::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Result;
use std::collections::{HashMap, HashSet};

pub type Env<D> = Vec<Val<D>>; // env[i] is the binding for $i
pub type Envs<D> = Vec<Env<D>>;
pub type EvalResults<D> = Vec<(Env<D>,Val<D>)>;

type EGraph = egg::EGraph<Lambda, LambdaAnalysis>;


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
    let inputs: Vec<Envs<D>> = tasks.iter().map(|t| t.inputs.clone()).collect();
    // we strip the toplevel lambdas before entering EGC
    let programs: Executable<D> = {
        let mut exprs: Vec<Expr> = tasks.iter().map(|t| t.program.clone()).collect();
        for (expr,envs) in exprs.iter_mut().zip(inputs.iter()) {
            let arity = envs[0].len();
            println!("arity {} solution: {}", arity, expr);
            assert!(envs.iter().all(|e| e.len() == arity));
            expr.strip_lambdas(arity);
        }
        Expr::programs(exprs).into()
    };

    let mut egraph: EGraph = Default::default();
    let eggid_of_expid: Vec<Id> = programs.expr.add_and_remap(&mut egraph);



    let roots: Vec<Id> = programs.expr.nodes[usize::from(programs.expr.root())].children().iter().copied().collect();
    let num_expids: usize = programs.expr.nodes.len() - 1; // 0..num_expids will be everything but the Programs node
    let num_eggids: usize = usize::from(*eggid_of_expid.iter().max().unwrap()) ; // 0..num_eggids will be everything but the Programs node

    // todo caching of executions will not be compositional sadly with this strategy - will just affect runtime not results
    let mut canonical_expid_of_eggid: Vec<usize> = (0..num_eggids).map(|i| eggid_of_expid.iter().position(|j| i==usize::from(*j)).unwrap()).collect();

    // execute the programs on their inputs to build up evalresults at each node.
    for (root_id,envs) in roots.iter().zip(inputs.iter()) {
        for env in envs.iter() {
            programs.eval_child_safe(*root_id, env).unwrap();
        }
    }

    let evalresults: Vec<EvalResults<D>> = (0..num_expids).map(|i| programs.evals_of_node(i.into())).collect();
    println!("{:?}",evalresults);
    // let mut free_vars: Vec<HashSet<i32>> = programs.expr.all_free_vars(false);
    // let mut costs: Vec<i32> = programs.expr.all_costs();
    // free_vars.pop(); // remove the Programs node

    assert_eq!(num_expids,evalresults.len());
    // assert_eq!(N,free_vars.len());

    let mut rewrites: Vec<Vec<Id>> = vec![vec![]; num_expids];

    // Now lets go full N^2 (not just N*(N-1)) and pick i,j to see if `i <- j` rewrite is valid
    for exprid in 0..num_expids {
        assert!(!evalresults[exprid].is_empty(), "no eval results for node {}: {}", exprid, programs.expr.to_string_uncurried(Some(exprid.into())));
        let eggid = eggid_of_expid[exprid];
        for eggid2 in (0..num_eggids).map(|i| Id::from(i)) {
            // check if i <- j is valid, so i.fvs must be a subset of j.fvs. Note that
            // it's more precise to use free_vars here than to use the env vecs because much
            // of the env might never be used.
            if !egraph[eggid].data.free_vars.is_subset(&egraph[eggid2].data.free_vars) {
                continue;
            }
            let expid2 = canonical_expid_of_eggid[usize::from(eggid2)];
            // execute j in each env of i, exiting early if you ever come across a crash or a differing result
            let mut ok = true;
            for (env,ires) in evalresults[exprid].iter() {
                match programs.eval_child_safe(expid2.into(), env) {
                    Ok(jres) => if jres != *ires { ok = false; break } // if result differs i <- j is not valid
                    Err(_) => {ok = false; break}, // if error then i <- j is not valid
                }
            }
            if !ok { continue; }
            rewrites[exprid].push(eggid2); // successfully found a rewrite that works!
        }
    }

    println!("done! found {} rewrites over {} nodes", rewrites.iter().map(|r| r.len()).sum::<usize>(), num_expids);

    for expid in 0..num_expids {
        if rewrites[expid].is_empty() {
            println!("node {} has no rewrites: {}", expid, programs.expr.to_string_uncurried(Some(expid.into())));
        } else {
            println!("node {} has {} rewrites: {}", expid, rewrites[expid].len(), programs.expr.to_string_uncurried(Some(expid.into())));
            for eggid2 in rewrites[expid].iter() {
                println!("  {} <- {}: {}", expid, eggid2, programs.expr.to_string_uncurried(Some(canonical_expid_of_eggid[usize::from(*eggid2)].into())));
            }
        }
    }

    println!("\nChoosing rewrites...\n");

    let mut rewritten = programs.clone();
    let mut worklist: Vec<Id> = roots.clone();

    while let Some(node) = worklist.pop() {
        println!("Rewriting {}: {}", node, programs.expr.to_string_uncurried(Some(node.into())));
        println!("Options:");
        for eggid2 in rewrites[usize::from(node)].iter() {
            println!("  {} <- {}: {}", node, eggid2, programs.expr.to_string_uncurried(Some(canonical_expid_of_eggid[usize::from(*eggid2)].into())));
        }

        let min_cost: i32 = rewrites[usize::from(node)].iter().map(|eggid2|egraph[*eggid2].data.inventionless_cost).min().unwrap();
        let small_rewrites: Vec<Id> = rewrites[usize::from(node)].iter().filter(|eggid2| egraph[**eggid2].data.inventionless_cost == min_cost).copied().collect::<Vec<_>>();

        // for now just choose the smallest Id to be canonical about it - in the future we'll be smarter than this
        let chosen_rewrite: Id = *small_rewrites.iter().min().unwrap();

        


    }
    

}