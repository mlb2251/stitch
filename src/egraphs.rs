// use itertools::Itertools;

use crate::*;
use ahash::{AHashSet};


pub type EGraph = egg::EGraph<Lambda, LambdaAnalysis>;

#[derive(Default)]
pub struct LambdaAnalysis;

/// The analysis data associated with each Lambda node
#[derive(Debug)]
pub struct Data {
    pub free_vars: AHashSet<i32>, // $i vars. For example (lam $2) has free_vars = {1}.
    pub free_ivars: AHashSet<i32>, // #i ivars
    pub inventionless_cost: i32,
}

impl Analysis<Lambda> for LambdaAnalysis {
    type Data = Data;
    fn merge(&mut self, _to: &mut Data, _from: Data) -> DidMerge {
        // we really shouldnt be merging anyone ever
        panic!("EClasses should never be merged because EGraph is only used as a structural hasher in Stitch");
        // assert_eq!(to.free_vars,from.free_vars);
        // assert_eq!(to.free_ivars,from.free_ivars);
        // assert_eq!(to.inventionless_cost,from.inventionless_cost);        
        // false // didnt modify anything
    }
    fn make(egraph: &EGraph, enode: &Lambda) -> Data {
        let mut free_vars: AHashSet<i32> = AHashSet::new();
        let mut free_ivars: AHashSet<i32> = AHashSet::new();
        match enode {
            Lambda::Var(i) => {
                free_vars.insert(*i);
            }
            Lambda::IVar(i) => {
                free_ivars.insert(*i);
            }
            Lambda::Prim(_) => {
            }
            Lambda::App([f, x]) => {
                // union of f and x
                free_vars.extend(egraph[*f].data.free_vars.iter());
                free_vars.extend(egraph[*x].data.free_vars.iter());
                free_ivars.extend(egraph[*f].data.free_ivars.iter());
                free_ivars.extend(egraph[*x].data.free_ivars.iter());
            }
            Lambda::Lam([b]) => {
                // body, subtract 1 from all values, remove the -1 if its in there
                free_vars.extend(egraph[*b].data.free_vars.iter()
                    .map(|x| x-1)
                    .filter(|x| *x >= 0));
                free_ivars.extend(egraph[*b].data.free_ivars.iter());
            }
            Lambda::Programs(programs) => {
                // assert no free variables in programs
                for (i,p) in programs.iter().enumerate() {
                    if !egraph[*p].data.free_vars.is_empty() {
                        panic!("Assert failed: free vars found in program {}:\n{}\n{:?}", i, extract(*p,egraph), egraph[*p].data.free_vars);
                    }
                }
                assert!(programs.iter().all(|p| egraph[*p].data.free_ivars.is_empty()));
            }
        }
        let inventionless_cost = match enode {
            Lambda::Var(_) | Lambda::IVar(_) | Lambda::Prim(_) => COST_TERMINAL,
            Lambda::App([f,x]) => {
                    COST_NONTERMINAL
                    + egraph[*f].data.inventionless_cost
                    + egraph[*x].data.inventionless_cost
                }
            Lambda::Lam([b]) => {
                COST_NONTERMINAL + egraph[*b].data.inventionless_cost
            }
            Lambda::Programs(ps) => {
                ps.iter().map(|p| egraph[*p].data.inventionless_cost).sum()
            }
        };
        Data {
               free_vars,
               free_ivars,
               inventionless_cost
            }
    }

    fn modify(_egraph: &mut EGraph, _id: Id) {
    }
}


/// the cost of a program, where `app` and `lam` cost 1, `programs` costs nothing,
/// `ivar` and `var` and `prim` cost 100.
pub struct ProgramCost {}
impl CostFunction<Lambda> for ProgramCost {
    type Cost = i32;
    fn cost<C>(&mut self, enode: &Lambda, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost
    {
        match enode {
            Lambda::Var(_) | Lambda::IVar(_) | Lambda::Prim(_) => COST_TERMINAL,
            Lambda::App([f, x]) => {
                COST_NONTERMINAL + costs(*f) + costs(*x)
            }
            Lambda::Lam([b]) => {
                COST_NONTERMINAL + costs(*b)
            }
            Lambda::Programs(ps) => {
                ps.iter()
                .map(|p|costs(*p))
                .sum()
            }
        }
    }
}

/// depth of a program. For example a leaf is depth 1.
pub struct ProgramDepth {}
impl CostFunction<Lambda> for ProgramDepth {
    type Cost = i32;
    fn cost<C>(&mut self, enode: &Lambda, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost
    {
        match enode {
            Lambda::Var(_) | Lambda::IVar(_) | Lambda::Prim(_) => 1,
            Lambda::App([f, x]) => {
                1 + std::cmp::max(costs(*f), costs(*x))
            }
            Lambda::Lam([b]) => {
                1 + costs(*b)
            }
            Lambda::Programs(ps) => {
                ps.iter()
                .map(|p|costs(*p))
                .max().unwrap()
            }
        }
    }
}


/// does a child first traversal of the egraph and returns a Vec<Id> in that
/// order. Notably an Id will never show up twice (if it showed up earlier
/// it wont show up again). Assumes no cycles in the EGraph.
/// Note that I'm pretty usre this will just return 0,1,2,3,... since due to structural
/// hashing that is a topological ordering
pub fn topological_ordering(root: Id, egraph: &EGraph) -> Vec<Id> {
    let mut vec = Vec::new();
    topological_ordering_rec(root, egraph, &mut vec);
    // let alt = (0..=usize::from(root)).collect::<Vec<usize>>().into_iter().map(|x| Id::from(x)).collect::<Vec<Id>>();
    // assert_eq!(vec, alt);
    vec
}

/// see `topological_ordering`
fn topological_ordering_rec(root: Id, egraph: &EGraph, vec: &mut Vec<Id>) {
    // assumes no cycles.
    // we require at this point that all eclasses only have ONE enode
    assert!(egraph[root].nodes.len() == 1);
    for child in egraph[root].nodes[0].children(){
        topological_ordering_rec(*child, egraph, vec);
    }
    if !vec.contains(&root) {
        // if we're already a child of someone else earlier we dont need to be readded
        vec.push(root);
    }
}

//#[inline(never)]
pub fn associate_tasks(programs_root: Id, egraph: &EGraph, treenodes: &[Id], task_of_root_idx: &[usize]) -> Vec<AHashSet<usize>> {

    // this is the map from egraph node ids to tasks (represented with unique usizes) that we will be building
    let mut tasks_of_node = vec![AHashSet::new(); treenodes.len()];

    let program_roots = egraph[programs_root].nodes[0].children();
    assert_eq!(program_roots.len(), task_of_root_idx.len());

    // since the tasks may not be listed in any specific order, we need to keep track of whether we've already
    // made an id for a given task or not
    for (program_root, task) in program_roots.iter().zip(task_of_root_idx) {
        associate_task_rec(*program_root, egraph, *task, &mut tasks_of_node)
    }

    // defensive sanity check that each entry is non-empty
    assert!(tasks_of_node.iter().all(|tasks| !tasks.is_empty()));

    tasks_of_node
}

fn associate_task_rec(node: Id, egraph: &EGraph, task_id: usize, tasks_of_node: &mut Vec<AHashSet<usize>>) {
    // tasks_of_node.entry(node).or_default().insert(task_id);
    tasks_of_node[usize::from(node)].insert(task_id);
    for child in egraph[node].nodes[0].children() {
        associate_task_rec(*child, egraph, task_id, tasks_of_node);
    }
}

/// Does debruijn index shifting of a subtree, incrementing all Vars by the given amount
#[inline] // useful to inline since callsite can usually tell which Shift type is happening allowing further optimization
pub fn shift(e: Id, incr_by: i32, egraph: &mut crate::EGraph, cache: &mut Option<RecVarModCache>) -> Option<Id> {
    let empty = &mut RecVarModCache::new();
    let seen: &mut RecVarModCache = cache.as_mut().unwrap_or(empty);

    recursive_var_mod(
        |actual_idx, _depth, _which_upward_ref, egraph| {
            Some(egraph.add(Lambda::Var(actual_idx + incr_by)))
        },
        false, // operate on Vars not IVars
        e,egraph,seen
    )
}

/// replaces upward refs to 0 with negative ivar
#[inline]
pub fn insert_arg_ivars(e: Id, set_to: i32, egraph: &mut crate::EGraph) -> Option<Id> {
    recursive_var_mod(
        |actual_idx, _depth, which_upward_ref, egraph| {
            if which_upward_ref == 0 {
                Some(egraph.add(Lambda::IVar(set_to)))
            } else {
                // leave unchanged
                Some(egraph.add(Lambda::Var(actual_idx)))
            }
        },
        false, // operate on Vars not IVars
        e,egraph,&mut RecVarModCache::new()
    )
}

// remap #i to $(i+depth) where depth is depth of #i in `e`
pub fn arg_ivars_to_vars(e: &mut Expr) {
    fn helper(id: Id, e: &mut Expr, depth: i32) {
        // println!("expr: {}", e.to_string_uncurried(Some(id)));

        match e.nodes[usize::from(id)].clone() {
            Lambda::Prim(_) | Lambda::Var(_) => {},
            Lambda::IVar(i) => {
                e.nodes[usize::from(id)] = Lambda::Var(i+depth)
            },
            Lambda::App([f,x]) => {
                helper(f, e, depth);
                helper(x, e, depth);
            },
            Lambda::Lam([b]) => {
                helper(b, e, depth+1);
            },
            _ => unreachable!()
        }
    }
    // println!("expr: {}", e);
    helper(e.root(), e, 0);
}


#[inline]
pub fn is_descendant(descendant: Id, ancestor: Id, egraph: &EGraph) -> bool {
    if descendant >= ancestor {
        return false; // by how structural hashing works descendants are always lower numbers
    }
    fn helper(descendant: Id, ancestor: Id, egraph: &EGraph) -> bool {
        if descendant == ancestor {
            return true;
        }
        match egraph[ancestor].nodes[0] {
            Lambda::Prim(_) | Lambda::Var(_) => false,
            Lambda::IVar(_) => false,
            Lambda::App([f,x]) => helper(descendant, f, egraph) || helper(descendant, x, egraph),
            Lambda::Lam([b]) => helper(descendant, b, egraph),
            _ => unreachable!()
        }
    }
    helper(descendant, ancestor, egraph)
}

// pub fn arity_inference(programs_root: Id, egraph: &EGraph, treenodes: &[Id]) -> Vec<usize> {
//     unimplemented!()
// }
