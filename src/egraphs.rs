use itertools::Itertools;

use crate::*;
use std::collections::{HashMap, HashSet};

pub type EGraph = egg::EGraph<Lambda, LambdaAnalysis>;

#[derive(Default)]
pub struct LambdaAnalysis;

/// The analysis data associated with each Lambda node
#[derive(Debug)]
pub struct Data {
    pub free_vars: HashSet<i32>, // $i vars. For example (lam $2) has free_vars = {1}.
    pub free_ivars: HashSet<i32>, // #i ivars
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
        let mut free_vars: HashSet<i32> = HashSet::new();
        let mut free_ivars: HashSet<i32> = HashSet::new();
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
                assert!(programs.iter().all(|p| egraph[*p].data.free_vars.is_empty()));
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
               free_vars: free_vars,
               free_ivars: free_ivars,
               inventionless_cost: inventionless_cost
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

pub fn associate_tasks(programs_root: Id, egraph: &EGraph, tasks: &Vec<String>) -> HashMap<Id, HashSet<String>> {
    let mut tasks_of_node = HashMap::new();
    let program_roots = egraph[programs_root].nodes[0].children();
    assert_eq!(program_roots.len(), tasks.len());
    for (program_root, task) in program_roots.iter().zip(tasks) {
        associate_task_rec(*program_root, egraph, task, &mut tasks_of_node)
    }
    tasks_of_node
}

fn associate_task_rec(node: Id, egraph: &EGraph, task: &String, tasks_of_node: &mut HashMap<Id, HashSet<String>>) {
    if !tasks_of_node.keys().contains(&node) {
        tasks_of_node.insert(node, HashSet::new());
    }
    let entry = tasks_of_node.get_mut(&node).unwrap();
    entry.insert(task.clone());
    for child in egraph[node].nodes[0].children() {
        associate_task_rec(*child, egraph, task, tasks_of_node);
    }
}