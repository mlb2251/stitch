use crate::*;
use clap::Parser;
use serde::Serialize;
use std::collections::{VecDeque, BinaryHeap};


// pub type PartialLambda = Option<Lambda>;
// pub type PartialExpr = Vec<PartialLambda>;

/// Top-down synthesis
#[derive(Parser, Debug, Serialize)]
#[clap(name = "Top-down synthesis")]
pub struct TopDownConfig {
    /// Max cost to enumerate to
    // #[clap(short = 'c', long, default_value = "10")]
    // pub max_cost: usize,

    /// print all exprs found at end
    #[clap(long)]
    pub print_found: bool,
}


// #[derive(Debug,Clone, Eq, PartialEq)]
// struct PartialExprHeavy {
//     cost: f32,
//     expr: Vec<Lambda>,
//     next_hole_parent: usize, // ptr into expr to the next app or lam that needs a righthand child
//     ctx: Context,
// }

// #[derive(Debug,Clone, Eq, PartialEq)]
// struct PartialExprLight {
//     cost: f32,
//     parent: usize,
//     holes: Vec<HoleEntry>,
// }

#[derive(Debug,Clone, PartialEq)]
pub struct WorklistItem {
    ll: f32,
    prods: Vec<Lambda>, // Vec of Prims and Vars only
    ctx: Context,
    holes: Vec<Hole>
}

impl Eq for WorklistItem {}


// #[derive(Debug,Clone, Eq, PartialEq)]
// struct HoleEntry {
//     parent: usize, // And since its a parent of ours, any ctx vars it refers to will still be in our ctx so we all good.
//     tp: usize, // this is just gonna be the typevar we can use to look up the real type in the context - important so we get the most updated version
// }



// partialord and ord for the binaryheap
impl PartialOrd for WorklistItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.ll.partial_cmp(&other.ll) // compares by ll
    }
}

impl Ord for WorklistItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.ll.partial_cmp(&other.ll).unwrap() // compares by ll and panics on mismatch
    }
}

impl WorklistItem {
    pub fn single_hole(tp: Type, env: VecDeque<Type>) -> WorklistItem {
        WorklistItem {
            ll: 0., // ll=0 -> P=1, then as we expand in new nodes we'll add their lls (multiplying the Ps)
            prods: vec![],
            ctx: Context::empty(),
            holes: vec![Hole::new(tp,env)],
        }
    }
}


pub struct Expansion {
    prod: Vec<Lambda>, // must be Prim(Symbol) | Var(i32)
    ctx: Context,
    holes: Vec<Hole>,
}

impl Expansion {
    fn new(prod: Vec<Lambda>, ctx: Context, holes: Vec<Hole>) -> Expansion {
        Expansion {prod, ctx, holes}
    }
}

#[derive(Debug,Clone, PartialEq, Eq)]
pub struct Hole {
    tp: Type,
    env: VecDeque<Type> // env[i] is $i
}

impl Hole {
    fn new(tp: Type, env: VecDeque<Type>) -> Hole {
        Hole {tp, env}
    }
}


pub trait ProbabilisticModel {
    fn expansion_unnormalized_ll(&self, prod: &Lambda) -> f32;

    fn likelihood(e: &Expr) -> f32 {
        // todo implement this recursively making use of expansion_unnormalized_ll
        unimplemented!()
    }

}

pub struct UniformModel {
    var_ll: f32,
    prim_ll: f32,
}

impl ProbabilisticModel for UniformModel {
    fn expansion_unnormalized_ll(&self, prod: &Lambda) -> f32 {
        match prod {
            Lambda::Var(_) => self.var_ll,
            Lambda::Prim(_) => self.prim_ll,
            _ => unreachable!()
        }
    }
}


pub fn expansions<D: Domain>(hole: &Hole, ctx: &Context) -> impl Iterator<Item=Expansion> + '_ {
    let hole_tp = hole.tp.apply_immut(ctx); // todo cut this? do it outside?
    assert!(!hole.tp.is_arrow());
    // loop over all dsl entries and all variables in the env
    D::dsl_entries().map(|entry| (Lambda::Prim(entry.name), &entry.tp))
        .chain(hole.env.iter().enumerate().map(|(i,tp)| (Lambda::Var(i as i32),tp)))
        .filter_map(|(prod, prod_tp)|
    {
            if ctx.might_unify(&hole_tp, prod_tp.return_type()) {
                let mut ctx = ctx.clone();
                let mut holes: Vec<Hole> = vec![];
                let prod_tp = prod_tp.instantiate(&mut ctx);
                if ctx.unify(&hole_tp, prod_tp.return_type()).is_ok() {
                    // add a new hole for each arg
                    for arg_tp in prod_tp.iter_args() {
                        // the hole type is the return type of the arg (bc all lambdas will be autofilled)
                        let new_hole_tp = arg_tp.return_type().clone();
                        // if this arg is higher order it may have arguments - we push those onto our new env 
                        let mut new_hole_env = hole.env.clone();
                        for inner_arg_tp in arg_tp.iter_args().cloned() {
                            new_hole_env.push_front(inner_arg_tp);
                        }
                        holes.push(Hole::new(new_hole_tp, new_hole_env))
                    }
                    return Some(Expansion::new(prod, ctx, holes))
                }
            }
            None
    })
}


// impl Expr {
//     // todo can optimize by having it take the Expr or Executable in as input, clear the nodes/cache/etc so
//     // you can reuse the same allocation.
//     pub fn from_prods<D: Domain>(prods: Vec<Lambda>, env: VecDeque<Type>) -> Expr {
//         // traverse right to left in bottom up order. Assumes that "prods" is a list of productions where you
//         // always expand the leftmost hole, so reading it backwards will give a particular right-child-first traversal
//         let mut no_parent = vec![];
//         for prod in prods.into_iter().rev() {
//             match prod {
//                 Lambda::Prim(p) =>
//                 Lambda::Var(i) => 
//             }
//         }
//     }
// }



pub fn top_down<D: Domain>(
    constants: &[(Expr,usize)],
    fns: &[(DSLEntry<D>,usize)],
    cfg: &TopDownConfig,
) {

    let tp = unimplemented!();
    let env = unimplemented!();

    let mut worklist: BinaryHeap<WorklistItem> = Default::default();
    let mut worklist_buf: Vec<WorklistItem> = vec![];
    let mut expansions_buf: Vec<(f32, Expansion)> = vec![];
    worklist.push(WorklistItem::single_hole(tp, env));

    // an expr we use as a scratchspace
    let mut expr = Expr::new(vec![]);


    loop {

        let item = match worklist.pop() {
            Some(item) => item,
            None => break,
        };


        let hole: Hole = item.holes.pop();
        let mut unnormalized_ll_total: f32::NEG_INFINITY; // start as ll=-inf -> P=0

        for expansion in expansions(&hole, &item.ctx) {
            unnormalized_ll_total = logsumexp(unnormalized_ll_total, expansion.unnormalized_ll);
            if item.holes.is_empty() && expansion.holes.is_empty() {
                // new completed program
                // todo run the program, see if it works, discard if not or keep if yes
                todo!()
            } else {
                // new partial program
                expansions_buf.push(expansion);
            }
        }
        // normalize the log likelihoods, calculate total log likelihood
        worklist_buf.extend(expansions_buf.drain(..).map(|(unnormalized_ll,expansion)| {
            // normalize the ll
            let ll = item.ll + (unnormalized_ll - unnormalized_ll_total);
            // extend prods and holes
            let mut prods = item.prods.clone();
            prods.push(expansion.prod.clone());
            let mut holes = item.holes.clone();
            holes.extend(expansion.holes.clone().into_iter().rev()); // todo i think rev() is right?
            WorklistItem::new(ll, item, holes, expansion.ctx.clone())
        }));

    }

}


/// numerically stable streaming logsumexp (base 2)
/// equivalent to log(exp(x) + exp(y))
/// same as ADDING probabilities in normal probability space
#[inline(always)]
fn logsumexp(x:f32, y: f32) -> f32 {
    let big = std::cmp::max(x,y);
    let smol = std::cmp::min(x,y);
    big + (1. + (smol - big).exp2()).log2()
}
