use crate::*;
use clap::Parser;
use serde::Serialize;
use std::{collections::{VecDeque, BinaryHeap}, default};
use ordered_float::NotNan;

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


#[derive(Clone, Debug, Default)]
struct Stats {
    num_eval_ok: usize,
    num_eval_err: usize,
    num_expansions: usize,
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
    pub ll: NotNan<f32>,
    pub expr: PartialExpr,
}

impl Eq for WorklistItem {}

#[derive(Debug,Clone, PartialEq, Eq)]
pub struct PartialExpr {
    expr: Vec<Lambda>, // expr
    ctx: Context, // typing context so far
    holes: Vec<Hole>, // holes so far
    prev_prod: Option<Lambda>, // previous production rule used, this is a Var | Prim or it's None if this is empty / the root
}

impl PartialExpr {
    fn new(expr: Vec<Lambda>, ctx: Context, holes: Vec<Hole>) -> PartialExpr {
        PartialExpr { expr, ctx, holes, prev_prod: None }
    }
    pub fn single_hole(tp: Type, env: VecDeque<Type>) -> PartialExpr {
        PartialExpr::new(vec![], Context::empty(), vec![Hole::new(tp,env,0)])
    }
}


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
        self.ll.partial_cmp(&other.ll).unwrap() // compares by ll but NaN != NaN
    }
}

impl WorklistItem {
    pub fn new(ll: NotNan<f32>, expr: PartialExpr) -> WorklistItem {
        WorklistItem { ll, expr }
    }
}


// pub struct Expansion {
//     prod: Vec<Lambda>, // must be Prim(Symbol) | Var(i32)
//     ctx: Context,
//     holes: Vec<Hole>,
// }

// impl Expansion {
//     fn new(prod: Vec<Lambda>, ctx: Context, holes: Vec<Hole>) -> Expansion {
//         Expansion {prod, ctx, holes}
//     }
// }

#[derive(Debug,Clone, PartialEq, Eq)]
pub struct Hole {
    tp: Type,
    env: VecDeque<Type>, // env[i] is $i
    parent: usize, // parent of the hole - either the hole is the child of a lam or the right side of an app
}

impl Hole {
    fn new(tp: Type, env: VecDeque<Type>, parent: usize) -> Hole {
        Hole {tp, env, parent}
    }
}


pub trait ProbabilisticModel {
    fn expansion_unnormalized_ll(&self, prod: &Lambda) -> NotNan<f32>;

    fn likelihood(e: &Expr) -> NotNan<f32> {
        // todo implement this recursively making use of expansion_unnormalized_ll
        unimplemented!()
    }

}

pub struct UniformModel {
    var_ll: NotNan<f32>,
    prim_ll: NotNan<f32>,
}

impl ProbabilisticModel for UniformModel {
    fn expansion_unnormalized_ll(&self, prod: &Lambda) -> NotNan<f32> {
        match prod {
            Lambda::Var(_) => self.var_ll,
            Lambda::Prim(_) => self.prim_ll,
            _ => unreachable!()
        }
    }
}

const SENTINEL: usize = usize::MAX;

#[inline]
fn fill_sentinel(node: &mut Lambda, id: usize) {
    match node {
        Lambda::App([f,x]) => {
            assert_eq!(usize::from(*x), SENTINEL);
            *x = Id::from(id);
        },
        Lambda::Lam([b]) => {
            assert_eq!(usize::from(*b), SENTINEL); 
            *b = Id::from(id); 
        },
        _ => unreachable!()
    }
}


/// returns an iterator over all possible partialexprs obtained by expanding `hole_idx` in `expr`.
pub fn expansions<D: Domain>(expr: &PartialExpr, hole_idx: usize) -> impl Iterator<Item=PartialExpr> + '_ {
    // let mut expr: PartialExpr = expr.clone();
    let hole: &Hole  = &expr.holes[hole_idx];
    // let env = hole.env.clone();
    let hole_tp = hole.tp.apply_immut(&expr.ctx); 
    assert!(!hole.tp.is_arrow());
    // loop over all dsl entries and all variables in the env
    D::dsl_entries().map(|entry| (Lambda::Prim(entry.name), &entry.tp))
        .chain(hole.env.iter().enumerate().map(|(i,tp)| (Lambda::Var(i as i32),tp)))
        .filter_map(move |(prod, prod_tp)|
    {
        // lightweight check for unification potential before doing the full clone
        if expr.ctx.might_unify(&hole_tp, prod_tp.return_type()) {
            let mut new_expr: PartialExpr = expr.clone();
            new_expr.holes.remove(hole_idx);
            let prod_tp: Type = prod_tp.instantiate(&mut new_expr.ctx);
            // full unififcation check
            if new_expr.ctx.unify(&hole_tp, prod_tp.return_type()).is_ok() {
                // push on the new primitive or var
                new_expr.prev_prod = Some(prod.clone());
                new_expr.expr.push(prod);
                let mut expr_so_far_idx = new_expr.expr.len() - 1;
                // add a new hole for each arg, along with any apps and lams
                for arg_tp in prod_tp.iter_args() {
                    // push on an app
                    new_expr.expr.push(Lambda::App([expr_so_far_idx.into(), SENTINEL.into()]));
                    expr_so_far_idx = new_expr.expr.len() - 1;

                    // if this arg is higher order it may have arguments - we push those types onto our new env and push lambdas
                    // into our expr
                    let mut new_hole_env = hole.env.clone();
                    for inner_arg_tp in arg_tp.iter_args().cloned() {
                        new_hole_env.push_front(inner_arg_tp);
                        new_expr.expr.push(Lambda::Lam([SENTINEL.into()]));
                        // adjust pointers so the previous node points to the new node we created
                        let last = new_expr.expr.len() - 1;
                        fill_sentinel(&mut new_expr.expr[last - 1], last);
                    }

                    // the hole type is the return type of the arg (bc all lambdas were autofilled)
                    let new_hole_tp = arg_tp.return_type().clone();
                    new_expr.holes.push(Hole::new(new_hole_tp, new_hole_env, new_expr.expr.len() - 1))
                }
                return Some(new_expr)
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



pub fn top_down<D: Domain, M: ProbabilisticModel>(
    model: M,
    tp: Type,
    // env: VecDeque<Type>,
    cfg: &TopDownConfig,
) {

    let mut stats = Stats::default();

    let exec_env = vec![];
    let env: VecDeque<Type> = Default::default();

    let mut worklist: BinaryHeap<WorklistItem> = Default::default();
    let mut worklist_buf: Vec<WorklistItem> = vec![];
    let mut expansions_buf: Vec<(NotNan<f32>, PartialExpr)> = vec![];
    worklist.push(WorklistItem::new(NotNan::new(0.).unwrap(), PartialExpr::single_hole(tp, env)));

    loop {

        let item = match worklist.pop() {
            Some(item) => item,
            None => break,
        };

        let mut unnormalized_ll_total = NotNan::new(f32::NEG_INFINITY).unwrap(); // start as ll=-inf -> P=0

        for expanded in expansions::<D>(&item.expr, item.expr.holes.len() - 1) {
            stats.num_expansions += 1;
            let unnormalized_ll = model.expansion_unnormalized_ll(expanded.prev_prod.as_ref().unwrap());
            unnormalized_ll_total = logsumexp(unnormalized_ll_total, unnormalized_ll);
            if expanded.holes.is_empty() {
                // new completed program
                // todo run the program, see if it works, discard if not or keep if yes
                if let Ok(res) = Executable::<D>::from(Expr::new(expanded.expr)).eval(&mut exec_env.clone()) {
                    stats.num_eval_ok += 1;
                } else {
                    stats.num_eval_err += 1;
                }
                todo!()
            } else {
                // new partial program
                expansions_buf.push((unnormalized_ll, expanded));
            }
        }
        // normalize the log likelihoods, calculate total log likelihood
        worklist_buf.extend(expansions_buf.drain(..).map(|(unnormalized_ll,expanded)| {
            // normalize the ll
            let ll = item.ll + (unnormalized_ll - unnormalized_ll_total);
            // extend prods and holes
            WorklistItem::new(ll, expanded)
        }));

    }

}


/// numerically stable streaming logsumexp (base 2)
/// equivalent to log(exp(x) + exp(y))
/// same as ADDING probabilities in normal probability space
#[inline(always)]
fn logsumexp(x: NotNan<f32>, y: NotNan<f32>) -> NotNan<f32> {
    let big = std::cmp::max(x,y);
    let smol = std::cmp::min(x,y);
    big + (1. + (smol - big).exp2()).log2()
}
