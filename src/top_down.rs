use crate::*;
use clap::Parser;
use itertools::Itertools;
use serde::Serialize;
use std::{collections::{VecDeque, BinaryHeap}, default, fmt::Display};
use ordered_float::NotNan;
use std::collections::HashMap;
use std::time::Duration;

// pub type PartialLambda = Option<Lambda>;
// pub type PartialExpr = Vec<PartialLambda>;

/// Top-down synthesis
#[derive(Parser, Debug, Serialize)]
#[clap(name = "Top-down synthesis")]
pub struct TopDownConfig {
    /// program to track
    #[clap(long)]
    pub t_track: Option<String>,
}



#[derive(Clone)]
pub struct Task<D: Domain> {
    name: String,
    tp: Type,
    ios: Vec<IO<D>>
}

impl<D:Domain> Task<D> {
    pub fn new(name: String, tp: Type, ios: Vec<IO<D>>) -> Task<D> {
        Task {name, tp, ios}
    }
}

#[derive(Clone)]
pub struct IO<D: Domain> {
    inputs: Vec<Val<D>>,
    output: Val<D>
}

impl<D:Domain> IO<D> {
    pub fn new(inputs: Vec<Val<D>>, output: Val<D>) -> IO<D> {
        IO { inputs, output }
    }
}








#[derive(Clone, Debug, Default)]
struct Stats {
    num_eval_ok: usize,
    num_eval_err: usize,
    num_processed: usize,
    num_finished: usize,
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
    root: Option<usize>, // root of the expression in `expr` or None if its a single hole
    ctx: Context, // typing context so far
    holes: Vec<Hole>, // holes so far
    prev_prod: Option<Lambda>, // previous production rule used, this is a Var | Prim or it's None if this is empty / the root
}

impl PartialExpr {
    pub fn new(expr: Vec<Lambda>, root: Option<usize>, ctx: Context, holes: Vec<Hole>) -> PartialExpr {
        PartialExpr { expr, root, ctx, holes, prev_prod: None }
    }
    pub fn single_hole(tp: Type, env: VecDeque<Type>) -> PartialExpr {
        PartialExpr::new(vec![], None, Context::empty(), vec![Hole::new(tp,env,SENTINEL)])
    }
}

impl Display for PartialExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // this expensive expr clone is silly to do lol
        write!(f, "{}", Expr::new(self.expr.clone()).to_string_uncurried(self.root.map(|x| Id::from(x))))
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
    fn expansion_unnormalized_ll(&self, prod: &Lambda, expr: &PartialExpr, hole_idx: usize) -> NotNan<f32>;

    fn likelihood(e: &Expr) -> NotNan<f32> {
        // todo implement this recursively making use of expansion_unnormalized_ll
        unimplemented!()
    }

}


// pub struct MaskPrimitives<M: ProbabilisticModel>
//  {
//     model: M,
//     masked: Vec<Symbol>,
// }

// impl<M: ProbabilisticModel> ProbabilisticModel for MaskPrimitives<M> {
//     fn expansion_unnormalized_ll(&self, prod: &Lambda, expr: &PartialExpr, hole_idx: usize) -> NotNan<f32> {
//         // mask out anything in self.masked to probability 0
//         if let Lambda::Prim(p) = prod {
//             if self.masked.contains(p) {
//                 return NotNan::new(f32::NEG_INFINITY).unwrap();
//             }
//         }
//         self.model.expansion_unnormalized_ll(prod, expr, hole_idx)
//     }
// }


// pub struct OverrideModel<M, F>
// where
//     M: ProbabilisticModel,
//     F: Fn(&Lambda, &PartialExpr, usize, NotNan<f32>) -> NotNan<f32>
// {
//     model: M,
//     f: F,
// }

// impl<M,F> ProbabilisticModel for OverrideModel<M,F>
// where
//     M: ProbabilisticModel,
//     F: Fn(&Lambda, &PartialExpr, usize, NotNan<f32>) -> NotNan<f32>
// {

//     fn expansion_unnormalized_ll(&self, prod: &Lambda, expr: &PartialExpr, hole_idx: usize) -> NotNan<f32> {
//         (self.f) (prod, expr, hole_idx, self.model.expansion_unnormalized_ll(prod, expr, hole_idx))
//     }
// }


/// This wraps a model to make it behave roughly like the DreamCoder enumerator, which when it detects a fixpoint operator
/// it give it 0% probability for using it lower in the program. Specifically what original DC does is
/// it forces the program to start with (lam (fix $0 ??)), then after the fact it may strip away that fix() operator if the function var
/// was never used in the body of the lambda.
/// For us fix_flip() is the DC style fixpoint operator, and we set fix() to P=0 as it's really just necessary internally to implement fix_flip().
/// In our case, we dont force it to start with a fix_flip() but instead let that just be the usual distribution for the toplevel operator,
/// but then if we ever try to expand into a fix_flip() and we're not at the top level then we set P=0 immediately.
/// Furthermore since the first argument of fix is always $0 we modify probabilities to force that too.
pub struct OrigamiModel<M: ProbabilisticModel> {
    model: M,
    fix_flip: Symbol,
    fix: Symbol
}

impl<M: ProbabilisticModel> OrigamiModel<M> {
    pub fn new(model: M, fix_flip: Symbol, fix: Symbol) -> Self {
        OrigamiModel { model, fix_flip, fix }
    }
}

impl<M: ProbabilisticModel> ProbabilisticModel for OrigamiModel<M> {
    fn expansion_unnormalized_ll(&self, prod: &Lambda, expr: &PartialExpr, hole_idx: usize) -> NotNan<f32> {
        // if this is not the very first expansion, and it's to a fix_flip() operator, then set the probability to 0
        if !expr.expr.is_empty() {
            if let Lambda::Prim(p) = prod {
                if *p == self.fix_flip {
                    return NotNan::new(f32::NEG_INFINITY).unwrap();
                }
            }
        }
        // if this is an expansion to the fix() operator, set it to 0
        if let Lambda::Prim(p) = prod {
            if *p == self.fix {
                return NotNan::new(f32::NEG_INFINITY).unwrap();
            }
        }
        // if we previously expanded with fix_flip(), then force next expansion (ie first argument) to be $0
        if let Some(Lambda::Prim(p)) = expr.prev_prod {
            if p == self.fix_flip {
                assert!(hole_idx == expr.holes.len() - 1); // we assume a left to right hole filling order for this to make sense (things were pushed on in opposite order hence we take the last hole), though you could change it
                if let Lambda::Var(0) = prod {
                    // doesnt really matter what we set this to as long as its not -inf, itll get normalized to ll=0 and P=1 since all other productions will be -inf
                    NotNan::new(-1.).unwrap();
                } else {
                    return NotNan::new(f32::NEG_INFINITY).unwrap();
                }
            }
        }
        // default
        self.model.expansion_unnormalized_ll(prod, expr, hole_idx)
    }
}


pub struct UniformModel {
    var_ll: NotNan<f32>,
    prim_ll: NotNan<f32>,
}

impl UniformModel {
    pub fn new(var_ll: NotNan<f32>, prim_ll: NotNan<f32>) -> UniformModel {
        UniformModel { var_ll, prim_ll }
    }
}

impl ProbabilisticModel for UniformModel {
    fn expansion_unnormalized_ll(&self, prod: &Lambda, expr: &PartialExpr, hole_idx: usize) -> NotNan<f32> {
        match prod {
            Lambda::Var(_) => self.var_ll,
            Lambda::Prim(_) => self.prim_ll,
            _ => unreachable!()
        }
    }
}


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
    // println!("hole type: {}", hole_tp);
    assert!(!hole.tp.is_arrow());
    // loop over all dsl entries and all variables in the env
    D::dsl_entries().map(|entry| (Lambda::Prim(entry.name), &entry.tp))
        .chain(hole.env.iter().enumerate().map(|(i,tp)| (Lambda::Var(i as i32),tp)))
        .filter_map(move |(prod, prod_tp)|
    {
        // println!("considering: {} :: {}", prod, prod_tp);
        // lightweight check for unification potential before doing the full clone
        if expr.ctx.might_unify(&hole_tp, prod_tp.return_type()) {
            // println!("passed might_unify");
            let mut new_expr: PartialExpr = expr.clone();
            new_expr.holes.remove(hole_idx);

            // instantiate if this wasnt a variable
            let prod_tp: Type = if let Lambda::Var(_) = prod {
                prod_tp.clone()
            } else {
                prod_tp.instantiate(&mut new_expr.ctx)
            };
            // println!("prod: {:?}", prod);
            // println!("hole parent:", )
            // full unification check
            if new_expr.ctx.unify(&hole_tp, prod_tp.return_type()).is_ok() {
                // println!("passed unify");
                // push on the new primitive or var
                new_expr.prev_prod = Some(prod.clone());
                new_expr.expr.push(prod);
                let mut expr_so_far_idx = new_expr.expr.len() - 1;
                let num_holes = prod_tp.arity();
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
                let len = new_expr.holes.len();
                new_expr.holes[len-num_holes..].reverse(); // reverse order of the ones we added
                if hole.parent != SENTINEL {
                    fill_sentinel(&mut new_expr.expr[hole.parent], expr_so_far_idx);
                } else {
                    // filling the single_hole so we can set our root
                    assert!(new_expr.root.is_none());
                    new_expr.root = Some(expr_so_far_idx);
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



// pub fn top_down<D: Domain, M: ProbabilisticModel>(
//     model: M,
//     all_tasks: &[Task<D>],
//     cfg: &TopDownConfig,
// ) {

//     println!("DSL:");
//     for entry in D::dsl_entries() {
//         println!("\t{} :: {}", entry.name, entry.tp);
//     }

//     let mut ll_record = NotNan::new(0.).unwrap();

//     let task_tps: HashMap<Type,Vec<Task<D>>> = all_tasks.iter().map(|task| (task.tp.clone(), task.clone())).into_group_map();
    

//     // assert!(tasks.iter().all(|task| task.tp == tp));

//     // for task in tasks {
//     //     assert_eq!(tp, task.tp.arity());
//     //     // let mut ctx = Context::empty();
//     //     // let tp_task = task.tp.instantiate(&mut ctx);
//     //     // let tp_overall = tp.instantiate(&mut ctx);
//     //     // ctx.unify(
//     //     //     &tp_task,
//     //     //     &tp_overall
//     //     // ).unwrap();
//     // }

//     for (tp, tasks) in task_tps.iter() {
//         println!("Searching for {tp} solutions:");
//         for task in tasks {
//             println!("\t{}", task.name)
//         }

//         let mut stats = Stats::default();

//         // if we want to wrap this in some lambdas and return it, then the outermost lambda should be the first type in
//         // the list of arg types. This will be the *largest* de bruijn index within the body of the program, therefore
//         // we should reverse the 
//         let mut env: VecDeque<Type> = tp.iter_args().cloned().collect();
//         env.make_contiguous().reverse();

//         let mut worklist: BinaryHeap<WorklistItem> = Default::default();
//         let mut worklist_buf: Vec<WorklistItem> = vec![];
//         let mut expansions_buf: Vec<(NotNan<f32>, PartialExpr)> = vec![];
//         let mut solved_buf: Vec<(NotNan<f32>, String, PartialExpr)> = vec![];
//         worklist.push(WorklistItem::new(NotNan::new(0.).unwrap(), PartialExpr::single_hole(tp.return_type().clone(), env)));

//         loop {

//             worklist.extend(worklist_buf.drain(..));

//             let item = match worklist.pop() {
//                 Some(item) => item,
//                 None => break,
//             };

//             if let Some(track) = &cfg.t_track {
//                 if !track.starts_with(item.expr.to_string().split("??").next().unwrap()) {
//                     continue;
//                 }
//             } 

//             if item.ll.trunc() < *ll_record {
//                 ll_record = NotNan::new(item.ll.trunc()).unwrap();
//                 println!("enumerated all programs under this ll: {} ({} wips processed; {} finished; worklist_size={})", item.ll, stats.num_processed, stats.num_finished, worklist.len());
//             }

//             // println!("{}: {} (ll={}; P={})", "expanding".yellow(), item.expr, item.ll, item.ll.exp());

//             let mut unnormalized_ll_total = NotNan::new(f32::NEG_INFINITY).unwrap(); // start as ll=-inf -> P=0

//             let hole_idx = item.expr.holes.len() - 1;
//             stats.num_processed += 1;

//             for expanded in expansions::<D>(&item.expr, hole_idx) {
//                 // println!("new expansion: {}", expanded);

//                 let unnormalized_ll = model.expansion_unnormalized_ll(expanded.prev_prod.as_ref().unwrap(), &item.expr, hole_idx);
//                 unnormalized_ll_total = logsumexp(unnormalized_ll_total, unnormalized_ll);
//                 if unnormalized_ll_total == f32::NEG_INFINITY {
//                     continue; // we skip adding -infs to the worklist entirely
//                 }

//                 if expanded.holes.is_empty() {
//                     // new completed program
//                     // todo run the program, see if it works, discard if not or keep if yes
//                     // let expr: Expr = "(fix_flip $0 (lam (lam (if (is_empty $0) 0 (+ ($1 (tail $0)) 1)))))".parse().unwrap();
//                     let expr = Expr::new(expanded.expr.clone());
//                     let mut exec = Executable::<D>::from(expr);
//                     stats.num_finished += 1;

//                     for task in tasks {
//                         let mut solved = true;
//                         for io in task.ios.iter() {
//                             // probably excessively much cloning and such here lol
//                             let mut exec_env: Vec<LazyVal<D>> = io.inputs.iter().map(|v| LazyVal::new_strict(v.clone())).collect();
//                             exec_env.reverse(); // for proper arg order

//                             // println!("about to exec");

//                             exec.set_timeout(Duration::from_millis(100));
//                             if let Ok(res) = exec.eval_child(Id::from(expanded.root.unwrap()), &mut exec_env.clone()) {
//                             // if let Ok(res) = exec.eval_child(exec.expr.root(),&mut exec_env.clone()) {
//                                 // println!("done");
//                                     stats.num_eval_ok += 1;

//                                 if res == io.output {
//                                     // println!("{} {} {:?}", expanded, "=>".green(), res);
//                                 } else {
//                                     // println!("{} {} {:?}", expanded, "=>".yellow(), res);
//                                     solved = false;
//                                     break
//                                 }

//                             } else {
//                                 // println!("done");

//                                 // println!("{} {} err", "=>".red(), expanded);
//                                 stats.num_eval_err += 1;
//                                 solved = false;
//                                 break
//                             }
//                         } 
//                         if solved {
//                             solved_buf.push((unnormalized_ll, task.name.clone(), expanded.clone()));
//                         }
//                     }
                    
//                 } else {
//                     // new partial program
//                     expansions_buf.push((unnormalized_ll, expanded));
//                 }
//                 // panic!("done")
//             }
//             // normalize the log likelihoods, calculate total log likelihood
//             worklist_buf.extend(expansions_buf.drain(..).map(|(unnormalized_ll,expanded)| {
//                 // normalize the ll
//                 let ll = item.ll + (unnormalized_ll - unnormalized_ll_total);
//                 // extend prods and holes
//                 // println!("{} ll={}", expanded, ll);
//                 WorklistItem::new(ll, expanded)
//             }));

//             for (unnormalized_ll, task_name, expanded) in solved_buf.iter() {
//                 // normalize the ll
//                 let ll = item.ll + (unnormalized_ll - unnormalized_ll_total);
//                 println!("{} {} [ll={}]: {}", "Solved".green(), task_name, ll, expanded);
//             }
//             solved_buf.clear();

//             // if stats.num_expansions >= 40 {
//             //     break
//             // }
//         }


//     }

// }


pub fn top_down_inplace<D: Domain, M: ProbabilisticModel>(
    model: M,
    all_tasks: &[Task<D>],
    cfg: &TopDownConfig,
) {

    println!("DSL:");
    for entry in D::dsl_entries() {
        println!("\t{} :: {}", entry.name, entry.tp);
    }

    let mut stats = Stats::default();

    let budget_decr = NotNan::new(1.5).unwrap();
    let mut upper_bound = NotNan::new(0.).unwrap();
    let mut lower_bound = upper_bound - budget_decr;

    let task_tps: HashMap<Type,Vec<Task<D>>> = all_tasks.iter().map(|task| (task.tp.clone(), task.clone())).into_group_map();

    loop {
        for (tp, tasks) in task_tps.iter() {
            println!("{:?}", stats);
            println!("Searching for {tp} solutions in range {lower_bound} <= ll <= {upper_bound}:");
            for task in tasks {
                println!("\t{}", task.name)
            }

            // if we want to wrap this in some lambdas and return it, then the outermost lambda should be the first type in
            // the list of arg types. This will be the *largest* de bruijn index within the body of the program, therefore
            // we should reverse the 
            let mut env: VecDeque<Type> = tp.iter_args().cloned().collect();
            env.make_contiguous().reverse();

            let mut worklist: Vec<WorklistItem> = Default::default();
            let mut worklist_buf: Vec<WorklistItem> = vec![];
            let mut expansions_buf: Vec<(NotNan<f32>, PartialExpr)> = vec![];
            let mut solved_buf: Vec<(NotNan<f32>, String, PartialExpr)> = vec![];
            worklist.push(WorklistItem::new(NotNan::new(0.).unwrap(), PartialExpr::single_hole(tp.return_type().clone(), env.clone())));

            loop {

                worklist.extend(worklist_buf.drain(..));

                let item = match worklist.pop() {
                    Some(item) => item,
                    None => break,
                };

                if item.ll <= lower_bound {
                    continue; 
                }

                if let Some(track) = &cfg.t_track {
                    if !track.starts_with(item.expr.to_string().split("??").next().unwrap()) {
                        continue;
                    }
                }

                // println!("{}: {} (ll={}; P={})", "expanding".yellow(), item.expr, item.ll, item.ll.exp());
                // println!("holes: {:?}", item.expr.holes);
                // println!("ctx: {:?}", item.expr.ctx);

                let mut unnormalized_ll_total = NotNan::new(f32::NEG_INFINITY).unwrap(); // start as ll=-inf -> P=0

                let hole_idx = item.expr.holes.len() - 1;
                stats.num_processed += 1;

                for expanded in expansions::<D>(&item.expr, hole_idx) {
                    // println!("new expansion: {}", expanded);

                    let unnormalized_ll = model.expansion_unnormalized_ll(expanded.prev_prod.as_ref().unwrap(), &item.expr, hole_idx);
                    unnormalized_ll_total = logsumexp(unnormalized_ll_total, unnormalized_ll);
                    if unnormalized_ll_total == f32::NEG_INFINITY {
                        continue; // we skip adding -infs to the worklist entirely
                    }

                    if expanded.holes.is_empty() {

                        // new completed program

                        if item.ll > upper_bound {
                            continue; 
                        }

                        // run the program, see if it works, discard if not or keep if yes
                        let expr = Expr::new(expanded.expr.clone());
                        // println!("{}", expanded);
                        // println!("{}", expanded.ctx);

                        // check for type errors:
                        expr.infer::<D>(Some(Id::from(expanded.root.unwrap())), &mut Context::empty(), &mut (env.clone())).unwrap();
                        let mut exec = Executable::<D>::from(expr);
                        stats.num_finished += 1;

                        for task in tasks {
                            let mut solved = true;
                            for io in task.ios.iter() {
                                // probably excessively much cloning and such here lol
                                let mut exec_env: Vec<LazyVal<D>> = io.inputs.iter().map(|v| LazyVal::new_strict(v.clone())).collect();
                                exec_env.reverse(); // for proper arg order

                                // println!("about to exec");

                                exec.set_timeout(Duration::from_millis(100));
                                if let Ok(res) = exec.eval_child(Id::from(expanded.root.unwrap()), &mut exec_env.clone()) {
                                // if let Ok(res) = exec.eval_child(exec.expr.root(),&mut exec_env.clone()) {
                                    // println!("done");
                                        stats.num_eval_ok += 1;

                                    if res == io.output {
                                        // println!("{} {} {:?}", expanded, "=>".green(), res);
                                    } else {
                                        // println!("{} {} {:?}", expanded, "=>".yellow(), res);
                                        solved = false;
                                        break
                                    }

                                } else {
                                    // println!("done");

                                    // println!("{} {} err", "=>".red(), expanded);
                                    stats.num_eval_err += 1;
                                    solved = false;
                                    break
                                }
                            } 
                            if solved {
                                solved_buf.push((unnormalized_ll, task.name.clone(), expanded.clone()));
                            }
                        }
                        
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
                    // println!("{} ll={}", expanded, ll);
                    WorklistItem::new(ll, expanded)
                }));

                for (unnormalized_ll, task_name, expanded) in solved_buf.iter() {
                    // normalize the ll
                    let ll = item.ll + (unnormalized_ll - unnormalized_ll_total);
                    println!("{} {} [ll={}]: {}", "Solved".green(), task_name, ll, expanded);
                    panic!("done");
                }
                solved_buf.clear();
            }


        }

        lower_bound -= budget_decr;
        upper_bound -= budget_decr;
    }

}




/// numerically stable streaming logsumexp (base 2)
/// equivalent to log(exp(x) + exp(y))
/// same as ADDING probabilities in normal probability space
#[inline(always)]
fn logsumexp(x: NotNan<f32>, y: NotNan<f32>) -> NotNan<f32> {
    if x.is_infinite() { return y }
    if y.is_infinite() { return x }
    let big = std::cmp::max(x,y);
    let smol = std::cmp::min(x,y);
    big + (1. + (smol - big).exp()).ln()
}
