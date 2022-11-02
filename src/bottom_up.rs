
use crate::*;
use ahash::{AHashMap};
use std::{time::Instant};
use clap::{Parser};
use serde::Serialize;
use itertools::Itertools;
// use serde_json::json;


/// Bottom-up synthesis
#[derive(Parser, Debug, Serialize)]
#[clap(name = "Bottom-up synthesis")]
pub struct BottomUpConfig {
    /// How big of a step to increase cost by for each round of bottom up
    #[clap(short, long, default_value = "1")]
    pub cost_step: usize,

    /// Max cost to enumerate to
    #[clap(short = 'c', long, default_value = "10")]
    pub max_cost: usize,

    /// print all exprs found at end
    #[clap(long)]
    pub print_found: bool,
}

#[derive(Clone)]
pub struct Found<D: Domain> {
    val: Val<D>, // value that was found
    id: Id, // expr that constructs it
    cost: usize, // cost of constructing it
}

#[derive(Clone)]
pub struct FoundExpr<D: Domain> {
    val: Val<D>, // value that was found
    expr: Expr, // expr that constructs it
    cost: usize, // cost of constructing it
}

impl <D: Domain> Found<D> {
    fn new(val: Val<D>, id: Id, cost: usize) -> Self {
        Found {
            val,
            id,
            cost,
        }
    }
}

impl <D: Domain> FoundExpr<D> {
    pub fn new(val: Val<D>, expr: Expr, cost: usize) -> Self {
        FoundExpr {
            val,
            expr,
            cost,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Stats {
    num_eval_ok: usize,
    num_eval_err: usize,
    num_not_seen: usize,
    num_yes_seen: usize,
    num_yes_seen_and_was_better: usize,
}

pub fn bottom_up<D: Domain>(
    // handle: &mut Evaluator<D>,
    initial: &[FoundExpr<D>],
    fns: &[(DSLEntry<D>,usize)],
    cfg: &BottomUpConfig,
) {

    let fns: Vec<(DSLEntry<D>,usize)>  = fns.iter().filter(|(entry, _)| entry.arity > 0).cloned().collect();

    let tstart = Instant::now();
    let mut stats: Stats = Default::default();

    let mut curr_cost = cfg.cost_step;
    let mut vals_of_type: AHashMap<Type,Vec<Found<D>>> = Default::default();

    // add each dsl fn so the ith dsl fn is expr.nodes[i] // todo programs() is a gross way to do this
    let mut handle: Expr = {
        let dsl_fns_expr: Expr = Expr::programs(fns.iter().map(|(entry,_)| Expr::prim(entry.name)).collect());
        let init_vals_expr: Expr = Expr::programs(initial.iter().map(|found_expr| found_expr.expr.clone()).collect());    
        Expr::programs(vec![dsl_fns_expr,init_vals_expr])
    };

    let mut seen: AHashMap<Val<D>,usize> = AHashMap::new();

    // ids for the exprs passed in as `initial`
    let init_val_ids: Vec<Id> = handle.get(handle.get_root().children()[1]).children().to_vec();

    println!("Productions:");
    for (f,cost) in fns.iter() {
        println!("(cost {}) {} :: {}", cost, f.name, f.tp);
    }

    println!("Initial:");
    for (i,found_expr) in initial.iter().enumerate() {
        let id = init_val_ids[i]; // ith child of programs node
        let found = Found::new(found_expr.val.clone(), id, found_expr.cost);
        let tp = found_expr.expr.infer::<D>(None, &mut Context::empty(), &mut Default::default()).unwrap();

        println!("(cost {}) {} :: {} => {:?}", found.cost, handle.to_string_uncurried(Some(found.id)), tp, found.val);

        vals_of_type.entry(tp).or_default().push(found.clone());
        seen.insert(found.val.clone(), found.cost);
    }



    while curr_cost < cfg.max_cost {
        // sort by the cost
        vals_of_type.values_mut().for_each(|vals| {
            vals.sort_by(|a,b| a.cost.cmp(&b.cost));
            // vals.dedup_by(|a,b| a.val == b.val);
        });

        let seen_types: Vec<Type> = vals_of_type.keys().cloned().collect();

        println!("new curr cost: {}", curr_cost);
        let mut new_vals_of_type: AHashMap<Type,Vec<Found<D>>> = Default::default();


        for (i_fn, (dsl_entry, fn_cost)) in fns.iter().enumerate() {
            // println!("trying fn: {}", dsl_entry.name);

            for (found_args, tp, cost) in ArgChoiceIterator::new(&vals_of_type, &seen_types, &dsl_entry.tp, *fn_cost, curr_cost, curr_cost - cfg.cost_step) {
                let args: Vec<LazyVal<D>> = found_args.iter().map(|&f| LazyVal::new_strict(f.val.clone())).collect();
                // println!("trying ({} {})", dsl_entry.name, found_cfg.iter().map(|arg| format!("{:?}",arg.val)).collect::<Vec<_>>().join(" "));
                if let Ok(val) = (D::lookup_fn_ptr(dsl_entry.name)) (args, &mut handle.as_eval(None)) {
                    stats.num_eval_ok += 1;
                    match seen.get(&val) {
                        None => {
                            stats.num_not_seen += 1;
                            let mut id = Id::from(i_fn); // assumes we constructed the ith fn primitive to be the ith element in handle.expr.nodes
                            for arg in found_args.iter() {
                                handle.nodes.push(Lambda::App([id,arg.id]));
                                id = Id::from(handle.nodes.len()-1);
                            }
                            new_vals_of_type.entry(tp).or_default().push(Found::new(val, id, cost));
                        }
                        Some(&old_cost) => {
                            stats.num_yes_seen += 1;
                            if old_cost > cost {
                                let mut id = Id::from(i_fn); // assumes we constructed the ith fn primitive to be the ith element in handle.expr.nodes
                                for arg in found_args.iter() {
                                    handle.nodes.push(Lambda::App([id,arg.id]));
                                    id = Id::from(handle.nodes.len()-1);
                                }
                                new_vals_of_type.entry(tp).or_default().push(Found::new(val, id, cost));
        
                            } else {
                                stats.num_yes_seen_and_was_better += 1;
                            }
                        }
                    }

                } else {
                    // Err from execution, discard
                    stats.num_eval_err += 1;
                }
            }
        }

        // deposit new vals into vals_of_type
        for (tp, new_vals) in new_vals_of_type.into_iter() {
            for found in new_vals.into_iter() {
                match seen.get(&found.val) {
                    None => {
                        seen.insert(found.val.clone(),found.cost);
                        vals_of_type.entry(tp.clone()).or_default().push(found.clone());
                        if cfg.print_found{
                            println!("(cost {}) {} :: {} => {:?}", found.cost, handle.to_string_uncurried(Some(found.id)), tp, found.val);
                        }
                    }
                    Some(&old_cost) => {
                        if old_cost > found.cost {
                            *seen.get_mut(&found.val).unwrap() = found.cost;
                            // removes old value
                            // todo this is prob v slow as implemented, could do faster with a binary search by cost or something which I guess works since we do assume vals_of_type is sorted by cost
                            // HOWEVER that invariant gets broken during this process so we should actually switch to doing a binary insertion if we do this.
                            // vals_of_type.get_mut(tp).unwrap().partition_point(|found| found.cost)
                            vals_of_type.get_mut(&tp).unwrap().retain(|f| f.val != found.val);

                            // add new value
                            vals_of_type.entry(tp.clone()).or_default().push(found.clone());
                            if cfg.print_found{
                                println!("(cost {}) {} :: {:?} -> {:?}", found.cost, handle.to_string_uncurried(Some(found.id)), tp, found.val);
                            }  
                        }
                    }
                }
            }
        }



        curr_cost += cfg.cost_step;
    }

    //todo add a sanity check that the length of seen equals the lengths of all val arrays. i bet theres an error and that wont be true lol
    println!("reached max cost");
    println!("Time: {}ms",tstart.elapsed().as_millis());
    println!("{:?}",stats);
    println!("num found: {}",seen.len());
    println!("num found per ms: {:.2}", seen.len() as f64 / tstart.elapsed().as_millis() as f64);
    println!("num eval total: {}",stats.num_eval_ok+stats.num_eval_err);
    println!("% eval ok: {:.2}%", stats.num_eval_ok as f64 / (stats.num_eval_ok + stats.num_eval_err) as f64 * 100.0);
    println!("num eval per ms: {:.2}",(stats.num_eval_ok+stats.num_eval_err) as f64 / tstart.elapsed().as_millis() as f64);
    println!("num found by type:\n\t{}", vals_of_type.iter().map(|(ty,vals)| format!("{}: {}", ty, vals.len())).collect::<Vec<_>>().join("\n\t"));

    // write a json out with everything that was found
    // let out = json!({
    //     "stats": {
    //         "num_eval_ok": num_eval_ok,
    //         "num_eval_err": num_eval_err,
    //         "num_eval_total": num_eval_ok+num_eval_err,
    //         "percent_eval_ok": num_eval_ok as f64 / (num_eval_ok + num_eval_err) as f64 * 100.0,
    //         "num_eval_per_ms": (num_eval_ok+num_eval_err) as f64 / tstart.elapsed().as_millis() as f64,
    //         "num_not_seen": num_not_seen,
    //         "num_yes_seen": num_yes_seen,
    //         "num_yes_seen_and_was_better": num_yes_seen_and_was_better,
    //     },
    // });


    // let out_path = cfg.out;
    // if let Some(out_path_dir) = out_path.parent() {
    //     if !out_path_dir.exists() {
    //         std::fs::create_dir_all(out_path_dir).unwrap();
    //     }
    // }
    // std::fs::write(out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    // println!("Wrote to {:?}", out_path);



}


struct ArgChoiceIterator<'a, D: Domain> {
    args: Vec<ArgState<'a,D>>,
    arg_tp_iter: Box<dyn Iterator<Item=(Vec<(&'a Type, &'a Type)>, Type)> + 'a>,
    vals_of_type: &'a AHashMap<Type,Vec<Found<D>>>, // vals[i] is the list of found vals for the ith arg
    return_tp: Option<Type>,
    fn_cost: usize,
    max_cost: usize,
    prev_max_cost: usize,
    prev_idx_to_inc: usize,
}

struct ArgState<'a, D: Domain> {
    i_vals: usize,
    tp: &'a Type,
    vals: &'a [Found<D>]
}

// struct ArgTypeIter<'a> {
//     fn_tp: &'a Type,
//     seen_types: &'a [Type],
// }
// impl<'a> Iterator for ArgTypeIter<'a> {
//     type Item = (Vec<(&'a Type, &'a Type)>, Type);

//     fn next(&mut self) -> Option<Self::Item> {

//     }
// }



impl <'a, D: Domain> ArgChoiceIterator<'a,D> {
    fn new(vals_of_type: &'a AHashMap<Type,Vec<Found<D>>>, seen_types: &'a [Type], fn_tp: &'a Type, fn_cost: usize, max_cost:  usize, prev_max_cost: usize) -> Self {
        assert!( max_cost > prev_max_cost);
        assert!(fn_tp.arity() > 0); // we use the empty .args list as a sentinel
        

        let mut arg_tp_iter = fn_tp.iter_args().map(|arg_tp|
            seen_types.iter()
                      .filter(move |seen_tp| Context::empty().unify(seen_tp, arg_tp).is_ok()) // filter for ones that unify with the expected type
                      .map(move |seen_tp| (seen_tp,arg_tp))
            ).multi_cartesian_product()
             .filter_map(move |seen_arg_tps|{
                // unify all the args together in one context to see if they're all mutually compatible
                let mut ctx = Context::empty();
                if !seen_arg_tps.iter().all(|(seen_tp, arg_tp)| {
                    let ty = arg_tp.apply(&mut ctx);
                    ctx.unify(seen_tp, &ty).is_ok()
                }) {
                    None // at least one unify() failure
                } else {
                    // Some(seen_arg_tps)
                    Some((seen_arg_tps, fn_tp.return_type().apply(&mut ctx)))
                }
             });

        // initialize the `args` field or make it an empty vector as a sentinel in case theres no first item in `arg_tp_iter`
        let (args, return_tp) = arg_tp_iter.next().map(|(seen_arg_tps, return_tp)| (seen_arg_tps.iter().map(|(seen_tp,_)| ArgState { i_vals: 0, tp: seen_tp, vals: &vals_of_type[seen_tp]}).collect(), Some(return_tp))).unwrap_or((vec![],None));

        ArgChoiceIterator {
            args,
            arg_tp_iter: Box::new(arg_tp_iter),
            vals_of_type,
            return_tp,
            fn_cost,
            max_cost,
            prev_max_cost,
            prev_idx_to_inc: 0,
        }
    }
    fn next_tps(&mut self) -> bool {
        match self.arg_tp_iter.next() {
            Some((seen_arg_tps, return_tp)) => {
                for (arg, (seen_tp,_)) in self.args.iter_mut().zip(seen_arg_tps.iter()) {
                    arg.i_vals = 0;
                    arg.tp = seen_tp;
                    arg.vals = &self.vals_of_type[seen_tp];
                }
                self.return_tp = Some(return_tp);
                true
            },
            None => {
                self.return_tp = None;
                false
            },
        }
    }
    fn rollover(&mut self) {
        let mut carry = false;
        for (i,arg) in self.args.iter_mut().enumerate() {
            if carry {
                arg.i_vals += 1; // carry the +1
                self.prev_idx_to_inc = i;
                carry = false;
            }
            if arg.i_vals >= arg.vals.len() {
                arg.i_vals = 0;
                carry = true;
            }
        }
        if carry {
            self.args.last_mut().unwrap().i_vals = self.args.last().unwrap().vals.len();
        }
    }
}


impl<'a, D: Domain> Iterator for ArgChoiceIterator<'a, D> {
    type Item = (Vec<&'a Found<D>>, Type, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.return_tp == None {
            return None // this is a sentinel
        }

        loop {
            // termination condition / set new types
            if self.args.last().unwrap().i_vals >= self.args.last().unwrap().vals.len() {
                if self.next_tps() {
                    continue
                } else {
                    return None;
                }
            }

            // check the cost, and if its too high then max out whatever the last thing
            // to increment was (which we know has all zeros to the left of it bc thats
            // how incrementing happens) so that the thing one higher than it will get incremented
            let cost: usize = self.fn_cost + self.args.iter().map(|arg| arg.vals[arg.i_vals].cost).sum::<usize>();
            
            if cost > self.max_cost {
                // skip ahead off the end bc we know theyll all be too expensive.
                self.args[self.prev_idx_to_inc].i_vals = self.args[self.prev_idx_to_inc].vals.len();
                debug_assert!(self.args[..self.prev_idx_to_inc].iter().all(|arg| arg.i_vals == 0));
                self.rollover();
                continue;
            }

            // check if cost is somethign we could have caught on a previous iteration
            if cost <= self.prev_max_cost {
                self.args.first_mut().unwrap().i_vals += 1;
                self.prev_idx_to_inc = 0;
                self.rollover();
                continue;
            }

            let res: Vec<&Found<D>> = self.args.iter().map(|arg| &arg.vals[arg.i_vals]).collect();

            // just increment the base
            self.args.first_mut().unwrap().i_vals += 1;
            self.prev_idx_to_inc = 0;
            self.rollover();


            return Some((res, self.return_tp.clone().unwrap(), cost))
        }
    }
}