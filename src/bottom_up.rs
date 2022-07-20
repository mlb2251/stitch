
use crate::*;
use ahash::AHashMap;
use std::time::Instant;
use serde_json::json;


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

pub fn bottom_up<D: Domain>(
    // handle: &mut Executable<D>,
    initial: &[FoundExpr<D>],
    fns: &[(DSLEntry<D>,usize)],
    max_cost:  usize,
    cost_delta: usize,
) {

    let tstart = Instant::now();
    let mut num_eval_err = 0;
    let mut num_eval_ok = 0;
    let mut num_not_seen = 0;
    let mut num_yes_seen = 0;
    let mut num_yes_seen_and_was_better = 0;

    let mut curr_cost = cost_delta;
    let mut vals_of_type: AHashMap<Type<D>,Vec<Found<D>>> = Default::default();
    let mut lambda_vals: Vec<Found<D>> = Default::default();
    let mut top_vals: Vec<Found<D>> = Default::default();

    // add each dsl fn so the ith dsl fn is expr.nodes[i] // todo programs() is a gross way to do this
    let mut handle: Executable<D> = {
        let dsl_fns_expr: Expr = Expr::programs(fns.iter().map(|(entry,_)| Expr::prim(entry.name)).collect());
        let init_vals_expr: Expr = Expr::programs(initial.iter().map(|found_expr| found_expr.expr.clone()).collect());    
        Expr::programs(vec![dsl_fns_expr,init_vals_expr]).into()
    };

    let init_val_ids: Vec<Id> = handle.expr.get(handle.expr.get_root().children()[1]).children().iter().cloned().collect();
    for (i,found_expr) in initial.iter().enumerate() {
        let id = init_val_ids[i]; // ith child of programs node
        let found = Found::new(found_expr.val.clone(), id, found_expr.cost);
        match &found_expr.val {
            Val::Dom(d)=> {
                vals_of_type.entry(Type::TDom(D::type_of_dom_val(d))).or_default().push(found.clone());
                println!("{:?} :: {:?}", d, D::type_of_dom_val(d));
            }
            _ => {
                lambda_vals.push(found.clone());
            }
        }
        top_vals.push(found.clone());
    }

    let mut seen: AHashMap<Val<D>,usize> = AHashMap::new();


    while curr_cost < max_cost {
        // sort by the cost
        vals_of_type.values_mut().for_each(|vals| {
            vals.sort_by(|a,b| a.cost.cmp(&b.cost));
            // vals.dedup_by(|a,b| a.val == b.val);
        });
        lambda_vals.sort_by(|a,b| a.cost.cmp(&b.cost));
        top_vals.sort_by(|a,b| a.cost.cmp(&b.cost));
        // lambda_vals.dedup_by(|a,b| a.val == b.val);

        println!("new curr cost: {}", curr_cost);
        let mut new_vals: Vec<Found<D>> = vec![];

        'next_fn:
        for (i_fn, (dsl_entry, fn_cost)) in fns.iter().enumerate() {
            // println!("trying fn: {}", dsl_entry.name);

            let mut vals: Vec<&[Found<D>]> = vec![];
            for ty in dsl_entry.arg_types.iter(){
                match vals_of_type.get(ty) {
                    Some(v) => {
                       assert!(!v.is_empty());
                       vals.push(&v[..]);
                    }
                    None => {
                        if *ty == Type::Top {
                            vals.push(&top_vals[..]);
                        } else {
                            // no vals of this type
                            continue 'next_fn;
                        }
                    }
                }
            };

            for (found_args,cost) in ArgChoiceIterator::new(&vals,dsl_entry.arity,*fn_cost,curr_cost, curr_cost - cost_delta) {
                let args: Vec<LazyVal<D>> = found_args.iter().map(|&f| LazyVal::new_strict(f.val.clone())).collect();
                // println!("trying ({} {})", dsl_entry.name, found_args.iter().map(|arg| format!("{:?}",arg.val)).collect::<Vec<_>>().join(" "));
                if let Ok(val) = (dsl_entry.dsl_fn) (args, &mut handle) {
                    num_eval_ok += 1;
                    match seen.get(&val) {
                        None => {
                            num_not_seen += 1;
                            // val has not been seen before!
                            // seen.insert(val.clone(),cost);

                            let mut id = Id::from(i_fn); // assumes we constructed the ith fn primitive to be the ith element in handle.expr.nodes
                            for arg in found_args.iter() {
                                handle.expr.nodes.push(Lambda::App([id,arg.id]));
                                id = Id::from(handle.expr.nodes.len()-1);
                            }
                            new_vals.push(Found::new(val, id, cost))
    
                        }
                        Some(&old_cost) => {
                            num_yes_seen += 1;
                            if old_cost > cost {
                                // update the seen value and push to new_vals
                                // Note this is safe even if we break the cost record more than once within a single iteration
                                // because only the final one from new_vals will remain at the end when we update our val vectors
                                // *seen.get_mut(&val).unwrap() = cost;

                                let mut id = Id::from(i_fn); // assumes we constructed the ith fn primitive to be the ith element in handle.expr.nodes
                                for arg in found_args.iter() {
                                    handle.expr.nodes.push(Lambda::App([id,arg.id]));
                                    id = Id::from(handle.expr.nodes.len()-1);
                                }
                                new_vals.push(Found::new(val, id, cost))
        
                            } else {
                                num_yes_seen_and_was_better += 1;
                            }
                        }
                    }

                } else {
                    // Err from execution, discard
                    num_eval_err += 1;
                }
            }
        }

        // deposit new vals into vals_of_type
        for found in new_vals.into_iter() {
            match seen.get(&found.val) {
                None => {
                    seen.insert(found.val.clone(),found.cost);
                    match &found.val {
                        Val::Dom(d)=> {
                            vals_of_type.entry(Type::TDom(D::type_of_dom_val(d))).or_default().push(found.clone());
                            top_vals.push(found.clone());
                            // println!("new val: {} :: {:?} -> {:?}", handle.expr.to_string_uncurried(Some(found.id)), D::type_of_dom_val(d), d);
                        }
                        _ => {
                            // discard i guess
                            println!("discarding {:?}", found.val);
                        }
                    }        
                }
                Some(&old_cost) => {
                    if old_cost > found.cost {
                        *seen.get_mut(&found.val).unwrap() = found.cost;
                        match &found.val {
                            Val::Dom(d)=> {
                                // remove old value - this is prob v slow as implemented, could do faster with a binary search by cost or something
                                vals_of_type.get_mut(&Type::TDom(D::type_of_dom_val(d))).unwrap().retain(|f| f.val != found.val);
                                top_vals.retain(|f| f.val != found.val);

                                // add new value
                                vals_of_type.entry(Type::TDom(D::type_of_dom_val(d))).or_default().push(found.clone());
                                top_vals.push(found.clone());
                                // println!("new val: {} :: {:?} -> {:?}", handle.expr.to_string_uncurried(Some(found.id)), D::type_of_dom_val(d), d);
                            }
                            _ => {
                                // discard i guess
                                println!("discarding {:?}", found.val);
                            }
                        }     
                    }
                }
            }
        }


        curr_cost += cost_delta;
    }

    //todo add a sanity check that the length of seen equals the lengths of all val arrays. i bet theres an error and that wont be true lol
    println!("reached max cost");
    println!("Time: {}ms",tstart.elapsed().as_millis());
    println!("num found: {}",seen.len());
    println!("num found per ms: {:.2}", seen.len() as f64 / tstart.elapsed().as_millis() as f64);
    println!("num eval ok: {}",num_eval_ok);
    println!("num eval err: {}",num_eval_err);
    println!("num eval total: {}",num_eval_ok+num_eval_err);
    println!("% eval ok: {:.2}%", num_eval_ok as f64 / (num_eval_ok + num_eval_err) as f64 * 100.0);
    println!("num eval per ms: {:.2}",(num_eval_ok+num_eval_err) as f64 / tstart.elapsed().as_millis() as f64);
    println!("num not seen: {}",num_not_seen);
    println!("num yes seen: {}",num_yes_seen);
    println!("num yes seen and was better: {}",num_yes_seen_and_was_better);

    // write a json out with everything that was found
    let out = json!({
        "stats": {
            "num_eval_ok": num_eval_ok,
            "num_eval_err": num_eval_err,
            "num_eval_total": num_eval_ok+num_eval_err,
            "percent_eval_ok": num_eval_ok as f64 / (num_eval_ok + num_eval_err) as f64 * 100.0,
            "num_eval_per_ms": (num_eval_ok+num_eval_err) as f64 / tstart.elapsed().as_millis() as f64,
            "num_not_seen": num_not_seen,
            "num_yes_seen": num_yes_seen,
            "num_yes_seen_and_was_better": num_yes_seen_and_was_better,
        },
    });


    // let out_path = args.out;
    // if let Some(out_path_dir) = out_path.parent() {
    //     if !out_path_dir.exists() {
    //         std::fs::create_dir_all(out_path_dir).unwrap();
    //     }
    // }
    // std::fs::write(out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    // println!("Wrote to {:?}", out_path);



}


struct ArgChoiceIterator<'a, D: Domain> {
    vals: &'a [ &'a [Found<D>] ], // vals[i] is the list of found vals for the ith arg
    idxs: Vec<usize>,
    arity: usize,
    fn_cost: usize,
    max_cost: usize,
    prev_max_cost: usize,
    prev_idx_to_inc: usize,
}

impl <'a, D: Domain> ArgChoiceIterator<'a,D> {
    fn new(vals: &'a [ &'a [Found<D>] ], arity: usize, fn_cost: usize, max_cost:  usize, prev_max_cost: usize) -> Self {
        assert!( max_cost > prev_max_cost);
        ArgChoiceIterator {
            vals,
            idxs: vec![0;arity],
            arity,
            fn_cost,
            max_cost,
            prev_max_cost,
            prev_idx_to_inc: 0,
        }
    }
    fn rollover(&mut self) {
        for i in 0..self.arity-1 {
            if self.idxs[i] >= self.vals[i].len() {
                self.idxs[i] = 0;
                self.idxs[i+1] += 1;
                self.prev_idx_to_inc = i+1;
            }
        }

    }
}


impl<'a, D: Domain> Iterator for ArgChoiceIterator<'a, D> {
    type Item = (Vec<&'a Found<D>>,usize);

    fn next(&mut self) -> Option<Self::Item> {
        loop {

            // termination condition
            if self.idxs[self.arity-1] >= self.vals[self.arity-1].len() {
                return None
            }

            // check the cost, and if its too high then max out whatever the last thing
            // to increment was (which we know has all zeros to the left of it bc thats
            // how incrementing happens) so that the thing one higher than it will get incremented
            let cost: usize = self.fn_cost + self.idxs.iter().enumerate().map(|(argi,i)| self.vals[argi][*i].cost).sum::<usize>();
            if cost > self.max_cost {
                // skip ahead off the end bc we know theyll all be too expensive.
                self.idxs[self.prev_idx_to_inc] = self.vals[self.prev_idx_to_inc].len();
                debug_assert!(self.idxs[..self.prev_idx_to_inc].iter().all(|i| *i == 0));
                self.rollover();
                continue;
            }

            // check if cost is somethign we could have caught on a previous iteration
            if cost <= self.prev_max_cost {
                self.idxs[0] += 1;
                self.prev_idx_to_inc = 0;
                self.rollover();
                continue;
            }

            let res: Vec<&Found<D>> = self.idxs.iter().enumerate().map(|(argi,i)| &self.vals[argi][*i]).collect();

            // just increment the base
            self.idxs[0] += 1;
            self.prev_idx_to_inc = 0;
            self.rollover();

            return Some((res,cost))
        }
    }
}