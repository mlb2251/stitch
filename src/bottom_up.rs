
use crate::*;
use ahash::AHashMap;


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
    let mut curr_cost = 0;
    let mut vals_of_type: AHashMap<D::Type,Vec<Found<D>>> = Default::default();
    let mut lambda_vals: Vec<Found<D>> = Default::default();

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
                vals_of_type.entry(D::type_of_dom_val(d)).or_default().push(found.clone());
                println!("{:?} :: {:?}", d, D::type_of_dom_val(d));
            }
            _ => {
                lambda_vals.push(found.clone());
            }
        }
    }

    let mut seen: AHashMap<Val<D>,usize> = AHashMap::new();

    // sort by the cost
    vals_of_type.values_mut().for_each(|vals| vals.sort_by(|a,b| a.cost.cmp(&b.cost)));
    lambda_vals.sort_by(|a,b| a.cost.cmp(&b.cost));

    while curr_cost < max_cost {
        println!("new curr cost: {}", curr_cost);
        let mut new_vals: Vec<Found<D>> = vec![];

        'next_fn:
        for (i_fn, (dsl_entry, fn_cost)) in fns.iter().enumerate() {
            println!("trying fn: {}", dsl_entry.name);

            let mut vals: Vec<&[Found<D>]> = vec![];
            for ty in dsl_entry.arg_types.iter(){
                match vals_of_type.get(ty) {
                    Some(v) => {
                       assert!(!v.is_empty());
                       vals.push(&v[..]);
                    }
                    None => {
                        // no vals of this type
                        continue 'next_fn;
                    }
                }
            };

            for (found_args,cost) in ArgChoiceIterator::new(&vals,dsl_entry.arity,*fn_cost,curr_cost) {
                let args: Vec<LazyVal<D>> = found_args.iter().map(|&f| LazyVal::new_strict(f.val.clone())).collect();
                println!("trying ({} {})", dsl_entry.name, found_args.iter().map(|arg| format!("{:?}",arg.val)).collect::<Vec<_>>().join(" "));
                if let Ok(val) = (dsl_entry.dsl_fn) (args, &mut handle) {
                    let mut do_add = false;
                    match seen.get(&val) {
                        None => {
                            // val has not been seen before!
                            do_add = true;
                        }
                        Some(&old_cost) => {
                            if old_cost > cost {
                                do_add = true;
                                // update the seen value and push to new_vals
                                // Note this is safe even if we break the cost record more than once within a single iteration
                                // because only the final one from new_vals will remain at the end when we update our val vectors
                                *seen.get_mut(&val).unwrap() = cost;
                            }
                        }
                    }
                    if do_add {
                        let mut id = Id::from(i_fn); // assumes we constructed the ith fn primitive to be the ith element in handle.expr.nodes
                        for arg in found_args.iter() {
                            handle.expr.nodes.push(Lambda::App([id,arg.id]));
                            id = Id::from(handle.expr.nodes.len()-1);
                        }                            
                        new_vals.push(Found::new(val, id, cost))
                    }
                } else {
                    // Err from execution, discard
                }
            }
        }

        // deposit new vals into vals_of_type

        curr_cost += cost_delta;
    }
    println!("reached max cost");
}


struct ArgChoiceIterator<'a, D: Domain> {
    vals: &'a [ &'a [Found<D>] ], // vals[i] is the list of found vals for the ith arg
    idxs: Vec<usize>,
    arity: usize,
    fn_cost: usize,
    max_cost: usize,
    prev_idx_to_inc: usize,
}

impl <'a, D: Domain> ArgChoiceIterator<'a,D> {
    fn new(vals: &'a [ &'a [Found<D>] ], arity: usize, fn_cost: usize, max_cost:  usize) -> Self {
        ArgChoiceIterator {
            vals,
            idxs: vec![0;arity],
            arity,
            fn_cost,
            max_cost,
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

            let res: Vec<&Found<D>> = self.idxs.iter().enumerate().map(|(argi,i)| &self.vals[argi][*i]).collect();

            // just increment the base
            self.idxs[0] += 1;
            self.prev_idx_to_inc = 0;
            self.rollover();

            return Some((res,cost))
        }
    }
}