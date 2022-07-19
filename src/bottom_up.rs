
use crate::*;
use std::collections::HashMap;

pub fn bottom_up<D: Domain>(
    terminals: &[(Val<D>,usize)],
    fns: &[(Symbol,DSLFn<D>,usize,usize)],
    max_cost:  usize,
) {
    let mut cost = 0;
    let mut vals: Vec<(Val<D>,usize)> = terminals.to_vec();

    // todo we gotta make an Executable i think for our handle to execute stuff otherwise
    // lambdas and such dont make sense. I guess we can just mutably push stuff onto the
    // expr whenever we want it on there? I think that only needs to happen at the end of
    // each round of bottom up since we only need the handle to contain stuff from the previous round.

    // sort by the cost
    vals.sort_by(|a,b| a.1.cmp(&b.1));

    // let handle: Executable<D> = unimplemented!();

    //todo assume unigram weights initially

    let handle = unimplemented!();
    // let mut weights_of_fname_arg: HashMap<(Symbol,usize),Vec<usize>> = Default::default();

    while cost < max_cost {
        let mut new_vals: Vec<(Val<D>,usize)> = vec![];

        for (fn_name,dsl_fn, arity, cost) in fns.iter() {
            // todo ok assume you have a vec of vals sorted by a bigram weight

                // todo hmmmm you need to sorta recurse down the arity to stop early this is actually pretty tricky!
                // todo also im not sure how exactly it looks with bigrams right like doesnt the node at the top of the
                // tree suddenly change in its contribution to the cost?
                
                // todo also note my old implementation has types and thats pretty nice actually but not rush here

            for (args,cost) in ArgChoiceIterator::new(&vals,*arity,*cost) {
                let new_args: Vec<LazyVal<D>> = args.iter().map(|&v| LazyVal::new_strict(v.clone())).collect();
                if let Ok(val) =  dsl_fn(new_args, &mut handle) {
                    new_vals.push((val,cost));
                }
            }
        }
    }
}


/// say we have 3 arguments and
// fn arg_iterator<D: Domain>(vals: &[(Val<D>,usize)], arity: usize) -> impl Iterator<Item=&(Val<D>,usize)> {
//     vals.into_iter()
// }


struct ArgChoiceIterator<'a, D: Domain> {
    vals: &'a [(Val<D>,usize)],
    idxs: Vec<usize>,
    arity: usize,
    max_cost: usize,
    greatest_idx_to_increment: usize,
}

impl <'a, D: Domain> ArgChoiceIterator<'a,D> {
    fn new(vals: &'a [(Val<D>,usize)], arity: usize, max_cost:  usize) -> Self {
        ArgChoiceIterator {
            vals,
            idxs: vec![0;arity],
            arity,
            max_cost,
            greatest_idx_to_increment: 0,
        }
    }
    fn rollover(&mut self) {
        for i in 0..self.arity-1 {
            if self.idxs[i] >= self.vals.len() {
                self.idxs[i] = 0;
                self.idxs[i+1] += 1;
                self.greatest_idx_to_increment = i+1;
            }
        }

    }
}


impl<'a, D: Domain> Iterator for ArgChoiceIterator<'a, D> {
    type Item = (Vec<&'a Val<D>>,usize);

    fn next(&mut self) -> Option<Self::Item> {
        loop {

            // termination condition
            if self.idxs[self.arity-1] >= self.vals.len() {
                return None
            }

            // check the cost, and if its too high then max out whatever the last thing
            // to increment was (which we know has all zeros to the left of it bc thats
            // how incrementing happens) so that the thing one higher than it will get incremented
            let cost: usize = self.idxs.iter().map(|i| self.vals[*i].1).sum();
            if cost > self.max_cost {
                // skip ahead off the end bc we know theyll all be too expensive.
                self.idxs[self.greatest_idx_to_increment] = self.vals.len();
                debug_assert!(self.idxs[..self.greatest_idx_to_increment].iter().all(|i| *i == 0));
                self.rollover();
                continue;
            }

            let res: Vec<&Val<D>> = self.idxs.iter().map(|i| &self.vals[*i].0).collect();


            // just increment the base
            self.idxs[0] += 1;
            self.greatest_idx_to_increment = 0;
            self.rollover();

            return Some((res,cost))
        }
    }
}