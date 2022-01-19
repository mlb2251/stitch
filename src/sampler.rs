use crate::*;
use std::rc::*;
use rand::prelude::thread_rng;
use rand::distributions::Distribution;
use rand::distributions::WeightedIndex;

//pub fn sample_program<D : Domain>(domain : D) {
pub fn sample_program() -> String {
    // Something like: Construct the PCFG from the domain's terminals + non-terminals.
    // Repeatedly sample from this PCFG, until no holes remain.

    // These are the NON-TERMINALS of the domain
    let mut grammar : Vec<(&str, usize, bool)> = domains::simple::PRIMS.values().cloned().filter_map(|nt|
        match nt {
            crate::Val::PrimFun(pf) => Some((pf.name.as_str(), pf.arity, true)),
            // TODO make the above use getters instead of just accessing the fields
            _ => None
        }
    ).collect();
    grammar.push(("lam", 1 as usize, false));

    // Now - before sampling or anything, also want to get the _terminals_.
    // There's an issue here with the fact that how many terminals we have depends on how "deep" into
    // lambdas we are - i.e. if we only have one lambda we can only generate $0, if we have two we can
    // generate $0 or $1, etc. To get around this, we have a single "var" terminal, which we then uniformly
    // concretize into a specific, depth-appropriate variable.
    let terms = vec!["_VAR", "-2", "-1", "0", "1", "2"];
    grammar.extend(terms.into_iter().map(|term| (term, 0 as usize, false)));
    //grammar.extend::<Vec<(String, usize)>>(terms.into_iter().map(|(s, a) : (&str, usize)| -> (String, usize) {
    //    (String::from(s), a)
    //}).collect());

    // Make the distribution over the grammar. For now, just making it uniform;
    // in general, it should be a PCFG given by the domain writer, probably.
    let weights = vec![1; grammar.len()];
    let pcfg_distrib = WeightedIndex::new(&weights).unwrap();
    let mut rng = thread_rng();

    // Every generated program first looks like (lam ??)
    let first_hole = util::Node::new_hole();
    let root_node = util::Node::new_internal_node(String::from("lam"), vec![Rc::clone(&first_hole)], false);

    let mut work_stack = vec![(Rc::clone(&first_hole), 0)];
    while !work_stack.is_empty() {
        let (hole, hole_depth) = work_stack.pop().unwrap();
        let (sampled_nt_or_term, arity, needs_app) = grammar[pcfg_distrib.sample(&mut rng)];

        let mut new_term_data = String::from(sampled_nt_or_term);  // String > &str due to dynamic var names ($0, $1, etc)
        let mut new_term_depth = hole_depth;

        let mut possible_vars = Vec::with_capacity(hole_depth+1);
        if sampled_nt_or_term == "_VAR" {
            // TODO make this LESS HACKY

            // When we generate a _VAR terminal, we uniformly replace it with
            // a variable (i.e. $0, $1, and so on) that is currently "in scope";
            // i.e. we do not want to generate $2 if we only have two lambdas above
            // this variable.
            let var_weights = vec![1; hole_depth+1];
            let var_distrib = WeightedIndex::new(&var_weights).unwrap();
            for i in 0..hole_depth+1 {
                let var_str = format!("${}", i);
                possible_vars.push(var_str);
            } // Again, surely there's a more elegant way of making a list [$0, $1, $2, ...]...
            new_term_data = possible_vars[var_distrib.sample(&mut rng)].clone();
        } else if sampled_nt_or_term == "lam" {
            // TODO make this less hacky too
            new_term_depth += 1;
        }

        // OK, we have now sampled something from our grammar to replace the hole with.
        // Whether the sampled term was a non-terminal or a terminal is implicitly
        // handled by the arity: we generate as many children (which will be holes) as
        // the term's arity, and add those children to the work queue.determined by the arity
        if arity > 0 {
            let mut new_holes = Vec::with_capacity(arity);
            for _ in 0..arity {
                let new_hole = util::Node::new_hole();
                new_holes.push(Rc::clone(&new_hole));
                work_stack.push((Rc::clone(&new_hole), new_term_depth));
            }
            // TODO there must be a more elegant two-liner for the above, right?
            // E.g. I'd ideally do something like new_holes = [new_hole() for _ in range(arity)],
            // work_stack.extend(new_holes.into_iter().map(|hole| hole.clone())).
            // Or something like that. Not sure how to do list comprehension-like things in Rust, though.

            // Replace the hole node with the sampled non-terminal
            let mut mh = hole.borrow_mut();
            mh.insert(new_term_data, new_holes, needs_app);
        } else {
            // Replace the hole node with the sampled terminal
            assert_eq!(needs_app, false);
            let mut mh = hole.borrow_mut();
            mh.insert(new_term_data, vec![], needs_app);
        }
    }

    // OK, now translate the generated program from tree-form to normal lambda-calc form...
    // TODO I guess this should use sexpr or something as an intermediary form?
    return root_node.borrow().to_string();
}