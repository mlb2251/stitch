use crate::*;
use std::rc::*;
use std::collections::HashMap;
use rand::prelude::thread_rng;
use rand::prelude::ThreadRng;
use rand::distributions::Distribution;
use rand::distributions::WeightedIndex;

// Generate the desired number of programs from the given domain.
// Each program is sampled pseudo-independently: the rng state is
// shared, but other than that there are no dependencies between samples.
// TODO this is probably a good candidate for quite trivial multi-threading
pub fn sample_n_programs<D: Domain>(n: usize) -> Vec<String> {
    let pcfg = D::pcfg();
    let tokens = D::tokens_with_arities();
    let mut rng = thread_rng();
    let mut programs = Vec::with_capacity(n);
    for _ in 0..n {
        programs.push(sample_program::<D>(&pcfg, &tokens, &mut rng));
    }
    return programs;
}

// Generates a single program from the given domain.
// The generation is done by top-down sampling, so results are non-deterministic, and their
// usefulness highly depend on the unigram PCFG passed to the function.
pub fn sample_program<D: Domain>(
    pcfg: &HashMap<String, WeightedIndex<usize>>,
    tokens: &Vec<(&'static str, usize)>,
    rng: &mut ThreadRng,
) -> String {

    // During generation, programs are represented internally as trees.
    // Every generated program first looks like (lam ??)
    let first_hole = util::Node::new_hole();
    let root_node = util::Node::new_internal_node(String::from(D::lambda()), vec![Rc::clone(&first_hole)]);

    // To keep track of which node to expand next, we make use of a "work stack". Each entry in the stack
    // is a hole to expand, the "depth" of the hole (the largest valid DeBruijin index for the hole),
    // and the token of the parent node (so that we know which entry in the PCFG to look at).
    let mut work_stack = vec![(Rc::clone(&first_hole), 0, String::from(D::lambda()))];
    while !work_stack.is_empty() {

        let (hole, hole_depth, prev_token) = work_stack.pop().unwrap();
        let distrib = pcfg.get(&prev_token).unwrap();

        let (sampled_token, arity) = tokens[distrib.sample(rng)];
        let mut new_node_data = String::from(sampled_token);
        let mut new_node_depth = hole_depth;

        if sampled_token == D::var() {
            // TODO make this LESS HACKY

            // When we generate a _VAR terminal, we uniformly replace it with
            // a variable (i.e. $0, $1, and so on) that is currently "in scope";
            // i.e. we do not want to generate $2 if we only have two lambdas above
            // this variable; this is when the hole "depth" comes into play.
            let mut possible_vars = Vec::with_capacity(hole_depth+1);
            let var_weights = vec![1; hole_depth+1];
            let var_distrib = WeightedIndex::new(&var_weights).unwrap();
            for i in 0..hole_depth+1 {
                let var_str = format!("${}", i);
                possible_vars.push(var_str);
            } // surely there's a more elegant way of making a list [$0, $1, $2, ...]...
            new_node_data = possible_vars[var_distrib.sample(rng)].clone();
        } else if sampled_token == D::lambda() {
            // TODO make this less hacky too

            // If we have generated a lambda, then we increment the depth for the
            // children of this node
            new_node_depth += 1;
        }

        // OK, we have now sampled something from our grammar to replace the hole with.
        // Whether the sampled term was a non-terminal or a terminal is implicitly
        // handled by the arity: we generate as many children (which will initally be holes) as
        // the term's arity, and add those children to the work queue so that they can be expanded.
        let mut new_holes = vec![];
        if arity > 0 {
            new_holes = Vec::with_capacity(arity);
            for _ in 0..arity {
                let new_hole = util::Node::new_hole();
                new_holes.push(Rc::clone(&new_hole));
                work_stack.push((Rc::clone(&new_hole), new_node_depth, new_node_data.clone()));
            }
            // TODO there must be a more elegant two-liner for the above, right?
            // E.g. I'd ideally do something like new_holes = [new_hole() for _ in range(arity)],
            // work_stack.extend(new_holes.into_iter().map(|hole| hole.clone())).
            // Or something like that. Not sure how to do list comprehension-like things in Rust, though.
        }

        // Now replace the hole in the tree with the sampled token, along with >= 0 new holes as children
        hole.borrow_mut().insert(new_node_data, new_holes);
    }

    // OK, we have now run out of entries on the work stack, so there must not be any holes left to expand
    // in the program.
    // Now translate the generated program from tree-form to normal lambda-calc form...
    // TODO I guess this should use sexpr or something as an intermediary form?
    return root_node.borrow().to_string();
}