use crate::*;
use lambdas::*;
use rand::seq::SliceRandom;
use rustc_hash::{FxHashMap,FxHashSet};
use std::convert::TryInto;
use std::fmt::{self, Formatter, Display};
use std::hash::Hash;
use itertools::Itertools;
use serde_json::json;
use clap::{Parser};
use serde::Serialize;
use std::thread;
use std::sync::Arc;
use parking_lot::Mutex;
use std::ops::DerefMut;
use std::collections::BinaryHeap;
use rand::Rng;



pub fn smc_expand(
    original_pattern: &Pattern,
    shared: &SharedData,
    rng: &mut impl rand::Rng,
) -> Option<Pattern> {
    let match_location = original_pattern.match_locations[rng.gen_range(0..original_pattern.match_locations.len())];
    let num_holes = original_pattern.holes.len();
    if num_holes == 0 {
        return None
    }
    let hole_idx: usize = shared.cfg.hole_choice.choose_hole(&original_pattern, &shared);
    let (mut holes_after_pop, hole_zid, arg_of_loc) = pop_hole(
        &original_pattern,
        &shared,
        hole_idx,
    );
    let expands_to = arg_of_loc[&match_location].expands_to.clone();
    let locs = original_pattern.match_locations.iter().filter(
        |&&loc| arg_of_loc[&loc].expands_to == expands_to
    ).cloned().collect::<Vec<_>>();
    perform_expansion(
        &original_pattern,
        &shared,
        &holes_after_pop,
        hole_zid,
        expands_to,
        locs,
        0,
        true,
    )
}

pub fn smc_expand_all(
    original_pattern: &Vec<Pattern>,
    shared: &SharedData,
    rng: &mut impl rand::Rng,
) -> Vec<Pattern> {
    let mut expanded_patterns = vec![];
    for pattern in original_pattern {
        if let Some(expanded) = smc_expand(pattern, shared, rng) {
            expanded_patterns.push(expanded);
        }
    }
    expanded_patterns
}

const TEMPERATURE: f64 = 1.0;
const ADDTL_WEIGHT: f64 = 1e-10;

fn compute_logweight(p: &Pattern) -> f64 {
    // TODO better utility function, probably should just be a field
    let utility = p.body_utility as usize * (p.match_locations.len() - 1);
    let logweight = (utility as f64).ln() * TEMPERATURE;
    logweight
}

fn resample(
    patterns: &[Pattern],
    rng: &mut impl rand::Rng,
    number: usize,
) -> Vec<Pattern> {
    let mut deduplicated = vec![];
    let mut counts = vec![];
    let mut deduplicated_pattern_to_idx: FxHashMap<Pattern, usize> = FxHashMap::default();
    for pattern in patterns {
        if let Some(idx) = deduplicated_pattern_to_idx.get_mut(pattern) {
            counts[*idx] += 1;
        } else {
            deduplicated.push(pattern.clone());
            counts.push(1);
            deduplicated_pattern_to_idx.insert(pattern.clone(), deduplicated.len() - 1);
        }
    }
    let logweights: Vec<f64> = deduplicated.iter().enumerate().map(|(i, p)|
        compute_logweight(p) + (counts[i] as f64).ln()
    ).collect();
    let total = modppl::logsumexp(&logweights);
    let mut weights = if total.is_infinite() {
        // if the total is infinite, we can't normalize, so we just return the patterns as is
        vec![1.0 * number as f64 / deduplicated.len() as f64; deduplicated.len()]
    } else {
        logweights.iter().map(|&w| (w - total).exp()).collect()
    };
    let mut result = vec![];
    for i in 0..weights.len() {
        while weights[i] >= 1.0 {
            result.push(deduplicated[i].clone());
            weights[i] -= 1.0;
        }
    }
    if (result.len() == number) {
        return result;
    }
    let weight_sum = weights.iter().sum::<f64>();
    weights.iter_mut().for_each(|w| *w /= weight_sum);
    for _ in result.len()..number {
        let idx = weighted_choice(&weights, rng);
        result.push(deduplicated[idx].clone());
    }
    return result;
}

fn weighted_choice(weights: &[f64], rng: &mut impl rand::Rng) -> usize {
    let mut cumulative = 0.0;
    let mut choice = 0;
    let r: f64 = rng.gen();
    for (i, &weight) in weights.iter().enumerate() {
        cumulative += weight;
        if r < cumulative {
            choice = i;
            break;
        }
    }
    choice
}

pub fn compression_step_smc(
    programs: &[ExprOwned],
    new_inv_name: &str, // name of the new invention, like "inv4"
    multistep_cfg: &MultistepCompressionConfig,
    tasks: &[String],
    weights: &[f32],
    very_first_cost: i32,
    name_mapping: &[(String, String)],
) -> Vec<CompressionStepResult> {
    let Some(shared) = construct_shared(
        programs,
        multistep_cfg,
        tasks,
        weights,
    ) else {
        return vec![];
    };

    let num_particles = 10;

    let mut patterns = {
        let mut shared_guard = shared.crit.lock();
        let mut crit = shared_guard.deref_mut();
        let worklist = crit.worklist.clone();
        worklist.iter().map(|item| item.pattern.clone()).collect::<Vec<_>>()
    };

    assert!(patterns.len() == 1);
    for _ in 2..num_particles {
        // clone the first pattern to create a new particle
        let first_pattern = patterns[0].clone();
        patterns.push(first_pattern)
    }
    
    let rng = &mut rand::thread_rng();

    loop {
        println!("patterns in worklist: {:?}", patterns.iter().map(|p| p.info(&shared)).collect::<Vec<_>>());
        patterns = smc_expand_all(&patterns, &shared, rng);
        if patterns.is_empty() {
            break;
        }
        patterns = resample(&patterns, rng, num_particles);
    }

    panic!("compression_step_smc is not implemented in this version of the code. Use compression_step instead.");
}