use crate::*;
use lambdas::*;
use rand::SeedableRng;
use rustc_hash::{FxHashMap};
use std::sync::Arc;

fn sample_new_ivar(
    original_pattern: &Pattern,
    shared: &SharedData,
    variable_ivar: usize,
    match_loc: &usize,
    rng: &mut impl rand::Rng,
) -> Option<usize> {
    let num_vars = get_num_variables(original_pattern);
    if num_vars <= 1 {
        return None; // no other variable to expand to
    }
    let mut new_ivar = rng.gen_range(0..num_vars - 1);
    if new_ivar >= variable_ivar {
        new_ivar += 1; // skip the variable we are expanding
    }
    let zid_original = get_zid_for_ivar(original_pattern, variable_ivar);
    let zid_new = get_zid_for_ivar(original_pattern, new_ivar);
    if shared.arg_of_zid_node[zid_original][match_loc].shifted_id == shared.arg_of_zid_node[zid_new][match_loc].shifted_id {
        return Some(new_ivar);
    }
    return None;
}

fn sample_variable_reuse_expansion(
    pattern: &Pattern,
    shared: &SharedData,
    variable_ivar: usize,
    match_location: usize,
    rng: &mut impl rand::Rng,
) -> Option<(Pattern, ExpandsTo)> {
    if let Some(new_ivar) = sample_new_ivar(pattern, shared, variable_ivar, &match_location, rng) {
        let zid_original = get_zid_for_ivar(pattern, variable_ivar);
        let zid_new = get_zid_for_ivar(pattern, new_ivar);
        let locs = compatible_locations(
            shared,
            pattern,
            &shared.arg_of_zid_node[zid_original],
            &shared.arg_of_zid_node[zid_new],
        );
        if !locs.is_empty() {
            let mut pattern = pattern.clone();
            pattern.match_locations = locs;
            let expands_to = ExpandsTo::IVar(new_ivar as i32);
            return Some((pattern, expands_to));
        }
    }
    None
}

fn sample_syntactic_expansion(
    original_pattern: &Pattern,
    arg_of_loc: &FxHashMap<Idx, Arg>,
    match_location: usize,
) -> (Pattern, ExpandsTo) {
    let expands_to = arg_of_loc[&match_location].expands_to.clone();
    let pattern = Pattern {
        holes: original_pattern.holes.clone(),
        match_locations: original_pattern.match_locations.iter().filter(
            |&loc| arg_of_loc[&loc].expands_to == expands_to
        ).cloned().collect(),
        first_zid_of_ivar: original_pattern.first_zid_of_ivar.clone(),
        arg_choices: original_pattern.arg_choices.clone(),
        body_utility: original_pattern.body_utility,
        utility_upper_bound: original_pattern.utility_upper_bound,
        tracked: original_pattern.tracked,
    };
    return (pattern, expands_to);
}

fn sample_expands_to(
    original_pattern: &Pattern,
    shared: &SharedData,
    arg_of_loc: &FxHashMap<Idx,Arg>,
    match_location: usize,
    variable_ivar: usize,
    rng: &mut impl rand::Rng,
) -> (Pattern, ExpandsTo) {
    if let Some(out) = sample_variable_reuse_expansion(
        original_pattern,
        shared,
        variable_ivar,
        match_location,
        rng,
    ) {
        return out;
    }
    return sample_syntactic_expansion(
        original_pattern,
        arg_of_loc,
        match_location,
    );
}

pub fn smc_expand(
    original_pattern: &Pattern,
    shared: &SharedData,
    rng: &mut impl rand::Rng,
) -> Option<Pattern> {
    let match_location = original_pattern.match_locations[rng.gen_range(0..original_pattern.match_locations.len())];
    let num_vars = get_num_variables(original_pattern);
    if num_vars == 0 {
        return None
    }
    let variable_ivar: usize = rng.gen_range(0..num_vars);
    let variable_zid = get_zid_for_ivar(original_pattern, variable_ivar);
    let arg_of_loc = &shared.arg_of_zid_node[variable_zid];
    let (pattern, expands_to) = sample_expands_to(original_pattern, shared, arg_of_loc, match_location, variable_ivar, rng);
    perform_expansion_variable(
        pattern,
        &shared,
        variable_ivar,
        expands_to,
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

fn calculate_utility(p: &Pattern) -> usize {
    p.body_utility as usize * (p.match_locations.len() - 1)
}

fn compute_logweight(p: &Pattern) -> f64 {
    let utility = calculate_utility(p);
    let logweight = (utility as f64).ln() * TEMPERATURE;
    logweight
}

fn resample(
    patterns: &[Pattern],
    rng: &mut impl rand::Rng,
    number: usize,
) -> Vec<Pattern> {
    let (deduplicated, counts) = do_deduplication(patterns);
    let logweights: Vec<f64> = deduplicated.iter().enumerate().map(|(i, p)|
        compute_logweight(p) + (counts[i] as f64).ln()
    ).collect();
    let total = modppl::logsumexp(&logweights);
    let mut weights = if total.is_infinite() {
        vec![1.0 * number as f64 / deduplicated.len() as f64; deduplicated.len()]
    } else {
        logweights.iter().map(|&w| number as f64 * (w - total).exp()).collect()
    };
    let mut result = vec![];
    for i in 0..weights.len() {
        while weights[i] >= 1.0 {
            result.push(deduplicated[i].clone());
            weights[i] -= 1.0;
        }
    }
    if result.len() == number {
        return result;
    }

    compute_cumulative_weights(&mut weights);
    
    for _ in result.len()..number {
        let idx = weighted_choice(&weights, rng);
        result.push(deduplicated[idx].clone());
    }
    return result;
}

fn do_deduplication(patterns: &[Pattern]) -> (Vec<Pattern>, Vec<i32>) {
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
    (deduplicated, counts)
}

fn compute_cumulative_weights(
    weights: &mut Vec<f64>,
) {
    let weight_sum = weights.iter().sum::<f64>();
    if weight_sum == 0.0 {
        let len= weights.len();
        weights.fill(1.0 / len as f64);
    }
    weights.iter_mut().for_each(|w| *w /= weight_sum);
    let mut accum = 0.0;
    for i in 0..weights.len() {
        accum += weights[i];
        weights[i] = accum;
    }
}

fn weighted_choice(cum_weights: &[f64], rng: &mut impl rand::Rng) -> usize {
    // println!("Choosing from weights: {:?}", cum_weights);
    let r: f64 = rng.gen();
    // println!("r: {:?}", r);
    return match cum_weights.binary_search_by(|&w| w.partial_cmp(&r).unwrap()) {
        Ok(idx) => idx,
        Err(idx) => idx,
    };
}

pub fn compression_step_smc(
    programs: &[ExprOwned],
    multistep_cfg: &MultistepCompressionConfig,
    tasks: &[String],
    weights: &[f32],
) -> Vec<CompressionStepResult> {
    let Some(shared) = construct_shared(
        programs,
        multistep_cfg,
        tasks,
        weights,
    ) else {
        return vec![];
    };

    let pattern = Pattern::single_var_from_shared(&shared);
    let mut patterns = vec![pattern; shared.cfg.smc_particles];
    
    let rng = &mut rand::rngs::StdRng::seed_from_u64(shared.cfg.seed);

    let mut best = patterns[0].clone();

    loop {
        patterns = smc_expand_all(&patterns, &shared, rng);
        if patterns.is_empty() {
            break;
        }
        patterns = resample(&patterns, rng, shared.cfg.smc_particles);
        for p in &patterns {
            if calculate_utility(p) > calculate_utility(&best) {
                best = p.clone();
            }
        }
    }

    let mut shared: SharedData = Arc::try_unwrap(shared).unwrap();

    let very_first_cost = shared.init_cost;

    let mut results = vec![];
    for (i, pattern) in [best].iter().enumerate() {
        let finished_pattern = FinishedPattern::new(pattern.clone(), &shared);
        let invention_name = format!("inv{}", i);
        let result = CompressionStepResult::new(
            finished_pattern,
            &invention_name,
            &mut shared,
            very_first_cost,
            &[],
            None,
        );
        results.push(result);
    }

    results
}
