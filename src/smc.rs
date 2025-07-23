use crate::*;
use lambdas::*;
use rand::SeedableRng;
use rustc_hash::{FxHashMap};
use std::sync::Arc;


fn filter_match_locs_for_syntactic_expansion(
    original_pattern: &Pattern,
    arg_of_loc: &FxHashMap<Idx, Arg>,
    expands_to: &ExpandsTo,
) -> Pattern {
    let pattern: Pattern = Pattern {
        holes: original_pattern.holes.clone(),
        match_locations: original_pattern.match_locations.iter().filter(
            |&loc| arg_of_loc[loc].expands_to == *expands_to
        ).cloned().collect(),
        pattern_args: original_pattern.pattern_args.clone(),
        body_utility: original_pattern.body_utility,
        utility_upper_bound: original_pattern.utility_upper_bound,
        tracked: original_pattern.tracked,
    };
    pattern
}

fn sample_expands_to(
    original_pattern: &Pattern,
    shared: &SharedData,
    arg_of_loc: &FxHashMap<Idx,Arg>,
    match_location: usize,
    variable_ivar: usize,
    rng: &mut impl rand::Rng,
) -> (Pattern, Option<ExpandsTo>) {
    if let Some(out) = sample_variable_reuse_expansion(
        original_pattern,
        shared,
        variable_ivar,
        match_location,
        rng,
    ) {
        return (out.0, Some(out.1));
    }
    let expands_to = arg_of_loc[&match_location].expands_to.clone();
    if valid_syntactic_expansion_loc(&shared.sym_var_info, &expands_to) {
        let pattern = filter_match_locs_for_syntactic_expansion(original_pattern, arg_of_loc, &expands_to);
        return (pattern, Some(expands_to));
    }
    return (original_pattern.clone(), None);
}

#[derive(Clone, Debug)]
struct Particle {
    pattern: Pattern,
    utility: usize,
    metavariables_valid: bool, // whether the metavariables in the pattern are valid
}

impl Particle {
    fn new(pattern: Pattern, shared: &SharedData, metavariables_valid: bool) -> Self {
        let utility = calculate_utility(&pattern, shared);
        Particle { pattern, utility, metavariables_valid }
    }
}

pub fn smc_expand_once(
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
    // println!("Checking consistency for variable ivar: {}, zid: {}", variable_ivar, variable_zid);
    let arg_of_loc = &shared.arg_of_zid_node[variable_zid];
    let (pattern, expands_to) = sample_expands_to(original_pattern, shared, arg_of_loc, match_location, variable_ivar, rng);
    let Some(expands_to) = expands_to else {
        return Some(original_pattern.clone()); // no valid expansion found
    };
    let mut pattern = perform_expansion_variable(
        pattern,
        shared,
        variable_ivar,
        expands_to.clone(),
    )?;
    pattern = clean_invalid_metavars(pattern, match_location, shared);
    Some(pattern)
}

pub fn clean_invalid_metavars(pattern: Pattern, representative_loc: Idx, shared: &SharedData) -> Pattern {
    let mut pattern = pattern;
    while let Some(unvalidated_ivar) = pattern.pattern_args.unvalidated_ivar() {
        let zid = pattern.pattern_args.variables[unvalidated_ivar as usize].0;
        let arg_of_loc = &shared.arg_of_zid_node[zid as usize];
        let arg_of_loc_this = arg_of_loc[&representative_loc].clone();
        let is_metavar = !invalid_metavar_location(shared, arg_of_loc_this.shifted_id);
        let is_symvar = shared.sym_var_info.as_ref().is_some_and(|symvar_info| symvar_info.is_symvar_spot(arg_of_loc_this.shifted_id));
        assert!(!is_metavar || !is_symvar, "Unvalidated ivar {} with zid {} is both a metavar and a symvar: {:?}", unvalidated_ivar, zid, arg_of_loc_this);
        if is_metavar {
            pattern.pattern_args.variables[unvalidated_ivar as usize].1 = VariableType::Metavar;
            pattern.match_locations.retain(
                |loc| !invalid_metavar_location(shared, arg_of_loc[loc].shifted_id)
            );
            continue;
        }
        if is_symvar {
            pattern.pattern_args.variables[unvalidated_ivar as usize].1 = VariableType::Symvar;
            pattern.match_locations.retain(
                |loc| shared.sym_var_info.as_ref().is_some_and(|symvar_info| symvar_info.is_symvar_spot(arg_of_loc[loc].shifted_id))
            );
            continue;
        }
        assert!(valid_syntactic_expansion_loc(&shared.sym_var_info, &arg_of_loc_this.expands_to));
        pattern.match_locations.retain(
            |loc| arg_of_loc[loc].expands_to == arg_of_loc_this.expands_to
        );
        pattern = perform_expansion_variable(
            pattern,
            shared,
            unvalidated_ivar as usize,
            arg_of_loc_this.expands_to.clone(),
        ).unwrap();
    };
    pattern
}

pub fn smc_expand_n_times(
    original_pattern: &Pattern,
    shared: &SharedData,
    rng: &mut impl rand::Rng,
    n: usize,
) -> Option<Pattern> {
    let mut num_rounds = 0;
    let mut pattern = original_pattern.clone();
    for _ in 0..n {
        if let Some(new_pattern) = smc_expand_once(&pattern, shared, rng) {
            pattern = new_pattern;
            num_rounds += 1;
        } else {
            return None; // no more expansions possible
        }
    }
    if num_rounds == 0 {
        return None; // no expansions were made
    }
    Some(pattern)
}

fn smc_expand_all(
    original_particles: &Vec<Particle>,
    shared: &SharedData,
    rng: &mut impl rand::Rng,
) -> Vec<Particle> {
    let mut expanded_particles = vec![];
    for particle in original_particles {
        if let Some(expanded) = smc_expand_n_times(&particle.pattern, shared, rng, shared.cfg.smc_expand_per_step) {
            // get a guarantee that the metavariables are valid from smc_expand_n_times
            expanded_particles.push(Particle::new(expanded, shared, true));
        }
    }
    expanded_particles
}

fn calculate_utility_fn(p: &Pattern, shared: &SharedData, use_fast_utility: bool) -> usize {
    let cost_leaf = shared.cfg.cost.cost_prim_default as Cost;
    let cost_app = shared.cfg.cost.cost_app as Cost;
    let util = if use_fast_utility {
        let updated_util = p.body_utility - cost_leaf - cost_app * get_num_variables(p) as Cost;
        updated_util * ((p.match_locations.iter().map(|loc| shared.num_paths_to_node[*loc])).sum::<Cost>() - 1)
    } else {
        compressive_utility(p, shared).util + noncompressive_utility(p.body_utility, &shared.cfg)
    };
    if util < cost_leaf {
        return cost_leaf as usize; // no utility if the compressive utility is negative
    }
    util as usize
}

fn calculate_utility(p: &Pattern, shared: &SharedData) -> usize {
    calculate_utility_fn(p, shared, shared.cfg.smc_fast_utility)
}


fn compute_logweight(p: &Particle) -> f64 {
    (p.utility as f64).ln()
}

fn resample(
    patterns: Vec<Particle>,
    rng: &mut impl rand::Rng,
    number: usize,
    shared: &SharedData,
) -> Vec<Particle> {
    let (deduplicated, counts) = do_deduplication(patterns);
    // println!("Counts after deduplication: {:?}", counts);
    // for i in 0..deduplicated.len() {
    //     if counts[i] > 1 {
    //         println!("Count: {} for pattern: {:?}", counts[i], deduplicated[i].info(shared));
    //     }
    // }
    let logweights: Vec<f64> = deduplicated.iter().enumerate().map(|(i, p)|
        (compute_logweight(p) + (counts[i] as f64).ln()) / shared.cfg.smc_temperature as f64
    ).collect();
    // println!("utilities: {:?}", deduplicated.iter().map(|x| calculate_utility(x, &shared)).collect::<Vec<_>>());
    // println!("Log weights: {:?}", logweights);
    let total = modppl::logsumexp(&logweights);
    let mut weights = if total.is_infinite() {
        vec![1.0 * number as f64 / deduplicated.len() as f64; deduplicated.len()]
    } else {
        logweights.iter().map(|&w| number as f64 * (w - total).exp()).collect()
    };
    // println!("Weights after normalization: {:?}", weights);
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
    result
}

fn do_deduplication(mut particles: Vec<Particle>) -> (Vec<Particle>, Vec<i32>) {
    particles.sort_by(
        |pat1, pat2|
            pat1.pattern.pattern_args.cmp(&pat2.pattern.pattern_args)
            .then_with(|| pat1.pattern.holes.cmp(&pat2.pattern.holes))
    );
    let mut counts = vec![];
    if particles.is_empty() {
        return (particles, counts);
    }
    let mut current_pattern = particles[0].pattern.clone();
    let mut deduplicated: Vec<Particle> = vec![particles[0].clone()];
    counts.push(1);
    for particle in particles {
        if particle.pattern == current_pattern {
            *counts.last_mut().unwrap() += 1;
        } else {
            current_pattern = particle.pattern.clone();
            deduplicated.push(particle);
            counts.push(1);
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
    } else {
        weights.iter_mut().for_each(|w| *w /= weight_sum);
    }
    let mut accum = 0.0;
    for w in weights {
        accum += *w;
        *w = accum;
    }
}

fn weighted_choice(cum_weights: &[f64], rng: &mut impl rand::Rng) -> usize {
    // println!("Choosing from weights: {:?}", cum_weights);
    let r: f64 = rng.gen();
    // println!("r: {:?}", r);
    match cum_weights.binary_search_by(|&w| w.partial_cmp(&r).unwrap()) {
        Ok(idx) => idx,
        Err(idx) => idx, // it could be inserted at idx, which means it's <= cum_weights[idx]
    }
}

pub fn compression_step_smc(
    programs: &[ExprOwned],
    inv_name: &str,
    multistep_cfg: &MultistepCompressionConfig,
    tasks: &[String],
    weights: &[f32],
    prev_results: &[CompressionStepResult],
) -> Vec<CompressionStepResult> {
    let Some(shared) = construct_shared(
        programs,
        multistep_cfg,
        tasks,
        weights,
        prev_results,
    ) else {
        return vec![];
    };

    // initially there's no safety guarantee
    let mut particles = vec![Particle::new(Pattern::single_var_from_shared(&shared), &shared, false); shared.cfg.smc_particles];
    
    let rng = &mut rand::rngs::StdRng::seed_from_u64(shared.cfg.seed);

    let mut best = None;

    let mut last_optimal_step = 0;

    for step in 0.. {
        if step - last_optimal_step > shared.cfg.smc_extra_steps {
            break; // stop if we haven't found a better pattern in a while
        }
        particles = smc_expand_all(&particles, &shared, rng);
        if particles.is_empty() {
            break;
        }
        particles = resample(particles, rng, shared.cfg.smc_particles, &shared);
        for p in &particles {
            if !p.metavariables_valid {
                // skip particles with invalid metavariables
                continue;
            }
            if p.utility <= best.as_ref().map(|p: &Particle| p.utility).unwrap_or(0) {
                continue;
            }
            if !TDFAInventionAnnotation::metavariables_are_consistent(&p.pattern, &shared) {
                continue; // skip patterns with inconsistent metavariables
            }
            TDFAInventionAnnotation::from_pattern(&p.pattern, &shared);
            best = Some(p.clone());
            if shared.cfg.verbose_best {
                println!("Step {}: New best pattern found: {}. Fast utility: {}, Accurate utility: {}",
                    step,
                    p.pattern.info(&shared),
                    calculate_utility_fn(&p.pattern, &shared, true),
                    calculate_utility_fn(&p.pattern, &shared, false)
                );
            }
            last_optimal_step = step;
        }
    }

    let mut shared: SharedData = Arc::try_unwrap(shared).unwrap();

    let very_first_cost = shared.init_cost;

    let Some (best) = best else {
        return vec![];
    };

    let finished_pattern = FinishedPattern::new(best.pattern.clone(), &shared);
    let result = CompressionStepResult::new(
        finished_pattern,
        inv_name,
        &mut shared,
        very_first_cost,
        &[],
        None,
    );
    vec![result]
}
