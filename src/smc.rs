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
    None
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
            |&loc| arg_of_loc[loc].expands_to == expands_to
        ).cloned().collect(),
        first_zid_of_ivar: original_pattern.first_zid_of_ivar.clone(),
        arg_choices: original_pattern.arg_choices.clone(),
        body_utility: original_pattern.body_utility,
        utility_upper_bound: original_pattern.utility_upper_bound,
        tracked: original_pattern.tracked,
    };
    (pattern, expands_to)
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
    sample_syntactic_expansion(
        original_pattern,
        arg_of_loc,
        match_location,
    )
}

#[derive(Clone, Debug)]
struct Particle {
    pattern: Pattern,
    utility: usize,
}

impl Particle {
    fn new(pattern: Pattern, shared: &SharedData) -> Self {
        let utility = calculate_utility(&pattern, shared);
        Particle { pattern, utility }
    }
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
        shared,
        variable_ivar,
        expands_to,
    )
}

fn smc_expand_all(
    original_particles: &Vec<Particle>,
    shared: &SharedData,
    rng: &mut impl rand::Rng,
) -> Vec<Particle> {
    let mut expanded_particles = vec![];
    for particle in original_particles {
        if let Some(expanded) = smc_expand(&particle.pattern, shared, rng) {
            expanded_particles.push(Particle::new(expanded, shared));
        }
    }
    expanded_particles
}

const TEMPERATURE: f64 = 1.5;

fn calculate_utility_fn(p: &Pattern, shared: &SharedData, use_fast_utility: bool) -> usize {
    let cost_leaf = shared.cfg.cost.cost_prim_default as i32;
    let cost_app: i32 = shared.cfg.cost.cost_app as i32;
    let util = if use_fast_utility {
        let updated_util = p.body_utility - cost_leaf - cost_app * get_num_variables(p) as i32;
        updated_util * ((p.match_locations.iter().map(|loc| shared.num_paths_to_node[*loc])).sum::<i32>() - 1)
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
        (compute_logweight(p) + (counts[i] as f64).ln()) / TEMPERATURE
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
            pat1.pattern.arg_choices.cmp(&pat2.pattern.arg_choices)
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
) -> Vec<CompressionStepResult> {
    let Some(shared) = construct_shared(
        programs,
        multistep_cfg,
        tasks,
        weights,
    ) else {
        return vec![];
    };

    let mut particles = vec![Particle::new(Pattern::single_var_from_shared(&shared), &shared); shared.cfg.smc_particles];
    
    let rng = &mut rand::rngs::StdRng::seed_from_u64(shared.cfg.seed);

    let mut best = particles[0].clone();

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
            if p.utility > best.utility {
                best = p.clone();
                if shared.cfg.verbose_best {
                    println!("Step {}: New best pattern found: {}. Fast utility: {}, Accurate utility: {}",
                        step,
                        best.pattern.info(&shared),
                        calculate_utility_fn(&best.pattern, &shared, true),
                        calculate_utility_fn(&best.pattern, &shared, false)
                    );
                }
                last_optimal_step = step;
            }
        }
    }

    let mut shared: SharedData = Arc::try_unwrap(shared).unwrap();

    let very_first_cost = shared.init_cost;

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
