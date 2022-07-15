// #![cfg(features="python")]

use stitch::*;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use serde_json::{json,Value};


/// Calls compression.rs::compression(), and has a similar API to bin/compress.rs
/// `programs` should be a list of program strings. `tasks` should be a list of task name strings,
/// with length equal to that of programs. Keyword arguments exist for
/// all the parameters of CompressionStepConfig (eg max_arity etc). `iterations` controls
/// the number of inventions that are returned.
/// Returns: a json string similar to the output of bin/compress.rs with some minor changes.
/// You can parse this string with `import json; json.loads(output)`.
#[pyfunction(
    programs,
    "*",
    max_arity = "2",
    threads = "1",
    batch = "1",
    dynamic_batch = "false",
    inv_candidates = "1",
    no_mismatch_check = "false",
    no_top_lambda = "false",
    track = "None",
    follow_track = "false",
    verbose_worklist = "false",
    verbose_best = "false",
    print_stats = "0",
    show_rewritten= "false",
    no_opt_free_vars= "false",
    no_opt_single_use= "false",
    no_opt_single_task= "false",
    no_opt_upper_bound= "false",
    no_opt_force_multiuse= "false",
    no_opt_useless_abstract= "false",
    no_opt_arity_zero= "false",
    no_stats= "false",
    no_other_util= "false",
    rewrite_check= "false",
    utility_by_rewrite= "false",
    dreamcoder_comparison= "false",
)]
fn compression(
    py: Python,
    programs: Vec<String>,
    iterations: usize,
    max_arity: usize,
    threads: usize,
    batch: usize,
    dynamic_batch: bool,
    inv_candidates: usize,
    no_mismatch_check: bool,
    no_top_lambda: bool,
    track: Option<String>,
    follow_track: bool,
    verbose_worklist: bool,
    verbose_best: bool,
    print_stats: usize,
    show_rewritten: bool,
    no_opt_free_vars: bool,
    no_opt_single_use: bool,
    no_opt_single_task: bool,
    no_opt_upper_bound: bool,
    no_opt_force_multiuse: bool,
    no_opt_useless_abstract: bool,
    no_opt_arity_zero: bool,
    no_stats: bool,
    no_other_util: bool,
    rewrite_check: bool,
    utility_by_rewrite: bool,
    dreamcoder_comparison: bool,    
) -> String {

    let cfg = CompressionStepConfig {
        max_arity,
        threads,
        batch,
        dynamic_batch,
        inv_candidates,
        hole_choice: HoleChoice::DepthFirst,
        no_mismatch_check,
        no_top_lambda,
        track,
        follow_track,
        verbose_worklist,
        verbose_best,
        print_stats,
        show_rewritten,
        no_opt_free_vars,
        no_opt_single_use,
        no_opt_single_task,
        no_opt_upper_bound,
        no_opt_force_multiuse,
        no_opt_useless_abstract,
        no_opt_arity_zero,
        no_stats,
        no_other_util,
        rewrite_check,
        utility_by_rewrite,
        dreamcoder_comparison,
    };

    // parse the program strings into expressions
    let train_programs: Vec<Expr> = programs.iter().map(|p| p.parse().unwrap()).collect();
    programs_info(&train_programs);
    let mut  num_prior_inventions = 0;
    while programs.iter().any(|p| p.contains(&format!("fn_{}", num_prior_inventions))) {
        num_prior_inventions += 1;
    }
    let mut tasks: Vec<String> = Vec::with_capacity(train_programs.len());
    for (task_num, _) in train_programs.iter().enumerate() {
        tasks.push(task_num.to_string());
    }
    let input = Input {
        train_programs: programs,
        test_programs: None,
        tasks,
        prev_dc_inv_to_inv_strs: Vec::new(),
    };

    let train_programs: Expr = Expr::programs(train_programs);

    if train_programs.to_string_curried(None).contains("(app (lam") {
        println!("Normal dreamcoder programs never have unapplied lambdas in them! Who knows what might happen if you run this. Probably it will be fine");
    }
    
    // release the GIL and call compression
    let step_results = py.allow_threads(||
        stitch::compression(&train_programs, &None, iterations, &cfg, &input.tasks, &input.prev_dc_inv_to_inv_strs)
    );

    // write everything to json
    let out = json!({
        "cmd": Value::Null,
        "args": cfg,
        "original_cost": train_programs.cost(),
        "original": train_programs.split_programs().iter().map(|p| p.to_string()).collect::<Vec<String>>(),
        "invs": step_results.iter().map(|inv| inv.json()).collect::<Vec<serde_json::Value>>(),
    });


    // return as something you could json.loads(out) from in python
    out.to_string()
}

/// A Python module implemented in Rust.
#[pymodule]
fn stitch(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    // m.add_function(wrap_pyfunction!(soot, m)?)?;
    m.add_function(wrap_pyfunction!(compression, m)?)?;
    Ok(())
}