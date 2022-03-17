use stitch::*;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use serde_json::json;


/// Calls compression.rs::compression(), and has a similar API to bin/compress.rs
/// `programs` should be a list of program strings. `tasks` should be a list of task name strings,
/// with length equal to that of programs. Keyword arguments exist for
/// all the parameters of CompressionStepConfig (eg max_arity etc). `iterations` controls
/// the number of inventions that are returned.
/// Returns: a json string similar to the output of bin/compress.rs with some minor changes.
/// You can parse this string with `import json; json.loads(output)`.
#[pyfunction(
    programs,
    tasks,
    "*",
    iterations = "3",
    max_arity = "2",
    threads = "1",
    inv_candidates = "1",
    fifo_worklist = "false",
    ascending_worklist = "false",
    lossy_candidates = "false",
    no_cache = "false",
    show_rewritten = "false",
    no_opt_free_vars = "false",
    no_opt_single_use = "false",
    no_opt_upper_bound = "false",
    no_opt_force_multiuse = "false",
    no_opt_useless_abstract = "false",
    no_stats = "false",
)]
fn compression(
    py: Python,
    programs: Vec<String>,
    tasks: Vec<String>,
    iterations: usize,
    max_arity: usize,
    threads: usize,
    inv_candidates: usize,
    fifo_worklist: bool,
    ascending_worklist: bool,
    lossy_candidates: bool,
    no_cache: bool,
    show_rewritten: bool,
    no_opt_free_vars: bool,
    no_opt_single_use: bool,
    no_opt_single_task: bool,
    no_opt_upper_bound: bool,
    no_opt_force_multiuse: bool,
    no_opt_useless_abstract: bool,
    no_stats: bool,
    no_ctx_thread: bool,
    no_other_util: bool,
    ) -> String {

    let cfg = CompressionStepConfig {
        max_arity,
        threads,
        inv_candidates,
        fifo_worklist,
        ascending_worklist,
        lossy_candidates,
        no_cache,
        show_rewritten,
        no_opt_free_vars,
        no_opt_single_use,
        no_opt_single_task,
        no_opt_upper_bound,
        no_opt_force_multiuse,
        no_opt_useless_abstract,
        no_stats,
        no_other_util,
        no_ctx_thread,
    };

    let programs: Vec<Expr> = programs.iter().map(|p| p.parse().unwrap()).collect();
    programs_info(&programs);
    let programs: Expr = Expr::programs(programs);

    // release the GIL and call compression
    let step_results = py.allow_threads(||
        stitch::compression(&programs, iterations, &cfg, &tasks)
    );

    let out = json!({
        "cfg": cfg,
        "iterations": iterations,
        "original_cost": programs.cost(),
        "original": programs.split_programs().iter().map(|p| p.to_string()).collect::<Vec<String>>(),
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