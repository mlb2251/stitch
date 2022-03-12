use stitch::*;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::collections::HashMap;


/// Runs compression
#[pyfunction(
    programs,
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
    no_opt_upper_bound: bool,
    no_opt_force_multiuse: bool,
    no_opt_useless_abstract: bool,
    no_stats: bool) -> HashMap<String,String> {

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
        no_opt_upper_bound,
        no_opt_force_multiuse,
        no_opt_useless_abstract,
        no_stats,
    };

    let programs: Vec<Expr> = programs.iter().map(|p| p.parse().unwrap()).collect();
    println!("{}","**********".blue().bold());
    println!("{}","* Stitch *".blue().bold());
    println!("{}","**********".blue().bold());
    programs_info(&programs);

    let programs: Expr = Expr::programs(programs);
    
    py.allow_threads(||
        unimplemented!()
    )
}

/// Formats the sum of two numbers as string.
// #[pyfunction(
//     a,
//     "*",
//     b = "10",
//     c = "10",
// )]
// #[pyo3(text_signature = "(a, *, b, c)")]
// fn soot(a: usize, b: usize) -> PyResult<String> {
//     Ok((a + b).to_string())
// }

/// Formats the sum of two numbers as string.
// #[pyfunction]
// fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//     Ok((a + b).to_string())
// }

/// A Python module implemented in Rust.
#[pymodule]
fn stitch(_py: Python, m: &PyModule) -> PyResult<()> {
    // m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    // m.add_function(wrap_pyfunction!(soot, m)?)?;
    m.add_function(wrap_pyfunction!(compression, m)?)?;
    Ok(())
}