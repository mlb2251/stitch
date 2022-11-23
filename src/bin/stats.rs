use stitch_core::*;
// extern crate log;
use clap::Parser;
use serde::Serialize;
use std::path::PathBuf;



/// Compression
#[derive(Parser, Debug, Serialize)]
#[clap(name = "Stitch")]
pub struct Args {
    /// json file to read compression input programs from
    #[clap(parse(from_os_str))]
    pub file: PathBuf,

    /// the format of the input file, e.g. 'programs-list' for a simple JSON array of programs
    /// or 'dreamcoder' for a JSON in the style expected by the original dreamcoder codebase.
    /// See [formats.rs] for options or to add new ones.
    #[clap(long, arg_enum, default_value = "programs-list")]
    pub fmt: InputFormat,

}

fn get_stats(data: &[i32]) -> (f64, f64) {
    let mean = data.iter().sum::<i32>() as f64 / data.len() as f64;
    let variance = data.iter().map(|x|f64::powi(*x as f64 - mean, 2)).sum::<f64>() / data.len() as f64;
    (mean, f64::sqrt(variance))
}

fn main() {

    let args = Args::parse();
    let programs = args.fmt.load_programs_and_tasks(&args.file).unwrap().train_programs;

    let programs: Vec<ExprOwned> = programs.iter().map(|p|{
        let mut set = ExprSet::empty(Order::ChildFirst, false, false);
        let idx = set.parse_extend(p).unwrap();
        ExprOwned::new(set,idx)
    }).collect();

    let cost_fn = ExprCost::dreamcoder();
    let length_fn = ExprCost::num_terminals();

    let costs = programs.iter().map(|p| p.cost(&cost_fn)).collect::<Vec<i32>>();
    let lengths = programs.iter().map(|p| p.cost(&length_fn)).collect::<Vec<i32>>();
    let depths = programs.iter().map(|p| p.depth() as i32).collect::<Vec<i32>>();

    let (mean_cost, std_cost) = get_stats(&costs);
    let (mean_length, std_length) = get_stats(&lengths);
    let (mean_depth, std_depth) = get_stats(&depths);

    println!("Number of programs: {}", programs.len());
    println!("Costs: mean={:.2}, std={:.2}", mean_cost, std_cost);
    println!("Lengths: mean={:.2}, std={:.2}", mean_length, std_length);
    println!("Depths: mean={:.2}, std={:.2}", mean_depth, std_depth);
}