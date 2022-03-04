use dreamegg::*;
use std::fs::File;
use serde_json::de::from_reader;
// extern crate log;
use clap::Parser;
use rand::seq::SliceRandom;
use serde::Serialize;
use std::path::PathBuf;
use serde_json::json;
use itertools::Itertools;


/// Args for compression
#[derive(Parser, Debug, Serialize)]
#[clap(name = "Stitch")]
pub struct Args {
    /// json file to read compression input programs from
    #[clap(short, long, parse(from_os_str), default_value = "data/train_19.json")]
    pub file: PathBuf,

    /// json output file
    #[clap(short, long, parse(from_os_str), default_value = "out/out.json")]
    pub out: PathBuf,

    /// Number of iterations to run compression for (number of inventions to find)
    #[clap(short, long, default_value = "3")]
    pub iterations: usize,

    /// shuffle order of set of inventions 
    #[clap(long)]
    pub shuffle: bool,

    /// truncate set of inventions to include only this many (happens after shuffle if shuffle is also specified)
    #[clap(long)]
    pub truncate: Option<usize>,

    #[clap(flatten)]
    pub step: CompressionStepConfig,
}

fn main() {
    procspawn::init();

    let args = Args::parse();

    // create a new directory for logging outputs
    let out_dir: String = format!("target/{}",timestamp());
    let out_dir_p = std::path::Path::new(out_dir.as_str());
    assert!(!out_dir_p.exists());
    std::fs::create_dir(out_dir_p).unwrap();

    let mut programs: Vec<String> = from_reader(File::open(&args.file).expect("file not found")).expect("json deserializing error");
    if args.shuffle {
        programs.shuffle(&mut rand::thread_rng());
    }
    if let Some(n) = args.truncate {
        programs.truncate(n);
    }
    
    let programs: Vec<Expr> = programs.iter().map(|p| p.parse().unwrap()).collect();

    for prog in programs.iter() {
        println!("{}", prog);
    }    

    programs_info(&programs);

    let programs: Expr = Expr::programs(programs);

    if programs.to_string_curried(None).contains("(app (lam") {
        println!("Normal dreamcoder programs never have unapplied lambdas in them! Who knows what might happen if you run this. Probably it will be fine");
    }

    compression(
        &programs,
        args.iterations,
        args.out.to_str().unwrap(),
        &args.step,
    );
}

fn compression(
    programs_expr: &Expr,
    iterations: usize,
    out_file: &str,
    cfg: &CompressionStepConfig,
) -> Vec<CompressionStepResult> {
    let mut rewritten: Expr = programs_expr.clone();
    let mut step_results: Vec<CompressionStepResult> = Default::default();

    let tstart = std::time::Instant::now();

    for i in 0..iterations {
        println!("\n=======Iteration {}=======",i);
        let inv_name = format!("inv{}",step_results.len());
        let res: Vec<CompressionStepResult> = compression_step(
            &rewritten,
            &inv_name,
            programs_expr.cost(),
            cfg,
            &step_results);
        if !res.is_empty() {
            let res: CompressionStepResult = res[0].clone();
            rewritten = res.rewritten.clone();
            println!("Chose Invention {}: {}\n{}", res.inv.name, res, res.rewritten);
            step_results.push(res);
        } else {
            println!("No inventions found at iteration {}",i);
            break;
        }
    }

    println!("\n=======Compression Summary=======");
    println!("Found {} inventions", step_results.len());
    println!("Cost Improvement: ({:.2}x better) {} -> {}", compression_factor(programs_expr,&rewritten), programs_expr.cost(), rewritten.cost());
    for i in 0..step_results.len() {
        let res = &step_results[i];
        println!("{} ({:.2}x wrt orig): {}" ,res.inv.name, compression_factor(programs_expr, &res.rewritten), res);
    }
    println!("Time: {}ms", tstart.elapsed().as_millis());

    let out = json!({
        "cmd": std::env::args().join(" "),
        // "args": args,
        "original_cost": programs_expr.cost(),
        "original": programs_expr.split_programs().iter().map(|p| p.to_string()).collect::<Vec<String>>(),
        "invs": step_results.iter().map(|inv| inv.json()).collect::<Vec<serde_json::Value>>(),
    });

    std::fs::write(out_file, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("Wrote to {:?}",out_file);
    step_results
}