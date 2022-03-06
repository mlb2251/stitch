use stitch::*;
use std::fs::File;
use serde_json::de::from_reader;
// extern crate log;
use clap::Parser;
use rand::seq::SliceRandom;
use serde::Serialize;
use std::path::PathBuf;
use serde_json::json;
use itertools::Itertools;


/// Compression
#[derive(Parser, Debug, Serialize)]
#[clap(name = "Stitch")]
pub struct Args {
    /// json file to read compression input programs from
    #[clap(parse(from_os_str))]
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

    /// use dreamcoder format
    #[clap(long)]
    pub dc_fmt: bool,

    #[clap(flatten)]
    pub step: CompressionStepConfig,
}

fn main() {
    // procspawn::init();
    let args = Args::parse();
    // create a new directory for logging outputs
    // let out_dir: String = format!("target/{}",timestamp());
    // let out_dir_p = std::path::Path::new(out_dir.as_str());
    // assert!(!out_dir_p.exists());
    // std::fs::create_dir(out_dir_p).unwrap();
    
    // load programs in from one of two different json formats depending on the --dc-fmt flag
    let mut programs: Vec<String> = if args.dc_fmt {
        // read dreamcoder format
        let json: serde_json::Value = from_reader(File::open(&args.file).expect("file not found")).expect("json deserializing error");
        let mut programs: Vec<String> = json["frontiers"].as_array().unwrap().iter().map(|f| f["programs"].as_array().unwrap().iter().map(|p|p["program"].as_str().unwrap().to_string())).flatten().collect();
        programs = programs.iter().map(|p| p.replace("(lambda ","(lam ")).collect();
        programs
    } else {
        from_reader(File::open(&args.file).expect("file not found")).expect("json deserializing error")
    };
    
    
    if args.shuffle {
        programs.shuffle(&mut rand::thread_rng());
    }
    if let Some(n) = args.truncate {
        programs.truncate(n);
    }
    
    // parse the program strings into expressions
    let programs: Vec<Expr> = programs.iter().map(|p| p.parse().unwrap()).collect();

    for prog in programs.iter() {
        println!("{}", prog);
    }

    programs_info(&programs);

    // build a single `Expr::Programs` node from these programs. Stitch uses these because often we want to treat
    // different parts of the same programs the same way that we treat different parts of different programs, so
    // treating everything as one big expression makes sense.
    let programs: Expr = Expr::programs(programs);

    if programs.to_string_curried(None).contains("(app (lam") {
        println!("Normal dreamcoder programs never have unapplied lambdas in them! Who knows what might happen if you run this. Probably it will be fine");
    }

    compression(&programs, &args);
}

fn compression(
    programs_expr: &Expr,
    args: &Args,
) -> Vec<CompressionStepResult> {
    let mut rewritten: Expr = programs_expr.clone();
    let mut step_results: Vec<CompressionStepResult> = Default::default();

    let tstart = std::time::Instant::now();

    for i in 0..args.iterations {
        println!("\n=======Iteration {}=======",i);
        let inv_name = format!("inv{}",step_results.len());

        // call actual compression
        let res: Vec<CompressionStepResult> = compression_step(
            &rewritten,
            &inv_name,
            &args.step,
            &step_results);

        if !res.is_empty() {
            // rewrite with the invention
            let res: CompressionStepResult = res[0].clone();
            rewritten = res.rewritten.clone();
            println!("Chose Invention {}: {}", res.inv.name, res);
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

    // write everything to json
    let out = json!({
        "cmd": std::env::args().join(" "),
        "args": args,
        "original_cost": programs_expr.cost(),
        "original": programs_expr.split_programs().iter().map(|p| p.to_string()).collect::<Vec<String>>(),
        "invs": step_results.iter().map(|inv| inv.json()).collect::<Vec<serde_json::Value>>(),
    });

    std::fs::write(&args.out, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("Wrote to {:?}",args.out);
    step_results
}