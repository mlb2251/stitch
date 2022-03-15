use stitch::*;
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

    /// the format of the input file, e.g. 'programs-list' for a simple JSON array of programs
    /// or 'dreamcoder' for a JSON in the style expected by the original dreamcoder codebase.
    /// See [formats.rs] for options or to add new ones.
    #[clap(long, arg_enum, default_value = "programs-list")]
    pub fmt: InputFormat,

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
    
    let mut programs: Vec<String> = args.fmt.load_programs(&args.file).unwrap();
    
    if args.shuffle {
        programs.shuffle(&mut rand::thread_rng());
    }
    if let Some(n) = args.truncate {
        programs.truncate(n);
    }
    
    // parse the program strings into expressions
    let programs: Vec<Expr> = programs.iter().map(|p| p.parse().unwrap()).collect();

    // for prog in programs.iter() {
    //     println!("{}", prog);
    // }
    println!("{}","**********".blue().bold());
    println!("{}","* Stitch *".blue().bold());
    println!("{}","**********".blue().bold());
    programs_info(&programs);

    // build a single `Expr::Programs` node from these programs. Stitch uses these because often we want to treat
    // different parts of the same programs the same way that we treat different parts of different programs, so
    // treating everything as one big expression makes sense.
    let programs: Expr = Expr::programs(programs);

    if programs.to_string_curried(None).contains("(app (lam") {
        println!("Normal dreamcoder programs never have unapplied lambdas in them! Who knows what might happen if you run this. Probably it will be fine");
    }

    let step_results = compression(&programs, args.iterations, &args.step);

    // write everything to json
    let out = json!({
        "cmd": std::env::args().join(" "),
        "args": args,
        "original_cost": programs.cost(),
        "original": programs.split_programs().iter().map(|p| p.to_string()).collect::<Vec<String>>(),
        "invs": step_results.iter().map(|inv| inv.json()).collect::<Vec<serde_json::Value>>(),
    });

    std::fs::write(&args.out, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("Wrote to {:?}",args.out);
}