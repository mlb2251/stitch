use dreamegg::*;
use std::fs::File;
use serde_json::de::from_reader;
// extern crate log;
use clap::Parser;
use rand::seq::SliceRandom;
use serde::Serialize;
use std::path::PathBuf;

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