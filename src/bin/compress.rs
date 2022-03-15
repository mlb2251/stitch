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

    /// disable all optimizations
    #[clap(long)]
    pub no_opt: bool,

    #[clap(flatten)]
    pub step: CompressionStepConfig,

}

fn main() {
    // procspawn::init();
    let mut args = Args::parse();
    if args.no_opt {
        args.step.no_opt();
    }
    // create a new directory for logging outputs
    // let out_dir: String = format!("target/{}",timestamp());
    // let out_dir_p = std::path::Path::new(out_dir.as_str());
    // assert!(!out_dir_p.exists());
    // std::fs::create_dir(out_dir_p).unwrap();
    
    // load programs in from one of two different json formats depending on the --dc-fmt flag
    let mut programs: Vec<String> = if args.dc_fmt {
        // read dreamcoder format
        let json: serde_json::Value = from_reader(File::open(&args.file).expect("file not found")).expect("json deserializing error");
        let mut programs: Vec<String> = json["frontiers"].as_array().unwrap_or_else(||panic!("json parse error, are you sure you wanted --dc-fmt ?")).iter().map(|f| f["programs"].as_array().unwrap().iter().map(|p|p["program"].as_str().unwrap().to_string())).flatten().collect();
        programs = programs.iter().map(|p| p.replace("(lambda ","(lam ")).collect();
        programs
    } else {
        from_reader(File::open(&args.file).expect("file not found")).unwrap_or_else(|_|panic!("json parse error, did you mean to include --dc-fmt ?"))
    };
    
    
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

    let out_path = &args.out;
    if let Some(out_path_dir) = out_path.parent() {
        if !out_path_dir.exists() {
            std::fs::create_dir_all(out_path_dir).unwrap();
        }
    }
    std::fs::write(out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("Wrote to {:?}", out_path);
}