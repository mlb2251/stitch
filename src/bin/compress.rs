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

    /// disable all optimizations
    #[clap(long)]
    pub no_opt: bool,

    #[clap(long)]
    pub split_train_test: bool,

    /// extracts argument values from the json; specifically assumes a key value pair like
    ///     "stitch_args": "data/dc/logo_iteration_1_stitchargs.json -a3 -t8 --fmt=dreamcoder --dreamcoder-drop-last --no-mismatch-check",
    /// in the toplevel dictionary of the json. All other commandline args get discarded when
    /// you specify this option.
    #[clap(long)]
    pub args_from_json: bool,

    #[clap(flatten)]
    pub step: CompressionStepConfig,

}

fn main() {
    // procspawn::init();
    let mut args = Args::parse();
    if args.args_from_json {
        let json = std::fs::read_to_string(&args.file).unwrap();
        let json: serde_json::Value = serde_json::from_str(&json).unwrap();
        // we want something that looks like "ignored_binary_name -a3 -t2"
        let mut args_str = String::from("compress ");
        args_str.push_str(json["stitch_args"].as_str().unwrap());
        args = Args::parse_from(args_str.split_whitespace());
    }

    if args.no_opt {
        args.step.no_opt();
    }
    // create a new directory for logging outputs
    // let out_dir: String = format!("target/{}",timestamp());
    // let out_dir_p = std::path::Path::new(out_dir.as_str());
    // assert!(!out_dir_p.exists());
    // std::fs::create_dir(out_dir_p).unwrap();
    
    let (mut train_programs,
        test_programs,
        tasks,
        num_prior_inventions) = args.fmt.load(&args.file, args.split_train_test).unwrap();
    
    if args.shuffle {
        train_programs.shuffle(&mut rand::thread_rng());
    }
    if let Some(n) = args.truncate {
        train_programs.truncate(n);
    }
    
    // parse the program strings into expressions
    let train_programs: Vec<Expr> = train_programs.iter().map(|p| p.parse().unwrap()).collect();
    let test_programs: Option<Vec<Expr>> = if let Some(ps) = test_programs { Some(ps.iter().map(|p| p.parse().unwrap()).collect()) } else { None };
    

    // for prog in programs.iter() {
    //     println!("{}", prog);
    // }
    println!("{}","**********".blue().bold());
    println!("{}","* Stitch *".blue().bold());
    println!("{}","**********".blue().bold());
    println!("{}","* Training programs: *".blue().bold());
    programs_info(&train_programs);
    println!("{}","* Testing programs: *".blue().bold());
    if let Some(test_ps) = &test_programs {
        programs_info(test_ps);
    } else {
        println!{"No test programs given"}
    }

    // build a single `Expr::Programs` node from these programs. Stitch uses these because often we want to treat
    // different parts of the same programs the same way that we treat different parts of different programs, so
    // treating everything as one big expression makes sense.
    let train_programs: Expr = Expr::programs(train_programs);
    let test_programs: Option<Expr> = test_programs.map(|ps|Expr::programs(ps));

    if train_programs.to_string_curried(None).contains("(app (lam")  {
        println!("Normal dreamcoder programs never have unapplied lambdas in them! Who knows what might happen if you run this. Probably it will be fine");
    }

    let step_results = compression(&train_programs, &test_programs, args.iterations, &args.step, &tasks, num_prior_inventions);

    // write everything to json
    let out = json!({
        "cmd": std::env::args().join(" "),
        "args": args,
        "train_original_cost": train_programs.cost(),
        "train_original": train_programs.split_programs().iter().map(|p| p.to_string()).collect::<Vec<String>>(),
        "test_original_cost": inspect(&test_programs, |ps| ps.cost()),
        "test_original": inspect(&test_programs, |ps| ps.split_programs().iter().map(|p| p.to_string()).collect::<Vec<String>>()),
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