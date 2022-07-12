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

    /// extracts argument values from the json; specifically assumes a key value pair like
    ///     "stitch_args": "data/dc/logo_iteration_1_stitchargs.json -a3 -t8 --fmt=dreamcoder --dreamcoder-drop-last --no-mismatch-check",
    /// in the toplevel dictionary of the json. All other commandline args get discarded when
    /// you specify this option.
    #[clap(long)]
    pub args_from_json: bool,

    /// saves the rewritten frontiers in an input-readable format
    #[clap(long)]
    pub save_rewritten: Option<PathBuf>,

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
    
    let mut input = args.fmt.load_programs_and_tasks(&args.file).unwrap();
    
    if args.shuffle {
        input.train_programs.shuffle(&mut rand::thread_rng());
    }
    if let Some(n) = args.truncate {
        input.train_programs.truncate(n);
    }
    
    // parse the program strings into expressions
    let train_programs: Vec<Expr> = input.train_programs.iter().map(|p| p.parse().unwrap()).collect();
    let test_programs: Option<Vec<Expr>> = input.test_programs.map(|ps| ps.iter().map(|p| p.parse().unwrap()).collect());

    // for prog in programs.iter() {
    //     println!("{}", prog);
    // }
    println!("{}","**********".blue().bold());
    println!("{}","* Stitch *".blue().bold());
    println!("{}","**********".blue().bold());
    programs_info(&train_programs);
    if let Some(ps) = &test_programs {
        println!("> Running with train/test split active");
        programs_info(ps);
    }

    // build a single `Expr::Programs` node from these programs. Stitch uses these because often we want to treat
    // different parts of the same programs the same way that we treat different parts of different programs, so
    // treating everything as one big expression makes sense.
    let train_programs: Expr = Expr::programs(train_programs);
    let test_programs: Option<Expr> = test_programs.map(|ps| Expr::programs(ps));

    if train_programs.to_string_curried(None).contains("(app (lam") {
        println!("Normal dreamcoder programs never have unapplied lambdas in them! Who knows what might happen if you run this. Probably it will be fine");
    }

    let step_results = compression(&train_programs, &test_programs, args.iterations, &args.step, &input.tasks, &input.prev_dc_inv_to_inv_strs);

    // write everything to json
    let out = json!({
        "cmd": std::env::args().join(" "),
        "args": args,
        "original_cost": train_programs.cost(),
        "original": train_programs.split_programs().iter().map(|p| p.to_string()).collect::<Vec<String>>(),
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

    if let Some(out_path) = args.save_rewritten {
        println!("Wrote to {:?}", out_path);
        std::fs::write(&out_path, serde_json::to_string_pretty(&step_results.iter().last().unwrap().rewritten.split_programs().iter().map(|p| p.to_string()).collect::<Vec<String>>()).unwrap()).unwrap();
    }

}

#[cfg(test)]
mod tests {
    use super::*;


    /**
     * Regression test.
     * Tests whether the top 10 inventions found after 10 iterations on data/dc/logo_iteration_1.json
     * matches those expected.
     */
    #[test]
    fn dc_logo_it_1_top_10_match_expected() {
    
        let input_file = std::path::Path::new("data/dc/logo_iteration_1.json");
        let input = InputFormat::Dreamcoder.load_programs_and_tasks(input_file).unwrap();
    
        let train_programs: Vec<Expr> = input.train_programs.iter().map(|p| p.parse().unwrap()).collect();
        let train_programs: Expr = Expr::programs(train_programs);

        // Run compression with the default argument values
        let step_results = compression(
            &train_programs,
            &None,
            10,
            &CompressionStepConfig::parse_from("compress -a2 ".split_whitespace()),
            &input.tasks,
            &input.prev_dc_inv_to_inv_strs);

        let inventions = step_results.iter().take(10).map(|inv| inv.inv.body.to_string()).collect::<Vec<String>>();
        // let expected_inventions = vec!["(lam (logo_forLoop #0 (lam (lam (#1 $0))) $0))", "(logo_FWRT #0 (logo_DIVA logo_UA #1))", "(logo_MULL logo_UL)", "(logo_FWRT logo_UL)", "(logo_FWRT (logo_MULL logo_epsL #0))", "(fn_9 #0 (lam (logo_GETSET (lam (#1 $0)) (fn_10 logo_ZL #0 $0))))", "(logo_DIVL logo_UL)", "(fn_9 #0 (logo_FWRT (fn_11 4) (logo_MULA (logo_DIVA logo_UA #0) #1)))", "(logo_PT (lam (fn_12 #0 $0)))", "(logo_forLoop logo_IFTY (lam (lam (logo_FWRT #0 #1 $0))))"];
        // let expected_inventions = vec!["(lam (logo_forLoop #0 (lam (lam (#1 $0))) $0))", "(logo_FWRT #0 (logo_DIVA logo_UA #1))", "(logo_MULL logo_UL)", "(logo_FWRT logo_UL)", "(logo_FWRT (logo_MULL logo_epsL #0))", "(fn_9 #0 (lam (logo_GETSET (lam (#1 $0)) (fn_10 logo_ZL #0 $0))))", "(logo_DIVL logo_UL)", "(fn_9 #0 (logo_FWRT (fn_11 4) (logo_MULA (logo_DIVA logo_UA #0) #1)))", "(logo_PT (lam (fn_12 #0 $0)))", "(logo_forLoop logo_IFTY (lam (lam (logo_FWRT #0 #1 $0))))"];
        let expected_inventions = vec!["(lam (logo_forLoop #0 #1 $0))",
                                                "(lam (logo_FWRT #1 #0 $0))",
                                                "(fn_9 #0 (lam (lam (logo_GETSET (fn_10 #1 logo_UL) (logo_FWRT logo_ZL (logo_DIVA logo_UA #0) $0)))))",
                                                "(fn_10 #1 (logo_MULL logo_UL #0))",
                                                "(logo_DIVA logo_UA)",
                                                "(fn_9 logo_IFTY (lam (fn_10 #1 (logo_MULL logo_epsL #0))))",
                                                "(logo_DIVL logo_UL)",
                                                "(lam (logo_FWRT logo_ZL #1 (#0 $0)))",
                                                "(logo_MULL logo_epsL)",
                                                "(logo_PT (fn_10 #0 logo_UL))"];
        
        // Assert single threaded results match expected results
        assert_eq!(inventions, expected_inventions);
    }

    /**
     * Regression test.
     * Tests whether the top 10 inventions found after 10 iterations on data/dc/logo_iteration_1.json
     * are the same both for single-threaded and 4-threaded runs.
     */
    #[test]
    fn dc_logo_it_1_top_10_st_match_mt() {

        let input_file = std::path::Path::new("data/dc/logo_iteration_1.json");
        let input = InputFormat::Dreamcoder.load_programs_and_tasks(input_file).unwrap();
    
        let train_programs: Vec<Expr> = input.train_programs.iter().map(|p| p.parse().unwrap()).collect();
        let train_programs: Expr = Expr::programs(train_programs);

        let step_results = compression(
            &train_programs,
            &None,
            10,
            &CompressionStepConfig::parse_from("compress -a2 ".split_whitespace()),
            &input.tasks,
            &input.prev_dc_inv_to_inv_strs);

        let singlethreaded_inventions = step_results.iter().take(10).map(|inv| inv.inv.body.to_string()).collect::<Vec<String>>();

        let step_results = compression(
            &train_programs,
            &None,
            10,
            &CompressionStepConfig::parse_from("compress -a2 -t4".split_whitespace()),
            &input.tasks,
            &input.prev_dc_inv_to_inv_strs);

        let multithreaded_inventions = step_results.iter().take(10).map(|inv| inv.inv.body.to_string()).collect::<Vec<String>>();

        // Assert single threaded results match multithreaded results
        assert_eq!(singlethreaded_inventions, multithreaded_inventions);
    }

    /**
     * Regression test.
     * Test whether context threading works by matching against expected inventions found
     * in data/basic/ctx_thread_1.json.
     */
    #[test]
    fn ctx_thread_1_match_expected() {
    
        let input_file = std::path::Path::new("data/basic/ctx_thread_1.json");
        let input = InputFormat::ProgramsList.load_programs_and_tasks(input_file).unwrap();
    
        let train_programs: Vec<Expr> = input.train_programs.iter().map(|p| p.parse().unwrap()).collect();
        let train_programs: Expr = Expr::programs(train_programs);

        // Run compression with the default argument values
        let step_results = compression(
            &train_programs,
            &None,
            10,
            &CompressionStepConfig::parse_from("compress".split_whitespace()),
            &input.tasks,
            &input.prev_dc_inv_to_inv_strs);

        let inventions = step_results.iter().take(2).map(|inv| inv.inv.body.to_string()).collect::<Vec<String>>();
        let expected_inventions = vec!["(+ #0 #0)", "(A (lam (lam (fn_0 (a b #0 $0 f)))))"];

        // Assert single threaded results match expected results
        assert_eq!(inventions, expected_inventions);
    }
}
