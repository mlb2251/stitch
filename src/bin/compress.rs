use stitch_core::*;
use clap::Parser;
use serde::Serialize;
use std::path::PathBuf;

/// Stitch
#[derive(Parser, Debug, Serialize)]
#[clap(name = "Stitch")]
pub struct Args {
    /// json file to read compression input programs from
    #[clap(parse(from_os_str))]
    pub file: PathBuf,

    /// json output file
    #[clap(short, long, parse(from_os_str), default_value = "out/out.json")]
    pub out: PathBuf,

    /// the format of the input file, e.g. 'programs-list' for a simple JSON array of programs
    /// or 'dreamcoder' for a JSON in the style expected by the original dreamcoder codebase.
    /// See [formats.rs] for options or to add new ones.
    #[clap(long, arg_enum, default_value = "programs-list")]
    pub fmt: InputFormat,

    /// saves the rewritten frontiers in an input-readable format in a separate json from the normal output json
    #[clap(long)]
    pub save_rewritten: Option<PathBuf>,    

    #[clap(flatten)]
    pub multistep: MultistepCompressionConfig,

}

fn main() {
    let args = Args::parse();

    let input = args.fmt.load_programs_and_tasks(&args.file).unwrap();

    let (step_results, json_res) = multistep_compression(&input.train_programs, input.tasks, None, input.name_mapping, None, &args.multistep);

    let out_path = &args.out;
    if let Some(out_path_dir) = out_path.parent() {
        if !out_path_dir.exists() {
            std::fs::create_dir_all(out_path_dir).unwrap();
        }
    }

    std::fs::write(out_path, serde_json::to_string_pretty(&json_res).unwrap()).unwrap();
    if !args.multistep.silent{ println!("Wrote to {out_path:?}") };
    if let Some(out_path) = args.save_rewritten {
        if !args.multistep.silent{ println!("Wrote rewritten things to {out_path:?}") };
        std::fs::write(&out_path, serde_json::to_string_pretty(&step_results.iter().last().unwrap().rewritten.iter().map(|p| p.to_string()).collect::<Vec<String>>()).unwrap()).unwrap();
    }

}




