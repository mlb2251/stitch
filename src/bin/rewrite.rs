/*
rewrite.rs: Utility entrypoint for rewriting programs given a library.

Usage (see arguments in extraction.rs)
cargo run --bin=rewrite --release
    --program_file # Programs to rewrite
    --inventions_file # JSON containing inventions
    --out # Where to put the outputs.
    --dc_fmt # Functions are written in the DreamCoder frontiers file format.

Sample command: cargo run --bin=rewrite --release -- --program-file data/logo/logo_dc.json --inventions-file out/out.json --dc-fmt
*/

use clap::Parser;
use serde::Serialize;
use serde_json::de::from_reader;
use serde_json::json;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use stitch_core::*;
use serde_json::Value;

// Args for rewrite.rs, which calls `rewrite_with_inventions`.
#[derive(Parser, Debug, Serialize)]
#[clap(name = "Rewrite")]
pub struct RewriteArgs {
    /// json file to read compression programs from.
    #[clap(short, long, parse(from_os_str), default_value = "data/logo/test_111.json")]
    pub program_file: PathBuf,

    /// Compression output to read the inventions from. Can use the compres
    #[clap(short, long, parse(from_os_str), default_value = "out/out.json")]
    pub inventions_file: PathBuf,

    /// json output file
    #[clap(short, long, parse(from_os_str), default_value = "out/extraction_out.json")]
    pub out: PathBuf,

    /// the format of the input file, e.g. 'programs-list' for a simple JSON array of programs
    /// or 'dreamcoder' for a JSON in the style expected by the original dreamcoder codebase.
    /// See [formats.rs] for options or to add new ones.
    #[clap(long, arg_enum, default_value = "programs-list")]
    pub fmt: InputFormat,

    /// return the rewritten programs in dreamcoder (#(lambda)) format
    #[clap(long)]
    pub dreamcoder_output: bool,

    #[clap(flatten)]
    pub cost: MultistepCompressionConfig,
}

// Match the relevant input format.
#[derive(Serialize)]
pub struct DcProgram {
    pub program: String,
}
#[derive(Serialize)]
pub struct DcFrontier {
    pub task: String,
    pub programs: Vec<DcProgram>,
}

fn main() {
    let args = RewriteArgs::parse();

    // Read in the programs and any previous inventions from the DSL.
    let input = args
        .fmt
        .load_programs_and_tasks(&args.program_file)
        .unwrap();

    // Read in library to rewrite.
    // This should be in {abstractions: [{name: , body:}]}
    let inventions_data: Value =
        from_reader(File::open(&args.inventions_file).expect("file not found"))
            .expect("json deserializing error");
    let inventions = inventions_data["abstractions"].as_array().unwrap();


    let mut name_mapping = input.name_mapping.clone().unwrap_or_default();
    name_mapping.extend(inventions.iter().map(|invention| (invention["name"].as_str().unwrap().to_string(),invention["dreamcoder"].as_str().unwrap().to_string())));
    name_mapping.sort_by_key(|(_, dc_name)| dc_name.len());

    let inventions: Vec<Invention> = inventions
        .iter()
        .map(Invention::from_compression_output)
        .collect();
     println!("Number of inventions: {}", inventions.len());

    let mut rewritten_frontiers: HashMap<String, Vec<String>> = HashMap::new();

    
    let rewritten: Vec<String> = rewrite_with_inventions(&input.train_programs, &inventions[..], &args.cost).0;

    match args.fmt {
        InputFormat::Dreamcoder => {

            // Rewrite back the lambda and optionally rewrite back the DC invention format.
            for (i, pretty_program) in rewritten.iter().enumerate() {
                let task_name = input.tasks.clone().map(|tasks| tasks[i].clone()).unwrap_or_else(||i.to_string());
                let mut pretty_program = pretty_program.to_string();
                if args.dreamcoder_output {
                    for (name, dc_translation) in name_mapping.iter().rev() {
                        pretty_program = replace_prim_with(&pretty_program, name, dc_translation);
                    }
                }

                rewritten_frontiers
                    .entry(task_name)
                    .or_default()
                    .push(
                        pretty_program
                            .replace("(lam ", "(lambda ")
                            .clone(),
                    );
            }
            fn rewritten_to_dc_fmt_frontiers(
                task_name: &str,
                string_programs: &[String],
            ) -> DcFrontier {
                let programs = string_programs
                    .iter()
                    .map(|p| DcProgram {
                        program: p.to_string(),
                    })
                    .collect();
                
                DcFrontier {
                    task: task_name.to_string(),
                    programs,
                }
            }
            let dc_fmt_frontiers: Vec<DcFrontier> = rewritten_frontiers
                .iter()
                .map(|(t, ps)| rewritten_to_dc_fmt_frontiers(t, ps))
                .collect();

            let json: Value = json!({ "frontiers": dc_fmt_frontiers });
            std::fs::write(&args.out, serde_json::to_string_pretty(&json).unwrap()).unwrap();
        },
        InputFormat::ProgramsList => {
            let json: Value = json!({ "rewritten": rewritten.iter().map(|p| p.to_string()).collect::<Vec<String>>() });
            std::fs::write(&args.out, serde_json::to_string_pretty(&json).unwrap()).unwrap();
        }
    }
}
