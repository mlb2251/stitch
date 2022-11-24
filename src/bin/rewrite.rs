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

// Args for rewrite.rs, which calls `rewrite_with_inventions`.
#[derive(Parser, Debug, Serialize)]
#[clap(name = "Rewrite")]
pub struct RewriteArgs {
    /// json file to read compression programs from.
    #[clap(
        short,
        long,
        parse(from_os_str),
        default_value = "data/logo/test_111.json"
    )]
    pub program_file: PathBuf,

    /// Compression output to read the inventions from. Can use the compres
    #[clap(short, long, parse(from_os_str), default_value = "out/out.json")]
    pub inventions_file: PathBuf,

    /// json output file
    #[clap(
        short,
        long,
        parse(from_os_str),
        default_value = "out/extraction_out.json"
    )]
    pub out: PathBuf,

    /// the format of the input file, e.g. 'programs-list' for a simple JSON array of programs
    /// or 'dreamcoder' for a JSON in the style expected by the original dreamcoder codebase.
    /// See [formats.rs] for options or to add new ones.
    #[clap(long, arg_enum, default_value = "programs-list")]
    pub fmt: InputFormat,

    /// return the rewritten programs in dreamcoder (#(lambda)) format
    #[clap(long)]
    pub dreamcoder_output: bool,
}

fn main() {
    let args = RewriteArgs::parse();

    let cost_fn = ExprCost::dreamcoder();

    // Read in the programs and any previous inventions from the DSL.
    let mut input = args
        .fmt
        .load_programs_and_tasks(&args.program_file)
        .unwrap();

    // Read in library to rewrite.
    // This should be in {invs: [{name: , body:}]}
    let inventions_data: serde_json::Value =
        from_reader(File::open(&args.inventions_file).expect("file not found"))
            .expect("json deserializing error");
    let inventions = inventions_data["invs"].as_array().unwrap();

    let mut dreamcoder_translation: Vec<(String,String)> = inventions.iter().map(|invention| (invention["name"].as_str().unwrap().to_string(),invention["dreamcoder"].as_str().unwrap().to_string())).collect();

    // Add in any previous inventions, and re-order this by string length again.
    dreamcoder_translation.append(&mut input.prev_dc_inv_to_inv_strs);
    dreamcoder_translation.sort_by_key(|(_, dc_name)| dc_name.len());

    let inventions: Vec<Invention> = inventions
        .iter()
        .map(|invention| Invention {
            body: {
                let mut set = ExprSet::empty(Order::ChildFirst, false, false);
                let idx = set.parse_extend(invention["body"].as_str().unwrap()).unwrap();
                ExprOwned::new(set, idx)
            },
            arity: invention["arity"].as_u64().unwrap() as usize,
            name: invention["name"].as_str().unwrap().parse().unwrap(),
        })
        .collect();
    println!("Number of inventions: {}", inventions.len());

    let mut rewritten_frontiers: HashMap<String, Vec<String>> = HashMap::new();

    let programs: Vec<ExprOwned> = input.train_programs.iter().map(|p|{
        let mut set = ExprSet::empty(Order::ChildFirst, false, false);
        let idx = set.parse_extend(p).unwrap();
        ExprOwned::new(set,idx)
    }).collect();
    programs_info(&programs, &cost_fn);

    let cfg = CompressionStepConfig::parse_from("compress".split_whitespace());
    let rewritten: Vec<ExprOwned> = rewrite_with_inventions(&programs, &inventions[..], &cfg);

    match args.fmt {
        InputFormat::Dreamcoder => {
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

            // Rewrite back the lambda and optionally rewrite back the DC invention format.
            for (i, pretty_program) in rewritten.iter().enumerate() {
                let task_name = input.tasks.clone().map(|tasks| tasks[i].clone()).unwrap_or_else(||i.to_string());
                let mut pretty_program = pretty_program.to_string();
                if args.dreamcoder_output {
                    for (name, dc_translation) in dreamcoder_translation.iter().rev() {
                        pretty_program = replace_prim_with(&pretty_program, name, dc_translation);
                    }
                }

                rewritten_frontiers
                    .entry(task_name)
                    .or_insert_with(Vec::new)
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

            let json: serde_json::Value = json!({ "frontiers": dc_fmt_frontiers });
            std::fs::write(&args.out, serde_json::to_string_pretty(&json).unwrap()).unwrap();
        },
        InputFormat::ProgramsList => {
            let json: serde_json::Value = json!({ "rewritten": rewritten.iter().map(|p| p.to_string()).collect::<Vec<String>>() });
            std::fs::write(&args.out, serde_json::to_string_pretty(&json).unwrap()).unwrap();
        },
        InputFormat::SplitProgramsList => {
            panic!("SplitProgramsList is not a valid format for --bin=rewrite")
        }
    }
}
