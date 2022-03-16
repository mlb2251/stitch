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
use stitch::*;

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
}

fn main() {
    procspawn::init();
    let args = RewriteArgs::parse();

    // Read in library to rewrite.
    // This should be in {invs: [{name: , body:}]}
    let inventions_data: serde_json::Value =
        from_reader(File::open(&args.inventions_file).expect("file not found"))
            .expect("json deserializing error");
    let inventions = inventions_data["invs"].as_array().unwrap();

    let inventions: Vec<Invention> = inventions
        .iter()
        .map(|invention| Invention {
            body: invention["body"].as_str().unwrap().parse().unwrap(),
            arity: invention["arity"].as_u64().unwrap() as usize,
            name: invention["name"].as_str().unwrap().parse().unwrap(),
        })
        .collect();
    println!("Number of inventions: {}", inventions.len());

    // Read in the programs. Maintain the DC format if we're using it.
    let mut programs: Vec<String> = Vec::new();
    let mut program_id_to_task_name: HashMap<usize, String> = HashMap::new();
    let mut rewritten_frontiers: HashMap<String, Vec<String>> = HashMap::new();
    match args.fmt {
        InputFormat::Dreamcoder => {
            // Read in frontiers to programs, but preserve their original task.
            let json: serde_json::Value =
                from_reader(File::open(&args.program_file).expect("file not found"))
                    .expect("json deserializing error");
            let frontiers = json["frontiers"].as_array().unwrap();
            println!("Read in {} frontiers", frontiers.len());
            let mut program_id = 0;
            for frontier in frontiers.into_iter() {
                let task_name = frontier["task"].as_str().unwrap().to_string();
                for dc_program in frontier["programs"].as_array().unwrap() {
                    let stitch_program = dc_program["program"]
                        .as_str()
                        .unwrap()
                        .to_string()
                        .replace("(lambda ", "(lam ");
                    program_id_to_task_name.insert(program_id, task_name.clone());
                    program_id += 1;
                    programs.push(stitch_program);
                }
            }
            println!("Read in {} programs from these frontiers", programs.len());
        },
        InputFormat::ProgramsList => {
            // Read in programs to rewrite.
            programs = from_reader(File::open(&args.program_file).expect("file not found"))
                .expect("json deserializing error");
        }
    }

    let programs: Vec<Expr> = programs.iter().map(|p| p.parse().unwrap()).collect();
    programs_info(&programs);

    let programs: Expr = Expr::programs(programs);
    let rewritten = rewrite_with_inventions(programs, &inventions[..]);

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
            for (i, pretty_program) in rewritten.split_programs().iter().enumerate() {
                let task_name = program_id_to_task_name[&i].clone();
                rewritten_frontiers
                    .entry(task_name)
                    .or_insert(Vec::new())
                    .push(
                        pretty_program
                            .to_string()
                            .replace("(lam ", "(lambda ")
                            .clone(),
                    );
            }
            fn rewritten_to_dc_fmt_frontiers(
                task_name: &String,
                string_programs: &Vec<String>,
            ) -> DcFrontier {
                let programs = string_programs
                    .iter()
                    .map(|p| DcProgram {
                        program: p.to_string(),
                    })
                    .collect();
                let frontier = DcFrontier {
                    task: task_name.to_string(),
                    programs: programs,
                };
                frontier
            }
            let dc_fmt_frontiers: Vec<DcFrontier> = rewritten_frontiers
                .iter()
                .map(|(t, ps)| rewritten_to_dc_fmt_frontiers(t, ps))
                .collect();

            let json: serde_json::Value = json!({ "frontiers": dc_fmt_frontiers });
            std::fs::write(&args.out, serde_json::to_string_pretty(&json).unwrap()).unwrap();
        },
        InputFormat::ProgramsList => {
            let json: serde_json::Value = json!({ "rewritten": rewritten.split_programs().iter().map(|p| p.to_string()).collect::<Vec<String>>() });
            std::fs::write(&args.out, serde_json::to_string_pretty(&json).unwrap()).unwrap();
        }
    }
}
