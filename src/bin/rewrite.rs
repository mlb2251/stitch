/*
rewrite.rs: Utility entrypoint for rewriting programs given a library.

Usage (see arguments in extraction.rs)
cargo run --bin=rewrite --release
    --program_file # Programs to rewrite
    --inventions_file # JSON containing inventions
    --out # Where to put the outptus

Sample command: cargo run --bin=rewrite --release -- --program-file data/test_111.json --inventions-file out/out.json
*/

use clap::Parser;
use dreamegg::*;
use serde_json::de::from_reader;
use serde_json::json;
use std::collections::HashMap;
use std::fs::File;

fn main() {
    procspawn::init();
    let args = ExtractionArgs::parse();

    // Read in library to rewrite.
    // This should be in {invs: [{name: , body:}]}
    let inventions_data: serde_json::Value =
        from_reader(File::open(&args.inventions_file).expect("file not found"))
            .expect("json deserializing error");
    let inventions = inventions_data["invs"].as_array().unwrap();

    let invention_names: Vec<String> = inventions
        .iter()
        .map(|invention| invention["name"].as_str().unwrap().parse().unwrap())
        .collect();

    let inventions: Vec<InventionExpr> = inventions
        .iter()
        .map(|invention| InventionExpr {
            body: invention["body"].as_str().unwrap().parse().unwrap(),
            arity: invention["arity"].as_u64().unwrap() as usize,
        })
        .collect();
    println!("Number of inventions: {}", inventions.len());

    inventions_info(&inventions);
    let invention_refs: Vec<&InventionExpr> = inventions.iter().map(|i| i).collect();
    let invention_refs: &[&InventionExpr] = &invention_refs[..];

    let name_refs: Vec<&str> = invention_names.iter().map(|n| n.as_str()).collect();
    let name_refs: &[&str] = &name_refs[..];

    // Read in the programs and maintain the frontier structure if we're using it.
    let mut programs: Vec<String> = Vec::new();
    let mut program_id_to_task_name: HashMap<usize, String> = HashMap::new();
    let mut rewritten_frontiers: HashMap<String, Vec<String>> = HashMap::new();
    if args.use_frontier_format {
        // Read in frontiers to programs
        let mut frontiers: HashMap<String, Vec<String>> =
            from_reader(File::open(&args.program_file).expect("file not found"))
                .expect("json deserializing error");
        println!("Read in {} frontiers", frontiers.len());

        // Read in frontiers to a list of programs, using the index as the ID.
        let mut program_id = 0;
        for (task_name, task_programs) in frontiers.into_iter() {
            for task_program in task_programs {
                program_id_to_task_name.insert(program_id, task_name.clone());
                program_id += 1;

                programs.push(task_program);
            }
        }
        println!("Read in {} programs from these frontiers", programs.len());
    } else {
        // Read in programs to rewrite.
        programs = from_reader(File::open(&args.program_file).expect("file not found"))
            .expect("json deserializing error");
    }

    let mut programs: Vec<Expr> = programs.iter().map(|p| p.parse().unwrap()).collect();
    programs_info(&programs);

    let programs: Expr = Expr::programs(programs);
    let mut rewritten: Expr = programs.clone();
    for i in 0..inventions.len() {
        let mut egraph: EGraph = Default::default();
        let programs_expr = &rewritten;
        let programs_node = egraph.add_expr(programs_expr.into());
        egraph.rebuild();
        let inv = &inventions[i];
        let inv_name = &invention_names[i];
        println!("Rewrote with invention: {}", inv_name);
        rewritten =
            rewrite_with_inventions(programs_node, &[&inv], &[inv_name.as_str()], &mut egraph);
    }

    if args.use_frontier_format {
        for (i, pretty_program) in pretty_programs(&rewritten).iter().enumerate() {
            let task_name = program_id_to_task_name[&i].clone();
            rewritten_frontiers
                .entry(task_name)
                .or_insert(Vec::new())
                .push(pretty_program.clone());
        }
        let json: serde_json::Value = json!(rewritten_frontiers);
        std::fs::write(&args.out, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    } else {
        let json: serde_json::Value = json!({ "rewritten": pretty_programs(&rewritten) });
        std::fs::write(&args.out, serde_json::to_string_pretty(&json).unwrap()).unwrap();
    }
}
