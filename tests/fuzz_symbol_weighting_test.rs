use std::time::Duration;
use std::thread;
use std::sync::mpsc;

use stitch_core::*;
use clap::Parser;
use std::fs;
use rand::prelude::*;
use test_case::test_matrix;


fn collect_test_files() -> Vec<std::path::PathBuf> {
    let mut test_files = Vec::new();
    for dir in ["data/cogsci", "data/basic"] {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                if entry.path().extension().map_or(false, |ext| ext == "json") {
                    test_files.push(entry.path());
                }
            }
        }
    }
    test_files
}

fn select_random_file<'a>(files: &'a [std::path::PathBuf], rng: &mut impl Rng) -> &'a std::path::Path {
    files.choose(rng).unwrap().as_path()
}

fn extract_unique_symbols(input: &Input) -> std::collections::HashSet<String> {
    let mut symbols = std::collections::HashSet::new();
    for program in &input.train_programs {
        let mut set = ExprSet::empty(Order::ChildFirst, false, false);
        let idx = set.parse_extend(program).unwrap();
        let expr = ExprOwned::new(set, idx);
        for i in 0..expr.set.len() {
            let node = &expr.set[i];
            if let Node::Prim(sym) = node {
                symbols.insert(sym.to_string());
            }
        }
    }
    symbols
}

fn select_random_symbols<'a>(symbols: &'a [String], n: usize, rng: &mut impl Rng) -> Vec<&'a String> {
    let mut indices: Vec<usize> = (0..symbols.len()).collect();
    indices.shuffle(rng);
    indices.into_iter().take(n).map(|i| &symbols[i]).collect()
}

fn generate_random_weights(selected_symbols: &[&String], rng: &mut impl Rng) -> serde_json::Value {
    let mut cost_prims = serde_json::json!({});
    for symbol in selected_symbols {
        let weight = if rng.gen_bool(0.5) {
            rng.gen_range(1..100)
        } else {
            rng.gen_range(100..1000)
        };
        cost_prims[symbol] = serde_json::json!(weight);
    }
    cost_prims
}

fn run_fuzz_compression(input: &Input, cost_prims: &serde_json::Value, test_file: &std::path::Path, weights: Option<Vec<f32>>) {
    let cost_prims_str = cost_prims.to_string();
    let args = vec!["compress", "-i1", "-a3", "--verbose-best", "--cost-prim", &cost_prims_str];
    println!("Running fuzz test with command: {}", args.join(" "));
    println!("Test file: {}", test_file.display());
    let cfg = MultistepCompressionConfig::parse_from(args);
    let _output = run_compression_testing_weighted(input, &cfg, weights);
}

#[test_matrix(1..=100)]
fn fuzz_test_symbol_weighting(seed: u64) {
    println!("\nRunning fuzz test with seed {}", seed);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let test_files = collect_test_files();
    let test_file = select_random_file(&test_files, &mut rng);
    let test_file = test_file.to_path_buf();
    let input = InputFormat::ProgramsList.load_programs_and_tasks(&test_file).unwrap();
    let symbols = extract_unique_symbols(&input);
    let mut symbols_vec: Vec<_> = symbols.into_iter().collect();
    symbols_vec.sort(); // Sort symbols for consistent selection
    let selected_symbols = select_random_symbols(&symbols_vec, 3, &mut rng);
    let cost_prims = generate_random_weights(&selected_symbols, &mut rng);

    let weights = if rng.gen_bool(0.5) {
        Some(input.train_programs.iter().map(|_| rng.gen_range(0.0..=2.0) as f32).collect())
    } else {
        None
    };

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        run_fuzz_compression(&input, &cost_prims, &test_file, weights);
        tx.send(()).unwrap();
    });

    match rx.recv_timeout(Duration::from_secs(300)) {
        Ok(_) => (),
        Err(_) => panic!("Test timed out after 300 seconds"),
    }
}
