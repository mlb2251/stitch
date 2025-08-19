use std::time::Duration;
use std::thread;
use std::sync::mpsc;

use rand_chacha::rand_core::{RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_chacha; // 0.3.0
use stitch_core::*;
use stitch_core::test_utils::compare_out_jsons_testing;
use std::fs;
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
    test_files.sort(); // Sort for consistent order
    test_files
}

fn select_random_file<'a>(files: &'a [std::path::PathBuf], rng: &mut ChaCha8Rng) -> String {
    // files.choose(rng).unwrap().to_str().unwrap().to_owned()
    let index = rng.next_u64() as usize % files.len();
    files[index].to_str().unwrap().to_owned()
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

fn select_random_symbols<'a>(symbols: &'a [String], n: usize, rng: &mut ChaCha8Rng) -> Vec<&'a String> {
    let mut selected = Vec::new();
    let mut indices: Vec<usize> = (0..symbols.len()).collect();
    while selected.len() < n && !indices.is_empty() {
        let index = rng.next_u64() as usize % indices.len();
        selected.push(&symbols[indices[index]]);
        indices.remove(index);
    }
    selected
}

fn generate_random_weights(selected_symbols: &[&String], rng: &mut ChaCha8Rng) -> serde_json::Value {
    let mut cost_prims = serde_json::json!({});
    for symbol in selected_symbols {
        let weight = if rng.next_u64() % 2 == 0 {
            rng.next_u64() % 100 + 1 // Random weight between 1 and 100
        } else {
            rng.next_u64() % 1000 + 101 // Random weight between 101 and 1100
        };
        cost_prims[symbol] = serde_json::json!(weight);
    }
    cost_prims
}

fn run_fuzz_compression(cost_prims: &serde_json::Value, test_file: &String, seed: u64) {
    let cost_prims_str = cost_prims.to_string();
    let args = format!("-i3 -a3 --verbose-best --cost-prim {}", shlex::try_quote(&cost_prims_str).unwrap());
    let output_file = format!("data/expected_outputs/fuzz/{:0>3}_{}.json", seed, test_file.split("/").last().unwrap().split(".").next().unwrap());
    compare_out_jsons_testing(test_file, &output_file, &args, InputFormat::ProgramsList);
}

#[test_matrix(1..=100)]
fn fuzz_test_symbol_weighting(seed: u64) {
    println!("\nRunning fuzz test with seed {}", seed);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let test_files = collect_test_files();
    let test_file = select_random_file(&test_files, &mut rng);
    let input = InputFormat::ProgramsList.load_programs_and_tasks(std::path::Path::new(&test_file)).unwrap();
    let symbols = extract_unique_symbols(&input);
    let mut symbols_vec: Vec<_> = symbols.into_iter().collect();
    symbols_vec.sort(); // Sort symbols for consistent selection
    let selected_symbols = select_random_symbols(&symbols_vec, 3, &mut rng);
    let cost_prims = generate_random_weights(&selected_symbols, &mut rng);

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _output = run_fuzz_compression(&cost_prims, &test_file, seed);
        tx.send(()).unwrap();
    });

    match rx.recv_timeout(Duration::from_secs(600)) {
        Ok(_) => (),
        Err(_) => panic!("Test timed out after 600 seconds"),
    }
}
