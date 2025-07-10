use stitch_core::*;
use std::path::PathBuf;
use std::time::Instant;
use std::fs;
use clap::Parser;
use rand::seq::SliceRandom;
use rand::thread_rng;

#[derive(Parser, Debug)]
#[clap(name = "benchmark_cogsci")]
struct Args {
    /// Number of repetitions
    #[clap(short, long, default_value = "1")]
    count: usize,
}

fn main() {
    let args = Args::parse();
    let mut geomeans = Vec::with_capacity(args.count);
    for _ in 0..args.count {
        let geo = benchmark_cogsci_geomean();
        println!("{geo:.2}");
        geomeans.push(geo);
    }
    if geomeans.len() > 1 {
        let (mean, lower, upper) = bootstrap_mean_ci(&geomeans, 10_000, 0.05);
        println!("Summary (95% CI, arithmetic mean): {mean:.2} [{lower:.2}, {upper:.2}]");
    }
}

fn benchmark_cogsci_geomean() -> f64 {
    let cogsci_dir = PathBuf::from("data/cogsci");
    // Check if directory exists
    if !cogsci_dir.exists() {
        eprintln!("Error: data/cogsci directory does not exist");
        std::process::exit(1);
    }
    // Get all JSON files in the directory
    let mut json_files = Vec::new();
    for entry in fs::read_dir(&cogsci_dir).expect("Failed to read data/cogsci directory") {
        let entry = entry.expect("Failed to read directory entry");
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "json") {
            let fname = path.file_name().unwrap().to_string_lossy();
            if fname == "castle.json" || fname == "city.json" {
                continue;
            }
            json_files.push(path);
        }
    }
    if json_files.is_empty() {
        eprintln!("Error: No JSON files found in data/cogsci");
        std::process::exit(1);
    }
    // Configuration for compression: -a3 -i10
    let mut cfg = MultistepCompressionConfig::default();
    cfg.step.max_arity = 3;  // -a3
    cfg.iterations = 10;     // -i10
    cfg.silent = true;       // Reduce output noise
    let mut compression_times = Vec::new();
    for file_path in json_files.iter() {
        // Load and parse the file (this time is NOT counted)
        let input = match InputFormat::ProgramsList.load_programs_and_tasks(file_path) {
            Ok(input) => input,
            Err(_) => {
                continue;
            }
        };
        // Start timing the compression (excluding file loading and JSON parsing)
        let start_time = Instant::now();
        // Run compression
        multistep_compression(
            &input.train_programs,
            input.tasks,
            None,
            input.name_mapping,
            None,
            &cfg
        );
        let compression_duration = start_time.elapsed();
        let compression_time_ms = compression_duration.as_millis() as f64;
        compression_times.push(compression_time_ms);
    }
    geometric_mean(&compression_times)
}

fn geometric_mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let product: f64 = data.iter().product();
    product.powf(1.0 / data.len() as f64)
}

fn bootstrap_mean_ci(data: &[f64], n_bootstrap: usize, alpha: f64) -> (f64, f64, f64) {
    let mut rng = thread_rng();
    let n = data.len();
    let mut means = Vec::with_capacity(n_bootstrap);
    for _ in 0..n_bootstrap {
        let sample: Vec<f64> = (0..n).map(|_| *data.choose(&mut rng).unwrap()).collect();
        let mean = sample.iter().sum::<f64>() / n as f64;
        means.push(mean);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lower_idx = ((alpha / 2.0) * n_bootstrap as f64).round() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64).round() as usize;
    let mean = data.iter().sum::<f64>() / n as f64;
    (
        mean,
        means[lower_idx.min(n_bootstrap - 1)],
        means[upper_idx.min(n_bootstrap - 1)],
    )
}
