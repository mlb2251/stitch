use criterion::{black_box, criterion_group, criterion_main, Criterion};
use stitch::*;
use serde_json::de::from_reader;
use std::fs::File;
use stitch::CompressionStepConfig;
use clap::Parser;

pub fn criterion_benchmark(c: &mut Criterion) {

    let programs: Vec<String> = from_reader(File::open("data/logo/train_200.json").expect("file not found")).unwrap();
    let programs: Vec<Expr> = programs.iter().map(|p| p.parse().unwrap()).collect();
    let programs: Expr = Expr::programs(programs);

    let args: Vec<&str> = "compress -a3".split_whitespace().collect();
    let cfg = CompressionStepConfig::parse_from(args);
    let iterations = 3;

    
    c.bench_function("compression logo/train_200.json", |b| b.iter(|| compression(black_box(&programs), black_box(&cfg), black_box(iterations))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);



fn compression(
    programs_expr: &Expr,
    cfg: &CompressionStepConfig,
    iterations: usize,
) -> Vec<CompressionStepResult> {
    let mut rewritten: Expr = programs_expr.clone();
    let mut step_results: Vec<CompressionStepResult> = Default::default();

    for _ in 0..iterations {
        let inv_name = format!("inv{}",step_results.len());

        // call actual compression
        let res: Vec<CompressionStepResult> = compression_step(
            &rewritten,
            &inv_name,
            cfg,
            &step_results);

        if !res.is_empty() {
            // rewrite with the invention
            let res: CompressionStepResult = res[0].clone();
            rewritten = res.rewritten.clone();
            step_results.push(res);
        } else {
            break;
        }
    }

    step_results
}