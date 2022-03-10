use stitch::*;
use criterion::*;
use serde_json::de::from_reader;
use std::fs::File;
use stitch::CompressionStepConfig;
use clap::Parser;

pub fn compression_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");
    group.sample_size(10);
    fn programs_from_file(file: &str) -> Expr {
        let programs: Vec<String> = from_reader(File::open(file).expect("file not found")).unwrap();
        let programs: Vec<Expr> = programs.iter().map(|p| p.parse().unwrap()).collect();
        Expr::programs(programs)
    }

    let programs_logo_train_200 = programs_from_file("data/logo/train_200.json");
    let programs_nuts_bolts = programs_from_file("data/cogsci/nuts-bolts.json");
    let programs_dials = programs_from_file("data/cogsci/dials.json");
    let programs_furniture = programs_from_file("data/cogsci/furniture.json");

    let cfg_arity2 = CompressionStepConfig::parse_from("compress --max-arity=2".split_whitespace());
    let cfg_arity3 = CompressionStepConfig::parse_from("compress --max-arity=3".split_whitespace());

    
    let tstart = std::time::Instant::now();
    group.bench_function("logo/train_200.json -i3 -a2", |b| b.iter(|| compression(black_box(&programs_logo_train_200), black_box(&cfg_arity2), black_box(3))));
    group.bench_function("cogsci/train_200.json -i3 -a3", |b| b.iter(|| compression(black_box(&programs_logo_train_200), black_box(&cfg_arity3), black_box(3))));
    // group.bench_function("cogsci/nuts-bolts.json -i3 -a2", |b| b.iter(|| compression(black_box(&programs_nuts_bolts), black_box(&cfg_arity2), black_box(3))));
    // group.bench_function("cogsci/nuts-bolts.json -i3 -a3", |b| b.iter(|| compression(black_box(&programs_nuts_bolts), black_box(&cfg_arity3), black_box(3))));
    // group.bench_function("cogsci/dials.json -i1 -a2", |b| b.iter(|| compression(black_box(&programs_dials), black_box(&cfg_arity2), black_box(1))));
    // group.bench_function("cogsci/furniture.json -i1 -a2", |b| b.iter(|| compression(black_box(&programs_furniture), black_box(&cfg_arity2), black_box(1))));
    group.finish();
    println!("Total `cargo bench` time: {}s", tstart.elapsed().as_secs());
}

criterion_group!(benches, compression_benchmark);
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