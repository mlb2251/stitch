// use stitch::*;
use criterion::*;
// use serde_json::de::from_reader;
// use std::fs::File;
// use stitch::CompressionStepConfig;
// use clap::Parser;

pub fn compression_benchmark(_c: &mut Criterion) {
    // let mut group = c.benchmark_group("compression");
    // group.sample_size(20);
    // fn programs_from_file(file: &str) -> Expr {
    //     let programs: Vec<String> = from_reader(File::open(file).expect("file not found")).unwrap();
    //     let programs: Vec<Expr> = programs.iter().map(|p| p.parse().unwrap()).collect();
    //     Expr::programs(programs)
    // }

    // // load programs
    // let programs_logo_train_200 = programs_from_file("data/logo/train_200.json");
    // let programs_nuts_bolts = programs_from_file("data/cogsci/nuts-bolts.json");
    // let programs_dials = programs_from_file("data/cogsci/dials.json");
    // let programs_furniture = programs_from_file("data/cogsci/furniture.json");

    // // build cfgs
    // let cfg_arity2_t1 = CompressionStepConfig::parse_from("compress --max-arity=2 -t1".split_whitespace());
    // let cfg_arity2 = CompressionStepConfig::parse_from("compress --max-arity=2 -t4".split_whitespace());
    // let cfg_arity3 = CompressionStepConfig::parse_from("compress --max-arity=3 -t4".split_whitespace());

    // // run benchmarks
    // let tstart = std::time::Instant::now();
    // group.bench_function("cogsci/nuts-bolts.json -i3 -a2 single-threaded", |b| b.iter(|| compression(black_box(&programs_logo_train_200), black_box(3), black_box(&cfg_arity2_t1))));
    // group.bench_function("logo/train_200.json -t4 -i3 -a2", |b| b.iter(|| compression(black_box(&programs_logo_train_200), black_box(3), black_box(&cfg_arity2))));
    // group.bench_function("cogsci/train_200.json -t4 -i3 -a3", |b| b.iter(|| compression(black_box(&programs_logo_train_200), black_box(3), black_box(&cfg_arity3))));
    // group.bench_function("cogsci/nuts-bolts.json -t4 -i3 -a2", |b| b.iter(|| compression(black_box(&programs_nuts_bolts), black_box(3), black_box(&cfg_arity2))));
    // group.bench_function("cogsci/nuts-bolts.json -t4 -i3 -a3", |b| b.iter(|| compression(black_box(&programs_nuts_bolts), black_box(3), black_box(&cfg_arity3))));
    // group.bench_function("cogsci/dials.json -t4 -i1 -a2", |b| b.iter(|| compression(black_box(&programs_dials), black_box(1), black_box(&cfg_arity2))));
    // group.bench_function("cogsci/furniture.json -t4 -i1 -a2", |b| b.iter(|| compression(black_box(&programs_furniture), black_box(1), black_box(&cfg_arity2))));
    // group.finish();
    // println!("Total `cargo bench` time: {}s", tstart.elapsed().as_secs());
    // println!("{}/target/criterion/compression/report/index.html", std::env::current_dir().unwrap().to_str().unwrap())
}

criterion_group!(benches, compression_benchmark);
criterion_main!(benches);