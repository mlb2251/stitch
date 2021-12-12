use dreamegg::*;
use std::fs::File;
use serde_json::de::from_reader;
// extern crate log;
use clap::Parser;


fn main() {
    procspawn::init();

    let args = CompressionArgs::parse();

    // create a new directory for logging outputs
    let out_dir: String = format!("target/{}",timestamp());
    let out_dir_p = std::path::Path::new(out_dir.as_str());
    assert!(!out_dir_p.exists());
    std::fs::create_dir(out_dir_p).unwrap();

    let mut programs: Vec<String> = from_reader(File::open(&args.file).expect("file not found")).expect("json deserializing error");
    programs.sort();
    programs.dedup();
    let mut programs: Vec<Expr> = programs.iter().map(|p| p.parse().unwrap()).collect();

    for prog in programs.iter() {
        println!("{}", prog);
    }

    // programs.sort_by(|a,b| a.cost().cmp(&b.cost()));
    

    programs_info(&programs);

    let programs: Expr = Expr::programs(programs);

    // todo this may not be an issue actually, just want this here to look out for it
    assert!(!programs.to_string_curried(None).contains("(app (lam"),
        "Normal dreamcoder programs never have unapplied lambdas in them! 
         Who knows what might happen if you run this. Side note you can probably
         inline them and it'd be fine (we've got a function for that!) and also
         who knows maybe it wouldnt be an issue in the first place");

    compression(&programs, &args, &out_dir);
}