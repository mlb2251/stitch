use itertools::Itertools;
use stitch::*;
// extern crate log;
use clap::Parser;
use serde::Serialize;
use std::{path::PathBuf, str::FromStr};



/// Compression
#[derive(Parser, Debug, Serialize)]
#[clap(name = "Stitch")]
pub struct Args {
    /// json file to read compression input programs from
    #[clap(parse(from_os_str))]
    pub file: PathBuf,

    /// the format of the input file, e.g. 'programs-list' for a simple JSON array of programs
    /// or 'dreamcoder' for a JSON in the style expected by the original dreamcoder codebase.
    /// See [formats.rs] for options or to add new ones.
    #[clap(long, arg_enum, default_value = "programs-list")]
    pub fmt: InputFormat,

    // The path to write the chunked programs to
    #[clap(long, short='o', parse(from_os_str))]
    pub out: Option<PathBuf>,

    // The number of chunks to split the dataset into
    #[clap(long, short='n')]
    pub num_chunks: usize,

}

fn main() {

    let args = Args::parse();
    let (programs, _, _, _) = args.fmt.load(&args.file, false).unwrap();

    let programs_sorted_depth = programs.clone().into_iter().sorted_by_key(|p| p.parse::<Expr>().unwrap().depth()).collect::<Vec<String>>();
    let programs_sorted_length = programs.clone().into_iter().sorted_by_key(|p| p.parse::<Expr>().unwrap().length()).collect::<Vec<String>>();

    println!("Number of programs (unchunked): {}", programs.len());
    println!("Number of chunks: {}", args.num_chunks);
    println!("Chunking will split dataset perfectly: {}", programs.len() % args.num_chunks == 0);

    let out_path_dir = &args.out.unwrap_or(PathBuf::from_str(format!("{}-chunked", args.file.file_stem().unwrap().to_str().unwrap()).as_str()).unwrap());
    if !out_path_dir.exists() {
        std::fs::create_dir_all(out_path_dir).unwrap();
    }

    for (i, chunk) in programs_sorted_depth.chunks(programs_sorted_depth.len() / args.num_chunks).enumerate() {
        let mut path: PathBuf = out_path_dir.clone();
        path.push("depth");
        std::fs::create_dir(&path);
        path.push(usize::to_string(&i));
        std::fs::write(path, serde_json::to_string_pretty(chunk).unwrap()).unwrap();
    }
    for (i, chunk) in programs_sorted_length.chunks(programs_sorted_length.len() / args.num_chunks).enumerate() {
        let mut path: PathBuf = out_path_dir.clone();
        path.push("length");
        std::fs::create_dir(&path);
        path.push(usize::to_string(&i));
        std::fs::write(path, serde_json::to_string_pretty(chunk).unwrap()).unwrap();
    }

    println!("Wrote to {:?}", out_path_dir.as_os_str());
}