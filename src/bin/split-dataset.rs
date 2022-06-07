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

    // The size of the window of programs to consider
    #[clap(long, short='w')]
    pub window_size: usize,

    // How many programs to jump when sliding the window.
    #[clap(long, short='s')]
    pub slide_length: usize,

}

fn main() {

    let args = Args::parse();
    let (programs, _, _, _) = args.fmt.load(&args.file, false).unwrap();

    let programs_sorted_depth = programs.clone().into_iter().sorted_by_key(|p| p.parse::<Expr>().unwrap().depth()).collect::<Vec<String>>();
    let programs_sorted_length = programs.clone().into_iter().sorted_by_key(|p| p.parse::<Expr>().unwrap().length()).collect::<Vec<String>>();

    println!("Number of programs: {}", programs.len());
    println!("Window size: {}", args.window_size);
    println!("Slide length: {}", args.slide_length);

    let out_path_dir = &args.out.unwrap_or(PathBuf::from_str(format!("{}-chunked", args.file.file_stem().unwrap().to_str().unwrap()).as_str()).unwrap());
    if !out_path_dir.exists() {
        std::fs::create_dir_all(out_path_dir).unwrap();
    }

    let mut window_start = 0;
    while window_start <= programs.len() - args.slide_length {
        let window_end = std::cmp::min(programs.len(), window_start + args.slide_length) - 1;

        // depth
        let mut path: PathBuf = out_path_dir.clone();
        path.push("depth");
        if !path.exists() { std::fs::create_dir(&path).unwrap(); }
        path.push(format!("{}-{}.json", &window_start, &window_end));
        let window = programs_sorted_depth.get(window_start..window_end+1).unwrap();
        std::fs::write(path, serde_json::to_string_pretty(window).unwrap()).unwrap();

        // length
        let mut path: PathBuf = out_path_dir.clone();
        path.push("length");
        if !path.exists() { std::fs::create_dir(&path).unwrap(); }
        path.push(format!("{}-{}.json", &window_start, &window_end));
        let window = programs_sorted_length.get(window_start..window_end+1).unwrap();
        std::fs::write(path, serde_json::to_string_pretty(window).unwrap()).unwrap();

        window_start = window_end + 1;

    }

    println!("Wrote to {:?}", out_path_dir.as_os_str());
}