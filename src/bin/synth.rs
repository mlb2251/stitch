use stitch::*;
use stitch::domains::prim_lists::*;
use std::iter::once;
use clap::Parser;
use serde::Serialize;
use std::path::PathBuf;



/// Synthesizer
#[derive(Parser, Debug, Serialize)]
#[clap(name = "Synthesizer")]
pub struct Args {
    /// json output file
    #[clap(short, long, parse(from_os_str), default_value = "out/out.json")]
    pub out: PathBuf,

    // How big of a step to increase cost by for each round of bottom up
    #[clap(short, long, default_value = "1")]
    pub cost_step: usize,

    // Max cost to enumerate to
    #[clap(short, long, default_value = "10")]
    pub max_cost: usize,

    // shuffle order of set of inventions 
    // #[clap(long)]
    // pub shuffle: bool,
}

fn main() {

    let mut args = Args::parse();

    let initial_strs: Vec<String> = (0..10).map(|i| i.to_string())
        .chain(once(String::from("[]"))).collect();

    let initial: Vec<FoundExpr<ListVal>> = initial_strs.iter().map(|s| {
        let expr = Expr::prim(s.into());
        FoundExpr::new(ListVal::val_of_prim(s.into()).unwrap(), expr, 1)
    }).collect();    

    let fns: Vec<(DSLEntry<ListVal>,usize)> = ListVal::get_dsl().entries.values()
        .map(|entry| (entry.clone(),1)).collect();

    bottom_up(&initial, &fns, args.max_cost, args.cost_step)

}
