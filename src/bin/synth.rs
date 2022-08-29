use stitch::*;
use stitch::domains::prim_lists::*;
use stitch::domains::simple::*;
use std::iter::once;
use clap::{Parser,ArgEnum};
use serde::Serialize;
use std::path::PathBuf;



/// Synthesizer
#[derive(Parser, Debug, Serialize)]
#[clap(name = "Synthesizer")]
pub struct Args {
    /// json output file
    #[clap(short, long, parse(from_os_str), default_value = "out/out.json")]
    pub out: PathBuf,

    /// Domain to enumerate from
    #[clap(short, long, arg_enum, default_value = "list")]
    pub domain: DomainChoice,

    #[clap(flatten)]
    pub cfg: BottomUpConfig,
}

#[derive(Debug, Clone, ArgEnum, Serialize)]
pub enum DomainChoice {
    List,
    Simple
}

fn main() {

    let args = Args::parse();

    match &args.domain {
        DomainChoice::List => prim_list_bottom_up(&args),
        DomainChoice::Simple => simple_bottom_up(&args)
    }

}

fn simple_bottom_up(args: &Args) {
    let initial_strs: Vec<String> = (0..3).map(|i| i.to_string())
    .chain(once(String::from("[1,2,3]"))).collect();

    let initial: Vec<FoundExpr<SimpleVal>> = initial_strs.iter().map(|s| {
        let expr = Expr::prim(s.into());
        FoundExpr::new(SimpleVal::val_of_prim(s.into()).unwrap(), expr, 1)
    }).collect();

    let fns: Vec<(DSLEntry<SimpleVal>,usize)> = SimpleVal::dsl_entries()
        .map(|entry| (entry.clone(),1)).collect();

    bottom_up(&initial, &fns, &args.cfg)
}

fn prim_list_bottom_up(args: &Args) {
    let initial_strs: Vec<String> = (0..10).map(|i| i.to_string())
        .chain(once(String::from("[]"))).collect();

    let initial: Vec<FoundExpr<ListVal>> = initial_strs.iter().map(|s| {
        let expr = Expr::prim(s.into());
        FoundExpr::new(ListVal::val_of_prim(s.into()).unwrap(), expr, 1)
    }).collect();    

    let fns: Vec<(DSLEntry<ListVal>,usize)> = ListVal::dsl_entries()
        .map(|entry| (entry.clone(),1)).collect();

    bottom_up(&initial, &fns, &args.cfg)
}

// fn main () {

// }