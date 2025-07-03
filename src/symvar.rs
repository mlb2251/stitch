use clap::Parser;

use lambdas::{Analysis, AnalyzedExpr, Expr, ExprSet, Idx, Symbol};
use serde::Serialize;

#[derive(Parser, Debug, Serialize, Clone)]
#[clap(name = "Stitch")]
pub struct SymvarConfig {
    /// If set, we will use symvars
    #[clap(long)]
    symvar_prefix: Option<char>,
}


#[derive(Debug, Clone)]
pub struct SymvarInfo {
    prefix: char,
    counts: AnalyzedExpr<SymvarCountAnalysis>,
}

impl SymvarInfo {
    pub fn new(set: &ExprSet, config: &SymvarConfig) -> Option<Self> {
        let prefix = config.symvar_prefix?;
        let mut counts = AnalyzedExpr::new(SymvarCountAnalysis);
        counts.analyze(set);

        Some(SymvarInfo { counts, prefix })
    }

    pub fn contains_symbols(&self, node: Idx) -> bool {
        self.counts[node] > 0
    }
    pub fn valid_symbol(&self, sym: &Symbol) -> bool {
        sym.starts_with(self.prefix)
    }
}

#[derive(Debug, Clone)]
struct SymvarCountAnalysis;
impl Analysis for SymvarCountAnalysis {
    type Item = i32;
    fn new(e: Expr, analyzed: &AnalyzedExpr<Self>) -> Self::Item {
        let mut count = 0;
        match e.node() {
            lambdas::Node::IVar(_) => {},
            lambdas::Node::Var(_, _) => {},
            lambdas::Node::Prim(sym) => {
                if sym.starts_with('&') {
                    count += 1; // count prim symbols that are not variables
                }
            },
            lambdas::Node::App(f, x) => {
                count += analyzed[*f];
                count += analyzed[*x];
            },
            lambdas::Node::Lam(b, _) => {
                count += analyzed[*b];
            }
        }
        count
    }
}
