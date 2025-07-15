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
    count_and_is_symvars: AnalyzedExpr<SymvarCountAnalysis>,
}

impl SymvarInfo {
    pub fn new(set: &ExprSet, config: &SymvarConfig) -> Option<Self> {
        let prefix = config.symvar_prefix?;
        let mut count_and_is_symvars = AnalyzedExpr::new(SymvarCountAnalysis);
        count_and_is_symvars.analyze(set);

        Some(SymvarInfo { count_and_is_symvars, prefix })
    }

    pub fn contains_symbols(&self, node: Idx) -> bool {
        self.count_and_is_symvars[node].0 > 0
    }
    pub fn valid_symbol(&self, sym: &Symbol) -> bool {
        sym.starts_with(self.prefix)
    }
    pub fn is_symvar_spot(&self, node: Idx) -> bool {
        self.count_and_is_symvars[node].1
    }
}

#[derive(Debug, Clone)]
struct SymvarCountAnalysis;
impl Analysis for SymvarCountAnalysis {
    type Item = (i32, bool);
    fn new(e: Expr, analyzed: &AnalyzedExpr<Self>) -> Self::Item {
        let mut count_and_is_symvar = (0, false);
        match e.node() {
            lambdas::Node::IVar(_) => {},
            lambdas::Node::Var(_, _) => {},
            lambdas::Node::Prim(sym) => {
                if sym.starts_with('&') {
                    count_and_is_symvar.0 += 1;
                    count_and_is_symvar.1 = true;
                }
            },
            lambdas::Node::App(f, x) => {
                count_and_is_symvar.0 += analyzed[*f].0;
                count_and_is_symvar.0 += analyzed[*x].0;
            },
            lambdas::Node::Lam(b, _) => {
                count_and_is_symvar.0 += analyzed[*b].0;
            }
        }
        count_and_is_symvar
    }
}
