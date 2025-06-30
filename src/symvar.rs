use lambdas::{Analysis, AnalyzedExpr, Expr};

#[derive(Debug, Clone)]
pub struct SymVarAnalysis;
impl Analysis for SymVarAnalysis {
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
