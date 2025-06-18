use core::panic;
use std::{collections::{HashMap, HashSet}};

use lambdas::{ExprSet, Idx, Node, Symbol};

use crate::{CompressionStepConfig};


type State = String;

#[derive(Debug, Clone)]
pub struct TDFA {
    root: State,
    dfa: HashMap<State, HashMap<Symbol, Vec<State>>>,
    valid_metavars: HashSet<State>,
}

impl TDFA {
    pub fn new(root: String, dfa: String, valid_metavars: Vec<State>) -> Self {
        let dfa: HashMap<State, HashMap<Symbol, Vec<State>>> = serde_json::from_str(&dfa).unwrap();
        let valid_metavars: HashSet<State> = valid_metavars.into_iter().collect();
        TDFA { root, dfa, valid_metavars }
    }

    pub fn annotate(
        &self,
        set: &ExprSet,
        roots: &[Idx],
    ) -> HashMap<usize, State> {
        let mut out = HashMap::new();
        for node in roots {
            self._annotate(set, *node, self.root.clone(), &mut out);
        }
        out
    }

    fn get_symbol_and_args(&self, set: &ExprSet, mut node: Idx) -> (Symbol, Vec<Idx>) {
        let mut children = vec![];
        loop {
            match &set[node] {
                Node::App(f, x) => {
                    children.push(*x);
                    node = *f;
                }
                Node::Prim(tag) => {
                    return (tag.clone(), children);
                }
                _ => {
                    panic!("Unexpected node type: {:?}", set[node]);
                }
            }
        }
    }

    fn _annotate(
        &self,
        set: &ExprSet,
        node: Idx,
        state: State,
        out: &mut HashMap<usize, State>,
    ) {
        out.insert(node, state.clone());
        match set[node] {
            Node::IVar(_) | Node::Lam(_, _) | Node::Var(_, _) => panic!("Not compatible"),
            Node::Prim(_) => return,
            Node::App(_, _) => {}
        }
        let (symbol, args) = self.get_symbol_and_args(set, node);
        let mut transitions = self.dfa.get(&state).and_then(|transitions| transitions.get(&symbol))
            .unwrap_or_else(|| {
                panic!("No transition for state: {:?} and symbol: {:?}", state, symbol);
            }).clone();
        transitions.reverse();
        assert!(args.len() % transitions.len() == 0, "Mismatch in number of transitions and arguments");
        for (i, arg) in args.iter().enumerate() {
            let next_state = transitions[i % transitions.len()].clone();
            self._annotate(set, *arg, next_state, out);
        }
    }
}

pub fn compute_invalid_metavar_location_of_node(cfg: &CompressionStepConfig, set: &ExprSet, roots: &[Idx]) -> Vec<bool> {
    let tdfa_string = std::fs::read_to_string(&cfg.tdfa_json_path).expect("Failed to read TDFA JSON file");
    let tdfa_root = cfg.tdfa_root.clone();
    let valid_metavars = serde_json::from_str::<Vec<State>>(&cfg.valid_metavars).expect("Failed to parse valid metavars JSON");
    let tdfa: TDFA = TDFA::new(
        tdfa_root,
        tdfa_string,
        valid_metavars,
    );
    let annotated = tdfa.annotate(set, roots);
    let mut invalid_metavars = vec![true; set.len()];
    for (idx, state) in annotated.iter() {
        if tdfa.valid_metavars.contains(state) {
            invalid_metavars[*idx] = false;
        }
    }
    invalid_metavars
}