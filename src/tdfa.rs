use core::panic;
use std::{collections::{HashMap, HashSet}};

use lambdas::{ExprSet, Idx, Node, Symbol};

use crate::{CompressionStepConfig, CompressionStepResult, Pattern, SharedData};


pub type State = String;

#[derive(Debug, Clone)]
pub struct TDFA {
    root: State,
    dfa: HashMap<State, HashMap<Symbol, Vec<State>>>,
    valid_metavars: HashSet<State>,
    tdfa_non_eta_long_states: HashSet<State>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TDFAInventionAnnotation {
    pub root_state: State,
    pub metavariable_states: Vec<State>,
}

impl TDFAInventionAnnotation {
    pub fn from_pattern(
        pattern: &Pattern,
        shared: &SharedData,
    ) -> Self {
        // println!("Match locations: {:?}", pattern.match_locations);
        // println!("Root symbols: {:?}", pattern.match_locations.iter().map(|&loc| shared.tdfa_symbol_of_node[loc].clone()).collect::<Vec<_>>());
        let annotation = TDFAInventionAnnotation::from_match_location(pattern, shared, pattern.match_locations[0]);
        // println!("TDFAInventionAnnotation: {:?}", annotation);
        for i in 1..pattern.match_locations.len() {
            assert!(annotation == TDFAInventionAnnotation::from_match_location(pattern, shared, pattern.match_locations[i]),
                "Inconsistent TDFAInventionAnnotation for match locations: {:?} and {:?}", 
                pattern.match_locations[0], pattern.match_locations[i]);
        }
        annotation
    }

    fn from_match_location(
        pattern: &Pattern,
        shared: &SharedData,
        match_location: Idx,
    ) -> Self {
        let root_sym: String = shared.tdfa_symbol_of_node[match_location].clone().unwrap();
        let mut ivar_states = vec![];
        pattern.first_zid_of_ivar.iter().for_each(|ivar_zid| {
            let node = shared.arg_of_zid_node[*ivar_zid].get(&match_location).unwrap().unshifted_id;
            let ivar_sym = shared.tdfa_symbol_of_node[node].clone().unwrap();
            ivar_states.push(ivar_sym);
        });
        Self {
            root_state: root_sym,
            metavariable_states: ivar_states,
        }
    }
}

impl TDFA {
    pub fn new(root: String, dfa: String, valid_metavars: Vec<State>, tdfa_non_eta_long_states: Vec<State>, prev_invs: Vec<(String, Option<TDFAInventionAnnotation>)>) -> Self {
        let mut dfa: HashMap<State, HashMap<Symbol, Vec<State>>> = serde_json::from_str(&dfa).unwrap();
        for (name, tdfa_annotation) in prev_invs {
            if let Some(annotation) = tdfa_annotation {
                println!("{:?} -> {:?}", name, annotation.metavariable_states);
                if !dfa.contains_key(&annotation.root_state) {
                    dfa.insert(annotation.root_state.clone(), HashMap::new());
                }
                let transitions = dfa.get_mut(&annotation.root_state).unwrap();
                transitions.insert(
                    Symbol::from(name),
                    annotation.metavariable_states,
                );
            } else {
                panic!("Previous invention annotation is None for DC invariant: {}", name);
            }
        }
        let valid_metavars: HashSet<State> = valid_metavars.into_iter().collect();
        let tdfa_non_eta_long_states: HashSet<State> = tdfa_non_eta_long_states.into_iter().collect();
        TDFA { root, dfa, tdfa_non_eta_long_states, valid_metavars }
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

    fn get_symbol_and_args(&self, set: &ExprSet, mut node: Idx) -> (Symbol, Vec<Idx>, Vec<Idx>) {
        let mut nodes = vec![];
        let mut children = vec![];
        loop {
            match &set[node] {
                Node::App(f, x) => {
                    nodes.push(*f);
                    children.push(*x);
                    node = *f;
                }
                Node::Prim(tag) => {
                    return (tag.clone(), nodes.clone(), children);
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
        // println!("Annotating node {:?}: {} with state: {:?}", node, set.get(node), state);
        out.insert(node, state.clone());
        match set[node] {
            Node::IVar(_) | Node::Lam(_, _) | Node::Var(_, _) => panic!("Not compatible"),
            Node::Prim(_) => return,
            Node::App(_, _) => {}
        }
        let (symbol, nodes, args) = self.get_symbol_and_args(set, node);
        let mut transitions = self.dfa.get(&state).and_then(|transitions| transitions.get(&symbol))
            .unwrap_or_else(|| {
                panic!("No transition for state: {:?} and symbol: {:?}", state, symbol);
            }).clone();
        transitions.reverse();
        let non_eta_long = self.tdfa_non_eta_long_states.contains(&state);
        // assert!(transitions.is_empty() || args.len() % transitions.len() == 0, "Mismatch in number of transitions and arguments");
        for (i, arg) in args.iter().enumerate() {
            let next_state = transitions[i % transitions.len()].clone();
            self._annotate(set, *arg, next_state, out);
        }
        if non_eta_long {
            assert!(transitions.len() == 1, "Non-eta long state {:?} has multiple transitions: {:?}", state, transitions);
            for child in nodes {
                out.insert(child, state.clone());
            }
        }
    }
}

pub fn compute_invalid_metavar_location_of_node(cfg: &CompressionStepConfig, set: &ExprSet, roots: &[Idx], prev_results: &[CompressionStepResult]) -> (Vec<Option<State>>, Vec<bool>) {
    let tdfa_string = std::fs::read_to_string(&cfg.tdfa_json_path).expect("Failed to read TDFA JSON file");
    let tdfa_root = cfg.tdfa_root.clone();
    let valid_metavars = serde_json::from_str::<Vec<State>>(&cfg.valid_metavars).expect("Failed to parse valid metavars JSON");
    let tdfa_non_eta_long_states = serde_json::from_str::<Vec<State>>(&cfg.tdfa_non_eta_long_states).expect("Failed to parse non-eta long states JSON");
    let tdfa: TDFA = TDFA::new(
        tdfa_root,
        tdfa_string,
        valid_metavars,
        tdfa_non_eta_long_states,
        prev_results.iter().map(|r| (r.inv.name.clone(), r.tdfa_annotation.clone())).collect::<Vec<_>>(),
    );
    let annotated = tdfa.annotate(set, roots);
    let mut symbols = vec![None; set.len()];
    let mut invalid_metavars = vec![true; set.len()];
    for (idx, state) in annotated.iter() {
        if tdfa.valid_metavars.contains(state) {
            invalid_metavars[*idx] = false;
        }
        symbols[*idx] = Some(state.clone());
    }
    (symbols, invalid_metavars)
}