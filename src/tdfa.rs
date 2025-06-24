use core::panic;
use std::collections::{HashMap, HashSet};

use clap::Parser;
use lambdas::{ExprSet, Idx, Node, Symbol};
use serde::Serialize;

use crate::{CompressionStepConfig, CompressionStepResult, Pattern, SharedData};


type State = String;

#[derive(Parser, Debug, Serialize, Clone)]
#[clap(name = "Stitch")]
pub struct TDFAConfig {
    /// If set, we will apply the given TDFA to the programs and use this to annotate the programs
    #[clap(long, default_value = "")]
    tdfa_json_path: String,

    /// The root of the TDFA to use for annotation
    #[clap(long, default_value = "")]
    tdfa_root: String,

    /// Metavariable locations that are valid for the TDFA
    #[clap(long, default_value = "")]
    valid_metavars: String,

    /// Root locations that are valid for the TDFA
    #[clap(long, default_value = "")]
    valid_roots: String,

    /// States of the TDFA that not in eta-long-form (e.g., (/seq A B C) makes (/seq A) a valid metavariable)
    #[clap(long, default_value = "")]
    tdfa_non_eta_long_states: String,
}

impl TDFAConfig {
    fn present(&self) -> bool {
        !self.tdfa_json_path.is_empty()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TDFAGlobalAnnotations {
    symbols: Vec<Option<State>>,
    invalid_metavars: Vec<bool>,
    invalid_roots: Vec<bool>,
}

impl TDFAGlobalAnnotations {
    pub fn new(cfg: &CompressionStepConfig, set: &ExprSet, roots: &[Idx], prev_results: &[CompressionStepResult]) -> Option<TDFAGlobalAnnotations> {
        if !cfg.tdfa.present() {
            return None;
        }
        let tdfa_cfg = cfg.tdfa.clone();
        let tdfa_string = std::fs::read_to_string(tdfa_cfg.tdfa_json_path).expect("Failed to read TDFA JSON file");
        let tdfa_root = tdfa_cfg.tdfa_root.clone();
        println!("{}", tdfa_cfg.valid_metavars);
        let valid_metavars = serde_json::from_str::<Vec<State>>(&tdfa_cfg.valid_metavars).expect("Failed to parse valid metavars JSON");
        let valid_roots = serde_json::from_str::<Vec<State>>(&tdfa_cfg.valid_roots).expect("Failed to parse valid roots JSON");
        let tdfa_non_eta_long_states: HashMap<State, State> = serde_json::from_str(&tdfa_cfg.tdfa_non_eta_long_states).expect("Failed to parse non-eta long states JSON");
        let tdfa: TDFA = TDFA::new(
            tdfa_root,
            tdfa_string,
            valid_metavars,
            valid_roots,
            tdfa_non_eta_long_states,
            prev_results.iter().map(|r| (r.inv.name.clone(), r.tdfa_annotation.clone())).collect::<Vec<_>>(),
        );
        let annotated = tdfa.annotate(set, roots);
        let mut symbols = vec![None; set.len()];
        let mut invalid_metavars = vec![true; set.len()];
        let mut invalid_roots = vec![true; set.len()];
        for (idx, state) in annotated.iter() {
            if tdfa.valid_metavars.contains(state) {
                invalid_metavars[*idx] = false;
            }
            if tdfa.valid_roots.contains(state) {
                invalid_roots[*idx] = false;
            }
            symbols[*idx] = Some(state.clone());
        }
        Some(
            TDFAGlobalAnnotations {
                symbols,
                invalid_metavars,
                invalid_roots,
            }
        )
    }
}

pub fn tdfa_invalid_metavar(
    global_annotations: &Option<TDFAGlobalAnnotations>,
    idx: Idx,
) -> bool {
    match global_annotations {
        Some(annotations) => annotations.invalid_metavars[idx],
        None => false,
    }
}

pub fn tdfa_invalid_root(
    global_annotations: &Option<TDFAGlobalAnnotations>,
    idx: Idx,
) -> bool {
    match global_annotations {
        Some(annotations) => annotations.invalid_roots[idx],
        None => false,
    }
}

#[derive(Debug, Clone)]
pub struct TDFA {
    root: State,
    dfa: HashMap<State, HashMap<Symbol, Vec<State>>>,
    valid_metavars: HashSet<State>,
    valid_roots: HashSet<State>,
    tdfa_non_eta_long_states: HashMap<State, State>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TDFAInventionAnnotation {
    root_state: State,
    metavariable_states: Vec<State>,
}

impl TDFAInventionAnnotation {
    pub fn from_pattern(
        pattern: &Pattern,
        shared: &SharedData
    ) -> Option<Self> {
        let Some(global_annotations) = &shared.tdfa_global_annotations else {
            return None;
        }; 
        let annotation = TDFAInventionAnnotation::from_match_location(pattern, shared, pattern.match_locations[0], global_annotations);
        for i in 1..pattern.match_locations.len() {
            assert!(annotation == TDFAInventionAnnotation::from_match_location(pattern, shared, pattern.match_locations[i], global_annotations),
                "Inconsistent TDFAInventionAnnotation for match locations: {:?} and {:?}", 
                pattern.match_locations[0], pattern.match_locations[i]);
        }
        Some(annotation)
    }

    fn from_match_location(
        pattern: &Pattern,
        shared: &SharedData,
        match_location: Idx,
        global_annotations: &TDFAGlobalAnnotations,
    ) -> Self {
        let root_sym: String = global_annotations.symbols[match_location].clone().unwrap();
        let mut ivar_states = vec![];
        pattern.first_zid_of_ivar.iter().for_each(|ivar_zid| {
            let node = shared.arg_of_zid_node[*ivar_zid].get(&match_location).unwrap().unshifted_id;
            let ivar_sym = global_annotations.symbols[node].clone().unwrap();
            ivar_states.push(ivar_sym);
        });
        Self {
            root_state: root_sym,
            metavariable_states: ivar_states,
        }
    }
}

impl TDFA {
    pub fn new(root: String, dfa: String, valid_metavars: Vec<State>, valid_roots: Vec<State>, tdfa_non_eta_long_states: HashMap<State, State>, prev_invs: Vec<(String, Option<TDFAInventionAnnotation>)>) -> Self {
        let mut dfa: HashMap<State, HashMap<Symbol, Vec<State>>> = serde_json::from_str(&dfa).unwrap();
        for (name, tdfa_annotation) in prev_invs {
            if let Some(annotation) = tdfa_annotation {
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
        let valid_roots: HashSet<State> = valid_roots.iter().cloned().collect();
        TDFA { root, dfa, tdfa_non_eta_long_states, valid_metavars, valid_roots }
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
                    nodes.reverse();
                    children.reverse();
                    return (tag.clone(), nodes, children);
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
        let (symbol, nodes, args) = self.get_symbol_and_args(set, node);
        let transitions = self.dfa.get(&state).and_then(|transitions| transitions.get(&symbol))
            .unwrap_or_else(|| {
                panic!("No transition for state: {:?} and symbol: {:?}", state, symbol);
            }).clone();

        let non_eta_long = self.tdfa_non_eta_long_states.contains_key(&state);
        let mut cur_arg = 0;
        if non_eta_long {
            // transitions are not in eta-long form.
            while cur_arg < transitions.len() {
                let next_state = transitions[cur_arg].clone();
                self._annotate(set, args[cur_arg], next_state, out);
                cur_arg += 1;
            }
            // now we annotate the rest of the args with the non-eta-long state
            let inner_state = self.tdfa_non_eta_long_states.get(&state).unwrap().clone();
            while cur_arg < args.len() {
                out.insert(nodes[cur_arg], state.clone());
                self._annotate(set, args[cur_arg], inner_state.clone(), out);
                cur_arg += 1;
            }
        } else {
            assert!(transitions.is_empty() || args.len() % transitions.len() == 0, "Mismatch in number of transitions and arguments");
            for (i, arg) in args.iter().enumerate() {
                let next_state = transitions[i % transitions.len()].clone();
                self._annotate(set, *arg, next_state, out);
            }
        }
    }
}
