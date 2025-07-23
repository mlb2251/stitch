use core::panic;
use std::collections::{HashMap, HashSet};

use clap::Parser;
use lambdas::{ExprOwned, ExprSet, Idx, Node, Symbol};
use serde::Serialize;

use crate::{CompressionStepConfig, CompressionStepResult, Pattern, SharedData, SymvarInfo};


type State = String;

#[derive(Parser, Debug, Serialize, Clone)]
#[clap(name = "Stitch")]
pub struct TDFAConfig {
    /// If set, we will apply the given TDFA to the programs and use this to annotate the programs
    #[clap(long, required = false)]
    tdfa_json_path: Option<String>,

    /// The root of the TDFA to use for annotation
    #[clap(long, required = false)]
    tdfa_root: Option<String>,

    /// Metavariable locations that are valid for the TDFA
    #[clap(long, required = false)]
    valid_metavars: Option<String>,

    /// Root locations that are valid for the TDFA
    #[clap(long, required = false)]
    valid_roots: Option<String>,

    /// States of the TDFA that not in eta-long-form (e.g., (/seq A B C) makes (/seq A) a valid metavariable)
    #[clap(long, required = false)]
    tdfa_non_eta_long_states: Option<String>,

    /// If set, when present in a symbol, we will take the part before the split as the symbol
    #[clap(long)]
    tdfa_split: Option<String>,
}

impl TDFAConfig {
    fn present(&self) -> bool {
        self.tdfa_json_path.is_some()
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TDFAGlobalAnnotations {
    symbols: Vec<Option<State>>,
    invalid_metavars: Vec<bool>,
    invalid_roots: Vec<bool>,
}

impl TDFAGlobalAnnotations {
    pub fn new(
        cfg: &CompressionStepConfig,
        set: &ExprSet,
        roots: &[Idx],
        prev_results: &[CompressionStepResult],
        sym_var_info: &Option<SymvarInfo>,
    ) -> Option<TDFAGlobalAnnotations> {
        if !cfg.tdfa.present() {
            return None;
        }
        let tdfa_cfg = cfg.tdfa.clone();
        let tdfa_string = std::fs::read_to_string(tdfa_cfg.tdfa_json_path.as_ref().expect("TDFA parameter 'tdfa_json_path' was not provided")).expect("Could not read file at path specified by 'tdfa_json_path'");
        let tdfa_root = tdfa_cfg.tdfa_root.clone().expect("TDFA parameter 'tdfa_root' was not provided");
        let valid_metavars = serde_json::from_str::<Vec<State>>(
            tdfa_cfg.valid_metavars.as_ref().expect("TDFA parameter 'valid_metavars' was not provided")
        ).expect("TDFA parameter 'valid_metavars' could not be parsed as a JSON array of states");
        let valid_roots = serde_json::from_str::<Vec<State>>(
            tdfa_cfg.valid_roots.as_ref().expect("TDFA parameter 'valid_roots' was not provided")
        ).expect("TDFA parameter 'valid_roots' could not be parsed as a JSON array of states");
        let tdfa_non_eta_long_states: HashMap<State, State> = serde_json::from_str(
            tdfa_cfg.tdfa_non_eta_long_states.as_ref().expect("TDFA parameter 'tdfa_non_eta_long_states' was not provided")
        ).expect("TDFA parameter 'tdfa_non_eta_long_states' could not be parsed as a JSON object mapping states");
        let tdfa: TDFA = TDFA::new(
            tdfa_root,
            tdfa_string,
            valid_metavars,
            valid_roots,
            tdfa_non_eta_long_states,
            prev_results.iter().map(|r| (r.inv.name.clone(), r.tdfa_annotation.clone())).collect::<Vec<_>>(),
            tdfa_cfg.tdfa_split.clone(),
        );
        let annotated = tdfa.annotate(set, roots, sym_var_info);
        let mut symbols: Vec<Option<State>> = vec![None; set.len()];
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
    split: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
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
        let annotation = TDFAInventionAnnotation::from_match_location(pattern, shared, pattern.match_locations[0], global_annotations).unwrap();
        for i in 1..pattern.match_locations.len() {
            let for_loc = TDFAInventionAnnotation::from_match_location(pattern, shared, pattern.match_locations[i], global_annotations).unwrap();
            if annotation != for_loc {
                panic!("Inconsistent TDFAInventionAnnotation for match locations: {:?} and {:?} ({:?} vs {:?})",
                    pattern.match_locations[0], pattern.match_locations[i],
                    annotation,
                    for_loc,
                );
            }
        }
        Some(annotation)
    }

    fn from_match_location(
        pattern: &Pattern,
        shared: &SharedData,
        match_location: Idx,
        global_annotations: &TDFAGlobalAnnotations,
    ) -> Option<Self> {
        let root_sym = global_annotations.symbols[match_location].clone()?;
        let mut ivar_states: Vec<String> = vec![];
        let all_found = pattern.pattern_args.iterate_one_zid_per_argument().all(|ivar_zid| {
            let Some(node) = shared.arg_of_zid_node[ivar_zid].get(&match_location) else {
                return false;
            };
            let Some(ivar_sym) = global_annotations.symbols[node.unshifted_id].clone() else {
                return false;
            };
            ivar_states.push(ivar_sym);
            true
        });
        if !all_found {
            return None;
        }
        Some (Self { root_state: root_sym, metavariable_states: ivar_states })
    }

    pub fn metavariables_are_consistent(
        pattern: &Pattern,
        shared: &SharedData,
    ) -> bool {
        let Some(global_annotations) = &shared.tdfa_global_annotations else {
            return true; // No TDFA, so no metavariable consistency check needed
        };
        // only check the first match location, since all should be consistent
        // the full consistency check is done in `TDFAInventionAnnotation::from_pattern`
        TDFAInventionAnnotation::from_match_location(pattern, shared, pattern.match_locations[0], global_annotations).is_some()
    }

}

impl TDFA {
    pub fn new(root: String, dfa: String, valid_metavars: Vec<State>, valid_roots: Vec<State>, tdfa_non_eta_long_states: HashMap<State, State>, prev_invs: Vec<(String, Option<TDFAInventionAnnotation>)>, split: Option<String>) -> Self {
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
        TDFA { root, dfa, tdfa_non_eta_long_states, valid_metavars, valid_roots, split }
    }

    pub fn annotate(
        &self,
        set: &ExprSet,
        roots: &[Idx],
        sym_var_info: &Option<SymvarInfo>,
    ) -> HashMap<usize, State> {
        let mut out = HashMap::new();
        for node in roots {
            self._annotate(set, *node, self.root.clone(), &mut out, sym_var_info);
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

    fn relevant_symbol(&self, symbol: &State) -> bool {
        self.valid_metavars.contains(symbol) || self.valid_roots.contains(symbol)
    }

    fn matters_to_annotate_node(&self, symbol_1: &State, symbol_2: &State, is_symvar_spot: bool) -> bool {
        if is_symvar_spot {
            return true;
        }
        if self.relevant_symbol(symbol_1) {
            return true;
        }
        if self.relevant_symbol(symbol_2) {
            return true;
        }
        false
    }

    fn _check_consistent(
        &self,
        set: &ExprSet,
        expr: Idx,
        existing_symbol: &State,
        new_symbol: &State,
        is_symvar_spot: bool,
    ) {
        if *existing_symbol == *new_symbol {
            return;
        }
        if !self.matters_to_annotate_node(existing_symbol, new_symbol, is_symvar_spot) {
            return;
        }
        panic!("Inconsistent symbols: {:?} and {:?} for expr {}", existing_symbol, new_symbol, ExprOwned {idx: expr, set: set.clone()});
    }

    fn _annotate(
        &self,
        set: &ExprSet,
        node: Idx,
        state: State,
        out: &mut HashMap<usize, State>,
        sym_var_info: &Option<SymvarInfo>,
    ) {
        if let Some(symbol) = &out.get(&node) {
            self._check_consistent(set, node, symbol, &state, sym_var_info.as_ref().is_some_and(|info| info.is_symvar_spot(node)));
        } else {
            out.insert(node, state.clone());
        }
        match set[node] {
            Node::IVar(_) | Node::Lam(_, _) | Node::Var(_, _) => panic!("Not compatible"),
            Node::Prim(_) => return,
            Node::App(_, _) => {}
        }
        let (symbol, nodes, args) = self.get_symbol_and_args(set, node);
        let symbol = self.process_split(symbol);
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
                self._annotate(set, args[cur_arg], next_state, out, sym_var_info);
                cur_arg += 1;
            }
            // now we annotate the rest of the args with the non-eta-long state
            let inner_state = self.tdfa_non_eta_long_states.get(&state).unwrap().clone();
            while cur_arg < args.len() {
                out.insert(nodes[cur_arg], state.clone());
                self._annotate(set, args[cur_arg], inner_state.clone(), out, sym_var_info);
                cur_arg += 1;
            }
        } else {
            assert!(transitions.is_empty() || args.len() % transitions.len() == 0, "Mismatch in number of transitions and arguments {}/{}: {} vs {}", state, symbol, transitions.len(), args.len());
            for (i, arg) in args.iter().enumerate() {
                let next_state = transitions[i % transitions.len()].clone();
                self._annotate(set, *arg, next_state, out, sym_var_info);
            }
        }
    }

    fn process_split(&self, symbol: Symbol) -> Symbol {
        let Some(split) = &self.split else {
            return symbol;
        };
        let Some(idx) = symbol.find(split) else {
            return symbol;
        };
        Symbol::from(symbol[..idx].to_string())
    }
}
