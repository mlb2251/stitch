use std::{fmt::{self, Formatter}, sync::Arc};

use itertools::Itertools;
use lambdas::{Idx, Node, Symbol, Tag, ZId, ZNode};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{invalid_metavar_location, Arg, Cost, LocationsForReusableArgs, Pattern, PatternArgs, SharedData, SymvarInfo, VariableType, ZIdExtension};

/// Tells us what a hole will expand into at this node.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
enum ExpandsToInner {
    // Literals
    Lam(Tag),
    App,
    Var(i32, Tag),
    Prim(Symbol),
    // IVar is an abstraction argument, which is a hole that can be filled with a value.
    IVar(i32, VariableType),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ExpandsTo(ExpandsToInner);

impl ExpandsTo {
    #[inline]
    /// true if expanding a node of this ExpandsTo will yield new holes
    #[allow(dead_code)]
    pub fn has_holes(&self) -> bool {
        let ExpandsTo(s) = self;
        match s {
            ExpandsToInner::Lam(_) => true,
            ExpandsToInner::App => true,
            ExpandsToInner::Var(_, _) => false,
            ExpandsToInner::Prim(_) => false,
            ExpandsToInner::IVar(_, _) => false,
        }
    }
    #[inline]
    pub fn is_lam(&self) -> bool {
        matches!(self, ExpandsTo(ExpandsToInner::Lam(_)))
    }

    #[inline]
    pub fn is_app(&self) -> bool {
        matches!(self, ExpandsTo(ExpandsToInner::App))
    }

    #[inline]
    pub fn is_prim_symbol(&self, sym_var_info: &SymvarInfo) -> bool {
        let ExpandsTo(ExpandsToInner::Prim(sym)) = self else {
            return false;
        };
        sym_var_info.valid_symbol(sym)
    }

    #[inline]
    pub fn free_variable(&self, num_variables: usize) -> bool {
        let ExpandsTo(s) = self;
        match s {
            ExpandsToInner::Var(i, _) => *i >= num_variables as i32,
            _ => false,
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn is_var(&self) -> bool {
        matches!(self, ExpandsTo(ExpandsToInner::IVar(_, _)))
    }

    #[inline]
    pub fn local_expansion_utility(&self, shared: &SharedData) -> Cost {
        let ExpandsTo(s) = self;
        let res = match &s {
            ExpandsToInner::Lam(_) => shared.cost_fn.cost_lam,
            ExpandsToInner::App => shared.cost_fn.cost_app,
            ExpandsToInner::Var(_, _) => shared.cost_fn.cost_var,
            ExpandsToInner::Prim(p) => shared.cost_fn.compute_cost_prim(p),
            ExpandsToInner::IVar(_, _) => 0,
        };
        res as Cost
    }

    #[inline]
    pub fn add_holes(&self, original_hole_zid_extension: &ZIdExtension, holes: &mut Vec<ZId>) {
        let ExpandsTo(expands_to) = self;
        match expands_to {
            ExpandsToInner::Lam(_) => {
                // add new holes
                holes.push(original_hole_zid_extension.body.unwrap());
            }
            ExpandsToInner::App => {
                // add new holes
                    holes.push(original_hole_zid_extension.func.unwrap());
                    holes.push(original_hole_zid_extension.arg.unwrap());
            }
            _ => {}
        }
    }

    pub fn add_variables(&self, original_hole_zid: ZId, pattern_args: &mut PatternArgs) {
        let ExpandsTo(expands_to) = self;
        if let ExpandsToInner::IVar(i, vt) = expands_to {
            pattern_args.add_var(*i as usize, original_hole_zid, *vt);
        }
    }
}


impl std::fmt::Display for ExpandsTo {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ExpandsTo(ExpandsToInner::Lam(tag)) => {
                write!(f, "(lam")?;
                if *tag != -1 {
                    write!(f, "_{tag}")?;
                }
                write!(f, " ??)")
            },
            ExpandsTo(ExpandsToInner::App) => write!(f, "(?? ??)"),
            ExpandsTo(ExpandsToInner::Var(v, tag)) => {
                write!(f, "${v}")?;
                if *tag != -1 {
                    write!(f, "_{tag}")?;
                }
                Ok(())
            },
            ExpandsTo(ExpandsToInner::Prim(p)) => write!(f, "{p}"),
            ExpandsTo(ExpandsToInner::IVar(v, _)) => write!(f, "#{v}"),
        }
    }
}


/// Used in debugging - tells you what you'd expect the next hole expansion to be
pub fn tracked_expands_to(pattern: &Pattern, hole_zid: ZId, shared: &SharedData) -> ExpandsTo {
    // apply the hole zipper to the original expr being tracked to get the subtree
    // this will expand into, then get the ExpandsTo of that
    let idx = shared.tracking.as_ref().unwrap().expr.immut().zip(&shared.zip_of_zid[hole_zid]).idx;
    match expands_to_of_node(&shared.tracking.as_ref().unwrap().expr.set[idx]) {
        ExpandsTo(ExpandsToInner::IVar(i, VariableType::Metavar)) => {
            ExpandsTo(ExpandsToInner::IVar(pattern.pattern_args.find_variable(shared, i as usize) as i32, VariableType::Metavar))
        }
        e => e
    }
}


pub fn expands_to_of_node(node: &Node) -> ExpandsTo {
    ExpandsTo(
        match node {
            Node::Var(i, tag) => ExpandsToInner::Var(*i, *tag),
            Node::Prim(p) => ExpandsToInner::Prim(p.clone()),
            Node::Lam(_, tag) => ExpandsToInner::Lam(*tag),
            Node::App(_,_) => ExpandsToInner::App,
            Node::IVar(i) => ExpandsToInner::IVar(*i, VariableType::Metavar),
        }
    )
}

#[inline]
pub fn get_syntactic_expansions(arg_of_loc: &FxHashMap<usize, Arg>, match_locations: Vec<usize>, sym_var_info: &Option<SymvarInfo>) -> Vec<(ExpandsTo, Vec<Idx>)> {
    match_locations.into_iter()
        .group_by(|loc| &arg_of_loc[loc].expands_to).into_iter()
        .filter(|(expands_to, _)| sym_var_info.as_ref().is_none_or(|s| !expands_to.is_prim_symbol(s)))
        .map(|(expands_to, locs)| (expands_to.clone(), locs.collect::<Vec<Idx>>()))
        .collect::<Vec<_>>()
}


//#[inline(never)]
/// Return options for what abstraction arguments (aka ivars, #i) can expand into. When expanding to an ivar that
/// already exists in the expression the match locations get subset to enforce the equality constraint - for example
/// in (* #0 #0) both #0s must be the same within each match location. For a fresh ivar that doesn't yet exist in a pattern,
/// we only allow if it is within our max arity limit.
pub fn get_ivars_expansions(original_pattern: &Pattern, arg_of_loc: &FxHashMap<Idx,Arg>, hole_zid: ZId, shared: &Arc<SharedData>) -> Vec<(ExpandsTo, Vec<Idx>)> {
    let mut ivars_expansions = vec![];

    if shared.cfg.no_curried_metavars {
        // dont allow any expansions that result in a metavar to the left of an app
        if let Some(ZNode::Func) = shared.zip_of_zid[hole_zid].last(){
            return ivars_expansions;
        }
    }
    let mut all_reusable_locs = FxHashSet::default();
    // consider all ivars used previously
    let mut locs_for_reusable = LocationsForReusableArgs::new(&original_pattern.match_locations);
    for var in 0..original_pattern.pattern_args.arity() {
        let locs = original_pattern.pattern_args.reusable_args_location(shared, var, arg_of_loc, &mut locs_for_reusable);
        if locs.is_empty() { continue; }
        all_reusable_locs.extend(locs.iter().cloned());
        ivars_expansions.push((ExpandsTo(ExpandsToInner::IVar(var as i32, VariableType::Metavar)), locs));
    }
    // also consider one ivar greater, if this is within the arity limit. This will match at all the same locations as the original.
    if original_pattern.pattern_args.num_ivars() < shared.cfg.max_arity {
        let var = original_pattern.pattern_args.arity();
        let mut locs = original_pattern.match_locations.clone();
        locs.retain(|loc| !invalid_metavar_location(shared, arg_of_loc[loc].shifted_id));
        ivars_expansions.push((ExpandsTo(ExpandsToInner::IVar(var as i32, VariableType::Metavar)), locs));
    }

    if let Some(sym_var_info) = shared.sym_var_info.as_ref() {
        let svar_locations = svar_locations(original_pattern, arg_of_loc, all_reusable_locs, sym_var_info);
        if !svar_locations.is_empty() {
            ivars_expansions.push((ExpandsTo(ExpandsToInner::IVar(original_pattern.pattern_args.arity() as i32, VariableType::Symvar)), svar_locations));
        }
    }


    ivars_expansions
}


pub fn svar_locations(original_pattern: &Pattern, arg_of_loc: &FxHashMap<Idx,Arg>, reusable_locs: FxHashSet<Idx>, sym_var_info: &SymvarInfo) -> Vec<Idx> {

    let mut locations = vec![];

    original_pattern.match_locations.iter().for_each(|loc| {
        if !arg_of_loc[loc].expands_to.is_prim_symbol(sym_var_info) {
            return;
        }
        if reusable_locs.contains(loc) {
            return;
        }

        locations.push(*loc);
    });
    locations
}
