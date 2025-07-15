use std::{fmt::{self, Formatter}, sync::Arc};

use itertools::Itertools;
use lambdas::{Idx, Node, Symbol, Tag, ZId, ZNode};
use rustc_hash::FxHashMap;

use crate::{compatible_locations, invalid_metavar_location, Arg, Cost, Pattern, PatternArgs, SharedData, ZIdExtension};

/// Tells us what a hole will expand into at this node.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
enum ExpandsToInner {
    // Literals
    Lam(Tag),
    App,
    Var(i32, Tag),
    Prim(Symbol),
    // IVar is an abstraction argument, which is a hole that can be filled with a value.
    // Corresponds to a "metavariable" in the Julia stitch implementation.
    IVar(i32),
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
            ExpandsToInner::IVar(_) => false,
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
    pub fn free_variable(&self, num_variables: usize) -> bool {
        let ExpandsTo(s) = self;
        match s {
            ExpandsToInner::Var(i, _) => *i >= num_variables as i32,
            _ => false,
        }
    }

    #[inline]
    #[allow(dead_code)]
    pub fn is_ivar(&self) -> bool {
        matches!(self, ExpandsTo(ExpandsToInner::IVar(_)))
    }

    #[inline]
    pub fn local_expansion_utility(&self, shared: &SharedData) -> Cost {
        let ExpandsTo(s) = self;
        let res = match s {
            ExpandsToInner::Lam(_) => shared.cost_fn.cost_lam,
            ExpandsToInner::App => shared.cost_fn.cost_app,
            ExpandsToInner::Var(_, _) => shared.cost_fn.cost_var,
            ExpandsToInner::Prim(p) => shared.cost_fn.compute_cost_prim(p),
            ExpandsToInner::IVar(_) => 0,
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

    pub fn add_ivar(&self, original_hole_zid: ZId, pattern_args: &mut PatternArgs) {
        let ExpandsTo(expands_to) = self;
        if let ExpandsToInner::IVar(i) = expands_to {
            pattern_args.add_ivar(*i as usize, original_hole_zid);
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
            ExpandsTo(ExpandsToInner::IVar(v)) => write!(f, "#{v}"),
        }
    }
}


/// Used in debugging - tells you what you'd expect the next hole expansion to be
pub fn tracked_expands_to(pattern: &Pattern, hole_zid: ZId, shared: &SharedData) -> ExpandsTo {
    // apply the hole zipper to the original expr being tracked to get the subtree
    // this will expand into, then get the ExpandsTo of that
    let idx = shared.tracking.as_ref().unwrap().expr.immut().zip(&shared.zip_of_zid[hole_zid]).idx;
    match expands_to_of_node(&shared.tracking.as_ref().unwrap().expr.set[idx]) {
        ExpandsTo(ExpandsToInner::IVar(i)) => {
            ExpandsTo(ExpandsToInner::IVar(pattern.pattern_args.find_variable(shared, i as usize) as i32))
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
            Node::IVar(i) => ExpandsToInner::IVar(*i),
        }
    )
}

#[inline]
pub fn get_syntactic_expansions(arg_of_loc: &FxHashMap<usize, Arg>, match_locations: Vec<usize>) -> Vec<(ExpandsTo, Vec<Idx>)> {
    match_locations.into_iter()
        .group_by(|loc| &arg_of_loc[loc].expands_to).into_iter()
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

    // consider all ivars used previously
    for ivar in 0..original_pattern.pattern_args.num_ivars() {
        let locs = original_pattern.pattern_args.reusable_args_location(shared, ivar, arg_of_loc, &original_pattern.match_locations);
        if locs.is_empty() { continue; }
        ivars_expansions.push((ExpandsTo(ExpandsToInner::IVar(ivar as i32)), locs));
    }
    // also consider one ivar greater, if this is within the arity limit. This will match at all the same locations as the original.
    if original_pattern.pattern_args.num_ivars() < shared.cfg.max_arity {
        let ivar = original_pattern.pattern_args.num_ivars();
        let mut locs = original_pattern.match_locations.clone();
        locs.retain(|loc| !invalid_metavar_location(shared, arg_of_loc[loc].shifted_id));
        ivars_expansions.push((ExpandsTo(ExpandsToInner::IVar(ivar as i32)), locs));
    }
    ivars_expansions
}


/* Perform expansions on variables -- largely for SMC */

pub fn perform_expansion_variable(
    pattern: Pattern,
    shared: &SharedData,
    variable_ivar: usize,
    expands_to: ExpandsTo,
    // locs: Vec<Idx>,
) -> Option<Pattern> {
    let mut pattern = pattern;
    let mut expands_to = expands_to;

    let variable_zids: Vec<usize> = pattern.pattern_args.remove_variable_at(variable_ivar, match &mut expands_to {
        ExpandsTo(ExpandsToInner::IVar(i)) => {
            Some(i)
        }
        _ => None,

    });

    let body_utility = pattern.body_utility +  expands_to.local_expansion_utility(shared) * variable_zids.len() as Cost;
    pattern.body_utility = body_utility;

    let num_vars = pattern.pattern_args.arity() as i32;

    let ExpandsTo(expands_to) = expands_to;

    for variable_zid in variable_zids {
        // add any new holes to the list of holes
        match expands_to {
            ExpandsToInner::Lam(_) => {
                // add new holes
                pattern.pattern_args.add_variable_at(shared.extensions_of_zid[variable_zid].body.unwrap(), num_vars); 
            }
            ExpandsToInner::App => {
                // add new holes
                pattern.pattern_args.add_variable_at(shared.extensions_of_zid[variable_zid].func.unwrap(), num_vars);
                pattern.pattern_args.add_variable_at(shared.extensions_of_zid[variable_zid].arg.unwrap(), num_vars + 1);
            }
            ExpandsToInner::IVar(i) => {
                pattern.pattern_args.add_variable_at(variable_zid, i);
            }
            _ => {}
        }
    }

    Some (pattern)
}


pub fn get_num_variables(pattern: &Pattern) -> usize {
    pattern.pattern_args.arity()
}

pub fn get_zid_for_ivar(pattern: &Pattern, ivar: usize) -> ZId {
    pattern.pattern_args.first_zid_of_ivar[ivar]
}


fn sample_new_ivar(
    original_pattern: &Pattern,
    shared: &SharedData,
    variable_ivar: usize,
    match_loc: &usize,
    rng: &mut impl rand::Rng,
) -> Option<usize> {
    let num_vars = get_num_variables(original_pattern);
    if num_vars <= 1 {
        return None; // no other variable to expand to
    }
    let mut new_ivar = rng.gen_range(0..num_vars - 1);
    if new_ivar >= variable_ivar {
        new_ivar += 1; // skip the variable we are expanding
    }
    let zid_original = get_zid_for_ivar(original_pattern, variable_ivar);
    let zid_new = get_zid_for_ivar(original_pattern, new_ivar);
    if shared.arg_of_zid_node[zid_original][match_loc].shifted_id == shared.arg_of_zid_node[zid_new][match_loc].shifted_id {
        return Some(new_ivar);
    }
    None
}

pub fn sample_variable_reuse_expansion(
    pattern: &Pattern,
    shared: &SharedData,
    variable_ivar: usize,
    match_location: usize,
    rng: &mut impl rand::Rng,
) -> Option<(Pattern, ExpandsTo)> {
    if let Some(new_ivar) = sample_new_ivar(pattern, shared, variable_ivar, &match_location, rng) {
        let zid_original = get_zid_for_ivar(pattern, variable_ivar);
        let zid_new = get_zid_for_ivar(pattern, new_ivar);
        let locs = compatible_locations(
            shared,
            pattern,
            &shared.arg_of_zid_node[zid_original],
            &shared.arg_of_zid_node[zid_new],
        );
        if !locs.is_empty() {
            let mut pattern = pattern.clone();
            pattern.match_locations = locs;
            let expands_to = ExpandsTo(ExpandsToInner::IVar(new_ivar as i32));
            return Some((pattern, expands_to));
        }
    }
    None
}
