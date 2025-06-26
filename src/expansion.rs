use std::{fmt::{self, Formatter}, sync::Arc};

use lambdas::{Idx, LabelledZId, Node, Symbol, Tag, ZId, ZNode};
use rustc_hash::FxHashMap;

use crate::{compatible_locations, invalid_metavar_location, Arg, Cost, Pattern, SharedData, ZIdExtension};

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

    pub fn add_ivar(&self, original_hole_zid: ZId, first_zid_of_ivar: &mut Vec<ZId>, arg_choices: &mut Vec<LabelledZId>) {
        let ExpandsTo(expands_to) = self;
        if let ExpandsToInner::IVar(i) = expands_to {
            arg_choices.push(LabelledZId::new(original_hole_zid, *i as usize));
            if *i as usize == first_zid_of_ivar.len() {
                first_zid_of_ivar.push(original_hole_zid);
            }
        }
    }
}


impl std::fmt::Display for ExpandsTo {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ExpandsTo(ExpandsToInner::Lam(tag)) => {
                write!(f, "(lam")?;
                if *tag != -1 {
                    write!(f, "_{}", tag)?;
                }
                write!(f, " ??)")
            },
            ExpandsTo(ExpandsToInner::App) => write!(f, "(?? ??)"),
            ExpandsTo(ExpandsToInner::Var(v, tag)) => {
                write!(f, "${v}")?;
                if *tag != -1 {
                    write!(f, "_{}", tag)?;
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
            // in the case where we're searching for an IVar we need to be robust to relabellings
            // since this doesn't have to be canonical. What we can do is we can look over
            // each ivar the the pattern has defined with a first zid in pattern.first_zid_of_ivar, and
            // if our expressions' zids_of_ivar[i] contains this zid then we know these two ivars
            // must correspond to each other in the pattern and the tracked expr and we can just return
            // the pattern version (`j` below).
            let zids = shared.tracking.as_ref().unwrap().zids_of_ivar[i as usize].clone();
            for (j,zid) in pattern.first_zid_of_ivar.iter().enumerate() {
                if zids.contains(zid) {
                    return ExpandsTo(ExpandsToInner::IVar(j as i32));
                }
            }
            // it's a new ivar that hasnt been used already so it must take on the next largest var number
            ExpandsTo(ExpandsToInner::IVar(pattern.first_zid_of_ivar.len() as i32))
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
    for ivar in 0..original_pattern.first_zid_of_ivar.len() {
        let arg_of_loc_ivar = &shared.arg_of_zid_node[original_pattern.first_zid_of_ivar[ivar]];
        let locs: Vec<Idx> = original_pattern.match_locations.iter()
            .filter(|loc:&&Idx|
                arg_of_loc[loc].shifted_id == 
                arg_of_loc_ivar[loc].shifted_id
                && !invalid_metavar_location(shared, arg_of_loc[loc].shifted_id)
            ).cloned().collect();
        if locs.is_empty() { continue; }
        ivars_expansions.push((ExpandsTo(ExpandsToInner::IVar(ivar as i32)), locs));
    }
    // also consider one ivar greater, if this is within the arity limit. This will match at all the same locations as the original.
    if original_pattern.first_zid_of_ivar.len() < shared.cfg.max_arity {
        let ivar = original_pattern.first_zid_of_ivar.len();
        let mut locs = original_pattern.match_locations.clone();
        locs.retain(|loc| !invalid_metavar_location(shared, arg_of_loc[loc].shifted_id));
        ivars_expansions.push((ExpandsTo(ExpandsToInner::IVar(ivar as i32)), locs));
    }
    ivars_expansions
}


/* Perform expansions on variables -- largely for SMC */

fn remove_variable_at(p: &mut Pattern, var_id: usize, expands_to: &mut ExpandsTo) -> Vec<ZId> {
    let ExpandsTo(expands_to) = expands_to;
    let mut zids = Vec::new();
    // remove the variable from the arg choices
    p.arg_choices.retain(|x: &LabelledZId| {
        if x.ivar == var_id {
            // if this is the variable we're removing, add its zid to the list of zids to remove
            // and return false to remove it from the arg choices
            zids.push(x.zid);
            return false;
        }
        true
    });
    p.arg_choices.iter_mut().for_each(|x: &mut LabelledZId| {
        if x.ivar > var_id {
            x.ivar -= 1; // decrement the ivar index for all variables after the one we're removing
        }
    });
    if let ExpandsToInner::IVar(i) = expands_to {
        assert!(*i != var_id as i32, "ExpandsTo::IVar should not be the variable we're removing");
        if *i > var_id as i32 {
            *i -= 1;
        }
    }
    // remove the variable from the first_zid_of_ivar
    p.first_zid_of_ivar.remove(var_id);
    zids
}

pub fn add_variable_at(p: &mut Pattern, at_loc: usize, var_id: i32) {
    p.arg_choices.push(LabelledZId::new(at_loc, var_id as usize));
    if var_id as usize ==  p.first_zid_of_ivar.len() {
        p.first_zid_of_ivar.push(at_loc);
    }
}


pub fn perform_expansion_variable(
    pattern: Pattern,
    shared: &SharedData,
    variable_ivar: usize,
    expands_to: ExpandsTo,
    // locs: Vec<Idx>,
) -> Option<Pattern> {
    let mut pattern = pattern;
    let mut expands_to = expands_to;

    let variable_zids: Vec<usize> = remove_variable_at(&mut pattern, variable_ivar, &mut expands_to);

    let body_utility = pattern.body_utility +  expands_to.local_expansion_utility(shared) * variable_zids.len() as Cost;
    pattern.body_utility = body_utility;

    let num_vars = pattern.first_zid_of_ivar.len() as i32;

    let ExpandsTo(expands_to) = expands_to;

    for variable_zid in variable_zids {
        // add any new holes to the list of holes
        match expands_to {
            ExpandsToInner::Lam(_) => {
                // add new holes
                add_variable_at(&mut pattern, shared.extensions_of_zid[variable_zid].body.unwrap(), num_vars); 
            }
            ExpandsToInner::App => {
                // add new holes
                add_variable_at(&mut pattern, shared.extensions_of_zid[variable_zid].func.unwrap(), num_vars);
                add_variable_at(&mut pattern, shared.extensions_of_zid[variable_zid].arg.unwrap(), num_vars + 1);
            }
            ExpandsToInner::IVar(i) => {
                add_variable_at(&mut pattern, variable_zid, i);
            }
            _ => {}
        }
    }

    Some (pattern)
}


pub fn get_num_variables(pattern: &Pattern) -> usize {
    pattern.first_zid_of_ivar.len()
}

pub fn get_zid_for_ivar(pattern: &Pattern, ivar: usize) -> ZId {
    pattern.first_zid_of_ivar[ivar]
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
