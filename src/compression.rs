use crate::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::{self, Formatter, Display};
use std::hash::Hash;
use itertools::Itertools;
use extraction::extract;
use serde_json::json;
use clap::{Parser};
use serde::Serialize;
use std::thread;
use std::sync::Arc;
use parking_lot::Mutex;
use std::ops::DerefMut;
use std::collections::BinaryHeap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Pattern {
    holes: Vec<ZId>, // in order of when theyre added NOT left to right
    arg_choices: Vec<ZId>, // a hole gets moved into here when it becomes an argchoice, again these are in order of when they were added
    match_locations: Vec<Id>, // places where it applies
    pruned_assignment_prefixes: Vec<Vec<i32>>,
    utility_upper_bound: i32,
    body_utility: i32, // the size (in `cost`) of a single use of the pattern body so far
}


#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
enum NodeType {
    Lam,
    App,
    Var(i32),
    Prim(Symbol)
}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Arg {
    id: Id,
    unshifted_id: Id, // in case `id` was shifted to make it an arg not sure if this will end up being useful
    cost: i32,
    node_type: NodeType,
}

/// The heap item used for heap-based worklists
#[derive(Debug,Clone, Eq, PartialEq)]
struct HeapItem {
    key: i32,
    pattern: Pattern,
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.key.partial_cmp(&other.key)
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}
impl HeapItem {
    fn new(pattern: Pattern) -> Self {
        HeapItem {
            key: pattern.body_utility * pattern.match_locations.len() as i32,
            pattern
        }
    }
}

struct MutableMultithreadData {
    donelist: Vec<FinishedPattern>,
    worklist: BinaryHeap<HeapItem>,
    lowest_donelist_utility:i32,
    utility_pruning_cutoff: i32,
}

fn get_worklist_item(
    worklist_buf: &mut Vec<HeapItem>,
    donelist_buf: &mut Vec<FinishedPattern>,
    shared: &Arc<Mutex<MutableMultithreadData>>,
    arg_of_zid_node: &HashMap<(ZId,Id),Arg>, // much like appzipper_of_node and yes shifted
    zids_of_node: &Arc<Vec<Vec<ZId>>>,
    egraph: &Arc<EGraph>,
    num_paths_to_node: &Arc<HashMap<Id,i32>>,
    stats: &Arc<Mutex<Stats>>,
    cfg: &Arc<CompressionStepConfig>,
    tasks_of_node: &Arc<HashMap<Id, HashSet<usize>>>,
    cost_of_node: &Arc<HashMap<Id,i32>>, // a simple direct lookup thats just like inventionless cost
) -> Option<(Pattern,i32,i32)> {
    // * MULTITHREADING: CRITICAL SECTION START *
    // take the lock, which will be released immediately when this scope exits
    let mut shared_guard = shared.lock();
    let shared: &mut MutableMultithreadData = shared_guard.deref_mut();
    let lowest_donelist_utility = shared.lowest_donelist_utility;
    let old_donelist_len = shared.donelist.len();
    // drain from donelist_buf into the actual donelist
    shared.donelist.extend(donelist_buf.drain(..).filter(|done| done.utility > lowest_donelist_utility));
    if !cfg.no_stats { stats.lock().deref_mut().finished_invs += shared.donelist.len() - old_donelist_len; };
    // sort + truncate + update utility_pruning_cutoff and lowest_donelist_utility
    update_donelist_shared(shared, &cfg); // this also updates utility_pruning_cutoff
    // pull out utility_pruning_cutoff now that it has been updated (not earlier)
    let utility_pruning_cutoff = shared.utility_pruning_cutoff;

    let old_worklist_len = shared.worklist.len();
    let worklist_buf_len = worklist_buf.len();
    // drain from worklist_buf into the actual worklist
    shared.worklist.extend(worklist_buf.drain(..).filter(|heap_item| heap_item.pattern.utility_upper_bound > utility_pruning_cutoff));
    // num pruned by upper bound = num we were gonna add minus change in worklist length
    if !cfg.no_stats { stats.lock().deref_mut().upper_bound_fired += worklist_buf_len - (shared.worklist.len() - old_worklist_len); };

    // loop until we get a new (unpruned) item from the worklist
    let heap_item = loop {
        let heap_item = match shared.worklist.pop() {
            Some(heap_item) => heap_item,
            None => return None, // worklist was empty! we're done!
        };
        // prune if upper bound is too low (cutoff may have increased in the time since this was added to the worklist)
        if cfg.no_opt_upper_bound || heap_item.pattern.utility_upper_bound > shared.utility_pruning_cutoff {
            break heap_item;
        } else {
            if !cfg.no_stats { stats.lock().deref_mut().upper_bound_fired += 1; };
        }
    };
    // * MULTITHREADING: CRITICAL SECTION END *

    return Some((heap_item.pattern, utility_pruning_cutoff, lowest_donelist_utility));
}


fn pattern_loop(
    shared: Arc<Mutex<MutableMultithreadData>>,
    arg_of_zid_node: &HashMap<(ZId,Id),Arg>, // much like appzipper_of_node and yes shifted
    zids_of_node: Arc<Vec<Vec<ZId>>>,
    egraph: Arc<EGraph>,
    num_paths_to_node: Arc<HashMap<Id,i32>>,
    stats: Arc<Mutex<Stats>>,
    cfg: Arc<CompressionStepConfig>,
    tasks_of_node: Arc<HashMap<Id, HashSet<usize>>>,
    cost_of_node: Arc<HashMap<Id,i32>>, // a simple direct lookup thats just like inventionless cost
) {
    
    let mut worklist_buf: Vec<HeapItem> = Default::default();
    let mut donelist_buf: Vec<_> = Default::default();

    loop {
        
        let (pattern, weak_utility_pruning_cutoff, weak_lowest_donelist_utility) =
        match get_worklist_item(
            &mut worklist_buf,
            &mut donelist_buf,
            &shared,
            &arg_of_zid_node,
            &zids_of_node,
            &egraph,
            &num_paths_to_node,
            &stats,
            &cfg,
            &tasks_of_node,
            &cost_of_node,
        ) {
            Some(pattern) => pattern,
            None => return,
        };


        // todo pick a hole in the pattern to expand


        // this insane little piece of code just figures out which hole will give us
        // a set of location groups such that we're maximizing for the size of the largest
        // group. So this is a max{ max{ ... }} hence the two max() calls.
        let hole_idx: usize = pattern.holes.iter().enumerate()
            .map(|(hole_idx,hole_zid)| (hole_idx, *pattern.match_locations.iter()
                .map(|loc| arg_of_zid_node[&(*hole_zid, *loc)].node_type.clone()).counts().values().max().unwrap())).max_by_key(|&(_,max_count)| max_count).unwrap().0;

        let mut new_pattern: Pattern = pattern.clone();

        // expand the hole into everything it can become
        let hole_zid: ZId = new_pattern.holes.remove(hole_idx);

        // sort so that the groupby will work
        let mut match_locations = pattern.match_locations.clone();
        match_locations.sort_unstable_by_key(|loc| arg_of_zid_node[&(hole_zid, *loc)].node_type.clone());

        for (node_type, locs) in match_locations.into_iter().group_by(|loc| arg_of_zid_node[&(hole_zid, *loc)].node_type.clone()).into_iter() {
            let locs: Vec<Id> = locs.collect();
            if !cfg.no_opt_single_use && locs.len() < 2 {
                if !cfg.no_stats { stats.lock().deref_mut().single_use_wip_fired += 1; };
                continue; // too few uses
            }
            // todo add single task pruning
            // todo add pruning if we JUST added a free var as our node_type
            let utility_upper_bound: i32 = locs.iter().map(|loc| cost_of_node[loc]).sum();
            if utility_upper_bound < weak_utility_pruning_cutoff {
                if !cfg.no_stats { stats.lock().deref_mut().upper_bound_fired += 1; };
                continue; // too low utility
            }
            // todo add structural penalty so it can actually go negative
            if utility_upper_bound > 0 {
                let mut new_pattern = new_pattern.clone();
                new_pattern.match_locations = locs;
                new_pattern.utility_upper_bound = utility_upper_bound;

                match node_type {
                    NodeType::Lam => {
                        // todo add 1 more hole to new_pattern.holes
                        
                        new_pattern.body_utility += COST_NONTERMINAL;
                    }
                    NodeType::App => {
                         // todo add 2 more holes to new_pattern.holes
                         new_pattern.body_utility += COST_NONTERMINAL;
                    }
                    NodeType::Var(_) => {
                        new_pattern.body_utility += COST_TERMINAL;
                    }
                    NodeType::Prim(_) => {
                        new_pattern.body_utility += COST_TERMINAL;
                    }
                }
                assignments_of_pattern(
                    new_pattern,
                    weak_utility_pruning_cutoff,
                    weak_lowest_donelist_utility,
                    &mut worklist_buf,
                    &mut donelist_buf,
                    &shared,
                    &arg_of_zid_node,
                    &zids_of_node,
                    &egraph,
                    &num_paths_to_node,
                    &stats,
                    &cfg,
                    &tasks_of_node,
                    &cost_of_node
                );
            }
        }

        // add an argchoice, as long as it's actually abstracting a different thing over all locations
        if new_pattern.match_locations.iter().map(|loc| arg_of_zid_node[&(hole_zid, *loc)].node_type.clone()).all_equal() {
            if !cfg.no_stats { stats.lock().deref_mut().useless_abstract_fired += 1; };
        } else {
            new_pattern.arg_choices.push(hole_zid);
            assignments_of_pattern(
                new_pattern,
                weak_utility_pruning_cutoff,
                weak_lowest_donelist_utility,
                &mut worklist_buf,
                &mut donelist_buf,
                &shared,
                &arg_of_zid_node,
                &zids_of_node,
                &egraph,
                &num_paths_to_node,
                &stats,
                &cfg,
                &tasks_of_node,
                &cost_of_node
            );  
        }
        
    }

}





#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FinishedPattern {
    zids: Vec<LabelledZId>, // a hole gets moved into here when it becomes an argchoice, again these are in order of when they were added
    match_locations: Vec<Id>, // places where it applies
    utility: i32,
    compressive_utility: i32,
    arity: usize,
}

impl FinishedPattern {
    fn new(pattern: &Pattern, asn: &Assignment, utility: i32, compressive_utility: i32) -> Self {
        FinishedPattern {
            zids: pattern.arg_choices.iter().zip(asn.ivars.iter()).map(|(argchoice_zid,ivar)| LabelledZId::new(*argchoice_zid,*ivar as usize)).collect(),
            match_locations: asn.match_locations.last().unwrap().clone(),
            utility: utility,
            compressive_utility: compressive_utility,
            arity: *asn.max_ivar_used.last().unwrap() as usize + 1,
        }
    }
}

struct Assignment {
    ivars: Vec<i32>,
    max_ivar_used: Vec<i32>,
    match_locations: Vec<Vec<Id>>,
    first_ivar_use: Vec<usize>, // first_ivar_use[i] gives the index of the first use of #i in ivars
    can_extend: bool, // are we allowed to extend the current assignment by adding more ivars?
    ptr: usize,
}

impl Assignment {
    fn prune_branch(
        &mut self,
        pattern: &mut Pattern,
    ) {
        // if this an in-progress pattern then we care to update pruned_assignment_prefixes
        if !pattern.holes.is_empty() {
            // throw out any prefixes that this prefix subsumes
            pattern.pruned_assignment_prefixes.retain(|prefix| !prefix.starts_with(&self.ivars));
            // add the new prefix maintaining sorted order
            match pattern.pruned_assignment_prefixes.binary_search(&self.ivars) {
                Err(idx) => {pattern.pruned_assignment_prefixes.insert(idx, self.ivars.clone());},
                Ok(_) => unreachable!()
            }
        }
        self.can_extend = false; // after pruning a branch you can't immediately extend
    }
    fn next(
        &mut self,
        pattern: &Pattern,
        arg_of_zid_node: &HashMap<(ZId,Id),Arg>,
        cfg: &Arc<CompressionStepConfig>,
    ) -> bool {
        loop {
            // this only happens on the very first step
            if self.ptr == 0 { return true; }

            // prefer extending if we're allowed to
            if self.can_extend && self.ivars.len() < pattern.arg_choices.len() {
                self.ivars.push(0);
                self.ptr += 1;
                self.max_ivar_used.push(self.max_ivar_used[self.ptr - 1]);
                // filter for locations where the multiuse equality constraint holds.
                // note that we know that ivars[0] == 0 ie the first argchoice is always #0
                // so this is easier than in other cases where we have to look it up.
                self.match_locations.push(pattern.match_locations.iter()
                    .filter(|loc|
                        arg_of_zid_node[&(pattern.arg_choices[self.ptr], **loc)].id == 
                        arg_of_zid_node[&(pattern.arg_choices[0], **loc)].id).cloned().collect());
                return true
            }

            // since we couldnt' extend, we'd like to increment our ivar.
            // this is allowed if we aren't breaking max_arity and we aren't
            // breaking the constraint that the first use of #1 must come after #0 etc
            // (here, as long as the past ivars contain something at least as big as our
            // current ivar, it's okay to increment our current ivar)
            let new_ivar = self.ivars[self.ptr] + 1;
            if new_ivar < cfg.max_arity as i32
                && self.ivars[self.ptr] <= self.max_ivar_used[self.ptr - 1]
            {
                // increment ivar and update max_ivar_used
                self.ivars[self.ptr] = new_ivar;
                if new_ivar > *self.max_ivar_used.last().unwrap() {
                    self.max_ivar_used[self.ptr] = new_ivar;
                    // since this is a new record for largest ivar, we should
                    // add it as a new entry in first_ivar_use too.
                    self.first_ivar_use.push(self.ptr);
                    assert_eq!((self.first_ivar_use.len() - 1) as i32, new_ivar);
                }

                let first_ivar_use = self.first_ivar_use[self.ivars[self.ptr] as usize];

                self.match_locations.push(pattern.match_locations.iter()
                    .filter(|loc|
                        arg_of_zid_node[&(pattern.arg_choices[self.ptr], **loc)].id == 
                        arg_of_zid_node[&(pattern.arg_choices[first_ivar_use], **loc)].id).cloned().collect());
                
                // having incremented an ivar, we are now allowed to extend if we weren't already
                self.can_extend = true;
                return true
            }

            // we couldnt increment our ivar so we need to backtrack
            if *self.first_ivar_use.last().unwrap() == self.ptr {
                self.first_ivar_use.pop();
            }
            self.ivars.pop();
            self.max_ivar_used.pop();
            self.match_locations.pop();
            self.ptr -= 1;
            self.can_extend = false; // you cant extend forward right after backtracking
            if self.ivars.len() == 1 {
                return false // we backtracked to [0] so we're done
            }
        }
    }
}

fn assignments_of_pattern(
    mut pattern: Pattern,
    weak_utility_pruning_cutoff: i32,
    weak_lowest_donelist_utility: i32,
    worklist_buf: &mut Vec<HeapItem>,
    donelist_buf: &mut Vec<FinishedPattern>,
    shared: &Arc<Mutex<MutableMultithreadData>>,
    arg_of_zid_node: &HashMap<(ZId,Id),Arg>, // much like appzipper_of_node and yes shifted
    zids_of_node: &Arc<Vec<Vec<ZId>>>,
    egraph: &Arc<EGraph>,
    num_paths_to_node: &Arc<HashMap<Id,i32>>,
    stats: &Arc<Mutex<Stats>>,
    cfg: &Arc<CompressionStepConfig>,
    tasks_of_node: &Arc<HashMap<Id, HashSet<usize>>>,
    cost_of_node: &Arc<HashMap<Id,i32>>, // a simple direct lookup thats just like inventionless cost
) {

    let mut unfinished_pattern_succeeded: bool = false;
    let is_finished_pattern: bool = pattern.holes.is_empty();

    let mut asn = Assignment {
        ivars: vec![0],
        max_ivar_used: vec![0],
        match_locations: vec![pattern.match_locations.clone()],
        first_ivar_use: vec![0],
        can_extend: true,
        ptr: 0,
    };


    while asn.next(&pattern, &arg_of_zid_node, &cfg) {

        // prune if not used in any places
        if asn.match_locations.len() == 0 {
            asn.prune_branch(&mut pattern);
            continue;
        }

        // prune if only used in a single place
        if asn.match_locations.len() == 1 {
            if is_finished_pattern {
                if !cfg.no_stats { stats.lock().single_use_done_fired += 1; }
                asn.prune_branch(&mut pattern);
                continue;
            } else if !cfg.no_opt_single_use {
                if !cfg.no_stats { stats.lock().single_use_wip_fired += 1; }
                asn.prune_branch(&mut pattern);
                continue;
            }
        }

        // backtrack if we know this to be a bad (too low utility) prefix
        if pattern.pruned_assignment_prefixes.binary_search(&asn.ivars).is_ok() {
            asn.prune_branch(&mut pattern);
            continue;
        }

        // check upper bound, and discard + add to bad prefix list if it fails
        let utility_upper_bound: i32 = asn.match_locations.last().unwrap().iter().map(|loc| cost_of_node[loc]).sum();
        if utility_upper_bound < weak_utility_pruning_cutoff {
            asn.prune_branch(&mut pattern);
            continue;
        }

        // if its a finished pattern and doesnt beat the lowest donelist utility, itll never survive
        if is_finished_pattern && utility_upper_bound <= weak_lowest_donelist_utility {
            asn.prune_branch(&mut pattern);
            continue
        }

        // check for useless abstractions (ie same arg everywhere) which might have arison from our narrowing of the match_locations
        if pattern.arg_choices.iter()
            .any(|argchoice_zid| asn.match_locations.last().unwrap().iter()
                .map(|loc| arg_of_zid_node[&(*argchoice_zid, *loc)].node_type.clone()).all_equal())
        {
            if !cfg.no_stats { stats.lock().deref_mut().useless_abstract_fired += 1; };
            asn.prune_branch(&mut pattern);
            continue; // useless abstraction
        }

        // if we've assigned all the argchoices to ivars (and we know we already had a reasonable utility). Note this branch is irrelevant if we've already succeeded once and we're an unfinished pattern
        if asn.ivars.len() == pattern.arg_choices.len() && !unfinished_pattern_succeeded {
            if is_finished_pattern {
                // add to donelist
                donelist_buf.push(FinishedPattern::new(
                    &pattern,
                    &asn,
                    // todo fix the fact that im just passing this in as if its compressive utility :)
                    utility_upper_bound,
                    utility_upper_bound,
                ));
                continue; 
            } else {
                // worklist case - note that at this point we know we wont be able to
                // completely prune this worklist item, so we can break early if we want
                // OR more likely we should just keep going to improve our `pruned_assignment_prefixes`
                // since if we dont do that now our children might have to do repeat that work separately
                // for each child.
                unfinished_pattern_succeeded = true;
                if cfg.break_early_assignment {
                    break
                }
                continue; 
            }
        }

    }


    if !is_finished_pattern && unfinished_pattern_succeeded {
        worklist_buf.push(HeapItem::new(pattern))
    }


}





/// At the end of the day we convert our Inventions into InventionExprs to make
/// them standalone without needing to carry the EGraph around to figure out what
/// the body Id points to.
#[derive(Debug, Clone)]
pub struct Invention {
    pub body: Expr, // invention body (not wrapped in lambdas)
    pub arity: usize,
    pub name: String,
}
impl Invention {
    pub fn new(body: Expr, arity: usize, name: &str) -> Self {
        Self { body, arity, name: String::from(name) }
    }
    /// replace any #i with args[i], returning a new expression
    pub fn apply(&self, args: &[Expr]) -> Expr {
        assert_eq!(args.len(), self.arity);
        let map: HashMap<i32, Expr> = args.iter().enumerate().map(|(i,e)| (i as i32, e.clone())).collect();
        ivar_replace(&self.body, self.body.root(), &map)
    }
}

impl Display for Invention {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "[{} arity={}: {}]", self.name, self.arity, self.body)
    }
}


/// Does debruijn index shifting of a subtree, incrementing all Vars by the given amount
#[inline] // useful to inline since callsite can usually tell which Shift type is happening allowing further optimization
pub fn shift(e: Id, incr_by: i32, egraph: &mut EGraph, cache: &mut Option<RecVarModCache>) -> Option<Id> {
    let empty = &mut RecVarModCache::new();
    let seen: &mut RecVarModCache = cache.as_mut().unwrap_or(empty);

    recursive_var_mod(
        |actual_idx, _depth, _which_upward_ref, egraph| {
            Some(egraph.add(Lambda::Var(actual_idx + incr_by)))
        },
        false, // operate on Vars not IVars
        e,egraph,seen
    )
}

/// A node in an ZPath
/// Ord: Func < Body < Arg
#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
enum ZNode {
    // * order of variants here is important because the derived Ord will use it
    Func, // zipper went into the function, so Id is the arg
    Body, 
    Arg, // zipper went into the arg, so Id is the function
}

/// "zipper id" each unique zipper gets referred to by its zipper id
type ZId = usize;

/// "zipper path" this is a path like "Func Body Arg Body Arg Arg"
#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
struct ZPath {
    path: Vec<ZNode>,
    threaded_vars: Vec<i32>,
}
// type ZPath = Vec<ZNode>;

/// A Zipper is a single-argument single-use invention, so it's a subtree from the
/// original program with exactly one invention variable #0 used in exactly one place.
/// 
/// A Zipper has a `.path` specifying the path it takes through an expression
/// eg "Func Body Arg Body Arg Arg", along with a `.left` and `.right` specifying
/// the off-zipper elements. For example when the zipper goes to the left (ie path has
/// an `Arg`) the off-zipper element is whatever the Function was in the `app(func,arg)`.
/// This is stored as a Some(Id) referencing the subtree from the original program. Nones are
/// used in cases where an off element doesnt exist. The lengths of all 3 of these fields are the same.
/// 
/// Illustration of the 3 vectors side by side:
/// ```
/// left     | path | right
/// -------------------
/// None     | Func |  Some(23)
/// None     | Body |  None
/// Some(33) | Arg  |  None
/// None     | Func |  Some(45)
/// ```
#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
struct Zipper {
    zpath: ZPath,
    left: Vec<Option<Id>>,
    right: Vec<Option<Id>>,
}

/// A zipper (single-arg single-use invention) applied to an argument
#[derive(Debug,Clone, Eq, PartialEq, Hash)]
struct AppZipper {
    zipper: Zipper,
    arg: Id,
}

/// a zid referencing a specific ZPath and a #i index
#[derive(Debug,Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
struct LabelledZId {
    zid: ZId,
    ivar: usize // which #i argument this is, which also corresponds to args[i] ofc
}

/// A Zipper Tuple. This is a multiarg multiuse invention consisting of a list of zippers. This is the core data structure
/// representing a partially or completely constructed invention. Zippers get merged into it, extending
/// the `elems` field by one, `divergence_idxs` by one, and either `multiarg` or `multiuse` by one depending
/// on whether the newly added zipper is reusing an existing argument. Only zippers that are larger than the rightmost zipper can be
/// added to a ztuple
#[derive(Debug,Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
struct ZTuple {
    elems: Vec<LabelledZId>, // list of zids labelled with #is
    divergence_idxs: Vec<usize>, // locations where the zippers diverge from each other (lengths is 1 less than the number of zippers)
    multiarg: Vec<ZId>, // len=arity, gives the first zid added for each arg
    multiuse: Vec<ZId>, // gives the 2nd and onward zids added for each arg (not the first, thats in multiarg)
    arity: usize,
}

/// A partially constructed invention (`.ztuple`) along with the nodes where it is usable (`.nodes`),
/// the utility of the concrete part of the invention for a single usage (`.left_utility`) and an upper
/// bound on the total utility over all usages (`.utility_upper_bound`),
#[derive(Debug,Clone, Eq, PartialEq, PartialOrd, Ord)]
struct WorklistItem {
    ztuple: ZTuple,
    nodes: Vec<Id>, // nodes in the group
    left_utility: i32, // utility of a single usage
    utility_upper_bound: i32, // upper bound utility over all usages
}

/// A completely finished invention (`.ztuple`) along with the nodes where it is used (`.nodes`)
/// and the total utility of it over all usages.
#[derive(Debug,Clone)]
pub struct FinishedItem {
    ztuple: ZTuple,
    nodes: Vec<Id>, // nodes in the group
    utility: i32,
    compressive_utility: i32,
}



/// This contains all the data that needs to be shared between threads running derive_inventions().
/// Notably we only take the lock on this data once at the start of a loop iteration. If you have
/// additional shared data that you want to access/modify more often then it should be handled
/// separately from this.
struct KMutableMultithreadData {
    donelist: Vec<FinishedItem>,
    worklist: VecDeque<WorklistItem>,
    lowest_donelist_utility:i32,
    utility_pruning_cutoff: i32,
}

/// Various tracking stats
#[derive(Clone,Default, Debug)]
struct Stats {
    partial_invs: usize,
    finished_invs: usize,
    upper_bound_fired: usize,
    free_vars_done_fired: usize,
    free_vars_wip_fired: usize,
    single_use_done_fired: usize,
    single_task_done_fired: usize,
    single_use_wip_fired: usize,
    force_multiuse_fired: usize,
    useless_abstract_fired: usize,
}

/// Args for compression step
#[derive(Parser, Debug, Serialize, Clone)]
#[clap(name = "Stitch")]
pub struct CompressionStepConfig {
    /// max arity of inventions to find (will find all from 0 to this number inclusive)
    #[clap(short='a', long, default_value = "2")]
    pub max_arity: usize,

    /// num threads (no parallelism if set to 1)
    #[clap(short='t', long, default_value = "1")]
    pub threads: usize,

    /// Number of invention candidates compression_step should return. Raising this may weaken the efficacy of upper bound pruning
    /// unless --lossy-candidates is enabled.
    #[clap(short='n', long, default_value = "1")]
    pub inv_candidates: usize,

    /// 
    #[clap(long)]
    pub break_early_assignment: bool,

    /// By default we use a LIFO worklist but this is certainly something to explore more
    /// and this flag makes it fifo https://github.com/mlb2251/stitch/issues/31
    #[clap(long)]
    pub fifo_worklist: bool,

    /// By default we sort the worklist in decreasing zipper order before starting to process it,
    /// but this swaps it to increasing order. https://github.com/mlb2251/stitch/issues/31
    #[clap(long)]
    pub ascending_worklist: bool,

    /// Turning this on means that only the top invention will be guaranteed to be the best invention,
    /// and the 2nd best invention may not be the actual second best invention. Basically, this just enables
    /// pruning of everything that's worse than the best invention which could cause speedups depending on the domain.
    #[clap(long)]
    pub lossy_candidates: bool,

    /// disable caching (though caching isn't used for much currently)
    #[clap(long)]
    pub no_cache: bool,

    /// print out programs rewritten under invention
    #[clap(long,short='r')]
    pub show_rewritten: bool,

    /// disable the free variable pruning optimization
    #[clap(long)]
    pub no_opt_free_vars: bool,

    /// disable the single usage pruning optimization
    #[clap(long)]
    pub no_opt_single_use: bool,

    /// disable the single task pruning optimization
    #[clap(long)]
    pub no_opt_single_task: bool,

    /// disable the upper bound pruning optimization
    #[clap(long)]
    pub no_opt_upper_bound: bool,

    /// disable the force multiuse pruning optimization
    #[clap(long)]
    pub no_opt_force_multiuse: bool,

    /// disable the useless abstraction pruning optimization
    #[clap(long)]
    pub no_opt_useless_abstract: bool,

    /// Disable stat logging - note that stat logging in multithreading requires taking a mutex
    /// so it could be a source of slowdown in the multithreaded case, hence this flag to disable it.
    /// From some initial tests it seems to cause no slowdown anyways though.
    #[clap(long)]
    pub no_stats: bool,

    /// disables other_utility so the only utility is based on compressivity
    #[clap(long)]
    pub no_other_util: bool,
    
    /// disables context threading
    #[clap(long)]
    pub no_ctx_thread: bool,

    /// turns on eta long form (WIP)
    #[clap(long)]
    pub eta_long: bool,
}

impl CompressionStepConfig {
    pub fn no_opt(&mut self) {
        self.no_opt_free_vars = true;
        self.no_opt_single_use = true;
        self.no_opt_single_task = true;
        self.no_opt_upper_bound = true;
        self.no_opt_force_multiuse = true;
        self.no_opt_useless_abstract = true;
    }
}

impl WorklistItem {
    fn new(ztuple: ZTuple, nodes: Vec<Id>, left_utility: i32, utility_upper_bound: i32) -> WorklistItem {
        WorklistItem { ztuple: ztuple, nodes: nodes, left_utility: left_utility, utility_upper_bound: utility_upper_bound }
    }
}

impl FinishedItem {
    fn new(ztuple: ZTuple, nodes: Vec<Id>, utility: i32, compressive_utility: i32) -> FinishedItem {
        FinishedItem { ztuple, nodes, utility, compressive_utility }
    }
    fn to_invention(&self, name: &str, appzipper_of_node_zid: &HashMap<(Id,ZId),AppZipper>, egraph: &EGraph ) -> Invention {
        Invention::new(self.ztuple.to_expr(self.nodes[0], appzipper_of_node_zid, egraph), self.ztuple.arity, name)
    }
}

// impl HeapItem {
//     fn new(item: WorklistItem) -> HeapItem {
//         HeapItem { key: item.ztuple.elems.last().unwrap().zid as i32, item: item }
//     }
// }

impl Zipper {
    fn empty() -> Zipper {
        Zipper { zpath: ZPath::empty(), left: vec![], right: vec![] }
    }
}

impl ZPath {
    fn empty() -> ZPath {
        ZPath { path: vec![], threaded_vars: vec![] }
    }
}

impl AppZipper {
    fn new(zipper: Zipper, arg: Id) -> AppZipper {
        AppZipper { zipper: zipper, arg: arg }
    }
    /// clone this applied single-arg single-use invention and extend the zipper by 1 at the top
    /// of the zipper. This is used when constructing the appzippers in a bottom up way
    #[inline]
    fn clone_prepend(&self, new: ZNode, id: Option<Id>) -> AppZipper {
        let mut appzipper: AppZipper = self.clone();
        match new {
            ZNode::Func => {
                assert!(id.is_some());
                appzipper.zipper.left.insert(0,None);
                appzipper.zipper.right.insert(0,id);
            },
            ZNode::Arg => {
                assert!(id.is_some());
                appzipper.zipper.left.insert(0,id);
                appzipper.zipper.right.insert(0,None);
            },
            ZNode::Body => {
                assert!(id.is_none());
                appzipper.zipper.left.insert(0,None);
                appzipper.zipper.right.insert(0,None);        
            },
        }
        appzipper.zipper.zpath.path.insert(0,new);
        appzipper
    }
}

impl LabelledZId {
    fn new(zid: ZId, ivar: usize) -> LabelledZId {
        LabelledZId { zid: zid, ivar: ivar }
    }
}

impl ZTuple {
    /// make an arity 0 ztuple 
    fn empty() -> ZTuple {
        ZTuple { elems: vec![], divergence_idxs: vec![], multiarg: vec![], multiuse: vec![], arity: 0}
    }
    /// make a new single-zipper ztuple
    fn single(zid: ZId) -> ZTuple {
        ZTuple { elems: vec![LabelledZId::new(zid, 0)], divergence_idxs: vec![], multiarg: vec![zid], multiuse: vec![], arity: 1 }
    }
    /// extend ztuple, returning a new ztuple with one extra argument (original is unchanged)
    fn extend(&self, elem: LabelledZId, div_idx: usize, is_multiuse: bool) -> ZTuple {
        let mut res = self.clone();
        res.divergence_idxs.push(div_idx);
        if is_multiuse {
            res.multiuse.push(elem.zid);
        } else {
            res.multiarg.push(elem.zid);
            res.arity += 1;
        }
        res.elems.push(elem);
        res
    }
    /// convert a ztuple to an Expr. This is for extracting out the final complete inventions at the very end so that
    /// there are no more ZIds or Ids and everything is self contained without references to shared data structures.
    fn to_expr(&self, node: Id, appzipper_of_node_zid: &HashMap<(Id,ZId),AppZipper>, egraph: &EGraph) -> Expr {
        if self.elems.is_empty() {
            return extract(node, egraph);  // arity 0
        }

        let mut elem_idx: usize = 0;
        let mut zipper: &Zipper = &appzipper_of_node_zid[&(node,self.elems[elem_idx].zid)].zipper;
        let mut depth: usize = zipper.zpath.path.len() - 1;
        fn start_ivar(ztuple: &ZTuple, elem_idx: usize, zipper: &Zipper) -> Expr {
            let mut expr = Expr::ivar(ztuple.elems[elem_idx].ivar as i32);
            // do any threading, turning #i into something like ((#i $1) $0)
            for var in zipper.zpath.threaded_vars.iter().rev() {
                expr = Expr::app(expr, Expr::var(*var as i32));
            }
            expr
        }
        let mut expr = start_ivar(self, elem_idx, zipper);
        let mut diverged: Vec<(usize,Expr)> = vec![];

        // we do this by a loop where we start at the bottom of the leftmost zipper and gradually extract the Expr bottom up,
        // and whenever we hit a divergence point we store the Expr and jump to the bottom of the next zipper and repeat, being
        // careful to pop and merge our stored expressions as we pass the divergence point a second time from the righthand side.
        loop {
            // encounter divergence point to our right
            if elem_idx < self.divergence_idxs.len() && depth == self.divergence_idxs[elem_idx] {
                // we should diverge to the right
                assert_eq!(zipper.zpath.path[depth], ZNode::Func);
                diverged.push((depth,expr));
                elem_idx += 1;
                zipper = &appzipper_of_node_zid[&(node,self.elems[elem_idx].zid)].zipper;
                depth = zipper.zpath.path.len() - 1;
                expr = start_ivar(self, elem_idx, zipper);
                continue;
            }
            // pass a divergence point to our left that we stored something for
            if !diverged.is_empty() && depth == diverged.last().unwrap().0 {
                // we should ignore our normal Some(f) and instead use the stored diverged expr
                assert_eq!(zipper.zpath.path[depth], ZNode::Arg);
                expr = Expr::app(diverged.pop().unwrap().1, expr);
                if depth == 0 { break }
                depth -= 1;
                continue;
            }

            // normal step upward by 1
            match (&zipper.zpath.path[depth], &zipper.left[depth], &zipper.right[depth]) {
                (ZNode::Arg, Some(f), None) => { expr = Expr::app(extract(*f,egraph), expr); },
                (ZNode::Func, None, Some(x)) => { expr = Expr::app(expr, extract(*x,egraph)); },
                (ZNode::Body, None, None) => { expr = Expr::lam(expr); },
                _ => panic!("malformed zipper"),
            }
            if depth == 0 { break }
            depth -= 1;
        }

        expr
    }
}

/// Construct all single-argument single-usage inventions in a bottom up manner. This returns around O(N^2) inventions
/// since it's any O(N) choice of a parent to be the root of the invention, and any choice of a single descendent of that
/// parent to be the abstracted #0. Returns a map from nodes to the list of single-arg single-use inventions that can be
/// used at that node.
fn get_appzippers(treenodes: &[Id], no_cache:bool, egraph: &mut EGraph, cfg: &CompressionStepConfig) -> HashMap<Id,Vec<AppZipper>> {
    let mut all_appzippers: HashMap<Id,Vec<AppZipper>> = Default::default();
    let cache: &mut Option<RecVarModCache> = &mut if no_cache { None } else { Some(HashMap::new()) };

    // todo i think we can do something reasonable here.
    // todo hard to decide exactly how it should go

    let mut zid_of_zpath: HashMap<Vec<ZNode>, ZId> = Default::default();
    let mut zpath_of_zid: Vec<Vec<ZNode>> = Default::default();
    let mut arg_of_zid_node: HashMap<(ZId,Id),Arg> = Default::default();
    let mut zids_of_node: HashMap<Id,Vec<ZId>> = Default::default();
    
    for treenode in treenodes.iter() {
        // println!("processing id={}: {}", treenode, extract(*treenode, egraph) );

        // im essentially using the egraph just for its structural hashing rn
        assert!(egraph[*treenode].nodes.len() == 1);
        // clone to appease the borrow checker
        let node = egraph[*treenode].nodes[0].clone();

        //==================================//
        // *** PROPAGATE/CREATE APPLAMS *** //
        //==================================//
        let mut appzippers: Vec<AppZipper> = vec![];
        
        // any node can become the identity function (the empty zipper)
        appzippers.push(AppZipper::new(Zipper::empty(), *treenode));

        match node {
            Lambda::IVar(_) => { panic!("attempted to abstract an IVar") }
            Lambda::Var(_) | Lambda::Prim(_) | Lambda::Programs(_) => {},
            Lambda::App([f,x]) => {

                let ref f_appzippers = all_appzippers[&f];
                let ref x_appzippers = all_appzippers[&x];

                // bubbling from the left:
                // (app f x) == (app (appzipper body arg) x) => (appzipper (app body upshift(x)) arg)
                // note no shifting is needed thanks to IVars
                for f_appzipper in f_appzippers.iter() {
                    // bubble out of function so zipper should point left so Func
                    let new: AppZipper = f_appzipper.clone_prepend(ZNode::Func,Some(x));
                    appzippers.push(new);
                }

                // bubbling from the right:
                // (app f x) == (app f (appzipper body arg)) => (appzipper (app upshift(f) body) arg)
                // note no shifting is needed thanks to IVars
                for x_appzipper in x_appzippers.iter() {
                    // bubble out of arg so zipper should point right so Arg
                    let new: AppZipper = x_appzipper.clone_prepend(ZNode::Arg,Some(f));
                    appzippers.push(new);
                }

            },
            Lambda::Lam([b]) => {
                let ref b_appzippers = all_appzippers[&b];
                // bubbling up over the lambda:
                // (lam b) == (lam (appzipper body arg)) => (appzipper (lam body) downshift(arg))
                // where:
                //  - arg must not have any upward refs to $0 in it   since we cant jump over a lambda we point to
                //    > (in the multiarg appzipper case, none of them can have $0)
                //  - in the pre-ivar era this required a RotateShift which turned out to be a huge speed bottleneck
                //    as it created tons of new nodes in the egraph. This is no longer needed with ivars. No shfiting at lal!

                for b_appzipper in b_appzippers.iter() {
                    // unless context threading is turned on, you can't bubble an appzipper over a lambda if its arg refers to the lambda!
                    if cfg.no_ctx_thread && egraph[b_appzipper.arg].data.free_vars.contains(&0) {
                        continue;
                    }

                    let mut new: AppZipper = b_appzipper.clone_prepend(ZNode::Body,None);
                    
                    // downshift the args since the lambda above them moved below them (earlier we made sure none of them had pointers to it)

                    if egraph[b_appzipper.arg].data.free_vars.contains(&0) {
                        // context threading: wrap the arg in a lambda but leave it otherwise unchanged
                        new.arg = egraph.add(Lambda::Lam([b_appzipper.arg]));
                        // depth in number of lambdas (including the newly added one)
                        let depth = new.zipper.zpath.path.iter().filter(|x| **x == ZNode::Body).count();
                        // thread variables equal to depth-1, for example if there are no lambdas (except the newly added one) this will thread $0
                        new.zipper.zpath.threaded_vars.push(depth as i32 - 1);
                        // todo modify zipper to account for this
                    } else {
                        // no threading and no $0 so it's safe to downshift everything in the argument
                        new.arg = shift(b_appzipper.arg, -1, egraph, cache).unwrap();
                    }
                    // println!("Bubbled over lam:\n\t{}\n{}", extract(*treenode,egraph), new.to_string(egraph));
                    appzippers.push(new);
                }
            },
        }
        all_appzippers.insert(*treenode, appzippers);
    }

    if cfg.eta_long {
        for treenode in treenodes.iter() {
            if let Lambda::App([f,_]) = egraph[*treenode].nodes[0] {
                // this for eta long form / dreamcoder compatability: no appzipper bodies can be rooted to the left of an App
                // because that means the body is a function type, which isnt allowed. For example an arity 2 invention with a
                // function type body would be effectively arity 3 and dreamcoder doesnt support this sort of thing.
                all_appzippers.get_mut(&f).unwrap().clear();
            }
        }
    }

    // note that we must be very careful pruning here. Most pruning isnt allowed, for example you cant prune things
    // that have free variables out bc if those free vars are on the leading edge you could still merge them away later
    all_appzippers.iter_mut().for_each(|(_,appzippers)| {
        appzippers.retain(|appzipper|
            !appzipper.zipper.zpath.path.is_empty() // no identity function
            // no toplevel abstraction. This is to mirror dreamcoder and so that
            // rewritten programs actually are things that a top down search could find,
            // in particular because when you come across an arrow typed hole in top down
            // search (eg a HOF argument) you autogenerate lambdas and then go into the body.
            && appzipper.zipper.zpath.path[0] != ZNode::Body 
        )});

    all_appzippers
}

/// utility of a fragment of a zipper, specifically a left edge (the left/right
/// distinction is just so we can include the nonterminal cost in the left edge)
#[inline]
fn left_edge_utility(edge: &[Option<Id>], egraph: &EGraph) -> i32 {
    edge.len() as i32 * COST_NONTERMINAL + // there is 1 nonterminal used at each node of the zipper of course (it's either an App or a Lam)
    edge.iter().filter_map(|option_id|
        option_id.map(|id| egraph[id].data.inventionless_cost)).sum::<i32>()
}
/// utility of the right a fragment of a zipper
#[inline]
fn right_edge_utility(edge: &[Option<Id>], egraph: &EGraph) -> i32 {
    edge.iter().filter_map(|option_id|
        option_id.map(|id| egraph[id].data.inventionless_cost)).sum::<i32>()
}

/// check if the edge has free variables
#[inline]
fn edge_has_free_vars(edge: &[Option<Id>], path: &[ZNode], mut depth: i32, egraph: &EGraph) -> bool {
    // return false;
    debug_assert_eq!(edge.len(), path.len());
    for (edge,node) in edge.iter().zip(path.iter()) {
        if *node == ZNode::Body {
            depth += 1;
            continue;
        }
        if let Some(id) = edge {
            if egraph[*id].data.free_vars.iter().any(|i| *i - depth >= 0) {
                return true;
            }
        }
    }
    return false;
}

/// Returns the first index where the two edges diverge
#[inline]
fn divergence_idx(left: &[ZNode], right: &[ZNode]) -> usize {
    for i in 0..left.len() {
        debug_assert!(i < right.len(), "right is a prefix of left");
        if left[i] != right[i] {
            debug_assert_eq!(left[i], ZNode::Func, "left: {:?}, right: {:?}", left, right);
            debug_assert_eq!(right[i], ZNode::Arg, "left: {:?}, right: {:?}", left, right);
            return i;
        }
    }
    panic!("right does not diverge from left")
}


// #[derive(ArgEnum, Clone, Debug, Serialize, Parser)]
// enum WorklistType {
//     #[clap(arg_enum, long = "foobaar")]
//     Fifo,
//     #[clap(arg_enum, long = "foofbar")]
//     Lifo
// }

// #[derive(ArgEnum,Clone, Debug, Serialize, Parser)]
// enum WorklistSort {
//     Forward,
//     #[clap(arg_enum, long = "foodbar")]
//     Reverse,
//     #[clap(arg_enum, long = "fzoobar")]
//     Shuffle,
// }


#[derive(Debug, Clone)]
pub struct CompressionStepResult {
    pub inv: Invention,
    pub rewritten: Expr,
    pub rewritten_dreamcoder: Vec<String>,
    pub done: FinishedItem,
    pub expected_cost: i32,
    pub final_cost: i32,
    pub multiplier: f64,
    pub multiplier_wrt_orig: f64,
    pub uses: i32,
    pub use_exprs: Vec<Expr>,
    pub use_args: Vec<Vec<Expr>>,
    pub dc_inv_str: String,
    pub initial_cost: i32,
}

impl CompressionStepResult {
    fn new(done: FinishedItem, programs_node: Id, inv_name: &str, appzipper_of_node_zid: &HashMap<(Id,ZId),AppZipper>,  num_paths_to_node: &HashMap<Id,i32>, egraph: &mut EGraph, past_invs: &Vec<CompressionStepResult>) -> Self {
        let initial_cost = egraph[programs_node].data.inventionless_cost;

        // cost of the very first initial program before any inventions
        let very_first_cost = if let Some(past_inv) = past_invs.first() { past_inv.initial_cost } else { initial_cost };

        let inv = done.to_invention(inv_name, appzipper_of_node_zid, egraph);
        let rewritten: Expr = rewrite_with_invention_egraph(programs_node, &inv, egraph);

        let expected_cost = initial_cost - done.compressive_utility;
        let final_cost = rewritten.cost();
        if expected_cost != final_cost {
            println!("*** expected cost {} != final cost {}", expected_cost, final_cost);
        }
        let multiplier = initial_cost as f64 / final_cost as f64;
        let multiplier_wrt_orig = very_first_cost as f64 / final_cost as f64;
        let uses = done.nodes.iter().map(|node| num_paths_to_node[node]).sum::<i32>();
        let use_exprs: Vec<Expr> = done.nodes.iter().map(|node| extract(*node, egraph)).collect();
        let use_args: Vec<Vec<Expr>> = done.nodes.iter().map(|node|
            done.ztuple.multiarg.iter().map(|zid|
                extract(appzipper_of_node_zid[&(*node,*zid)].arg, egraph)
            ).collect()).collect();
        
        // dreamcoder compatability
        let dc_inv_str: String = dc_inv_str(&inv, past_invs);
        // Rewrite to dreamcoder syntax with all past invention
        // we rewrite "inv1)" and "inv1 " instead of just "inv1" because we dont want to match on "inv10"
        let rewritten_dreamcoder: Vec<String> = rewritten.split_programs().iter().map(|p|{
            let mut res = p.to_string();
            for past_inv in past_invs {
                res = replace_prim_with(&res, &past_inv.inv.name, &past_inv.dc_inv_str);
                // res = res.replace(&format!("{})",past_inv.inv.name), &format!("{})",past_inv.dc_inv_str));
                // res = res.replace(&format!("{} ",past_inv.inv.name), &format!("{} ",past_inv.dc_inv_str));
            }
            res = replace_prim_with(&res, &inv_name, &dc_inv_str);
            // res = res.replace(&format!("{})",inv_name), &format!("{})",dc_inv_str));
            // res = res.replace(&format!("{} ",inv_name), &format!("{} ",dc_inv_str));
            res
        }).collect();

        CompressionStepResult { inv, rewritten, rewritten_dreamcoder, done, expected_cost, final_cost, multiplier, multiplier_wrt_orig, uses, use_exprs, use_args, dc_inv_str, initial_cost }
    }
    pub fn json(&self) -> serde_json::Value {        
        let use_exprs: Vec<String> = self.use_exprs.iter().map(|expr| expr.to_string()).collect();
        let use_args: Vec<String> = self.use_args.iter().map(|args| format!("{} {}", self.inv.name, args.iter().map(|expr| expr.to_string()).collect::<Vec<String>>().join(" "))).collect();
        let all_uses: Vec<serde_json::Value> = use_exprs.iter().zip(use_args.iter()).sorted().map(|(expr,args)| json!({args: expr})).collect();

        json!({            
            "body": self.inv.body.to_string(),
            "dreamcoder": self.dc_inv_str,
            "arity": self.inv.arity,
            "name": self.inv.name,
            "rewritten": self.rewritten.split_programs().iter().map(|p| p.to_string()).collect::<Vec<String>>(),
            "rewritten_dreamcoder": self.rewritten_dreamcoder,
            "utility": self.done.utility,
            "expected_cost": self.expected_cost,
            "final_cost": self.final_cost,
            "multiplier": self.multiplier,
            "multiplier_wrt_orig": self.multiplier_wrt_orig,
            "num_uses": self.uses,
            "uses": all_uses,
        })
    }
}

impl fmt::Display for CompressionStepResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.expected_cost != self.final_cost {
            write!(f,"[cost mismatch of {}] ", self.expected_cost - self.final_cost)?;
        }
        write!(f, "utility: {} | final_cost: {} | {:.2}x | uses: {} | body: {}",
            self.done.utility, self.final_cost, self.multiplier, self.uses, self.inv)
    }
}

/// sort the donelist by utility, truncate to cfg.inv_candidates, update the lowest_donelist_utility to be the lowest utility,
/// update utility_pruning_cutoff to be the highest utility if --lossy-candidates is set else the lowest utility
fn update_donelist(donelist: &mut Vec<FinishedItem>, cfg: &CompressionStepConfig, lowest_donelist_utility: &mut i32, utility_pruning_cutoff: &mut i32) {
    // sort in decreasing order by utility primarily, and break ties using the ztuple (just in order to be deterministic!)
    donelist.sort_unstable_by(|a,b| (b.utility,&b.ztuple).cmp(&(a.utility,&a.ztuple)));
    donelist.truncate(cfg.inv_candidates);
    *lowest_donelist_utility = donelist.last().map(|x|x.utility).unwrap_or(0);
    *utility_pruning_cutoff = if cfg.lossy_candidates { donelist.first().map(|x|x.utility).unwrap_or(0) } else { donelist.last().map(|x|x.utility).unwrap_or(0) };
}

/// sort the donelist by utility, truncate to cfg.inv_candidates, update the lowest_donelist_utility to be the lowest utility,
/// update utility_pruning_cutoff to be the highest utility if --lossy-candidates is set else the lowest utility
fn kupdate_donelist_shared(shared: &mut KMutableMultithreadData, cfg: &CompressionStepConfig) {
    // sort in decreasing order by utility primarily, and break ties using the ztuple (just in order to be deterministic!)
    shared.donelist.sort_unstable_by(|a,b| (b.utility,&b.ztuple).cmp(&(a.utility,&a.ztuple)));
    shared.donelist.truncate(cfg.inv_candidates);
    shared.lowest_donelist_utility = shared.donelist.last().map(|x|x.utility).unwrap_or(0);
    shared.utility_pruning_cutoff = if cfg.lossy_candidates { shared.donelist.first().map(|x|x.utility).unwrap_or(0) } else { shared.donelist.last().map(|x|x.utility).unwrap_or(0) };
}

/// sort the donelist by utility, truncate to cfg.inv_candidates, update the lowest_donelist_utility to be the lowest utility,
/// update utility_pruning_cutoff to be the highest utility if --lossy-candidates is set else the lowest utility
fn update_donelist_shared(shared: &mut MutableMultithreadData, cfg: &CompressionStepConfig) {
    // sort in decreasing order by utility primarily, and break ties using the zids (just in order to be deterministic!)
    shared.donelist.sort_unstable_by(|a,b| (b.utility,&b.zids).cmp(&(a.utility,&a.zids)));
    shared.donelist.truncate(cfg.inv_candidates);
    shared.lowest_donelist_utility = shared.donelist.last().map(|x|x.utility).unwrap_or(0);
    shared.utility_pruning_cutoff = if cfg.lossy_candidates { shared.donelist.first().map(|x|x.utility).unwrap_or(0) } else { shared.donelist.last().map(|x|x.utility).unwrap_or(0) };
}



// /// sort the donelist by utility, truncate to cfg.inv_candidates, update the lowest_donelist_utility to be the lowest utility,
// /// update utility_pruning_cutoff to be the highest utility if --lossy-candidates is set else the lowest utility
// fn update_donelist_threaded(shared: Arc<Mutex<KMutableMultithreadData>>, cfg: Arc<CompressionStepConfig>) {
//     // sort in decreasing order by utility primarily, and break ties using the ztuple (just in order to be deterministic!)
//     donelist.sort_unstable_by(|a,b| (b.utility,&b.ztuple).cmp(&(a.utility,&a.ztuple)));
//     donelist.truncate(cfg.inv_candidates);
//     *lowest_donelist_utility = donelist.last().map(|x|x.utility).unwrap_or(0);
//     *utility_pruning_cutoff = if cfg.lossy_candidates { donelist.first().map(|x|x.utility).unwrap_or(0) } else { donelist.last().map(|x|x.utility).unwrap_or(0) };
// }

/// This utility directly corresponds to decrease in program cost of the
/// final program tree once it has been rewritten with the invention. Program
/// cost is leaf_nodes * 100 + non_leaf_nodes * 1, so dividing a cost (or a utility)
/// by 100 will give approximately the number of leaf nodes.
/// 
/// At a very high level, we can calculate this utility as:
///     (num places the invention is useful) * (size of invention body)
/// However it's a little more complicated due to inventions re-using their variables
/// and the slight cost of using the invention primitive itself.
/// 
/// `left_utility`: utility of a single use of the whole invention except the righthand edge
/// `right_utility`: utility of the righthand edge of the invention
fn compressive_utility(
    body_utility: i32,
    ztuple: &ZTuple,
    nodes: &Vec<Id>,
    num_paths_to_node: &HashMap<Id,i32>,
    egraph: &EGraph,
    appzipper_of_node_zid: &HashMap<(Id,ZId),AppZipper>,
) -> i32 {


    // tiny penalty: there's one extra `lam` introduced (compared to original programs) for each instance of threading used in each arg
    let ctx_thread_penalty = - COST_NONTERMINAL * ztuple.multiarg.iter().map(|zid| appzipper_of_node_zid[&(nodes[0],*zid)].zipper.zpath.threaded_vars.len()).sum::<usize>() as i32;
    // it costs a tiny bit to apply the invention, for example (app (app inv0 x) y) incurs a cost
    // of COST_TERMINAL for the `inv0` primitive and 2 * COST_NONTERMINAL for the two `app`s.
    let app_penalty = - (COST_TERMINAL + COST_NONTERMINAL * ztuple.arity as i32) + ctx_thread_penalty;
    // multiuse utility depends on the size of the argument that's being used in multiple places. We can
    // look up that argument using appzipper_of_node_zid since ztuple.multiuses gives us the zids for the multiuse
    // cases (leaving out the original use)
    let global_multiuse_utility = ztuple.multiuse.iter()
        .map(|&arg_zid| // for each extra use of a multiuse arg
            nodes.iter().map(|node| // for each node
                num_paths_to_node[node] * // account for same node being used in multiple subtrees
                egraph[appzipper_of_node_zid[&(*node,arg_zid)].arg].data.inventionless_cost
            ).sum::<i32>()
        ).sum::<i32>();
    // total number of places the invention is used. num_paths_to_node accounts for structural hashing
    let num_uses = nodes.iter().map(|node| num_paths_to_node[node]).sum::<i32>();
    // utility = num_uses * (cost of applying invention + invention body utility) + multiuse utility
    num_uses * (app_penalty + body_utility) + global_multiuse_utility
}

/// This utility is just for any utility terms that we care about that don't directly correspond
/// to changes in size that come from rewriting with an invention
fn other_utility(
    body_utility: i32,
    cfg: &CompressionStepConfig,
) -> i32 {
    if cfg.no_other_util { return 0; }
    // this is a bit like the structure penalty from dreamcoder except that
    // that penalty uses inlined versions of nested inventions.
    let structure_penalty = - body_utility * 3 / 2;
    structure_penalty
}

/// This takes a partial invention and gives an upper bound on the maximum
/// compressive_utility() that any completed offspring of this partial invention could have.
fn compressive_utility_upper_bound(
    left_utility: i32,
    global_right_utility_upper_bound: i32,
    ztuple: &ZTuple,
    nodes: &Vec<Id>,
    num_paths_to_node: &HashMap<Id,i32>,
    egraph: &EGraph,
    appzipper_of_node_zid: &HashMap<(Id,ZId),AppZipper>,
) -> i32 {
    // safe bound: this penalty will only increase in offspring so doing the penalty-so-far like this is safe
    let ctx_thread_penalty = - COST_NONTERMINAL * ztuple.multiarg.iter().map(|zid| appzipper_of_node_zid[&(nodes[0],*zid)].zipper.zpath.threaded_vars.len()).sum::<usize>() as i32;
    // safe bound: arity will only increase in offspring inventions, so this term will only
    // get more negative, so this bound is safe.
    let app_penalty = - (COST_TERMINAL + COST_NONTERMINAL * ztuple.arity as i32) + ctx_thread_penalty;
    // safe bound: this is an exact utility for all the multiuse that's happened so far. As long
    // as right_utility_upper_bound incorporates any benefits from possible future multiuse, this is okay.
    let global_multiuse_utility = ztuple.multiuse.iter()
        .map(|&arg_zid| // for each extra use of a multiuse arg
            nodes.iter().map(|node| // for each node
                num_paths_to_node[node] * // account for same node being used in multiple subtrees
                egraph[appzipper_of_node_zid[&(*node,arg_zid)].arg].data.inventionless_cost
            ).sum::<i32>()
        ).sum::<i32>();
    // safe bound: number of usage locations will only decrease in offspring inventions
    let num_uses = nodes.iter().map(|node| num_paths_to_node[node]).sum::<i32>();
    // safe bound: summing a bunch of safe bounds is safe
    num_uses * (app_penalty + left_utility) + global_multiuse_utility + global_right_utility_upper_bound
}

/// This takes a partial invention and gives an upper bound on the maximum
/// other_utility() that any completed offspring of this partial invention could have.
fn other_utility_upper_bound(
    left_utility: i32,
    cfg: &CompressionStepConfig,
) -> i32 {
    if cfg.no_other_util { return 0; }
    // safe bound: since structure_penalty is negative an upper bound is anything less negative or exact. Since
    // left_utility < body_utility we know that this will be a less negative bound.
    let structure_penalty = - left_utility * 3 / 2;
    structure_penalty
}

/// Multistep compression. See `compression_step` if you'd just like to do a single step of compression.
pub fn compression(
    programs_expr: &Expr,
    iterations: usize,
    cfg: &CompressionStepConfig,
    tasks: &Vec<String>,
) -> Vec<CompressionStepResult> {
    let mut rewritten: Expr = programs_expr.clone();
    let mut step_results: Vec<CompressionStepResult> = Default::default();

    let tstart = std::time::Instant::now();

    for i in 0..iterations {
        println!("{}",format!("\n=======Iteration {}=======",i).blue().bold());
        let inv_name = format!("fn_{}",step_results.len());

        // call actual compression
        let res: Vec<CompressionStepResult> = compression_step(
            &rewritten,
            &inv_name,
            &cfg,
            &step_results,
            tasks);

        if !res.is_empty() {
            // rewrite with the invention
            let res: CompressionStepResult = res[0].clone();
            rewritten = res.rewritten.clone();
            println!("Chose Invention {}: {}", res.inv.name, res);
            step_results.push(res);
        } else {
            println!("No inventions found at iteration {}",i);
            break;
        }
    }

    println!("{}","\n=======Compression Summary=======".blue().bold());
    println!("Found {} inventions", step_results.len());
    println!("Cost Improvement: ({:.2}x better) {} -> {}", compression_factor(programs_expr,&rewritten), programs_expr.cost(), rewritten.cost());
    for i in 0..step_results.len() {
        let res = &step_results[i];
        println!("{} ({:.2}x wrt orig): {}" ,res.inv.name, compression_factor(programs_expr, &res.rewritten), res);
    }
    println!("Time: {}ms", tstart.elapsed().as_millis());
    step_results
}

/// Takes a set of programs as an Expr with Programs as its root, and does one full step of compresison.
/// Returns the top Inventions and the Expr rewritten under that invention along with other useful info in CompressionStepResult
/// The number of inventions returned is based on cfg.inv_candidates
pub fn compression_step(
    programs_expr: &Expr,
    new_inv_name: &str, // name of the new invention, like "inv4"
    cfg: &CompressionStepConfig,
    past_invs: &Vec<CompressionStepResult>, // past inventions we've found
    tasks: &Vec<String>,
) -> Vec<CompressionStepResult> {

    // build the egraph. We'll just be using this as a structural hasher we don't use rewrites at all. All eclasses will always only have one node.
    let mut egraph: EGraph = Default::default();
    let programs_node = egraph.add_expr(programs_expr.into());
    egraph.rebuild();

    // println!("Initial egraph:\n\t{}\n", egraph_info(&egraph));
    // if args.render_initial {
    //     save(&egraph, "0_programs", &out_dir);
    // }

    let treenodes: Vec<Id> = topological_ordering(programs_node,&egraph);
    assert!(usize::from(*treenodes.iter().max().unwrap()) == treenodes.len() - 1); // ensures we can safely just use Vecs of length treenodes.len() to store various nodewise things

    // populate num_paths_to_node so we know how many different parts of the programs tree
    // a node participates in (ie multiple uses within a single program or among programs)
    let num_paths_to_node: HashMap<Id,i32> = num_paths_to_node(programs_node, &treenodes, &egraph);

    let tstart_total = std::time::Instant::now();

    let tstart = std::time::Instant::now();
    let all_appzippers = get_appzippers(&treenodes, cfg.no_cache, &mut egraph, cfg);
    println!("get_appzippers: {:?}ms", tstart.elapsed().as_millis());

    let tstart = std::time::Instant::now();

    // flatten all the appzippers (single arg single use inventions) to get the list of zipper paths, then sort/dedup.
    let mut zpaths: Vec<ZPath> = all_appzippers.values().flatten().map(|appzipper| appzipper.zipper.zpath.clone()).collect();
    println!("{} total paths (incl dupes)", zpaths.len());
    zpaths.sort();
    zpaths.dedup();
    println!("{} paths", zpaths.len());
    println!("collect paths and dedup: {:?}ms", tstart.elapsed().as_millis());

    // define all the important data structures for compression
    let mut appzipper_of_node_zid: HashMap<(Id,ZId),AppZipper> = Default::default(); // lookup an appzipper from a node and zid
    let mut zids_of_node: Vec<Vec<ZId>> = vec![vec![]; treenodes.len()]; // lookup all zids that a node can use
    let mut nodes_of_zid: Vec<Vec<Id>> = vec![vec![]; zpaths.len()]; // look up all nodes that a zid can be used at
    let mut first_mergeable_zid_of_zid: Vec<ZId> = Default::default(); // used for speed; lets you quickly lookup the smallest mergable (non-suffix) zid larger than your current zid
    let mut worklist: VecDeque<WorklistItem> = Default::default(); // worklist that holds partially constructed inventions
    // let mut worklist: BinaryHeap<HeapItem> = Default::default();
    let mut donelist: Vec<FinishedItem> = Default::default(); // completed inventions will go here
    let tasks_of_node: HashMap<Id, HashSet<usize>> = associate_tasks(programs_node, &egraph, tasks);

    // populate first_mergeable_zid_of_zid
    for (i,zpath) in zpaths.iter().enumerate() {
        // first path after `i` where the path isnt a prefix is the first mergeable one
        // (note partition_point points to the first elem where the predicate is FALSE assuming the 
        // vec already starts with all Trues and ends with all Falses)
        first_mergeable_zid_of_zid.push(zpaths[i..].partition_point(|p| p.path.starts_with(&zpath.path)) + i);
    }

    let tstart = std::time::Instant::now();

    // populate zids_of_node, nodes_of_zid, and appzipper_of_node_zid
    for (treenode,appzippers) in all_appzippers {
        for appzipper in appzippers {
            if let Ok(i) = zpaths.binary_search(&appzipper.zipper.zpath) {
                zids_of_node[usize::from(treenode)].push(i);
                nodes_of_zid[i].push(treenode);
                appzipper_of_node_zid.insert((treenode,i),appzipper.clone());
            } else { unreachable!() }
        }
    }

    println!("binary search to set up data structs: {:?}ms", tstart.elapsed().as_millis());

    let tstart = std::time::Instant::now();

    // arity 0 inventions
    for node in treenodes.iter() {
        if *node == programs_node { continue; }
        if !egraph[*node].data.free_vars.is_empty() { continue; }
        if tasks_of_node[&node].len() < 2 { continue; }
        // Note that "single use" pruning is intentionally not done here,
        // since any invention specific to a node will by definition only
        // be useful at that node
        
        let ztuple = ZTuple::empty();
        let nodes = vec![*node];
        let body_utility = egraph[*node].data.inventionless_cost;
        let compressive_utility = compressive_utility(body_utility, &ztuple, &nodes, &num_paths_to_node, &egraph, &appzipper_of_node_zid);
        let utility = compressive_utility + other_utility(body_utility, cfg);
        if utility <= 0 { continue; }

        donelist.push(FinishedItem::new(ztuple, nodes, utility, compressive_utility));
    }
    println!("got {} arity zero inventions ({:?}ms)", donelist.len(), tstart.elapsed().as_millis());

    let mut lowest_donelist_utility = 0;
    let mut utility_pruning_cutoff = 0;

    // sort and truncate
    update_donelist(&mut donelist, &cfg, &mut lowest_donelist_utility, &mut utility_pruning_cutoff);

    let mut stats: Stats = Default::default();


    let tstart = std::time::Instant::now();

    // **********************
    // * INITIAL INVENTIONS *
    // **********************
    // (these are the single-arg single-use inventions)
    initial_inventions(
        &appzipper_of_node_zid,
        &nodes_of_zid,
        &mut worklist,
        &mut donelist,
        &egraph,
        &mut lowest_donelist_utility,
        &mut utility_pruning_cutoff,
        &num_paths_to_node,
        &mut stats,
        cfg,
        &tasks_of_node,
    );

    println!("initial_inventions(): {:?}ms", tstart.elapsed().as_millis());
    println!("initial worklist length: {}", worklist.len());
    if let Some(size) = worklist.iter().map(|ztg| ztg.nodes.len()).max() {
        println!("largest ztuple group: {}", size);
    }
    println!("avg ztuple group: {}", worklist.iter().map(|ztg| ztg.nodes.len()).sum::<usize>() as f64 / worklist.len() as f64);

    println!("total prep: {:?}ms", tstart_total.elapsed().as_millis());

    println!("deriving inventions...");
    let tstart = std::time::Instant::now();

    // worklist.make_contiguous().shuffle(&mut rand::thread_rng()); // shuffle
    if cfg.ascending_worklist {
        worklist.make_contiguous().sort(); // ascending sort order
    } else {
        worklist.make_contiguous().sort_by(|a, b| b.cmp(a)); // reverse sort order
    }

    // At this point we transition to multithreading. We don't do this earlier because it'd require messy mutexes during single threaded code.
    // So now here we shadow over all our variables to make atomically refcounted pointers to them (`Arc`), along with `Mutex`s for things
    // that aren't read-only. 
    let shared = Arc::new(Mutex::new(KMutableMultithreadData { donelist, worklist, lowest_donelist_utility, utility_pruning_cutoff}));
    let stats =                      Arc::new(Mutex::new(stats));
    let appzipper_of_node_zid =      Arc::new(appzipper_of_node_zid);
    let zids_of_node =               Arc::new(zids_of_node);
    let first_mergeable_zid_of_zid = Arc::new(first_mergeable_zid_of_zid);
    let egraph =                     Arc::new(egraph);
    let num_paths_to_node =          Arc::new(num_paths_to_node);
    // unfortunately since we dont own `cfg` we can't `move` it into an Arc so we need to duplicate it here - no worries it's lightweight
    let cfg: Arc<CompressionStepConfig> = Arc::new(cfg.clone()); 
    let tasks_of_node = Arc::new(tasks_of_node);
    
    // *********************
    // * DERIVE INVENTIONS *
    // *********************
    // (this is finding all the higher-arity multi-use inventions through stitching)
    if cfg.threads == 1 {
        // Single threaded
        derive_inventions(
            Arc::clone(&shared),
            Arc::clone(&appzipper_of_node_zid),
            Arc::clone(&zids_of_node),
            Arc::clone(&first_mergeable_zid_of_zid),
            Arc::clone(&egraph),
            Arc::clone(&num_paths_to_node),
            Arc::clone(&stats),
            Arc::clone(&cfg),
            Arc::clone(&tasks_of_node),
        )
    } else {
        // Multithreaded
        let mut handles = vec![];
        for _ in 0..cfg.threads {
            // clone the Arcs to have copies for this thread
            let shared =                     Arc::clone(&shared);
            let appzipper_of_node_zid =      Arc::clone(&appzipper_of_node_zid);
            let zids_of_node =               Arc::clone(&zids_of_node);
            let first_mergeable_zid_of_zid = Arc::clone(&first_mergeable_zid_of_zid);
            let egraph =                     Arc::clone(&egraph);
            let num_paths_to_node =          Arc::clone(&num_paths_to_node);
            let stats =                      Arc::clone(&stats);
            let cfg =                        Arc::clone(&cfg);
            let tasks_of_node = Arc::clone(&tasks_of_node);
            
            // launch thread to just call derive_inventions()
            handles.push(thread::spawn(move || {
                derive_inventions(
                    shared,
                    appzipper_of_node_zid,
                    zids_of_node,
                    first_mergeable_zid_of_zid,
                    egraph,
                    num_paths_to_node,
                    stats,
                    cfg,
                    tasks_of_node,
                )}));
        }
        // wait for all threads to finish (when all have empty worklists)
        for handle in handles {
            handle.join().unwrap();
        }
    }


    let mut shared_guard = shared.lock();
    let shared: &mut KMutableMultithreadData = shared_guard.deref_mut();

    assert!(shared.worklist.is_empty());
    kupdate_donelist_shared(shared, &cfg);
    println!("{:?}", stats.lock().deref_mut());

    let elapsed_derive_inventions = tstart.elapsed().as_millis();

    println!("\nderive_inventions() done: {:?}ms\n", elapsed_derive_inventions);
    println!("total everything: {:?}ms", tstart_total.elapsed().as_millis());

    

    let orig_cost = egraph[programs_node].data.inventionless_cost;

    let mut results: Vec<CompressionStepResult> = vec![];

    let mut egraph: EGraph = Arc::try_unwrap(egraph).unwrap();

    // construct CompressionStepResults and print some info about them)
    println!("Cost before: {}", orig_cost);
    for (i,done) in shared.donelist.iter().enumerate() {
        let res = CompressionStepResult::new(done.clone(), programs_node, new_inv_name, &appzipper_of_node_zid, &num_paths_to_node, &mut egraph, past_invs);

        println!("{}: {}", i, res);
        if cfg.show_rewritten {
            println!("rewritten:\n{}", res.rewritten.split_programs().iter().map(|p|p.to_string()).collect::<Vec<_>>().join("\n"));
        }
        results.push(res);
        // if args.render_inventions {
        //     inv_expr.save(&format!("fn_{}",i), &out_dir);
        // }
    }


    println!("Final donelist length: {}",shared.donelist.len());
    println!("derive_inventions() took: {}ms ***\n", elapsed_derive_inventions);

    results
}

/// Finds the initial set of single-arg single-use inventions from the appzippers. This updates `donelist` with the
/// discovered inventions and pushes all the partial inventions to `worklist`. No stitching is done at this point, that
/// will all be done during `derive_inventions`. This just gets the worklist in its initial state!
#[inline(never)]
fn initial_inventions(
    appzipper_of_node_zid: &HashMap<(Id,ZId),AppZipper>,
    nodes_of_zid: &Vec<Vec<Id>>,
    worklist: &mut VecDeque<WorklistItem>,
    // worklist: &mut BinaryHeap<HeapItem>,
    donelist: &mut Vec<FinishedItem>,
    egraph: &EGraph,
    lowest_donelist_utility: &mut i32,
    utility_pruning_cutoff: &mut i32,
    num_paths_to_node: &HashMap<Id,i32>,
    stats: &mut Stats,
    cfg: &CompressionStepConfig,
    tasks_of_node: &HashMap<Id, HashSet<usize>>,
) {
    for (zid,nodes) in nodes_of_zid.iter().enumerate() {
        let ztuple = ZTuple::single(zid);

        // 1) Define keys that we will use to index into our zippers
        let left_edge_key = |node: &Id| appzipper_of_node_zid[&(*node,zid)].zipper.left.as_slice();
        let path_key = |node: &Id| appzipper_of_node_zid[&(*node,zid)].zipper.zpath.path.as_slice();
        let right_edge_key = |node: &Id| appzipper_of_node_zid[&(*node,zid)].zipper.right.as_slice();
        let both_edge_key = |node: &Id| (appzipper_of_node_zid[&(*node,zid)].zipper.left.as_slice(),
                                         appzipper_of_node_zid[&(*node,zid)].zipper.right.as_slice());


        // 2) Sort our nodes by their both_edge_key (which also sorts them by their left_edge_key since `left` is a prefix of `both`)
        //    and then group adjacent nodes that are equal in terms of `left` or `both` keys, creating two sets of groups.
        let mut nodes = nodes.clone();
        nodes.sort_unstable_by_key(&both_edge_key);
        let left_groups = group_by_key(nodes.clone(), left_edge_key);
        let both_groups = group_by_key(nodes, both_edge_key);

        // *******************
        // * ADD TO DONELIST *
        // *******************
        for group in both_groups {
            // prune finished inventions that are only useful at one node
            if !cfg.no_opt_single_use && group.len() <= 1 {
                stats.single_use_done_fired += 1;
                continue;
            }
            // prune finished inventions that have free variables in them
            if  edge_has_free_vars(left_edge_key(&group[0]), path_key(&group[0]),  0, &egraph) ||
                edge_has_free_vars(right_edge_key(&group[0]), path_key(&group[0]),  0, &egraph) {
                stats.free_vars_done_fired += 1;
                continue;
            }
            // calculate utility of this single-arg single-use invention
            let left_utility = left_edge_utility(left_edge_key(&group[0]), &egraph);
            let right_utility = right_edge_utility(right_edge_key(&group[0]), &egraph);
            let compressive_utility = compressive_utility(left_utility + right_utility, &ztuple, &group, num_paths_to_node, egraph, appzipper_of_node_zid);
            let utility = compressive_utility + other_utility(left_utility + right_utility, cfg);

            // prune finished inventions specific to one single task
            if !cfg.no_opt_single_task
                    && group.iter().all(|node| tasks_of_node[&node].len() == 1)
                    && group.iter().all(|node| tasks_of_node[&group[0]].iter().next() == tasks_of_node[&node].iter().next()) {
                stats.single_task_done_fired += 1;
                continue;
            }
            // push to donelist if better than worst thing on donelist
            if utility >= 0 && utility > *lowest_donelist_utility {
                donelist.push(FinishedItem::new(ztuple.clone(), group, utility, compressive_utility));
                // if you beat the cutoff, we need to update the cutoff (regardless of whether its --lossy-candidates or not)
                if utility > *utility_pruning_cutoff {
                    update_donelist(donelist, &cfg, lowest_donelist_utility, utility_pruning_cutoff);
                }
            }
            stats.finished_invs += 1;
        }

        // *******************
        // * ADD TO WORKLIST *
        // *******************
        for group in left_groups {
            // prune partial inventions that are only useful at one node
            if !cfg.no_opt_single_use && group.len() <= 1 {
                // println!("rejected bc <= 1: {}", ZTuple::single(zid).to_expr(group[0], &appzipper_of_node_zid, &egraph));
                stats.single_use_wip_fired += 1;
                continue;
            }

            // prune partial inventions specific to one single task
            if !cfg.no_opt_single_task
                    && group.iter().all(|node| tasks_of_node[&node].len() == 1)
                    && group.iter().all(|node| tasks_of_node[&group[0]].iter().next() == tasks_of_node[&node].iter().next()) {
                stats.single_task_done_fired += 1;
                continue;
            }

            // prune partial inentions that contain free variables in their concrete part
            if !cfg.no_opt_free_vars && edge_has_free_vars(left_edge_key(&group[0]), path_key(&group[0]),  0, &egraph) {
                // panic!("hey");
                stats.free_vars_wip_fired += 1;
                continue;
            }
            // println!("passed: {}", ZTuple::single(zid).to_expr(group[0], &appzipper_of_node_zid, &egraph));
            // upper bound the utility of the partial invention
            let left_utility = left_edge_utility(left_edge_key(&group[0]), &egraph);
            let global_right_utility_upper_bound = group.iter().map(|node| num_paths_to_node[node] * right_edge_utility(right_edge_key(node), &egraph)).sum::<i32>();
            let upper_bound = other_utility_upper_bound(left_utility, &cfg) + compressive_utility_upper_bound(left_utility, global_right_utility_upper_bound, &ztuple, &group, num_paths_to_node, egraph, appzipper_of_node_zid);
            // push to worklist if utility upper bound is good enough
            if cfg.no_opt_upper_bound || (upper_bound >= 0 && upper_bound > *utility_pruning_cutoff) {
                worklist.push_back(WorklistItem::new(ztuple.clone(), group, left_utility, upper_bound));
                // worklist.push(HeapItem::new(WorklistItem::new(ZTuple::single(zid), group, left_utility, upper_bound)));
                // worklist.sort_by_key(|wi| -wi.left_utility);
            } else {
                stats.upper_bound_fired += 1;
            }
        }
    }
    update_donelist(donelist, &cfg, lowest_donelist_utility, utility_pruning_cutoff);
}



#[inline(never)]
fn derive_inventions(
    shared: Arc<Mutex<KMutableMultithreadData>>,
    appzipper_of_node_zid: Arc<HashMap<(Id,ZId),AppZipper>>,
    zids_of_node: Arc<Vec<Vec<ZId>>>,
    first_mergeable_zid_of_zid: Arc<Vec<ZId>>,
    egraph: Arc<EGraph>,
    num_paths_to_node: Arc<HashMap<Id,i32>>,
    stats: Arc<Mutex<Stats>>,
    cfg: Arc<CompressionStepConfig>,
    tasks_of_node: Arc<HashMap<Id, HashSet<usize>>>,
) {
    let mut worklist_buf: Vec<WorklistItem> = Default::default();
    let mut donelist_buf: Vec<FinishedItem> = Default::default();

    // let mut till_shuffle = 100;

    loop {

        // * MULTITHREADING: CRITICAL SECTION START *
        let wi = {
            // take the lock, which will be released immediately when this scope exits
            let mut shared_guard = shared.lock();
            let shared: &mut KMutableMultithreadData = shared_guard.deref_mut();
            let lowest_donelist_utility = shared.lowest_donelist_utility;
            let old_donelist_len = shared.donelist.len();
            // drain from donelist_buf into the actual donelist
            shared.donelist.extend(donelist_buf.drain(..).filter(|done| done.utility > lowest_donelist_utility));
            if !cfg.no_stats { stats.lock().deref_mut().finished_invs += shared.donelist.len() - old_donelist_len; };
            // sort + truncate + update utility_pruning_cutoff and lowest_donelist_utility
            kupdate_donelist_shared(shared, &cfg); // this also updates utility_pruning_cutoff
            // pull out utility_pruning_cutoff now that it has been updated (not earlier)
            let utility_pruning_cutoff = shared.utility_pruning_cutoff;

            let old_worklist_len = shared.worklist.len();
            let worklist_buf_len = worklist_buf.len();
            // drain from worklist_buf into the actual worklist
            shared.worklist.extend(worklist_buf.drain(..).filter(|done| done.utility_upper_bound > utility_pruning_cutoff));
            // num pruned by upper bound = num we were gonna add minus change in worklist length
            if !cfg.no_stats { stats.lock().deref_mut().upper_bound_fired += worklist_buf_len - (shared.worklist.len() - old_worklist_len); };

            // loop until we get a new (unpruned) item from the worklist
            let wi = loop {
                let next = if cfg.fifo_worklist { shared.worklist.pop_front() } else { shared.worklist.pop_back() };
                let wi = match next {
                    Some(wi) => wi,
                    None => return, // worklist was empty! we're done!
                };
                // prune if upper bound is too low (cutoff may have increased in the time since this was added to the worklist)
                if cfg.no_opt_upper_bound || wi.utility_upper_bound > shared.utility_pruning_cutoff {
                    break wi
                } else {
                    if !cfg.no_stats { stats.lock().deref_mut().upper_bound_fired += 1; };
                }
            };
            wi
        };
        // * MULTITHREADING: CRITICAL SECTION END *

        if !cfg.no_stats { stats.lock().deref_mut().partial_invs += 1; };

        let rightmost_zid: ZId = wi.ztuple.elems.last().unwrap().zid;
        let first_mergeable_zid: ZId = first_mergeable_zid_of_zid[rightmost_zid];
        let mut possible_elems: Vec<(LabelledZId,Id)> = vec![];

        // collect all the possible LabelledZIds; these essentially correspond to the different zippers (labelled with #i variables) that
        // we could choose to merge in. We collect these from each of the nodes in the group.
        for node in wi.nodes.iter() {
            // skip over the zids that are prefixes - partition point will binarysearch for the first case where the predicate is false.
            // this works nicely since all (unusuable) prefix ones come before all nonprefix ones and first_mergeable_zid tells us the first nonprefix one
            let zids = &zids_of_node[usize::from(*node)];
            let start: usize = zids.partition_point(|zid| *zid < first_mergeable_zid);
            for zid in &zids[start..] {
                // merging rightmost_zid and zid is possible as long as either arity or multiuse check out

                // add any multiarg
                if wi.ztuple.arity < cfg.max_arity {
                    possible_elems.push((LabelledZId::new(*zid, wi.ztuple.arity), *node));
                }
                // add any multiuse
                let arg = appzipper_of_node_zid[&(*node,*zid)].arg;
                for (argi,arg_zid) in wi.ztuple.multiarg.iter().enumerate() {
                    if arg == appzipper_of_node_zid[&(*node, *arg_zid)].arg {
                        possible_elems.push((LabelledZId::new(*zid, argi), *node));
                    }
                }
            }
        }
        
        // sort by zid (and ivar) (and Id though we dont care about that)
        possible_elems.sort(); // sorting is important!
        // Itertools::group_by(key: F)
        for (elem, subset) in &Itertools::group_by(possible_elems.into_iter(), |(elem, _node)| elem.clone()) {
            let mut nodes: Vec<Id> = subset.map(|(_elem, node)| node).collect();

            // if all usage locations of this partial invention take the SAME argument for the new variable, then prune
            // this partial invention because it's strictly better to inline that argument into the body and not abstract it
            if !cfg.no_opt_useless_abstract && nodes.iter().all(|node| appzipper_of_node_zid[&(nodes[0],elem.zid)].arg == appzipper_of_node_zid[&(*node,elem.zid)].arg ) {
                continue;
            }

            let num_nodes = nodes.len();
            // this partial invention is only used in a single place to lets prune it
            if !cfg.no_opt_single_use && num_nodes == 1 {
                if !cfg.no_stats { stats.lock().deref_mut().single_use_wip_fired += 1; };
                continue;
            }
            let is_multiuse = elem.ivar < wi.ztuple.arity; // multiuse means an old index within the old arity range was reused

            // divergence point doesnt depend on the specific node so we'll just use the first one
            let div_idx = divergence_idx(appzipper_of_node_zid[&(nodes[0],rightmost_zid)].zipper.zpath.path.as_slice(),
                                         appzipper_of_node_zid[&(nodes[0],elem.zid)].zipper.zpath.path.as_slice());
            
            let div_depth = appzipper_of_node_zid[&(nodes[0],rightmost_zid)].zipper.zpath.path[..div_idx].iter().filter(|x| **x == ZNode::Body).count() as i32;

            let new_ztuple: ZTuple = wi.ztuple.extend(elem.clone(), div_idx, is_multiuse);

            // define key functions for grabbing all the slices of zipper we care about
            // left_fold_key is the left inner side of the fold which is rightmost_zid.RIGHT (not LEFT)
            let left_fold_key =  |node: &Id| &appzipper_of_node_zid[&(*node,rightmost_zid)].zipper.right[div_idx+1..];
            let left_fold_path_key =  |node: &Id| &appzipper_of_node_zid[&(*node,rightmost_zid)].zipper.zpath.path[div_idx+1..];
            let right_fold_key = |node: &Id| &appzipper_of_node_zid[&(*node,elem.zid)].zipper.left[div_idx+1..];
            let right_fold_path_key = |node: &Id| &appzipper_of_node_zid[&(*node,elem.zid)].zipper.zpath.path[div_idx+1..];

            let fold_key = |node: &Id| (&appzipper_of_node_zid[&(*node,rightmost_zid)].zipper.right[div_idx+1..],
                                        &appzipper_of_node_zid[&(*node,elem.zid)].zipper.left[div_idx+1..]);
            let right_edge_key = |node: &Id| appzipper_of_node_zid[&(*node,elem.zid)].zipper.right.as_slice();
            let right_path_key = |node: &Id| appzipper_of_node_zid[&(*node,elem.zid)].zipper.zpath.path.as_slice();

            let both_edge_key = |node: &Id| (&appzipper_of_node_zid[&(*node,rightmost_zid)].zipper.right[div_idx+1..],
                                             &appzipper_of_node_zid[&(*node,elem.zid)].zipper.left[div_idx+1..],
                                             appzipper_of_node_zid[&(*node,elem.zid)].zipper.right.as_slice());

            // sorting by `both` will also sort by fold_key since the latter is a prefix of the former
            nodes.sort_unstable_by_key(&both_edge_key);

            let fold_groups = group_by_key(nodes.clone(), fold_key);
            let both_groups = group_by_key(nodes, both_edge_key);
            let num_offspring = fold_groups.len();

            // *******************
            // * ADD TO DONELIST *
            // *******************
            for group in both_groups {
                // if groups are singletons or contain free variables, skip them
                if !cfg.no_opt_single_use && group.len() <= 1 {
                    if !cfg.no_stats { stats.lock().deref_mut().single_use_done_fired += 1; };
                    continue;
                }

                // prune partial inventions specific to one single task
                if !cfg.no_opt_single_task
                        && group.iter().all(|node| tasks_of_node[&node].len() == 1)
                        && group.iter().all(|node| tasks_of_node[&group[0]].iter().next() == tasks_of_node[&node].iter().next()) {
                    stats.lock().deref_mut().single_task_done_fired += 1;
                    continue;
                }
                // prune inventions that contain free variables
                if edge_has_free_vars(left_fold_key(&group[0]), left_fold_path_key(&group[0]),  div_depth, &egraph) ||
                    edge_has_free_vars(right_fold_key(&group[0]), right_fold_path_key(&group[0]),  div_depth, &egraph) ||
                    edge_has_free_vars(right_edge_key(&group[0]), right_path_key(&group[0]),  0, &egraph) {
                    if !cfg.no_stats { stats.lock().deref_mut().free_vars_done_fired += 1; };
                    continue;
                }
                // Calculate utility
                // the left side of the fold is a RIGHT-facing edge (since it faces into the fold) hence it's right_edge_utility for the left_fold_key
                let left_utility = wi.left_utility + right_edge_utility(left_fold_key(&group[0]), &*egraph) + left_edge_utility(right_fold_key(&group[0]), &*egraph);
                let right_utility = right_edge_utility(right_edge_key(&group[0]), &*egraph);
                let compressive_utility = compressive_utility(left_utility + right_utility, &new_ztuple, &group, &*num_paths_to_node, &*egraph, &*appzipper_of_node_zid);
                let utility = compressive_utility + other_utility(left_utility + right_utility, &cfg);
                if utility >= 0 {
                    donelist_buf.push(FinishedItem::new(new_ztuple.clone(), group, utility, compressive_utility));
                }
            }
    
            // *******************
            // * ADD TO WORKLIST *
            // *******************
            for group in fold_groups {
                // prune partial inventions that are only useful at one node
                if !cfg.no_opt_single_use && group.len() <= 1 {
                    if !cfg.no_stats { stats.lock().deref_mut().single_use_wip_fired += 1; };
                    continue;
                }

                // prune partial inventions specific to one single task
                if !cfg.no_opt_single_task
                        && group.iter().all(|node| tasks_of_node[&node].len() == 1)
                        && group.iter().all(|node| tasks_of_node[&group[0]].iter().next() == tasks_of_node[&node].iter().next()) {
                    stats.lock().deref_mut().single_task_done_fired += 1;
                    continue;
                }
                // prune partial inventions that contain free variables in their concrete part
                if !cfg.no_opt_free_vars && 
                   (edge_has_free_vars(left_fold_key(&group[0]), left_fold_path_key(&group[0]),  div_depth, &egraph) ||
                    edge_has_free_vars(right_fold_key(&group[0]), right_fold_path_key(&group[0]),  div_depth, &egraph)) {
                        if !cfg.no_stats { stats.lock().deref_mut().free_vars_wip_fired += 1; };
                    continue;
                }

                // Calculate utility
                
                let left_utility = wi.left_utility + right_edge_utility(left_fold_key(&group[0]), &*egraph) + left_edge_utility(right_fold_key(&group[0]), &*egraph);
                let global_right_utility_upper_bound = group.iter().map(|node| num_paths_to_node[node] * right_edge_utility(right_edge_key(node), &*egraph)).sum::<i32>();
                let upper_bound = other_utility_upper_bound(left_utility, &cfg) + compressive_utility_upper_bound(left_utility, global_right_utility_upper_bound, &new_ztuple, &group, &*num_paths_to_node, &*egraph, &*appzipper_of_node_zid);

                if upper_bound >= 0 {
                    worklist_buf.push(WorklistItem::new(new_ztuple.clone(), group, left_utility, upper_bound));
                }

            }

            // a multiuse invention that is present at all the nodes from the original worklist AND
            // has all the same non-leading-edge so it only has one offspring. It is strictly beneficial (or breakeven
            // for single leaf nodes) to accept this multiuse, so we can just Break before looking at any higher zid 
            // merges and instead let this newly pushed multiuse thing be the one that merges with those future things.
            if !cfg.no_opt_force_multiuse && is_multiuse && num_nodes == wi.nodes.len() && num_offspring == 1 {
                if !cfg.no_stats { stats.lock().deref_mut().force_multiuse_fired += 1; };
                break;
            }
        }
    }
}

