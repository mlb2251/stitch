use std::ops::DerefMut;
use std::hash::Hash;
use itertools::Itertools;

use lambdas::*;
use rustc_hash::FxHashMap;

use crate::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum VariableType {
    Metavar,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct TypedLabeledZId {
    pub zid: usize,
    pub ivar: u32,
    vtype: u32,
}

#[derive(Debug, Clone, Default)]
pub struct PatternArgs {
    arg_choices: Vec<TypedLabeledZId>, // a hole gets moved into here when it becomes an abstraction argument, again these are in order of when they were added
    first_zid_of_var: Vec<ZId>, //first_zid_of_ivar[i] gives the index zipper to the ith argument (#i), i.e. this is zipper is also somewhere in arg_choices
}

// impl PartialEq for TypedLabeledZId {
//     fn eq(&self, other: &Self) -> bool {
//         self.zid == other.zid && self.ivar == other.ivar
//     }
// }

// impl Eq for TypedLabeledZId {
//     // Eq is automatically implemented if PartialEq is implemented
// }

// impl Hash for TypedLabeledZId {
//     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//         self.zid.hash(state);
//         self.ivar.hash(state);
//     }
// }

// impl PartialOrd for TypedLabeledZId {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         Some(self.cmp(other))
//     }
// }

// impl Ord for TypedLabeledZId {
//     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
//         self.zid.cmp(&other.zid)
//             .then_with(|| self.ivar.cmp(&other.ivar))
//     }
// }

impl PartialEq for PatternArgs {
    fn eq(&self, other: &Self) -> bool {
        // only compare the arg_choices, since the first_zid_of_ivar is just a mapping to the arg_choices
        self.arg_choices == other.arg_choices
    }
}

impl Eq for PatternArgs {
    // Eq is automatically implemented if PartialEq is implemented
}

impl Hash for PatternArgs {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // only hash the arg_choices, since the first_zid_of_ivar is just a mapping to the arg_choices
        self.arg_choices.hash(state);
    }
}

impl PartialOrd for PatternArgs {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PatternArgs {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // only compare the arg_choices, since the first_zid_of_ivar is just a mapping to the arg_choices
        self.arg_choices.cmp(&other.arg_choices)
    }
}


impl PatternArgs {
    pub fn arity(&self) -> usize {
        self.first_zid_of_var.len()
    }

    pub fn num_ivars(&self) -> usize {
        self.first_zid_of_var.len()
    }

    #[inline]
    pub fn iterate_arguments(&self) -> impl Iterator<Item = &TypedLabeledZId> {
        self.arg_choices.iter()
    }

    #[inline]
    pub fn iterate_one_zid_per_argument(&self) -> impl Iterator<Item = ZId> + '_ {
        self.first_zid_of_var.iter().cloned()
    }
    
    pub fn add_var(&mut self, ivar: usize, zid: ZId, vtype: VariableType) {
        self.arg_choices.push(TypedLabeledZId { zid, ivar: ivar as u32, vtype: 0 as u32 });
        if ivar == self.first_zid_of_var.len() {
            self.first_zid_of_var.push(zid);
            assert!(vtype == VariableType::Metavar, "Only metavars are supported for now");
            // self.type_of_var.push(vtype);
        }
    }

    pub fn use_args(&self, shared: &SharedData, node: &Idx) -> Vec<ZId> {
        self.first_zid_of_var.iter().map(|zid|
            shared.arg_of_zid_node[*zid][node].shifted_id
        ).collect()
    }

    pub fn multiuses(&self) -> Vec<(ZId, Cost)> {
        // returns the zids of the first zipper of each var, which is used to check for multiuse
        self.arg_choices.iter().map(|labelled|labelled.ivar).counts()
            .iter().filter_map(|(ivar,count)| if *count > 1 { Some((self.first_zid_of_var[*ivar as usize], (*count-1) as Cost)) } else { None }).collect()
    }

    pub fn has_free_ivars(&self, shared: &SharedData, loc: &Idx) -> bool {
        //  if there are any free ivars in the arg at this location then we can't apply this invention here so *total* util should be 0
        for i in 0..self.first_zid_of_var.len() {
            let zid = self.first_zid_of_var[i];
            let shifted_arg = shared.arg_of_zid_node[zid][loc].shifted_id;
            if !shared.analyzed_ivars[shifted_arg].is_empty() {
                return true;
            }
        }
        false
    }

    pub fn is_useless_abstract(&self, shared: &SharedData, locs: &[Idx]) ->  bool {
        // Pruning (ARGUMENT CAPTURE): check for useless abstractions (ie ones that take the same arg everywhere). We check for this all the time, not just when adding a new variables,
        // because subsetting of match_locations can turn previously useful abstractions into useless ones. In the paper this is referred to as "argument capture"
        if !shared.cfg.no_opt_useless_abstract {
            // note I believe it'd be save to iterate over first_zid_of_ivar instead
            for argchoice in self.arg_choices.iter() {
                // if its the same arg in every place, and doesnt have any free vars (ie it's safe to inline)
                if locs.iter().map(|loc| shared.arg_of_zid_node[argchoice.zid][loc].shifted_id).all_equal()
                    && shared.analyzed_free_vars[shared.arg_of_zid_node[argchoice.zid][&locs[0]].shifted_id].is_empty()
                {
                    if !shared.cfg.no_stats { shared.stats.lock().deref_mut().useless_abstract_fired += 1; };
                    return true;
                }
            }
        }
        false
    }

    pub fn is_redundant_argument(&self, shared: &SharedData, locs: &[Idx]) -> bool {
        // PRUNING (REDUNDANT ARGUMENT) if two different ivars #i and #j have the same arg at every location, then we can prune this pattern
        // because there must exist another pattern where theyre just both the same ivar. Note that this pruning
        // happens here and not just at the ivar creation point because new subsetting can happen. In this paper this is referred to as
        // "redundant argument elimination".
        if !shared.cfg.no_opt_force_multiuse {
            // for all pairs of ivars #i and #j, get the first zipper and compare the arg value across all locations
            for (i, ivar_zid_1) in self.first_zid_of_var.iter().enumerate() {
                let arg_of_loc_1 = &shared.arg_of_zid_node[*ivar_zid_1];
                for ivar_zid_2 in self.first_zid_of_var.iter().skip(i+1) {
                    let arg_of_loc_2 = &shared.arg_of_zid_node[*ivar_zid_2];
                    if locs.iter().all(|loc|
                        arg_of_loc_1[loc].shifted_id == arg_of_loc_2[loc].shifted_id)
                    {
                        if !shared.cfg.no_stats { shared.stats.lock().deref_mut().force_multiuse_fired += 1; };
                        return true;
                    }
                }
            }
        }
        false
    }
    pub fn reusable_args_location(&self, shared: &SharedData, ivar: Idx, arg_of_loc: &FxHashMap<Idx, Arg>, match_locations: &[Idx]) -> Vec<Idx> {
        let arg_of_loc_ivar = &shared.arg_of_zid_node[self.first_zid_of_var[ivar]];
        match_locations.iter()
            .filter(|loc:&&Idx|
                arg_of_loc[loc].shifted_id == 
                arg_of_loc_ivar[loc].shifted_id
                && !invalid_metavar_location(shared, arg_of_loc[loc].shifted_id)
            ).cloned().collect()
    }

    pub fn find_variable(&self, shared: &SharedData, i: Idx) -> Idx {
        // in the case where we're searching for an IVar we need to be robust to relabellings
        // since this doesn't have to be canonical. What we can do is we can look over
        // each ivar the the pattern has defined with a first zid in pattern.first_zid_of_ivar, and
        // if our expressions' zids_of_ivar[i] contains this zid then we know these two ivars
        // must correspond to each other in the pattern and the tracked expr and we can just return
        // the pattern version (`j` below).
        let zids = shared.tracking.as_ref().unwrap().zids_of_ivar[i].clone();
        for (j,zid) in self.first_zid_of_var.iter().enumerate() {
            if zids.contains(zid) {
                return j
            }
        }
        // it's a new ivar that hasnt been used already so it must take on the next largest var number
        self.first_zid_of_var.len()
    }

}
