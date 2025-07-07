use std::ops::DerefMut;
use itertools::Itertools;

use lambdas::*;
use rustc_hash::FxHashMap;

use crate::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy, PartialOrd, Ord)]
pub enum VariableType {
    Metavar,
    Symvar,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct PatternArgs {
    arg_choices: Vec<LabelledZId>, // a hole gets moved into here when it becomes an abstraction argument, again these are in order of when they were added
    variables: Vec<(u32, VariableType)>, //first_zid_of_ivar[i] gives the index zipper to the ith argument (#i), i.e. this is zipper is also somewhere in arg_choices
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
        self.variables.len()
    }

    pub fn num_ivars(&self) -> usize {
        self.variables.iter().filter(|(_,t)| *t == VariableType::Metavar).count()
    }

    #[inline]
    pub fn iterate_arguments(&self) -> impl Iterator<Item = &LabelledZId> {
        self.arg_choices.iter()
    }

    #[inline]
    pub fn iterate_one_zid_per_argument(&self) -> impl Iterator<Item = ZId> + '_ {
        self.variables.iter().map(|(zid, _)| *zid as ZId)
    }
    
    pub fn add_var(&mut self, ivar: usize, zid: ZId, vtype: VariableType) {
        self.arg_choices.push(LabelledZId::new(zid, ivar));
        if ivar == self.variables.len() {
            self.variables.push((zid as u32, vtype));
        }
    }

    pub fn use_args(&self, shared: &SharedData, node: &Idx) -> Vec<ZId> {
        self.variables.iter().map(|(zid, _)|
            shared.arg_of_zid_node[*zid as usize][node].shifted_id
        ).collect()
    }

    pub fn multiuses(&self) -> Vec<(ZId, Cost)> {
        // returns the zids of the first zipper of each var, which is used to check for multiuse
        self.arg_choices.iter().map(|labelled|labelled.ivar).counts()
            .iter().filter_map(|(ivar,count)| if *count > 1 { Some((self.variables[*ivar].0 as ZId, (*count-1) as Cost)) } else { None }).collect()
    }

    pub fn has_free_ivars(&self, shared: &SharedData, loc: &Idx) -> bool {
        //  if there are any free ivars in the arg at this location then we can't apply this invention here so *total* util should be 0
        for i in 0..self.variables.len() {
            if self.variables[i].1 != VariableType::Metavar {
                continue;
            }
            let zid = self.variables[i].0 as ZId;
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
                if self.variables[argchoice.ivar].1 != VariableType::Metavar {
                    continue;
                }
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
            for (i,(ivar_zid_1, type_1)) in self.variables.iter().enumerate() {
                if *type_1 != VariableType::Metavar {
                    continue;
                }
                let arg_of_loc_1 = &shared.arg_of_zid_node[*ivar_zid_1 as ZId];
                for (j, (ivar_zid_2, type_2)) in self.variables.iter().enumerate().skip(i+1) {
                    if *type_2 != VariableType::Metavar {
                        continue;
                    }
                    let arg_of_loc_2 = &shared.arg_of_zid_node[*ivar_zid_2 as ZId];
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

    pub fn find_variable(&self, shared: &SharedData, i: Idx) -> Idx {
        // in the case where we're searching for an IVar we need to be robust to relabellings
        // since this doesn't have to be canonical. What we can do is we can look over
        // each ivar the the pattern has defined with a first zid in pattern.first_zid_of_ivar, and
        // if our expressions' zids_of_ivar[i] contains this zid then we know these two ivars
        // must correspond to each other in the pattern and the tracked expr and we can just return
        // the pattern version (`j` below).
        let zids = shared.tracking.as_ref().unwrap().zids_of_ivar[i].clone();
        for (j,(zid, _)) in self.variables.iter().enumerate() {
            if zids.contains(&(*zid as ZId)) {
                return j
            }
        }
        // it's a new ivar that hasnt been used already so it must take on the next largest var number
        self.variables.len()
    }

}

pub struct LocationsForReusableArgs<'a> {
    all_locs: &'a Vec<Idx>,
    // lazily computed on an as-needed basis
    sym_locs: Option<Vec<Idx>>,
}

impl LocationsForReusableArgs<'_> {
    pub fn new(all_locs: &Vec<Idx>) -> LocationsForReusableArgs {
        LocationsForReusableArgs {
            all_locs,
            sym_locs: None,
        }
    }

    fn sym_locs<'a>(&'a mut self, arg_of_loc: &FxHashMap<Idx, Arg>, sym_var_info: &SymvarInfo) -> &'a Vec<Idx> {
        if self.sym_locs.is_some() {
            return self.sym_locs.as_ref().unwrap();
        }
        let locs: Vec<_> = self.all_locs.iter()
            .filter(|loc| arg_of_loc[loc].expands_to.is_prim_symbol(sym_var_info))
            .cloned().collect();
        self.sym_locs = Some(locs.clone());
        self.sym_locs.as_mut().unwrap()
    }

    fn relevant_locs<'a>(&'a mut self, var_type: VariableType, arg_of_loc: &FxHashMap<Idx, Arg>, sym_var_info: &Option<SymvarInfo>) -> &'a Vec<Idx> {
        match var_type {
            // should  be safe because this only happens if there's a symvar
            VariableType::Symvar => self.sym_locs(arg_of_loc, sym_var_info.as_ref().unwrap()),
            VariableType::Metavar => self.all_locs,
        }
    }
}

impl PatternArgs {
    pub fn reusable_args_location(&self, shared: &SharedData, ivar: Idx, arg_of_loc: &FxHashMap<Idx, Arg>, match_locations: &mut LocationsForReusableArgs) -> Vec<Idx> {
        let (first_zid_of_var, type_of_var) = self.variables[ivar];
        let arg_of_loc_ivar = &shared.arg_of_zid_node[first_zid_of_var as ZId];
        let require_valid = match type_of_var {
            VariableType::Metavar => true,
            VariableType::Symvar => false,
        };
        match_locations.relevant_locs(type_of_var, arg_of_loc, &shared.sym_var_info).iter()
            .filter(|loc:&&Idx|
                arg_of_loc[loc].shifted_id == 
                arg_of_loc_ivar[loc].shifted_id
                && (!require_valid || !invalid_metavar_location(shared, arg_of_loc[loc].shifted_id))
            ).cloned().collect()
    }
}