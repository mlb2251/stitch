use crate::*;

pub fn add_variable_at(p: &mut Pattern, at_loc: usize, var_id: i32) {
    p.arg_choices.push(LabelledZId::new(at_loc, var_id as usize));
    if var_id as usize ==  p.first_zid_of_ivar.len() {
        p.first_zid_of_ivar.push(at_loc);
    }
}

pub fn remove_variable_at(p: &mut Pattern, var_id: usize, expands_to: &mut ExpandsTo) -> Vec<ZId> {
    let mut zids = Vec::new();
    // remove the variable from the arg choices
    p.arg_choices.retain(|x: &LabelledZId| {
        if (x.ivar as usize) == var_id {
            // if this is the variable we're removing, add its zid to the list of zids to remove
            // and return false to remove it from the arg choices
            zids.push(x.zid);
            return false;
        }
        return true; // keep this arg choice
    });
    p.arg_choices.iter_mut().for_each(|x: &mut LabelledZId| {
        if x.ivar > var_id {
            x.ivar -= 1; // decrement the ivar index for all variables after the one we're removing
        }
    });
    if let ExpandsTo::IVar(i) = expands_to {
        assert!(*i != var_id as i32, "ExpandsTo::IVar should not be the variable we're removing");
        if *i > var_id as i32 {
            *i -= 1;
        }
    }
    // remove the variable from the first_zid_of_ivar
    p.first_zid_of_ivar.remove(var_id);
    zids
}

pub fn get_num_variables(pattern: &Pattern) -> usize {
    // the number of variables is the number of first_zid_of_ivar entries
    pattern.first_zid_of_ivar.len()
}

pub fn get_zid_for_ivar(pattern: &Pattern, ivar: usize) -> ZId {
    // get the first zid of the ivar
    pattern.first_zid_of_ivar[ivar]
}


pub fn check_consistency(shared: &SharedData, p: &Pattern) {
    // let num_vars: usize = get_num_variables(p);
    for labeled in p.arg_choices.iter() {
        // check that the ivar is within bounds
        let arg_of_loc = &shared.arg_of_zid_node[labeled.zid];
        for loc in p.match_locations.iter() {
            assert!(arg_of_loc.contains_key(loc), "Variable id={}, zid={} at location {} is not consistent with shared data", labeled.ivar, labeled.zid, loc);
        }
    }
    for (ivar, zid) in p.first_zid_of_ivar.iter().enumerate() {
        for labeled in p.arg_choices.iter() {
            if labeled.ivar == ivar {
                // check that they expand to the same thing
                let arg_of_loc_1 = &shared.arg_of_zid_node[labeled.zid];
                let arg_of_loc_2 = &shared.arg_of_zid_node[*zid];
                // println!("Checking consistency for variable id={} (zid={} vs zid={})", ivar, zid, labeled.zid);
                for loc in p.match_locations.iter() {
                    assert!(arg_of_loc_1.contains_key(loc) && arg_of_loc_2.contains_key(loc), 
                        "Variable id={} at location {} is not consistent with shared data: {:?} vs {:?}", ivar, loc, arg_of_loc_1.get(loc), arg_of_loc_2.get(loc));
                    assert_eq!(arg_of_loc_1[loc].shifted_id, arg_of_loc_2[loc].shifted_id,
                        "Variable id={} at location {} has different shifted ids: {} vs {}", ivar, loc, arg_of_loc_1[loc].shifted_id, arg_of_loc_2[loc].shifted_id);
                    assert_eq!(arg_of_loc_1[loc].expands_to, arg_of_loc_2[loc].expands_to,
                        "Variable id={} at location {} expands to different things: {} vs {}", ivar, loc, arg_of_loc_1[loc].expands_to, arg_of_loc_2[loc].expands_to);
                }
            }
        }
    }
    // for i in 0..num_vars {
    //     let variable_zid = get_zid_for_ivar(p, i);
    //     let arg_of_loc = &shared.arg_of_zid_node[variable_zid];
    //     for loc in p.match_locations.iter() {
    //         assert!(arg_of_loc.contains_key(loc), "Variable {} at location {} is not consistent with shared data", i, loc);
    //     }
    // }
}

pub fn perform_expansion_variable(
    pattern: Pattern,
    shared: &SharedData,
    variable_ivar: usize,
    expands_to: ExpandsTo,
    // locs: Vec<Idx>,
) -> Option<Pattern> {
    // for debugging
    // TODO double check
    // let tracked = original_pattern.tracked && expands_to == tracked_expands_to(&original_pattern, variable_zid, &shared);
    // if tracked { found_tracked = true; }
    // if shared.cfg.follow_prune && !tracked { return None; }
    let mut pattern = pattern;
    let mut expands_to = expands_to;


    // check_consistency(shared, &pattern);
    // println!("expands_to: {:?}", expands_to);
    // println!("pattern: {:?}", pattern);
    // update the body utility
    let body_utility = pattern.body_utility +  compute_body_utility_change(shared, &expands_to);

    // assert!(shared.cfg.no_opt_upper_bound || !holes_after_pop.is_empty() || !original_pattern.arg_choices.is_empty() || expands_to.has_holes() || expands_to.is_ivar(),
            // "unexpected arity 0 invention: upper bounds + priming with arity 0 inventions should have prevented this");
    // assert!(shared.cfg.no_opt_upper_bound || (locs.len() > 1 || !shared.egraph[locs[0]].data.free_vars.is_empty()),
    //         "single-use pruning doesn't seem to be happening, it should be an automatic side effect of upper bounds + priming with arity zero inventions (as long as they dont have free vars)\n{}\n{}\n{}\n{}\n{}", original_pattern.to_expr(&shared), extract(locs[0], &shared.egraph), expands_to,  util_upper_bound, weak_utility_pruning_cutoff);

    // build our new pattern with all the variables we've just defined. Copy in the argchoices and prefixes
    // from the old pattern.
    // new_pattern.match_locations = locs;
    pattern.body_utility = body_utility;

    // println!("targeting ivar: {}", variable_ivar);
    // println!("expands_to: {:?}", expands_to);
    // println!("original: {:?}", new_pattern);
    let variable_zids: Vec<usize> = remove_variable_at(&mut pattern, variable_ivar, &mut expands_to);

    // println!("pattern after remove: {:?}", pattern);
    // println!("expands_to after remove: {:?}", expands_to);
    // println!("variable_zids: {:?}", variable_zids);
    // println!("after removing variable: {:?}", new_pattern);
    // println!("variable_zids: {:?}", variable_zids);

    let num_vars = pattern.first_zid_of_ivar.len() as i32;

    for variable_zid in variable_zids {
        // add any new holes to the list of holes
        match expands_to {
            ExpandsTo::Lam(_) => {
                // add new holes
                add_variable_at(&mut pattern, shared.extensions_of_zid[variable_zid].body.unwrap(), num_vars); 
            }
            ExpandsTo::App => {
                // add new holes
                add_variable_at(&mut pattern, shared.extensions_of_zid[variable_zid].func.unwrap(), num_vars);
                add_variable_at(&mut pattern, shared.extensions_of_zid[variable_zid].arg.unwrap(), num_vars + 1);
            }
            _ => {}
        }

        if let ExpandsTo::IVar(i) = expands_to {
            add_variable_at(&mut pattern, variable_zid, i);
        }
    }
    // println!("pattern after add: {:?}", pattern);

    // println!("after adding variable: {:?}", new_pattern);

    // check_consistency(shared, &pattern);


    Some (pattern)
}
