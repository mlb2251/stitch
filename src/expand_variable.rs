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
    pattern.first_zid_of_ivar.len()
}

pub fn get_zid_for_ivar(pattern: &Pattern, ivar: usize) -> ZId {
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

    let body_utility = pattern.body_utility +  compute_body_utility_change(shared, &expands_to) * variable_zids.len() as i32;
    pattern.body_utility = body_utility;

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
            ExpandsTo::IVar(i) => {
                add_variable_at(&mut pattern, variable_zid, i);
            }
            _ => {}
        }
    }

    Some (pattern)
}
