use crate::*;

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
