use crate::*;
use lambdas::*;
use compression::*;

/// a rule for determining when to shift and by how much.
/// if anything points above the `depth_cutoff` (absolute depth
/// of lambdas from the top of the program) it gets shifted by `shift`. If
/// more than one ShiftRule applies then more then one shift will happen.
struct ShiftRule {
    depth_cutoff: i32, 
    shift: i32,
}

pub fn rewrite_fast(
    pattern: &FinishedPattern,
    shared: &SharedData,
    inv_name: &Node,
    cost_fn: &ExprCost
) -> Vec<ExprOwned>
{
    //  if !shared.cfg.silent { println!("rewriting with {}", pattern.info(&shared)) }
    #[allow(clippy::too_many_arguments)]
    fn helper(
        owned_set: &mut ExprSet,
        pattern: &FinishedPattern,
        shared: &SharedData,
        unshifted_id: Idx,
        total_depth: i32, // depth from the very root of the program down
        shift_rules: &mut Vec<ShiftRule>,
        inv_name: &Node,
        refinements: Option<(&Vec<Idx>,i32)>
    ) -> Idx
    {
        // we search using the the *unshifted* one since its an original program tree node
        if pattern.pattern.match_locations.binary_search(&unshifted_id).is_ok() // if the pattern matches here
           && (!pattern.util_calc.corrected_utils.contains_key(&unshifted_id) // and either we have no conflict (ie corrected_utils doesnt have an entry)
             || pattern.util_calc.corrected_utils[&unshifted_id]) // or we have a conflict but we choose to accept it (which is contextless in this top down approach so its the right move)
           && refinements.is_none() // AND we can't currently be in a refinement where rewriting is forbidden
        //    && !pattern.pattern.first_zid_of_ivar.iter().any(|zid| // and there are no negative vars anywhere in the arguments
        //         shared.egraph[shared.arg_of_zid_node[*zid][&unshifted_id].Idx].data.free_vars.iter().any(|var| *var < 0))
        {
            //  if !shared.cfg.silent { println!("inv applies at unshifted={} with shift={}", extract(unshifted_id,&shared.egraph), shift) }
            let mut expr = owned_set.add(inv_name.clone());
            // wrap the prim in all the Apps to args
            for (_ivar,zid) in pattern.pattern.first_zid_of_ivar.iter().enumerate() {
                let arg: &Arg = &shared.arg_of_zid_node[*zid][&unshifted_id];

                if arg.shift != 0 {
                    shift_rules.push(ShiftRule{depth_cutoff: total_depth, shift: arg.shift});
                }
                let rewritten_arg = helper(owned_set, pattern, shared, arg.unshifted_id, total_depth, shift_rules, inv_name, None);
                if arg.shift != 0 {
                    shift_rules.pop(); // pop the rule back off after
                }
                expr = owned_set.add(Node::App(expr, rewritten_arg));
            }
            return expr
        }
        //  if !shared.cfg.silent { println!("descending: {}", extract(unshifted_id,&shared.egraph)) }

        if let Some((refinements,arg_depth)) = refinements.as_ref() {
            if let Some(idx) = refinements.iter().position(|r| *r == unshifted_id) {
                //  if !shared.cfg.silent { println!("found refinement!!!") }
                // todo should this be `idx` or `refinements.len()-1-idx`?
                return owned_set.add(Node::Var(total_depth - arg_depth + idx as i32)); // if we didnt pass thru any lams on the way this would just be $0 and thus refer to the ExprOwned::lam() wrapping our helper() call
            }
        }


        match &shared.set[unshifted_id] {
            Node::Prim(p) => owned_set.add(Node::Prim(p.clone())),
            Node::Var(i) => {
                let mut j = *i;
                for rule in shift_rules.iter() {
                    // take "i" steps upward from current depth and see if you meet or pass the cutoff.
                    // exactly equaling the cutoff counts as needing shifting.
                    if total_depth - i <= rule.depth_cutoff {
                        j += rule.shift;
                    }
                }
                if let Some((refinements,arg_depth)) = refinements.as_ref() {
                    // we're inside the *shifted arg* of a refinement so this var has already been shifted a bit btw
                    // tho thats kinda irrelevant right here
                    if j >= *arg_depth {
                        // j is pointing above the invention so we need to upshift it a bit to account for the new lambdas we added
                        j += refinements.len() as i32;
                    }
                }
                assert!(j >= 0, "{}", pattern.to_expr(shared));
                owned_set.add(Node::Var(j))
            }, // we extract from the *shifted* one since thats the real one
            Node::App(unshifted_f,unshifted_x) => {
                let f = helper(owned_set, pattern, shared, *unshifted_f, total_depth, shift_rules, inv_name, refinements);
                let x = helper(owned_set, pattern, shared, *unshifted_x, total_depth, shift_rules, inv_name, refinements);
                owned_set.add(Node::App(f,x))
            },
            Node::Lam(unshifted_b) => {
                let b = helper(owned_set, pattern, shared, *unshifted_b, total_depth + 1, shift_rules, inv_name, refinements);
                owned_set.add(Node::Lam(b))
            },
            Node::IVar(_) => {
                unreachable!()
            },
        }
    }

    let shift_rules = &mut vec![];
    let rewritten_exprs: Vec<ExprOwned> = shared.roots.iter().map(|root| {
        let mut owned_set = ExprSet::empty(Order::ChildFirst, false, false); // need struct hash off for cost_span later
        let idx = helper(&mut owned_set, pattern, shared, *root, 0, shift_rules, inv_name, None);
        ExprOwned { set: owned_set, idx }
    }).collect();

    if !shared.cfg.no_mismatch_check && !shared.cfg.utility_by_rewrite {
        assert_eq!(
            shared.root_idxs_of_task.iter().map(|root_idxs|
                root_idxs.iter().map(|idx| rewritten_exprs[*idx].cost(cost_fn)).min().unwrap()
            ).sum::<i32>(),
            shared.init_cost - pattern.util_calc.util,
            "\n{}\n", pattern.info(shared)
        );
    }

    rewritten_exprs
}

// /// Rewrite with the given abstractions
pub fn rewrite_with_inventions(
    programs: &[ExprOwned],
    invs: &[Invention],
    cfg: &CompressionStepConfig,
) -> Vec<ExprOwned> {
    if invs.is_empty() {
        return programs.to_vec()
    }
    // programs.to_vec()
    let follow = Some(invs.to_vec());
    let step_results = compression(programs, invs.len(), cfg, None, &[], follow);

    // return the last one - note that if an abstraction wasn't used anywhere it will not be included in the step_results so this
    // may be shorter than invs.len(), however we do ensure that we continue searching for the rest of the abstractions if this happens
    // anyways.
    step_results.last().map(|res|res.rewritten.clone()).unwrap_or_else(||programs.to_vec())
}