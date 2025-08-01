use crate::*;
use lambdas::*;


pub fn min_cost(programs: &[ExprOwned], weights: &Option<Vec<f32>>, tasks: &Option<Vec<String>>, cost_fn: &ExprCost) -> Cost {
    let weights = weights.clone().unwrap_or(vec![1.0; programs.len()]);
    if let Some(tasks) = tasks {
        let mut unique_tasks = tasks.to_vec();
        unique_tasks.sort();
        unique_tasks.dedup();
        unique_tasks.iter().map(|task|
            tasks.iter().zip(programs.iter().zip(weights.iter())).filter_map(|(t,(p,w))| if task == t { Some((p.cost(cost_fn) as f32 * w).round() as Cost) } else { None })
            .min().unwrap()
        ).sum::<Cost>()
    } else {
        programs.iter().zip(weights.iter()).map(|(e,w)| (e.cost(cost_fn) as f32 * w).round() as Cost).sum::<Cost>()
    }
}

/// print some info about a Vec of programs
pub fn programs_info(programs: &[ExprOwned], cost_fn: &ExprCost) {
    let max_cost = programs.iter().map(|e| e.cost(cost_fn)).max().unwrap();
    let max_depth = programs.iter().map(|e| e.depth()).max().unwrap();
    println!("Programs:");
    println!("\t num: {}",programs.len());
    println!("\t max cost: {max_cost}");
    println!("\t max depth: {max_depth}"); 
}

/// provides a timestamp as a string in a format you can use for file/folder names: YYYY-MM-DD_HH-MM-SS
pub fn timestamp() -> String {
    format!("{}", chrono::Local::now().format("%Y-%m-%d_%H-%M-%S"))
}


pub fn compression_factor(original: Cost, compressed: Cost) -> f64 {
    original as f64 / compressed as f64
}

/// Replace the ivars in an expr with vars
pub fn ivar_to_dc(e: &mut ExprMut, depth: i32, arity: i32) {
    match e.node().clone() {
        // dreamcoder doesn't care about the tags, so just use -1 for all of them
        Node::IVar(i) => *e.node() = Node::Var(depth + (arity - 1 - i), -1), // the higher the ivar the smaller the var
        Node::Var(_, _) => {},
        Node::Prim(_) => {},
        Node::App(f,x) => {
            ivar_to_dc(&mut e.get(f), depth, arity);
            ivar_to_dc(&mut e.get(x), depth, arity);
        },
        Node::Lam(b, _) => {
            ivar_to_dc(&mut e.get(b), depth+1, arity);
        },
    }
}

pub fn dc_inv_str(inv: &Invention, dreamcoder_translations: &[(String, String)]) -> String {
    let mut body = inv.body.clone();
    assert!(body.set.struct_hash.is_none()); // we dont attempt to maintain the struct hash
    ivar_to_dc(&mut body.as_mut(), 0, inv.arity as i32);

    // wrap in lambdas for dremacoder
    for _ in 0..inv.arity {
        // dreamcoder doesn't care about the tags, so just use -1 for all of them
        body.idx = body.set.add(Node::Lam(body.idx, -1));
    }
    // add the "#" that dreamcoder wants and change lam -> lambda
    let mut res: String = format!("#{body}");
    res = res.replace("(lam ", "(lambda ").replace("(lam_", "(lambda_");
    // inline any past inventions using their dc_inv_str. Match on "fn_i)" and "fn_i " to avoid matching fn_1 on fn_10 or any other prefix
    for (inv_name, dc_translation) in dreamcoder_translations.iter() {
        res = replace_prim_with(&res, inv_name, dc_translation);
        // res = res.replace(&format!("{})",past_step_result.inv.name), &format!("{})",past_step_result.dc_inv_str));
        // res = res.replace(&format!("{} ",past_step_result.inv.name), &format!("{} ", past_step_result.dc_inv_str));
    }
    res
}

pub fn replace_prim_with(s: &str, prim: &str, new: &str) -> String {
    let mut res: String = s.to_string();
    res = res.replace(&format!(" {prim})"), &format!(" {new})"));
    // we need to do the " {} " case twice to handle multioverlaps like fn_i fn_i fn_i fn_i which will replace at locations 1 and 3
    // in the first replace() and 2 and 4 in the second replace due to overlapping matches.
    res = res.replace(&format!(" {prim} "), &format!(" {new} "));
    res = res.replace(&format!(" {prim} "), &format!(" {new} "));
    assert!(!res.contains(&format!(" {prim} ")));
    res = res.replace(&format!("({prim} "), &format!("({new} "));
    if res.starts_with(&format!("{prim} ")) {
        res = format!("{} {}", new, &res[prim.len()..]);
    }
    if res.ends_with(&format!(" {prim}")) {
        res = format!("{} {}", &res[..res.len()-prim.len()], new);
    }
    if res == prim {
        res = new.to_string();
    }
    res
}

/// Returns a vec from node Idx to number of places that node is used in the tree. Essentially this just
/// follows all paths down from the root and logs how many times it encounters each node
pub fn num_paths_to_node(roots: &[Idx], corpus_span: &Span, set: &ExprSet) -> (Vec<Cost>, Vec<Vec<Cost>>) {
    let mut num_paths_to_node_by_root_idx: Vec<Vec<Cost>> = vec![vec![0; corpus_span.len()]; roots.len()];

    fn helper(num_paths_to_node: &mut Vec<Cost>, idx: Idx, set: &ExprSet) {
        // num_paths_to_node.insert(*child, num_paths_to_node[node] + 1);
        num_paths_to_node[idx] += 1;
        for child in set.get(idx).children() {
            helper(num_paths_to_node, child, set);
        }
    }

    let mut num_paths_to_node_all: Vec<Cost> = vec![0; corpus_span.len()];
    num_paths_to_node_by_root_idx.iter_mut().enumerate().for_each(|(i,num_paths_to_node)| {
        helper(num_paths_to_node, roots[i], set);
        for i in corpus_span.clone() {
            num_paths_to_node_all[i] += num_paths_to_node[i];
        }
    });
    
    (num_paths_to_node_all, num_paths_to_node_by_root_idx)
}


pub fn zipper_replace(mut expr: ExprOwned, zipper: &[ZNode], new: Node) -> ExprOwned {
    let idx = expr.immut().zip(zipper).idx;
    *expr.as_mut().get_node_mut(idx) = new;
    expr
}

pub fn compute_parents_of(set: &ExprSet) -> Vec<Vec<Idx>> {
    let mut parent_of_node = vec![vec![]; set.len()];
    for (idx, node) in set.nodes.iter().enumerate() {
        match node {
            Node::App(f, x) => {
                parent_of_node[*f].push(idx);
                parent_of_node[*x].push(idx);
            },
            Node::Lam(b, _) => {
                parent_of_node[*b].push(idx);
            },
            Node::Prim(_) | Node::Var(_, _) | Node::IVar(_) => {
                // these have no children
            },
        }
    }
    parent_of_node
}