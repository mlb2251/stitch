use crate::abstraction_learning::*;
use crate::abstraction_learning::egraphs::EGraph;
use crate::expr::*;
use rustc_hash::{FxHashMap};
use std::hash::Hash;

/// print some info about a Vec of programs
pub fn programs_info(programs: &[Expr]) {
    let max_cost = programs.iter().map(|p| p.cost()).max().unwrap();
    let max_depth = programs.iter().map(|p| p.depth()).max().unwrap();
    println!("Programs:");
    println!("\t num: {}",programs.len());
    println!("\t max cost: {}",max_cost);
    println!("\t max depth: {}",max_depth);
}

/// provides a timestamp as a string in a format you can use for file/folder names: YYYY-MM-DD_HH-MM-SS
pub fn timestamp() -> String {
    format!("{}", chrono::Local::now().format("%Y-%m-%d_%H-%M-%S"))
}

pub fn save(egraph: &EGraph, name: &str, outdir: &str) 
{
    egraph.dot().to_png(format!("{}/{}.png",outdir,name)).unwrap();
}

pub fn egraph_info(egraph: &EGraph) -> String 
{
    format!("{} nodes, {} classes, {} memo", egraph.total_number_of_nodes(), egraph.number_of_classes(), egraph.total_size())
}


pub fn compression_factor(original: &Expr, compressed: &Expr) -> f64 {
    f64::from(original.cost())/f64::from(compressed.cost())
}

/// Replace the ivars in an expr based on an i32->Expr map
pub fn ivar_replace(e: &Expr, child: Id, map: &FxHashMap<i32, Expr>) -> Expr {
    match e.get(child) {
        Lambda::IVar(i) => map.get(i).unwrap_or(e).clone(),
        Lambda::Var(v) => Expr::var(*v),
        Lambda::Prim(p) => Expr::prim(*p),
        Lambda::App([f,x]) => Expr::app(ivar_replace(e, *f, map), ivar_replace(e, *x, map)),
        Lambda::Lam([b]) => Expr::lam(ivar_replace(e, *b, map)),
        Lambda::Programs(_) => panic!("why would you do this")
    }
}

/// Replace the ivars in an expr based on an i32->Expr map
pub fn ivar_to_dc(e: &Expr, child: Id, depth: i32, arity: i32) -> Expr {
    match e.get(child) {
        Lambda::IVar(i) => Expr::var(depth + (arity - 1 - i)), // the higher the ivar the smaller the var
        Lambda::Var(v) => Expr::var(*v),
        Lambda::Prim(p) => Expr::prim(*p),
        Lambda::App([f,x]) => Expr::app(ivar_to_dc(e, *f, depth, arity), ivar_to_dc(e, *x, depth, arity)),
        Lambda::Lam([b]) => Expr::lam(ivar_to_dc(e, *b, depth+1, arity)),
        Lambda::Programs(_) => panic!("why would you do this")
    }
}

pub fn dc_inv_str(inv: &Invention, dreamcoder_translations: &[(String, String)]) -> String {
    let mut body: Expr = ivar_to_dc(&inv.body, inv.body.root(), 0, inv.arity as i32);
    // wrap in lambdas for dremacoder
    for _ in 0..inv.arity {
        body = Expr::lam(body);
    }
    // add the "#" that dreamcoder wants and change lam -> lambda
    let mut res: String = format!("#{}", body);
    res = res.replace("(lam ", "(lambda ");
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
    res = res.replace(&format!(" {})",prim), &format!(" {})",new));
    // we need to do the " {} " case twice to handle multioverlaps like fn_i fn_i fn_i fn_i which will replace at locations 1 and 3
    // in the first replace() and 2 and 4 in the second replace due to overlapping matches.
    res = res.replace(&format!(" {} ",prim), &format!(" {} ",new));
    res = res.replace(&format!(" {} ",prim), &format!(" {} ",new));
    assert!(!res.contains(&format!(" {} ",prim)));
    res = res.replace(&format!("({} ",prim), &format!("({} ",new));
    if res.starts_with(&format!("{} ",prim)) {
        res = format!("{} {}", new, &res[prim.len()..]);
    }
    if res.ends_with(&format!(" {}",prim)) {
        res = format!("{} {}", &res[..res.len()-prim.len()], new);
    }
    if res == prim {
        res = new.to_string();
    }
    res
}

/// cache for shift()
pub type RecVarModCache = FxHashMap<(Id,i32),Option<Id>>;


/// This is a helper function for implementing various recursive operations that only
/// modify Var or IVar constructs (use `ivars=true` to run this on all ivars). Just provide
/// a function that you want to call on each Var to determine what to replace it with. The function
/// signature should be `(actual_idx, depth, which_upward_ref, egraph) -> Option<Id>`.
/// 
/// We recurse over the full graph rooted at `eclass` and replace any `Var` (or `IVar` if `ivars=true`)
/// with the result of calling the function with:
/// * `actual_idx`: if we're matching on Var(i) this is i
/// * `depth`: how many Lamdas are between this Var and the original toplevel eclass this was called on
/// * `which_upward_ref`: this is just actual_idx-depth
/// * `egraph`: the EGraph we're operating on
/// 
/// Note that we wont touch any branches of the tree that dont have free variables with respect to the toplevel,
/// so this will never be called on some Var(i) if it is not considered a free variable in `eclass` as a whole.
/// 
/// This function is fairly efficient. We cache both within and between calls to it, it uses
/// the enode data that tells us if there are no free variables in a branch (and thus it can be ignored),
/// it operates on the structurally hashed form of the graph, etc.
pub fn recursive_var_mod(
    var_mod: impl Fn(i32, i32, i32, &mut EGraph) -> Option<Id>,
    ivars: bool,
    eclass:Id,
    egraph: &mut EGraph,
    seen: &mut RecVarModCache
    ) -> Option<Id>
    {
        recursive_var_mod_helper(
            &var_mod,
            ivars,
            eclass,
            0,
            egraph,
            seen,
        )
}

/// see `recursive_var_mod`
fn recursive_var_mod_helper(
    var_mod: &impl Fn(i32, i32, i32, &mut EGraph) -> Option<Id>,
    ivars: bool, // whether to run this on vars or ivars
    eclass:Id,
    depth: i32,
    egraph: &mut EGraph,
    seen : &mut RecVarModCache,
    ) -> Option<Id>
    {
        // important invariant for ivars=false case: a $i with i==depth would be a $0 pointer at the top level
        // meaning i<depth is an internal pointer that doesnt break the top level
        let eclass = egraph.find(eclass);
        let key = (eclass,depth);

        if seen.contains_key(&key) {
            return seen[&key];
        }

        if  (ivars && egraph[eclass].data.free_ivars.is_empty())
        || (!ivars && egraph[eclass].data.free_vars.iter().all(|i| *i < depth)) {
            // if we're replacing ivars and theres no ivars in this subtree, we can return early
            // if we're replacing vars, from our invariant (above) we know i<depth is an internal pointer that doesnt point out of the top level so again we can return early
            seen.insert(key, Some(eclass));
            return Some(eclass)
        }

        // this is for loop breaking (though there shouldnt be loops in my new DAG setup anyways)
        seen.insert(key, None);
        
        // if you want a multiple-node-per-eclass version of this that unions together the stuff from diff branches, see my old code!
        assert!(egraph[eclass].nodes.len() == 1);
        // clone to appease the borrow checker
        let enode = egraph[eclass].nodes[0].clone();

        let new_eclass = match enode {
            Lambda::Var(i) => {
                if ivars {
                    panic!("unreachable, Var doesnt have free IVars")
                }
                assert!(i >= depth); // otherwise we should have returned earlier
                // by our invariant be have i-depth as the toplevel version of this index
                var_mod(i, depth, i-depth, egraph)
            }
            Lambda::IVar(i) => {
                if !ivars {
                    panic!("unreachable, IVar doesnt have free Vars")
                }
                var_mod(i, depth, i-depth, egraph)
            }
            Lambda::Prim(_) => {
                panic!("unreachable, Prim never has free vars/ivars")
            }
            Lambda::App([f, x]) => {
                // recurse in each (class shift will return early if no shifting is needed) and build a new App
                let fnew_opt = recursive_var_mod_helper(var_mod, ivars, f, depth, egraph, seen);
                let xnew_opt = recursive_var_mod_helper(var_mod, ivars, x, depth, egraph, seen);
                match (fnew_opt,xnew_opt) {
                    (Some(fnew),Some(xnew)) => Some(egraph.add(Lambda::App([fnew, xnew]))),
                    _ => None,
                }
            }
            Lambda::Lam([b]) => {
                // increment depth
                recursive_var_mod_helper(var_mod, ivars, b, depth+1, egraph, seen)
                .map(|bnew| egraph.add(Lambda::Lam([bnew])))
            }
            Lambda::Programs(_) => {
                panic!("attempted to shift a Programs node")
            }
        };

        if let Some(new_eclass) = new_eclass {
            let new_eclass = egraph.find(new_eclass);
            seen.insert(key, Some(new_eclass));
            Some(new_eclass)
        } else {
            None
        }
}

#[inline]
/// Takes a SORTED vector of copyable items and a key function and groups adjacent equal-key items
/// into subvectors, returning the vector of these subvectors.
pub fn group_by_key<T: Copy, U: Ord>(v: Vec<T>, key: impl Fn(&T)->U) -> Vec<Vec<T>> {
    let mut group = vec![v[0]];
    let mut groups = vec![];
    
    for i in 1..v.len() {
        // group zippers by their left sides being the same
        if key(&v[i]) == key(&v[i-1]) {
            // add on to old ztuplegroup
            group.push(v[i]);
        } else {
            // start a new ztuplegroup
            groups.push(group);
            group = vec![v[i]];
        }
    }
    groups.push(group);
    groups
}

/// Returns a vec from node id to number of places that node is used in the tree. Essentially this just
/// follows all paths down from the root and logs how many times it encounters each node
pub fn num_paths_to_node(roots: &[Id], treenodes: &[Id], egraph: &EGraph) -> (Vec<i32>, Vec<Vec<i32>>) {
    let mut num_paths_to_node_by_root_idx: Vec<Vec<i32>> = vec![vec![0; treenodes.len()]; roots.len()];
    // treenodes.iter().for_each(|treenode| {
    //     num_paths_to_node.insert(*treenode, 0);
    // });
    fn helper(num_paths_to_node: &mut Vec<i32>, node: &Id, egraph: &EGraph) {
        // num_paths_to_node.insert(*child, num_paths_to_node[node] + 1);
        num_paths_to_node[usize::from(*node)] += 1;
        for child in egraph[*node].nodes[0].children() {
            helper(num_paths_to_node, child, egraph);
        }
    }
    let mut num_paths_to_node_all: Vec<i32> = vec![0; treenodes.len()];
    num_paths_to_node_by_root_idx.iter_mut().enumerate().for_each(|(i,num_paths_to_node)| {
        helper(num_paths_to_node, &roots[i], egraph);
        for i in 0..treenodes.len() {
            num_paths_to_node_all[i] += num_paths_to_node[i];
        }
    });
    
    (num_paths_to_node_all, num_paths_to_node_by_root_idx)
}

/// same as Itertools::counts() but returns an FxHashMap instead of a HashMap
pub fn counts_ahash<T: Hash + Eq + Clone>(v: &[T]) -> FxHashMap<T, usize>
{
    let mut counts = FxHashMap::default();
    v.iter().for_each(|item| *counts.entry(item.clone()).or_default() += 1);
    counts
}


// pub trait IterUtil : Iterator {
//     /// same as Itertools::counts() but returns an FxHashMap instead of a HashMap
//     fn counts_ahash(self) -> FxHashMap<Self::Item, usize>
//     where
//         Self: Sized,
//         Self::Item: Eq + Hash,
//     {
//         let mut counts = FxHashMap::default();
//         self.for_each(|item| *counts.entry(item).or_default() += 1);
//         counts
//     }
// }