use crate::*;
use std::collections::{HashMap, HashSet};
use compression::*;



/// convert an egraph Id to an Expr. Assumes one node per class (just picks the first node). Note
/// that this could cause an infinite loop if the egraph didnt just have a single node in a class
/// and instead the first node had a self loop.
pub fn extract(eclass: Id, egraph: &EGraph) -> Expr {
    debug_assert!(egraph[eclass].nodes.len() == 1);
    match &egraph[eclass].nodes[0] {
        Lambda::Prim(p) => Expr::prim(*p),
        Lambda::Var(i) => Expr::var(*i),
        Lambda::IVar(i) => Expr::ivar(*i),
        Lambda::App([f,x]) => Expr::app(extract(*f,egraph), extract(*x,egraph)),
        Lambda::Lam([b]) => Expr::lam(extract(*b,egraph)),
        Lambda::Programs(roots) => Expr::programs(roots.iter().map(|r| extract(*r,egraph)).collect()),
    }
}

/// like extract() but works on nodes
pub fn extract_enode(enode: &Lambda, egraph: &EGraph) -> Expr {
    match enode {
        Lambda::Prim(p) => Expr::prim(*p),
        Lambda::Var(i) => Expr::var(*i),
        Lambda::IVar(i) => Expr::ivar(*i),
        Lambda::App([f,x]) => Expr::app(extract(*f,egraph),extract(*x,egraph)),
        Lambda::Lam([b]) => Expr::lam(extract(*b,egraph)),
        _ => {panic!("not rendered")},
    }
}

/// These are like Inventions but with a pointer to the body instead of an Expr
#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
struct PtrInvention {
    pub body:Id, // this will be a subtree which can have IVars
    pub arity: usize, // also equal to max ivar in subtree + 1
    pub name: String
}
impl PtrInvention {
    pub fn new(body:Id, arity: usize, name: String) -> Self {
        PtrInvention {
            body,
            arity,
            name
        }
    }
}

/// Same as `rewrite_with_invention` but for multiple inventions, rewriting with one after another in order, compounding on each other
pub fn rewrite_with_inventions(
    e: Expr,
    invs: &[Invention]
) -> Expr {
    let mut egraph = EGraph::default();
    let root = egraph.add_expr(&e.into());
    rewrite_with_inventions_egraph(root, invs, &mut egraph)
}

/// Rewrite `root` using an invention `inv`. This will use inventions everywhere
/// as long as it decreases the cost. It will account for the fact that using an invention
/// in a child could prevent the use of the invention in the parent - it will always do whatever
/// gives the lowest cost.
/// 
/// For the `EGraph` argument here you can either pass in a fresh egraph constructed by `let mut egraph = EGraph::new(); egraph.add_expr(expr.into())`
/// or if you make repeated calls to this function feel free to pass in the same egraph over and over. It doesn't matter what is in the EGraph already.

pub fn rewrite_with_invention(
    e: Expr,
    inv: &Invention,
) -> Expr {
    let mut egraph = EGraph::default();
    let root = egraph.add_expr(&e.into());
    rewrite_with_invention_egraph(root, inv, &mut egraph)
}

/// Same as `rewrite_with_invention_egraph` but for multiple inventions, rewriting with one after another in order, compounding on each other
pub fn rewrite_with_inventions_egraph(
    root: Id,
    invs: &[Invention],
    egraph: &mut EGraph,
) -> Expr {
    let mut root = root;
    for inv in invs.iter() {
        let expr = rewrite_with_invention_egraph(root, inv, egraph);
        root = egraph.add_expr(&expr.into());
    }
    extract(root,egraph)
}

/// Same as `rewrite_with_invention` but operates on an egraph instead of an Expr.
/// 
/// For the `EGraph` argument here you can either pass in a fresh egraph constructed by `let mut egraph = EGraph::new(); egraph.add_expr(expr.into())`
/// or if you make repeated calls to this function feel free to pass in the same egraph over and over. It doesn't matter what is in the EGraph already
/// as long as `root` is in it.
pub fn rewrite_with_invention_egraph(
    root: Id,
    inv: &Invention,
    egraph: &mut EGraph,
) -> Expr {
    let inv: PtrInvention = PtrInvention::new(egraph.add_expr(&inv.body.clone().into()), inv.arity, inv.name.clone());

    let treenodes = topological_ordering(root, egraph);

    let mut nodecost_of_treenode: HashMap<Id,NodeCost> = Default::default();
    
    for treenode in treenodes.iter() {
        // println!("processing id={}: {}", treenode, extract(*treenode, egraph) );

        // clone to appease the borrow checker
        let node = egraph[*treenode].nodes[0].clone();

        let mut nodecost = NodeCost::new(egraph[*treenode].data.inventionless_cost);

        // trying to use the invs at this node
        if let Some(args) = match_expr_with_inv(*treenode, &inv, &mut nodecost_of_treenode, egraph) {
            let cost: i32 =
                COST_TERMINAL // the new primitive for this invention
                + COST_NONTERMINAL * inv.arity as i32 // the chain of app()s needed to apply the new primitive
                + args.iter()
                    .map(|id| nodecost_of_treenode[&id]
                        .cost_under_inv(&inv)) // cost under ANY of the invs since we allow multiple to be used!
                    .sum::<i32>(); // sum costs of actual args
                    nodecost.new_cost_under_inv(inv.clone(), cost, Some(args));
        }


        // inventions based on specific node type
        match node {
            Lambda::IVar(_) => { unreachable!() }
            Lambda::Var(_) | Lambda::Prim(_) => {},
            Lambda::App([f,x]) => {
                let ref f_nodecost = nodecost_of_treenode[&f];
                let ref x_nodecost = nodecost_of_treenode[&x];
                                
                // costs with inventions as 1 + fcost + xcost. Use inventionless cost as a default.
                // if either fcost or xcost is None (ie infinite)
                let fcost = f_nodecost.cost_under_inv(&inv);
                let xcost = x_nodecost.cost_under_inv(&inv);
                let cost = COST_NONTERMINAL+fcost+xcost;
                nodecost.new_cost_under_inv(inv.clone(), cost, None);
            }
            Lambda::Lam([b]) => {
                // just map +1 over the costs
                let ref b_nodecost = nodecost_of_treenode[&b];
                let bcost = b_nodecost.cost_under_inv(&inv);
                nodecost.new_cost_under_inv(inv.clone(), bcost + COST_NONTERMINAL, None);
            }
            Lambda::Programs(roots) => {
                // no filtering for 2+ uses because we're just doing rewriting here
                let cost = roots.iter().map(|root| {
                        nodecost_of_treenode[root].cost_under_inv(&inv)
                    }).sum();
                    nodecost.new_cost_under_inv(inv.clone(), cost, None);
            }
        }

        nodecost_of_treenode.insert(*treenode, nodecost);
    }

    // Now that we've calculated all the costs, we can extract the cheapest one
    extract_from_nodecosts(root, &inv, &nodecost_of_treenode, egraph)
}

fn extract_from_nodecosts(
    root: Id,
    inv: &PtrInvention,
    nodecost_of_treenode: &HashMap<Id,NodeCost>,
    egraph: &EGraph,
) -> Expr {

    let target_cost = nodecost_of_treenode[&root].cost_under_inv(&inv);

    if let Some((inv,_cost,args)) = nodecost_of_treenode[&root].top_invention() {
        if let Some(args) = args {
            // invention was used here
            let mut expr = Expr::prim(inv.name.clone().into());
            // wrap the new primitive in app() calls. Note that you pass in the $0 args LAST given how appapplamlam works
            // todo perhaps this shouldnt be a .rev() - related to what theo found
            for arg in args.iter() {
                let arg_expr = extract_from_nodecosts(*arg, &inv, nodecost_of_treenode, egraph);
                expr = Expr::app(expr,arg_expr);
            }
            assert_eq!(target_cost,expr.cost());
            return expr
        } else {
            // inventions were used in our children
            let expr: Expr = match &egraph[root].nodes[0] {
                Lambda::Prim(_) | Lambda::Var(_) | Lambda::IVar(_) => {unreachable!()},
                Lambda::App([f,x]) => {
                    let f_expr = extract_from_nodecosts(*f, &inv, nodecost_of_treenode, egraph);
                    let x_expr = extract_from_nodecosts(*x, &inv, nodecost_of_treenode, egraph);
                    Expr::app(f_expr,x_expr)
                },
                Lambda::Lam([b]) => {
                    let b_expr = extract_from_nodecosts(*b, &inv, nodecost_of_treenode, egraph);
                    Expr::lam(b_expr)
                }
                Lambda::Programs(roots) => {
                    let root_exprs: Vec<Expr> = roots.iter()
                        .map(|r| extract_from_nodecosts(*r, &inv, nodecost_of_treenode, egraph))
                        .collect();
                    Expr::programs(root_exprs)
                }
            };
            assert_eq!(target_cost,expr.cost());
            return expr
        }
    } else {
        // no invention was useful, just return original tree
        let expr =  extract(root, egraph);
        assert_eq!(target_cost,expr.cost());
        return expr
    }
}

/// There will be one of these structs associated with each node, and it keeps
/// track of the best inventions for that node, their costs, and their arguments.
#[derive(Debug,Clone)]
struct NodeCost {
    inventionless_cost: i32,
    inventionful_cost: HashMap<PtrInvention, (i32,Option<Vec<Id>>)>, // i32 = cost; and Some(args) gives the arguments if the invention is used at this node
}

impl NodeCost {
    fn new(inventionless_cost: i32) -> Self {
        Self {
            inventionless_cost: inventionless_cost,
            inventionful_cost: HashMap::new()
        }
    }
    /// cost under an invention if it's useful for this node, else inventionless cost
    fn cost_under_inv(&self, inv: &PtrInvention) -> i32 {
        self.inventionful_cost.get(inv).map(|x|x.0).unwrap_or(self.inventionless_cost)
    }
    /// improve the cost using a new invention, or do nothing if we've already seen
    /// a better cost for this invention. Also skip if inventionless cost is better.
    fn new_cost_under_inv(&mut self, inv: PtrInvention, cost:i32, args: Option<Vec<Id>>) {
        if cost < self.inventionless_cost {
            if !self.inventionful_cost.contains_key(&inv)
               || cost < self.inventionful_cost[&inv].0  {
                self.inventionful_cost.insert(inv, (cost,args));
            }
        }
    }
    /// Get the top inventions in decreasing order of cost
    #[allow(dead_code)] // todo at some point add tests for this
    fn top_inventions(&self) -> Vec<PtrInvention> {
        let mut top_inventions: Vec<PtrInvention> = self.inventionful_cost.keys().cloned().collect();
        top_inventions.sort_by(|a,b| self.inventionful_cost[a].0.cmp(&self.inventionful_cost[b].0));
        top_inventions
    }
    /// Get the top inventions in decreasing order of cost
    fn top_invention(&self) -> Option<(PtrInvention,i32,Option<Vec<Id>>)> {
        self.inventionful_cost.iter().min_by_key(|(_k,v)| v.0).map(|(k,v)| (k.clone(),v.0,v.1.clone()))
    }
}


fn match_expr_with_inv(
    root: Id,
    inv: &PtrInvention,
    best_inventions_of_treenode: &mut HashMap<Id, NodeCost>,
    egraph: &mut EGraph,
) -> Option<Vec<Id>> {
    let mut args: Vec<Option<Id>> = vec![None;inv.arity];
    let threadables = threadables_of_inv(inv.clone(), egraph);
    if match_expr_with_inv_rec(root, inv.body, 0, &mut args, &threadables, best_inventions_of_treenode, egraph) {
        assert!(args.iter().all(|x| x.is_some()), "{:?}\n{}\n{}", args, extract(root,egraph), extract(inv.body,egraph)); // if any didnt unwrap() fine that would mean some variable wasnt used at all in the invention body
        Some(args.iter().map(|arg| arg.unwrap()).collect()) 
    } else {
        None
    }
}

fn match_expr_with_inv_rec(
    root: Id,
    inv: Id,
    depth: i32,
    args: &mut [Option<Id>],
    threadables: &HashSet<Id>,
    best_inventions_of_treenode: &mut HashMap<Id, NodeCost>,
    egraph: &mut EGraph,
) -> bool {
    // println!("comparing:\n\t{}\n\t{}",
    //     extract(root, egraph).to_string(),
    //     extract(inv, egraph).to_string()
    // );
    // println!("processing:\n\troot:{}\n\tinv:{} ts:{}", extract(root,egraph), extract(inv,egraph), threadables.contains(&inv));
    match (&egraph[root].nodes[0].clone(), &egraph[inv].nodes[0].clone()) { // clone for the borrow checker
        (Lambda::Prim(p), Lambda::Prim(q)) => { p == q },
        (Lambda::Var(i), Lambda::Var(j)) => { i == j },
        (root_node, Lambda::App([g,y])) if threadables.contains(&inv) => {
            // todo this whole section is a nightmare so make sure there arent bugs

            // a thread site only applies when the set of internal pointers is the same
            // as the thread site's set of pointers.
            let internal_free_vars: HashSet<i32> = egraph[root].data.free_vars.iter().filter(|i| **i < depth).cloned().collect();
            let num_to_thread = internal_free_vars.len() as i32;
            if internal_free_vars == egraph[inv].data.free_vars {
                // println!("threading");
                // free vars match exactly so we could thread here note that if we match here than an inner thread site wont match.
                // however, also note that there some chance a nonthreading approach could work too which is always simpler,
                // for example when matching (#0 $0) against (inc $0) we can simply set #0=inc instead of #0=(lam (inc $0))
                // lets clone our args and reset if this fails
                if let Lambda::App([f,x]) = root_node {
                    let cloned_args: Vec<_> = args.iter().cloned().collect();
                    if match_expr_with_inv_rec(*f, *g, depth, args, threadables, best_inventions_of_treenode, egraph)
                    && match_expr_with_inv_rec(*x, *y, depth, args, threadables, best_inventions_of_treenode, egraph) {
                        return true;
                    }
                    args.clone_from_slice(cloned_args.as_slice());
                }

                // Now lets build the desired argument out of `root` by basically
                // following what bubbling up would normally do: downshifting for each
                // internal lambda in the invention, except adding an extra lambda on top
                // in the case of threaded variables
                let mut arg = root;
                for i in 0..depth {
                    if egraph[inv].data.free_vars.contains(&i) {
                        // protect $0 before continuing with the shift
                        arg = egraph.add(Lambda::Lam([arg]));
                    }
                    arg = shift(arg, -1, egraph, &mut None).unwrap();
                }

                // now copy over the best_inventions
                if !best_inventions_of_treenode.contains_key(&arg) {
                    let mut cloned = best_inventions_of_treenode[&root].clone();
                    cloned.inventionless_cost += COST_NONTERMINAL * num_to_thread;
                    // we'll force this arg to not use a toplevel invention at the "lam" node hence the None
                    cloned.inventionful_cost.iter_mut().for_each(|(_key, val)| {val.0 += COST_NONTERMINAL * num_to_thread; val.1 = None});
                    best_inventions_of_treenode.insert(arg,cloned);
                }

                let ivar = *egraph[inv].data.free_ivars.iter().next().unwrap() as usize;

                // now finally check that these results align
                if let Some(v) = args[ivar] {
                    arg == v // if #j was bound to some id `v` before, then `root` must be `v` for this to match
                } else {
                    args[ivar] = Some(arg);
                    // println!("Assigned #{} = {}", ivar, extract(arg,egraph));
                    true
                }
            } else {
                // not threadable case 
                if let Lambda::App([f,x]) = root_node {
                    return match_expr_with_inv_rec(*f, *g, depth, args, threadables, best_inventions_of_treenode, egraph)
                    && match_expr_with_inv_rec(*x, *y, depth, args, threadables, best_inventions_of_treenode, egraph)
                }
                false
            }
        },
        (Lambda::App([f,x]), Lambda::App([g,y])) => {
            // not threadable case 
            return match_expr_with_inv_rec(*f, *g, depth, args, threadables, best_inventions_of_treenode, egraph)
            && match_expr_with_inv_rec(*x, *y, depth, args, threadables, best_inventions_of_treenode, egraph)
        }
        (Lambda::Lam([b]), Lambda::Lam([c])) => {
            match_expr_with_inv_rec(*b, *c, depth+1, args, threadables, best_inventions_of_treenode, egraph)
        },
        (_, Lambda::IVar(j)) => {
            // We need to bind #j to `root`
            // First `root` needs to be downshifted by `depth`. There are 3 cases:
            let shifted_root: Id = if egraph[root].data.free_vars.is_empty() {
                // 1. `root` has no free variables so no shifting is needed
                root
            } else if egraph[root].data.free_vars.iter().min().unwrap() - depth >= 0 {
                // 2. `root` has free variables but they all point outside the invention so are safe to decrement
                let shifted_root = shift(root, -depth, egraph, &mut None).unwrap();
                // copy the cost of the unshifted node to the shifted node (see PR#1 comments for why this is safe)
                if !best_inventions_of_treenode.contains_key(&shifted_root) {
                    let cloned = best_inventions_of_treenode[&root].clone();
                    best_inventions_of_treenode.insert(shifted_root,cloned);
                }
                shifted_root
            } else {
                return false // threading needed but this is not a thread site
            };

            if let Some(v) = args[*j as usize] {
                shifted_root == v // if #j was bound to some id `v` before, then `root` must be `v` for this to match
            } else {
                args[*j as usize] = Some(shifted_root);
                // println!("Assigned #{} = {}", j, extract(shifted_root,egraph));
                true
            }
         },
        _ => { false }
    }
}

fn threadables_of_inv(inv: PtrInvention, egraph: &EGraph) -> HashSet<Id> {
    // a threadable is a (app #i $j) or (app <threadable> $j)
    // assert j > k sanity check
    // println!("Invention: {}", inv.to_expr(egraph));
    let mut threadables: HashSet<Id> = Default::default();
    let nodes = topological_ordering(inv.body, egraph);
    for node in nodes {
        if let Lambda::App([f,x]) = egraph[node].nodes[0] {
            if matches!(egraph[x].nodes[0], Lambda::Var(_)) {
                if matches!(egraph[f].nodes[0], Lambda::IVar(_)) ||
                  threadables.contains(&f) {
                    threadables.insert(node);
                    // println!("Identified threadable: {}", extract(node,egraph));
                }
            }
        }
    }
    threadables
}