use crate::*;
use std::collections::{HashSet,HashMap};
use std::fmt::{self, Formatter, Display};
use clap::Parser;
use std::path::PathBuf;

/// Args for compression
#[derive(Parser, Debug)]
#[clap(name = "Dream Egg")]
pub struct CompressionArgs {
    /// json file to read compression input programs from
    #[clap(short, long, parse(from_os_str), default_value = "data/train_19.json")]
    pub file: PathBuf,

    /// Number of iterations to run compression for
    #[clap(short, long, default_value = "3")]
    pub iterations: usize,

    /// max arity of inventions
    #[clap(short='a', long, default_value = "2")]
    pub max_arity: usize,

    /// beam size
    // #[clap(short, long, default_value = "10000000")]
    // beam_size: usize,

    /// disable caching
    #[clap(long)]
    pub no_cache: bool,

    /// whether to render the inventions
    #[clap(long)]
    pub render_inventions: bool,

    /// render the final egraph
    #[clap(long)]
    pub render_final: bool,

    /// render initial egraph
    #[clap(long)]
    pub render_initial: bool,

    /// number of inventions to print - set to 0 if you dont want to print inventions at all
    #[clap(long, default_value="0")]
    pub print_inventions: usize,
}

/// nonterminals ("app" and "lam") cost 1/100th of a terminal ("var", "ivar", "prim"). This is because nonterminals
/// can be autofilled based on the type of the hole you're filling during most search methods.
const COST_NONTERMINAL: i32 = 1;
const COST_TERMINAL: i32 = 100;

type EGraph = egg::EGraph<Lambda, LambdaAnalysis>;

/// The analysis data associated with each Lambda node
#[derive(Debug)]
pub struct Data {
    free_vars: HashSet<i32>, // $i vars. For example (lam $2) has free_vars = {1}.
    free_ivars: HashSet<i32>, // #i ivars
    inventionless_cost: i32,
}

/// An invention we've found (ie a learned function we can use to compress the program).
/// Inventions have a body + an arity
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct Invention {
    body:Id, // this will be a subtree which can have IVars
    arity: usize // also equal to max ivar in subtree + 1
}

/// At the end of the day we convert our Inventions into InventionExprs to make
/// them standalone without needing to carry the EGraph around to figure out what
/// the body Id points to.
#[derive(Debug, Clone)]
pub struct InventionExpr {
    body: Expr, // invention body (not wrapped in lambdas)
    arity: usize
}

impl Display for InventionExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "(arity={}: {})", self.arity, self.body)
    }
}

impl Invention {
    fn new(body:Id, arity: usize) -> Invention {
        Invention { body, arity }
    }
    // fn canonicalize(&mut self, egraph: &EGraph) {
    //     self.body = egraph.find(self.body);
    // }
    fn is_canonical(&self, egraph: &mut EGraph) -> bool {
        self.body == egraph.find(self.body)
    }
    fn valid_invention(&self, egraph: &EGraph) -> bool {
        // even invalid Inventions are important as parts of AppLams that will propagate recursively upward,
        // This checks that there aren't any upward refs that go beyond the args of the AppLam itself
        // egraph[self.body].data.free_vars.iter().all(|i| *i < (self.arity as i32))
        egraph[self.body].data.free_vars.is_empty()
    }
    fn to_expr(&self, egraph: &EGraph) -> InventionExpr {
        // wrap body in lambdas
        let expr = extract(self.body, &egraph);
        InventionExpr {body: expr, arity:self.arity}
    }
}

/// An AppLam is an applied lambda, so in lambda calculus it would look like (app (lam ...) ...)
/// The lambda's body is in the `inv: Invention` field.
/// 
/// Note that this actually captures multiarg applams. The first argument .args[0] corresponds to
/// the #0 free ivar in the body. This means technically if you were to write out what a 2-arg applam
/// might look like it would be (app (app (lam (lam ...)) arg1) arg0) which is a bit backwards
/// from what you might expect (but think about where a $0 would point and it makes sense)
/// 
/// But in reality there are no apps and no lams, everything is implicitly captured in the AppLam. The
/// Invention does NOT have a lam() at the top.
#[derive(Debug, Clone)]
struct AppLam {
    inv: Invention,
    args: Vec<Id>, // these should be (possibly shifted) subtrees of the original tree. No IVars.
}

impl AppLam {
    fn new(body: Id, args: Vec<Id>) -> AppLam {
        AppLam {
            inv: Invention::new(body, args.len()),
            args: args,
        }
    }
    // fn canonicalize(&mut self, egraph: &mut EGraph) {
    //     self.inv.canonicalize(egraph);
    //     for arg in &mut self.args {
    //         if !canonical(arg, egraph) {
    //             *arg = egraph.find(*arg);
    //         }
    //     }
    // }
    fn is_canonical(&self, egraph: &mut EGraph) -> bool {
        self.inv.is_canonical(egraph) &&
        self.args.iter().all(|arg| canonical(arg, egraph))
    }
    /// unions together all the upward refs of body + args to get the free variables of this applam
    fn free_vars(&self, egraph: &mut EGraph) -> HashSet<i32> {
        let mut free_vars: HashSet<i32> = egraph[self.inv.body].data.free_vars.clone();
        for arg in self.args.iter() {
            free_vars.extend(egraph[*arg].data.free_vars.clone());
        }
        free_vars
    }
    fn to_string(&self, egraph: &EGraph) -> String {
        format!("inv:{}\narg:{}",
            self.inv.to_expr(egraph),
            self.args.iter().map(|arg| extract(*arg, egraph).to_string()).collect::<Vec<_>>().join("\narg:")
        )
    }

}

/// There will be one of these structs associated with each node, and it keeps
/// track of the best inventions for that node, their costs, and their arguments.
#[derive(Debug,Clone)]
struct NodeCost {
    inventionless_cost: i32,
    inventionful_cost: HashMap<Invention, (i32,Option<Vec<Id>>)>, // i32 = cost; and Some(args) gives the arguments if the invention is used at this node
}

impl NodeCost {
    fn new(inventionless_cost: i32) -> Self {
        Self {
            inventionless_cost: inventionless_cost,
            inventionful_cost: HashMap::new()
        }
    }
    /// cost under an invention if it's useful for this node, else inventionless cost
    fn cost_under_inv(&self, inv: &Invention) -> i32 {
        self.inventionful_cost.get(inv).map(|x|x.0).unwrap_or(self.inventionless_cost)
    }
    /// min cost under any of a list of invs
    fn cost_under_invs(&self, invs: &[Invention]) -> i32 {
        invs.iter().map(|inv| self.cost_under_inv(inv)).min().unwrap()
    }
    /// improve the cost using a new invention, or do nothing if we've already seen
    /// a better cost for this invention. Also skip if inventionless cost is better.
    fn new_cost_under_inv(&mut self, inv: Invention, cost:i32, args: Option<Vec<Id>>) {
        if cost < self.inventionless_cost {
            if !self.inventionful_cost.contains_key(&inv)
               || cost < self.inventionful_cost[&inv].0  {
                self.inventionful_cost.insert(inv, (cost,args));
            }
        }
    }
    /// Get the top inventions in decreasing order of cost
    fn top_inventions(&self) -> Vec<Invention> {
        let mut top_inventions: Vec<Invention> = self.inventionful_cost.keys().cloned().collect();
        top_inventions.sort_by(|a,b| self.inventionful_cost[a].0.cmp(&self.inventionful_cost[b].0));
        top_inventions
    }
    /// Get the top inventions in decreasing order of cost
    fn top_invention(&self) -> Option<(Invention,i32,Option<Vec<Id>>)> {
        self.inventionful_cost.iter().min_by_key(|(_k,v)| v.0).map(|(k,v)| (*k,v.0,v.1.clone()))
    }
}

/// convert an egraph Id to an Expr by extracting the expression 
fn extract_old(eclass: Id, egraph: &EGraph) -> Expr {
    let mut extractor = Extractor::new(&egraph, ProgramCost{});
    let (_,p) = extractor.find_best(eclass);
    p.into()
}

/// convert an egraph Id to an Expr. Assumes one node per class (just picks the first node). Note
/// that this could cause an infinite loop if the egraph didnt just have a single node in a class
/// and instead the first node had a self loop.
fn extract(eclass: Id, egraph: &EGraph) -> Expr {
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
fn extract_enode(enode: &Lambda, egraph: &EGraph) -> Expr {
    match enode {
        Lambda::Prim(p) => Expr::prim(*p),
        Lambda::Var(i) => Expr::var(*i),
        Lambda::IVar(i) => Expr::ivar(*i),
        Lambda::App([f,x]) => Expr::app(extract(*f,egraph),extract(*x,egraph)),
        Lambda::Lam([b]) => Expr::lam(extract(*b,egraph)),
        _ => {panic!("not rendered")},
    }
}


fn threadables_of_inv(inv: Invention, egraph: &EGraph) -> HashSet<Id> {
    // a threadable is a (app #i $j) or (app <threadable> $j)
    // assert j > k sanity check
    // println!("Invention: {}", inv.to_expr(egraph));
    let mut threadables: HashSet<Id> = Default::default();
    let nodes = toplogical_ordering(inv.body, egraph);
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


fn match_expr_with_inv(
    root: Id,
    inv: &Invention,
    best_inventions_of_treenode: &mut HashMap<Id, NodeCost>,
    egraph: &mut EGraph,
) -> Option<Vec<Id>> {
    let mut args: Vec<Option<Id>> = vec![None;inv.arity];
    let threadables = threadables_of_inv(*inv, egraph);
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
                    arg = shift(arg, Shift::ShiftVar(-1), egraph, None).unwrap();
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
                let shifted_root = shift(root, Shift::ShiftVar(-depth), egraph, None).unwrap();
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




/// Rewrite `root` using an invention `inv`. This will use inventions everywhere
/// as long as it decreases the cost. It will account for the fact that using an invention
/// in a child could prevent the use of the invention in the parent - it will always do whatever
/// gives the lowest cost.
fn rewrite_with_inventions(
    root: Id,
    invs: &[Invention],
    replace_invs_with: &[&str],
    egraph: &mut EGraph,
) -> Expr {
    let root = egraph.find(root);

    let treenodes = toplogical_ordering(root, egraph);

    let mut nodecost_of_treenode: HashMap<Id,NodeCost> = Default::default();
    
    for treenode in treenodes.iter() {
        // println!("processing id={}: {}", treenode, extract(*treenode, egraph) );

        // clone to appease the borrow checker
        let node = egraph[*treenode].nodes[0].clone();

        let mut nodecost = NodeCost::new(egraph[*treenode].data.inventionless_cost);

        // trying to use the invs at this node
        for inv in invs.iter() {
            if let Some(args) = match_expr_with_inv(*treenode, inv, &mut nodecost_of_treenode, egraph) {
                let cost: i32 =
                    COST_TERMINAL // the new primitive for this invention
                    + COST_NONTERMINAL * inv.arity as i32 // the chain of app()s needed to apply the new primitive
                    + args.iter()
                        .map(|id| nodecost_of_treenode[&id]
                            .cost_under_invs(invs)) // cost under ANY of the invs since we allow multiple to be used!
                        .sum::<i32>(); // sum costs of actual args
                        nodecost.new_cost_under_inv(*inv, cost, Some(args));
            }
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
                for inv in invs.iter() {
                    let fcost = f_nodecost.cost_under_inv(inv);
                    let xcost = x_nodecost.cost_under_inv(inv);
                    let cost = COST_NONTERMINAL+fcost+xcost;
                    nodecost.new_cost_under_inv(*inv, cost, None);
                }
            }
            Lambda::Lam([b]) => {
                // just map +1 over the costs
                let ref b_nodecost = nodecost_of_treenode[&b];
                for inv in invs.iter() {
                    let bcost = b_nodecost.cost_under_inv(inv);
                    nodecost.new_cost_under_inv(*inv, bcost + COST_NONTERMINAL, None);
                }
            }
            Lambda::Programs(roots) => {
                // no filtering for 2+ uses because we're just doing rewriting here
                for inv in invs.iter() {
                    let cost = roots.iter().map(|root| {
                            nodecost_of_treenode[root].cost_under_inv(inv)
                        }).sum();
                        nodecost.new_cost_under_inv(*inv, cost, None);
                }
            }
        }

        nodecost_of_treenode.insert(*treenode, nodecost);
    }

    // Now that we've calculated all the costs, we can extract the cheapest one
    extract_from_nodecosts(root, invs, &nodecost_of_treenode, replace_invs_with, egraph)
}

fn extract_from_nodecosts(
    root: Id,
    invs: &[Invention],
    nodecost_of_treenode: &HashMap<Id,NodeCost>,
    replace_invs_with: &[&str],
    egraph: &EGraph,
) -> Expr {

    let target_cost = nodecost_of_treenode[&root].cost_under_invs(invs);

    if let Some((inv,cost,args)) = nodecost_of_treenode[&root].top_invention() {
        if let Some(args) = args {
            // invention was used here
            let prim: &str = replace_invs_with[invs.iter().position(|x| *x == inv).unwrap()];
            let mut expr = Expr::prim(prim.into());
            // wrap the new primitive in app() calls. Note that you pass in the $0 args LAST given how appapplamlam works
            // todo perhaps this shouldnt be a .rev() - related to what theo found
            for arg in args.iter().rev() {
                let arg_expr = extract_from_nodecosts(*arg, invs, nodecost_of_treenode, replace_invs_with, egraph);
                expr = Expr::app(expr,arg_expr);
            }
            assert_eq!(target_cost,expr.cost());
            return expr
        } else {
            // inventions were used in our children
            let expr: Expr = match &egraph[root].nodes[0] {
                Lambda::Prim(_) | Lambda::Var(_) | Lambda::IVar(_) => {unreachable!()},
                Lambda::App([f,x]) => {
                    let f_expr = extract_from_nodecosts(*f, invs, nodecost_of_treenode, replace_invs_with, egraph);
                    let x_expr = extract_from_nodecosts(*x, invs, nodecost_of_treenode, replace_invs_with, egraph);
                    Expr::app(f_expr,x_expr)
                },
                Lambda::Lam([b]) => {
                    let b_expr = extract_from_nodecosts(*b, invs, nodecost_of_treenode, replace_invs_with, egraph);
                    Expr::lam(b_expr)
                }
                Lambda::Programs(roots) => {
                    let root_exprs: Vec<Expr> = roots.iter()
                        .map(|r| extract_from_nodecosts(*r, invs, nodecost_of_treenode, replace_invs_with, egraph))
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




/// Extracts an expression under an invention. This rewrites the expression to use the invention
/// if it decreases the cost.
/// 
/// todo The current implementation requires the results of the full compression search, however
/// it should be possible to do this in a much more efficient way that works even on things
/// that weren't part of the original set of expressions. That would be easy with no HOFs and
/// might be more difficult with HOFs but is probably still possible
fn extract_under_inv(
    root: Id,
    inv: Invention,
    replace_inv_with: &str,
    applams_of_treenode: &HashMap<Id,Vec<AppLam>>,
    best_inventions_of_treenode: &HashMap<Id,NodeCost>,
    egraph: &EGraph,
) -> Expr {
    let root = egraph.find(root);
    let target_cost:i32 = best_inventions_of_treenode[&root].cost_under_inv(&inv);

    if best_inventions_of_treenode[&root].inventionful_cost.contains_key(&inv)
       && applams_of_treenode[&root].iter().any(|applam| applam.inv == inv) {
        let applam: Vec<AppLam> = applams_of_treenode[&root].iter().filter(|applam| applam.inv == inv).cloned().collect();
        let applam = if applam.len() > 1 {
            applam.iter().min_by_key(|al| al.args.iter().map(|arg| egraph[*arg].data.inventionless_cost).sum::<i32>()).unwrap()  // TODO make this cheaper
        } else {
            &applam[0]
        };
        let mut expr = Expr::prim(replace_inv_with.into());
        // wrap the new primitive in app() calls. Note that you pass in the $0 args LAST given how appapplamlam works
        for arg in applam.args.iter().rev() {
            let arg_expr = extract_under_inv(*arg, inv, replace_inv_with, applams_of_treenode, best_inventions_of_treenode, egraph);
            expr = Expr::app(expr,arg_expr);
        }
        assert_eq!(target_cost,expr.cost(),
          "\n{}\n{}\n",
          extract(root,egraph),
          inv.to_expr(egraph));
        return expr
    }
    
    assert!(egraph[root].nodes.len() == 1);
    let expr: Expr = match &egraph[root].nodes[0] {
        Lambda::Prim(p) => {
            Expr::prim(*p)
        },
        Lambda::Var(i) => {
            Expr::var(*i)
        },
        Lambda::IVar(_) => {
            panic!("Shouldn't be extracting an IVar under an invention")
            //into_expr.add(Lambda::IVar(*i))
        },
        Lambda::App([f,x]) => {
            let f_expr = extract_under_inv(*f, inv, replace_inv_with, applams_of_treenode, best_inventions_of_treenode, egraph);
            let x_expr = extract_under_inv(*x, inv, replace_inv_with, applams_of_treenode, best_inventions_of_treenode, egraph);
            Expr::app(f_expr,x_expr)
        },
        Lambda::Lam([b]) => {
            let b_expr = extract_under_inv(*b, inv, replace_inv_with, applams_of_treenode, best_inventions_of_treenode, egraph);
            Expr::lam(b_expr)
        }
        Lambda::Programs(roots) => {
            let root_exprs: Vec<Expr> = roots.iter()
                .map(|r| extract_under_inv(*r, inv, replace_inv_with, applams_of_treenode, best_inventions_of_treenode, egraph))
                .collect();
            Expr::programs(root_exprs)
        }
    };

    assert_eq!(target_cost,expr.cost());
    expr
}


#[inline]
fn canonical(id:&Id, egraph: &EGraph) -> bool {
    egraph.find(*id) == *id
}

/// Narrows a beam. Not actually used currently since Invention beam size wasn't an issue at all (AppLam was
/// but it's way less clear how to narrow that beam)
fn narrow_beam(beam: &mut HashMap<Invention,i32>, beam_size: usize) {
    if beam.len() < beam_size {
        return
    }
    // println!("Need to narrow beam! (worth turning this print message off if it ever actually prints)");
    let num_to_drop = beam_size - beam.len();
    let mut costs: Vec<(Invention,i32)> = beam.iter().map(|(id,cost)|(*id,*cost)).collect();
    // DECREASING order of cost (since i do cost2.cmp(cost1))
    costs.sort_by(|(_,cost1),(_,cost2)| cost2.cmp(cost1));
    for (id,_) in costs.iter().take(num_to_drop) {
        beam.remove(id);
    }
}

#[derive(Default)]
pub struct LambdaAnalysis;

impl Analysis<Lambda> for LambdaAnalysis {
    type Data = Data;
    // fn merge(&self, to: &mut Self::Data, from: Self::Data) -> bool;
    fn merge(&mut self, to: &mut Data, from: Data) -> egg::DidMerge {
        // we really shouldnt be merging anyone ever rn I think.
        panic!("shouldn't be merging");

        assert_eq!(to.free_vars,from.free_vars);
        assert_eq!(to.free_ivars,from.free_ivars);
        assert_eq!(to.inventionless_cost,from.inventionless_cost);

        // keep the lowest inventionless cost
        // modified |= merge_inventionless(&mut to.inventionless_cost_any, &from.inventionless_cost_any);
        
        DidMerge(false,false) // didnt modify anything
    }

    fn make(egraph: &EGraph, enode: &Lambda) -> Data {
        let mut free_vars: HashSet<i32> = HashSet::new();
        let mut free_ivars: HashSet<i32> = HashSet::new();
        match enode {
            Lambda::Var(i) => {
                free_vars.insert(*i);
            }
            Lambda::IVar(i) => {
                free_ivars.insert(*i);
            }
            Lambda::Prim(_) => {
            }
            Lambda::App([f, x]) => {
                // union of f and x
                free_vars.extend(egraph[*f].data.free_vars.iter());
                free_vars.extend(egraph[*x].data.free_vars.iter());
                free_ivars.extend(egraph[*f].data.free_ivars.iter());
                free_ivars.extend(egraph[*x].data.free_ivars.iter());
            }
            Lambda::Lam([b]) => {
                // body, subtract 1 from all values, remove the -1 if its in there
                free_vars.extend(egraph[*b].data.free_vars.iter()
                    .map(|x| x-1)
                    .filter(|x| *x >= 0));
                free_ivars.extend(egraph[*b].data.free_ivars.iter());
            }
            Lambda::Programs(programs) => {
                // assert no free variables in programs
                //assert!(programs.iter().all(|p| egraph[*p].data.free_vars.is_empty()));
                assert!(programs.iter().all(|p| egraph[*p].data.free_ivars.is_empty()));
            }
        }
        let inventionless_cost = match enode {
            Lambda::Var(_) | Lambda::IVar(_) | Lambda::Prim(_) => COST_TERMINAL,
            Lambda::App([f,x]) => {
                    COST_NONTERMINAL
                    + egraph[*f].data.inventionless_cost
                    + egraph[*x].data.inventionless_cost
                }
            Lambda::Lam([b]) => {
                COST_NONTERMINAL + egraph[*b].data.inventionless_cost
            }
            Lambda::Programs(ps) => {
                ps.iter().map(|p| egraph[*p].data.inventionless_cost).sum()
            }
        };
        Data {
               free_vars: free_vars,
               free_ivars: free_ivars,
               inventionless_cost: inventionless_cost
            }
    }

    fn modify(_egraph: &mut EGraph, _id: Id) {
    }
}


/// Does debruijn index shifting of a subtree. Note that the type of shifting is specified by
/// the `shift` argument.
/// 
/// Shift variants:
/// * ShiftVar(i32) -> increment all free variables by the given amount
/// * ShiftIVar(i32) -> increment all ivars by given amount
/// * TableShiftIVar(Vec<i32>) -> increment ivar #i by the amount given by table[i]
#[inline] // useful to inline since callsite can usually tell which Shift type is happening allowing further optimization
fn shift(e: Id, shift: Shift, egraph: &mut EGraph, caches: Option<&mut CacheGenerator>) -> Option<Id> {
    let mut empty = HashMap::new();
    let seen = match caches {
        Some(caches) => caches.get(&shift),
        None => &mut empty,
    };
    match shift {
        Shift::ShiftVar(incr_by) => recursive_var_mod(
            |actual_idx, _depth, _which_upward_ref, egraph| {
                Some(egraph.add(Lambda::Var(actual_idx + incr_by)))
            },
            false, // operate on Vars
            e,egraph,seen
        ),
        Shift::ShiftIVar(incr_by) => recursive_var_mod(
            |actual_idx, _depth, _which_upward_ref, egraph| {
                // note this is IVars so depth and which_upward_ref are meaningless to us
                Some(egraph.add(Lambda::IVar(actual_idx + incr_by)))
            },
            true, // operate on IVars
            e,egraph,seen
        ),
        Shift::TableShiftIVar(shift_table) => recursive_var_mod(
            |actual_idx, _depth, _which_upward_ref, egraph| {
                // shift variable up or down whatever the shift table says it should be
                // note this is IVars so depth and which_upward_ref are meaningless to us
                Some(egraph.add(Lambda::IVar(actual_idx + shift_table[actual_idx as usize])))
            },
            true, // operate on IVars
            e,egraph,seen
        )
    }
}



// not used in the new verison but should be compatible with everything we've got!
// fn inline(e: Id, replace_with: Id, egraph: &mut EGraph, seen: &mut RecVarModCache) -> Option<Id> {
//     recursive_var_mod(
//         |actual_idx, depth, which_upward_ref, egraph| {
//             if which_upward_ref == 0 {
//                 // ShifterVM { incr_by: depth }.recursive_var_mod(replace_with, egraph)
//                 shift(replace_with, depth, egraph, None) // note i have it just make a new hashmap on the spot for this, caching would be better
//             } else {
//                 // we need to decrement this by 1 since its a pointer above the lambda we removed
//                 Some(egraph.add(Lambda::Var(actual_idx - 1)))
//             }
//         },
//         e,egraph, seen
//     )
// }

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
fn recursive_var_mod(
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


/// the cost of a program, where `app` and `lam` cost 1, `programs` costs nothing,
/// `ivar` and `var` and `prim` cost 100.
pub struct ProgramCost {}
impl CostFunction<Lambda> for ProgramCost {
    type Cost = i32;
    fn cost<C>(&mut self, enode: &Lambda, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost
    {
        match enode {
            Lambda::Var(_) | Lambda::IVar(_) | Lambda::Prim(_) => COST_TERMINAL,
            Lambda::App([f, x]) => {
                COST_NONTERMINAL + costs(*f) + costs(*x)
            }
            Lambda::Lam([b]) => {
                COST_NONTERMINAL + costs(*b)
            }
            Lambda::Programs(ps) => {
                ps.iter()
                .map(|p|costs(*p))
                .sum()
            }
        }
    }
}

/// depth of a program. For example a leaf is depth 1.
pub struct ProgramDepth {}
impl CostFunction<Lambda> for ProgramDepth {
    type Cost = i32;
    fn cost<C>(&mut self, enode: &Lambda, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost
    {
        match enode {
            Lambda::Var(_) | Lambda::IVar(_) | Lambda::Prim(_) => 1,
            Lambda::App([f, x]) => {
                1 + std::cmp::max(costs(*f), costs(*x))
            }
            Lambda::Lam([b]) => {
                1 + costs(*b)
            }
            Lambda::Programs(ps) => {
                ps.iter()
                .map(|p|costs(*p))
                .max().unwrap()
            }
        }
    }
}


/// does a child first traversal of the egraph and returns a Vec<Id> in that
/// order. Notably an Id will never show up twice (if it showed up earlier
/// it wont show up again). Assumes no cycles in the EGraph.
fn toplogical_ordering(root: Id, egraph: &EGraph) -> Vec<Id> {
    let mut vec = Vec::new();
    toplogical_ordering_rec(root, egraph, &mut vec);
    vec
}

/// see `toplogical_ordering`
fn toplogical_ordering_rec(root: Id, egraph: &EGraph, vec: &mut Vec<Id>) {
    // assumes no cycles.
    // we require at this point that all eclasses only have ONE enode
    assert!(egraph[root].nodes.len() == 1);
    for child in egraph[root].nodes[0].children(){
        toplogical_ordering_rec(*child, egraph, vec);
    }
    if !vec.contains(&root) {
        // if we're already a child of someone else earlier we dont need to be readded
        vec.push(root);
    }
}

/// cache for shift()
type RecVarModCache = HashMap<(Id,i32),Option<Id>>;

/// types of debruijn index shifts.
/// * ShiftVar(i32) -> increment all free variables by the given amount
/// * ShiftIVar(i32) -> increment all ivars by given amount
/// * TableShiftIVar(Vec<i32>) -> increment ivar #i by the amount given by table[i]
#[derive(Debug,Clone,Eq,PartialEq,Hash)]
enum Shift {
    ShiftVar(i32), // shift $i to be $(i+incr_by)
    ShiftIVar(i32), // shift #i to be #(i+incr_by)
    TableShiftIVar(Vec<i32>), // shift #i to be #(i+table[#i]) ie look up the shift amount in the table
}

/// generates caches for shift()
struct CacheGenerator {
    caches: HashMap<Shift,RecVarModCache>,
    enabled: bool,
}
impl CacheGenerator {
    fn new(enabled: bool) -> CacheGenerator {
        CacheGenerator { caches: Default::default(), enabled: enabled }
    }
    fn get(&mut self, context: &Shift) -> &mut RecVarModCache {
        if !self.enabled {
            // wipe the cache before returning it
            self.caches.insert(context.clone(),Default::default());
         }
        if !self.caches.contains_key(&context) {
            self.caches.insert(context.clone(),Default::default());
        } 
        self.caches.get_mut(&context).unwrap()
    }
}


#[inline(always)]
fn build_shift_table(applam_shift: &AppLam, applam_noshift: &AppLam) -> (Vec<i32>, Vec<Id>) {
    let mut shift_table = vec![]; // just gonna assume nobody wants an arity greater than 10 (for static speed)
    let mut to_remove = vec![];
    let mut shift_rest_by = applam_noshift.inv.arity as i32; // normal amt we shift x by, except if there are merges to be done. If a merge happens all the higher x vars get shifted less, and the specific x var gets shifted a very specific amount
    for (x_idx,xarg) in applam_shift.args.iter().enumerate() {
        if let Some(f_idx) = applam_noshift.args.iter().position(|farg| farg == xarg) {
            // we found a match! $x_idx should map to the same thing as $f_idx.
            // remember, our body currently has $x_idx at the toplevel so now
            // we want to shift it by $(f_idx-x_idx) so that it ends up as f_idx.
            shift_table.push((f_idx as i32) - (x_idx as i32));
            to_remove.push(true);
            shift_rest_by -= 1; // effectively downshifts all the higher args now that this one is gone
        } else {
            // shift fully without merging
            shift_table.push(shift_rest_by);
            to_remove.push(false);
        }
    }

    // remove the args from xargs that we can merge into fargs
    let new_x_applam_args: Vec<Id> = applam_shift.args.iter()
        .zip(to_remove)
        .filter(|(_,remove)| !*remove)
        .map(|(xarg,_)| xarg)
        .cloned().collect();
    
    let mut new_applam_args = applam_noshift.args.clone();
    new_applam_args.extend(new_x_applam_args);
    (shift_table, new_applam_args)
}

/// result of beta_inversions(). This struct feels pretty subject to change, it's a bit
/// of a pain to work with these _of_treenode objects.
struct InversionResult {
    applams_of_treenode: HashMap<Id,Vec<AppLam>>,
    best_inventions_of_treenode: HashMap<Id,NodeCost>
}

/// This is the main workhorse of compression. Takes a child-first ordering of nodes in an EGraph
/// (assumed to be acyclic) and finds all the possible useful inventions up to the given arity.
#[inline(never)] // for flamegraph debugging
fn beta_inversions(
    treenodes: &Vec<Id>,
    max_arity: usize,
    // beam_size: usize,
    no_cache: bool,
    egraph: &mut EGraph
) -> InversionResult {
    // one vector of applams per tree node
    let mut applams_of_treenode: HashMap<Id,Vec<AppLam>> = Default::default();
    let mut best_inventions_of_treenode: HashMap<Id,NodeCost> = Default::default();
    
    let ivar0: Id = egraph.add(Lambda::IVar(0));

    // Caches - these give us considerable speedup (26s -> 18s)
    let caches = &mut CacheGenerator::new(!no_cache);

    for treenode in treenodes.iter() {
        // println!("processing id={}: {}", treenode, extract(*treenode, egraph) );

        // im essentially using the egraph just for its structural hashing rn
        assert!(egraph[*treenode].nodes.len() == 1);
        // clone to appease the borrow checker
        let node = egraph[*treenode].nodes[0].clone();

        // its very very very important that these are all canonical because
        // we treat Id equality as true equality in various cases which is only true when theyre canonical
        debug_assert!(node.children().iter().all(|c| applams_of_treenode[c].iter().all(|applam| applam.is_canonical(egraph))));
        debug_assert!(node.children().iter().all(|c| best_inventions_of_treenode[c].inventionful_cost.keys().all(|inv| inv.is_canonical(egraph))));

        //==================================//
        // *** PROPAGATE/CREATE APPLAMS *** //
        //==================================//
        
        let mut applams: Vec<AppLam> = Vec::new();
        let mut downshifted_applam_args: Vec<(Id,Id)> = Vec::new(); // minor implementation detail
        
        // any node can become the identity applam
        applams.push(AppLam::new(ivar0, vec![*treenode]));

        match node {
            Lambda::IVar(_) => {
                panic!("attempted to abstract an IVar");
            }
            Lambda::Var(_) | Lambda::Prim(_) | Lambda::Programs(_) => {},
            Lambda::App([f,x]) => {
                let ref f_applams = applams_of_treenode[&f];
                let ref x_applams = applams_of_treenode[&x];

                // bubbling from the left:
                // (app f x) == (app (applam body arg) x) => (applam (app body upshift(x)) arg)
                // note no shifting is needed thanks to IVars
                for f_applam in f_applams.iter() {
                    let new_applam_body = egraph.add(Lambda::App([f_applam.inv.body,x]));
                    applams.push(AppLam::new(new_applam_body, f_applam.args.clone()));
                }

                // bubbling from the right:
                // (app f x) == (app f (applam body arg)) => (applam (app upshift(f) body) arg)
                // note no shifting is needed thanks to IVars
                for x_applam in x_applams.iter() {
                    let new_applam_body = egraph.add(Lambda::App([f, x_applam.inv.body]));
                    applams.push(AppLam::new(new_applam_body, x_applam.args.clone()));
                }

                // println!("f_applam x_applam pairwise product size: {} x {} -> {}",f_applams.len(), x_applams.len(), f_applams.len() * x_applams.len());

                for f_applam in f_applams.iter() {
                    for x_applam in x_applams.iter() {
                        // making a higher arity applam out of two diff applams and merging any shared arguments. Higher arity applam looks a bit like:
                        // (app f x) == (app (applam body1 arg1) (applam body2 arg2)) => (applam (app body1 upshift_ivars(body2)) arg1 arg2).
                        //
                        // Note that (applam body arg0 arg1) means arg0 will fill upward refs to $0 and arg1 will fill upward refs to $1
                        // so somewhat confusingly (applam body arg0 arg1) == (app (app (lam (lam body)) arg1) arg0)
                        //
                        // Merging: when f_applam and x_applam have identical args we can merge them
                        // (app f x) == (app (applam body1 arg) (applam body2 arg)) => (applam (app body1 body2) arg)
                        // 
                        // Everything in between: more generically, a pair of applams might have some vars that are shared and
                        // some that are unique to each of them. First we calculate the `overlap` between the two to determine
                        // if the resulting merge will be too high in arity and we should just fail fast. If it's not too high
                        // in arity, then we proceed with a merge. We merge the ivars in `x` into the ivars in `f`, that is `f`
                        // remains completely unchanged while shared arguments in `x` get remapped to point to `f` vars and nonshared
                        // arguments in `x` get upshifted since we're using the lower ivars are the `f` ivars and the higher ones
                        // as the `x` ivars. We accomplish all this by constructing a table that says how much each var should shift by,
                        // and then use a shift(Shift::TableShiftIVar) to perform this remapping.

                        // how many args are shared between `f` and `x`?
                        let overlap: usize = f_applam.args.iter().filter(|farg| x_applam.args.contains(farg)).count();
                        if f_applam.inv.arity + x_applam.inv.arity - overlap > max_arity {
                            continue; // too high in arity
                        }

                        if overlap > 0 {
                            // overlap case - need to merge

                            // x_shift_table[i] tells us how much to shift #i in x_applam.body
                            // (note without merging this would be the arity of f_applam)
                            let (shift_table,new_applam_args) = build_shift_table(&f_applam, &x_applam);
                            let shifted_f_applam_body = shift(f_applam.inv.body, Shift::TableShiftIVar(shift_table), egraph, Some(caches)).unwrap();
                            let new_applam_body = egraph.add(Lambda::App([shifted_f_applam_body, x_applam.inv.body]));

                            applams.push(AppLam::new(new_applam_body, new_applam_args));
                        } else {
                            // no overlap case - no merging, just upshift the x vars and we're done.
                            // We will use the lower indices for f_applam and will upshift x_applam to occupy the higher indices.
                            let shifted_x_applam_body = shift(x_applam.inv.body, Shift::ShiftIVar(f_applam.inv.arity as i32), egraph, Some(caches)).unwrap();

                            let new_applam_body = egraph.add(Lambda::App([f_applam.inv.body,shifted_x_applam_body]));
                            let mut new_applam_args = f_applam.args.clone();
                            new_applam_args.extend(x_applam.args.clone());
                            applams.push(AppLam::new(new_applam_body, new_applam_args));
                        };
                    }
                    
                }

            },
            Lambda::Lam([b]) => {
                let ref b_applams = applams_of_treenode[&b];
                // bubbling up over the lambda:
                // (lam b) == (lam (applam body arg)) => (applam (lam body) downshift(arg))
                // where:
                //  - arg must not have any upward refs to $0 in it since we cant jump over a lambda we point to
                //    > (in the multiarg applam case, none of them can have $0)
                //  - in the pre-ivar era this required a RotateShift which turned out to be a huge speed bottleneck
                //    as it created tons of new nodes in the egraph. This is no longer needed with ivars. No shfiting at all!

                for b_applam in b_applams.iter() {
                    // can't bubble an applam over a lambda if its arg refers to the lambda!
                    if b_applam.args.iter().any(|arg| egraph[*arg].data.free_vars.contains(&0)) {
                        
                    } else {

                    }
                    
                    let mut body = b_applam.inv.body;
                    // downshift the args since the lambda above them moved below them (earlier we made sure none of them had pointers to it)
                    let new_args: Vec<Id> = b_applam.args.iter().enumerate().map(
                        |(i, arg)| {
                            if egraph[*arg].data.free_vars.contains(&0) {
                                body = recursive_var_mod(
                                    |actual_idx, depth, _which_upward_ref, egraph| {
                                        if actual_idx == i as i32 {
                                            let f = egraph.add(Lambda::IVar(actual_idx));
                                            let x = egraph.add(Lambda::Var(depth));
                                            Some(egraph.add(Lambda::App([f, x])))
                                        } else {
                                            Some(egraph.add(Lambda::IVar(actual_idx)))  // i.e. do nothing
                                        }
                                    },
                                    true, // operate on IVars
                                    body, egraph, &mut HashMap::new()
                                ).unwrap();
                                let new_arg = egraph.add(Lambda::Lam([*arg]));
                                if !best_inventions_of_treenode.contains_key(&new_arg) {
                                    let mut cloned = best_inventions_of_treenode[arg].clone();
                                    cloned.inventionless_cost += COST_NONTERMINAL;
                                    // we'll force this arg to not use a toplevel invention at the "lam" node hence the None
                                    cloned.inventionful_cost.iter_mut().for_each(|(_key, val)| {val.0 += COST_NONTERMINAL; val.1 = None});
                                    best_inventions_of_treenode.insert(new_arg,cloned);
                                }
                                new_arg
                            } else {
                                let shifted_arg = shift(*arg, Shift::ShiftVar(-1), egraph, Some(caches)).unwrap();
                                downshifted_applam_args.push((shifted_arg, *arg));
                                shifted_arg
                            }
                        }
                    ).collect();

                    // add the downshifted args to a worklist discussed below. Worklist needed bc of borrow checker.
                    //downshifted_applam_args.extend(new_args.iter().cloned().zip(b_applam.args.iter().cloned())); // (new,old)

                    let new_applam_body = egraph.add(Lambda::Lam([body]));
                    applams.push(AppLam::new(new_applam_body, new_args));
                }
            },
        }

        // let child_applams: Vec<usize> = node.children().iter()
        //     .map(|id| applams_of_treenode[id].len())
        //     .collect();
        // println!("{:?}:\n\t{:?} -> {}\n\t{}",node,child_applams, applams.len(), egraph.total_size());

        // Processing downshifted_applam_args. The following is really just an implementation detail. If it proved
        // to be a bottleneck it could absolutely be done differently.
        //
        // Downshifting args in the "bubbling over lam" case is sortof a big deal because other than this moment we have had the invariant that args
        // are always subtrees of the original program (and are smaller than their parent and thus have already been
        // processed in the child-first traversal). Now we have just created some args that weren't in the original program
        // and thus aren't in our best_inventions list. Luckily if you think a bit you can see that downshifting
        // all free vars in an expression results in something that can be compressed in exactly the same ways as the original
        // expression (assuming we're compressing using a valid invention that doesnt itself have free vars). Please correct
        // me if I'm wrong. But basically TLDR we can just duplicate the list of best inventions and
        // use it for the shifted one. Side note if the shifted guy is already in our best_inventions list then it must already
        // have the right inventions for it and everything so we can skip that.
        // However note the applams can be cloned too but you do need to downshift all their args since their args are all part of a toplevel
        // applam. Then you need to repeat the shifting for them too if need be...luckily arguments must get strictly
        // smaller so it will converge and probably very fast. This is all handled here.
        // todo note that if this were a slowdown you could deal with it differently thru pointers for sure but I think it will be fine
        loop {
            match downshifted_applam_args.pop() {
                Some((new_arg,old_arg)) => {
                    if !best_inventions_of_treenode.contains_key(&new_arg) {
                        let cloned = best_inventions_of_treenode[&old_arg].clone();
                        best_inventions_of_treenode.insert(new_arg,cloned);
                        let new_applams = applams_of_treenode[&old_arg].iter()
                            .map(|applam|{
                                let new_args = applam.args.iter()
                                .map(|arg|{
                                    let arg_mod = shift(*arg, Shift::ShiftVar(-1), egraph, Some(caches)).unwrap();
                                    downshifted_applam_args.push((arg_mod,*arg)); // recursively fix these children
                                    arg_mod
                                }).collect();
                                AppLam::new(applam.inv.body,new_args)
                            }
                            ).collect();
                        applams_of_treenode.insert(new_arg,new_applams);
                    }
                }
                None => break,
            }
        }

        //===================================//
        // *** CALCULATE BEST INVENTIONS *** //
        //===================================//

        let mut best_inventions = NodeCost::new(egraph[*treenode].data.inventionless_cost);

        // For each applam that doesnt have any free variables, we can call it a complete invention
        // and apply it here! Our cost is the basically the cost of our arguments plus a bit extra.
        for applam in applams.iter() {
            if applam.inv.valid_invention(egraph) && applam.inv.body != ivar0 {
                let cost: i32 =
                    COST_TERMINAL // the new primitive for this invention
                    + COST_NONTERMINAL * applam.inv.arity as i32 // the chain of app()s needed to apply the new primitive
                    + applam.args.iter()
                        .map(|id| best_inventions_of_treenode[&id]
                        .cost_under_inv(&applam.inv))
                        .sum::<i32>(); // sum costs of actual args
                best_inventions.new_cost_under_inv(applam.inv, cost, Some(applam.args.clone()));
            }
        }
                
        // inventions that helped our children
        let child_inventions: Vec<Invention> = node.children().iter()
            .map(|id| best_inventions_of_treenode[id].inventionful_cost.keys().cloned())
            .flatten()
            .collect();

        // inventions based on specific node type
        match node {
            Lambda::IVar(_) => {
                panic!("unreachable, should have crashed in previous match statement");
            }
            Lambda::Var(_) | Lambda::Prim(_) => {},
            Lambda::App([f,x]) => {
                let ref f_best_inventions = best_inventions_of_treenode[&f];
                let ref x_best_inventions = best_inventions_of_treenode[&x];
                                
                // costs with inventions as 1 + fcost + xcost. Use inventionless cost as a default.
                // if either fcost or xcost is None (ie infinite)
                for inv in child_inventions {
                    let fcost = f_best_inventions.cost_under_inv(&inv);
                    let xcost = x_best_inventions.cost_under_inv(&inv);
                    let cost = COST_NONTERMINAL+fcost+xcost;
                    best_inventions.new_cost_under_inv(inv, cost, None);
                }
            }
            Lambda::Lam([b]) => {
                // just map +1 over the costs
                let ref b_best_inventions = best_inventions_of_treenode[&b];
                for (inv,cost) in b_best_inventions.inventionful_cost.iter() {
                    best_inventions.new_cost_under_inv(*inv, cost.0 + COST_NONTERMINAL, None);
                }
            }
            Lambda::Programs(roots) => {
                // union together all the useful inventions of diff programs
                
                // count num occurences of each invention
                let mut counts: HashMap<Invention,i32> = child_inventions.iter().map(|i| (*i,0)).collect();
                for inv in child_inventions {
                    counts.insert(inv, counts[&inv] + 1);
                }

                // keep only inventions used by 2+ programs
                // (otherwise it's pretty boring and just abstracts out an entire program)
                let inventions: Vec<Invention> = counts.iter()
                    .filter_map(|(i,c)| if *c > 1 { Some(*i) } else { None }).collect();
                
                for inv in inventions {
                    let cost = roots.iter().map(|root| {
                            best_inventions_of_treenode[root].cost_under_inv(&inv)
                        }).sum();
                    best_inventions.new_cost_under_inv(inv, cost, None);
                }
            }
        }
        // narrow_beam(&mut best_inventions.inventionful_cost, beam_size);

        applams_of_treenode.insert(*treenode, applams);
        best_inventions_of_treenode.insert(*treenode, best_inventions);

    }
    InversionResult {
        applams_of_treenode: applams_of_treenode,
        best_inventions_of_treenode: best_inventions_of_treenode,
    }
}

struct CompressionResult {
    inv: InventionExpr,
    rewritten: Expr,
}

/// takes a (programs ...) expr, returns the best Invention and the Expr rewritten under that invention
fn compression_step(
    programs_expr: &Expr,
    args: &CompressionArgs,
    out_dir: &str,
    new_inv_name: &str,
) -> Option<CompressionResult> {

    // build the egraph. We'll just be using this as a structural hasher we don't use rewrites at all. All eclasses will always only have one node.
    let mut egraph: EGraph = Default::default();
    let programs_id = egraph.add_expr(programs_expr.into());
    egraph.rebuild();

    println!("Initial egraph:\n\t{}\n", egraph_info(&egraph));
    if args.render_initial {
        save(&egraph, "0_programs", &out_dir);
    }

    let tstart = std::time::Instant::now();

    let treenodes: Vec<Id> = toplogical_ordering(programs_id, &egraph);
    // println!("Topological ordering:");
    // treenodes.iter().for_each(|&id| {
    //     println!("id={}: {}", id, extract(id,&egraph));
    // });

    let InversionResult { applams_of_treenode, best_inventions_of_treenode} =
        beta_inversions(
            &treenodes,
            args.max_arity,
            // args.beam_size,
            args.no_cache,
            &mut egraph
        );

    egraph.rebuild(); // hopefully doesnt matter at all anyways if we're just using the egraph as a structurla hasher (not sure if we needed to do this thruout inversions)

    let elapsed = tstart.elapsed().as_millis();

    println!("Inventionless (cost={:?}):\n{}\n",
        egraph[programs_id].data.inventionless_cost,
        extract(programs_id, &egraph)
    );

    let top_invs: Vec<Invention> = best_inventions_of_treenode[&programs_id].top_inventions();

    // print the top args.print_inventions inventions
    for (i,inv) in top_invs.iter().take(args.print_inventions).enumerate() {
        let inv_expr = inv.to_expr(&egraph).body;
        let inv_str = &format!("inv{}_{}",inv.body,inv.arity);
        let rewritten = rewrite_with_inventions(programs_id,&[*inv], &[inv_str], &mut egraph);
        println!("\nInvention {} {:?} (inv_cost={:?}; rewritten_cost={:?}):\n{}\n Rewritten:\n{}",
            i,
            inv,
            inv_expr.cost(),
            rewritten.cost(),
            inv_expr,
            rewritten,
        );
        if args.render_inventions {
            inv_expr.save(&format!("inv{}",i), &out_dir);
        }
    }

    println!("Final egraph: {}",egraph_info(&egraph));

    // print out the largest variable we've seen (useful to make sure our egraph isnt exploding due to Vars)
    for i in 0..1000 {
        if format!("(${})",i).parse::<Pattern<Lambda>>().unwrap().search(&egraph).is_empty() {
            println!("Largest variable: ${}",i-1);
            break;
        }
    }

    println!("Cands useful at top level: {}",top_invs.len());
    println!("Core stuff took: {}ms ***\n", elapsed);

    if args.render_final {
        println!("Rendering final egraph");
        save(&egraph, "final", &out_dir);
    }

    if top_invs.is_empty() {
        return None
    }

    // return the top invention
    let top_inv = top_invs[0].clone();
    let top_inv_expr = top_inv.to_expr(&egraph);
    // let top_inv_rewritten = extract_under_inv(programs_id, top_inv.clone(), new_inv_name, &applams_of_treenode, &best_inventions_of_treenode, &egraph);
    let new_top_inv_rewritten = rewrite_with_inventions(programs_id,&[top_inv.clone()], &[new_inv_name], &mut egraph);
    println!("Expected cost: {}", best_inventions_of_treenode[&programs_id].cost_under_inv(&top_inv));
    // println!("old rewriter cost: {}", top_inv_rewritten.cost());
    println!("new rewriter cost: {}", new_top_inv_rewritten.cost());
    Some(CompressionResult {
        inv: top_inv_expr,
        rewritten: new_top_inv_rewritten,
    })
}

pub fn compression(
    programs_expr: &Expr,
    args: &CompressionArgs,
    out_dir: &str,
) -> (Vec<(InventionExpr,String)>,Expr) {
    let mut rewritten: Expr = programs_expr.clone();
    let mut invs: Vec<(InventionExpr,String)> = Default::default();
    let mut rewrittens: Vec<Expr> = Default::default();
    let mut cost_improvement: Vec<i32> = Default::default();

    let tstart = std::time::Instant::now();

    for i in 0..args.iterations {
        println!("\n=======Iteration {}=======",i);
        let inv_name = format!("inv{}",invs.len());
        if let Some(res) = compression_step(&rewritten, args, out_dir, &inv_name) {
            rewritten = res.rewritten.clone();
            println!("Chose Invention {}: {}\nRewritten (cost={})\n{}", inv_name, res.inv, res.rewritten.cost(), res.rewritten);
            invs.push((res.inv,inv_name));
            rewrittens.push(res.rewritten);
        } else {
            println!("No inventions found at iteration {}",i);
            break;
        }
        
    }

    println!("\n=======Compression Summary=======");
    println!("Found {} inventions", invs.len());
    println!("Cost Improvement: ({:.2}x better) {} -> {}", compression_factor(programs_expr,&rewritten), programs_expr.cost(), rewritten.cost());
    let mut prev_rewritten = programs_expr;
    for i in 0..invs.len() {
        let (inv,inv_name) = &invs[i];
        let rewritten = &rewrittens[i];
        println!("({:.2}x wrt prev; {:.2}x wrt orig) {}: {}", compression_factor(prev_rewritten, &rewritten), compression_factor(programs_expr, &rewritten), inv_name, inv);
        prev_rewritten = &rewritten;
    }
    println!("Time: {}ms", tstart.elapsed().as_millis());


    (invs,rewritten)
}