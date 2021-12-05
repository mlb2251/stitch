use crate::*;
use std::collections::{HashSet,HashMap};
use std::fmt::{self, Formatter, Display};
use clap::Parser;
use std::path::PathBuf;
use std::hash::Hash;

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
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
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
/// track of the best inventions for that node.
#[derive(Debug,Clone)]
struct BestInventions {
    inventionless_cost: i32,
    inventionful_cost: HashMap<Invention, i32>,
}

impl BestInventions {
    fn new(inventionless_cost: i32) -> BestInventions {
        BestInventions {
            inventionless_cost: inventionless_cost,
            inventionful_cost: HashMap::new()
        }
    }
    /// cost under an invention if it's useful for this node, else inventionless cost
    fn cost_under_inv(&self, inv: &Invention) -> i32 {
        self.inventionful_cost.get(inv).cloned().unwrap_or(self.inventionless_cost)
    }
    /// improve the cost using a new invention, or do nothing if we've already seen
    /// a better cost for this invention. Also skip if inventionless cost is better.
    fn new_cost_under_inv(&mut self, inv: Invention, cost:i32) {
        if cost < self.inventionless_cost {
            if !self.inventionful_cost.contains_key(&inv)
               || cost < self.inventionful_cost[&inv]  {
                self.inventionful_cost.insert(inv, cost);
            }
        }
    }
    /// Get the top inventions in decreasing order of cost
    fn top_inventions(&self) -> Vec<Invention> {
        let mut top_inventions: Vec<Invention> = self.inventionful_cost.keys().cloned().collect();
        top_inventions.sort_by(|a,b| self.inventionful_cost[a].cmp(&self.inventionful_cost[b]));
        top_inventions
    }
}

/// convert an egraph Id to an Expr by extracting the expression
fn extract(eclass: Id, egraph: &EGraph) -> Expr {
    let mut extractor = Extractor::new(&egraph, ProgramCost{});
    let (_,p) = extractor.find_best(eclass);
    p.into()
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
    best_inventions_of_treenode: &HashMap<Id,BestInventions>,
    egraph: &EGraph,
) -> Expr {
    let root = egraph.find(root);
    let target_cost:i32 = best_inventions_of_treenode[&root].cost_under_inv(&inv);

    if best_inventions_of_treenode[&root].inventionful_cost.contains_key(&inv)
       && applams_of_treenode[&root].iter().any(|applam| applam.inv == inv) {
        let applam: Vec<AppLam> = applams_of_treenode[&root].iter().filter(|applam| applam.inv == inv).cloned().collect();
        assert!(applam.len() == 1);
        let applam = &applam[0];
        let mut expr = Expr::prim(replace_inv_with.into());
        // wrap the new primitive in app() calls. Note that you pass in the $0 args LAST given how appapplamlam works
        for arg in applam.args.iter().rev() {
            let arg_expr = extract_under_inv(*arg, inv, replace_inv_with, applams_of_treenode, best_inventions_of_treenode, egraph);
            expr = Expr::app(expr,arg_expr);
        }
        assert_eq!(target_cost,expr.cost());
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
    fn merge(&self, to: &mut Data, from: Data) -> bool {
        // we really shouldnt be merging anyone ever rn I think.
        panic!("shouldn't be merging");

        assert_eq!(to.free_vars,from.free_vars);
        assert_eq!(to.free_ivars,from.free_ivars);
        assert_eq!(to.inventionless_cost,from.inventionless_cost);

        // keep the lowest inventionless cost
        // modified |= merge_inventionless(&mut to.inventionless_cost_any, &from.inventionless_cost_any);
        
        false // didnt modify anything
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
                assert!(programs.iter().all(|p| egraph[*p].data.free_vars.is_empty()));
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

/// path down a tree. false = children[0]; true = children[1]
type Zipper = Vec<bool>;
type InvId = Id;


#[derive(Debug,Clone)]
struct AppliedInv1 {
    body: InvId,
    arg: Id, // from original tree modulo shifting
    zipper: Zipper, // useful for performing efficient merges
}

impl AppliedInv1 {
    fn new(body: InvId, arg: Id, zipper: Zipper) -> AppliedInv1 {
        AppliedInv1 { body, arg, zipper }
    }
}

#[derive(Debug,Clone,Eq,PartialEq,Hash)]
struct Inv {
    multiarg_bodies: Vec<InvId>, // like inv1.body
    multiuse_bodies: Vec<(usize,InvId)> // Id is  .body, usize says which `inv` this multiuse is merged with
}
impl Inv {
    fn new(multiarg_bodies: Vec<InvId>, multiuse_bodies: Vec<(usize,InvId)>) -> Inv {
        Inv { multiarg_bodies, multiuse_bodies }
    }
}

#[derive(Debug, Clone)]
struct AppliedInv {
    inv: Inv,
    args: Vec<Id>,
    multiarg_zippers: Vec<Zipper>, // has the unique ones then the multiuse ones
    multiuse_zippers: Vec<Zipper>,
}
impl AppliedInv {
    fn new(inv: Inv, args: Vec<Id>, multiarg_zippers: Vec<Zipper>, multiuse_zippers: Vec<Zipper>) -> AppliedInv {
        AppliedInv { inv, args, multiarg_zippers, multiuse_zippers }
    }
    #[inline]
    fn zippers_interfere(&self, appinv1: &AppliedInv1) -> bool {
        // merge works if inv1.zipper is not a prefix of any of our zippers or vis versa
        // (note that the prefix case is a path towards adding higher order functions, though
        // it would take a good bit of extra work to make that work)
        self.multiarg_zippers.iter()
            .chain(self.multiuse_zippers.iter())
            .any(|z| z.starts_with(&appinv1.zipper) || appinv1.zipper.starts_with(&z)) //||
        // self.multiuses.iter().any(|(inv,_)| inv.zipper.starts_with(&inv1.zipper) || inv1.zipper.startås_with(&inv.zipper))
    }
    #[inline]
    fn merge_multiarg(&self, appinv1: &AppliedInv1, max_arity: usize) -> Option<AppliedInv> {
        if self.args.len() >= max_arity {
            return None; // would exceed arity
        }
        if self.zippers_interfere(&appinv1) {
            return None; // zipper is ancestor of other zipper
        }
        let mut new_appinv = self.clone();
        new_appinv.inv.multiarg_bodies.push(appinv1.body.clone());
        new_appinv.args.push(appinv1.arg.clone());
        new_appinv.multiarg_zippers.push(appinv1.zipper.clone());
        Some(new_appinv)
    }
    #[inline]
    fn merge_multiuse(&self, appinv1: &AppliedInv1) -> Option<Vec<AppliedInv>> {
        if !self.args.iter().any(|arg| *arg == appinv1.arg) {
            return None // no shared arg
        }
        if self.zippers_interfere(&appinv1) {
            return None; // zipper is ancestor of other zipper
        }
        let mut res =  vec![];
        for (i,inv) in self.args.iter().enumerate().filter(|(_,arg)| **arg == appinv1.arg) {
            let mut new_appinv = self.clone();
            new_appinv.inv.multiuse_bodies.push((i,appinv1.body.clone()));
            new_appinv.multiuse_zippers.push(appinv1.zipper.clone());
            res.push(new_appinv);
        }
        Some(res)
    }
    // #[inline]
    // fn cost(&self, costs: &HashMap<Id,i32>) -> i32 {
    //     COST_TERMINAL // the new primitive for this invention
    //     + COST_NONTERMINAL * self.invs.len() as i32 // the chain of app()s needed to apply the new primitive
    //     + self.invs.iter()
    //         .map(|appinv1| costs[&appinv1.arg])
    //         .sum::<i32>(); // sum costs of actual args
    // }
}

// impl PartialEq for Inv1 {
//     fn eq(&self, other: &Inv1) -> bool {
//         self.body == other.body // comparison is just based on the .body
//     }
// }
// impl Eq for Inv1 {}

// #[derive(Debug,Clone)]
// struct Inv {
//     invs: Vec<Inv1>,
// }

// #[derive(Debug,Clone)]
// struct AppliedInv {
//     inv: Inv,
//     args: Vec<Id>, // from original tree modulo shifting
// }

// struct Inv {
//     trees: Vec<Id>, // from original tree but with a single #0
//     origin: Id, // where in the original tree this was created.
//     zippers: Vec<Vec<bool>>, // alternative representation of `.trees`
// }




/// result of beta_inversions(). This struct feels pretty subject to change, it's a bit
/// of a pain to work with these _of_treenode objects.
struct InversionResult {
    applams_of_treenode: HashMap<Id,Vec<AppLam>>,
    best_inventions_of_treenode: HashMap<Id,BestInventions>
}

fn get_treenode_to_roots(roots: &Vec<Id>, egraph: &EGraph) -> HashMap<Id,Vec<Id>> {
    let mut treenode_to_roots: HashMap<Id,Vec<Id>> = Default::default();
    fn get_treenode_to_roots_rec(treenode: Id, root: Id, treenode_to_roots: &mut HashMap<Id,Vec<Id>>, egraph: &EGraph) {
        treenode_to_roots.entry(treenode).or_default().push(root);
        egraph[treenode].nodes[0].children().iter().for_each(|child| get_treenode_to_roots_rec(*child, root, treenode_to_roots, egraph));
    }
    roots.iter().for_each(|root| get_treenode_to_roots_rec(*root, *root, &mut treenode_to_roots, egraph));
    treenode_to_roots
}

fn get_appinv1s(treenodes: &[Id], no_cache:bool, egraph: &mut EGraph) -> (HashMap<Id,Vec<AppliedInv1>>, HashMap<Id,Id>) {
    let mut all_appinv1s: HashMap<Id,Vec<AppliedInv1>> = Default::default();
    let identity_body = egraph.add(Lambda::IVar(0));
    let caches = &mut CacheGenerator::new(!no_cache);
    
    // keys are shifted treenodes values are original treenodes. Useful since shifted ones can use same inventions as originals
    let mut shifted_treenodes: HashMap<Id,Id> = Default::default();

    for treenode in treenodes.iter() {
        // println!("processing id={}: {}", treenode, extract(*treenode, egraph) );

        // im essentially using the egraph just for its structural hashing rn
        assert!(egraph[*treenode].nodes.len() == 1);
        // clone to appease the borrow checker
        let node = egraph[*treenode].nodes[0].clone();

        // its very very very important that these are all canonical because
        // we treat Id equality as true equality in various cases which is only true when theyre canonical
        // debug_assert!(node.children().iter().all(|c| applams_of_treenode[c].iter().all(|applam| applam.is_canonical(egraph))));
        // debug_assert!(node.children().iter().all(|c| best_inventions_of_treenode[c].inventionful_cost.keys().all(|inv| inv.is_canonical(egraph))));

        //==================================//
        // *** PROPAGATE/CREATE APPLAMS *** //
        //==================================//
        let mut appinv1s: Vec<AppliedInv1> = Default::default();
        
        // any node can become the identity
        appinv1s.push(AppliedInv1::new(identity_body.clone(), *treenode, vec![]));

        match node {
            Lambda::IVar(_) => {
                panic!("attempted to abstract an IVar");
            }
            Lambda::Var(_) | Lambda::Prim(_) | Lambda::Programs(_) => {},
            Lambda::App([f,x]) => {
                let ref f_appinv1s = all_appinv1s[&f];
                let ref x_appinv1s = all_appinv1s[&x];

                // bubbling from the left:
                // (app f x) == (app (appinv1 body arg) x) => (appinv1 (app body upshift(x)) arg)
                // note no shifting is needed thanks to IVars
                for f_appinv1 in f_appinv1s.iter() {
                    let body = egraph.add(Lambda::App([f_appinv1.body,x]));
                    let mut zipper = vec![false];
                    zipper.extend(f_appinv1.zipper.clone());
                    appinv1s.push(AppliedInv1::new(body, f_appinv1.arg, zipper));
                }

                // bubbling from the right:
                // (app f x) == (app f (appinv1 body arg)) => (appinv1 (app upshift(f) body) arg)
                // note no shifting is needed thanks to IVars
                for x_appinv1 in x_appinv1s.iter() {
                    let body = egraph.add(Lambda::App([x,x_appinv1.body]));
                    let mut zipper = vec![true];
                    zipper.extend(x_appinv1.zipper.clone());
                    appinv1s.push(AppliedInv1::new(body, x_appinv1.arg, zipper));
                }
            },
            Lambda::Lam([b]) => {
                let ref b_appinv1s = all_appinv1s[&b];
                // bubbling up over the lambda:
                // (lam b) == (lam (appinv1 body arg)) => (appinv1 (lam body) downshift(arg))
                // where:
                //  - arg must not have any upward refs to $0 in it since we cant jump over a lambda we point to
                //    > (in the multiarg appinv1 case, none of them can have $0)
                //  - in the pre-ivar era this required a RotateShift which turned out to be a huge speed bottleneck
                //    as it created tons of new nodes in the egraph. This is no longer needed with ivars. No shfiting at lal!

                for b_appinv1 in b_appinv1s.iter() {
                    // can't bubble an appinv1 over a lambda if its arg refers to the lambda!
                    if egraph[b_appinv1.arg].data.free_vars.contains(&0) {
                        continue;
                    }
                    
                    // downshift the args since the lambda above them moved below them (earlier we made sure none of them had pointers to it)
                    let new_arg: Id = shift(b_appinv1.arg, Shift::ShiftVar(-1), egraph, Some(caches)).unwrap();

                    // to keep track of the fact that this shifted treenode can use the same inventions as the original
                    shifted_treenodes.insert(new_arg, b_appinv1.arg);

                    let body = egraph.add(Lambda::Lam([b_appinv1.body]));
                    let mut zipper = vec![false];
                    zipper.extend(b_appinv1.zipper.clone());
                    appinv1s.push(AppliedInv1::new(body, new_arg, zipper));
                }
            },
        }

        all_appinv1s.insert(*treenode, appinv1s);
    }

    // remove any appinv1s that have free variables & remove identity functions
    all_appinv1s = all_appinv1s.into_iter()
        .map(|(treenode,appinv1s)|{
            let new_appinv1s: Vec<AppliedInv1> = appinv1s.into_iter()
                .filter(|appinv1|
                    !egraph[appinv1.body].data.free_vars.is_empty() && appinv1.body != identity_body
                ).collect();
            (treenode,new_appinv1s)
        })
        .collect();

    (all_appinv1s,shifted_treenodes)
}


/// This is the main workhorse of compression. Takes a child-first ordering of nodes in an EGraph
/// (assumed to be acyclic) and finds all the possible useful inventions up to the given arity.
#[inline(never)] // for flamegraph debugging
fn beta_inversions(
    programs_node: Id,
    max_arity: usize,
    // beam_size: usize,
    no_cache: bool,
    egraph: &mut EGraph
) -> InversionResult {

    let treenodes: Vec<Id> = toplogical_ordering(programs_node,egraph);
    let roots: Vec<Id> = egraph[programs_node].nodes[0].children().iter().cloned().collect();

    // lets you lookup which roots a treenode is a descendent of
    let mut treenode_to_roots: HashMap<Id,Vec<Id>> = get_treenode_to_roots(&roots, egraph);
    treenode_to_roots.insert(programs_node,vec![]); // Programs node has no roots

    let tstart = std::time::Instant::now();
    let (mut all_appinv1s, shifted_treenodes) = get_appinv1s(&treenodes, no_cache, egraph);
    println!("get_appinv1s: {:?}ms", tstart.elapsed().as_millis());

    // from inv1 body to set of roots that it's used under
    let tstart = std::time::Instant::now();
    let mut usages: HashMap<Id,HashSet<Id>> = Default::default();
    for (treenode,appinv1s) in all_appinv1s.iter() {
        for appinv1 in appinv1s.iter() {
            usages.entry(appinv1.body).or_default().extend(treenode_to_roots[treenode].clone());
        }
    }

    println!("{} invs", usages.len());

    // prune down to ones that are used in multiple places
    let mut invs: Vec<Id> = usages.iter()
        .filter_map(|(inv,usages)| if usages.len() > 1 {Some(inv)} else {None})
        .cloned().collect();
    invs.sort();

    all_appinv1s = all_appinv1s.into_iter()
        .map(|(treenode,appinv1s)|{
            let mut new_appinv1s: Vec<AppliedInv1> = appinv1s.into_iter()
                .filter(|appinv1|
                    invs.contains(&appinv1.body)
                ).collect();
            new_appinv1s.sort_by(|a,b| a.body.cmp(&b.body));
            (treenode,new_appinv1s)
        })
        .collect();

    println!("filtered out single use: {:?}ms", tstart.elapsed().as_millis());
    println!("{} invs", invs.len());

    let tstart = std::time::Instant::now();

    let mut all_derived_invs: HashMap<Id,Vec<AppliedInv>> = Default::default();

    for base_inv in invs.iter() {
        for node in treenodes.iter() {
            let appinv1s = &all_appinv1s[&node];
            let idx = match appinv1s.binary_search_by_key(base_inv, |appinv1| appinv1.body) {
                Ok(idx) => idx,
                Err(_) => continue,
            };

            let base_appinv1: &AppliedInv1 = &all_appinv1s[&node][idx];

            // invs built from the original iteratively. The usize is to track the largest Id used so far (to make it easy)
            let mut derived_invs: Vec<(AppliedInv,usize)> = vec![(AppliedInv::new(
                Inv::new(vec![*base_inv], vec![]),
                vec![base_appinv1.arg], // args
                vec![base_appinv1.zipper.clone()], // multiarg zipper
                vec![], // multiuse zipper
            ),idx)];
            let mut offset: usize = 0;
            while offset < derived_invs.len() {
                let skip_to: usize = derived_invs[offset].1;
                for (i,appinv1) in appinv1s[skip_to+1..].iter().enumerate() {
                    debug_assert!(appinv1.body > *base_inv, "wasnt sorted!!");
                    if let Some(new_derived_inv) = derived_invs[offset].0.merge_multiarg(appinv1,max_arity) {
                        derived_invs.push((new_derived_inv,skip_to+1+i));
                    }
                    if let Some(new_derived_invs) = derived_invs[offset].0.merge_multiuse(appinv1) {
                        derived_invs.extend(new_derived_invs.into_iter().map(|inv|(inv,skip_to+1+i)));
                    }
                }
                offset += 1;
            }
            all_derived_invs.entry(*node).or_default().extend(derived_invs.into_iter().map(|(inv,_)| inv));
        }
    }

    println!("derived all inventions: {:?}ms", tstart.elapsed().as_millis());


    // usage counts
    let tstart = std::time::Instant::now();
    let mut usages: HashMap<Inv,HashSet<Id>> = Default::default();
    for (treenode,derived_invs) in all_derived_invs.iter() {
        for derived_inv in derived_invs.iter() {
            usages.entry(derived_inv.inv.clone()).or_default().extend(treenode_to_roots[treenode].clone());
        }
    }

    println!("{} derived invs before pruning", usages.len());

    // prune down to ones that arent used in multiple places
    let invs: Vec<Inv> = usages.iter()
        .filter_map(|(inv,usages)| if usages.len() > 1 {Some(inv)} else {None})
        .cloned().collect();
    
    let to_remove: Vec<Inv> = usages.iter()
        .filter_map(|(inv,usages)| if usages.len() == 1 {Some(inv)} else {None})
        .cloned().collect();
    for (_, derived_invs) in all_derived_invs.iter_mut() {
        derived_invs.retain(|derived_inv| !to_remove.contains(&derived_inv.inv));
    }

    println!("filtered out single use derived: {:?}ms", tstart.elapsed().as_millis());
    println!("{} derived invs", invs.len());




    // all_derived_invs = all_derived_invs.into_iter()
    //     .map(|(treenode,derived_invs)|{
    //         let mut new_derived_invs: Vec<AppliedInv> = derived_invs.into_iter()
    //             .filter(|appinv1|
    //                 invs.contains(&appinv1.to_inv())
    //             ).collect();
    //         (treenode,new_derived_invs)
    //     })
    //     .collect();





    
    // for treenode in treenodes.iter() {
    //     let applams = &applams_of_treenode[treenode];
    //     let valid = applams.iter().filter(|applam| applam.inv.valid_invention(egraph)).collect::<Vec<_>>();
    //     let best_inventions = &best_inventions_of_treenode[treenode];
    //     let expr = extract(*treenode, egraph);
    //     println!("id: {}, cost: {}, depth: {}  applams: {} valid: {}", treenode, expr.cost(), expr.depth(), applams.len(), valid.len());
    // }
    unimplemented!()
}


struct NodeCost {
    inventionless_cost: i32,
    inventionful_cost: HashMap<Inv, i32>
}
impl NodeCost {
    fn new(inventionless_cost: i32) -> Self {
        NodeCost {
            inventionless_cost,
            inventionful_cost: Default::default()
        }
    }
    /// improve the cost using a new invention, or do nothing if we've already seen
    /// a better cost for this invention. Also skip if inventionless cost is better.
    fn new_cost_under_inv(&mut self, inv: &Inv, cost:i32) {
        if cost < self.inventionless_cost {
            if !self.inventionful_cost.contains_key(inv)
               || cost < self.inventionful_cost[&inv]  {
                self.inventionful_cost.insert(inv.clone(), cost);
            }
        }
    }
    /// cost under an invention if it's useful for this node, else inventionless cost
    fn cost_under_inv(&self, inv: &Inv) -> i32 {
        self.inventionful_cost.get(inv).cloned().unwrap_or(self.inventionless_cost)
    }
}

struct NodeCosts {
    costs: HashMap<Id,NodeCost>,
    remap: HashMap<Id,Id>
}
impl NodeCosts {
    fn new(treenodes: &[Id], remap: HashMap<Id,Id>, egraph: &EGraph) -> Self {
        let costs = treenodes.iter().map(|node| (*node,NodeCost::new(egraph[*node].data.inventionless_cost))).collect();
        NodeCosts { costs, remap }
    }
    fn cost_under_inv(&self, node: Id, inv: &Inv) -> i32 {
        let remapped_node = if self.costs.contains_key(&node) {node} else {self.remap[&node]};
        self.costs[&remapped_node].cost_under_inv(inv)
    }
    fn new_cost_under_inv(&mut self, node: Id, inv: &Inv, cost:i32) {
        self.costs.get_mut(&node).unwrap().new_cost_under_inv(inv, cost);
    }
    fn useful_invs(&self, node: Id) -> Vec<Inv> {
        let remapped_node = if self.costs.contains_key(&node) {node} else {self.remap[&node]};
        self.costs[&remapped_node].inventionful_cost.keys().cloned().collect()
    }
}


fn best_inventions(invs_of_node: &HashMap<Id,Vec<AppliedInv>>, remap: &HashMap<Id,Id>, programs_node: Id, egraph: &EGraph) {
    let treenodes: Vec<Id> = toplogical_ordering(programs_node,egraph);

    // first get inventionless costs
    let mut node_costs = NodeCosts::new(&treenodes, remap.clone(), egraph);

    for node in treenodes.iter() {
        // using an invention at this node
        for appinv in invs_of_node[node].iter() {
            let cost: i32 =
                COST_TERMINAL // the new primitive for this invention
                + COST_NONTERMINAL * appinv.args.len() as i32 // the chain of app()s needed to apply the new primitive
                + appinv.args.iter()
                    .map(|arg| node_costs.cost_under_inv(*node, &appinv.inv))
                    .sum::<i32>(); // sum costs of actual args
            node_costs.new_cost_under_inv(*node, &appinv.inv, cost);
        }

        let enode = egraph[*node].nodes[0].clone();

        // inventions that helped our children
        let child_inventions: Vec<Inv> = enode.children().iter()
            .map(|id| node_costs.useful_invs(*id))
            .flatten()
            .collect();
        
        match enode {
            Lambda::IVar(_) => { panic!("unreachable"); }
            Lambda::Var(_) | Lambda::Prim(_) => {},
            Lambda::App([f,x]) => {                                
                for inv in child_inventions {
                    let fcost = node_costs.cost_under_inv(f, &inv);
                    let xcost = node_costs.cost_under_inv(x, &inv);
                    let cost = COST_NONTERMINAL+fcost+xcost;
                    node_costs.new_cost_under_inv(*node, &inv, cost);
                }
            }
            Lambda::Lam([b]) => {
                // just map +1 over the costs
                for inv in child_inventions {
                    let bcost = node_costs.cost_under_inv(b, &inv);
                    let cost = COST_NONTERMINAL+bcost;
                    node_costs.new_cost_under_inv(*node, &inv, cost);
                }
            }
            Lambda::Programs(roots) => {
                // union together all the useful inventions of diff programs
                
                // count num occurences of each invention
                let mut counts: HashMap<Inv,i32> = child_inventions.iter().cloned().map(|i| (i,0)).collect();
                for inv in child_inventions {
                    counts.insert(inv.clone(), counts[&inv] + 1);
                }

                // keep only inventions used by 2+ programs
                let inventions: Vec<Inv> = counts.into_iter()
                    .filter_map(|(i,c)| if c > 1 { Some(i) } else { None }).collect();
                
                for inv in inventions {
                    let cost = roots.iter().map(|root| {
                            node_costs.cost_under_inv(*root, &inv)
                        }).sum();
                    node_costs.new_cost_under_inv(*node, &inv, cost);
                }
            }
        }

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
            programs_id,
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
        let rewritten = extract_under_inv(programs_id, *inv, inv_str, &applams_of_treenode, &best_inventions_of_treenode, &egraph);
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
        if search(format!("(${})",i).as_str(),&egraph).is_empty() {
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
    let top_inv_rewritten = extract_under_inv(programs_id, top_inv.clone(), new_inv_name, &applams_of_treenode, &best_inventions_of_treenode, &egraph);
    Some(CompressionResult {
        inv: top_inv_expr,
        rewritten: top_inv_rewritten,
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