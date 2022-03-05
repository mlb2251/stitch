use crate::*;
use std::collections::{HashSet,HashMap,BinaryHeap};
use std::fmt::{self, Formatter, Display};
use std::hash::Hash;
use itertools::Itertools;
use extraction::extract;
use serde_json::json;
use rand::seq::SliceRandom;


use clap::Parser;
use serde::Serialize;
use std::path::PathBuf;


/// The analysis data associated with each Lambda node
#[derive(Debug)]
pub struct Data {
    pub free_vars: HashSet<i32>, // $i vars. For example (lam $2) has free_vars = {1}.
    pub free_ivars: HashSet<i32>, // #i ivars
    pub inventionless_cost: i32,
}

/// At the end of the day we convert our Inventions into InventionExprs to make
/// them standalone without needing to carry the EGraph around to figure out what
/// the body Id points to.
#[derive(Debug, Clone)]
pub struct Invention {
    pub body: Expr, // invention body (not wrapped in lambdas)
    pub arity: usize,
    pub name: String,
}
impl Invention {
    pub fn new(body: Expr, arity: usize, name: &str) -> Self {
        Self { body, arity, name: String::from(name) }
    }
    /// replace any #i with args[i], returning a new expression
    pub fn apply(&self, args: &[Expr]) -> Expr {
        assert_eq!(args.len(), self.arity);
        let map: HashMap<i32, Expr> = args.iter().enumerate().map(|(i,e)| (i as i32, e.clone())).collect();
        ivar_replace(&self.body, self.body.root(), &map)
    }
}

impl Display for Invention {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "[{} arity={}: {}]", self.name, self.arity, self.body)
    }
}


/// Does debruijn index shifting of a subtree, incrementing all Vars by the given amount
#[inline] // useful to inline since callsite can usually tell which Shift type is happening allowing further optimization
pub fn shift(e: Id, incr_by: i32, egraph: &mut EGraph, cache: &mut Option<RecVarModCache>) -> Option<Id> {
    let empty = &mut RecVarModCache::new();
    let seen: &mut RecVarModCache = cache.as_mut().unwrap_or(empty);

    recursive_var_mod(
        |actual_idx, _depth, _which_upward_ref, egraph| {
            Some(egraph.add(Lambda::Var(actual_idx + incr_by)))
        },
        false, // operate on Vars not IVars
        e,egraph,seen
    )
}

/// A node in an ZPath
/// Ord: Func < Body < Arg
#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
enum ZNode {
    // * order of variants here is important because the derived Ord will use it
    Func, // zipper went into the function, so Id is the arg
    Body, 
    Arg, // zipper went into the arg, so Id is the function
}

/// "zipper id" each unique zipper gets referred to by its zipper id
type ZId = usize;
/// "zipper path" this is a path like "Func Body Arg Body Arg Arg"
type ZPath = Vec<ZNode>;

/// A Zipper is a single-argument single-use invention, so it's a subtree from the
/// original program with exactly one invention variable #0 used in exactly one place.
/// 
/// A Zipper has a `.path` specifying the path it takes through an expression
/// eg "Func Body Arg Body Arg Arg", along with a `.left` and `.right` specifying
/// the off-zipper elements. For example when the zipper goes to the left (ie path has
/// an `Arg`) the off-zipper element is whatever the Function was in the `app(func,arg)`.
/// This is stored as a Some(Id) referencing the subtree from the original program. Nones are
/// used in cases where an off element doesnt exist. The lengths of all 3 of these fields are the same.
/// 
/// Illustration of the 3 vectors side by side:
/// ```
/// left     | path | right
/// -------------------
/// None     | Func |  Some(23)
/// None     | Body |  None
/// Some(33) | Arg  |  None
/// None     | Func |  Some(45)
/// ```
#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
struct Zipper {
    path: ZPath,
    left: Vec<Option<Id>>,
    right: Vec<Option<Id>>,
}

/// A zipper (single-arg single-use invention) applied to an argument
#[derive(Debug,Clone, Eq, PartialEq, Hash)]
struct AppZipper {
    zipper: Zipper,
    arg: Id,
}

/// a zid referencing a specific ZPath and a #i index
#[derive(Debug,Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
struct LabelledZId {
    zid: ZId,
    ivar: usize // which #i argument this is, which also corresponds to args[i] ofc
}

/// A Zipper Tuple. This is a multiarg multiuse invention consisting of a list of zippers. This is the core data structure
/// representing a partially or completely constructed invention. Zippers get merged into it, extending
/// the `elems` field by one, `divergence_idxs` by one, and either `multiarg` or `multiuse` by one depending
/// on whether the newly added zipper is reusing an existing argument. Only zippers that are larger than the rightmost zipper can be
/// added to a ztuple
#[derive(Debug,Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
struct ZTuple {
    elems: Vec<LabelledZId>, // list of zids labelled with #is
    divergence_idxs: Vec<usize>, // locations where the zippers diverge from each other (lengths is 1 less than the number of zippers)
    multiarg: Vec<ZId>, // len=arity, gives the first zid added for each arg
    multiuse: Vec<ZId>, // gives the 2nd and onward zids added for each arg (not the first, thats in multiarg)
    arity: usize,
}

/// A partially constructed invention (`.ztuple`) along with the nodes where it is usable (`.nodes`),
/// the utility of the concrete part of the invention for a single usage (`.left_utility`) and an upper
/// bound on the total utility over all usages (`.utility_upper_bound`),
#[derive(Debug,Clone, Eq, PartialEq, PartialOrd, Ord)]
struct WorklistItem {
    ztuple: ZTuple,
    nodes: Vec<Id>, // nodes in the group
    left_utility: i32, // utility of a single usage
    utility_upper_bound: i32, // upper bound utility over all usages
}

/// A completely finished invention (`.ztuple`) along with the nodes where it is used (`.nodes`)
/// and the total utility of it over all usages.
#[derive(Debug,Clone)]
pub struct FinishedItem {
    ztuple: ZTuple,
    nodes: Vec<Id>, // nodes in the group
    utility: i32,
}

/// The heap item used for heap-based worklists
#[derive(Debug,Clone, Eq, PartialEq, PartialOrd, Ord)]
struct HeapItem {
    key: i32,
    item: WorklistItem,
}

impl WorklistItem {
    fn new(ztuple: ZTuple, nodes: Vec<Id>, left_utility: i32, utility_upper_bound: i32) -> WorklistItem {
        WorklistItem { ztuple: ztuple, nodes: nodes, left_utility: left_utility, utility_upper_bound: utility_upper_bound }
    }
}

impl FinishedItem {
    fn new(ztuple: ZTuple, nodes: Vec<Id>, utility: i32) -> FinishedItem {
        FinishedItem { ztuple, nodes, utility }
    }
    fn to_invention(&self, name: &str, appzipper_of_node_zid: &HashMap<(Id,ZId),AppZipper>, egraph: &EGraph ) -> Invention {
        Invention::new(self.ztuple.to_expr(self.nodes[0], appzipper_of_node_zid, egraph), self.ztuple.arity, name)
    }
}

impl HeapItem {
    fn new(item: WorklistItem) -> HeapItem {
        HeapItem { key: item.ztuple.elems.last().unwrap().zid as i32, item: item }
    }
}

impl Zipper {
    fn new(path: ZPath, left: Vec<Option<Id>>, right: Vec<Option<Id>> ) -> Zipper {
        Zipper { path, left, right }
    }
}

impl AppZipper {
    fn new(zipper: Zipper, arg: Id) -> AppZipper {
        AppZipper { zipper: zipper, arg: arg }
    }
    /// clone this applied single-arg single-use invention and extend the zipper by 1 at the top
    /// of the zipper. This is used when constructing the appzippers in a bottom up way
    #[inline]
    fn clone_prepend(&self, new: ZNode, id: Option<Id>) -> AppZipper {
        let mut appzipper: AppZipper = self.clone();
        match new {
            ZNode::Func => {
                assert!(id.is_some());
                appzipper.zipper.left.insert(0,None);
                appzipper.zipper.right.insert(0,id);
            },
            ZNode::Arg => {
                assert!(id.is_some());
                appzipper.zipper.left.insert(0,id);
                appzipper.zipper.right.insert(0,None);
            },
            ZNode::Body => {
                assert!(id.is_none());
                appzipper.zipper.left.insert(0,None);
                appzipper.zipper.right.insert(0,None);        
            },
        }
        appzipper.zipper.path.insert(0,new);
        appzipper
    }
}

impl LabelledZId {
    fn new(zid: ZId, ivar: usize) -> LabelledZId {
        LabelledZId { zid: zid, ivar: ivar }
    }
}

impl ZTuple {
    /// make an arity 0 ztuple 
    fn empty() -> ZTuple {
        ZTuple { elems: vec![], divergence_idxs: vec![], multiarg: vec![], multiuse: vec![], arity: 0}
    }
    /// make a new single-zipper ztuple
    fn single(zid: ZId) -> ZTuple {
        ZTuple { elems: vec![LabelledZId::new(zid, 0)], divergence_idxs: vec![], multiarg: vec![zid], multiuse: vec![], arity: 1 }
    }
    /// extend ztuple, returning a new ztuple with one extra argument (original is unchanged)
    fn extend(&self, elem: LabelledZId, div_idx: usize, is_multiuse: bool) -> ZTuple {
        let mut res = self.clone();
        res.divergence_idxs.push(div_idx);
        if is_multiuse {
            res.multiuse.push(elem.zid);
        } else {
            res.multiarg.push(elem.zid);
            res.arity += 1;
        }
        res.elems.push(elem);
        res
    }
    /// convert a ztuple to an Expr. This is for extracting out the final complete inventions at the very end so that
    /// there are no more ZIds or Ids and everything is self contained without references to shared data structures.
    fn to_expr(&self, node: Id, appzipper_of_node_zid: &HashMap<(Id,ZId),AppZipper>, egraph: &EGraph) -> Expr {
        if self.elems.is_empty() {
            return extract(node, egraph);  // arity 0
        }

        let mut elem_idx: usize = 0;
        let mut zipper: &Zipper = &appzipper_of_node_zid[&(node,self.elems[elem_idx].zid)].zipper;
        let mut depth: usize = zipper.path.len() - 1;
        let mut expr = Expr::ivar(self.elems[elem_idx].ivar as i32);
        let mut diverged: Vec<(usize,Expr)> = vec![];

        // we do this by a loop where we start at the bottom of the leftmost zipper and gradually extract the Expr bottom up,
        // and whenever we hit a divergence point we store the Expr and jump to the bottom of the next zipper and repeat, being
        // careful to pop and merge our stored expressions as we pass the divergence point a second time from the righthand side.
        loop {
            // encounter divergence point to our right
            if elem_idx < self.divergence_idxs.len() && depth == self.divergence_idxs[elem_idx] {
                // we should diverge to the right
                assert_eq!(zipper.path[depth], ZNode::Func);
                diverged.push((depth,expr));
                elem_idx += 1;
                zipper = &appzipper_of_node_zid[&(node,self.elems[elem_idx].zid)].zipper;
                depth = zipper.path.len() - 1;
                expr = Expr::ivar(self.elems[elem_idx].ivar as i32);
                continue;
            }
            // pass a divergence point to our left that we stored something for
            if !diverged.is_empty() && depth == diverged.last().unwrap().0 {
                // we should ignore our normal Some(f) and instead use the stored diverged expr
                assert_eq!(zipper.path[depth], ZNode::Arg);
                expr = Expr::app(diverged.pop().unwrap().1, expr);
                if depth == 0 { break }
                depth -= 1;
                continue;
            }

            // normal step upward by 1
            match (&zipper.path[depth], &zipper.left[depth], &zipper.right[depth]) {
                (ZNode::Arg, Some(f), None) => { expr = Expr::app(extract(*f,egraph), expr); },
                (ZNode::Func, None, Some(x)) => { expr = Expr::app(expr, extract(*x,egraph)); },
                (ZNode::Body, None, None) => { expr = Expr::lam(expr); },
                _ => panic!("malformed zipper"),
            }
            if depth == 0 { break }
            depth -= 1;
        }

        expr
    }
}

/// Construct all single-argument single-usage inventions in a bottom up manner. This returns around O(N^2) inventions
/// since it's any O(N) choice of a parent to be the root of the invention, and any choice of a single descendent of that
/// parent to be the abstracted #0. Returns a map from nodes to the list of single-arg single-use inventions that can be
/// used at that node.
fn get_appzippers(treenodes: &[Id], no_cache:bool, egraph: &mut EGraph) -> HashMap<Id,Vec<AppZipper>> {
    let mut all_appzippers: HashMap<Id,Vec<AppZipper>> = Default::default();
    let cache: &mut Option<RecVarModCache> = &mut if no_cache { None } else { Some(HashMap::new()) };
    
    for treenode in treenodes.iter() {
        // println!("processing id={}: {}", treenode, extract(*treenode, egraph) );

        // im essentially using the egraph just for its structural hashing rn
        assert!(egraph[*treenode].nodes.len() == 1);
        // clone to appease the borrow checker
        let node = egraph[*treenode].nodes[0].clone();

        //==================================//
        // *** PROPAGATE/CREATE APPLAMS *** //
        //==================================//
        let mut appzippers: Vec<AppZipper> = vec![];
        
        // any node can become the identity function (the empty zipper)
        appzippers.push(AppZipper::new(Zipper::new(vec![],vec![],vec![]), *treenode));

        match node {
            Lambda::IVar(_) => { panic!("attempted to abstract an IVar") }
            Lambda::Var(_) | Lambda::Prim(_) | Lambda::Programs(_) => {},
            Lambda::App([f,x]) => {
                let ref f_appzippers = all_appzippers[&f];
                let ref x_appzippers = all_appzippers[&x];

                // bubbling from the left:
                // (app f x) == (app (appzipper body arg) x) => (appzipper (app body upshift(x)) arg)
                // note no shifting is needed thanks to IVars
                for f_appzipper in f_appzippers.iter() {
                    // bubble out of function so zipper should point left so Func
                    let new: AppZipper = f_appzipper.clone_prepend(ZNode::Func,Some(x));
                    appzippers.push(new);
                }

                // bubbling from the right:
                // (app f x) == (app f (appzipper body arg)) => (appzipper (app upshift(f) body) arg)
                // note no shifting is needed thanks to IVars
                for x_appzipper in x_appzippers.iter() {
                    // bubble out of arg so zipper should point right so Arg
                    let new: AppZipper = x_appzipper.clone_prepend(ZNode::Arg,Some(f));
                    appzippers.push(new);
                }
            },
            Lambda::Lam([b]) => {
                let ref b_appzippers = all_appzippers[&b];
                // bubbling up over the lambda:
                // (lam b) == (lam (appzipper body arg)) => (appzipper (lam body) downshift(arg))
                // where:
                //  - arg must not have any upward refs to $0 in it   since we cant jump over a lambda we point to
                //    > (in the multiarg appzipper case, none of them can have $0)
                //  - in the pre-ivar era this required a RotateShift which turned out to be a huge speed bottleneck
                //    as it created tons of new nodes in the egraph. This is no longer needed with ivars. No shfiting at lal!

                for b_appzipper in b_appzippers.iter() {
                    // can't bubble an appzipper over a lambda if its arg refers to the lambda!
                    // todo make it handle the threading case i figured out with theo
                    if egraph[b_appzipper.arg].data.free_vars.contains(&0) {
                        continue;
                    }

                    let mut new: AppZipper = b_appzipper.clone_prepend(ZNode::Body,None);
                    
                    // downshift the args since the lambda above them moved below them (earlier we made sure none of them had pointers to it)

                    if egraph[b_appzipper.arg].data.free_vars.contains(&0) {
                        // context threading
                        // todo currently this branch will never be taken thanks to the `continue;` above. However when you do remove that continue
                        // todo you want to modify this to also change the zipper in some way to indicate that this is context threaded.
                        new.arg = egraph.add(Lambda::Lam([b_appzipper.arg]));
                    } else {
                        // no threading
                        new.arg = shift(b_appzipper.arg, -1, egraph, cache).unwrap();
                    }
                    // println!("Bubbled over lam:\n\t{}\n{}", extract(*treenode,egraph), new.to_string(egraph));
                    appzippers.push(new);
                }
            },
        }
        all_appzippers.insert(*treenode, appzippers);
    }

    // remove all the identity functions.
    // note that we must be very careful pruning here. Most pruning isnt allowed, for example you cant prune things
    // that have free variables out bc if those free vars are on the leading edge you could still merge them away later
    all_appzippers.iter_mut().for_each(|(_,appzippers)| {
        appzippers.retain(|appzipper| !appzipper.zipper.path.is_empty());
    });

    all_appzippers
}

/// utility of a fragment of a zipper, specifically a left edge (the left/right
/// distinction is just so we can include the nonterminal cost in the left edge)
#[inline]
fn left_edge_utility(edge: &[Option<Id>], egraph: &EGraph) -> i32 {
    edge.len() as i32 * COST_NONTERMINAL + // there is 1 nonterminal used at each node of the zipper of course (it's either an App or a Lam)
    edge.iter().filter_map(|option_id|
        option_id.map(|id| egraph[id].data.inventionless_cost)).sum::<i32>()
}
/// utility of the right a fragment of a zipper
#[inline]
fn right_edge_utility(edge: &[Option<Id>], egraph: &EGraph) -> i32 {
    edge.iter().filter_map(|option_id|
        option_id.map(|id| egraph[id].data.inventionless_cost)).sum::<i32>()
}

/// check if the edge has free variables
#[inline]
fn edge_has_free_vars(edge: &[Option<Id>], path: &[ZNode], mut depth: i32, egraph: &EGraph) -> bool {
    // return false;
    debug_assert_eq!(edge.len(), path.len());
    for (edge,node) in edge.iter().zip(path.iter()) {
        if *node == ZNode::Body {
            depth += 1;
            continue;
        }
        if let Some(id) = edge {
            if egraph[*id].data.free_vars.iter().any(|i| *i - depth >= 0) {
                return true;
            }
        }
    }
    return false;
}

/// Returns the first index where the two edges diverge
#[inline]
fn divergence_idx(left: &[ZNode], right: &[ZNode]) -> usize {
    for i in 0..left.len() {
        debug_assert!(i < right.len(), "right is a prefix of left");
        if left[i] != right[i] {
            debug_assert_eq!(left[i], ZNode::Func, "left: {:?}, right: {:?}", left, right);
            debug_assert_eq!(right[i], ZNode::Arg, "left: {:?}, right: {:?}", left, right);
            return i;
        }
    }
    panic!("right does not diverge from left")
}

/// Various tracking stats
#[derive(Clone,Default, Debug)]
struct Stats {
    num_wip: i32,
    num_done: i32,
    upper_bound_fired: i32,
    free_vars_done_fired: i32,
    free_vars_wip_fired: i32,
    single_use_done_fired: i32,
    single_use_wip_fired: i32,
    force_multiuse_fired: i32,
}

/// Args for compression step
#[derive(Parser, Debug, Serialize)]
#[clap(name = "Stitch")]
pub struct CompressionStepConfig {
    /// max arity of inventions to find
    #[clap(short='a', long, default_value = "2")]
    pub compress_max_arity: usize,

    /// disable caching (though caching isn't used for much currently)
    #[clap(long)]
    pub compress_no_cache: bool,

    /// print out programs rewritten under invention
    #[clap(long,short='r')]
    pub compress_show_rewritten: bool,

    /// disable the free variable pruning optimization
    #[clap(long)]
    pub no_opt_free_vars: bool,

    /// disable the single usage pruning optimization
    #[clap(long)]
    pub no_opt_single_use: bool,

    /// disable the upper bound pruning optimization
    #[clap(long)]
    pub no_opt_upper_bound: bool,

    /// disable the force multiuse pruning optimization
    #[clap(long)]
    pub no_opt_force_multiuse: bool,
}

#[derive(Debug, Clone)]
pub struct CompressionStepResult {
    pub inv: Invention,
    pub rewritten: Expr,
    pub done: FinishedItem,
    pub final_cost: i32,
    pub multiplier: f64,
    pub final_cost_rewritten: i32,
    pub multiplier_rewritten: f64,
    pub multiplier_wrt_orig: f64,
    pub uses: i32,
    pub use_exprs: Vec<Expr>,
    pub use_args: Vec<Vec<Expr>>,
    pub dc_inv_str: String,
    pub initial_cost: i32,
}

impl CompressionStepResult {
    fn new(done: FinishedItem, programs_node: Id, inv_name: &str, appzipper_of_node_zid: &mut HashMap<(Id,ZId),AppZipper>,  num_paths_to_node: &HashMap<Id,i32>, egraph: &mut EGraph, past_invs: &Vec<CompressionStepResult>) -> Self {
        let initial_cost = egraph[programs_node].data.inventionless_cost;

        // cost of the very first initial program before any inventions
        let very_first_cost = if let Some(past_inv) = past_invs.first() { past_inv.initial_cost } else { initial_cost };

        let inv = done.to_invention(inv_name, appzipper_of_node_zid, egraph);
        let rewritten: Expr = rewrite_with_invention(programs_node, &inv, egraph);
        let final_cost = initial_cost - done.utility;
        let multiplier = initial_cost as f64 / final_cost as f64;
        let final_cost_rewritten = rewritten.cost();
        let multiplier_rewritten = initial_cost as f64 / final_cost_rewritten as f64;
        let multiplier_wrt_orig = very_first_cost as f64 / final_cost_rewritten as f64;
        let uses = done.nodes.iter().map(|node| num_paths_to_node[node]).sum::<i32>();
        let use_exprs: Vec<Expr> = done.nodes.iter().map(|node| extract(*node, egraph)).collect();
        let use_args: Vec<Vec<Expr>> = done.nodes.iter().map(|node|
            done.ztuple.multiarg.iter().map(|zid|
                extract(appzipper_of_node_zid[&(*node,*zid)].arg, egraph)
            ).collect()).collect();
        
        // dreamcoder compatability
        let dc_inv_str: String = dc_inv_str(&inv, past_invs);
        CompressionStepResult { inv, rewritten, done, final_cost, multiplier, final_cost_rewritten, multiplier_rewritten, multiplier_wrt_orig, uses, use_exprs, use_args, dc_inv_str, initial_cost }
    }
    pub fn json(&self) -> serde_json::Value {        
        let use_exprs: Vec<String> = self.use_exprs.iter().map(|expr| expr.to_string()).collect();
        let use_args: Vec<String> = self.use_args.iter().map(|args| format!("{} {}", self.inv.name, args.iter().map(|expr| expr.to_string()).collect::<Vec<String>>().join(" "))).collect();
        let all_uses: Vec<serde_json::Value> = use_exprs.iter().zip(use_args.iter()).map(|(expr,args)| json!({args: expr})).collect();

        json!({            
            "body": self.inv.body.to_string(),
            "dreamcoder": self.dc_inv_str,
            "arity": self.inv.arity,
            "name": self.inv.name,
            "rewritten": self.rewritten.split_programs().iter().map(|p| p.to_string()).collect::<Vec<String>>(),
            "utility": self.done.utility,
            "final_cost": self.final_cost,
            "multiplier": self.multiplier,
            "final_cost_rewritten": self.final_cost_rewritten,
            "multiplier_rewritten": self.multiplier_rewritten,
            "multiplier_wrt_orig": self.multiplier_wrt_orig,
            "num_uses": self.uses,
            "uses": all_uses,
        })
    }
}

impl fmt::Display for CompressionStepResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "utility: {} (final_cost: ({},{}); ({:.2}x,{:.2}x)) | uses: {} | body: {}",
            self.done.utility, self.final_cost, self.final_cost_rewritten, self.multiplier, self.multiplier_rewritten, self.uses, self.inv)
    }
}

/// takes a set of programs as an Expr with Programs at its root, and does one full step of compresison.
/// Returns the top Inventions and the Expr rewritten under that invention along with other useful info in CompressionStepResult
pub fn compression_step(
    programs_expr: &Expr,
    new_inv_name: &str, // name of the new invention, like "inv4"
    cfg: &CompressionStepConfig,
    past_invs: &Vec<CompressionStepResult>, // past inventions we've found
) -> Vec<CompressionStepResult> {

    // build the egraph. We'll just be using this as a structural hasher we don't use rewrites at all. All eclasses will always only have one node.
    let mut egraph: EGraph = Default::default();
    let programs_node = egraph.add_expr(programs_expr.into());
    egraph.rebuild();

    // println!("Initial egraph:\n\t{}\n", egraph_info(&egraph));
    // if args.render_initial {
    //     save(&egraph, "0_programs", &out_dir);
    // }

    let treenodes: Vec<Id> = toplogical_ordering(programs_node,&egraph);
    assert!(usize::from(*treenodes.iter().max().unwrap()) == treenodes.len() - 1); // ensures we can safely just use Vecs of length treenodes.len() to store various nodewise things

    // populate num_paths_to_node so we know how many different parts of the programs tree
    // a node participates in (ie multiple uses within a single program or among programs)
    let num_paths_to_node: HashMap<Id,i32> = num_paths_to_node(programs_node, &treenodes, &egraph);

    let tstart_total = std::time::Instant::now();

    let tstart = std::time::Instant::now();
    let all_appzippers = get_appzippers(&treenodes, cfg.compress_no_cache, &mut egraph);
    println!("get_appzippers: {:?}ms", tstart.elapsed().as_millis());

    let tstart = std::time::Instant::now();

    // flatten all the appzippers (single arg single use inventions) to get the list of zipper paths, then sort/dedup.
    let mut paths: Vec<ZPath> = all_appzippers.values().flatten().map(|appzipper| appzipper.zipper.path.clone()).collect();
    println!("{} total paths (incl dupes)", paths.len());
    paths.sort();
    paths.dedup();
    println!("{} paths", paths.len());
    println!("collect paths and dedup: {:?}ms", tstart.elapsed().as_millis());

    // define all the important data structures for compression
    let mut appzipper_of_node_zid: HashMap<(Id,ZId),AppZipper> = Default::default(); // lookup an appzipper from a node and zid
    let mut zids_of_node: Vec<Vec<ZId>> = vec![vec![]; treenodes.len()]; // lookup all zids that a node can use
    let mut nodes_of_zid: Vec<Vec<Id>> = vec![vec![]; paths.len()]; // look up all nodes that a zid can be used at
    let mut first_mergeable_zid_of_zid: Vec<ZId> = Default::default(); // used for speed; lets you quickly lookup the smallest mergable (non-suffix) zid larger than your current zid
    let mut worklist: Vec<WorklistItem> = Default::default(); // worklist that holds partially constructed inventions
    // let mut worklist: BinaryHeap<HeapItem> = Default::default();
    let mut donelist: Vec<FinishedItem> = Default::default(); // completed inventions will go here

    // populate first_mergeable_zid_of_zid
    for (i,path) in paths.iter().enumerate() {
        // first path after `i` where the path isnt a prefix is the first mergeable one
        // (note partition_point points to the first elem where the predicate is FALSE assuming the 
        // vec already starts with all Trues and ends with all Falses)
        first_mergeable_zid_of_zid.push(paths[i..].partition_point(|p| p.starts_with(path)) + i);
    }

    let tstart = std::time::Instant::now();

    // populate zids_of_node, nodes_of_zid, and appzipper_of_node_zid
    for (treenode,appzippers) in all_appzippers {
        for appzipper in appzippers {
            if let Ok(i) = paths.binary_search(&appzipper.zipper.path) {
                zids_of_node[usize::from(treenode)].push(i);
                nodes_of_zid[i].push(treenode);
                appzipper_of_node_zid.insert((treenode,i),appzipper.clone());
            } else { unreachable!() }
        }
    }

    println!("binary search to set up data structs: {:?}ms", tstart.elapsed().as_millis());

    let tstart = std::time::Instant::now();

    // arity 0 inventions
    for node in treenodes.iter() {
        if *node == programs_node { continue; }
        // utility is just size * usages and then -COST_TERMINAL for the `inv` primitive
        let structure_penalty = - egraph[*node].data.inventionless_cost * 3 / 2;
        let utility = num_paths_to_node[&node] * (egraph[*node].data.inventionless_cost - COST_TERMINAL) + structure_penalty;
        if utility == 0 { continue; }
        donelist.push(FinishedItem::new(ZTuple::empty(),vec![*node], utility));
    }
    println!("got {} arity zero inventions ({:?}ms)", donelist.len(), tstart.elapsed().as_millis());


    let max_donelist: usize = 100; // todo revisit

    // sort + truncate donelist; update lowest_donelist_utility
    donelist.sort_unstable_by_key(|item| -item.utility);
    donelist.truncate(max_donelist);

    let mut lowest_donelist_utility = donelist.last().map(|x|x.utility).unwrap_or(0);
    let mut utility_pruning_cutoff = donelist.first().map(|x|x.utility).unwrap_or(0); // todo adjust
    let mut stats: Stats = Default::default();


    let tstart = std::time::Instant::now();

    // put together the initial set of single-arg single-use inventions from the appzippers
    initial_inventions(
        &appzipper_of_node_zid,
        &nodes_of_zid,
        &mut worklist,
        &mut donelist,
        &egraph,
        &mut lowest_donelist_utility,
        &mut utility_pruning_cutoff,
        max_donelist,
        &num_paths_to_node,
        &mut stats,
        cfg,
    );

    println!("initial_inventions(): {:?}ms", tstart.elapsed().as_millis());
    println!("initial worklist length: {}", worklist.len());
    if let Some(size) = worklist.iter().map(|ztg| ztg.nodes.len()).max() {
        println!("largest ztuple group: {}", size);
    }
    println!("avg ztuple group: {}", worklist.iter().map(|ztg| ztg.nodes.len()).sum::<usize>() as f64 / worklist.len() as f64);

    // todo not sure if its the right move to sort by this see discussion in notes "sort the worklist by upper bound"
    // worklist.sort_by_key(|wi| wi.utility_upper_bound);

    println!("total prep: {:?}ms", tstart_total.elapsed().as_millis());

    println!("deriving inventions...");
    let tstart = std::time::Instant::now();

    // derive inventions by merging
    derive_inventions(
        &appzipper_of_node_zid,
        &zids_of_node,
        &first_mergeable_zid_of_zid,
        &mut worklist,
        &mut donelist,
        cfg.compress_max_arity,
        &egraph,
        &mut lowest_donelist_utility,
        &mut utility_pruning_cutoff,
        max_donelist,
        &num_paths_to_node,
        &mut stats,
        cfg,
    );

    let elapsed_derive_inventions = tstart.elapsed().as_millis();

    println!("\nderive_inventions() done: {:?}ms\n", elapsed_derive_inventions);
    println!("total everything: {:?}ms", tstart_total.elapsed().as_millis());

    

    let orig_cost = egraph[programs_node].data.inventionless_cost;

    let mut results: Vec<CompressionStepResult> = vec![];

    // construct CompressionStepResults and print some info about them
    println!("Cost before: {}", orig_cost);
    for (i,done) in donelist.iter().enumerate().take(10) {
        let res = CompressionStepResult::new(done.clone(), programs_node, new_inv_name, &mut appzipper_of_node_zid, &num_paths_to_node, &mut egraph, past_invs);

        println!("{}: {}", i, res);
        if cfg.compress_show_rewritten {
            println!("rewritten: {}", res.rewritten);
        }
        results.push(res);
        // if args.render_inventions {
        //     inv_expr.save(&format!("inv{}",i), &out_dir);
        // }
    }

    // we sort again here because technically the costs might not be quite right if the rewrite_with_invention actually gives
    // a slightly different utility than the normal utility. This would indicate a bug and shouldn't happen often, but in case
    // there are small justifiable reasons for the mismatch we do this.
    results.sort_by_key(|res| res.final_cost_rewritten);
    

    // println!("Final egraph: {}",egraph_info(&egraph));

    // print out the largest variable we've seen (useful to make sure our egraph isnt exploding due to Vars)
    // for i in 0..100 {
    //     if search(format!("(${})",i).as_str(),&egraph).is_empty() {
    //         println!("Largest variable: ${}",i-1);
    //         break;
    //     }
    // }

    println!("Final donelist length: {}",donelist.len());
    println!("derive_inventions() took: {}ms ***\n", elapsed_derive_inventions);

    results
}

/// Finds the initial set of single-arg single-use inventions from the appzippers. This updates `donelist` with the
/// discovered inventions and pushes all the partial inventions to `worklist`. No stitching is done at this point, that
/// will all be done during `derive_inventions`. This just gets the worklist in its initial state!
#[inline(never)]
fn initial_inventions(
    appzipper_of_node_zid: &HashMap<(Id,ZId),AppZipper>,
    nodes_of_zid: &Vec<Vec<Id>>,
    worklist: &mut Vec<WorklistItem>,
    // worklist: &mut BinaryHeap<HeapItem>,
    donelist: &mut Vec<FinishedItem>,
    egraph: &EGraph,
    lowest_donelist_utility: &mut i32,
    utility_pruning_cutoff: &mut i32,
    max_donelist: usize,
    num_paths_to_node: &HashMap<Id,i32>,
    stats: &mut Stats,
    cfg: &CompressionStepConfig,
) {
    for (zid,nodes) in nodes_of_zid.iter().enumerate() {
        // 1) Define keys that we will use to index into our zippers
        let left_edge_key = |node: &Id| appzipper_of_node_zid[&(*node,zid)].zipper.left.as_slice();
        let path_key = |node: &Id| appzipper_of_node_zid[&(*node,zid)].zipper.path.as_slice();
        let right_edge_key = |node: &Id| appzipper_of_node_zid[&(*node,zid)].zipper.right.as_slice();
        let both_edge_key = |node: &Id| (appzipper_of_node_zid[&(*node,zid)].zipper.left.as_slice(),
                                         appzipper_of_node_zid[&(*node,zid)].zipper.right.as_slice());


        // 2) Sort our nodes by their both_edge_key (which also sorts them by their left_edge_key since `left` is a prefix of `both`)
        //    and then group adjacent nodes that are equal in terms of `left` or `both` keys, creating two sets of groups.
        let mut nodes = nodes.clone();
        nodes.sort_unstable_by_key(&both_edge_key);
        let left_groups = group_by_key(nodes.clone(), left_edge_key);
        let both_groups = group_by_key(nodes, both_edge_key);

        // *******************
        // * ADD TO DONELIST *
        // *******************
        for group in both_groups {
            // prune finished inventions that are only useful at one node
            if !cfg.no_opt_single_use && group.len() <= 1 {
                stats.single_use_done_fired += 1;
                continue;
            }
            // prune finished inventions that have free variables in them
            if !cfg.no_opt_free_vars && 
               (edge_has_free_vars(left_edge_key(&group[0]), path_key(&group[0]),  0, &egraph) ||
                edge_has_free_vars(right_edge_key(&group[0]), path_key(&group[0]),  0, &egraph)) {
                stats.free_vars_done_fired += 1;
                continue;
            }
            // calculate utility of this single-arg single-use invention
            let utility = {
                let left_utility = left_edge_utility(left_edge_key(&group[0]), &egraph);
                let right_utility = right_edge_utility(right_edge_key(&group[0]), &egraph);
                let arity_utility = -COST_NONTERMINAL * 1; // arity is 1
                let structure_penalty = - (left_utility + right_utility) * 3 / 2;
                let multiuse_utility = 0; // can't have multiuse here
                let num_uses = group.iter().map(|node| num_paths_to_node[node]).sum::<i32>();
                num_uses * (-COST_TERMINAL + left_utility + right_utility + arity_utility) + multiuse_utility + structure_penalty
            };
            // push to donelist if utility is good enough
            if utility > *lowest_donelist_utility {
                donelist.push(FinishedItem::new(ZTuple::single(zid), group, utility));
                if utility > *utility_pruning_cutoff {
                    *utility_pruning_cutoff = utility;
                }
            }
            stats.num_done += 1;
        }

        // *******************
        // * ADD TO WORKLIST *
        // *******************
        for group in left_groups {
            // prune partial inventions that are only useful at one node
            if !cfg.no_opt_single_use && group.len() <= 1 {
                // println!("rejected bc <= 1: {}", ZTuple::single(zid).to_expr(group[0], &appzipper_of_node_zid, &egraph));
                stats.single_use_wip_fired += 1;
                continue;
            }
            // prune partial inentions that contain free variables in their concrete part
            if !cfg.no_opt_free_vars && edge_has_free_vars(left_edge_key(&group[0]), path_key(&group[0]),  0, &egraph) {
                // panic!("hey");
                stats.free_vars_wip_fired += 1;
                continue;
            }
            // println!("passed: {}", ZTuple::single(zid).to_expr(group[0], &appzipper_of_node_zid, &egraph));

            // upper bound the utility of the partial invention
            let left_utility = left_edge_utility(left_edge_key(&group[0]), &egraph);
            let upper_bound = {
                let right_utility_upper_bound = group.iter().map(|node| num_paths_to_node[node] * right_edge_utility(right_edge_key(node), &egraph)).sum::<i32>();
                let arity_utility = -COST_NONTERMINAL * 1; // arity is 1
                let multiuse_utility = 0; // can't have multiuse here, and upper bound accounts for future multiuse
                let num_uses = group.iter().map(|node| num_paths_to_node[node]).sum::<i32>();
                num_uses * (-COST_TERMINAL + left_utility + arity_utility) + multiuse_utility + right_utility_upper_bound
            };
            // push to worklist if utility upper bound is good enough
            if !cfg.no_opt_upper_bound || upper_bound > *utility_pruning_cutoff {
                worklist.push(WorklistItem::new(ZTuple::single(zid), group, left_utility, upper_bound));
                // worklist.push(HeapItem::new(WorklistItem::new(ZTuple::single(zid), group, left_utility, upper_bound)));
                // worklist.sort_by_key(|wi| -wi.left_utility);
            } else {
                stats.upper_bound_fired += 1;
            }
        }
    }

    // sort + truncate donelist; update lowest_donelist_utility
    donelist.sort_unstable_by_key(|item| -item.utility);
    donelist.truncate(max_donelist);
    if !donelist.is_empty() { *lowest_donelist_utility = donelist.last().unwrap().utility; }
}



#[inline(never)]
fn derive_inventions(
    appzipper_of_node_zid: &HashMap<(Id,ZId),AppZipper>,
    zids_of_node: &Vec<Vec<ZId>>,
    first_mergeable_zid_of_zid: &Vec<ZId>,
    worklist: &mut Vec<WorklistItem>,
    // worklist: &mut BinaryHeap<HeapItem>,
    donelist: &mut Vec<FinishedItem>,
    max_arity: usize,
    egraph: &EGraph,
    // upper_bound_cutoff: &mut i32,
    lowest_donelist_utility: &mut i32,
    utility_pruning_cutoff: &mut i32,
    max_donelist: usize,
    num_paths_to_node: &HashMap<Id,i32>,
    stats: &mut Stats,
    cfg: &CompressionStepConfig,
) {

    // let mut till_shuffle = 100;
    // todo could parallelize 
    while let Some(wi) = worklist.pop() {
        // let wi = wi.item;
        // println!("processing {}", num_processed);
        
        // prune if upper bound is too low (cutoff may have increased in the time since this was added to the worklist)
        if !cfg.no_opt_upper_bound && wi.utility_upper_bound <= *utility_pruning_cutoff {
            stats.upper_bound_fired += 1;
            continue;
        }
        stats.num_wip += 1;

        // till_shuffle -= 1;
        // if till_shuffle == 0 {
        //     till_shuffle = 1000;
        //     worklist.shuffle(&mut rand::thread_rng());
        // }

        let rightmost_zid: ZId = wi.ztuple.elems.last().unwrap().zid;
        let first_mergeable_zid: ZId = first_mergeable_zid_of_zid[rightmost_zid];
        let mut possible_elems: Vec<(LabelledZId,Id)> = vec![];

        // collect all the possible LabelledZIds
        for node in wi.nodes.iter() {
            // skip over the zids that are prefixes - partition point will binarysearch for the first case where the predicate is false.
            // this works nicely since all (unusuable) prefix ones come before all nonprefix ones and first_mergeable_zid tells us the first nonprefix one
            let zids = &zids_of_node[usize::from(*node)];
            let start: usize = zids.partition_point(|zid| *zid < first_mergeable_zid);
            for zid in &zids[start..] {
                // merging rightmost_zid and zid is possible as long as either arity or multiuse check out

                // add any multiarg
                if wi.ztuple.arity < max_arity {
                    possible_elems.push((LabelledZId::new(*zid, wi.ztuple.arity), *node));
                }
                // add any multiuse
                let arg = appzipper_of_node_zid[&(*node,*zid)].arg;
                for (argi,arg_zid) in wi.ztuple.multiarg.iter().enumerate() {
                    if arg == appzipper_of_node_zid[&(*node, *arg_zid)].arg {
                        possible_elems.push((LabelledZId::new(*zid, argi), *node));
                    }
                }
            }
        }
        
        // sort by zid (and ivar) (and Id though we dont care about that)
        possible_elems.sort();
        // Itertools::group_by(key: F)
        for (elem, subset) in &Itertools::group_by(possible_elems.into_iter(), |(elem, _node)| elem.clone()) {
            let mut nodes: Vec<Id> = subset.map(|(_elem, node)| node).collect();
            let num_nodes = nodes.len();
            let is_multiuse = elem.ivar < wi.ztuple.arity; // multiuse means an old index within the old arity range was reused

            // divergence point doesnt depend on the specific node so we'll just use the first one
            let div_idx = divergence_idx(appzipper_of_node_zid[&(nodes[0],rightmost_zid)].zipper.path.as_slice(),
                                         appzipper_of_node_zid[&(nodes[0],elem.zid)].zipper.path.as_slice());
            
            let div_depth = appzipper_of_node_zid[&(nodes[0],rightmost_zid)].zipper.path[..div_idx].iter().filter(|x| **x == ZNode::Body).count() as i32;

            let new_ztuple: ZTuple = wi.ztuple.extend(elem.clone(), div_idx, is_multiuse);

            // define key functions for grabbing all the slices of zipper we care about
            // left_fold_key is the left inner side of the fold which is rightmost_zid.RIGHT (not LEFT)
            let left_fold_key =  |node: &Id| &appzipper_of_node_zid[&(*node,rightmost_zid)].zipper.right[div_idx+1..];
            let left_fold_path_key =  |node: &Id| &appzipper_of_node_zid[&(*node,rightmost_zid)].zipper.path[div_idx+1..];
            let right_fold_key = |node: &Id| &appzipper_of_node_zid[&(*node,elem.zid)].zipper.left[div_idx+1..];
            let right_fold_path_key = |node: &Id| &appzipper_of_node_zid[&(*node,elem.zid)].zipper.path[div_idx+1..];

            let fold_key = |node: &Id| (&appzipper_of_node_zid[&(*node,rightmost_zid)].zipper.right[div_idx+1..],
                                        &appzipper_of_node_zid[&(*node,elem.zid)].zipper.left[div_idx+1..]);
            let right_edge_key = |node: &Id| appzipper_of_node_zid[&(*node,elem.zid)].zipper.right.as_slice();
            let right_path_key = |node: &Id| appzipper_of_node_zid[&(*node,elem.zid)].zipper.path.as_slice();

            let both_edge_key = |node: &Id| (&appzipper_of_node_zid[&(*node,rightmost_zid)].zipper.right[div_idx+1..],
                                             &appzipper_of_node_zid[&(*node,elem.zid)].zipper.left[div_idx+1..],
                                             appzipper_of_node_zid[&(*node,elem.zid)].zipper.right.as_slice());

            // sorting by `both` will also sort by fold_key since the latter is a prefix of the former
            nodes.sort_unstable_by_key(&both_edge_key);

            let fold_groups = group_by_key(nodes.clone(), fold_key);
            let both_groups = group_by_key(nodes, both_edge_key);
            let num_offspring = fold_groups.len();

            // *******************
            // * ADD TO DONELIST *
            // *******************
            for group in both_groups {
                // if groups are singletons or contain free variables, skip them
                if !cfg.no_opt_single_use && group.len() <= 1 {
                    stats.single_use_done_fired += 1;
                    continue;
                }
                if  !cfg.no_opt_free_vars &&
                   (edge_has_free_vars(left_fold_key(&group[0]), left_fold_path_key(&group[0]),  div_depth, &egraph) ||
                    edge_has_free_vars(right_fold_key(&group[0]), right_fold_path_key(&group[0]),  div_depth, &egraph) ||
                    edge_has_free_vars(right_edge_key(&group[0]), right_path_key(&group[0]),  0, &egraph)) {
                    stats.free_vars_done_fired += 1;
                    continue;
                }
                // the left side of the fold is a RIGHT-facing edge (since it faces into the fold) hence it's right_edge_utility for the left_fold_key
                let fold_utility = right_edge_utility(left_fold_key(&group[0]), egraph) + left_edge_utility(right_fold_key(&group[0]), egraph);
                let left_utility = wi.left_utility + fold_utility;
                let right_utility = right_edge_utility(right_edge_key(&group[0]), egraph);
                let arity_utility = -COST_NONTERMINAL * new_ztuple.arity as i32; // new arity
                // let arity_penalty = - (new_ztuple.arity as i32); // extra penalty for higher arity to break ties in rare cases
                // arity penalty case: arity=2 (#0 (foo #1)) is the same utility as arity=1 (foo #0) or something like that, so we need to break the tie

                // a bit like the DC structure penalty
                let structure_penalty = - (left_utility + right_utility) * 3 / 2;

                // multiuse utility depends on the size of the argument that's being used in multiple places. We can
                // look up that argument using appzipper_of_node_zid since ztuple.multiuses gives us the zids for the multiuse
                // cases (leaving out the original use)
                let multiuse_utility = new_ztuple.multiuse.iter()
                    .map(|&arg_zid| // for each extra use of a multiuse arg
                        group.iter().map(|node| // for each node
                            num_paths_to_node[node] * // account for same node being used in multiple subtrees
                            egraph[appzipper_of_node_zid[&(*node,arg_zid)].arg].data.inventionless_cost
                        ).sum::<i32>()
                    ).sum::<i32>();
                
                let num_uses = group.iter().map(|node| num_paths_to_node[node]).sum::<i32>();
                let utility = num_uses * (-COST_TERMINAL + left_utility + right_utility + arity_utility) + multiuse_utility + structure_penalty;
                if utility > *lowest_donelist_utility {
                    donelist.push(FinishedItem::new(new_ztuple.clone(), group, utility));
                    if utility > *utility_pruning_cutoff {
                        *utility_pruning_cutoff = utility;
                    }
                }
                stats.num_done += 1;
            }
    
            // *******************
            // * ADD TO WORKLIST *
            // *******************
            for group in fold_groups {
                // if groups are singletons or the fold contains free variables, skip them
                if !cfg.no_opt_single_use && group.len() <= 1 {
                    stats.single_use_wip_fired += 1;
                    continue;
                }
                if !cfg.no_opt_free_vars && 
                   (edge_has_free_vars(left_fold_key(&group[0]), left_fold_path_key(&group[0]),  div_depth, &egraph) ||
                    edge_has_free_vars(right_fold_key(&group[0]), right_fold_path_key(&group[0]),  div_depth, &egraph)) {
                    stats.free_vars_wip_fired += 1;
                    continue;
                }
                // todo filter out ones w free vars too
                let fold_utility = right_edge_utility(left_fold_key(&group[0]), egraph) + left_edge_utility(right_fold_key(&group[0]), egraph);
                let left_utility = wi.left_utility + fold_utility;
                let upper_bound = {
                    let right_utility_upper_bound = group.iter().map(|node| num_paths_to_node[node] * right_edge_utility(right_edge_key(node), egraph)).sum::<i32>();
                    
                    let arity_utility = -COST_NONTERMINAL * new_ztuple.arity as i32; // new arity
                    // multiuse utility depends on the size of the argument that's being used in multiple places. We can
                    // look up that argument using appzipper_of_node_zid since ztuple.multiuses gives us the zids for the multiuse
                    // cases (leaving out the original use)
                    let multiuse_utility = new_ztuple.multiuse.iter()
                        .map(|&arg_zid| // for each extra use of a multiuse arg
                            group.iter().map(|node| // for each node
                                num_paths_to_node[node] * // account for same node being used in multiple subtrees
                                egraph[appzipper_of_node_zid[&(*node,arg_zid)].arg].data.inventionless_cost
                            ).sum::<i32>()
                        ).sum::<i32>();

                    let num_uses = group.iter().map(|node| num_paths_to_node[node]).sum::<i32>();
                    num_uses * (-COST_TERMINAL + left_utility + arity_utility) + multiuse_utility + right_utility_upper_bound
                };
                if !cfg.no_opt_upper_bound || upper_bound > *utility_pruning_cutoff {
                    // worklist.push(HeapItem::new(WorklistItem::new(new_ztuple.clone(), group, left_utility, upper_bound)));
                    worklist.push(WorklistItem::new(new_ztuple.clone(), group, left_utility, upper_bound));
                    // worklist.sort_by_key(|wi| -wi.left_utility);
                } else {
                    stats.upper_bound_fired += 1;
                }
            }

            // a multiuse invention that is present at all the nodes from the original worklist AND
            // has all the same non-leading-edge so it only has one offspring. It is strictly beneficial (or breakeven
            // for single leaf nodes) to accept this multiuse, so we can just Break before looking at any higher zid 
            // merges and instead let this newly pushed multiuse thing be the one that merges with those future things.
            if !cfg.no_opt_force_multiuse && is_multiuse && num_nodes == wi.nodes.len() && num_offspring == 1 {
                stats.force_multiuse_fired += 1;
                break; // todo I would be a little careful with this optimization, disable it by commenting this if you suspect its not sound. It should be fine but I wrote it at night.
            }
        }

        if donelist.len() > std::cmp::max(1000, if max_donelist != usize::MAX { max_donelist*4 } else { max_donelist }) {
            donelist.sort_unstable_by_key(|item| -item.utility);
            donelist.truncate(max_donelist);
            *lowest_donelist_utility = donelist.last().unwrap().utility;
        }

    }

    assert!(worklist.is_empty());

    donelist.sort_unstable_by_key(|item| -item.utility);
    donelist.truncate(max_donelist);
    if !donelist.is_empty() { *lowest_donelist_utility = donelist.last().unwrap().utility; }

    println!("{:?}", stats);
}

