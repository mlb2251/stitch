use crate::*;
use std::collections::{HashSet,HashMap,BinaryHeap};
use std::fmt::{self, Formatter, Display};
use clap::Parser;
use std::path::PathBuf;
use std::hash::Hash;
use std::cmp::Ordering;
use itertools::Itertools;
use extraction::extract;
use serde_json::json;
use serde::Serialize;

/// Args for compression
#[derive(Parser, Debug, Serialize)]
#[clap(name = "Stitch")]
pub struct CompressionArgs {
    /// json file to read compression input programs from
    #[clap(short, long, parse(from_os_str), default_value = "data/train_19.json")]
    pub file: PathBuf,

    /// output file
    #[clap(short, long, parse(from_os_str), default_value = "out/out.json")]
    pub out: PathBuf,

    /// Number of iterations to run compression for
    #[clap(short, long, default_value = "3")]
    pub iterations: usize,

    /// max arity of inventions
    #[clap(short='a', long, default_value = "2")]
    pub max_arity: usize,

    /// 
    #[clap(long,short='r')]
    pub show_rewritten: bool,

    /// 
    #[clap(long)]
    pub shuffle: bool,

    /// 
    #[clap(long)]
    pub truncate: Option<usize>,

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
pub const COST_NONTERMINAL: i32 = 1;
pub const COST_TERMINAL: i32 = 100;

pub type EGraph = egg::EGraph<Lambda, LambdaAnalysis>;

/// The analysis data associated with each Lambda node
#[derive(Debug)]
pub struct Data {
    pub free_vars: HashSet<i32>, // $i vars. For example (lam $2) has free_vars = {1}.
    pub free_ivars: HashSet<i32>, // #i ivars
    pub inventionless_cost: i32,
}


/// An invention we've found (ie a learned function we can use to compress the program).
/// Inventions have a body + an arity
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct Invention {
    pub body:Id, // this will be a subtree which can have IVars
    pub arity: usize // also equal to max ivar in subtree + 1
}
impl Invention {
    pub fn new(body:Id, arity: usize) -> Self {
        Invention {
            body,
            arity
        }
    }
}

/// At the end of the day we convert our Inventions into InventionExprs to make
/// them standalone without needing to carry the EGraph around to figure out what
/// the body Id points to.
#[derive(Debug, Clone)]
pub struct InventionExpr {
    pub body: Expr, // invention body (not wrapped in lambdas)
    pub arity: usize
}
impl InventionExpr {
    pub fn new(body: Expr, arity: usize) -> Self {
        Self { body, arity }
    }
}

impl Display for InventionExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "(arity={}: {})", self.arity, self.body)
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
pub fn shift(e: Id, shift: Shift, egraph: &mut EGraph, caches: Option<&mut CacheGenerator>) -> Option<Id> {
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
pub fn toplogical_ordering(root: Id, egraph: &EGraph) -> Vec<Id> {
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
pub enum Shift {
    ShiftVar(i32), // shift $i to be $(i+incr_by)
    ShiftIVar(i32), // shift #i to be #(i+incr_by)
    TableShiftIVar(Vec<i32>), // shift #i to be #(i+table[#i]) ie look up the shift amount in the table
}

/// generates caches for shift()
pub struct CacheGenerator {
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


// fn rewrite_with_finisheditem()



/// A node in an Zipper
/// Ord: Func < Body < Arg
// #[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
// enum ZNode {
//     // * order of variants here is important because the derived Ord will use it
//     Func(Id), // zipper went into the function, so Id is the arg
//     Body, 
//     Arg(Id), // zipper went into the arg, so Id is the function
// }

/// A node in an Zipper
/// Ord: Func < Body < Arg
#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
enum ZNode {
    // * order of variants here is important because the derived Ord will use it
    Func, // zipper went into the function, so Id is the arg
    Body, 
    Arg, // zipper went into the arg, so Id is the function
}

type ZId = usize; // zipper id
type ZPath = Vec<ZNode>;

/// a 1 arg invention
#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
struct Zipper {
    path: ZPath,
    left: Vec<Option<Id>>,
    right: Vec<Option<Id>>,
}


/// a 1 arg applied invention
#[derive(Debug,Clone, Eq, PartialEq, Hash)]
struct AppZipper {
    zipper: Zipper,
    arg: Id,
}

#[derive(Debug,Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
struct ZTupleElem {
    zid: ZId,
    ivar: usize // which #i argument this is, which also corresponds to args[i] ofc
}

/// a multiarg multiuse invention
#[derive(Debug,Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
struct ZTuple {
    elems: Vec<ZTupleElem>,
    divergence_idxs: Vec<usize>,
    multiarg: Vec<ZId>, // len=arity, gives the first zid added for each arg
    multiuse: Vec<ZId>, // gives the 2nd and onward zids added for each arg (not the first, thats in multiarg)
    arity: usize,
}

/// a multiarg multiuse invention applied
#[derive(Debug,Clone, Eq, PartialEq, Hash)]
struct AppZTuple {
    ztuple: ZTuple,
    args: Vec<Id>,
}

// can add upper bound utility and such here later too
#[derive(Debug,Clone, Eq, PartialEq, PartialOrd, Ord)]
struct WorklistItem {
    ztuple: ZTuple,
    nodes: Vec<Id>, // nodes in the group
    left_utility: i32, // utility of a single usage
    utility_upper_bound: i32, // upper bound utility over all usages
}

#[derive(Debug,Clone, Eq, PartialEq, PartialOrd, Ord)]
struct HeapItem {
    key: i32,
    item: WorklistItem,
}
impl HeapItem {
    fn new(item: WorklistItem) -> HeapItem {
        HeapItem { key: item.ztuple.elems.last().unwrap().zid as i32, item: item }
    }
}

// can add upper bound utility and such here later too
#[derive(Debug,Clone)]
struct FinishedItem {
    ztuple: ZTuple,
    nodes: Vec<Id>, // nodes in the group
    utility: i32,
}

#[derive(Debug,Clone)]
struct InventionItem {
    item: FinishedItem,
    expr: Expr,
    usages: i32,
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

impl ZTupleElem {
    fn new(zid: ZId, ivar: usize) -> ZTupleElem {
        ZTupleElem { zid: zid, ivar: ivar }
    }
}

impl ZTuple {
    fn new(elems: Vec<ZTupleElem>, divergence_idxs: Vec<usize>, multiarg: Vec<usize>, multiuse: Vec<usize>, arity: usize) -> ZTuple {
        ZTuple { elems, divergence_idxs, multiarg, multiuse, arity }
    }
    fn single(zid: ZId) -> ZTuple {
        ZTuple::new(vec![ZTupleElem::new(zid, 0)], vec![], vec![zid], vec![], 1)
    }
    fn extend(&self, elem: ZTupleElem, div_idx: usize, is_multiuse: bool) -> ZTuple {
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
    fn to_expr(&self, node: Id, appzipper_of_node_zid: &HashMap<(Id,ZId),AppZipper>, egraph: &EGraph) -> Expr {
        let mut elem_idx: usize = 0;
        let mut zipper: &Zipper = &appzipper_of_node_zid[&(node,self.elems[elem_idx].zid)].zipper;
        let mut depth: usize = zipper.path.len() - 1;
        let mut expr = Expr::ivar(self.elems[elem_idx].ivar as i32);
        let mut diverged: Vec<(usize,Expr)> = vec![];

        loop {
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

impl WorklistItem {
    fn new(ztuple: ZTuple, nodes: Vec<Id>, left_utility: i32, utility_upper_bound: i32) -> WorklistItem {
        WorklistItem { ztuple: ztuple, nodes: nodes, left_utility: left_utility, utility_upper_bound: utility_upper_bound }
    }
}

impl FinishedItem {
    fn new(ztuple: ZTuple, nodes: Vec<Id>, utility: i32) -> FinishedItem {
        FinishedItem { ztuple, nodes, utility }
    }
}

fn get_appzippers(treenodes: &[Id], no_cache:bool, egraph: &mut EGraph) -> HashMap<Id,Vec<AppZipper>> {
    let mut all_appzippers: HashMap<Id,Vec<AppZipper>> = Default::default();
    let caches = &mut CacheGenerator::new(!no_cache);
    
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
                        new.arg = shift(b_appzipper.arg, Shift::ShiftVar(-1), egraph, Some(caches)).unwrap();
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

#[inline]
fn group_by_key<T: Copy, U: Ord>(v: Vec<T>, key: impl Fn(&T)->U) -> Vec<Vec<T>> {
    // sort so that all equal elements are adjacent

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


/// Utility
/// The utility of an invention is how useful it is at compressing the program.
/// utility(inv) = { (-NONTERMINAL_COST * arity) +  (COST_NONTERMINAL * total_path_len) + (sum of inventionless costs along edges) + -COST_TERMINAL } (for each arg #i, (num_usages - 1) * arg.inventionless_cost) + (-COST_NONTERMINAL * num_usages)
///                  ^ cost of Apps to use inv      ^ all these nonterms used to be in the original program, hence they count toward utility
///                                                   note that in practice we have to be careful not to double-count shared path prefixes
///                                                   but we can handle this easily through the "fold" and leading/tailing edge setup
///                                                                                       ^ again all these subtrees used to be in the original program so they count toward utility.
///                                                                                       and again we must be careful not to double-count shared path prefixes, which will again
///                                                                                       be naturally handled by the fold setup.
///                                                                                                                                                       ^ this captures multiuse inventions, where for each additional use
///                                                                                                                                                       you gain (+ size_of_arg) utility. Notably this cost is specific to
///                                                                                                                                                       the location that it is used and specific arguments passed in.
/// implementation: we'll build up this utility as we go. We'll lump the path length                                                  ^ theres a COST_TERMINAL whenever you use an invention for the `inv` primitive itself
/// term into the left_edge_utility. 

/// utility of a fragment of a zipper, specifically a left edge (the left/right
/// distinction is just so we can include the nonterminal cost in the left edge)
#[inline]
fn left_edge_utility(edge: &[Option<Id>], egraph: &EGraph) -> i32 {
    edge.len() as i32 * COST_NONTERMINAL +
    edge.iter().filter_map(|option_id|
        option_id.map(|id| egraph[id].data.inventionless_cost)).sum::<i32>()
}
#[inline]
fn right_edge_utility(edge: &[Option<Id>], egraph: &EGraph) -> i32 {
    edge.iter().filter_map(|option_id|
        option_id.map(|id| egraph[id].data.inventionless_cost)).sum::<i32>()
}

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

#[inline]
fn divergence_idx(left: &[ZNode], right: &[ZNode]) -> usize {
    // find the first index where the two edges diverge

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


/// takes a (programs ...) expr, returns the best Invention and the Expr rewritten under that invention
fn compression_step(
    programs_expr: &Expr,
    args: &CompressionArgs,
    out_dir: &str,
    new_inv_name: &str,
    very_first_cost: i32,
) -> Vec<CompressionStepResult> {

    // build the egraph. We'll just be using this as a structural hasher we don't use rewrites at all. All eclasses will always only have one node.
    let mut egraph: EGraph = Default::default();
    let programs_node = egraph.add_expr(programs_expr.into());
    egraph.rebuild();

    println!("Initial egraph:\n\t{}\n", egraph_info(&egraph));
    if args.render_initial {
        save(&egraph, "0_programs", &out_dir);
    }

    let tstart = std::time::Instant::now();

    let treenodes: Vec<Id> = toplogical_ordering(programs_node,&egraph);
    assert!(usize::from(*treenodes.iter().max().unwrap()) == treenodes.len() - 1); // ensures we can safely just use Vecs of length treenodes.len() to store various nodewise things

    // populate num_paths_to_node so we know how many different parts of the programs tree
    // a node participates in (ie multiple uses within a single program or among programs)
    let mut num_paths_to_node: HashMap<Id,i32> = HashMap::new();
    treenodes.iter().for_each(|treenode| {
        num_paths_to_node.insert(*treenode, 0);
    });
    fn helper(num_paths_to_node: &mut HashMap<Id,i32>, node: &Id, egraph: &EGraph) {
        // num_paths_to_node.insert(*child, num_paths_to_node[node] + 1);
        *num_paths_to_node.get_mut(node).unwrap() += 1;
        for child in egraph[*node].nodes[0].children() {
            helper(num_paths_to_node, &child, egraph);
        }
    }
    helper(&mut num_paths_to_node, &programs_node, &egraph);

    let tstart_total = std::time::Instant::now();

    let tstart = std::time::Instant::now();
    let all_appzippers = get_appzippers(&treenodes, args.no_cache, &mut egraph);
    println!("get_appzippers: {:?}ms", tstart.elapsed().as_millis());


    // from inv1 body to set of roots that it's used under
    let tstart = std::time::Instant::now();

    let mut paths: Vec<ZPath> = all_appzippers.values().flatten().map(|appzipper| appzipper.zipper.path.clone()).collect();
    println!("{} total paths (incl dupes)", paths.len());
    paths.sort();
    paths.dedup();
    println!("{} paths", paths.len());
    println!("collect paths and dedup: {:?}ms", tstart.elapsed().as_millis());

    let mut appzipper_of_node_zid: HashMap<(Id,ZId),AppZipper> = Default::default();
    let mut zids_of_node: Vec<Vec<ZId>> = vec![vec![]; treenodes.len()];
    let mut nodes_of_zid: Vec<Vec<Id>> = vec![vec![]; paths.len()];
    let mut first_mergeable_zid_of_zid: Vec<ZId> = Default::default();
    let mut worklist: Vec<WorklistItem> = Default::default();
    // let mut worklist: BinaryHeap<HeapItem> = Default::default();
    let mut donelist: Vec<FinishedItem> = Default::default();

    for (i,path) in paths.iter().enumerate() {
        // first path after `i` where the path isnt a prefix is the first mergeable one
        // (note partition_point points to the first elem where the predicate is FALSE assuming the 
        // vec already starts with all Trues and ends with all Falses)
        first_mergeable_zid_of_zid.push(paths[i..].partition_point(|p| p.starts_with(path)) + i);
    }

    let tstart = std::time::Instant::now();


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

    // build up the initial worklist
    let max_donelist: usize = 100;
    // let mut upper_bound_cutoff: i32 = 0;
    let mut lowest_donelist_utility = 0;
    let mut best_utility = 0;

    let mut stats: Stats = Default::default();

    initial_inventions(
        &appzipper_of_node_zid,
        &nodes_of_zid,
        &mut worklist,
        &mut donelist,
        &egraph,
        &mut lowest_donelist_utility,
        &mut best_utility,
        max_donelist,
        &num_paths_to_node,
        &mut stats,
    );

    println!("initial worklist length: {}", worklist.len());
    println!("set up the worklist: {:?}ms", tstart.elapsed().as_millis());
    println!("largest ztuple group: {}", worklist.iter().map(|ztg| ztg.nodes.len()).max().unwrap());
    println!("avg ztuple group: {}", worklist.iter().map(|ztg| ztg.nodes.len()).sum::<usize>() as f64 / worklist.len() as f64);


    // todo not sure if its the right move to sort by this see discussion in notes "sort the worklist by upper bound"
    // worklist.sort_by_key(|wi| wi.utility_upper_bound);

    println!("total prep: {:?}ms", tstart_total.elapsed().as_millis());
    
    // let tstart = std::time::Instant::now();
    // let num_invs = appzipper_of_node_zid.values().map(|appzipper| appzipper.zipper.clone()).collect::<HashSet<Zipper>>().len();
    // println!("counted inventions: {:?}ms", tstart.elapsed().as_millis());
    // println!("{} single arg invs", num_invs);

    println!("deriving inventions...");
    let tstart = std::time::Instant::now();

    derive_inventions(
        &appzipper_of_node_zid,
        &zids_of_node,
        &first_mergeable_zid_of_zid,
        &mut worklist,
        &mut donelist,
        args.max_arity,
        &egraph,
        &mut lowest_donelist_utility,
        &mut best_utility,
        max_donelist,
        &num_paths_to_node,
        &mut stats,
    );


    println!("\ndone deriving inventions: {:?}ms\n", tstart.elapsed().as_millis());

    println!("total everything: {:?}ms", tstart_total.elapsed().as_millis());

    let elapsed = tstart.elapsed().as_millis();

    let orig_cost = egraph[programs_node].data.inventionless_cost;

    let mut results: Vec<CompressionStepResult> = vec![];

    println!("Cost before: {}", orig_cost);
    for (i,done) in donelist.iter().enumerate().take(10) {
        let res = CompressionStepResult::new(done.clone(), programs_node, very_first_cost, new_inv_name, &mut appzipper_of_node_zid, &num_paths_to_node, &mut egraph);

        println!("{}: {}", i, res);
        if args.show_rewritten {
            println!("rewritten: {}", res.rewritten);
        }
        results.push(res);
        // todo also add printing the actual body if --show-rewritten is set
        // if args.render_inventions {
        //     inv_expr.save(&format!("inv{}",i), &out_dir);
        // }
    }

    // sort now that we have the exact costs by rewriting
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
    println!("Core stuff took: {}ms ***\n", elapsed);


    results
}

#[inline(never)]
fn initial_inventions(
    appzipper_of_node_zid: &HashMap<(Id,ZId),AppZipper>,
    nodes_of_zid: &Vec<Vec<Id>>,
    worklist: &mut Vec<WorklistItem>,
    // worklist: &mut BinaryHeap<HeapItem>,
    donelist: &mut Vec<FinishedItem>,
    egraph: &EGraph,
    lowest_donelist_utility: &mut i32,
    best_utility: &mut i32,
    max_donelist: usize,
    num_paths_to_node: &HashMap<Id,i32>,
    stats: &mut Stats,
) {
    for (zid,nodes) in nodes_of_zid.iter().enumerate() {
        let left_edge_key = |node: &Id| appzipper_of_node_zid[&(*node,zid)].zipper.left.as_slice();
        let path_key = |node: &Id| appzipper_of_node_zid[&(*node,zid)].zipper.path.as_slice();
        let right_edge_key = |node: &Id| appzipper_of_node_zid[&(*node,zid)].zipper.right.as_slice();
        let both_edge_key = |node: &Id| (appzipper_of_node_zid[&(*node,zid)].zipper.left.as_slice(),
                                         appzipper_of_node_zid[&(*node,zid)].zipper.right.as_slice());
        let mut nodes = nodes.clone();

        // sorting by `both` means elements with the same `both` key will be adjacent, AND
        // elements with the same `left` key will be contiguous (since the `left` key is a prefix of the `both` key)
        nodes.sort_unstable_by_key(&both_edge_key);

        let left_groups = group_by_key(nodes.clone(), left_edge_key);
        let both_groups = group_by_key(nodes, both_edge_key);

        // finish any inventions
        for group in both_groups {
            // if groups are singletons or contain free variables, skip them
            if group.len() <= 1 {
                stats.single_use_done_fired += 1;
                continue;
            }
            if edge_has_free_vars(left_edge_key(&group[0]), path_key(&group[0]),  0, &egraph) ||
               edge_has_free_vars(right_edge_key(&group[0]), path_key(&group[0]),  0, &egraph) {
                stats.free_vars_done_fired += 1;
                continue;
            }
            let utility = {
                let left_utility = left_edge_utility(left_edge_key(&group[0]), &egraph);
                let right_utility = right_edge_utility(right_edge_key(&group[0]), &egraph);
                let arity_utility = -COST_NONTERMINAL * 1; // arity is 1
                let multiuse_utility = 0; // can't have multiuse here
                let num_uses = group.iter().map(|node| num_paths_to_node[node]).sum::<i32>();
                num_uses * (-COST_TERMINAL + left_utility + right_utility + arity_utility) + multiuse_utility
            };
            if utility > *lowest_donelist_utility {
                donelist.push(FinishedItem::new(ZTuple::single(zid), group, utility));
                if utility > *best_utility {
                    *best_utility = utility;
                }
            }
            stats.num_done += 1;
        }

        // extend the worklist
        for group in left_groups {
            // if groups are singletons or contain free variables on the left edge (which can never be changed), discard them
            if group.len() <= 1 {
                // println!("rejected bc <= 1: {}", ZTuple::single(zid).to_expr(group[0], &appzipper_of_node_zid, &egraph));
                stats.single_use_wip_fired += 1;
                continue;
            }
            if edge_has_free_vars(left_edge_key(&group[0]), path_key(&group[0]),  0, &egraph) {
                // panic!("hey");
                stats.free_vars_wip_fired += 1;
                continue;
            }
            // println!("passed: {}", ZTuple::single(zid).to_expr(group[0], &appzipper_of_node_zid, &egraph));


            let left_utility = left_edge_utility(left_edge_key(&group[0]), &egraph);
            let upper_bound = {
                let right_utility_upper_bound = group.iter().map(|node| num_paths_to_node[node] * right_edge_utility(right_edge_key(node), &egraph)).sum::<i32>();
                let arity_utility = -COST_NONTERMINAL * 1; // arity is 1
                let multiuse_utility = 0; // can't have multiuse here, and upper bound accounts for future multiuse
                let num_uses = group.iter().map(|node| num_paths_to_node[node]).sum::<i32>();
                num_uses * (-COST_TERMINAL + left_utility + arity_utility) + multiuse_utility + right_utility_upper_bound
            };
            if upper_bound > *best_utility {
                worklist.push(WorklistItem::new(ZTuple::single(zid), group, left_utility, upper_bound));
                // worklist.push(HeapItem::new(WorklistItem::new(ZTuple::single(zid), group, left_utility, upper_bound)));
                // worklist.sort_by_key(|wi| -wi.left_utility);
            } else {
                stats.upper_bound_fired += 1;
            }
        }
    }

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
    best_utility: &mut i32,
    max_donelist: usize,
    num_paths_to_node: &HashMap<Id,i32>,
    stats: &mut Stats,
) {

    // todo ofc can parallelize this 
    while let Some(wi) = worklist.pop() {
        // let wi = wi.item;
        // println!("processing {}", num_processed);
        // check upper bound;
        if wi.utility_upper_bound <= *best_utility {
            stats.upper_bound_fired += 1;
            continue;
        }
        stats.num_wip += 1;

        let rightmost_zid: ZId = wi.ztuple.elems.last().unwrap().zid;
        let first_mergeable_zid: ZId = first_mergeable_zid_of_zid[rightmost_zid];


        let mut possible_elems: Vec<(ZTupleElem,Id)> = vec![];

        // collect all the possible ztupleelems
        for node in wi.nodes.iter() {
            // skip over the zids that are prefixes - partition point will binarysearch for the first case where the predicate is false.
            // this works nicely since all (unusuable) prefix ones come before all nonprefix ones and first_mergeable_zid tells us the first nonprefix one
            let zids = &zids_of_node[usize::from(*node)];
            let start: usize = zids.partition_point(|zid| *zid < first_mergeable_zid);
            for zid in &zids[start..] {
                // merging rightmost_zid and zid is possible as long as either arity or multiuse check out

                // add any multiarg
                if wi.ztuple.arity < max_arity {
                    possible_elems.push((ZTupleElem::new(*zid, wi.ztuple.arity), *node));
                }
                // add any multiuse
                let arg = appzipper_of_node_zid[&(*node,*zid)].arg;
                for (argi,arg_zid) in wi.ztuple.multiarg.iter().enumerate() {
                    if arg == appzipper_of_node_zid[&(*node, *arg_zid)].arg {
                        possible_elems.push((ZTupleElem::new(*zid, argi), *node));
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

            // finish any inventions
            for group in both_groups {
                // if groups are singletons or contain free variables, skip them
                if group.len() <= 1 {
                    stats.single_use_done_fired += 1;
                    continue;
                }
                if  edge_has_free_vars(left_fold_key(&group[0]), left_fold_path_key(&group[0]),  div_depth, &egraph) ||
                    edge_has_free_vars(right_fold_key(&group[0]), right_fold_path_key(&group[0]),  div_depth, &egraph) ||
                    edge_has_free_vars(right_edge_key(&group[0]), right_path_key(&group[0]),  0, &egraph) {
                    stats.free_vars_done_fired += 1;
                    continue;
                }
                // the left side of the fold is a RIGHT-facing edge (since it faces into the fold) hence it's right_edge_utility for the left_fold_key
                let fold_utility = right_edge_utility(left_fold_key(&group[0]), egraph) + left_edge_utility(right_fold_key(&group[0]), egraph);
                let left_utility = wi.left_utility + fold_utility;
                let right_utility = right_edge_utility(right_edge_key(&group[0]), egraph);
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
                let utility = num_uses * (-COST_TERMINAL + left_utility + right_utility + arity_utility) + multiuse_utility;
                if utility > *lowest_donelist_utility {
                    donelist.push(FinishedItem::new(new_ztuple.clone(), group, utility));
                    if utility > *best_utility {
                        *best_utility = utility;
                    }
                }
                stats.num_done += 1;
            }
    
            // extend the worklist
            for group in fold_groups {
                // if groups are singletons or the fold contains free variables, skip them
                if group.len() <= 1 {
                    stats.single_use_wip_fired += 1;
                    continue;
                }
                if edge_has_free_vars(left_fold_key(&group[0]), left_fold_path_key(&group[0]),  div_depth, &egraph) ||
                    edge_has_free_vars(right_fold_key(&group[0]), right_fold_path_key(&group[0]),  div_depth, &egraph) {
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
                if upper_bound > *best_utility {
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
            if is_multiuse && num_nodes == wi.nodes.len() && num_offspring == 1 {
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

#[derive(Debug, Clone)]
pub struct CompressionStepResult {
    inv: InventionExpr,
    inv_name: String,
    rewritten: Expr,
    done: FinishedItem,
    final_cost: i32,
    multiplier: f64,
    final_cost_rewritten: i32,
    multiplier_rewritten: f64,
    multiplier_wrt_orig: f64,
    uses: i32,
    use_exprs: Vec<Expr>,
    use_args: Vec<Vec<Expr>>
}

impl CompressionStepResult {
    fn new(done: FinishedItem, programs_node: Id, very_first_cost: i32, inv_name: &str, appzipper_of_node_zid: &mut HashMap<(Id,ZId),AppZipper>,  num_paths_to_node: &HashMap<Id,i32>, egraph: &mut EGraph) -> Self {
        let orig_cost = egraph[programs_node].data.inventionless_cost;

        let inv = InventionExpr::new(done.ztuple.to_expr(done.nodes[0], appzipper_of_node_zid, egraph), done.ztuple.arity);
        let rewritten: Expr = rewrite_with_inventions(programs_node, &[&inv], &[inv_name], egraph);
        let final_cost = orig_cost - done.utility;
        let multiplier = orig_cost as f64 / final_cost as f64;
        let final_cost_rewritten = rewritten.cost();
        let multiplier_rewritten = orig_cost as f64 / final_cost_rewritten as f64;
        let multiplier_wrt_orig = very_first_cost as f64 / final_cost_rewritten as f64;
        let uses = done.nodes.iter().map(|node| num_paths_to_node[node]).sum::<i32>();
        let use_exprs: Vec<Expr> = done.nodes.iter().map(|node| extract(*node, egraph)).collect();
        let use_args: Vec<Vec<Expr>> = done.nodes.iter().map(|node|
            done.ztuple.multiarg.iter().map(|zid|
                extract(appzipper_of_node_zid[&(*node,*zid)].arg, egraph)
            ).collect()).collect();
        CompressionStepResult { inv, inv_name: String::from(inv_name), rewritten, done, final_cost, multiplier, final_cost_rewritten, multiplier_rewritten, multiplier_wrt_orig, uses, use_exprs, use_args }
    }
    fn json(&self) -> serde_json::Value {
        let use_exprs: Vec<String> = self.use_exprs.iter().map(|expr| expr.to_string()).collect();
        let use_args: Vec<String> = self.use_args.iter().map(|args| format!("{} {}", self.inv_name, args.iter().map(|expr| expr.to_string()).collect::<Vec<String>>().join(" "))).collect();
        let all_uses: Vec<serde_json::Value> = use_exprs.iter().zip(use_args.iter()).map(|(expr,args)| json!({args: expr})).collect();

        json!({            
            "body": self.inv.body.to_string(),
            "arity": self.inv.arity,
            "name": self.inv_name,
            "rewritten": self.rewritten.to_string(),
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


pub fn compression(
    programs_expr: &Expr,
    args: &CompressionArgs,
    out_dir: &str,
) -> Vec<CompressionStepResult> {
    let mut rewritten: Expr = programs_expr.clone();
    let mut invs: Vec<CompressionStepResult> = Default::default();

    let tstart = std::time::Instant::now();

    for i in 0..args.iterations {
        println!("\n=======Iteration {}=======",i);
        let inv_name = format!("inv{}",invs.len());
        let res: Vec<CompressionStepResult> = compression_step(&rewritten, args, out_dir, &inv_name, programs_expr.cost());
        if !res.is_empty() {
            let res: CompressionStepResult = res[0].clone();
            rewritten = res.rewritten.clone();
            println!("Chose Invention {}: {}\n{}", res.inv_name, res, res.rewritten);
            invs.push(res);
        } else {
            println!("No inventions found at iteration {}",i);
            break;
        }
    }

    println!("\n=======Compression Summary=======");
    println!("Found {} inventions", invs.len());
    println!("Cost Improvement: ({:.2}x better) {} -> {}", compression_factor(programs_expr,&rewritten), programs_expr.cost(), rewritten.cost());
    for i in 0..invs.len() {
        let inv = &invs[i];
        println!("{} ({:.2}x wrt orig): {}" ,inv.inv_name, compression_factor(programs_expr, &inv.rewritten), inv);
    }
    println!("Time: {}ms", tstart.elapsed().as_millis());

    let out = json!({
        "cmd": std::env::args().join(" "),
        "args": args,
        "original_cost": programs_expr.cost(),
        "original": programs_expr.to_string(),
        "invs": invs.iter().map(|inv| inv.json()).collect::<Vec<serde_json::Value>>(),
    });

    std::fs::write(&args.out, serde_json::to_string_pretty(&out).unwrap()).unwrap();

    // serde_json::to_writer(&std::fs::File::create(&args.out).unwrap(), &out).unwrap();
    println!("Wrote to {:?}",args.out);
    invs
}