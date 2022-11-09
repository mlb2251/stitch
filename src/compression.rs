use crate::*;
use crate::egraphs::EGraph;
use lambdas::*;
use rustc_hash::{FxHashMap,FxHashSet};
use std::fmt::{self, Formatter, Display};
use std::hash::Hash;
use itertools::Itertools;
use rewriting::extract;
use serde_json::json;
use clap::{Parser};
use serde::Serialize;
use std::thread;
use std::sync::Arc;
use parking_lot::Mutex;
use std::ops::DerefMut;
use std::collections::BinaryHeap;
use rand::Rng;

/// Args for compression step
#[derive(Parser, Debug, Serialize, Clone)]
#[clap(name = "Stitch")]
pub struct CompressionStepConfig {
    /// max arity of abstractions to find (will find all from 0 to this number inclusive)
    #[clap(short='a', long, default_value = "2")]
    pub max_arity: usize,

    /// number of threads (no parallelism if set to 1)
    #[clap(short='t', long, default_value = "1")]
    pub threads: usize,

    /// Disable stat logging - note that stat logging in multithreading requires taking a mutex
    /// so it can be a source of slowdown in the massively multithreaded case, hence this flag to disable it.
    #[clap(long)]
    pub no_stats: bool,

    /// how many worklist items a thread will take at once
    #[clap(short='b', long, default_value = "1")]
    pub batch: usize,

    /// threads will autoadjust how large their batches are based on the worklist size
    #[clap(long)]
    pub dynamic_batch: bool,

    /// Number of invention candidates compression_step should return in a *single* step. Note that
    /// these will be the top n optimal candidates modulo subsumption pruning (and the top-1  is guaranteed
    /// to be globally optimal)
    #[clap(short='n', long, default_value = "1")]
    pub inv_candidates: usize,

    /// Method for choosing hole to expand at each step, doesn't have a huge effect
    #[clap(long, arg_enum, default_value = "depth-first")]
    pub hole_choice: HoleChoice,

    /// disables the safety check for the utility being correct; you only want
    /// to do this if you truly dont mind unsoundness for a minute
    #[clap(long)]
    pub no_mismatch_check: bool,

    /// makes it so inventions cant start with a lambda at the top
    #[clap(long)]
    pub no_top_lambda: bool,

    /// for debugging: pattern or abstraction to track
    #[clap(long)]
    pub track: Option<String>,

    /// for debugging: prunes all branches except the one that leads to the `--track` abstraction
    #[clap(long)]
    pub follow_track: bool,

    /// prints every worklist item as it is processed (will slow things down a ton due to rendering out expressins)
    #[clap(long)]
    pub verbose_worklist: bool,
    
    /// prints whenever a new best abstraction is found
    #[clap(long)]
    pub verbose_best: bool,

    /// print stats this often (0 means never)
    #[clap(long, default_value = "0")]
    pub print_stats: usize,

    /// print out programs rewritten under abstraction
    #[clap(long,short='r')]
    pub show_rewritten: bool,

    /// disables the edge case handling where argument capture needs to be inverted for optimality
    #[clap(long,short='r')]
    pub no_inv_arg_cap: bool,

    /// disable the free variable pruning optimization
    #[clap(long)]
    pub no_opt_free_vars: bool,

    /// disable the single structurally hashed subtree match pruning
    #[clap(long)]
    pub no_opt_single_use: bool,

    /// disable the single task pruning optimization
    #[clap(long)]
    pub no_opt_single_task: bool,

    /// disable the upper bound pruning optimization
    #[clap(long)]
    pub no_opt_upper_bound: bool,

    /// disable the force multiuse pruning optimization
    #[clap(long)]
    pub no_opt_force_multiuse: bool,

    /// disable the useless abstraction pruning optimization 
    #[clap(long)]
    pub no_opt_useless_abstract: bool,

    /// disable the arity zero priming optimization
    #[clap(long)]
    pub no_opt_arity_zero: bool,

    /// makes it so utility is based purely on corpus size without adding
    /// in the abstraction size
    #[clap(long)]
    pub no_other_util: bool,

    /// whenever you finish an invention do a full rewrite to check
    /// that rewriting doesnt raise a cost mismatch exception
    #[clap(long)]
    pub rewrite_check: bool,

    /// calculate utility exhaustively by performing a full rewrite;
    /// mainly used when cost mismatches are happening and we need something slow but accurate
    #[clap(long)]
    pub utility_by_rewrite: bool,

    /// anything related to running a dreamcoder comparison
    #[clap(long)]
    pub dreamcoder_comparison: bool,
    
}

impl CompressionStepConfig {
    pub fn no_opt(&mut self) {
        self.no_opt_free_vars = true;
        self.no_opt_single_task = true;
        self.no_opt_upper_bound = true;
        self.no_opt_force_multiuse = true;
        self.no_opt_useless_abstract = true;
        self.no_opt_arity_zero = true;
    }
}



/// A Pattern is a partial invention with holes. The simplest pattern is the single hole `??` which
/// matches at all nodes in the program set. From this single hole in a top-down manner we grow more complex
/// patterns like `(+ ?? ??)` and `(+ 3 (* ?? ??))`. Expanding a hole in a pattern always results in a pattern
/// that matches at a subset of the places that the original pattern matched.
/// 
/// `match_locations` is the list of structurally hashed nodes where the pattern matches.
/// `holes` is the list of zippers that point from the root of the pattern to the holes.
/// `arg_choices` is the same as `holes` but for the invention arguments like #i
/// `body_utility` is the cost of the non-hole non-argchoice parts of the pattern so far
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pattern {
    pub holes: Vec<ZId>, // in order of when theyre added NOT left to right
    arg_choices: Vec<LabelledZId>, // a hole gets moved into here when it becomes an argchoice, again these are in order of when they were added
    pub first_zid_of_ivar: Vec<ZId>, //first_zid_of_ivar[i] gives the index of the first use of #i in arg_choices
    pub match_locations: Vec<Id>, // places where it applies
    pub utility_upper_bound: i32,
    pub body_utility: i32, // the size (in `cost`) of a single use of the pattern body so far
    pub tracked: bool, // for debugging
}

#[allow(clippy::ptr_arg)]
fn zipper_replace(expr: &Expr, zip: &Zip, new: &str) -> Expr {
    let child = apply_zipper(expr,zip).unwrap();
    // clone and overwrite that node
    let mut res = expr.clone();
    res.nodes[usize::from(child)] = Lambda::Prim(new.into());
    res
}
/// replaces the node at the end of the zipper with `new` prim,
/// returning the new expression
#[allow(clippy::ptr_arg)]
fn apply_zipper(expr: &Expr, zip: &Zip) -> Option<Id> {
    let mut child = expr.root();
    for znode in zip.iter() {
        child = match (znode, expr.get(child)) {
            (ZNode::Body, Lambda::Lam([b])) => *b,
            (ZNode::Func, Lambda::App([f,_])) => *f,
            (ZNode::Arg, Lambda::App([_,x])) => *x,
            (_,_) => return None // no zipper works here
        };
    }
    Some(child)
}


/// returns the vec of zippers to each ivar
fn zids_of_ivar_of_expr(expr: &Expr, zid_of_zip: &FxHashMap<Zip,ZId>) -> Vec<Vec<ZId>> {

    // quickly determine arity
    let mut arity = 0;
    for node in expr.nodes.iter() {
        if let Lambda::IVar(ivar) = node {
            if ivar + 1 > arity {
                arity = ivar + 1;
            }
        }
    }

    let mut curr_zip: Zip = vec![];
    let mut zids_of_ivar = vec![vec![]; arity as usize];

    fn helper(curr_node: Id, expr: &Expr, curr_zip: &mut Zip, zids_of_ivar: &mut Vec<Vec<ZId>>, zid_of_zip: &FxHashMap<Zip,ZId>) {
        match expr.get(curr_node) {
            Lambda::Prim(_) => {},
            Lambda::Var(_) => {},
            Lambda::IVar(i) => {
                zids_of_ivar[*i as usize].push(zid_of_zip[curr_zip]);
            },
            Lambda::Lam([b]) => {
                curr_zip.push(ZNode::Body);
                helper(*b, expr, curr_zip, zids_of_ivar, zid_of_zip);
                curr_zip.pop();
            }
            Lambda::App([f,x]) => {
                curr_zip.push(ZNode::Func);
                helper(*f, expr, curr_zip, zids_of_ivar, zid_of_zip);
                curr_zip.pop();
                curr_zip.push(ZNode::Arg);
                helper(*x, expr, curr_zip, zids_of_ivar, zid_of_zip);
                curr_zip.pop();
            }
            _ => unreachable!(),
        }
        
    }
    // we can pick any match location
    helper(expr.root(), expr, &mut curr_zip, &mut zids_of_ivar, zid_of_zip);

    zids_of_ivar
}


impl Pattern {
    /// create a single hole pattern `??`
    //#[inline(never)]
    fn single_hole(treenodes: &[Id], cost_of_node_all: &[i32], num_paths_to_node: &[i32], egraph: &EGraph, cfg: &CompressionStepConfig) -> Self {
        let body_utility = 0;
        let mut match_locations = treenodes.to_owned();
        match_locations.sort(); // we assume match_locations is always sorted
        if cfg.no_top_lambda {
            match_locations.retain(|node| expands_to_of_node(&egraph[*node].nodes[0]) != ExpandsTo::Lam);
        }
        let utility_upper_bound = utility_upper_bound(&match_locations, body_utility, cost_of_node_all, num_paths_to_node, cfg);
        Pattern {
            holes: vec![EMPTY_ZID], // (zid 0 is the empty zipper)
            arg_choices: vec![],
            first_zid_of_ivar: vec![],
            match_locations, // single hole matches everywhere
            utility_upper_bound,
            body_utility, // 0 body utility
            tracked: cfg.track.is_some(),
        }
    }
    /// convert pattern to an Expr with `??` in place of holes and `?#` in place of argchoices
    fn to_expr(&self, shared: &SharedData) -> Expr {
        let mut curr_zip: Zip = vec![];
        // map zids to zips with a bool thats true if this is a hole and false if its a future ivar
        let zips: Vec<(Zip,Expr)> = self.holes.iter().map(|zid| (shared.zip_of_zid[*zid].clone(), Expr::prim("??".into())))
            .chain(self.arg_choices.iter()
            .map(|labelled_zid| (shared.zip_of_zid[labelled_zid.zid].clone(), Expr::ivar(labelled_zid.ivar as i32)))).collect();


        fn helper(curr_node: Id, curr_zip: &mut Zip, zips: &[(Zip,Expr)], shared: &SharedData) -> Expr {
            match zips.iter().find(|(zip,_)| zip == curr_zip) {
                // current zip matches a hole
                Some((_,e)) => e.clone(),
                // no ivar zip match, so recurse
                None => {
                    match &shared.node_of_id[usize::from(curr_node)] {
                        Lambda::Prim(p) => Expr::prim(*p),
                        Lambda::Var(v) => Expr::var(*v),
                        Lambda::Lam([b]) => {
                            curr_zip.push(ZNode::Body);
                            let b_expr = helper(*b, curr_zip, zips, shared);
                            curr_zip.pop();
                            Expr::lam(b_expr) 
                        }
                        Lambda::App([f,x]) => {
                            curr_zip.push(ZNode::Func);
                            let f_expr = helper(*f, curr_zip, zips, shared);
                            curr_zip.pop();
                            curr_zip.push(ZNode::Arg);
                            let x_expr = helper(*x, curr_zip, zips, shared);
                            curr_zip.pop();
                            Expr::app(f_expr, x_expr)
                        }
                        _ => unreachable!(),
                    }
                }
            }
            
        }
        // we can pick any match location
        helper(self.match_locations[0], &mut curr_zip, &zips, shared)
    }
    fn show_track_expansion(&self, hole_zid: ZId, shared: &SharedData) -> String {
        let mut s = zipper_replace(&self.to_expr(shared), &shared.zip_of_zid[hole_zid], "<REPLACE>" ).to_string();
        s = s.replace(&"<REPLACE>", &format!("{}",tracked_expands_to(self, hole_zid, shared)).magenta().bold().to_string());
        s
    }
    pub fn info(&self, shared: &SharedData) -> String {
        format!("{}: utility_upper_bound={}, body_utility={}, match_locations={}, usages={}",self.to_expr(shared), self.utility_upper_bound, self.body_utility, self.match_locations.len(), self.match_locations.iter().map(|loc|shared.num_paths_to_node[usize::from(*loc)]).sum::<i32>())
    }
}

/// The child-ignoring value of a node in the original set of programs. This tells us
/// what the hole will expand into at this node.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum ExpandsTo {
    Lam,
    App,
    Var(i32),
    Prim(Symbol),
    IVar(i32),
}

impl ExpandsTo {
    #[inline]
    /// true if expanding a node of this ExpandsTo will yield new holes
    #[allow(dead_code)]
    fn has_holes(&self) -> bool {
        match self {
            ExpandsTo::Lam => true,
            ExpandsTo::App => true,
            ExpandsTo::Var(_) => false,
            ExpandsTo::Prim(_) => false,
            ExpandsTo::IVar(_) => false,
        }
    }
    #[inline]
    #[allow(dead_code)]
    fn is_ivar(&self) -> bool {
        matches!(self, ExpandsTo::IVar(_))
    }
}

impl std::fmt::Display for ExpandsTo {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ExpandsTo::Lam => write!(f, "(lam ??)"),
            ExpandsTo::App => write!(f, "(?? ??)"),
            ExpandsTo::Var(v) => write!(f, "${}", v),
            ExpandsTo::Prim(p) => write!(f, "{}", p),
            ExpandsTo::IVar(v) => write!(f, "#{}", v),
        }
    }
}

/// a list of znodes, representing a path through a tree (a zipper)
pub type Zip = Vec<ZNode>;
/// the index of the empty zid `[]` in the list of zippers
const EMPTY_ZID: ZId = 0;

/// an argument to an abstraction. `id` is the main field here, we can use
/// it to lookup the corresponding tree using egraph[id]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Arg {
    pub shifted_id: Id,
    pub unshifted_id: Id, // in case `id` was shifted to make it an arg not sure if this will end up being useful
    pub shift: i32,
    pub cost: i32,
    pub expands_to: ExpandsTo,
}

/// ExpandsTo from a &Lambda node. Returns None if this is
/// and IVar (which is not considered a node type) and crashes
/// on Programs node.
fn expands_to_of_node(node: &Lambda) -> ExpandsTo {
    match node {
        Lambda::Var(i) => ExpandsTo::Var(*i),
        Lambda::Prim(p) => {
            if *p == Symbol::from("?#") {
                panic!("I still need to handle this") // todo
            } else {
                ExpandsTo::Prim(*p)
            }
        },
        Lambda::Lam(_) => ExpandsTo::Lam,
        Lambda::App(_) => ExpandsTo::App,
        Lambda::IVar(i) => ExpandsTo::IVar(*i),
        _ => unreachable!()
    }
}

/// Returns Some(ExpandsTo) for what we expect the hole to expand to to follow
/// the target, and returns None if we expect it to become a ?# argchoice.
fn tracked_expands_to(pattern: &Pattern, hole_zid: ZId, shared: &SharedData) -> ExpandsTo {
    // apply the hole zipper to the original expr being tracked to get the subtree
    // this will expand into, then get the ExpandsTo of that
    let id =  apply_zipper(&shared.tracking.as_ref().unwrap().expr, &shared.zip_of_zid[hole_zid]).unwrap();
    match expands_to_of_node(shared.tracking.as_ref().unwrap().expr.get(id)) {
        ExpandsTo::IVar(i) => {
            // in the case where we're searching for an IVar we need to be robust to relabellings
            // since this doesn't have to be canonical. What we can do is we can look over
            // each ivar the the pattern has defined with a first zid in pattern.first_zid_of_ivar, and
            // if our expressions' zids_of_ivar[i] contains this zid then we know these two ivars
            // must correspond to each other in the pattern and the tracked expr and we can just return
            // the pattern version (`j` below).
            let zids = shared.tracking.as_ref().unwrap().zids_of_ivar[i as usize].clone();
            for (j,zid) in pattern.first_zid_of_ivar.iter().enumerate() {
                if zids.contains(zid) {
                    return ExpandsTo::IVar(j as i32);
                }
            }
            // it's a new ivar that hasnt been used already so it must take on the next largest var number
            ExpandsTo::IVar(pattern.first_zid_of_ivar.len() as i32)
        }
        e => e
    }
}

/// The heap item used for heap-based worklists. Holds a pattern
#[derive(Debug,Clone, Eq, PartialEq)]
pub struct HeapItem {
    key: i32,
    pattern: Pattern,
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.key.partial_cmp(&other.key)
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}
impl HeapItem {
    fn new(pattern: Pattern) -> Self {
        HeapItem {
            // key: pattern.body_utility * pattern.match_locations.iter().map(|loc|num_paths_to_node[loc]).sum::<i32>(),
            key: pattern.utility_upper_bound,
            // system time is suuuper slow btw you want to do something else
            // key: std::time::SystemTime::now().duration_since(std::time::SystemTime::UNIX_EPOCH).unwrap().as_nanos() as i32,
            pattern
        }
    }
}


/// This is the multithread data locked during the critical section of the algorithm.
#[derive(Debug, Clone)]
pub struct CriticalMultithreadData {
    donelist: Vec<FinishedPattern>,
    worklist: BinaryHeap<HeapItem>,
    utility_pruning_cutoff: i32,
    active_threads: FxHashSet<std::thread::ThreadId>, // list of threads currently holding worklist items
}

/// All the data shared among threads, mostly read-only
/// except for the mutexes
#[derive(Debug)]
pub struct SharedData {
    pub crit: Mutex<CriticalMultithreadData>,
    pub arg_of_zid_node: Vec<FxHashMap<Id,Arg>>,
    pub cost_fn: ProgramCost,
    pub treenodes: Vec<Id>,
    pub node_of_id: Vec<Lambda>,
    pub programs_node: Id,
    pub roots: Vec<Id>,
    pub zids_of_node: FxHashMap<Id,Vec<ZId>>,
    pub zip_of_zid: Vec<Zip>,
    pub zid_of_zip: FxHashMap<Zip, ZId>,
    pub extensions_of_zid: Vec<ZIdExtension>,
    pub egraph: EGraph,
    pub num_paths_to_node: Vec<i32>,
    pub num_paths_to_node_by_root_idx: Vec<Vec<i32>>,
    pub tasks_of_node: Vec<FxHashSet<usize>>,
    pub task_name_of_task: Vec<String>,
    pub task_of_root_idx: Vec<usize>,
    pub root_idxs_of_task: Vec<Vec<usize>>,
    pub cost_of_node_once: Vec<i32>,
    pub cost_of_node_all: Vec<i32>,
    pub free_vars_of_node: Vec<FxHashSet<i32>>,
    pub init_cost: i32,
    pub init_cost_by_root_idx: Vec<i32>,
    pub first_train_cost: i32,
    pub stats: Mutex<Stats>,
    pub cfg: CompressionStepConfig,
    pub tracking: Option<Tracking>,
}

/// Used for debugging tracking information
#[derive(Debug)]
pub struct Tracking {
    expr: Expr,
    zids_of_ivar: Vec<Vec<ZId>>,
}

impl CriticalMultithreadData {
    /// Create a new mutable multithread data struct with
    /// a worklist that just has a single hole on it
    fn new(donelist: Vec<FinishedPattern>, treenodes: &[Id], cost_of_node_all: &[i32], num_paths_to_node: &[i32], egraph: &EGraph, cfg: &CompressionStepConfig) -> Self {
        // push an empty hole onto a new worklist
        let mut worklist = BinaryHeap::new();
        worklist.push(HeapItem::new(Pattern::single_hole(treenodes, cost_of_node_all, num_paths_to_node, egraph, cfg)));
        
        let mut res = CriticalMultithreadData {
            donelist,
            worklist,
            utility_pruning_cutoff: 0,
            active_threads: FxHashSet::default(),
        };
        res.update(cfg);
        res
    }
    /// sort the donelist by utility, truncate to cfg.inv_candidates, update 
    /// update utility_pruning_cutoff to be the lowest utility
    //#[inline(never)]
    fn update(&mut self, cfg: &CompressionStepConfig) {
        // sort in decreasing order by utility primarily, and break ties using the argchoice zids (just in order to be deterministic!)
        // let old_best = self.donelist.first().map(|x|x.utility).unwrap_or(0);
        self.donelist.sort_unstable_by(|a,b| (b.utility,&b.pattern.arg_choices).cmp(&(a.utility,&a.pattern.arg_choices)));
        self.donelist.truncate(cfg.inv_candidates);
        // the cutoff is the lowest utility
        self.utility_pruning_cutoff = if cfg.no_opt_upper_bound { 0 } else { std::cmp::max(0,self.donelist.last().map(|x|x.utility).unwrap_or(0)) };
    }
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
        let map: FxHashMap<i32, Expr> = args.iter().enumerate().map(|(i,e)| (i as i32, e.clone())).collect();
        ivar_replace(&self.body, self.body.root(), &map)
    }
}

impl Display for Invention {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "[{} arity={}: {}]", self.name, self.arity, self.body)
    }
}

/// A node in an ZPath
/// Ord: Func < Body < Arg
#[derive(Debug, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum ZNode {
    // * order of variants here is important because the derived Ord will use it
    Func, // zipper went into the function, so Id is the arg
    Body, 
    Arg, // zipper went into the arg, so Id is the function
}

/// "zipper id" each unique zipper gets referred to by its zipper id
pub type ZId = usize;

/// a zid referencing a specific ZPath and a #i index
#[derive(Debug,Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
struct LabelledZId {
    zid: ZId,
    ivar: usize // which #i argument this is, which also corresponds to args[i] ofc
}

/// Various tracking stats
#[derive(Clone,Default, Debug)]
pub struct Stats {
    worklist_steps: usize,
    finished: usize,
    calc_final_utility: usize,
    calc_unargcap: usize,
    donelist_push: usize,
    azero_calc_util: usize,
    azero_calc_unargcap: usize,
    upper_bound_fired: usize,
    // conflict_upper_bound_fired: usize,
    free_vars_fired: usize,
    single_use_fired: usize,
    single_task_fired: usize,
    useless_abstract_fired: usize,
    force_multiuse_fired: usize,
}

/// a strategy for choosing which hole to expand next in a partial pattern
#[derive(Debug, Clone, clap::ArgEnum, Serialize)]
pub enum HoleChoice {
    Random,
    BreadthFirst,
    DepthFirst,
    MaxLargestSubset,
    HighEntropy,
    LowEntropy,
    MaxCost,
    MinCost,
    ManyGroups,
    FewGroups,
    FewApps,
}

impl HoleChoice {
    //#[inline(never)]
    fn choose_hole(&self, pattern: &Pattern, shared: &SharedData) -> usize {
        if pattern.holes.len() == 1 {
            return 0;
        }
        match *self {
            HoleChoice::BreadthFirst => 0,
            HoleChoice::DepthFirst => pattern.holes.len() - 1,
            HoleChoice::Random => {
                let mut rng = rand::thread_rng();
                rng.gen_range(0..pattern.holes.len())
            },
            HoleChoice::FewApps => {
                pattern.holes.iter().enumerate().map(|(hole_idx,hole_zid)|
                    (hole_idx, pattern.match_locations.iter().filter(|loc|shared.arg_of_zid_node[*hole_zid][loc].expands_to == ExpandsTo::App).count()))
                        .min_by_key(|x|x.1).unwrap().0
            }
            HoleChoice::MaxCost => {
                pattern.holes.iter().enumerate().map(|(hole_idx,hole_zid)|
                    (hole_idx, pattern.match_locations.iter().map(|loc|shared.arg_of_zid_node[*hole_zid][loc].cost).sum::<i32>()))
                        .max_by_key(|x|x.1).unwrap().0
            }
            HoleChoice::MinCost => {
                pattern.holes.iter().enumerate().map(|(hole_idx,hole_zid)|
                    (hole_idx, pattern.match_locations.iter().map(|loc|shared.arg_of_zid_node[*hole_zid][loc].cost).sum::<i32>()))
                        .min_by_key(|x|x.1).unwrap().0
            }
            HoleChoice::MaxLargestSubset => {
                // todo warning this is extremely slow, partially bc of counts() but I think
                // mainly because where there are like dozens of holes doing all these lookups and clones and hashmaps is a LOT
                pattern.holes.iter().enumerate()
                    .map(|(hole_idx,hole_zid)| (hole_idx, *pattern.match_locations.iter()
                        .map(|loc| shared.arg_of_zid_node[*hole_zid][loc].expands_to.clone()).counts().values().max().unwrap())).max_by_key(|&(_,max_count)| max_count).unwrap().0
            }
            _ => unimplemented!()
        }
    }
}

impl LabelledZId {
    fn new(zid: ZId, ivar: usize) -> LabelledZId {
        LabelledZId { zid, ivar }
    }
}

/// tells you which zid if any you would get if you extended the depth
/// (of whatever the current zid is) with any of these znodes.
#[derive(Clone,Debug)]
pub struct ZIdExtension {
    body: Option<ZId>,
    arg: Option<ZId>,
    func: Option<ZId>,
}

/// empties worklist_buf and donelist_buf into the shared worklist while holding the mutex, updates
/// the donelist and cutoffs, and grabs and returns a new worklist item along with new cutoff bounds.
//#[inline(never)]
fn get_worklist_item(
    worklist_buf: &mut Vec<HeapItem>,
    donelist_buf: &mut Vec<FinishedPattern>,
    shared: &Arc<SharedData>,
) -> Option<(Vec<Pattern>,i32)> {

    // * MULTITHREADING: CRITICAL SECTION START *
    // take the lock, which will be released immediately when this scope exits
    let mut shared_guard = shared.crit.lock();
    let mut crit: &mut CriticalMultithreadData = shared_guard.deref_mut();
    let old_best_utility = crit.donelist.first().map(|x|x.utility).unwrap_or(0);
    let old_donelist_len = crit.donelist.len();
    let old_utility_pruning_cutoff = crit.utility_pruning_cutoff;
    // drain from donelist_buf into the actual donelist
    crit.donelist.extend(donelist_buf.drain(..).filter(|done| done.utility > old_utility_pruning_cutoff));
    if !shared.cfg.no_stats { shared.stats.lock().deref_mut().finished += crit.donelist.len() - old_donelist_len; };
    // sort + truncate + update utility_pruning_cutoff
    crit.update(&shared.cfg); // this also updates utility_pruning_cutoff

    if shared.cfg.verbose_best && crit.donelist.first().map(|x|x.utility).unwrap_or(0) > old_best_utility {

        let new_expected_cost = shared.first_train_cost - crit.donelist.first().unwrap().compressive_utility + crit.donelist.first().unwrap().to_expr(&shared).cost(&shared.cost_fn);
        let trainratio = shared.first_train_cost as f64 / new_expected_cost as f64;
        // println!("{} @ step={} util={} trainratio={:.2} for {}", "[new best utility]".blue(), shared.stats.lock().deref_mut().worklist_steps, shared.first_train_cost as f64/ new_expected_cost as f64, crit.donelist.first().unwrap().info(shared));
        println!("{} @ step={} util={} trainratio={:.2} for {}", "[new best utility]".blue(), shared.stats.lock().deref_mut().worklist_steps, crit.donelist.first().unwrap().utility, trainratio, crit.donelist.first().unwrap().info(shared));
    }

    // pull out the newer version of this now that its been updated, since we're returning it at the end
    let mut utility_pruning_cutoff = crit.utility_pruning_cutoff;

    let old_worklist_len = crit.worklist.len();
    let worklist_buf_len = worklist_buf.len();
    // drain from worklist_buf into the actual worklist
    crit.worklist.extend(worklist_buf.drain(..).filter(|heap_item| heap_item.pattern.utility_upper_bound > utility_pruning_cutoff));
    // num pruned by upper bound = num we were gonna add minus change in worklist length
    if !shared.cfg.no_stats { shared.stats.lock().deref_mut().upper_bound_fired += worklist_buf_len - (crit.worklist.len() - old_worklist_len); };

    let mut returned_items = vec![];

    // try to get a new worklist item
    crit.active_threads.remove(&thread::current().id()); // remove ourself from the active threads
    // println!("worklist len: {}", crit.worklist.len());

    loop {
        // with dynamic batch size, take worklist_size/num_threads items from the worklist
        let batch_size = if shared.cfg.dynamic_batch { std::cmp::max(1, crit.worklist.len() / shared.cfg.threads ) } else { shared.cfg.batch };
        while crit.worklist.is_empty() {
            if !returned_items.is_empty() {
                // give up and return whatever we've got
                crit.active_threads.insert(thread::current().id());
                return Some((returned_items, utility_pruning_cutoff));
            }
            if crit.active_threads.is_empty() {
                return None // all threads are stuck waiting for work so we're all done
            }
            // the worklist is empty but someone else currently has a worklist item so we should give up our lock then take it back
            drop(shared_guard);
            shared_guard = shared.crit.lock();
            crit = shared_guard.deref_mut();
            // update our cutoff in case it changed
            utility_pruning_cutoff = crit.utility_pruning_cutoff;
        }
        
        let heap_item = crit.worklist.pop().unwrap();
        // prune if upper bound is too low (cutoff may have increased in the time since this was added to the worklist)
        if shared.cfg.no_opt_upper_bound || heap_item.pattern.utility_upper_bound > utility_pruning_cutoff {
            // we got one!
            returned_items.push(heap_item.pattern);
            if returned_items.len() == batch_size {
                // we got enough, so return it
                crit.active_threads.insert(thread::current().id());
                return Some((returned_items, utility_pruning_cutoff));
            }
        } else if !shared.cfg.no_stats { shared.stats.lock().deref_mut().upper_bound_fired += 1; }
    }
    // * MULTITHREADING: CRITICAL SECTION END *
}

// pub fn blackbox1<T>(dummy: T) -> T {
//     unsafe {
//         let ret = std::ptr::read_volatile(&dummy);
//         std::mem::forget(dummy);
//         ret
//     }
// }
// pub fn blackbox2<T>(dummy: T) -> T {
//     unsafe {
//         let ret = std::ptr::read_volatile(&dummy);
//         std::mem::forget(dummy);
//         ret
//     }
// }
// pub fn blackbox3<T>(dummy: T) -> T {
//     unsafe {
//         let ret = std::ptr::read_volatile(&dummy);
//         std::mem::forget(dummy);
//         ret
//     }
// }
// pub fn blackbox<T>(dummy: T) -> T {
//     unsafe {
//         let ret = std::ptr::read_volatile(&dummy);
//         std::mem::forget(dummy);
//         ret
//     }
// }

/// The core top down branch and bound search
fn stitch_search(
    shared: Arc<SharedData>,
) {
    
    // local buffers to eventually pour into the global worklist and donelist when we take the mutex
    let mut worklist_buf: Vec<HeapItem> = Default::default();
    let mut donelist_buf: Vec<_> = Default::default();

    loop {

        // get a new worklist item along with pruning cutoffs
        let (patterns, mut weak_utility_pruning_cutoff) =
            match get_worklist_item(
                &mut worklist_buf,
                &mut donelist_buf,
                &shared,
            ) {
                Some(pattern) => pattern,
                None => return,
        };

        for original_pattern in patterns {

            if !shared.cfg.no_stats { shared.stats.lock().deref_mut().worklist_steps += 1; };
            if !shared.cfg.no_stats && shared.cfg.print_stats > 0 &&  shared.stats.lock().deref_mut().worklist_steps % shared.cfg.print_stats == 0 { println!("{:?} \n\t@ [bound={}; uses={}] chose: {}",shared.stats.lock().deref_mut(),   original_pattern.utility_upper_bound, original_pattern.match_locations.iter().map(|loc| shared.num_paths_to_node[usize::from(*loc)]).sum::<i32>(), original_pattern.to_expr(&shared)); };

            if shared.cfg.verbose_worklist {
                println!("[bound={}; uses={}] chose: {}", original_pattern.utility_upper_bound, original_pattern.match_locations.iter().map(|loc| shared.num_paths_to_node[usize::from(*loc)]).sum::<i32>(), original_pattern.to_expr(&shared));
            }

            // choose which hole we're going to expand
            let hole_idx: usize = shared.cfg.hole_choice.choose_hole(&original_pattern, &shared);

            // pop that hole form the list of holes
            let mut holes_after_pop: Vec<ZId> = original_pattern.holes.clone();
            let hole_zid: ZId = holes_after_pop.remove(hole_idx);

            // get the hashmap of args for this hole
            let arg_of_loc = &shared.arg_of_zid_node[hole_zid];

            // sort the match locations by node type (ie what theyll expand into) so that we can do a group_by() on
            // node type in order to iterate over all the different expansions
            // We also sort secondarily by `loc` to ensure each groupby subsequence has the locations in sorted order
            let mut match_locations = original_pattern.match_locations.clone();
            match_locations.sort_by_cached_key(|loc| (&arg_of_loc[loc].expands_to, *loc));

            let ivars_expansions = get_ivars_expansions(&original_pattern, arg_of_loc, &shared);

            let mut found_tracked = false;
            // for each way of expanding the hole...

            'expansion:
                for (expands_to, locs) in match_locations.into_iter()
                .group_by(|loc| &arg_of_loc[loc].expands_to).into_iter()
                .map(|(expands_to, locs)| (expands_to.clone(), locs.collect::<Vec<Id>>()))
                .chain(ivars_expansions.into_iter())
            {
                // for debugging
                let tracked = original_pattern.tracked && expands_to == tracked_expands_to(&original_pattern, hole_zid, &shared);
                if tracked { found_tracked = true; }
                if shared.cfg.follow_track && !tracked { continue 'expansion; }


                // prune inventions that only match at a single unique (structurally hashed) subtree. This only applies if we
                // also are priming with arity 0 inventions. Basically if something only matches at one subtree then the best you can
                // do is the arity zero invention which is the whole subtree, and since we already primed with arity 0 inventions we can
                // prune here. The exception is when there are free variables so arity 0 wouldn't have applied.
                // Also, note that upper bounding + arity 0 priming does nearly perfectly handle this already, but there are cases where
                // you can't improve your structure penalty bound enough to catch everything hence this separate single_use thing.
                if !shared.cfg.no_opt_single_use && !shared.cfg.no_opt_arity_zero && locs.len()  == 1 && shared.free_vars_of_node[usize::from(locs[0])].is_empty() {
                    if !shared.cfg.no_stats { shared.stats.lock().deref_mut().single_use_fired += 1; }
                    continue 'expansion;
                }

                // prune inventions specific to one single task
                if !shared.cfg.no_opt_single_task
                        && locs.iter().all(|node| shared.tasks_of_node[usize::from(*node)].len() == 1)
                        && locs.iter().all(|node| shared.tasks_of_node[usize::from(locs[0])].iter().next() == shared.tasks_of_node[usize::from(*node)].iter().next()) {
                    if !shared.cfg.no_stats { shared.stats.lock().deref_mut().single_task_fired += 1; }
                    if tracked { println!("{} single task pruned when expanding {} to {}", "[TRACK]".red().bold(), original_pattern.to_expr(&shared), zipper_replace(&original_pattern.to_expr(&shared), &shared.zip_of_zid[hole_zid], &format!("<{}>",expands_to))); }
                    continue 'expansion;
                }

                // check for free variables: if an invention has free variables in the body then it's not a real function and we can discard it
                // Here we just check if our expansion just yielded a variable, and if that is bound based on how many lambdas there are above it.
                if true {  // TODO: condition should be "!shared.cfg.no_opt_free_vars" once this is no longer unsound
                    if let ExpandsTo::Var(i) = expands_to {
                        if i >= shared.zip_of_zid[hole_zid].iter().filter(|znode|**znode == ZNode::Body).count() as i32 {
                            if !shared.cfg.no_stats { shared.stats.lock().deref_mut().free_vars_fired += 1; };
                            if tracked { println!("{} pruned by free var in body when expanding {} to {}", "[TRACK]".red().bold(), original_pattern.to_expr(&shared), original_pattern.show_track_expansion(hole_zid, &shared)); }
                            continue 'expansion; // free var
                        }
                    }
                }

                // check for useless abstractions (ie ones that take the same arg everywhere). We check for this all the time, not just when adding a new variables,
                // because subsetting of match_locations can turn previously useful abstractions into useless ones.
                if !shared.cfg.no_opt_useless_abstract {
                    for argchoice in original_pattern.arg_choices.iter(){
                        // if its the same arg in every place
                        if locs.iter().map(|loc| shared.arg_of_zid_node[argchoice.zid][loc].shifted_id).all_equal()
                        {
                            if !shared.cfg.no_stats { shared.stats.lock().deref_mut().useless_abstract_fired += 1; };
                            continue 'expansion; // useless abstraction
                        }
                    }

                }


                // update the body utility
                let body_utility = original_pattern.body_utility +  match expands_to {
                    ExpandsTo::Lam | ExpandsTo::App => COST_NONTERMINAL,
                    ExpandsTo::Var(_) | ExpandsTo::Prim(_) => COST_TERMINAL,
                    ExpandsTo::IVar(_) => 0,
                };
                // update the upper bound
                let util_upper_bound: i32 = utility_upper_bound(&locs, body_utility, &shared.cost_of_node_all, &shared.num_paths_to_node, &shared.cfg);
                assert!(util_upper_bound <= original_pattern.utility_upper_bound);

                // branch and bound: if the upper bound is less than the best invention we've found so far (our cutoff), we can discard this pattern
                if !shared.cfg.no_opt_upper_bound && util_upper_bound <= weak_utility_pruning_cutoff {
                    if !shared.cfg.no_stats { shared.stats.lock().deref_mut().upper_bound_fired += 1; };
                    if tracked { println!("{} upper bound ({} < {}) pruned when expanding {} to {}", "[TRACK]".red().bold(), util_upper_bound, weak_utility_pruning_cutoff, original_pattern.to_expr(&shared), original_pattern.show_track_expansion(hole_zid, &shared)); }
                    continue 'expansion; // too low utility
                }

                // assert!(shared.cfg.no_opt_upper_bound || !holes_after_pop.is_empty() || !original_pattern.arg_choices.is_empty() || expands_to.has_holes() || expands_to.is_ivar(),
                        // "unexpected arity 0 invention: upper bounds + priming with arity 0 inventions should have prevented this");
                // assert!(shared.cfg.no_opt_upper_bound || (locs.len() > 1 || !shared.egraph[locs[0]].data.free_vars.is_empty()),
                //         "single-use pruning doesn't seem to be happening, it should be an automatic side effect of upper bounds + priming with arity zero inventions (as long as they dont have free vars)\n{}\n{}\n{}\n{}\n{}", original_pattern.to_expr(&shared), extract(locs[0], &shared.egraph), expands_to,  util_upper_bound, weak_utility_pruning_cutoff);

                // add any new holes to the list of holes
                let mut holes = holes_after_pop.clone();
                match expands_to {
                    ExpandsTo::Lam => {
                        // add new holes
                        holes.push(shared.extensions_of_zid[hole_zid].body.unwrap());
                    }
                    ExpandsTo::App => {
                        // add new holes
                            holes.push(shared.extensions_of_zid[hole_zid].func.unwrap());
                            holes.push(shared.extensions_of_zid[hole_zid].arg.unwrap());
                    }
                    _ => {}
                }

                let mut arg_choices = original_pattern.arg_choices.clone();
                let mut first_zid_of_ivar = original_pattern.first_zid_of_ivar.clone();
                if let ExpandsTo::IVar(i) = expands_to {
                    arg_choices.push(LabelledZId::new(hole_zid, i as usize));
                    if i as usize == original_pattern.first_zid_of_ivar.len() {
                        first_zid_of_ivar.push(hole_zid);
                    }
                }

                // if two different ivars #i and #j have the same arg at every location, then we can prune this pattern
                // because there must exist another pattern where theyre just both the same ivar. Note that this pruning
                // happens here and not just at the ivar creation point because new subsetting can happen
                if !shared.cfg.no_opt_force_multiuse {
                    // for all pairs of ivars #i and #j, get the first zipper and compare the arg value across all locations
                    for (i,ivar_zid_1) in first_zid_of_ivar.iter().enumerate() {
                        let arg_of_loc_1 = &shared.arg_of_zid_node[*ivar_zid_1];
                        for ivar_zid_2 in first_zid_of_ivar.iter().skip(i+1) {
                            let arg_of_loc_2 = &shared.arg_of_zid_node[*ivar_zid_2];
                            if locs.iter().all(|loc|
                                arg_of_loc_1[loc].shifted_id == arg_of_loc_2[loc].shifted_id)
                            {
                                if !shared.cfg.no_stats { shared.stats.lock().deref_mut().force_multiuse_fired += 1; };
                                if tracked { println!("{} force multiuse pruned when expanding {} to {}", "[TRACK]".red().bold(), original_pattern.to_expr(&shared), original_pattern.show_track_expansion(hole_zid, &shared)); }
                                continue 'expansion;
                            }
                        }
                    }
                }

            // build our new pattern with all the variables we've just defined. Copy in the argchoices and prefixes
            // from the old pattern.
            let new_pattern = Pattern {
                holes,
                arg_choices,
                first_zid_of_ivar,
                match_locations: locs,
                utility_upper_bound: util_upper_bound,
                body_utility,
                tracked
            };

            // new_pattern.utility_upper_bound = utility_upper_bound_with_conflicts(&new_pattern, body_utility_no_refinement + refinement_body_utility, &shared);
            // // branch and bound again
            // if !shared.cfg.no_opt_upper_bound && new_pattern.utility_upper_bound <= weak_utility_pruning_cutoff {
            //     if !shared.cfg.no_stats { shared.stats.lock().deref_mut().conflict_upper_bound_fired += 1; };
            //     if tracked { println!("{} upper bound ({} < {}) pruned when expanding {} to {}", "[TRACK]".red().bold(), util_upper_bound, weak_utility_pruning_cutoff, original_pattern.to_expr(&shared), original_pattern.show_track_expansion(hole_zid, &shared)); }
            //     continue 'expansion; // too low utility
            // }



            if new_pattern.holes.is_empty() {
                // it's a finished pattern

                let mut finished_pattern = FinishedPattern::new(new_pattern, &shared);

                if !shared.cfg.no_stats { shared.stats.lock().calc_final_utility += 1; };

                if finished_pattern.compressive_utility <= weak_utility_pruning_cutoff {
                    continue 'expansion // todo could add a tracked{} printing thing here
                }

                if !shared.cfg.no_stats { shared.stats.lock().calc_unargcap += 1; };
                inverse_argument_capture(&mut finished_pattern, &shared.cfg, &shared.zip_of_zid, &shared.node_of_id, &shared.cost_of_node_once, &shared.arg_of_zid_node, &shared.extensions_of_zid, &shared.egraph);

                if finished_pattern.utility <= weak_utility_pruning_cutoff {
                    continue 'expansion // todo could add a tracked{} printing thing here
                }

                if !shared.cfg.no_stats { shared.stats.lock().donelist_push += 1; };

                if shared.cfg.rewrite_check {
                    // run rewriting just to make sure the assert in it passes
                    rewrite_fast(&finished_pattern, &shared, "fake_inv", &shared.cost_fn);
                }

                if tracked {
                    println!("{} pushed {} to donelist (util: {})", "[TRACK:DONE]".green().bold(), finished_pattern.to_expr(&shared), finished_pattern.utility);
                }

                if shared.cfg.inv_candidates == 1 && finished_pattern.utility > weak_utility_pruning_cutoff {
                    // if we're only looking for one invention, we can directly update our cutoff here
                    weak_utility_pruning_cutoff = finished_pattern.utility;
                }

                donelist_buf.push(finished_pattern);

                } else {
                    // it's a partial pattern so just add it to the worklist
                    if tracked { println!("{} pushed {} to work list (bound: {})", "[TRACK]".green().bold(), original_pattern.show_track_expansion(hole_zid, &shared), new_pattern.utility_upper_bound); }
                    worklist_buf.push(HeapItem::new(new_pattern))
                }
            }

            if original_pattern.tracked && !found_tracked {
                // let new = format!("<{}>",tracked_expands_to(&original_pattern, hole_zid, &shared));
                // let mut s = original_pattern.to_expr(&shared).zipper_replace(&shared.zip_of_zid[hole_zid], &new ).to_string();
                // s = s.replace(&new, &new.clone().magenta().bold().to_string());
            println!("{} pruned when expanding because there were no match locations for the target expansion of {} to {}", "[TRACK]".red().bold(), original_pattern.to_expr(&shared), original_pattern.show_track_expansion(hole_zid, &shared));
            }
        
        }
    }

}

//#[inline(never)]
fn get_ivars_expansions(original_pattern: &Pattern, arg_of_loc: &FxHashMap<Id,Arg>, shared: &Arc<SharedData>) -> Vec<(ExpandsTo, Vec<Id>)> {
    let mut ivars_expansions = vec![];
    // consider all ivars used previously
    for ivar in 0..original_pattern.first_zid_of_ivar.len() {
        let arg_of_loc_ivar = &shared.arg_of_zid_node[original_pattern.first_zid_of_ivar[ivar]];
        let locs: Vec<Id> = original_pattern.match_locations.iter()
            .filter(|loc|
                arg_of_loc[loc].shifted_id == 
                arg_of_loc_ivar[loc].shifted_id).cloned().collect();
        if locs.is_empty() { continue; }
        ivars_expansions.push((ExpandsTo::IVar(ivar as i32), locs));
    }
    // also consider one ivar greater, if this is within the arity limit. This will match at all the same locations as the original.
    if original_pattern.first_zid_of_ivar.len() < shared.cfg.max_arity {
        let ivar = original_pattern.first_zid_of_ivar.len();
        let locs = original_pattern.match_locations.clone();
        ivars_expansions.push((ExpandsTo::IVar(ivar as i32), locs));
    }
    ivars_expansions
}


/// A finished invention
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinishedPattern {
    pub pattern: Pattern,
    pub utility: i32,
    pub compressive_utility: i32,
    pub util_calc: UtilityCalculation,
    pub arity: usize,
    pub usages: i32,
}

impl FinishedPattern {
    //#[inline(never)]
    fn new(pattern: Pattern, shared: &SharedData) -> Self {
        let arity = pattern.first_zid_of_ivar.len();
        let usages = pattern.match_locations.iter().map(|loc| shared.num_paths_to_node[usize::from(*loc)]).sum();
        let compressive_utility = compressive_utility(&pattern,shared);
        let noncompressive_utility = noncompressive_utility(pattern.body_utility, &shared.cfg);
        let utility = noncompressive_utility + compressive_utility.util;
        assert!(utility <= pattern.utility_upper_bound, "{} BUT utility is higher: {} (usages: {})", pattern.info(shared), utility, usages);
        let mut res = FinishedPattern {
            pattern,
            utility,
            compressive_utility: compressive_utility.util,
            util_calc: compressive_utility,
            arity,
            usages,
        };
        if shared.cfg.utility_by_rewrite {
            let rewritten: Vec<Expr> = rewrite_fast(&res, shared, "fake_inv", &shared.cost_fn);
            res.compressive_utility = shared.init_cost - shared.root_idxs_of_task.iter().map(|root_idxs|
                root_idxs.iter().map(|idx| rewritten[*idx].cost(&shared.cost_fn)).min().unwrap()
            ).sum::<i32>();
            // res.compressive_utility = shared.init_cost - rewritten.iter().map(|e|e.cost()).sum::<i32>();
            res.util_calc.util = res.compressive_utility;
            res.utility = res.compressive_utility + noncompressive_utility;
        }
        res
    }
    // convert finished invention to an Expr
    pub fn to_expr(&self, shared: &SharedData) -> Expr {
        self.pattern.to_expr(shared)
    }
    pub fn to_invention(&self, name: &str, shared: &SharedData) -> Invention {
        Invention::new(self.to_expr(shared), self.arity, name)
    }
    pub fn info(&self, shared: &SharedData) -> String {
        format!("{} -> finished: utility={}, compressive_utility={}, arity={}, usages={}",self.pattern.info(shared), self.utility, self.compressive_utility, self.arity, self.usages)
    }

}
// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
// struct Refinement {
//     refined_subtree: Id, // the thing you can refine out
//     uses: HashMap<Id,i32>, // map from loc to number of times it's used
//     refined_subtree_cost: i32, // the compressive utility gained by refining it
// }


/// figure out all the N^2 zippers from choosing any given node and then choosing a descendant and returning the zipper from
/// the node to the descendant. We also collect a bunch of other useful stuff like the argument you would get if you abstracted
/// the descendant and introduced an invention rooted at the ancestor node.
#[allow(clippy::type_complexity)]
//#[inline(never)]
fn get_zippers(
    treenodes: &[Id],
    cost_of_node_once: &[i32],
    egraph: &mut EGraph,
) -> (FxHashMap<Zip, ZId>, Vec<Zip>, Vec<FxHashMap<Id,Arg>>, FxHashMap<Id,Vec<ZId>>,  Vec<ZIdExtension>) {
    let cache: &mut Option<RecVarModCache> = &mut Some(FxHashMap::default());

    let mut zid_of_zip: FxHashMap<Zip, ZId> = Default::default();
    let mut zip_of_zid: Vec<Zip> = Default::default();
    let mut arg_of_zid_node: Vec<FxHashMap<Id,Arg>> = Default::default();
    let mut zids_of_node: FxHashMap<Id,Vec<ZId>> = Default::default();

    zid_of_zip.insert(vec![], EMPTY_ZID);
    zip_of_zid.push(vec![]);
    arg_of_zid_node.push(FxHashMap::default());
    
    // loop over all nodes in all programs in bottom up order
    for treenode in treenodes.iter() {
        // println!("processing id={}: {}", treenode, extract(*treenode, egraph) );

        // im essentially using the egraph just for its structural hashing rn
        assert!(egraph[*treenode].nodes.len() == 1);
        // clone to appease the borrow checker
        let node = egraph[*treenode].nodes[0].clone();
        
        // any node can become the identity function (the empty zipper with itself as the arg)
        let mut zids: Vec<ZId> = vec![EMPTY_ZID];
        arg_of_zid_node[EMPTY_ZID].insert(*treenode,
            Arg { shifted_id: *treenode, unshifted_id: *treenode, shift: 0, cost: cost_of_node_once[usize::from(*treenode)], expands_to: expands_to_of_node(&node) });
        
        match node {
            Lambda::IVar(_) => { panic!("attempted to abstract an IVar") }
            Lambda::Var(_) | Lambda::Prim(_) | Lambda::Programs(_) => {},
            Lambda::App([f,x]) => {
                // bubble from `f`
                for f_zid in zids_of_node[&f].iter() {
                    // clone and extend zip to get new zid for this node
                    let mut zip = zip_of_zid[*f_zid].clone();
                    zip.insert(0,ZNode::Func);
                    let zid = zid_of_zip.entry(zip.clone()).or_insert_with(|| {
                        let zid = zip_of_zid.len();
                        zip_of_zid.push(zip);
                        arg_of_zid_node.push(FxHashMap::default());
                        zid
                    });
                    // add new zid to this node
                    zids.push(*zid);
                    // give it the same arg
                    let arg = arg_of_zid_node[*f_zid][&f].clone();
                    arg_of_zid_node[*zid].insert(*treenode, arg);
                }

                // bubble from `x`
                for x_zid in zids_of_node[&x].iter() {
                    // clone and extend zip to get new zid for this node
                    let mut zip = zip_of_zid[*x_zid].clone();
                    zip.insert(0,ZNode::Arg);
                    let zid = zid_of_zip.entry(zip.clone()).or_insert_with(|| {
                        let zid = zip_of_zid.len();
                        zip_of_zid.push(zip);
                        arg_of_zid_node.push(FxHashMap::default());
                        zid
                    });
                    // add new zid to this node
                    zids.push(*zid);
                    // give it the same arg
                    let arg = arg_of_zid_node[*x_zid][&x].clone();
                    arg_of_zid_node[*zid].insert(*treenode, arg);

                }
            },
            Lambda::Lam([b]) => {
                for b_zid in zids_of_node[&b].iter() {

                    // clone and extend zip to get new zid for this node
                    let mut zip = zip_of_zid[*b_zid].clone();
                    zip.insert(0,ZNode::Body);
                    let zid = zid_of_zip.entry(zip.clone()).or_insert_with(|| {
                        let zid = zip_of_zid.len();
                        zip_of_zid.push(zip.clone());
                        arg_of_zid_node.push(FxHashMap::default());
                        zid
                    });
                    // add new zid to this node
                    zids.push(*zid);
                    // shift the arg but keep the unshifted part the same
                    let mut arg: Arg = arg_of_zid_node[*b_zid][&b].clone();

                    if !egraph[arg.shifted_id].data.free_vars.is_empty() {
                        // println!("stepping from child: {}", extract(b, egraph));
                        // println!("stepping to parent : {}", extract(*treenode, egraph));
                        // println!("b_zid: {}; b_zip: {:?}", b_zid, zip_of_zid[*b_zid]);
                        // println!("shift from: {}", extract(arg.id, egraph));
                        // println!("shift to:   {}", extract(arg.id, egraph));
                        // println!("total shift: {}", arg.shift);
                        if egraph[arg.shifted_id].data.free_vars.contains(&0) {
                            // we  go one less than the depth from the root to the arg. That way $0 when we're hopping
                            // the only  lambda in existence will map to depth_root_to_arg-1 = 1-1 = 0 -> #0 which will then
                            // be transformed back #0 -> $0 + depth = $0 + 0 = $0 if we thread it directly for example.
                            let depth_root_to_arg = zip.iter().filter(|x| **x == ZNode::Body).count() as i32;
                            arg.shifted_id = insert_arg_ivars(arg.shifted_id, depth_root_to_arg-1, egraph).unwrap();
                        }
                        arg.shifted_id = shift(arg.shifted_id, -1, egraph, cache).unwrap();
                        arg.shift -= 1;
                    }
                    arg_of_zid_node[*zid].insert(*treenode, arg);
                }            },
        }
        zids_of_node.insert(*treenode, zids);
    }

    let extensions_of_zid = zip_of_zid.iter().map(|zip| {
        let mut zip_body = zip.clone();
        zip_body.push(ZNode::Body);
        let mut zip_arg = zip.clone();
        zip_arg.push(ZNode::Arg);
        let mut zip_func = zip.clone();
        zip_func.push(ZNode::Func);
        ZIdExtension {
            body: zid_of_zip.get(&zip_body).copied(),
            arg: zid_of_zip.get(&zip_arg).copied(),
            func: zid_of_zip.get(&zip_func).copied(),
        }
    }).collect();

    (zid_of_zip,
    zip_of_zid,
    arg_of_zid_node,
    zids_of_node,
    extensions_of_zid)
}

/// the complete result of a single step of compression, this is a somewhat expensive data structure
/// to create.
#[derive(Debug, Clone)]
pub struct CompressionStepResult {
    pub inv: Invention,
    pub rewritten: Expr,
    pub rewritten_dreamcoder: Vec<String>,
    pub done: FinishedPattern,
    pub expected_cost: i32,
    pub final_cost: i32,
    pub multiplier: f64,
    pub multiplier_wrt_orig: f64,
    pub uses: i32,
    pub use_exprs: Vec<Expr>,
    pub use_args: Vec<Vec<Expr>>,
    pub dc_inv_str: String,
    pub initial_cost: i32,
}

impl CompressionStepResult {
    fn new(done: FinishedPattern, inv_name: &str, shared: &mut SharedData, past_invs: &[CompressionStepResult], prev_dc_inv_to_inv_strs: &[(String, String)]) -> Self {

        // cost of the very first initial program before any inventions
        let very_first_cost = if let Some(past_inv) = past_invs.first() { past_inv.initial_cost } else { shared.init_cost };

        let inv = done.to_invention(inv_name, shared);
        let rewritten = rewrite_fast(&done, shared, &inv.name, &shared.cost_fn);

        let expected_cost = shared.init_cost - done.compressive_utility;
        // let final_cost = rewritten.cost();
        let final_cost = shared.root_idxs_of_task.iter().map(|root_idxs|
            root_idxs.iter().map(|idx| rewritten[*idx].cost(&shared.cost_fn)).min().unwrap()
        ).sum::<i32>();
        if expected_cost != final_cost {
            println!("*** expected cost {} != final cost {}", expected_cost, final_cost);
        }
        let multiplier = shared.init_cost as f64 / final_cost as f64;
        let multiplier_wrt_orig = very_first_cost as f64 / final_cost as f64;
        let uses = done.usages;
        let use_exprs: Vec<Expr> = done.pattern.match_locations.iter().map(|node| extract(*node, &shared.egraph)).collect();
        let use_args: Vec<Vec<Expr>> = done.pattern.match_locations.iter().map(|node|
            done.pattern.first_zid_of_ivar.iter().map(|zid|
                extract(shared.arg_of_zid_node[*zid][node].shifted_id, &shared.egraph)
            ).collect()).collect();
        
        // Combine the past_invs with the existing dreamcoder inventions.
        let mut dreamcoder_translations: Vec<(String, String)>  = past_invs.iter().map(|compression_step_result| (compression_step_result.inv.name.clone(), compression_step_result.dc_inv_str.clone())).collect();

        dreamcoder_translations.extend(prev_dc_inv_to_inv_strs.iter().cloned());

        // dreamcoder compatability
        let dc_inv_str: String = dc_inv_str(&inv, &dreamcoder_translations);
        // Rewrite to dreamcoder syntax with all past invention
        // we rewrite "inv1)" and "inv1 " instead of just "inv1" because we dont want to match on "inv10"

        let rewritten_dreamcoder: Vec<String> = rewritten.iter().map(|p|{
            let mut res = p.to_string();
            for (prev_inv_name, prev_dc_inv_str) in prev_dc_inv_to_inv_strs {
                res = replace_prim_with(&res, prev_inv_name, prev_dc_inv_str);
                // res = res.replace(&format!("{})",past_inv.inv.name), &format!("{})",past_inv.dc_inv_str));
                // res = res.replace(&format!("{} ",past_inv.inv.name), &format!("{} ",past_inv.dc_inv_str));
            }

            // Now go ahead and replace the current invention.
            res = replace_prim_with(&res, inv_name, &dc_inv_str);
            // res = res.replace(&format!("{})",inv_name), &format!("{})",dc_inv_str));
            // res = res.replace(&format!("{} ",inv_name), &format!("{} ",dc_inv_str));
            res = res.replace("(lam ","(lambda ");
            res
        }).collect();

        CompressionStepResult { inv, rewritten: Expr::programs(rewritten), rewritten_dreamcoder, done, expected_cost, final_cost, multiplier, multiplier_wrt_orig, uses, use_exprs, use_args, dc_inv_str, initial_cost: shared.init_cost }
    }
    pub fn json(&self) -> serde_json::Value {        
        let use_exprs: Vec<String> = self.use_exprs.iter().map(|expr| expr.to_string()).collect();
        let use_args: Vec<String> = self.use_args.iter().map(|args| format!("{} {}", self.inv.name, args.iter().map(|expr| expr.to_string()).collect::<Vec<String>>().join(" "))).collect();
        let all_uses: Vec<serde_json::Value> = use_exprs.iter().zip(use_args.iter()).sorted().map(|(expr,args)| json!({args: expr})).collect();

        json!({            
            "body": self.inv.body.to_string(),
            "dreamcoder": self.dc_inv_str,
            "arity": self.inv.arity,
            "name": self.inv.name,
            "rewritten": self.rewritten.split_programs().iter().map(|p| p.to_string()).collect::<Vec<String>>(),
            "rewritten_dreamcoder": self.rewritten_dreamcoder,
            "utility": self.done.utility,
            "expected_cost": self.expected_cost,
            "final_cost": self.final_cost,
            "multiplier": self.multiplier,
            "multiplier_wrt_orig": self.multiplier_wrt_orig,
            "num_uses": self.uses,
            "uses": all_uses,
        })
    }
}

impl fmt::Display for CompressionStepResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.expected_cost != self.final_cost {
            write!(f,"[cost mismatch of {}] ", self.expected_cost - self.final_cost)?;
        }
        write!(f, "utility: {} | final_cost: {} | {:.2}x | uses: {} | body: {}",
            self.done.utility, self.final_cost, self.multiplier, self.uses, self.inv)
    }
}

/// calculates the total upper bound on compressive + noncompressive utility
//#[inline(never)]
fn utility_upper_bound(
    match_locations: &[Id],
    body_utility_lower_bound: i32,
    cost_of_node_all: &[i32],
    num_paths_to_node: &[i32],
    cfg: &CompressionStepConfig,
) -> i32 {
    compressive_utility_upper_bound(match_locations, cost_of_node_all, num_paths_to_node)
        + noncompressive_utility_upper_bound(body_utility_lower_bound, cfg)
}

/// This utility is just for any utility terms that we care about that don't directly correspond
/// to changes in size that come from rewriting with an invention
//#[inline(never)]
fn noncompressive_utility(
    body_utility: i32,
    cfg: &CompressionStepConfig,
) -> i32 {
    if cfg.no_other_util { return 0; }
    // this is a bit like the structure penalty from dreamcoder except that
    // that penalty uses inlined versions of nested inventions.
    // 0
    - body_utility
}

/// This takes a partial invention and gives an upper bound on the maximum
/// compressive_utility() that any completed offspring of this partial invention could have.
//#[inline(never)]
fn compressive_utility_upper_bound(
    match_locations: &[Id],
    cost_of_node_all: &[i32],
    num_paths_to_node: &[i32],
) -> i32 {
    match_locations.iter().map(|node|
        cost_of_node_all[usize::from(*node)] 
        - num_paths_to_node[usize::from(*node)] * COST_TERMINAL).sum::<i32>()
    
    // shared.init_cost - shared.root_idxs_of_task.iter().map(|root_idxs|
    //     root_idxs.iter().map(|idx| shared.init_cost_by_root_idx[*idx] - adjusted_util_by_root_idx[*idx]).min().unwrap()
    // ).sum::<i32>()
}

/// calculates the total upper bound on compressive + noncompressive utility
// //#[inline(never)]
// fn utility_upper_bound_with_conflicts(
//     pattern: &Pattern,
//     body_utility_with_refinement_lower_bound: i32,
//     shared: &SharedData,
// ) -> i32 {
//     let utility_of_loc_once: Vec<i32> = pattern.match_locations.iter().map(|node|
//         shared.cost_of_node_once[usize::from(*node)] - COST_TERMINAL).collect();
//     let compressive_utility: i32 = pattern.match_locations.iter()
//         .zip(utility_of_loc_once.iter())
//         .map(|(loc,utility)| utility * shared.num_paths_to_node[usize::from(*loc)])
//         .sum();
//     use_conflicts(pattern, utility_of_loc_once, compressive_utility, shared).util + noncompressive_utility_upper_bound(body_utility_with_refinement_lower_bound, &shared.cfg)
// }


/// This takes a partial invention and gives an upper bound on the maximum
/// other_utility() that any completed offspring of this partial invention could have.
//#[inline(never)]
fn noncompressive_utility_upper_bound(
    body_utility_lower_bound: i32,
    cfg: &CompressionStepConfig,
) -> i32 {
    // 0
    if cfg.no_other_util { return 0; }
    - body_utility_lower_bound
    // safe bound: since structure_penalty is negative an upper bound is anything less negative or exact. Since
    // left_utility < body_utility we know that this will be a less negative bound.
    
}

//#[inline(never)]
fn compressive_utility(pattern: &Pattern, shared: &SharedData) -> UtilityCalculation {

    // * BASIC CALCULATION
    // Roughly speaking compressive utility is num_usages(invention) * size(invention), however there are a few extra
    // terms we need to take care of too.

    let utility_of_loc_once: Vec<i32> = get_utility_of_loc_once(pattern, shared);

    let (cumulative_utility_of_node, corrected_utils) = bottom_up_utility_correction(pattern,shared,&utility_of_loc_once);

    let compressive_utility: i32 = shared.init_cost - shared.root_idxs_of_task.iter().map(|root_idxs|
        root_idxs.iter().map(|idx| shared.init_cost_by_root_idx[*idx] - cumulative_utility_of_node[usize::from(shared.roots[*idx])]).min().unwrap()
    ).sum::<i32>();

    // pattern.match_locations.

    UtilityCalculation { util: compressive_utility, corrected_utils }
}

//#[inline(never)]
fn get_utility_of_loc_once(pattern: &Pattern, shared: &SharedData) -> Vec<i32> {
    // it costs a tiny bit to apply the invention, for example (app (app inv0 x) y) incurs a cost
    // of COST_TERMINAL for the `inv0` primitive and 2 * COST_NONTERMINAL for the two `app`s.
    // Also an extra COST_NONTERMINAL for each argument that is refined (for the lambda).
    let app_penalty = - (COST_TERMINAL + COST_NONTERMINAL * pattern.first_zid_of_ivar.len() as i32);

    // get a list of (ivar,usages-1) filtering out things that are only used once, this will come in handy for adding multi-use utility later
    let ivar_multiuses: Vec<(usize,i32)> = pattern.arg_choices.iter().map(|labelled|labelled.ivar).counts()
        .iter().filter_map(|(ivar,count)| if *count > 1 { Some((*ivar, (*count-1) as i32)) } else { None }).collect();

    pattern.match_locations.iter().map(|loc| {

        //  if there are any free ivars in the arg at this location then we can't apply this invention here so *total* util should be 0
        for (_ivar,zid) in pattern.first_zid_of_ivar.iter().enumerate() {
            let shifted_arg = shared.arg_of_zid_node[*zid][loc].shifted_id;
            if !shared.egraph[shifted_arg].data.free_ivars.is_empty() {
                return 0; // set whole util to 0 for this loc, causing an autoreject
            }
        }

        // println!("calculating util of {}", extract(*loc, &shared.egraph));
        // compressivity of body (no refinement) minus slight penalty from the application
        let base_utility = pattern.body_utility + app_penalty;
        // println!("base {}", base_utility);

        // for each extra usage of an argument, we gain the cost of that argument as
        // extra utility. Note we use `first_zid_of_ivar` since it doesn't matter which
        // of the zids we use as long as it corresponds to the right ivar
        let multiuse_utility = ivar_multiuses.iter().map(|(ivar,count)|
            count * shared.arg_of_zid_node[pattern.first_zid_of_ivar[*ivar]][loc].cost
        ).sum::<i32>();
        // println!("multiuse {}", multiuse_utility);

        base_utility + multiuse_utility
    }).collect()
}

//#[inline(never)]
fn bottom_up_utility_correction(pattern: &Pattern, shared:&SharedData, utility_of_loc_once: &[i32]) -> (Vec<i32>,FxHashMap<Id,bool>) {
    let mut cumulative_utility_of_node: Vec<i32> = vec![0; shared.treenodes.len()];
    let mut corrected_utils: FxHashMap<Id,bool> = Default::default();

    for node in shared.treenodes.iter() {

        let utility_without_rewrite: i32 = match &shared.node_of_id[usize::from(*node)] {
            Lambda::Lam([b]) => cumulative_utility_of_node[usize::from(*b)],
            Lambda::App([f,x]) => cumulative_utility_of_node[usize::from(*f)] + cumulative_utility_of_node[usize::from(*x)],
            Lambda::Prim(_) | Lambda::Var(_) => 0,
            Lambda::IVar(_) | Lambda::Programs(_) => unreachable!(),
        };

        assert!(utility_without_rewrite >= 0);

        if let Ok(idx) = pattern.match_locations.binary_search(node) {
            // this node is a potential rewrite location

            let utility_of_args: i32 = pattern.first_zid_of_ivar.iter()
                .map(|zid| cumulative_utility_of_node[usize::from(shared.arg_of_zid_node[*zid][node].unshifted_id)])
                .sum();
            let utility_with_rewrite = utility_of_args + utility_of_loc_once[idx];

            let chose_to_rewrite = utility_with_rewrite > utility_without_rewrite;

            cumulative_utility_of_node[usize::from(*node)] = std::cmp::max(utility_with_rewrite, utility_without_rewrite);

            corrected_utils.insert(*node,chose_to_rewrite);


        } else if utility_without_rewrite != 0 {
            cumulative_utility_of_node[usize::from(*node)] = utility_without_rewrite;
        }
    }
    (cumulative_utility_of_node,corrected_utils)
}


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UtilityCalculation {
    pub util: i32,
    pub corrected_utils: FxHashMap<Id,bool>, // whether to accept
}

pub fn inverse_delta(cost_once: i32, usages: i32, arg_uses: usize) -> (i32, i32, i32) {
    let compressive_delta = - (cost_once + COST_NONTERMINAL) * usages;
    let noncompressive_delta = arg_uses as i32 * (cost_once - COST_TERMINAL) ;
    (compressive_delta,noncompressive_delta, compressive_delta+noncompressive_delta)
}

pub fn inverse_argument_capture(finished: &mut FinishedPattern, cfg: &CompressionStepConfig, zip_of_zid: &[Zip], node_of_id: &[Lambda], cost_of_node_once: &[i32], arg_of_zid_node: &[FxHashMap<Id,Arg>], extensions_of_zid: &[ZIdExtension], egraph: &EGraph) {
    if cfg.no_inv_arg_cap {
        return
    }
    while finished.arity < cfg.max_arity {
        let counts = use_counts(&finished.pattern, node_of_id, zip_of_zid, arg_of_zid_node, extensions_of_zid, egraph);
        let best = counts.iter()
            .filter(|(arg_id,(cost,zids))| zids.len() > finished.usages as usize)
            .max_by_key(|(arg_id,(cost,zids))|{
                inverse_delta(*cost, finished.usages, zids.len()).2
            });
        if let Some((arg_id, (cost,zids))) = best {

            let (compressive_delta,
                 noncompressive_delta,
                 delta) = inverse_delta(*cost, finished.usages, zids.len());
            
            if delta < 0 {
                return
            }
            let ivar = finished.arity;
            finished.pattern.arg_choices.extend(zids.iter().map(|&zid| LabelledZId { zid, ivar }));
            finished.pattern.first_zid_of_ivar.push(zids[0]);
            finished.compressive_utility += compressive_delta;
            finished.util_calc.util += compressive_delta;
            finished.utility += delta;
            finished.arity +=1;
            println!("UNARG")
        } else {
            return
        }
    }
}

fn use_counts(pattern: &Pattern, node_of_id: &[Lambda], zip_of_zid: &[Zip], arg_of_zid_node: &[FxHashMap<Id,Arg>], extensions_of_zid: &[ZIdExtension], egraph: &EGraph) -> FxHashMap<Id,(i32,Vec<ZId>)> {
    let mut curr_zip: Zip = vec![];
    let curr_zid: ZId = EMPTY_ZID;
    let zids = &pattern.arg_choices[..];

    // map zids to zips with a bool thats true if this is a hole and false if its a future ivar
    let zips: Vec<Zip> = zids.iter()
        .map(|labelled_zid| zip_of_zid[labelled_zid.zid].clone()).collect();

    let mut counts: FxHashMap<Id,(i32,Vec<ZId>)> = Default::default();

    fn helper(curr_node: Id, match_loc: Id, curr_zip: &mut Zip, curr_zid: ZId, zips: &[Zip], zids: &[LabelledZId], node_of_id: &[Lambda], arg_of_zid_node: &[FxHashMap<Id,Arg>], extensions_of_zid: &[ZIdExtension], egraph: &EGraph,  counts: &mut FxHashMap<Id,(i32,Vec<ZId>)>) {
        if zids.iter().any(|labelled| labelled.zid == curr_zid){
            return // current zip matches an arg
        }
        // if curr_zip is not a prefix of any arg zipper, then increment its count
        if zips.iter().all(|zip| !zip.starts_with(curr_zip)) {
            // also make sure its valid ie doesnt have any free ivars as ew do during normal checks
            let arg = arg_of_zid_node[curr_zid].get(&match_loc).unwrap();
            if egraph[arg.shifted_id].data.free_ivars.is_empty() {
                counts.entry(arg.shifted_id)
                    .or_insert_with(||(arg.cost, vec![]))
                    .1.push(curr_zid);
            }
        }
        match &node_of_id[usize::from(curr_node)] {
            Lambda::Prim(_) => {},
            Lambda::Var(_) => {},
            Lambda::Lam([b]) => {
                curr_zip.push(ZNode::Body);
                let new_zid = extensions_of_zid[curr_zid].body.unwrap();
                helper(*b, match_loc, curr_zip, new_zid, zips, zids, node_of_id, arg_of_zid_node, extensions_of_zid, egraph, counts);
                curr_zip.pop();
            }
            Lambda::App([f,x]) => {
                curr_zip.push(ZNode::Func);
                let new_zid = extensions_of_zid[curr_zid].func.unwrap();
                helper(*f, match_loc, curr_zip, new_zid, zips, zids, node_of_id, arg_of_zid_node, extensions_of_zid, egraph, counts);
                curr_zip.pop();
                curr_zip.push(ZNode::Arg);
                let new_zid = extensions_of_zid[curr_zid].arg.unwrap();
                helper(*x, match_loc, curr_zip, new_zid, zips, zids, node_of_id, arg_of_zid_node, extensions_of_zid, egraph, counts);
                curr_zip.pop();
            }
            _ => unreachable!(),
        }
    }
    // we can pick any match location
    helper(pattern.match_locations[0], pattern.match_locations[0], &mut curr_zip, curr_zid, &zips, zids, node_of_id, arg_of_zid_node, extensions_of_zid, egraph, &mut counts);
    counts
}


// fn use_counts(pattern: &Pattern, node_of_id: &[Lambda], zip_of_zid: &[Zip]) -> FxHashMap<Id,usize> {
//     let mut curr_zip: Zip = vec![];
//     // map zids to zips with a bool thats true if this is a hole and false if its a future ivar
//     let zips: Vec<Zip> = pattern.arg_choices.iter()
//         .map(|labelled_zid| zip_of_zid[labelled_zid.zid].clone()).collect();

//     let mut counts: FxHashMap<Id,usize> = Default::default();

//     fn helper(curr_node: Id, curr_zip: &mut Zip, zips: &[Zip], node_of_id: &[Lambda], counts: &mut FxHashMap<Id,usize>) {
//         if zips.iter().contains(curr_zip){
//             return // current zip matches an arg
//         }
//         // if curr_zip is not a prefix of any arg zipper, then increment its count
//         if zips.iter().all(|zip| !zip.starts_with(curr_zip)) {
//             *counts.entry(arg_of_zid_node[cur]).or_default() += 1;
//         }
//         match &node_of_id[usize::from(curr_node)] {
//             Lambda::Prim(_) => {},
//             Lambda::Var(_) => {},
//             Lambda::Lam([b]) => {
//                 curr_zip.push(ZNode::Body);
//                 helper(*b, curr_zip, zips, node_of_id, counts);
//                 curr_zip.pop();
//             }
//             Lambda::App([f,x]) => {
//                 curr_zip.push(ZNode::Func);
//                 helper(*f, curr_zip, zips, node_of_id, counts);
//                 curr_zip.pop();
//                 curr_zip.push(ZNode::Arg);
//                 helper(*x, curr_zip, zips, node_of_id, counts);
//                 curr_zip.pop();
//             }
//             _ => unreachable!(),
//         }
//     }
//     // we can pick any match location
//     helper(pattern.match_locations[0], &mut curr_zip, &zips, node_of_id, &mut counts);
//     counts
// }


/// Multistep compression. See `compression_step` if you'd just like to do a single step of compression.
pub fn compression(
    train_programs_expr: &Expr,
    test_programs_expr: &Option<Expr>,
    iterations: usize,
    cfg: &CompressionStepConfig,
    tasks: &[String],
    prev_dc_inv_to_inv_strs: &[(String, String)],
    cost_fn: &ProgramCost,
) -> Vec<CompressionStepResult> {
    let num_prior_inventions = prev_dc_inv_to_inv_strs.len();
    let mut rewritten: Expr = train_programs_expr.clone();
    let mut step_results: Vec<CompressionStepResult> = Default::default();

    let tstart = std::time::Instant::now();

    for i in 0..iterations {
        println!("{}",format!("\n=======Iteration {}=======",i).blue().bold());
        let inv_name = format!("fn_{}", num_prior_inventions + step_results.len());

        // call actual compression
        let res: Vec<CompressionStepResult> = compression_step(
            &rewritten,
            &inv_name,
            cfg,
            &step_results,
            tasks,
            prev_dc_inv_to_inv_strs,
            cost_fn);

        if !res.is_empty() {
            // rewrite with the invention
            let res: CompressionStepResult = res[0].clone();
            rewritten = res.rewritten.clone();
            println!("Chose Invention {}: {}", res.inv.name, res);
            step_results.push(res);
        } else {
            println!("No inventions found at iteration {}",i);
            break;
        }
    }

    println!("{}","\n=======Compression Summary=======".blue().bold());
    println!("Found {} inventions", step_results.len());
    println!("Cost Improvement: ({:.2}x better) {} -> {}", compression_factor(train_programs_expr, &rewritten, cost_fn), train_programs_expr.cost(cost_fn), rewritten.cost(cost_fn));
    for res in step_results.iter() {
        println!("{} ({:.2}x wrt orig): {}" , res.inv.name.clone().blue(), compression_factor(train_programs_expr, &res.rewritten, cost_fn), res);
    }
    println!("Time: {}ms", tstart.elapsed().as_millis());
    if cfg.follow_track && !(
        cfg.no_opt_free_vars
        && cfg.no_opt_single_task
        && cfg.no_opt_upper_bound
        && cfg.no_opt_force_multiuse
        && cfg.no_opt_useless_abstract
        && cfg.no_opt_arity_zero)
    {
        println!("{} you often want to run --follow-track with --no-opt otherwise your target may get pruned", "[WARNING]".yellow());
    }

    if let Some(e) = test_programs_expr {
        println!("Test set compression with all inventions applied: {}", compression_factor(e, &rewrite_with_inventions(e.clone(), &step_results.iter().map(|r| r.inv.clone()).collect::<Vec<Invention>>(), cost_fn), cost_fn));
    }
    step_results
}

/// Takes a set of programs as an Expr with Programs as its root, and does one full step of compresison.
/// Returns the top Inventions and the Expr rewritten under that invention along with other useful info in CompressionStepResult
/// The number of inventions returned is based on cfg.inv_candidates
pub fn compression_step(
    programs_expr: &Expr,
    new_inv_name: &str, // name of the new invention, like "inv4"
    cfg: &CompressionStepConfig,
    past_invs: &[CompressionStepResult], // past inventions we've found
    task_name_of_root_idx: &[String],
    prev_dc_inv_to_inv_strs: &[(String, String)],
    cost_fn: &ProgramCost,
) -> Vec<CompressionStepResult> {

    let tstart_total = std::time::Instant::now();
    let tstart_prep = std::time::Instant::now();
    let mut tstart = std::time::Instant::now();

    // build the egraph. We'll just be using this as a structural hasher we don't use rewrites at all. All eclasses will always only have one node.
    let mut egraph: EGraph = Default::default();
    let programs_node = egraph.add_expr(programs_expr.into());
    egraph.rebuild();

    let first_train_cost = egraph[programs_node].data.inventionless_cost; // This is used for --verbose-print

    println!("set up egraph: {:?}ms", tstart.elapsed().as_millis());
    tstart = std::time::Instant::now();

    let roots: Vec<Id> = egraph[programs_node].nodes[0].children().to_vec();


    // all nodes in child-first order except for the Programs node
    let mut treenodes: Vec<Id> = topological_ordering(programs_node,&egraph);
    assert!(treenodes.iter().enumerate().all(|(i,node)| i == usize::from(*node)));
    // assert_eq!(treenodes.iter().map(|n| usize::from(*n)).collect::<Vec<_>>(), (0..treenodes.len()).collect::<Vec<_>>());
    let node_of_id: Vec<Lambda> = treenodes.iter().map(|node| egraph[*node].nodes[0].clone()).collect();
    treenodes.retain(|id| *id != programs_node);

    println!("got roots, treenodes, and cloned egraph contents: {:?}ms", tstart.elapsed().as_millis());
    tstart = std::time::Instant::now();

    // populate num_paths_to_node so we know how many different parts of the programs tree
    // a node participates in (ie multiple uses within a single program or among programs)
    let (num_paths_to_node, num_paths_to_node_by_root_idx) : (Vec<i32>, Vec<Vec<i32>>) = num_paths_to_node(&roots, &treenodes, &egraph);

    println!("num_paths_to_node(): {:?}ms", tstart.elapsed().as_millis());
    tstart = std::time::Instant::now();


    let mut task_name_of_task: Vec<String> = vec![];
    let mut task_of_root_idx: Vec<usize> = vec![];
    let mut root_idxs_of_task: Vec<Vec<usize>> = vec![];
    for (root_idx,task_name) in task_name_of_root_idx.iter().enumerate() {
        let task = task_name_of_task.iter().position(|name| name == task_name)
            .unwrap_or_else(||{
                task_name_of_task.push(task_name.clone());
                root_idxs_of_task.push(vec![]);
                task_name_of_task.len() - 1
            });
        task_of_root_idx.push(task);
        root_idxs_of_task[task].push(root_idx);
    }
    let tasks_of_node: Vec<FxHashSet<usize>> = associate_tasks(programs_node, &egraph, &treenodes, &task_of_root_idx);

    let init_cost_by_root_idx: Vec<i32> = roots.iter().map(|id| egraph[*id].data.inventionless_cost).collect();
    // assert_eq!(init_cost, init_cost_by_root_idx.iter().sum::<i32>());
    let init_cost: i32 = root_idxs_of_task.iter().map(|root_idxs|
        root_idxs.iter().map(|idx| init_cost_by_root_idx[*idx]).min().unwrap()
    ).sum();
    //  = egraph[programs_node].data.inventionless_cost;

    println!("associate_tasks() and other task stuff: {:?}ms", tstart.elapsed().as_millis());
    println!("num unique tasks: {}", task_name_of_task.len());
    println!("num unique programs: {}", roots.len());
    tstart = std::time::Instant::now();

    // arity inference
    // let mut arity_of_node: Vec<usize> = arity_inference(programs_node, &egraph, &treenodes);

    // cost of a single usage of a node (same as inventionless_cost)
    let cost_of_node_once: Vec<i32> = treenodes.iter().map(|node| egraph[*node].data.inventionless_cost).collect();
    // cost of a single usage times number of paths to node
    let cost_of_node_all: Vec<i32> = treenodes.iter().map(|node| cost_of_node_once[usize::from(*node)] * num_paths_to_node[usize::from(*node)]).collect();

    let free_vars_of_node: Vec<FxHashSet<i32>> = treenodes.iter().map(|node| egraph[*node].data.free_vars.clone()).collect();

    println!("cost_of_node structs: {:?}ms", tstart.elapsed().as_millis());
    tstart = std::time::Instant::now();

    let (zid_of_zip,
        zip_of_zid,
        arg_of_zid_node,
        zids_of_node,
        extensions_of_zid) = get_zippers(&treenodes, &cost_of_node_once, &mut egraph);
    
    println!("get_zippers(): {:?}ms", tstart.elapsed().as_millis());
    tstart = std::time::Instant::now();
    
    println!("{} zips", zip_of_zid.len());
    println!("arg_of_zid_node size: {}", arg_of_zid_node.len());

    // set up tracking if any
    let tracking: Option<Tracking> = cfg.track.as_ref().map(|s|{
        let expr: Expr = s.parse().unwrap();
        let zids_of_ivar = zids_of_ivar_of_expr(&expr, &zid_of_zip);
        Tracking { expr, zids_of_ivar }
    });

    println!("Tracking setup: {:?}ms", tstart.elapsed().as_millis());

    let mut stats: Stats = Default::default();

    tstart = std::time::Instant::now();

    // define all the important data structures for compression
    let mut donelist: Vec<FinishedPattern> = Default::default(); // completed inventions will go here    

    let mut azero_pruning_cutoff = 0;

    // arity 0 inventions
    if !cfg.no_opt_arity_zero {
        for node in treenodes.iter() {

            // check for free vars: inventions with free vars in the body are not well-defined functions
            // and should thus be discarded
            if !cfg.no_opt_free_vars && !egraph[*node].data.free_vars.is_empty() {
                if !cfg.no_stats { stats.free_vars_fired += 1; };
                continue;
            }

            // check whether this invention is useful in > 1 task
            if !cfg.no_opt_single_task && tasks_of_node[usize::from(*node)].len() < 2 {
                if !cfg.no_stats { stats.single_task_fired += 1; };
                continue;
            }
            // Note that "single use" pruning is intentionally not done here,
            // since any invention specific to a node will by definition only
            // be useful at that node

            let match_locations = vec![*node];
            let body_utility = cost_of_node_once[usize::from(*node)];
            // compressive_utility for arity-0 is cost_of_node_all[node] minus the penalty of using the new prim

            let compressive_utility: i32 = init_cost - root_idxs_of_task.iter().map(|root_idxs|
                root_idxs.iter().map(|idx| init_cost_by_root_idx[*idx] - num_paths_to_node_by_root_idx[*idx][usize::from(*node)] * (cost_of_node_once[usize::from(*node)] - COST_TERMINAL))
                    .min().unwrap()
            ).sum::<i32>();
            // println!("utility: {}", compressive_utility);
            
            // let compressive_utility = cost_of_node_all[usize::from(*node)] - num_paths_to_node[usize::from(*node)] * COST_TERMINAL;
            let utility = compressive_utility + noncompressive_utility(body_utility, cfg);
            if utility <= 0 { continue; }


            if !cfg.no_stats { stats.azero_calc_util += 1; };

            if compressive_utility <= azero_pruning_cutoff {
                continue // upper bound pruning
            }

            let pattern = Pattern {
                holes: vec![],
                arg_choices: vec![],
                first_zid_of_ivar: vec![],
                match_locations,
                utility_upper_bound: utility,
                body_utility,
                tracked: false,
            };
            let mut finished_pattern = FinishedPattern {
                pattern,
                utility,
                compressive_utility,
                util_calc: UtilityCalculation { util: compressive_utility, corrected_utils: Default::default()},
                arity: 0,
                usages: num_paths_to_node[usize::from(*node)]
            };

            inverse_argument_capture(&mut finished_pattern, &cfg, &zip_of_zid, &node_of_id, &cost_of_node_once, &arg_of_zid_node, &extensions_of_zid, &egraph);
            if !cfg.no_stats { stats.azero_calc_unargcap += 1; };

            if finished_pattern.utility <= azero_pruning_cutoff {
                continue // upper bound pruning
            }

            if cfg.inv_candidates == 1 && finished_pattern.utility > azero_pruning_cutoff {
                // if we're only looking for one invention, we can directly update our cutoff here
                azero_pruning_cutoff = finished_pattern.utility
            }
            donelist.push(finished_pattern);
        }
    }

    println!("arity 0: {:?}ms", tstart.elapsed().as_millis());
    tstart = std::time::Instant::now();

    println!("got {} arity zero inventions", donelist.len());

    let crit = CriticalMultithreadData::new(donelist, &treenodes, &cost_of_node_all, &num_paths_to_node, &egraph, cfg);
    let shared = Arc::new(SharedData {
        crit: Mutex::new(crit),
        arg_of_zid_node,
        cost_fn: cost_fn.clone(),
        treenodes: treenodes.clone(),
        node_of_id,
        programs_node,
        roots,
        zids_of_node,
        zip_of_zid,
        zid_of_zip,
        extensions_of_zid,
        egraph,
        num_paths_to_node,
        num_paths_to_node_by_root_idx,
        tasks_of_node,
        task_name_of_task,
        task_of_root_idx,
        root_idxs_of_task,
        cost_of_node_once,
        cost_of_node_all,
        free_vars_of_node,
        init_cost,
        init_cost_by_root_idx,
        first_train_cost,
        stats: Mutex::new(stats),
        cfg: cfg.clone(),
        tracking,
    });

    println!("built SharedData: {:?}ms", tstart.elapsed().as_millis());
    tstart = std::time::Instant::now();

    if cfg.verbose_best {
        let mut crit = shared.crit.lock();
        if !crit.deref_mut().donelist.is_empty() {
            let best_util = crit.deref_mut().donelist.first().unwrap().utility;
            let best_expr: String = crit.deref_mut().donelist.first().unwrap().info(&shared);
            let new_expected_cost = first_train_cost - crit.donelist.first().unwrap().compressive_utility + crit.donelist.first().unwrap().to_expr(&shared).cost(&shared.cost_fn);
            let trainratio = first_train_cost as f64/new_expected_cost as f64;
            println!("{} @ step=0 util={} trainratio={:.2} for {}", "[new best utility]".blue(), best_util, trainratio, best_expr);
        }
    }

    println!("TOTAL PREP: {:?}ms", tstart_prep.elapsed().as_millis());

    println!("running pattern search...");

    // *****************
    // * STITCH SEARCH *
    // *****************
    // (this is finding all the higher-arity multi-use inventions through stitching)
    if cfg.threads == 1 {
        // Single threaded
        stitch_search(Arc::clone(&shared));
    } else {
        // Multithreaded
        let mut handles = vec![];
        for _ in 0..cfg.threads {
            // clone the Arcs to have copies for this thread
            let shared = Arc::clone(&shared);
            
            // launch thread to just call stitch_search()
            handles.push(thread::spawn(move || {
                stitch_search(shared);
            }));
        }
        // wait for all threads to finish (when all have empty worklists)
        for handle in handles {
            handle.join().unwrap();
        }
    }

    println!("TOTAL SEARCH: {:?}ms", tstart.elapsed().as_millis());
    println!("TOTAL PREP + SEARCH: {:?}ms", tstart_total.elapsed().as_millis());


    tstart = std::time::Instant::now();

    // at this point we hold the only reference so we can get rid of the Arc
    let mut shared: SharedData = Arc::try_unwrap(shared).unwrap();

    // one last .update()
    shared.crit.lock().deref_mut().update(cfg);

    println!("{:?}", shared.stats.lock().deref_mut());
    assert!(shared.crit.lock().deref_mut().worklist.is_empty());

    let donelist: Vec<FinishedPattern> = shared.crit.lock().deref_mut().donelist.clone();

    if cfg.dreamcoder_comparison {
        println!("Timing point 1 (from the start of compression_step to final donelist): {:?}ms", tstart_total.elapsed().as_millis());
        println!("Timing Comparison Point A (search) (millis): {}", tstart_total.elapsed().as_millis());
        let tstart_rewrite = std::time::Instant::now();
        rewrite_fast(&donelist[0], &shared, new_inv_name, cost_fn);
        println!("Timing point 2 (rewriting the candidate): {:?}ms", tstart_rewrite.elapsed().as_millis());
        println!("Timing Comparison Point B (search+rewrite) (millis): {}", tstart_total.elapsed().as_millis());
    }

    let mut results: Vec<CompressionStepResult> = vec![];

    // construct CompressionStepResults and print some info about them)
    println!("Cost before: {}", shared.init_cost);
    for (i,done) in donelist.iter().enumerate() {
        let res = CompressionStepResult::new(done.clone(), new_inv_name, &mut shared, past_invs, prev_dc_inv_to_inv_strs);

        println!("{}: {}", i, res);
        if cfg.show_rewritten {
            println!("rewritten:\n{}", res.rewritten.split_programs().iter().map(|p|p.to_string()).collect::<Vec<_>>().join("\n"));
        }
        results.push(res);
    }
    println!("post stuff: {:?}ms", tstart.elapsed().as_millis());

    results
}
