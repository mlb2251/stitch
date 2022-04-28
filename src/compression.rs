use crate::*;
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Formatter, Display};
use std::hash::Hash;
use itertools::Itertools;
use extraction::extract;
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
    /// max arity of inventions to find (will find all from 0 to this number inclusive)
    #[clap(short='a', long, default_value = "2")]
    pub max_arity: usize,

    /// num threads (no parallelism if set to 1)
    #[clap(short='t', long, default_value = "1")]
    pub threads: usize,

    /// how many worklist items a thread will take at once
    #[clap(short='b', long, default_value = "1")]
    pub batch: usize,

    /// threads will autoadjust how large their batches are based on the worklist size
    #[clap(long)]
    pub dynamic_batch: bool,

    /// disables refinement
    #[clap(long)]
    pub refine: bool,

    /// max refinement size
    #[clap(long)]
    pub max_refinement_size: Option<i32>,

    /// max number of refined out args that can be passed into a #i
    #[clap(long, default_value = "1")]
    pub max_refinement_arity: usize,

    /// Number of invention candidates compression_step should return. Raising this may weaken the efficacy of upper bound pruning
    #[clap(short='n', long, default_value = "1")]
    pub inv_candidates: usize,

    /// pattern or invention to track
    #[clap(long, arg_enum, default_value = "min-cost")]
    pub hole_choice: HoleChoice,

    /// disables the safety check for the utility being correct; you only want
    /// to do this if you truly dont mind unsoundness for a minute
    #[clap(long)]
    pub no_mismatch_check: bool,

    /// inventions cant start with a Lambda
    #[clap(long)]
    pub no_top_lambda: bool,

    /// pattern or invention to track
    #[clap(long)]
    pub track: Option<String>,

    /// refined version of pattern or invention to track
    #[clap(long)]
    pub track_refined: Option<String>,

    /// pattern or invention to track
    #[clap(long)]
    pub follow_track: bool,

    /// print out each step of what gets popped off the worklist
    #[clap(long)]
    pub verbose_worklist: bool,

    /// whenever a new best thing is found, print it
    #[clap(long)]
    pub verbose_best: bool,

    /// print stats this often (0 means never)
    #[clap(long, default_value = "0")]
    pub print_stats: usize,

    /// for dreamcoder comparison only: this makes stitch drop its final searchh
    /// result and return one less invention than you asked for while still
    /// doing the work of finding that last invention. This simulations how dreamcoder
    /// finds and rejects its final candidate
    #[clap(long)]
    pub dreamcoder_drop_last: bool,

    /// disable caching (though caching isn't used for much currently)
    #[clap(long)]
    pub no_cache: bool,

    /// print out programs rewritten under invention
    #[clap(long,short='r')]
    pub show_rewritten: bool,

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

    /// Disable stat logging - note that stat logging in multithreading requires taking a mutex
    /// so it could be a source of slowdown in the multithreaded case, hence this flag to disable it.
    /// From some initial tests it seems to cause no slowdown anyways though.
    #[clap(long)]
    pub no_stats: bool,

    /// disables other_utility so the only utility is based on compressivity
    #[clap(long)]
    pub no_other_util: bool,

    /// whenever you finish an invention do a full rewrite to check that rewriting doesnt raise a mismatch exception
    #[clap(long)]
    pub rewrite_check: bool,

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
    pub refinements: Vec<Option<Vec<Id>>>, // refinements[i] gives the list of refinements for #i
    pub match_locations: Vec<Id>, // places where it applies
    pub utility_upper_bound: i32,
    pub body_utility_no_refinement: i32, // the size (in `cost`) of a single use of the pattern body so far
    pub refinement_body_utility: i32, // modifier on body_utility to include the full size account for refinement
    pub tracked: bool, // for debugging
}

impl Expr {
    fn zipper_replace(&self, zip: &Zip, new: &str) -> Expr {
        let child = self.apply_zipper(zip).unwrap();
        // clone and overwrite that node
        let mut res = self.clone();
        res.nodes[usize::from(child)] = Lambda::Prim(new.into());
        res
    }
    /// replaces the node at the end of the zipper with `new` prim,
    /// returning the new expression
    fn apply_zipper(&self, zip: &Zip) -> Option<Id> {
        let mut child = self.root();
        for znode in zip.iter() {
            child = match (znode, self.get(child)) {
                (ZNode::Body, Lambda::Lam([b])) => *b,
                (ZNode::Func, Lambda::App([f,_])) => *f,
                (ZNode::Arg, Lambda::App([_,x])) => *x,
                (_,_) => return None // no zipper works here
            };
        }
        Some(child)
    }
}

/// returns the vec of zippers to each ivar
fn zids_of_ivar_of_expr(expr: &Expr, zid_of_zip: &HashMap<Zip,ZId>) -> Vec<Vec<ZId>> {

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

    fn helper(curr_node: Id, expr: &Expr, curr_zip: &mut Zip, zids_of_ivar: &mut Vec<Vec<ZId>>, zid_of_zip: &HashMap<Zip,ZId>) {
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
    fn single_hole(treenodes: &Vec<Id>, cost_of_node_all: &HashMap<Id,i32>, num_paths_to_node: &HashMap<Id,i32>, egraph: &crate::EGraph, cfg: &CompressionStepConfig) -> Self {
        let body_utility_no_refinement = 0;
        let refinement_body_utility = 0;
        let mut match_locations = treenodes.clone();
        match_locations.sort(); // we assume match_locations is always sorted
        if cfg.no_top_lambda {
            match_locations.retain(|node| expands_to_of_node(&egraph[*node].nodes[0]) != ExpandsTo::Lam);
        }
        let utility_upper_bound = utility_upper_bound(&match_locations, body_utility_no_refinement + refinement_body_utility, cost_of_node_all, num_paths_to_node, cfg);
        Pattern {
            holes: vec![EMPTY_ZID], // (zid 0 is the empty zipper)
            arg_choices: vec![],
            first_zid_of_ivar: vec![],
            refinements: vec![],
            match_locations, // single hole matches everywhere
            utility_upper_bound,
            body_utility_no_refinement, // 0 body utility
            refinement_body_utility, // 0 body utility
            tracked: cfg.track.is_some(),
        }
    }
    /// convert pattern to an Expr with `??` in place of holes and `?#` in place of argchoices
    fn to_expr(&self, shared: &SharedData) -> Expr {
        let mut curr_zip: Zip = vec![];
        // map zids to zips with a bool thats true if this is a hole and false if its a future ivar
        let zips: Vec<(Zip,Expr)> = self.holes.iter().map(|zid| (shared.zip_of_zid[*zid].clone(), Expr::prim("??".into())))
            .chain(self.arg_choices.iter().map(|labelled_zid| (shared.zip_of_zid[labelled_zid.zid].clone(),
                if let Some(refinements) = self.refinements[labelled_zid.ivar].as_ref() {

                    // extract the refinement and remap #i to $(i+depth) where depth is depth of #i in `extracted`
                    let mut extracted = refinements.iter().map(|refinement| extract(*refinement, &shared.egraph)).collect::<Vec<_>>();
                    extracted.iter_mut().for_each(|e| arg_ivars_to_vars(e));
                    let mut expr = Expr::ivar(labelled_zid.ivar as i32);
                    // todo are these applied in the right order?
                    for e in extracted {
                        expr = Expr::app(expr,e);
                    }
                    expr
                } else {
                    Expr::ivar(labelled_zid.ivar as i32)
                }))).collect();


        fn helper(curr_node: Id, curr_zip: &mut Zip, zips: &Vec<(Zip,Expr)>, shared: &SharedData) -> Expr {
            match zips.iter().find(|(zip,_)| zip == curr_zip) {
                // current zip matches a hole
                Some((_,e)) => e.clone(),
                // no ivar zip match, so recurse
                None => {
                    match &shared.egraph[curr_node].nodes[0] {
                        Lambda::Prim(p) => Expr::prim(*p),
                        Lambda::Var(v) => Expr::var(*v),
                        Lambda::Lam([b]) => {
                            curr_zip.push(ZNode::Body);
                            let b_expr = helper(*b, curr_zip, &zips, shared);
                            curr_zip.pop();
                            Expr::lam(b_expr) 
                        }
                        Lambda::App([f,x]) => {
                            curr_zip.push(ZNode::Func);
                            let f_expr = helper(*f, curr_zip, &zips, shared);
                            curr_zip.pop();
                            curr_zip.push(ZNode::Arg);
                            let x_expr = helper(*x, curr_zip, &zips, shared);
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
        let mut s = self.to_expr(shared).zipper_replace(&shared.zip_of_zid[hole_zid], &"<REPLACE>" ).to_string();
        s = s.replace(&"<REPLACE>", &format!("{}",tracked_expands_to(self, hole_zid, shared)).clone().magenta().bold().to_string());
        s
    }
    pub fn info(&self, shared: &SharedData) -> String {
        format!("{}: utility_upper_bound={}, body_utility=({},{}), refinements={}, match_locations={}, usages={}",self.to_expr(shared), self.utility_upper_bound, self.body_utility_no_refinement, self.refinement_body_utility, self.refinements.iter().filter(|x|x.is_some()).count(), self.match_locations.len(), self.match_locations.iter().map(|loc|shared.num_paths_to_node[loc]).sum::<i32>())
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
    fn is_ivar(&self) -> bool {
        match self {
            ExpandsTo::IVar(_) => true,
            _ => false
        }
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
    let id = shared.tracking.as_ref().unwrap().expr
        .apply_zipper(&shared.zip_of_zid[hole_zid]).unwrap();
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
            return ExpandsTo::IVar(pattern.first_zid_of_ivar.len() as i32);
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
    fn new(pattern: Pattern, num_paths_to_node: &HashMap<Id,i32>) -> Self {
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
    active_threads: HashSet<std::thread::ThreadId>, // list of threads currently holding worklist items
}

/// All the data shared among threads, mostly read-only
/// except for the mutexes
#[derive(Debug)]
pub struct SharedData {
    pub crit: Mutex<CriticalMultithreadData>,
    pub arg_of_zid_node: Vec<HashMap<Id,Arg>>,
    pub treenodes: Vec<Id>,
    pub programs_node: Id,
    pub zids_of_node: HashMap<Id,Vec<ZId>>,
    pub zip_of_zid: Vec<Zip>,
    pub zid_of_zip: HashMap<Zip, ZId>,
    pub extensions_of_zid: Vec<ZIdExtension>,
    // pub refinables_of_shifted_arg: HashMap<Id,Vec<Id>>,
    // pub uses_of_zid_refinable_loc: HashMap<(ZId,Id,Id),i32>,
    pub uses_of_shifted_arg_refinement: HashMap<Id,HashMap<Id,usize>>,
    pub egraph: EGraph,
    pub num_paths_to_node: HashMap<Id,i32>,
    pub tasks_of_node: HashMap<Id, HashSet<usize>>,
    pub cost_of_node_once: HashMap<Id,i32>,
    pub cost_of_node_all: HashMap<Id,i32>,
    pub stats: Mutex<Stats>,
    pub cfg: CompressionStepConfig,
    pub tracking: Option<Tracking>,
}

/// Used for debugging tracking information
#[derive(Debug)]
pub struct Tracking {
    expr: Expr,
    zids_of_ivar: Vec<Vec<ZId>>,
    refined: Option<Expr>,
}

impl CriticalMultithreadData {
    /// Create a new mutable multithread data struct with
    /// a worklist that just has a single hole on it
    fn new(donelist: Vec<FinishedPattern>, treenodes: &Vec<Id>, cost_of_node_all: &HashMap<Id,i32>, num_paths_to_node: &HashMap<Id,i32>, egraph: &crate::EGraph, cfg: &CompressionStepConfig) -> Self {
        // push an empty hole onto a new worklist
        let mut worklist = BinaryHeap::new();
        worklist.push(HeapItem::new(Pattern::single_hole(treenodes, cost_of_node_all, num_paths_to_node, egraph, cfg),num_paths_to_node));
        
        let mut res = CriticalMultithreadData {
            donelist,
            worklist,
            utility_pruning_cutoff: 0,
            active_threads: HashSet::new(),
        };
        res.update(cfg);
        res
    }
    /// sort the donelist by utility, truncate to cfg.inv_candidates, update 
    /// update utility_pruning_cutoff to be the lowest utility
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
        let map: HashMap<i32, Expr> = args.iter().enumerate().map(|(i,e)| (i as i32, e.clone())).collect();
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
enum ZNode {
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
    upper_bound_fired: usize,
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
    First,
    Last,
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
    fn choose_hole(&self, pattern: &Pattern, shared: &SharedData) -> usize {
        if pattern.holes.len() == 1 {
            return 0;
        }
        match self {
            &HoleChoice::First => 0,
            &HoleChoice::Last => pattern.holes.len() - 1,
            &HoleChoice::Random => {
                let mut rng = rand::thread_rng();
                rng.gen_range(0..pattern.holes.len())
            },
            &HoleChoice::FewApps => {
                pattern.holes.iter().enumerate().map(|(hole_idx,hole_zid)|
                    (hole_idx, pattern.match_locations.iter().filter(|loc|shared.arg_of_zid_node[*hole_zid][loc].expands_to == ExpandsTo::App).count()))
                        .min_by_key(|x|x.1).unwrap().0
            }
            &HoleChoice::MaxCost => {
                pattern.holes.iter().enumerate().map(|(hole_idx,hole_zid)|
                    (hole_idx, pattern.match_locations.iter().map(|loc|shared.arg_of_zid_node[*hole_zid][loc].cost).sum::<i32>()))
                        .max_by_key(|x|x.1).unwrap().0
            }
            &HoleChoice::MinCost => {
                pattern.holes.iter().enumerate().map(|(hole_idx,hole_zid)|
                    (hole_idx, pattern.match_locations.iter().map(|loc|shared.arg_of_zid_node[*hole_zid][loc].cost).sum::<i32>()))
                        .min_by_key(|x|x.1).unwrap().0
            }
            &HoleChoice::MaxLargestSubset => {
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
        LabelledZId { zid: zid, ivar: ivar }
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
        println!("{} @ step={} util={} for {}", "[new best utility]".blue(), shared.stats.lock().deref_mut().worklist_steps, crit.donelist.first().unwrap().utility, crit.donelist.first().unwrap().info(shared));
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

    // with dynamic batch size, take worklist_size/num_threads items from the worklist
    let batch_size = if shared.cfg.dynamic_batch { std::cmp::max(1, crit.worklist.len() / shared.cfg.threads ) } else { shared.cfg.batch };
    loop {
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
        } else {
            if !shared.cfg.no_stats { shared.stats.lock().deref_mut().upper_bound_fired += 1; };
        }
    }
    // * MULTITHREADING: CRITICAL SECTION END *
}

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
            if !shared.cfg.no_stats { if shared.cfg.print_stats > 0 &&  shared.stats.lock().deref_mut().worklist_steps % shared.cfg.print_stats == 0 { println!("{:?}",shared.stats.lock().deref_mut()); }};

            if shared.cfg.verbose_worklist {
                println!("[prio={}; uses={}] chose: {}", original_pattern.utility_upper_bound, original_pattern.match_locations.len(), original_pattern.to_expr(&shared));
            }

            // choose which hole we're going to expand
            let hole_idx: usize = shared.cfg.hole_choice.choose_hole(&original_pattern, &shared);

            // pop that hole form the list of holes
            let mut holes_after_pop: Vec<ZId> = original_pattern.holes.clone();
            let hole_zid: ZId = holes_after_pop.remove(hole_idx);

            // get the hashmap of args for this hole
            let ref arg_of_loc = shared.arg_of_zid_node[hole_zid];

            // sort the match locations by node type (ie what theyll expand into) so that we can do a group_by() on
            // node type in order to iterate over all the different expansions
            // We also sort secondarily by `loc` to ensure each groupby subsequence has the locations in sorted order
            let mut match_locations = original_pattern.match_locations.clone();
            match_locations.sort_unstable_by_key(|loc| (arg_of_loc[loc].expands_to.clone(), *loc));

            let mut ivars_expansions = vec![];

            // consider all ivars used previously
            for ivar in 0..original_pattern.first_zid_of_ivar.len() {
                let ref arg_of_loc_ivar = shared.arg_of_zid_node[original_pattern.first_zid_of_ivar[ivar]];
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

            let mut found_tracked = false;
            // for each way of expanding the hole...
            'expansion:
                for (expands_to, locs) in match_locations.into_iter()
                .group_by(|loc| arg_of_loc[loc].expands_to.clone()).into_iter()
                .map(|(expands_to, locs)| (expands_to, locs.collect::<Vec<Id>>()))
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
                if !shared.cfg.no_opt_single_use && !shared.cfg.no_opt_arity_zero && locs.len()  == 1 && shared.egraph[locs[0]].data.free_vars.is_empty() {
                    if !shared.cfg.no_stats { shared.stats.lock().deref_mut().single_use_fired += 1; }
                    continue 'expansion;
                }

                // prune inventions specific to one single task
                if !shared.cfg.no_opt_single_task
                        && locs.iter().all(|node| shared.tasks_of_node[&node].len() == 1)
                        && locs.iter().all(|node| shared.tasks_of_node[&locs[0]].iter().next() == shared.tasks_of_node[&node].iter().next()) {
                    if !shared.cfg.no_stats { shared.stats.lock().deref_mut().single_task_fired += 1; }
                    if tracked { println!("{} single task pruned when expanding {} to {}", "[TRACK]".red().bold(), original_pattern.to_expr(&shared), original_pattern.to_expr(&shared).zipper_replace(&shared.zip_of_zid[hole_zid], &format!("<{}>",expands_to))); }
                    continue 'expansion;
                }

                // check for free variables: if an invention has free variables in the body then it's not a real function and we can discard it
                // Here we just check if our expansion just yielded a variable, and if that is bound based on how many lambdas there are above it.
                if true || !shared.cfg.no_opt_free_vars {
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
                            // AND there's no potential for refining that arg
                            && (!shared.cfg.refine || locs.iter().all(|loc| shared.egraph[shared.arg_of_zid_node[argchoice.zid][loc].shifted_id].data.free_ivars.is_empty()))
                        {
                            if !shared.cfg.no_stats { shared.stats.lock().deref_mut().useless_abstract_fired += 1; };
                            continue 'expansion; // useless abstraction
                        }
                    }

                }


                // update the body utility
                let body_utility_no_refinement = original_pattern.body_utility_no_refinement +  match expands_to {
                    ExpandsTo::Lam | ExpandsTo::App => COST_NONTERMINAL,
                    ExpandsTo::Var(_) | ExpandsTo::Prim(_) => COST_TERMINAL,
                    ExpandsTo::IVar(_) => 0,
                };
                let refinement_body_utility = original_pattern.refinement_body_utility;

                // update the upper bound
                let util_upper_bound: i32 = utility_upper_bound(&locs, body_utility_no_refinement + refinement_body_utility, &shared.cost_of_node_all, &shared.num_paths_to_node, &shared.cfg);

                assert!(util_upper_bound <= original_pattern.utility_upper_bound);

                // branch and bound: if the upper bound is less than the best invention we've found so far (our cutoff), we can discard this pattern
                if !shared.cfg.no_opt_upper_bound && util_upper_bound <= weak_utility_pruning_cutoff {
                    if !shared.cfg.no_stats { shared.stats.lock().deref_mut().upper_bound_fired += 1; };
                    if tracked { println!("{} upper bound ({} < {}) pruned when expanding {} to {}", "[TRACK]".red().bold(), util_upper_bound, weak_utility_pruning_cutoff, original_pattern.to_expr(&shared), original_pattern.show_track_expansion(hole_zid, &shared)); }
                    continue 'expansion; // too low utility
                }

                assert!(shared.cfg.no_opt_upper_bound || !(holes_after_pop.is_empty() && original_pattern.arg_choices.is_empty() && !expands_to.has_holes() && !expands_to.is_ivar()),
                        "unexpected arity 0 invention: upper bounds + priming with arity 0 inventions should have prevented this");
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
                let mut refinements = original_pattern.refinements.clone();
                if let ExpandsTo::IVar(i) = expands_to {
                    arg_choices.push(LabelledZId::new(hole_zid, i as usize));
                    if i as usize == original_pattern.first_zid_of_ivar.len() {
                        first_zid_of_ivar.push(hole_zid);
                        refinements.push(None)
                    }
                }

                // if two different ivars #i and #j have the same arg at every location, then we can prune this pattern
                // because there must exist another pattern where theyre just both the same ivar. Note that this pruning
                // happens here and not just at the ivar creation point because new subsetting can happen
                if !shared.cfg.no_opt_force_multiuse {
                    // for all pairs of ivars #i and #j, get the first zipper and compare the arg value across all locations
                    for (i,ivar_zid_1) in first_zid_of_ivar.iter().enumerate() {
                        let ref arg_of_loc_1 = shared.arg_of_zid_node[*ivar_zid_1];
                        for ivar_zid_2 in first_zid_of_ivar.iter().skip(i+1) {
                            let ref arg_of_loc_2 = shared.arg_of_zid_node[*ivar_zid_2];
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
            let mut new_pattern = Pattern {
                holes,
                arg_choices,
                first_zid_of_ivar,
                refinements,
                match_locations: locs,
                utility_upper_bound: util_upper_bound,
                body_utility_no_refinement,
                refinement_body_utility,
                tracked
            };


            if new_pattern.holes.is_empty() {
                // it's a finished pattern
                // refinement

                if shared.cfg.refine {
                    if tracked {
                        println!("{} refining {}", "[TRACK:REFINE]".yellow().bold(), new_pattern.to_expr(&shared));
                    }

                    let mut best_refinement: Vec<Option<Vec<Id>>> = new_pattern.refinements.clone(); // initially all Nones
                    let mut best_utility = noncompressive_utility(new_pattern.body_utility_no_refinement + new_pattern.refinement_body_utility, &shared.cfg) + compressive_utility(&new_pattern,&shared).util;
                    let mut best_refinement_body_utility = 0;
                    assert!(new_pattern.refinement_body_utility == 0);
                    assert!(new_pattern.refinements.iter().all(|refinement| refinement.is_none()));

                    // get all refinement options for each arg, deduped
                    let mut refinements_by_arg: Vec<Vec<Id>> = new_pattern.first_zid_of_ivar.iter().map(|zid| 
                        new_pattern.match_locations.iter().flat_map(|loc|
                            shared.uses_of_shifted_arg_refinement.get(&shared.arg_of_zid_node[*zid][loc].shifted_id).map(|uses_of_refinement| uses_of_refinement.keys())
                        ).flatten().cloned().collect::<HashSet<_>>().into_iter().collect()).collect();

                    for (i,_) in new_pattern.first_zid_of_ivar.iter().enumerate() { // for each arg
                        if new_pattern.arg_choices.iter().filter(|l |l.ivar == i).count() > 1 {
                            refinements_by_arg[i] = Vec::new(); // todo limitation: we dont refine multiuse
                        }
                    }

                    let mut num_refinements = 0;

                    'refinements: for refinements in refinements_by_arg.into_iter()
                        .map(|refinements|
                                (1..=shared.cfg.max_refinement_arity).map(move |k| refinements.clone().into_iter()
                                    .combinations(k))
                                .flatten()
                                .map(|r| Some(r))
                                .chain(std::iter::once(None))
                                )
                        .multi_cartesian_product()
                    {
                        num_refinements += 1;
                        // insert the refinement
                        new_pattern.refinements = refinements.clone();
                        // body grows by an APP and the refined out subtree's size
                        new_pattern.refinement_body_utility = refinements.iter()
                            .flat_map(|r| r)
                            .map(|r| r.iter()
                                .map(|r_id| COST_NONTERMINAL + shared.egraph[*r_id].data.inventionless_cost).sum::<i32>()).sum::<i32>();

                        let utility = noncompressive_utility(new_pattern.body_utility_no_refinement + new_pattern.refinement_body_utility, &shared.cfg) + compressive_utility(&new_pattern,&shared).util;
                        if utility > best_utility {
                            best_refinement = refinements.clone();
                            best_utility = utility;
                            best_refinement_body_utility = new_pattern.refinement_body_utility;
                        }
                        if tracked {
                            println!("{} refined to {} (util: {})", "[TRACK:REFINE]".yellow().bold(), new_pattern.to_expr(&shared), utility);
                            println!("{:?}", refinements);
                            if let Some(track_refined) = &shared.tracking.as_ref().unwrap().refined {
                                let refined = new_pattern.to_expr(&shared).to_string();
                                let track_refined = track_refined.to_string();
                                if refined == track_refined {
                                    println!("{} previous refinement was the tracked one! Forcing it to accept that one", "[TRACK:REFINE]".green().bold());
                                    best_refinement = refinements.clone();
                                    best_refinement_body_utility = new_pattern.refinement_body_utility;
                                    new_pattern.refinement_body_utility = 0;
                                    break 'refinements;
                                }
                            }
                        }
                        // reset body utility
                        new_pattern.refinement_body_utility = 0;
                    }
                    if num_refinements > 1000 {
                        println!("[many refinements] tried {} refinements for {}", num_refinements, new_pattern.to_expr(&shared));
                    }

                    // set to the best
                    new_pattern.refinements = best_refinement.clone();
                    new_pattern.refinement_body_utility = best_refinement_body_utility;
                }

                let finished_pattern = FinishedPattern::new(new_pattern, &shared);

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
                    if tracked { println!("{} pushed {} to worklist (bound: {})", "[TRACK]".green().bold(), original_pattern.show_track_expansion(hole_zid, &shared), new_pattern.utility_upper_bound); }
                    worklist_buf.push(HeapItem::new(new_pattern, &shared.num_paths_to_node))
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
    fn new(pattern: Pattern, shared: &SharedData) -> Self {
        let arity = pattern.first_zid_of_ivar.len();
        let usages = pattern.match_locations.iter().map(|loc| shared.num_paths_to_node[loc]).sum();
        let compressive_utility = compressive_utility(&pattern,shared);
        let noncompressive_utility = noncompressive_utility(pattern.body_utility_no_refinement + pattern.refinement_body_utility, &shared.cfg);
        let utility = noncompressive_utility + compressive_utility.util;
        assert!(utility <= pattern.utility_upper_bound, "{} BUT utility is higher: {} (usages: {})", pattern.info(&shared), utility, usages);
        FinishedPattern {
            pattern,
            utility,
            compressive_utility: compressive_utility.util,
            util_calc: compressive_utility,
            arity,
            usages,
        }
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

/// return all possible refinements you could use here along with counts for how many of each you can use
fn get_refinements_of_shifted_id(shifted_id: Id, egraph: &crate::EGraph, cfg: &CompressionStepConfig) -> HashMap<Id,usize>
{
    fn helper(id: Id, egraph: &crate::EGraph, cfg: &CompressionStepConfig, refinements: &mut Vec<Id>) {
        let ivars =  egraph[id].data.free_ivars.len();
        if ivars == 0 {
            return; // todo limitation: we dont thread things that dont have ivars, for sortof good reasons.
        }
        
        if !egraph[id].data.free_vars.is_empty() {
            return; // if something has free vars and theyre not turned into ivars they must either be refs to lambdas within the arg or refs ABOVE the invention which in either case we can't refine
        }

        if egraph[id].data.inventionless_cost <= cfg.max_refinement_size.unwrap_or(i32::MAX) {
            refinements.push(id);
        }

        // ivar!
        for child in egraph[id].nodes[0].children().iter() {
            helper(*child, egraph, cfg, refinements);
        }
    }
    let mut refinements = vec![];
    helper(shifted_id, egraph, cfg, &mut refinements);
    refinements.into_iter().counts()
}

/// figure out all the N^2 zippers from choosing any given node and then choosing a descendant and returning the zipper from
/// the node to the descendant. We also collect a bunch of other useful stuff like the argument you would get if you abstracted
/// the descendant and introduced an invention rooted at the ancestor node.
fn get_zippers(
    treenodes: &[Id],
    cost_of_node_once: &HashMap<Id,i32>,
    no_cache: bool,
    egraph: &mut crate::EGraph,
    cfg: &CompressionStepConfig
) -> (HashMap<Zip, ZId>, Vec<Zip>, Vec<HashMap<Id,Arg>>, HashMap<Id,Vec<ZId>>,  Vec<ZIdExtension>, HashMap<Id,HashMap<Id,usize>>) {
    let cache: &mut Option<RecVarModCache> = &mut if no_cache { None } else { Some(HashMap::new()) };

    let mut zid_of_zip: HashMap<Zip, ZId> = Default::default();
    let mut zip_of_zid: Vec<Zip> = Default::default();
    let mut arg_of_zid_node: Vec<HashMap<Id,Arg>> = Default::default();
    let mut zids_of_node: HashMap<Id,Vec<ZId>> = Default::default();


    // let mut refinements_of_shifted_arg: HashMap<Id,HashSet<Id>> = Default::default();
    // let mut uses_of_zid_refinable_loc: HashMap<(ZId,Id,Id),i32> = Default::default();
    let mut uses_of_shifted_arg_refinement: HashMap<Id,HashMap<Id,usize>> = Default::default();


    zid_of_zip.insert(vec![], EMPTY_ZID);
    zip_of_zid.push(vec![]);
    arg_of_zid_node.push(HashMap::new());
    assert!(EMPTY_ZID == 0);
    
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
            Arg { shifted_id: *treenode, unshifted_id: *treenode, shift: 0, cost: cost_of_node_once[treenode], expands_to: expands_to_of_node(&node) });
        
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
                        arg_of_zid_node.push(HashMap::new());
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
                        arg_of_zid_node.push(HashMap::new());
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
                        arg_of_zid_node.push(HashMap::new());
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
                        if cfg.refine {
                            // refinements:
                            if !uses_of_shifted_arg_refinement.contains_key(&arg.shifted_id) {
                                uses_of_shifted_arg_refinement.insert(arg.shifted_id,get_refinements_of_shifted_id(arg.shifted_id, &egraph, cfg));
                            }
                            // refinements_of_shifted_arg.entry(arg.shifted_id).or_default().extend(refinement_counts.keys().cloned());
                            // uses_of_zid_refinable_loc
                        }
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
    extensions_of_zid,
    uses_of_shifted_arg_refinement)
}

#[derive(Debug, Clone)]
pub struct CompressionStepData {
    pub rewritten: Expr,
    pub rewritten_dreamcoder: Vec<String>,
    pub initial_cost: i32,
    pub expected_cost: i32,
    pub final_cost: i32,
    pub multiplier: f64,
    pub multiplier_wrt_orig: f64,
    pub uses: i32,
    pub use_exprs: Vec<Expr>,
    pub use_args: Vec<Vec<Expr>>,
    is_test: bool,
}

impl CompressionStepData {
    fn new(done: FinishedPattern, programs_node: Id, inv: &Invention, shared: &mut SharedData, past_invs: &Vec<CompressionStepResult>, very_first_cost: Option<i32>, do_fast_rewrite: bool, is_test: bool) -> Self {
        let initial_cost = shared.egraph[programs_node].data.inventionless_cost;
        let roots: Vec<Id> = shared.egraph[programs_node].nodes[0].children().iter().cloned().collect();

        // cost of the very first initial program before any inventions
        let very_first_cost = very_first_cost.unwrap_or(initial_cost);

        let inv_name = &inv.name;
        let rewritten = if do_fast_rewrite {
            Expr::programs(rewrite_fast(&done, roots.clone(), &shared, inv_name))
        } else {
            let r = Expr::programs(roots.clone().into_iter().map(|r| rewrite_with_invention_egraph(r, inv, &mut shared.egraph)).collect());
            if !is_test {
                let fast = Expr::programs(rewrite_fast(&done, roots, &shared, inv_name)).to_string();
                assert_eq!(r.to_string(), fast);
            }
            r
        };

        let expected_cost = if !is_test {
            initial_cost - done.compressive_utility
        } else {
            -1  // no sense of expected cost for test sets
        };
        let final_cost = rewritten.cost();
        if !is_test && expected_cost != final_cost {
            println!("*** expected cost {} != final cost {}", expected_cost, final_cost);
        }
        let multiplier = initial_cost as f64 / final_cost as f64;
        let multiplier_wrt_orig = very_first_cost as f64 / final_cost as f64;
        let uses = rewritten.to_string()
                                        .as_bytes()
                                        .windows(inv_name.len() + 1)
                                        .filter(|&wdw| (inv_name.to_owned() + " ").as_bytes() == wdw
                                                              || (inv_name.to_owned() + ")").as_bytes() == wdw)
                                        .count() as i32;
        let use_exprs: Vec<Expr> = vec![];
        let use_args: Vec<Vec<Expr>> = vec![vec![]];
        
        // dreamcoder compatability
        let dc_inv_str: String = dc_inv_str(&inv, past_invs);
        // Rewrite to dreamcoder syntax with all past invention
        // we rewrite "inv1)" and "inv1 " instead of just "inv1" because we dont want to match on "inv10"
        let rewritten_dreamcoder: Vec<String> = rewritten.split_programs().iter().map(|p|{
            let mut res = p.to_string();
            for past_inv in past_invs {
                res = replace_prim_with(&res, &past_inv.inv.name, &past_inv.dc_inv_str);
                // res = res.replace(&format!("{})",past_inv.inv.name), &format!("{})",past_inv.dc_inv_str));
                // res = res.replace(&format!("{} ",past_inv.inv.name), &format!("{} ",past_inv.dc_inv_str));
            }
            res = replace_prim_with(&res, inv_name, &dc_inv_str);
            // res = res.replace(&format!("{})",inv_name), &format!("{})",dc_inv_str));
            // res = res.replace(&format!("{} ",inv_name), &format!("{} ",dc_inv_str));
            res = res.replace("(lam ","(lambda ");
            res
        }).collect();

        CompressionStepData {rewritten, rewritten_dreamcoder, initial_cost, expected_cost, final_cost, multiplier, multiplier_wrt_orig, uses, use_exprs, use_args, is_test}
    }

    pub fn json(&self, inv_name: &str) -> serde_json::Value {        
        let use_exprs: Vec<String> = self.use_exprs.iter().map(|expr| expr.to_string()).collect();
        let use_args: Vec<String> = self.use_args.iter().map(|args| format!("{} {}", inv_name, args.iter().map(|expr| expr.to_string()).collect::<Vec<String>>().join(" "))).collect();
        let all_uses: Vec<serde_json::Value> = use_exprs.iter().zip(use_args.iter()).sorted().map(|(expr,args)| json!({args: expr})).collect();

        json!({            
            "rewritten": self.rewritten.split_programs().iter().map(|p| p.to_string()).collect::<Vec<String>>(),
            "rewritten_dreamcoder": self.rewritten_dreamcoder,
            "expected_cost": self.expected_cost,
            "final_cost": self.final_cost,
            "multiplier": self.multiplier,
            "multiplier_wrt_orig": self.multiplier_wrt_orig,
            "num_uses": self.uses,
            "uses": all_uses,
        })
    }
}

impl fmt::Display for CompressionStepData {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let dataset = if self.is_test { "test" } else { "train" };
        write!(f, "final_cost ({0}): {1} | multiplier ({0}): {2:.2}x | uses ({0}): {3}",
            dataset,
            self.final_cost,
            self.multiplier,
            self.uses
        )
    }
}

/// the complete result of a single step of compression, this is a somewhat expensive data structure
/// to create.
#[derive(Debug, Clone)]
pub struct CompressionStepResult {
    pub inv: Invention,
    pub done: FinishedPattern,
    pub dc_inv_str: String,
    pub train_data: CompressionStepData,
    pub test_data: Option<CompressionStepData>,
}

impl CompressionStepResult {
    fn new(done: FinishedPattern, train_programs_node: Id, test_programs_node: Option<Id>, inv_name: &str, shared: &mut SharedData, past_invs: &Vec<CompressionStepResult>, first_train_cost: Option<i32>, first_test_cost: Option<i32>) -> Self {

        let inv = done.to_invention(inv_name, shared);

        let train_data = CompressionStepData::new(done.clone(), train_programs_node, &inv, shared, past_invs, first_train_cost, false, false);
        let test_data = test_programs_node.map(|n| CompressionStepData::new(done.clone(), n, &inv, shared, past_invs, first_test_cost, false, true));

        let dc_inv_str: String = dc_inv_str(&inv, past_invs);

        CompressionStepResult { inv, done, dc_inv_str, train_data, test_data }
    }

    pub fn json(&self) -> serde_json::Value {        

        json!({            
            "body": self.inv.body.to_string(),
            "dreamcoder": self.dc_inv_str,
            "arity": self.inv.arity,
            "name": self.inv.name,
            "utility": self.done.utility,
            "train_result": self.train_data.json(&self.inv.name),
            "test_result": self.test_data.as_ref().map(|d| d.json(&self.inv.name).to_string()),
        })
    }
}

impl fmt::Display for CompressionStepResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.train_data.expected_cost != self.train_data.final_cost {
            write!(f,"[cost mismatch of {} in training data] ", self.train_data.expected_cost - self.train_data.final_cost)?;
        }
        write!(f, "utility: {} | {}{} | body: {}",
            self.done.utility,
            self.train_data,
            self.test_data.as_ref().map_or("".to_string(), |d| format!(" | {}", d)),
            self.inv)
    }
}

/// calculates the total upper bound on compressive + noncompressive utility
#[inline]
fn utility_upper_bound(
    match_locations: &Vec<Id>,
    body_utility_with_refinement_lower_bound: i32,
    cost_of_node_all: &HashMap<Id,i32>,
    num_paths_to_node: &HashMap<Id,i32>,
    cfg: &CompressionStepConfig,
) -> i32 {
    compressive_utility_upper_bound(match_locations, cost_of_node_all, num_paths_to_node)
        + noncompressive_utility_upper_bound(body_utility_with_refinement_lower_bound, cfg)
}

/// This utility is just for any utility terms that we care about that don't directly correspond
/// to changes in size that come from rewriting with an invention
fn noncompressive_utility(
    body_utility_with_refinement: i32,
    cfg: &CompressionStepConfig,
) -> i32 {
    if cfg.no_other_util { return 0; }
    // this is a bit like the structure penalty from dreamcoder except that
    // that penalty uses inlined versions of nested inventions.
    let structure_penalty = - body_utility_with_refinement;
    structure_penalty
}

/// This takes a partial invention and gives an upper bound on the maximum
/// compressive_utility() that any completed offspring of this partial invention could have.
#[inline]
fn compressive_utility_upper_bound(
    match_locations: &Vec<Id>,
    cost_of_node_all: &HashMap<Id,i32>,
    num_paths_to_node: &HashMap<Id,i32>
) -> i32 {
    match_locations.iter().map(|node|
        cost_of_node_all[node] 
        - num_paths_to_node[node] * COST_TERMINAL).sum::<i32>()
    // COST_TERMINAL is from cost of the invention primitive
}

/// This takes a partial invention and gives an upper bound on the maximum
/// other_utility() that any completed offspring of this partial invention could have.
#[inline]
fn noncompressive_utility_upper_bound(
    body_utility_with_refinement_lower_bound: i32,
    cfg: &CompressionStepConfig,
) -> i32 {
    if cfg.no_other_util { return 0; }
    // safe bound: since structure_penalty is negative an upper bound is anything less negative or exact. Since
    // left_utility < body_utility we know that this will be a less negative bound.
    let structure_penalty = - body_utility_with_refinement_lower_bound;
    structure_penalty
}

fn compressive_utility(pattern: &Pattern, shared: &SharedData) -> UtilityCalculation {

    // * BASIC CALCULATION
    // Roughly speaking compressive utility is num_usages(invention) * size(invention), however there are a few extra
    // terms we need to take care of too.

    // get a list of (ivar,usages-1) filtering out things that are only used once, this will come in handy for adding multi-use utility later
    let ivar_multiuses: Vec<(usize,i32)> = pattern.arg_choices.iter().map(|labelled|labelled.ivar).counts()
        .iter().filter_map(|(ivar,count)| if *count > 1 { Some((*ivar, (*count-1) as i32)) } else { None }).collect();

    // (parent,child) show up in here if they conflict
    let mut refinement_conflicts: HashSet<(Id,Id)> = Default::default();
    for r in pattern.refinements.iter().filter_map(|r|r.as_ref()) {
        for ancestor in r.iter() {
            for descendant in r.iter() {
                if ancestor != descendant && is_descendant(*descendant, *ancestor, &shared.egraph) {
                    refinement_conflicts.insert((*ancestor, *descendant));
                }
            }
        }
    }

    // it costs a tiny bit to apply the invention, for example (app (app inv0 x) y) incurs a cost
    // of COST_TERMINAL for the `inv0` primitive and 2 * COST_NONTERMINAL for the two `app`s.
    // Also an extra COST_NONTERMINAL for each argument that is refined (for the lambda).
    let app_penalty = - (COST_TERMINAL + COST_NONTERMINAL * pattern.first_zid_of_ivar.len() as i32 + COST_NONTERMINAL * pattern.refinements.iter().map(|r|if let Some(refinements) = r {refinements.len() as i32} else {0}).sum::<i32>());


    let utility_of_loc_once: Vec<i32> = pattern.match_locations.iter().map(|loc| {
        // println!("calculating util of {}", extract(*loc, &shared.egraph));
        // compressivity of body (no refinement) minus slight penalty from the application
        let base_utility = pattern.body_utility_no_refinement + app_penalty;
        // println!("base {}", base_utility);

        // each use of the refined out arg gives a benefit equal to the size of the arg
        let refinement_utility: i32 = pattern.refinements.iter().enumerate().filter(|(_,r)| r.is_some()).map(|(ivar,r)| {
            let refinements = r.as_ref().unwrap();
            // grab shifted arg
            let shifted_arg: Id = shared.arg_of_zid_node[pattern.first_zid_of_ivar[ivar]][loc].shifted_id;
            if let Some(uses_of_refinement) =  shared.uses_of_shifted_arg_refinement.get(&shifted_arg) {
                return refinements.iter().map(|refinement| {
                    if let Some(uses) = uses_of_refinement.get(&refinement) {
                        // we subtract COST_TERMINAL because we need to leave behind a $i in place of it in the arg
                        let mut util = (*uses as i32) * (shared.egraph[*refinement].data.inventionless_cost - COST_TERMINAL);
                        // println!("gained util {} from {}", util, extract(*refinement, &shared.egraph));
                        for r in refinements.iter().filter(|r| refinement_conflicts.contains(&(*refinement,**r))) {
                            // we (an ancestor) conflicted with a descendant so we lose some of that descendants util
                            // todo: importantly this doesnt when a grandparent negates both a parent and a child... 
                            // todo that would be necessary for 3+ refinements and would be closer to our full conflict resolution setup
                            assert!(refinements.len() < 3);
                            util -=  (*uses as i32) * (shared.egraph[*r].data.inventionless_cost - COST_TERMINAL);
                        }
                        util
                    } else { 0 }
                }).sum::<i32>()
            }
            // if uses_of_shifted_arg_refinement lacks this shifted_arg then it must not have any refinements so we must not be getting any refinement gain here
            // likewise if the inner hashmap uses_of_shifted_arg_refinement[shifted_arg] lacks this refinement then we wont get any benefit
            0 
        }).sum();

        // println!("refinement {}", refinement_utility);


        // the bad refinement override: if there are any free ivars in the arg at this location (ignoring the refinement itself if there
        // is one) then we can't apply this invention here so *total* util should be 0
        for (ivar,zid) in pattern.first_zid_of_ivar.iter().enumerate() {
            let shifted_arg = shared.arg_of_zid_node[*zid][loc].shifted_id;
            if has_free_ivars(shifted_arg, &pattern.refinements[ivar], &shared.egraph) {
                return 0; // set whole util to 0 for this loc, causing an autoreject
            }
        }

        // for each extra usage of an argument, we gain the cost of that argument as
        // extra utility. Note we use `first_zid_of_ivar` since it doesn't matter which
        // of the zids we use as long as it corresponds to the right ivar
        let multiuse_utility = ivar_multiuses.iter().map(|(ivar,count)|
            count * shared.arg_of_zid_node[pattern.first_zid_of_ivar[*ivar]][loc].cost
        ).sum::<i32>();
        // println!("multiuse {}", multiuse_utility);

        // multiply all this utility by the number of times this node shows up
        base_utility + multiuse_utility + refinement_utility
        }).collect();


    let compressive_utility: i32 = pattern.match_locations.iter()
        .zip(utility_of_loc_once.iter())
        .map(|(loc,utility)| utility * shared.num_paths_to_node[loc])
        .sum();

    // * ACCOUNTING FOR USE CONFLICTS:

    // todo opt include holes too
    // zips and ivars
    let zips: Vec<(Zip,usize)> = pattern.arg_choices.iter().map(|labelled_zid| (shared.zip_of_zid[labelled_zid.zid].clone(),labelled_zid.ivar)).collect();
    
    {
        // assertion to make sure pattern.match_locations is sorted (for binary searching + bottom up iterating)
        let mut largest_seen = -1;
        assert!(pattern.match_locations.iter().all(|x| {
            let res = largest_seen < usize::from(*x) as i32;
            largest_seen = usize::from(*x) as i32;
            res
            }));
    }

    // the idea here is we want the fast-path to be the case where no conflicts happen. If no conflicts happen, there should be
    // zero heap allocations in this whole section! Since empty vecs and hashmaps dont cause allocations yet.
    let mut corrected_utils: HashMap<Id,CorrectedUtil> = Default::default();
    let mut global_correction = 0; // this is going to get added to the compressive_utility at the end to correct for use-conflicts

    // bottom up traversal since we assume match_locations is sorted
    for (loc_idx,loc) in pattern.match_locations.iter().enumerate() {
        // get all the nodes this could conflict with (by idx within `locs` not by id)
        let mut conflict_idxs: Vec<(Id,usize)> = vec![];
        for (zip,ivar) in zips.iter().filter(|(zip,_)| !zip.is_empty()) {
            let mut id = loc;
            // for all except the last node in the zipper, push the childs location on as a potential conflict
            for znode in zip[..zip.len()-1].iter() {
                // step one deeper
                id = match (znode, &shared.egraph[*id].nodes[0]) {
                    (ZNode::Body, Lambda::Lam([b])) => b,
                    (ZNode::Func, Lambda::App([f,_])) => f,
                    (ZNode::Arg, Lambda::App([_,x])) => x,
                    _ => unreachable!()
                };
                // if its also a location, push it to the conflicts list (do NOT dedup)
                if let Ok(idx) = pattern.match_locations.binary_search(id) {
                    conflict_idxs.push((*id,idx));
                }
            }
            // if this is a refinement, push every descendant of the unshifted argument including it itself as a potential conflict
            if let Some(_) = pattern.refinements[*ivar] {
                fn helper(id: Id, shared: &SharedData, conflict_idxs: &mut Vec<(Id,usize)>, pattern: &Pattern) {
                    if let Ok(idx) = pattern.match_locations.binary_search(&id) {
                        conflict_idxs.push((id,idx));
                    }
                    match &shared.egraph[id].nodes[0] {
                        Lambda::Lam([b]) => {helper(*b, shared, conflict_idxs, pattern);},
                        Lambda::App([f,x]) => {
                            helper(*f, shared, conflict_idxs, pattern);
                            helper(*x, shared, conflict_idxs, pattern);
                        }
                        Lambda::Prim(_) | Lambda::Var(_) | Lambda::IVar(_) => {},
                        _ => unreachable!()
                    }
                }
                let unshifted_arg: Id = shared.arg_of_zid_node[pattern.first_zid_of_ivar[*ivar]][loc].unshifted_id;
                helper(unshifted_arg, shared, &mut conflict_idxs, pattern);
            }
        }

        // now we basically record how much we would affect global utility by if we accept vs reject vs choose the best of those options.
        // and recording this will let us change our mind later if we decide to force-reject something

        // if we reject using the invention at this node, we just lose its utility
        let reject = - utility_of_loc_once[loc_idx];

        // Rare case: when utility_of_loc_once is <=0, then reject is >=0 and of course we should do it
        // (it benefits us or rather brings us back to 0, and leaves maximal flexibility for other things to be accepted/rejected).
        // and theres nothing else we need to account for here.
        if reject >= 0 {
            global_correction += reject * shared.num_paths_to_node[loc];
            corrected_utils.insert(*loc, CorrectedUtil {
                accept: false, // we rejected
                best_util_correction: reject, // we rejected
                util_change_to_reject: 0 // we rejected so no change to reject
            });
            continue
        }

        // common case: no conflicts
        // (this has to come AFTER the possible forced rejection)
        if conflict_idxs.is_empty() { continue; }
        
        // if we accept using the invention at this node everywhere, we lose the util of the difference of the best choice of each descendant vs the reject choice
        // so for example if all the conflicts had chosen to Reject anyways then this would be 0 (optimal)
        // but if some chose to Accept then our Accept correction will include the difference caused by forcing them to reject
        // This is easiest to understand if you think of reject as "the effect on global util of rejecting at a single location"
        // and likewise for accept and best.
        let accept = conflict_idxs.iter()
            .map(|(id,idx)|
                corrected_utils.get(id).map(|x|x.util_change_to_reject)
                // if it's not in corrected_utils, it must have had no conflicts so we must be switching from accept to reject with no other side effects
                // so we do (reject - accept) = (- util(idx) - 0) = - util(idx)
                // where accept was 0 since it caused no conflicts
                .unwrap_or_else(|| - utility_of_loc_once[*idx]) 
            ).sum();

        // lets accept the less negative of the options
        let best_util_correction = std::cmp::max(reject,accept);

        // update global correction with this applied to all our nodes (note that the same choice makes sense for all nodes
        // from the point of view of this being the top of the tree - it's our parents job to use change_to_reject if they
        // want to reject only certain ones of us)
        global_correction += best_util_correction * shared.num_paths_to_node[loc];

        let util_change_to_reject = reject - best_util_correction;

        corrected_utils.insert(*loc, CorrectedUtil {
            accept: best_util_correction == accept,
            best_util_correction,
            util_change_to_reject
        });

        // Involved example:
        // A -> B -> C  (ie A conflicts with B conflicts with C; and A is the parent)
        // and also A -> C
        // 
        // First we calculate C.accept C.reject
        // B.reject as - util(B)
        // B.accept as (C.reject - C.best)
        // A.reject as - util(A)
        // A.accept as (B.reject - B.best) + (C.reject - C.best)
        // did we double count C in here since it was a child of both B and A (I mean literally a child of both not just when struct hashed)
        // if B.best was B.reject, then it would involve allowing C.best to happen so that's good that we have the (C.reject - C.best) term
        //
        // if B.best = B.reject:
        // A.accept = (B.reject - B.best) + (C.reject - C.best)
        //          = (C.reject - C.best) = force C to reject
        // which is good because since B was `reject` then yes A needs to include the C rejection term
        //
        // if B.best = B.accept:
        // A.accept = (B.reject - B.best) + (C.reject - C.best)
        //          = (B.reject - B.accept) + (C.reject - C.best)
        //          = (B.reject - (C.reject - C.best)) + (C.reject - C.best)
        //          = B.reject = - util(B)
        // which is good because we've already modified the global util to incorporate C rejection when we decided that B.best 
        // was B.accept, so it's good that the C terms cancel out here. You can think of what happened like this: we force-reject C
        // which creates a (C.reject - C.best) term, but then we force reject B and since B.best was B.accept which involved C rejection,
        // we get another (C.reject - C.best) term that cancels out the first
        //
        // if B.best was B.accept, then (B.reject - B.best) = (B.reject - B.accept) = (B.reject - (C.reject - C.best))

    }

    UtilityCalculation { util: (compressive_utility + global_correction), corrected_utils}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UtilityCalculation {
    pub util: i32,
    pub corrected_utils: HashMap<Id,CorrectedUtil>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorrectedUtil {
    pub accept: bool, // whether it's the best choice to accept applying the invention at this node when there are no other parent nodes above us (ignoring context, would this be the right choice?)
    pub best_util_correction: i32, // the change in utility that this choice would cause. Always <= 0
    pub util_change_to_reject: i32, // if accept=false this is 0 otherwise it's the difference in utility between accept and reject. Always <= 0.
}



/// Multistep compression. See `compression_step` if you'd just like to do a single step of compression.
pub fn compression(
    train_programs_expr: &Expr,
    test_programs_expr: &Option<Expr>,
    iterations: usize,
    cfg: &CompressionStepConfig,
    tasks: &Vec<String>,
    num_prior_inventions: usize,
) -> Vec<CompressionStepResult> {

    if cfg.follow_track && !(
           cfg.no_opt_free_vars
        && cfg.no_opt_single_use
        && cfg.no_opt_single_task
        && cfg.no_opt_upper_bound
        && cfg.no_opt_force_multiuse
        && cfg.no_opt_useless_abstract)
    {
        println!("{} you often want to run --follow-track with --no-opt otherwise your target may get pruned", "[WARNING]".yellow());
    }

    let mut train_rewritten: Expr = train_programs_expr.clone();
    let mut test_rewritten: Option<Expr> = test_programs_expr.clone();
    let mut step_results: Vec<CompressionStepResult> = Default::default();

    let tstart = std::time::Instant::now();

    for i in 0..iterations {
        println!("{}",format!("\n=======Iteration {}=======",i).blue().bold());
        let inv_name = format!("fn_{}", num_prior_inventions + step_results.len());

        // call actual compression
        let res: Vec<CompressionStepResult> = compression_step(
            &train_rewritten,
            &test_rewritten,
            &inv_name,
            &cfg,
            &step_results,
            tasks);

        if !res.is_empty() {
            // rewrite with the invention
            let res: CompressionStepResult = res[0].clone();
            train_rewritten = res.train_data.rewritten.clone();
            test_rewritten = res.test_data.as_ref().map(|d| d.rewritten.clone());
            println!("Chose Invention {}: {}", res.inv.name, res);
            step_results.push(res);
        } else {
            println!("No inventions found at iteration {}",i);
            break;
        }
    }

    if cfg.dreamcoder_drop_last {
        println!("{}",format!("{}","[--dreamcoder-drop-last] dropping final invention".yellow().bold()));
        step_results.pop();
    }

    println!("{}","\n=======Compression Summary=======".blue().bold());
    println!("Found {} inventions", step_results.len());
    let total_inv_sizes: f64 = step_results.iter().map(|r| r.inv.body.cost()).sum::<i32>() as f64;
    println!("Training Set Cost Improvement: ({:.2}x better) {} -> {}", compression_factor(train_programs_expr, &train_rewritten, total_inv_sizes), train_programs_expr.cost(), train_rewritten.cost());
    if let Some(e) = test_programs_expr.as_ref() {
        println!("Testing Set Cost Improvement: ({:.2}x better) {} -> {}", compression_factor(e, &test_rewritten.as_ref().unwrap(), total_inv_sizes), e.cost(), test_rewritten.unwrap().cost());
    }
    for i in 0..step_results.len() {
        let res = &step_results[i];
        println!("{}: {}", res.inv.name.clone().blue(), res);
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
    step_results
}

/// Takes a training set of programs as an Expr with Programs as its root, and does one full step of compresison.
/// Returns the top Inventions and the Expr rewritten under that invention along with other useful info in CompressionStepResult
/// The number of inventions returned is based on cfg.inv_candidates.
/// If test_programs_expr is not None, the test programs will also be rewritten under the chosen invention(s).
pub fn compression_step(
    train_programs_expr: &Expr,
    test_programs_expr: &Option<Expr>,
    new_inv_name: &str, // name of the new invention, like "inv4"
    cfg: &CompressionStepConfig,
    past_invs: &Vec<CompressionStepResult>, // past inventions we've found
    tasks: &Vec<String>,
) -> Vec<CompressionStepResult> {

    let tstart_total = std::time::Instant::now();
    let tstart_prep = std::time::Instant::now();
    let mut tstart = std::time::Instant::now();

    // build the egraph. We'll just be using this as a structural hasher we don't use rewrites at all. All eclasses will always only have one node.
    let mut egraph: EGraph = Default::default();
    let train_programs_node = egraph.add_expr(train_programs_expr.into());
    let test_programs_node = test_programs_expr.as_ref().map(|e| egraph.add_expr(e.into()));
    egraph.rebuild();

    println!("set up egraph: {:?}ms", tstart.elapsed().as_millis());
    tstart = std::time::Instant::now();

    let roots: Vec<Id> = egraph[train_programs_node].nodes[0].children().iter().cloned().collect();

    // all nodes in child-first order except for the Programs node
    let mut treenodes: Vec<Id> = topological_ordering(train_programs_node, &egraph);
    treenodes.retain(|id| *id != train_programs_node);

    println!("got roots and treenodes: {:?}ms", tstart.elapsed().as_millis());
    tstart = std::time::Instant::now();

    // populate num_paths_to_node so we know how many different parts of the programs tree
    // a node participates in (ie multiple uses within a single program or among programs)
    let num_paths_to_node: HashMap<Id,i32> = num_paths_to_node(&roots, &treenodes, &egraph);

    println!("num_paths_to_node(): {:?}ms", tstart.elapsed().as_millis());
    tstart = std::time::Instant::now();

    let tasks_of_node: HashMap<Id, HashSet<usize>> = associate_tasks(train_programs_node, &egraph, tasks);

    println!("associate_tasks(): {:?}ms", tstart.elapsed().as_millis());
    tstart = std::time::Instant::now();

    // cost of a single usage of a node (same as inventionless_cost)
    let cost_of_node_once: HashMap<Id,i32> = treenodes.iter().map(|node| (*node,egraph[*node].data.inventionless_cost)).collect();
    // cost of a single usage times number of paths to node
    let cost_of_node_all: HashMap<Id,i32> = treenodes.iter().map(|node| (*node,cost_of_node_once[node] * num_paths_to_node[node])).collect();

    println!("cost_of_node structs: {:?}ms", tstart.elapsed().as_millis());
    tstart = std::time::Instant::now();

    let (zid_of_zip,
        zip_of_zid,
        arg_of_zid_node,
        zids_of_node,
        extensions_of_zid,
        uses_of_shifted_arg_refinement) = get_zippers(&treenodes, &cost_of_node_once, cfg.no_cache, &mut egraph, cfg);
    
    println!("get_zippers(): {:?}ms", tstart.elapsed().as_millis());
    tstart = std::time::Instant::now();
    
    println!("{} zips", zip_of_zid.len());
    println!("arg_of_zid_node size: {}", arg_of_zid_node.len());

    // set up tracking if any
    let tracking: Option<Tracking> = cfg.track.as_ref().map(|s|{
        let expr: Expr = s.parse().unwrap();
        let zids_of_ivar = zids_of_ivar_of_expr(&expr, &zid_of_zip);
        let refined = cfg.track_refined.as_ref().map(|s| s.parse().unwrap());
        Tracking { expr, zids_of_ivar, refined }
    });

    println!("Tracking setup: {:?}ms", tstart.elapsed().as_millis());

    let mut stats: Stats = Default::default();

    tstart = std::time::Instant::now();

    // define all the important data structures for compression
    let mut donelist: Vec<FinishedPattern> = Default::default(); // completed inventions will go here    

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
            if !cfg.no_opt_single_task && tasks_of_node[&node].len() < 2 {
                if !cfg.no_stats { stats.single_task_fired += 1; };
                continue;
            }
            // Note that "single use" pruning is intentionally not done here,
            // since any invention specific to a node will by definition only
            // be useful at that node

            let match_locations = vec![*node];
            let body_utility_no_refinement = cost_of_node_once[node];
            let refinement_body_utility = 0;
            // compressive_utility for arity-0 is cost_of_node_all[node] minus the penalty of using the new prim
            let compressive_utility = cost_of_node_all[node] - num_paths_to_node[node] * COST_TERMINAL;
            let utility = compressive_utility + noncompressive_utility(body_utility_no_refinement + refinement_body_utility, cfg);
            if utility <= 0 { continue; }

            let pattern = Pattern {
                holes: vec![],
                arg_choices: vec![],
                first_zid_of_ivar: vec![],
                refinements: vec![],
                match_locations,
                utility_upper_bound: utility,
                body_utility_no_refinement,
                refinement_body_utility,
                tracked: false,
            };
            let finished_pattern = FinishedPattern {
                pattern,
                utility,
                compressive_utility,
                util_calc: UtilityCalculation { util: compressive_utility, corrected_utils: Default::default()},
                arity: 0,
                usages: num_paths_to_node[node]
            };
            donelist.push(finished_pattern);
        }
    }

    println!("arity 0: {:?}ms", tstart.elapsed().as_millis());
    tstart = std::time::Instant::now();

    println!("got {} arity zero inventions", donelist.len());

    let crit = CriticalMultithreadData::new(donelist, &treenodes, &cost_of_node_all, &num_paths_to_node, &egraph, &cfg);
    let shared = Arc::new(SharedData {
        crit: Mutex::new(crit),
        arg_of_zid_node,
        treenodes: treenodes.clone(),
        programs_node: train_programs_node,
        zids_of_node,
        zip_of_zid,
        zid_of_zip,
        extensions_of_zid,
        uses_of_shifted_arg_refinement,
        egraph,
        num_paths_to_node,
        tasks_of_node,
        cost_of_node_once,
        cost_of_node_all,
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
            println!("{} @ step=0 util={} for {}", "[new best utility]".blue(), best_util, best_expr);
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
    shared.crit.lock().deref_mut().update(&cfg);

    println!("{:?}", shared.stats.lock().deref_mut());
    assert!(shared.crit.lock().deref_mut().worklist.is_empty());

    let donelist: Vec<FinishedPattern> = shared.crit.lock().deref_mut().donelist.clone();

    if cfg.dreamcoder_comparison {
        println!("Timing point 1 (from the start of compression_step to final donelist): {:?}ms", tstart_total.elapsed().as_millis());
        println!("Timing Comparison Point A (search) (millis): {}", tstart_total.elapsed().as_millis());
        let tstart_rewrite = std::time::Instant::now();
        rewrite_fast(&donelist[0], roots, &shared, new_inv_name);
        println!("Timing point 2 (rewriting the candidate): {:?}ms", tstart_rewrite.elapsed().as_millis());
        println!("Timing Comparison Point B (search+rewrite) (millis): {}", tstart_total.elapsed().as_millis());
    }

    let orig_train_cost = shared.egraph[train_programs_node].data.inventionless_cost;
    let orig_test_cost = test_programs_node.map(|n|shared.egraph[n].data.inventionless_cost);

    let mut results: Vec<CompressionStepResult> = vec![];

    // construct CompressionStepResults and print some info about them)
    println!("Training cost before: {}", orig_train_cost);
    for (i,done) in donelist.iter().enumerate() {
        let res = CompressionStepResult::new(
            done.clone(),
            train_programs_node,
            test_programs_node,
            new_inv_name,
            &mut shared,
            past_invs,
            Some(orig_train_cost),
            orig_test_cost);

        println!("{}: {}", i, res);
        if cfg.show_rewritten {
            println!("rewritten (train):\n{}", &res.train_data.rewritten.split_programs().iter().map(|p|p.to_string()).collect::<Vec<_>>().join("\n"));
            println!("rewritten (test):\n{}",
                res.test_data.as_ref().map_or("null".to_string(), |d| d.rewritten.split_programs().iter().map(|p|p.to_string()).collect::<Vec<_>>().join("\n"))
            );
        }
        results.push(res);
    }
    println!("post stuff: {:?}ms", tstart.elapsed().as_millis());

    results
}