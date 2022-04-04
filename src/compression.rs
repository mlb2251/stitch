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

    /// pattern or invention to track
    #[clap(long)]
    pub follow_track: bool,

    /// print out each step of what gets popped off the worklist
    #[clap(long)]
    pub verbose_worklist: bool,

    /// whenever a new best thing is found, print it
    #[clap(long)]
    pub verbose_best: bool,

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

    /// disable the single usage pruning optimization
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

    /// Disable stat logging - note that stat logging in multithreading requires taking a mutex
    /// so it could be a source of slowdown in the multithreaded case, hence this flag to disable it.
    /// From some initial tests it seems to cause no slowdown anyways though.
    #[clap(long)]
    pub no_stats: bool,

    /// disables other_utility so the only utility is based on compressivity
    #[clap(long)]
    pub no_other_util: bool,
}

impl CompressionStepConfig {
    pub fn no_opt(&mut self) {
        self.no_opt_free_vars = true;
        self.no_opt_single_use = true;
        self.no_opt_single_task = true;
        self.no_opt_upper_bound = true;
        self.no_opt_force_multiuse = true;
        self.no_opt_useless_abstract = true;
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
        let body_utility = 0;
        let mut match_locations = treenodes.clone();
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
            .chain(self.arg_choices.iter().map(|labelled_zid| (shared.zip_of_zid[labelled_zid.zid].clone(),Expr::ivar(labelled_zid.ivar as i32)))).collect();


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
    pub id: Id,
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
    fn new(pattern: Pattern) -> Self {
        HeapItem {
            // key: pattern.body_utility * pattern.match_locations.len() as i32,
            key: pattern.utility_upper_bound,
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
    pub roots: Vec<Id>,
    pub zids_of_node: HashMap<Id,Vec<ZId>>,
    pub zip_of_zid: Vec<Zip>,
    pub zid_of_zip: HashMap<Zip, ZId>,
    pub extensions_of_zid: Vec<ZIdExtension>,
    pub descendants_of_node: HashMap<Id,Vec<Id>>,
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
}

impl CriticalMultithreadData {
    /// Create a new mutable multithread data struct with
    /// a worklist that just has a single hole on it
    fn new(donelist: Vec<FinishedPattern>, treenodes: &Vec<Id>, cost_of_node_all: &HashMap<Id,i32>, num_paths_to_node: &HashMap<Id,i32>, egraph: &crate::EGraph, cfg: &CompressionStepConfig) -> Self {
        // push an empty hole onto a new worklist
        let mut worklist = BinaryHeap::new();
        worklist.push(HeapItem::new(Pattern::single_hole(treenodes, cost_of_node_all, num_paths_to_node, egraph, cfg)));
        
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
        self.utility_pruning_cutoff = self.donelist.last().map(|x|x.utility).unwrap_or(0);
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
    single_task_fired: usize,
    single_use_fired: usize,
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
) -> Option<(Pattern,i32)> {

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
        println!("{} @ step={} util={} for {}", "[new best utility]".blue(), shared.stats.lock().deref_mut().worklist_steps, crit.donelist.first().unwrap().utility, crit.donelist.first().unwrap().to_expr(shared));
    }

    // pull out the newer version of this now that its been updated, since we're returning it at the end
    let mut utility_pruning_cutoff = crit.utility_pruning_cutoff;

    let old_worklist_len = crit.worklist.len();
    let worklist_buf_len = worklist_buf.len();
    // drain from worklist_buf into the actual worklist
    crit.worklist.extend(worklist_buf.drain(..).filter(|heap_item| heap_item.pattern.utility_upper_bound > utility_pruning_cutoff));
    // num pruned by upper bound = num we were gonna add minus change in worklist length
    if !shared.cfg.no_stats { shared.stats.lock().deref_mut().upper_bound_fired += worklist_buf_len - (crit.worklist.len() - old_worklist_len); };

    // try to get a new worklist item
    loop {
        while crit.worklist.is_empty() {
            crit.active_threads.remove(&thread::current().id());
            if crit.active_threads.is_empty() {
                return None
            }
            // the worklist is empty but someone else currently has a worklist item so we should give up our lock then take it back
            drop(shared_guard);
            shared_guard = shared.crit.lock();
            crit = shared_guard.deref_mut();
            // update our cutoff in case it changed
            utility_pruning_cutoff = crit.utility_pruning_cutoff;
        }
        
        let  heap_item = crit.worklist.pop().unwrap();
        // prune if upper bound is too low (cutoff may have increased in the time since this was added to the worklist)
        if shared.cfg.no_opt_upper_bound || heap_item.pattern.utility_upper_bound > utility_pruning_cutoff {
            // we got one!
            crit.active_threads.insert(thread::current().id());
            return Some((heap_item.pattern, utility_pruning_cutoff));
        }
        if !shared.cfg.no_stats { shared.stats.lock().deref_mut().upper_bound_fired += 1; };
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
        let (original_pattern, mut weak_utility_pruning_cutoff) =
            match get_worklist_item(
                &mut worklist_buf,
                &mut donelist_buf,
                &shared,
            ) {
                Some(pattern) => pattern,
                None => return,
        };

        if !shared.cfg.no_stats { shared.stats.lock().deref_mut().worklist_steps += 1; };

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
                    arg_of_loc[loc].id == 
                    arg_of_loc_ivar[loc].id).cloned().collect();
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
        'outer:
            for (expands_to, locs) in match_locations.into_iter()
            .group_by(|loc| arg_of_loc[loc].expands_to.clone()).into_iter()
            .map(|(expands_to, locs)| (expands_to, locs.collect::<Vec<Id>>()))
            .chain(ivars_expansions.into_iter())
        {
            // for debugging
            let tracked = original_pattern.tracked && expands_to == tracked_expands_to(&original_pattern, hole_zid, &shared);
            if tracked { found_tracked = true; }
            if shared.cfg.follow_track && !tracked { continue; }

            // check for arity 0 inventions; these were previously handled and can be skipped
            if holes_after_pop.is_empty() && original_pattern.arg_choices.is_empty() && !expands_to.has_holes() && !expands_to.is_ivar() {
                continue; 
            }

            // check for inventions that are only useful at a single node. And invention that is arity>0 but is only useful at a single
            // structurally hashed node must not do any actual useful abstraction so we can discard it
            if !shared.cfg.no_opt_single_use && locs.len() < 2 {
                if !shared.cfg.no_stats { shared.stats.lock().deref_mut().single_use_fired += 1; };
                if tracked { println!("{} single use pruned when expanding {} to {}", "[TRACK]".red().bold(), original_pattern.to_expr(&shared), original_pattern.show_track_expansion(hole_zid, &shared)); }
                continue; // too few uses
            }

            // prune inventions specific to one single task
            if !shared.cfg.no_opt_single_task
                    && locs.iter().all(|node| shared.tasks_of_node[&node].len() == 1)
                    && locs.iter().all(|node| shared.tasks_of_node[&locs[0]].iter().next() == shared.tasks_of_node[&node].iter().next()) {
                if !shared.cfg.no_stats { shared.stats.lock().deref_mut().single_task_fired += 1; }
                if tracked { println!("{} single task pruned when expanding {} to {}", "[TRACK]".red().bold(), original_pattern.to_expr(&shared), original_pattern.to_expr(&shared).zipper_replace(&shared.zip_of_zid[hole_zid], &format!("<{}>",expands_to))); }
                continue;
            }

            // check for free variables: if an invention has free variables in the body then it's not a real function and we can discard it
            // Here we just check if our expansion just yielded a variable, and if that is bound based on how many lambdas there are above it.
            if !shared.cfg.no_opt_free_vars {
                if let ExpandsTo::Var(i) = expands_to {
                    if i >= shared.zip_of_zid[hole_zid].iter().filter(|znode|**znode == ZNode::Body).count() as i32 {
                        if !shared.cfg.no_stats { shared.stats.lock().deref_mut().free_vars_fired += 1; };
                        if tracked { println!("{} pruned by free var in body when expanding {} to {}", "[TRACK]".red().bold(), original_pattern.to_expr(&shared), original_pattern.show_track_expansion(hole_zid, &shared)); }
                        continue; // free var
                    }
                }
            }

            // check for useless abstractions (ie ones that take the same arg everywhere). We check for this all the time, not just when adding a new variables,
            // because subsetting of match_locations can turn previously useful abstractions into useless ones.
            if !shared.cfg.no_opt_useless_abstract &&
                original_pattern.arg_choices.iter()
                .any(|argchoice| locs.iter()
                    .map(|loc| shared.arg_of_zid_node[argchoice.zid][loc].id.clone()).all_equal())
            {
                if !shared.cfg.no_stats { shared.stats.lock().deref_mut().useless_abstract_fired += 1; };
                continue; // useless abstraction
            }

            // update the body utility
            let body_utility = original_pattern.body_utility +  match expands_to {
                ExpandsTo::Lam | ExpandsTo::App => COST_NONTERMINAL,
                ExpandsTo::Var(_) | ExpandsTo::Prim(_) => COST_TERMINAL,
                ExpandsTo::IVar(_) => 0,
            };

            // update the upper bound
            let utility_upper_bound: i32 = utility_upper_bound(&locs, body_utility, &shared.cost_of_node_all, &shared.num_paths_to_node, &shared.cfg);

            assert!(utility_upper_bound <= original_pattern.utility_upper_bound);

            // prune if utility upper bound is negative
            if utility_upper_bound <= 0 {
                if tracked { println!("{} <= 0 utility pruned ({}) when expanding {} to {}", "[TRACK]".red().bold(), utility_upper_bound, original_pattern.to_expr(&shared), original_pattern.show_track_expansion(hole_zid, &shared)); }
                continue; // too low utility
            }

            // branch and bound: if the upper bound is less than the best invention we've found so far (our cutoff), we can discard this pattern
            if !shared.cfg.no_opt_upper_bound && utility_upper_bound <= weak_utility_pruning_cutoff {
                if !shared.cfg.no_stats { shared.stats.lock().deref_mut().upper_bound_fired += 1; };
                if tracked { println!("{} upper bound ({} < {}) pruned when expanding {} to {}", "[TRACK]".red().bold(), utility_upper_bound, weak_utility_pruning_cutoff, original_pattern.to_expr(&shared), original_pattern.show_track_expansion(hole_zid, &shared)); }
                continue; // too low utility
            }

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
                    let ref arg_of_loc_1 = shared.arg_of_zid_node[*ivar_zid_1];
                    for ivar_zid_2 in first_zid_of_ivar.iter().skip(i+1) {
                        let ref arg_of_loc_2 = shared.arg_of_zid_node[*ivar_zid_2];
                        if locs.iter().all(|loc|
                            arg_of_loc_1[loc].id == arg_of_loc_2[loc].id)
                        {
                            if !shared.cfg.no_stats { shared.stats.lock().deref_mut().force_multiuse_fired += 1; };
                            if tracked { println!("{} force multiuse pruned when expanding {} to {}", "[TRACK]".red().bold(), original_pattern.to_expr(&shared), original_pattern.show_track_expansion(hole_zid, &shared)); }
                            continue 'outer;
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
                utility_upper_bound,
                body_utility,
                tracked
            };


            if new_pattern.holes.is_empty() {
                // it's a finished pattern

                // check if free vars (including negatives) in args
                // todo change this to be negative ivars + add refinement
                for zid in new_pattern.first_zid_of_ivar.iter() {
                    let ref arg_of_node = shared.arg_of_zid_node[*zid];
                    for loc in new_pattern.match_locations.iter() {
                        if shared.egraph[arg_of_node[loc].id].data.free_vars.iter().any(|var| *var < 0) {
                            if tracked { println!("{} discarding finished_pattern because one of its args has negative vars: {}", "[TRACK]".red().bold(), new_pattern.to_expr(&shared)); }
                            continue 'outer;
                        }
                    }

                }

                let finished_pattern = FinishedPattern::new(new_pattern, &shared);
                if tracked {
                    println!("{} pushed {} to donelist (util: {})", "[TRACK:DONE]".green().bold(), finished_pattern.to_expr(&shared), finished_pattern.utility);
                }
                if shared.cfg.inv_candidates == 1 {
                    // if we're only looking for one invention, we can directly update our cutoff here
                    weak_utility_pruning_cutoff = finished_pattern.utility;
                }

                donelist_buf.push(finished_pattern);

            } else {
                // it's a partial pattern so just add it to the worklist
                if tracked { println!("{} pushed {} to worklist (bound: {})", "[TRACK]".green().bold(), original_pattern.show_track_expansion(hole_zid, &shared), new_pattern.utility_upper_bound); }
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
        let noncompressive_utility = noncompressive_utility(pattern.body_utility, &shared.cfg);
        let utility = noncompressive_utility + compressive_utility.util;
        assert!(utility <= pattern.utility_upper_bound);
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

}

/// figure out all the N^2 zippers from choosing any given node and then choosing a descendant and returning the zipper from
/// the node to the descendant. We also collect a bunch of other useful stuff like the argument you would get if you abstracted
/// the descendant and introduced an invention rooted at the ancestor node.
fn get_zippers(
    treenodes: &[Id],
    cost_of_node_once: &HashMap<Id,i32>,
    no_cache: bool,
    egraph: &mut crate::EGraph,
    _cfg: &CompressionStepConfig
) -> (HashMap<Zip, ZId>, Vec<Zip>, Vec<HashMap<Id,Arg>>, HashMap<Id,Vec<ZId>>,  Vec<ZIdExtension>, HashMap<Id,Vec<Id>>) {
    let cache: &mut Option<RecVarModCache> = &mut if no_cache { None } else { Some(HashMap::new()) };

    let mut zid_of_zip: HashMap<Zip, ZId> = Default::default();
    let mut zip_of_zid: Vec<Zip> = Default::default();
    let mut arg_of_zid_node: Vec<HashMap<Id,Arg>> = Default::default();
    let mut zids_of_node: HashMap<Id,Vec<ZId>> = Default::default();
    let mut descendants_of_node: HashMap<Id,Vec<Id>> = Default::default();

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
            Arg { id: *treenode, unshifted_id: *treenode, shift: 0, cost: cost_of_node_once[treenode], expands_to: expands_to_of_node(&node) });
        
        descendants_of_node.insert(*treenode, vec![]);

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
                let mut descendants: Vec<Id> = descendants_of_node[&f].iter().chain(descendants_of_node[&x].iter()).cloned().collect();
                descendants.push(x);
                descendants.push(f);
                descendants_of_node.get_mut(&treenode).unwrap().extend(descendants.into_iter());
            },
            Lambda::Lam([b]) => {
                for b_zid in zids_of_node[&b].iter() {
                    // todo add negative ivars here

                    // clone and extend zip to get new zid for this node
                    let mut zip = zip_of_zid[*b_zid].clone();
                    zip.insert(0,ZNode::Body);
                    let zid = zid_of_zip.entry(zip.clone()).or_insert_with(|| {
                        let zid = zip_of_zid.len();
                        zip_of_zid.push(zip);
                        arg_of_zid_node.push(HashMap::new());
                        zid
                    });
                    // add new zid to this node
                    zids.push(*zid);
                    // shift the arg but keep the unshifted part the sam
                    let mut arg: Arg = arg_of_zid_node[*b_zid][&b].clone();
                    arg.id = shift(arg.id, -1, egraph, cache).unwrap();
                    arg.shift -= 1;
                    arg_of_zid_node[*zid].insert(*treenode, arg);
                }
                let mut descendants: Vec<Id> = descendants_of_node[&b].clone();
                descendants.push(b);
                descendants_of_node.get_mut(&treenode).unwrap().extend(descendants.into_iter());
            },
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
    descendants_of_node)
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
    fn new(done: FinishedPattern, programs_node: Id, inv_name: &str, shared: &mut SharedData, past_invs: &Vec<CompressionStepResult>) -> Self {
        let initial_cost = shared.egraph[programs_node].data.inventionless_cost;

        // cost of the very first initial program before any inventions
        let very_first_cost = if let Some(past_inv) = past_invs.first() { past_inv.initial_cost } else { initial_cost };

        let inv = done.to_invention(inv_name, shared);
        let rewritten = Expr::programs(rewrite_fast(&done, &shared, &inv.name));


        let expected_cost = initial_cost - done.compressive_utility;
        let final_cost = rewritten.cost();
        if expected_cost != final_cost {
            println!("*** expected cost {} != final cost {}", expected_cost, final_cost);
        }
        let multiplier = initial_cost as f64 / final_cost as f64;
        let multiplier_wrt_orig = very_first_cost as f64 / final_cost as f64;
        let uses = done.usages;
        let use_exprs: Vec<Expr> = done.pattern.match_locations.iter().map(|node| extract(*node, &shared.egraph)).collect();
        let use_args: Vec<Vec<Expr>> = done.pattern.match_locations.iter().map(|node|
            done.pattern.first_zid_of_ivar.iter().map(|zid|
                extract(shared.arg_of_zid_node[*zid][node].id, &shared.egraph)
            ).collect()).collect();
        
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
            res = replace_prim_with(&res, &inv_name, &dc_inv_str);
            // res = res.replace(&format!("{})",inv_name), &format!("{})",dc_inv_str));
            // res = res.replace(&format!("{} ",inv_name), &format!("{} ",dc_inv_str));
            res
        }).collect();

        CompressionStepResult { inv, rewritten, rewritten_dreamcoder, done, expected_cost, final_cost, multiplier, multiplier_wrt_orig, uses, use_exprs, use_args, dc_inv_str, initial_cost }
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
#[inline]
fn utility_upper_bound(
    match_locations: &Vec<Id>,
    body_utility: i32,
    cost_of_node_all: &HashMap<Id,i32>,
    num_paths_to_node: &HashMap<Id,i32>,
    cfg: &CompressionStepConfig,
) -> i32 {
    compressive_utility_upper_bound(match_locations, cost_of_node_all, num_paths_to_node)
        + noncompressive_utility_upper_bound(body_utility, cfg)
}

/// This utility is just for any utility terms that we care about that don't directly correspond
/// to changes in size that come from rewriting with an invention
fn noncompressive_utility(
    body_utility: i32,
    cfg: &CompressionStepConfig,
) -> i32 {
    if cfg.no_other_util { return 0; }
    // this is a bit like the structure penalty from dreamcoder except that
    // that penalty uses inlined versions of nested inventions.
    let structure_penalty = - (body_utility * 3 / 2);
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
    body_utility: i32,
    cfg: &CompressionStepConfig,
) -> i32 {
    if cfg.no_other_util { return 0; }
    // safe bound: since structure_penalty is negative an upper bound is anything less negative or exact. Since
    // left_utility < body_utility we know that this will be a less negative bound.
    let structure_penalty = - (body_utility * 3 / 2);
    structure_penalty
}

fn compressive_utility(pattern: &Pattern, shared: &SharedData) -> UtilityCalculation {

    // * BASIC CALCULATION:

    // get a list of (ivar,usages-1) filtering out things that are only used once
    let ivar_multiuses: Vec<(usize,i32)> = pattern.arg_choices.iter().map(|labelled|labelled.ivar).counts()
        .iter().filter_map(|(ivar,count)| if *count > 1 { Some((*ivar, (*count-1) as i32)) } else { None }).collect();

    // it costs a tiny bit to apply the invention, for example (app (app inv0 x) y) incurs a cost
    // of COST_TERMINAL for the `inv0` primitive and 2 * COST_NONTERMINAL for the two `app`s.
    let app_penalty = - (COST_TERMINAL + COST_NONTERMINAL * pattern.first_zid_of_ivar.len() as i32);


    let utility_of_loc_once: Vec<i32> = pattern.match_locations.iter().map(|loc| {
        // compressivity of body minus slight penalty from the application
        let base_utility = pattern.body_utility + app_penalty;
        // for each extra usage of an argument, we gain the cost of that argument as
        // extra utility. Note we use `first_zid_of_ivar` since it doesn't matter which
        // of the zids we use as long as it corresponds to the right ivar
        let multiuse_utility = ivar_multiuses.iter().map(|(ivar,count)|
            count * shared.arg_of_zid_node[pattern.first_zid_of_ivar[*ivar]][loc].cost
        ).sum::<i32>();
        // multiply all this utility by the number of times this node shows up
        base_utility + multiuse_utility
        }).collect();


    let compressive_utility: i32 = pattern.match_locations.iter()
        .zip(utility_of_loc_once.iter())
        .map(|(loc,utility)| utility * shared.num_paths_to_node[loc])
        .sum();

    // * ACCOUNTING FOR USE CONFLICTS:

    // todo opt include holes too
    let zips: Vec<Zip> = pattern.arg_choices.iter().map(|labelled_zid| shared.zip_of_zid[labelled_zid.zid].clone()).collect();
    
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
        for zip in zips.iter().filter(|zip| !zip.is_empty()) {
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
        }

        // common case: no conflcits
        if conflict_idxs.is_empty() { continue; }

        // now we basically record how much we would affect global utility by if we accept vs reject vs choose the best of those options.
        // and recording this will let us change our mind later if we decide to force-reject something

        // if we reject using the invention at this node, we just lose its utility
        let reject = - utility_of_loc_once[loc_idx];
        
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
    programs_expr: &Expr,
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

    let mut rewritten: Expr = programs_expr.clone();
    let mut step_results: Vec<CompressionStepResult> = Default::default();

    let tstart = std::time::Instant::now();

    for i in 0..iterations {
        println!("{}",format!("\n=======Iteration {}=======",i).blue().bold());
        let inv_name = format!("fn_{}", num_prior_inventions + step_results.len());

        // call actual compression
        let res: Vec<CompressionStepResult> = compression_step(
            &rewritten,
            &inv_name,
            &cfg,
            &step_results,
            tasks);

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

    if cfg.dreamcoder_drop_last {
        println!("{}",format!("{}","[--dreamcoder-drop-last] dropping final invention".yellow().bold()));
        step_results.pop();
    }

    println!("{}","\n=======Compression Summary=======".blue().bold());
    println!("Found {} inventions", step_results.len());
    println!("Cost Improvement: ({:.2}x better) {} -> {}", compression_factor(programs_expr,&rewritten), programs_expr.cost(), rewritten.cost());
    for i in 0..step_results.len() {
        let res = &step_results[i];
        println!("{} ({:.2}x wrt orig): {}" ,res.inv.name.clone().blue(), compression_factor(programs_expr, &res.rewritten), res);
    }
    println!("Time: {}ms", tstart.elapsed().as_millis());
    step_results
}

/// Takes a set of programs as an Expr with Programs as its root, and does one full step of compresison.
/// Returns the top Inventions and the Expr rewritten under that invention along with other useful info in CompressionStepResult
/// The number of inventions returned is based on cfg.inv_candidates
pub fn compression_step(
    programs_expr: &Expr,
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
    let programs_node = egraph.add_expr(programs_expr.into());
    egraph.rebuild();

    println!("set up egraph: {:?}ms", tstart.elapsed().as_millis());
    tstart = std::time::Instant::now();

    let roots: Vec<Id> = egraph[programs_node].nodes[0].children().iter().cloned().collect();

    // all nodes in child-first order except for the Programs node
    let mut treenodes: Vec<Id> = topological_ordering(programs_node,&egraph);
    treenodes.retain(|id| *id != programs_node);

    println!("got roots and treenodes: {:?}ms", tstart.elapsed().as_millis());
    tstart = std::time::Instant::now();

    // populate num_paths_to_node so we know how many different parts of the programs tree
    // a node participates in (ie multiple uses within a single program or among programs)
    let num_paths_to_node: HashMap<Id,i32> = num_paths_to_node(&roots, &treenodes, &egraph);

    println!("num_paths_to_node(): {:?}ms", tstart.elapsed().as_millis());
    tstart = std::time::Instant::now();

    let tasks_of_node: HashMap<Id, HashSet<usize>> = associate_tasks(programs_node, &egraph, tasks);

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
        descendants_of_node) = get_zippers(&treenodes, &cost_of_node_once, cfg.no_cache, &mut egraph, cfg);
    
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

    // arity 0 inventions
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
        let body_utility = cost_of_node_once[node];
        // compressive_utility for arity-0 is cost_of_node_all[node] minus the penalty of using the new prim
        let compressive_utility = cost_of_node_all[node] - num_paths_to_node[node] * COST_TERMINAL;
        let utility = compressive_utility + noncompressive_utility(body_utility, cfg);
        if utility <= 0 { continue; }

        let pattern = Pattern {
            holes: vec![],
            arg_choices: vec![],
            first_zid_of_ivar: vec![],
            match_locations,
            utility_upper_bound: utility,
            body_utility,
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

    println!("arity 0: {:?}ms", tstart.elapsed().as_millis());
    tstart = std::time::Instant::now();

    println!("got {} arity zero inventions", donelist.len());

    let crit = CriticalMultithreadData::new(donelist, &treenodes, &cost_of_node_all, &num_paths_to_node, &egraph, &cfg);
    let shared = Arc::new(SharedData {
        crit: Mutex::new(crit),
        arg_of_zid_node,
        treenodes: treenodes.clone(),
        programs_node,
        roots,
        zids_of_node,
        zip_of_zid,
        zid_of_zip,
        extensions_of_zid,
        descendants_of_node,
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
        let best_util = crit.deref_mut().donelist.first().unwrap().utility;
        let best_expr: Expr = crit.deref_mut().donelist.first().unwrap().to_expr(&shared);
        println!("{} @ step=0 util={} for {}", "[new best utility]".blue(), best_util, best_expr);
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

    let orig_cost = shared.egraph[programs_node].data.inventionless_cost;

    let mut results: Vec<CompressionStepResult> = vec![];

    // construct CompressionStepResults and print some info about them)
    println!("Cost before: {}", orig_cost);
    for (i,done) in donelist.iter().enumerate() {
        let res = CompressionStepResult::new(done.clone(), programs_node, new_inv_name, &mut shared, past_invs);

        println!("{}: {}", i, res);
        if cfg.show_rewritten {
            println!("rewritten:\n{}", res.rewritten.split_programs().iter().map(|p|p.to_string()).collect::<Vec<_>>().join("\n"));
        }
        results.push(res);
    }
    println!("post stuff: {:?}ms", tstart.elapsed().as_millis());

    results
}