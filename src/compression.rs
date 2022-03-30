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
    /// unless --lossy-candidates is enabled.
    #[clap(short='n', long, default_value = "1")]
    pub inv_candidates: usize,

    /// pattern or invention to track
    #[clap(long, arg_enum, default_value = "min-cost")]
    pub hole_choice: HoleChoice,

    /// inventions cant start with a Lambda
    #[clap(long)]
    pub no_top_lambda: bool,

    /// pattern or invention to track
    #[clap(long)]
    pub track: Option<String>,

    /// print out each step of what gets popped off the worklist
    #[clap(long)]
    pub verbose_worklist: bool,

    /// whenever a new best thing is found, print it
    #[clap(long)]
    pub verbose_best: bool,

    /// 
    #[clap(long)]
    pub break_early_assignment: bool,

    /// Turning this on means that only the top invention will be guaranteed to be the best invention,
    /// and the 2nd best invention may not be the actual second best invention. Basically, this just enables
    /// pruning of everything that's worse than the best invention which could cause speedups depending on the domain.
    #[clap(long)]
    pub lossy_candidates: bool,

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
/// `pruned_assignment_prefixes` is an implementation detail for optimizing variable assignments
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Pattern {
    holes: Vec<ZId>, // in order of when theyre added NOT left to right
    arg_choices: Vec<ZId>, // a hole gets moved into here when it becomes an argchoice, again these are in order of when they were added
    match_locations: Vec<Id>, // places where it applies
    pruned_assignment_prefixes: Vec<Vec<i32>>,
    utility_upper_bound: i32,
    body_utility: i32, // the size (in `cost`) of a single use of the pattern body so far
    tracked: bool, // for debugging
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

impl Pattern {
    /// create a single hole pattern `??`
    fn single_hole(treenodes: &Vec<Id>, cost_of_node_all: &HashMap<Id,i32>, num_paths_to_node: &HashMap<Id,i32>, egraph: &EGraph, cfg: &CompressionStepConfig) -> Self {
        let body_utility = 0;
        let mut match_locations = treenodes.clone();
        if cfg.no_top_lambda {
            match_locations.retain(|node| node_type_of_node(&egraph[*node].nodes[0]).unwrap() != NodeType::Lam);
        }
        let utility_upper_bound = utility_upper_bound(&match_locations, body_utility, cost_of_node_all, num_paths_to_node, cfg);
        Pattern {
            holes: vec![EMPTY_ZID], // (zid 0 is the empty zipper)
            arg_choices: vec![],
            match_locations, // single hole matches everywhere
            pruned_assignment_prefixes: vec![],
            utility_upper_bound,
            body_utility, // 0 body utility
            tracked: cfg.track.is_some(),
        }
    }
    /// convert pattern to an Expr with `??` in place of holes and `?#` in place of argchoices
    fn to_expr(&self, shared: &SharedData) -> Expr {
        let mut curr_zip: Zip = vec![];
        // map zids to zips with a bool thats true if this is a hole and false if its a future ivar
        let zips: Vec<(Zip,bool)> = self.holes.iter().map(|zid| (shared.zip_of_zid[*zid].clone(), true)).
            chain(self.arg_choices.iter().map(|zid| (shared.zip_of_zid[*zid].clone(), false))).collect();


        fn helper(curr_node: Id, curr_zip: &mut Zip, zips: &Vec<(Zip,bool)>, shared: &SharedData) -> Expr {
            match zips.iter().find(|(zip,_)| zip == curr_zip) {
                // current zip matches a hole
                Some((_,true)) => Expr::prim("??".into()),
                // current zip matches an ivar
                Some((_,false)) => Expr::prim("?#".into()),
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
}

/// The child-ignoring value of a node in the original set of programs. This tells us
/// what the hole will expand into at this node.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd)]
enum NodeType {
    Lam,
    App,
    Var(i32),
    Prim(Symbol)
}

impl NodeType {
    #[inline]
    /// true if expanding a node of this nodetype will yield new holes
    fn has_holes(&self) -> bool {
        match self {
            NodeType::Lam => true,
            NodeType::App => true,
            NodeType::Var(_) => false,
            NodeType::Prim(_) => false,
        }
    }
}

impl std::fmt::Display for NodeType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            NodeType::Lam => write!(f, "lam"),
            NodeType::App => write!(f, "app"),
            NodeType::Var(v) => write!(f, "${}", v),
            NodeType::Prim(p) => write!(f, "{}", p),
        }
    }
}

/// a list of znodes, representing a path through a tree (a zipper)
type Zip = Vec<ZNode>;
/// the index of the empty zid `[]` in the list of zippers
const EMPTY_ZID: ZId = 0;

/// an argument to an abstraction. `id` is the main field here, we can use
/// it to lookup the corresponding tree using egraph[id]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Arg {
    id: Id,
    unshifted_id: Id, // in case `id` was shifted to make it an arg not sure if this will end up being useful
    cost: i32,
    node_type: NodeType,
}

impl Arg {
    fn new(id: Id, unshifted_id: Id, cost: i32, node_type: NodeType) -> Self {
        Arg {
            id,
            unshifted_id,
            cost,
            node_type
        }
    }
}

/// NodeType from a &Lambda node. Returns None if this is
/// and IVar (which is not considered a node type) and crashes
/// on Programs node.
fn node_type_of_node(node: &Lambda) -> Option<NodeType> {
    match node {
        Lambda::Var(i) => Some(NodeType::Var(*i)),
        Lambda::Prim(p) => {
            if *p == Symbol::from("?#") { None } else {
                Some(NodeType::Prim(*p))
            }
        },
        Lambda::Lam(_) => Some(NodeType::Lam),
        Lambda::App(_) => Some(NodeType::App),
        Lambda::IVar(_) => None,
        _ => unreachable!()
    }
}

/// Returns Some(nodetype) for what we expect the hole to expand to to follow
/// the target, and returns None if we expect it to become a ?# argchoice.
fn tracked_node_type(hole_zid: ZId, shared: &SharedData) -> Option<NodeType> {
    let id = shared.tracking.as_ref().unwrap().expr
        .apply_zipper(&shared.zip_of_zid[hole_zid]).unwrap();
    node_type_of_node(shared.tracking.as_ref().unwrap().expr.get(id))
}

/// The heap item used for heap-based worklists. Holds a pattern
#[derive(Debug,Clone, Eq, PartialEq)]
struct HeapItem {
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
struct CriticalMultithreadData {
    donelist: Vec<FinishedPattern>,
    worklist: BinaryHeap<HeapItem>,
    lowest_donelist_utility:i32,
    utility_pruning_cutoff: i32,
    active_threads: HashSet<std::thread::ThreadId>, // list of threads currently holding worklist items
}

/// All the data shared among threads, mostly read-only
/// except for the mutexes
#[derive(Debug)]
struct SharedData {
    crit: Mutex<CriticalMultithreadData>,
    arg_of_zid_node: Vec<HashMap<Id,Arg>>,
    #[allow(dead_code)]
    treenodes: Vec<Id>,
    #[allow(dead_code)]
    zids_of_node: HashMap<Id,Vec<ZId>>,
    zip_of_zid: Vec<Zip>,
    extensions_of_zid: Vec<ZIdExtension>,
    descendants_of_node: HashMap<Id,Vec<Id>>,
    egraph: EGraph,
    num_paths_to_node: HashMap<Id,i32>,
    #[allow(dead_code)]
    tasks_of_node: HashMap<Id, HashSet<usize>>,
    #[allow(dead_code)]
    cost_of_node_once: HashMap<Id,i32>,
    cost_of_node_all: HashMap<Id,i32>,
    stats: Mutex<Stats>,
    cfg: CompressionStepConfig,
    tracking: Option<Tracking>,
}

/// Used for debugging tracking information
#[derive(Debug)]
struct Tracking {
    expr: Expr,
}

impl CriticalMultithreadData {
    /// Create a new mutable multithread data struct with
    /// a worklist that just has a single hole on it
    fn new(donelist: Vec<FinishedPattern>, treenodes: &Vec<Id>, cost_of_node_all: &HashMap<Id,i32>, num_paths_to_node: &HashMap<Id,i32>, egraph: &EGraph, cfg: &CompressionStepConfig) -> Self {
        // push an empty hole onto a new worklist
        let mut worklist = BinaryHeap::new();
        worklist.push(HeapItem::new(Pattern::single_hole(treenodes, cost_of_node_all, num_paths_to_node, egraph, cfg)));
        
        let mut res = CriticalMultithreadData {
            donelist,
            worklist,
            lowest_donelist_utility: 0,
            utility_pruning_cutoff: 0,
            active_threads: HashSet::new(),
        };
        res.update(cfg);
        res
    }
    /// sort the donelist by utility, truncate to cfg.inv_candidates, update the lowest_donelist_utility to be the lowest utility,
    /// update utility_pruning_cutoff to be the highest utility if --lossy-candidates is set else the lowest utility
    fn update(&mut self, cfg: &CompressionStepConfig) {
        // sort in decreasing order by utility primarily, and break ties using the zids (just in order to be deterministic!)
        // let old_best = self.donelist.first().map(|x|x.utility).unwrap_or(0);
        self.donelist.sort_unstable_by(|a,b| (b.utility,&b.labelled_zids).cmp(&(a.utility,&a.labelled_zids)));
        self.donelist.truncate(cfg.inv_candidates);
        self.lowest_donelist_utility = self.donelist.last().map(|x|x.utility).unwrap_or(0);
        self.utility_pruning_cutoff = if cfg.lossy_candidates { self.donelist.first().map(|x|x.utility).unwrap_or(0) } else { self.donelist.last().map(|x|x.utility).unwrap_or(0) };
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
type ZId = usize;

/// a zid referencing a specific ZPath and a #i index
#[derive(Debug,Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
struct LabelledZId {
    zid: ZId,
    ivar: usize // which #i argument this is, which also corresponds to args[i] ofc
}

/// Various tracking stats
#[derive(Clone,Default, Debug)]
struct Stats {
    partial_invs: usize,
    finished_invs: usize,
    upper_bound_fired: usize,
    free_vars_wip_fired: usize,
    single_use_done_fired: usize,
    single_task_done_fired: usize,
    single_use_wip_fired: usize,
    useless_abstract_fired: usize,
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
                    (hole_idx, pattern.match_locations.iter().filter(|loc|shared.arg_of_zid_node[*hole_zid][loc].node_type == NodeType::App).count()))
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
                        .map(|loc| shared.arg_of_zid_node[*hole_zid][loc].node_type.clone()).counts().values().max().unwrap())).max_by_key(|&(_,max_count)| max_count).unwrap().0
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
struct ZIdExtension {
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
) -> Option<(Pattern,i32,i32)> {

    // * MULTITHREADING: CRITICAL SECTION START *
    // take the lock, which will be released immediately when this scope exits
    let mut shared_guard = shared.crit.lock();
    let mut crit: &mut CriticalMultithreadData = shared_guard.deref_mut();
    let lowest_donelist_utility = crit.lowest_donelist_utility;
    let old_best_utility = crit.donelist.first().map(|x|x.utility).unwrap_or(0);
    let old_donelist_len = crit.donelist.len();
    // drain from donelist_buf into the actual donelist
    crit.donelist.extend(donelist_buf.drain(..).filter(|done| done.utility > lowest_donelist_utility));
    if !shared.cfg.no_stats { shared.stats.lock().deref_mut().finished_invs += crit.donelist.len() - old_donelist_len; };
    // sort + truncate + update utility_pruning_cutoff and lowest_donelist_utility
    crit.update(&shared.cfg); // this also updates utility_pruning_cutoff

    if shared.cfg.verbose_best && crit.donelist.first().map(|x|x.utility).unwrap_or(0) > old_best_utility {
        println!("{} @ step={} util={} for {}", "[new best utility]".blue(), shared.stats.lock().deref_mut().partial_invs, crit.donelist.first().unwrap().utility, crit.donelist.first().unwrap().to_expr(shared));
    }

    // pull out the newer versions of these now that theyve been updated, since we're returning them at the end
    let mut lowest_donelist_utility = crit.lowest_donelist_utility;
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
            // update our cutoffs in case they changed
            lowest_donelist_utility = crit.lowest_donelist_utility;
            utility_pruning_cutoff = crit.utility_pruning_cutoff;
        }
        
        let  heap_item = crit.worklist.pop().unwrap();
        // prune if upper bound is too low (cutoff may have increased in the time since this was added to the worklist)
        if shared.cfg.no_opt_upper_bound || heap_item.pattern.utility_upper_bound > utility_pruning_cutoff {
            // we got one!
            crit.active_threads.insert(thread::current().id());
            return Some((heap_item.pattern, utility_pruning_cutoff, lowest_donelist_utility));
        }
        if !shared.cfg.no_stats { shared.stats.lock().deref_mut().upper_bound_fired += 1; };
    }
    // * MULTITHREADING: CRITICAL SECTION END *
}


fn stitch_search(
    shared: Arc<SharedData>,
) {
    
    // local buffers to eventually pour into the global worklist and donelist when we take the mutex
    let mut worklist_buf: Vec<HeapItem> = Default::default();
    let mut donelist_buf: Vec<_> = Default::default();

    loop {

        // get a new worklist item along with pruning cutoffs
        let (original_pattern, mut weak_utility_pruning_cutoff, mut weak_lowest_donelist_utility) =
            match get_worklist_item(
                &mut worklist_buf,
                &mut donelist_buf,
                &shared,
            ) {
                Some(pattern) => pattern,
                None => return,
        };

        if !shared.cfg.no_stats { shared.stats.lock().deref_mut().partial_invs += 1; };

        if shared.cfg.verbose_worklist {
            println!("[prio={}; uses={}] chose: {}", original_pattern.match_locations.len() as i32 * original_pattern.body_utility, original_pattern.match_locations.len(), original_pattern.to_expr(&shared));
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
        let mut match_locations = original_pattern.match_locations.clone();
        match_locations.sort_unstable_by_key(|loc| arg_of_loc[loc].node_type.clone());

        // for each way of expanding the hole...
        for (node_type, locs) in match_locations.into_iter()
            .group_by(|loc| arg_of_loc[loc].node_type.clone()).into_iter()
        {
            // for debugging
            let tracked = original_pattern.tracked && Some(node_type.clone()) == tracked_node_type(hole_zid, &shared);

            // check for arity 0 inventions; these were previously handled and can be skipped
            if holes_after_pop.is_empty() && original_pattern.arg_choices.is_empty() && !node_type.has_holes() {
                continue; 
            }

            let locs: Vec<Id> = locs.collect();

            // check for inventions that are only useful at a single node. And invention that is arity>0 but is only useful at a single
            // structurally hashed node must not do any actual useful abstraction so we can discard it
            if !shared.cfg.no_opt_single_use && locs.len() < 2 {
                if !shared.cfg.no_stats { shared.stats.lock().deref_mut().single_use_wip_fired += 1; };
                if tracked { println!("{} single use pruned when expanding {} to {}", "[TRACK]".red().bold(), original_pattern.to_expr(&shared), original_pattern.to_expr(&shared).zipper_replace(&shared.zip_of_zid[hole_zid], &format!("<{}>",node_type))); }
                continue; // too few uses
            }

            // check for free variables: if an invention has free variables in the body then it's not a real function and we can discard it
            // Here we just check if our expansion just yielded a variable, and if that is bound based on how many lambdas there are above it.
            if !shared.cfg.no_opt_free_vars {
                if let NodeType::Var(i) = node_type {
                    if i >= shared.zip_of_zid[hole_zid].iter().filter(|znode|**znode == ZNode::Body).count() as i32 {
                        if !shared.cfg.no_stats { shared.stats.lock().deref_mut().free_vars_wip_fired += 1; };
                        if tracked { println!("{} pruned by free var in body when expanding {} to {}", "[TRACK]".red().bold(), original_pattern.to_expr(&shared), original_pattern.to_expr(&shared).zipper_replace(&shared.zip_of_zid[hole_zid], &format!("<{}>",node_type))); }
                        continue; // free var
                    }
                }
            }

            // check for useless abstractions (ie ones that take the same arg everywhere). We check for this all the time, not just when adding a new variables,
            // because subsetting of match_locations can turn previously useful abstractions into useless ones.
            if original_pattern.arg_choices.iter()
            .any(|argchoice_zid| locs.iter()
                .map(|loc| shared.arg_of_zid_node[*argchoice_zid][loc].id.clone()).all_equal())
            {
                if !shared.cfg.no_stats { shared.stats.lock().deref_mut().useless_abstract_fired += 1; };
                continue; // useless abstraction
            }

            // update the body utility
            let body_utility = original_pattern.body_utility +  match node_type {
                NodeType::Lam | NodeType::App => COST_NONTERMINAL,
                NodeType::Var(_) | NodeType::Prim(_) => COST_TERMINAL,
            };

            // update the upper bound
            let utility_upper_bound: i32 = utility_upper_bound(&locs, body_utility, &shared.cost_of_node_all, &shared.num_paths_to_node, &shared.cfg);

            // prune if utility upper bound is negative
            if utility_upper_bound <= 0 {
                if tracked { println!("{} <= 0 utility pruned ({}) when expanding {} to {}", "[TRACK]".red().bold(), utility_upper_bound, original_pattern.to_expr(&shared), original_pattern.to_expr(&shared).zipper_replace(&shared.zip_of_zid[hole_zid], &format!("<{}>",node_type))); }
                continue; // too low utility
            }

            // branch and bound: if the upper bound is less than the best invention we've found so far (our cutoff), we can discard this pattern
            if utility_upper_bound < weak_utility_pruning_cutoff {
                if !shared.cfg.no_stats { shared.stats.lock().deref_mut().upper_bound_fired += 1; };
                if tracked { println!("{} upper bound ({} < {}) pruned when expanding {} to {}", "[TRACK]".red().bold(), utility_upper_bound, weak_utility_pruning_cutoff, original_pattern.to_expr(&shared), original_pattern.to_expr(&shared).zipper_replace(&shared.zip_of_zid[hole_zid], &format!("<{}>",node_type))); }
                continue; // too low utility
            }

            // add any new holes to the list of holes
            let mut holes = holes_after_pop.clone();
            match node_type {
                NodeType::Lam => {
                    // add new holes
                    holes.push(shared.extensions_of_zid[hole_zid].body.unwrap());
                }
                NodeType::App => {
                    // add new holes
                        holes.push(shared.extensions_of_zid[hole_zid].func.unwrap());
                        holes.push(shared.extensions_of_zid[hole_zid].arg.unwrap());
                }
                _ => {}
            }

            // build our new pattern with all the variables we've just defined. Copy in the argchoices and prefixes
            // from the old pattern.
            let new_pattern = Pattern {
                holes,
                arg_choices: original_pattern.arg_choices.clone(),
                match_locations: locs,
                pruned_assignment_prefixes: original_pattern.pruned_assignment_prefixes.clone(),
                utility_upper_bound,
                body_utility,
                tracked
            };

            if new_pattern.holes.is_empty() {
                // it's a finished pattern so we get the assignments and they'll get pushed to the donelist.
                // (we intentionally only call this on partial inventions in the argchoice case below because it's a little expensive
                // and fairly unlikely to be a speedup except in the argchoice case).
                assignments_of_pattern(
                    new_pattern,
                    &mut weak_utility_pruning_cutoff,
                    &mut weak_lowest_donelist_utility,
                    &mut worklist_buf,
                    &mut donelist_buf,
                    &shared,
                );
            } else {
                // it's a partial pattern so just add it to the worklist
                if tracked { println!("{} pushed {} to worklist", "[TRACK]".green().bold(), original_pattern.to_expr(&shared).zipper_replace(&shared.zip_of_zid[hole_zid], &format!("<{}>",node_type))); }
                worklist_buf.push(HeapItem::new(new_pattern))
            }
        }

        // ***********************
        // * adding an argchoice *
        // ***********************
        // In this section we expand the hole that we just popped into an argchoice

        // for debugging
        let tracked = original_pattern.tracked && tracked_node_type(hole_zid, &shared).is_none();

        // add an argchoice, as long as it's actually abstracting a different thing over all locations
        if original_pattern.match_locations.iter().map(|loc| arg_of_loc[loc].id.clone()).all_equal() {
            // useless abstraction: takes the same argument at all locations
            if tracked { println!("{} useless abstraction pruned when expanding {} to {}", "[TRACK]".red().bold(), original_pattern.to_expr(&shared), original_pattern.to_expr(&shared).zipper_replace(&shared.zip_of_zid[hole_zid], "?#")); }
            if !shared.cfg.no_stats { shared.stats.lock().deref_mut().useless_abstract_fired += 1; };
        } else {
            // add the argchoice! Leave everything else as-is, just push the old hole zipper onto the argchoice zipper list
            let mut arg_choices = original_pattern.arg_choices.clone();
            arg_choices.push(hole_zid);
            let new_pattern = Pattern {
                holes: holes_after_pop,
                arg_choices,
                match_locations: original_pattern.match_locations.clone(),
                pruned_assignment_prefixes: original_pattern.pruned_assignment_prefixes.clone(),
                utility_upper_bound: original_pattern.utility_upper_bound,
                body_utility: original_pattern.body_utility,
                tracked
            };
            
            // if this is a partial invention then find assignments and update the worklist with one item,
            // otherwise if its a finished invention then find assignments and update the donelist with many items
            assignments_of_pattern(
                new_pattern,
                &mut weak_utility_pruning_cutoff,
                &mut weak_lowest_donelist_utility,
                &mut worklist_buf,
                &mut donelist_buf,
                &shared
            );
        }
        
    }

}




/// A finished invention
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FinishedPattern {
    labelled_zids: Vec<LabelledZId>, // a hole gets moved into here when it becomes an argchoice, again these are in order of when they were added
    match_locations: Vec<Id>, // places where it applies
    first_zid_of_ivar: Vec<ZId>, // map from first instance of an ivar to a zid
    utility: i32,
    compressive_utility: i32,
    arity: usize,
    usages: i32,
}

impl FinishedPattern {
    fn new(pattern: &Pattern, asn: &Assignment, shared: &SharedData) -> Self {
        let labelled_zids: Vec<LabelledZId> = pattern.arg_choices.iter().zip(asn.ivars.iter()).map(|(argchoice_zid,ivar)| LabelledZId::new(*argchoice_zid,*ivar as usize)).collect();
        let arity = *asn.max_ivar_used.last().unwrap() as usize + 1;
        let first_zid_of_ivar: Vec<ZId> = (0..arity).map(|ivar| labelled_zids.iter().find(|labelled| labelled.ivar == ivar).unwrap().zid).collect();
        let match_locations = asn.match_locations.last().unwrap().clone();
        let usages = match_locations.iter().map(|loc| shared.num_paths_to_node[loc]).sum();
        let compressive_utility = compressive_utility(
            &labelled_zids,
            &first_zid_of_ivar,
            arity,
            pattern.body_utility,
            &match_locations,
            shared
        );
        let noncompressive_utility = noncompressive_utility(pattern.body_utility, &shared.cfg);
        FinishedPattern {
            labelled_zids,
            match_locations,
            first_zid_of_ivar,
            utility: compressive_utility + noncompressive_utility,
            compressive_utility,
            arity,
            usages,
        }
    }
    // convert finished invention to an Expr
    fn to_expr(&self, shared: &SharedData) -> Expr {
        let mut curr_zip: Zip = vec![];
        // map zids to zips
        let zips: Vec<(Zip,i32)> = self.labelled_zids.iter().map(|labelled| (shared.zip_of_zid[labelled.zid].clone(), labelled.ivar as i32)).collect();

        fn helper(curr_node: Id, curr_zip: &mut Zip, zips: &Vec<(Zip,i32)>, shared: &SharedData) -> Expr {
            match zips.iter().find(|(zip,_)| zip == curr_zip) {
                // current zip matches an ivar zip so just insert the ivar
                Some((_,ivar)) => Expr::ivar(*ivar),
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
    fn to_invention(&self, name: &str, shared: &SharedData) -> Invention {
        Invention::new(self.to_expr(shared), self.arity, name)
    }

}

/// @theoxo this is super likely to change dont spend time understandign it
struct Assignment {
    ivars: Vec<i32>,
    max_ivar_used: Vec<i32>,
    match_locations: Vec<Vec<Id>>,
    first_ivar_use: Vec<usize>, // first_ivar_use[i] gives the index of the first use of #i in ivars
    can_extend: bool, // are we allowed to extend the current assignment by adding more ivars?
    ptr: usize,
    first_step: bool,
}

impl Assignment {
    /// @theoxo this is super likely to change dont spend time understandign it
    fn prune_branch(
        &mut self,
        pattern: &mut Pattern,
    ) {
        // if this an in-progress pattern then we care to update pruned_assignment_prefixes
        if !pattern.holes.is_empty() {
            // throw out any prefixes that this prefix subsumes
            pattern.pruned_assignment_prefixes.retain(|prefix| !prefix.starts_with(&self.ivars));
            // add the new prefix maintaining sorted order
            match pattern.pruned_assignment_prefixes.binary_search(&self.ivars) {
                Err(idx) => {pattern.pruned_assignment_prefixes.insert(idx, self.ivars.clone());},
                Ok(_) => unreachable!()
            }
        }
        self.can_extend = false; // after pruning a branch you can't immediately extend
    }
    /// @theoxo this is super likely to change dont spend time understandign it
    fn next(
        &mut self,
        pattern: &Pattern,
        shared: &SharedData,
    ) -> bool {
        loop {
            // this only happens on the very first step
            if self.first_step {
                self.first_step = false;
                return true; 
            }


            // prefer extending if we're allowed to
            if self.can_extend && self.ivars.len() < pattern.arg_choices.len() {
                // println!("extending");
                // todo at this point add auto-incrementing by arbitrary amts when you see that it doesnt cause subsetting
                self.ivars.push(0);
                self.ptr += 1;
                self.max_ivar_used.push(self.max_ivar_used[self.ptr - 1]);

                // backtrack if we know this to be a bad (too low utility) prefix
                if pattern.pruned_assignment_prefixes.binary_search(&self.ivars).is_ok() {
                    self.can_extend = false;
                    self.match_locations.push(vec![]); // todo ack hacky
                    // dont extend match location
                    if pattern.tracked { println!("prune by pruned_assignment_prefixes: {:?}", self.ivars); }
                    continue;
                }

                // filter for locations where the multiuse equality constraint holds.
                // note that we know that ivars[0] == 0 ie the first argchoice is always #0
                // so this is easier than in other cases where we have to look it up.
                let ref arg_of_node_self = shared.arg_of_zid_node[pattern.arg_choices[self.ptr]];
                let ref arg_of_node_ivar0 = shared.arg_of_zid_node[pattern.arg_choices[0]];
                // println!("{:?}", self.ivars);
                self.match_locations.push(self.match_locations.last().unwrap().iter()
                    .filter(|loc|
                        arg_of_node_self[*loc].id == 
                        arg_of_node_ivar0[*loc].id).cloned().collect());
                return true
            }


            // it's not our first step, but we're back to [0] so we're done
            if self.ivars.len() == 1 {
                return false // we backtracked to [0] so we're done
            }

            // since we couldnt' extend, we'd like to increment our ivar.
            // this is allowed if we aren't breaking max_arity and we aren't
            // breaking the constraint that the first use of #1 must come after #0 etc
            // (here, as long as the past ivars contain something at least as big as our
            // current ivar, it's okay to increment our current ivar)
            let new_ivar = self.ivars[self.ptr] + 1;

            if new_ivar < shared.cfg.max_arity as i32
                && self.ivars[self.ptr] <= self.max_ivar_used[self.ptr - 1]
            {
                // println!("incrementing");
                // increment ivar and update max_ivar_used
                self.ivars[self.ptr] = new_ivar;
                if new_ivar > *self.max_ivar_used.last().unwrap() {
                    self.max_ivar_used[self.ptr] = new_ivar;
                    // since this is a new record for largest ivar, we should
                    // add it as a new entry in first_ivar_use too.
                    self.first_ivar_use.push(self.ptr);
                    assert_eq!((self.first_ivar_use.len() - 1) as i32, new_ivar);
                }

                let first_ivar_use = self.first_ivar_use[self.ivars[self.ptr] as usize];

                // backtrack if we know this to be a bad (too low utility) prefix
                if pattern.pruned_assignment_prefixes.binary_search(&self.ivars).is_ok() {
                    self.can_extend = false;
                    // dont extend match location
                    if pattern.tracked { println!("prune by pruned_assignment_prefixes: {:?}", self.ivars); }
                    continue;
                }

                // when you increment, you get to pop the previous match_locations and
                // push a new one thats subsetting on the one *before* that one.
                let ref arg_of_node_self = shared.arg_of_zid_node[pattern.arg_choices[self.ptr]];
                let ref arg_of_node_other = shared.arg_of_zid_node[pattern.arg_choices[first_ivar_use]];
                self.match_locations.pop();
                self.match_locations.push(self.match_locations.last().unwrap().iter()
                    .filter(|loc|
                        arg_of_node_self[loc].id == 
                        arg_of_node_other[loc].id).cloned().collect());
                
                // having incremented an ivar, we are now allowed to extend if we weren't already
                self.can_extend = true;
                return true
            }

            // we couldnt increment our ivar so we need to backtrack
            if *self.first_ivar_use.last().unwrap() == self.ptr {
                self.first_ivar_use.pop();
            }
            self.ivars.pop();
            self.max_ivar_used.pop();
            self.match_locations.pop();
            self.ptr -= 1;
            self.can_extend = false; // you cant extend forward right after backtracking
        }
    }
}

/// this finds the set of assignments for a finished or unfinished pattern. This is VERY LIKELY TO CHANGE
/// so @theoxo dont spent time understanding this
fn assignments_of_pattern(
    mut pattern: Pattern,
    weak_utility_pruning_cutoff: &mut i32,
    weak_lowest_donelist_utility: &mut i32,
    worklist_buf: &mut Vec<HeapItem>,
    donelist_buf: &mut Vec<FinishedPattern>,
    shared: &Arc<SharedData>,
) {
    let mut unfinished_pattern_succeeded: bool = false;
    let is_finished_pattern: bool = pattern.holes.is_empty();

    let mut asn = Assignment {
        ivars: vec![0],
        max_ivar_used: vec![0],
        match_locations: vec![pattern.match_locations.clone()],
        first_ivar_use: vec![0],
        can_extend: true,
        ptr: 0,
        first_step: true,
    };

    if pattern.tracked && is_finished_pattern {
        println!("{} Pattern finished, looking for assignments: {}", "[TRACK]".yellow().bold(), pattern.to_expr(&shared));
    }


    while asn.next(&pattern, shared) {
        if pattern.tracked { println!("trying assignment: {:?}", asn.ivars); }
        // println!("ptr: {}", asn.ptr);
        // println!("ivars: {:?}", asn.ivars);

        // prune if not used in any places
        if asn.match_locations.last().unwrap().len() == 0 {
            asn.prune_branch(&mut pattern);
            if pattern.tracked { println!("prune by 0 uses: {:?}", asn.ivars); }
            continue;
        }

        // prune if only used in a single place
        if asn.match_locations.last().unwrap().len() == 1 {
            // panic!("single {:?} for {}", asn.ivars, pattern.to_expr(shared));
            if is_finished_pattern {
                if pattern.tracked { println!("prune by 1 use: {:?}", asn.ivars); }
                if !shared.cfg.no_stats { shared.stats.lock().single_use_done_fired += 1; }
                asn.prune_branch(&mut pattern);
                continue;
            } else if !shared.cfg.no_opt_single_use {
                if pattern.tracked { println!("prune by 1 use: {:?}", asn.ivars); }
                if !shared.cfg.no_stats { shared.stats.lock().single_use_wip_fired += 1; }
                asn.prune_branch(&mut pattern);
                continue;
            }
        }

        

        // check upper bound, and discard + add to bad prefix list if it fails
        let utility_upper_bound: i32 = utility_upper_bound(asn.match_locations.last().unwrap(), pattern.body_utility, &shared.cost_of_node_all, &shared.num_paths_to_node, &shared.cfg);
        if utility_upper_bound < *weak_utility_pruning_cutoff {
            asn.prune_branch(&mut pattern);
            if pattern.tracked { println!("prune by upper bound: {:?}", asn.ivars); }
            continue;
        }

        // if its a finished pattern and doesnt beat the lowest donelist utility, itll never survive
        if is_finished_pattern && utility_upper_bound <= *weak_lowest_donelist_utility {
            if pattern.tracked { println!("prune by lowest_donelist_utility: {:?}", asn.ivars); }
            asn.prune_branch(&mut pattern);
            continue
        }

        // check for useless abstractions (ie same arg everywhere) which might have arison from our narrowing of the match_locations
        if pattern.arg_choices.iter()
            .any(|argchoice_zid| asn.match_locations.last().unwrap().iter()
                .map(|loc| shared.arg_of_zid_node[*argchoice_zid][loc].id.clone()).all_equal())
        {
            if !shared.cfg.no_stats { shared.stats.lock().deref_mut().useless_abstract_fired += 1; };
            asn.prune_branch(&mut pattern);
            if pattern.tracked { println!("prune by useless abstraction: {:?}", asn.ivars); }
            continue; // useless abstraction
        }

        // if we've assigned all the argchoices to ivars (and we know we already had a reasonable utility). Note this branch is irrelevant if we've already succeeded once and we're an unfinished pattern
        if asn.ivars.len() == pattern.arg_choices.len() && !unfinished_pattern_succeeded {
            if is_finished_pattern {
                // add to donelist
                // todo add refinement here
                // we do need to prune negatives in the args, I'll do that here since later we'll be refining here
                
                let finished_pattern = FinishedPattern::new(
                    &pattern,
                    &asn,
                    &shared
                );
                // todo migrate this over to using negative ivars some time
                if finished_pattern.first_zid_of_ivar.iter().any(|zid|
                    shared.egraph[shared.arg_of_zid_node[*zid][&finished_pattern.match_locations[0]].id].data.free_vars.iter().any(|free_var|
                        *free_var < 0))
                {
                    if pattern.tracked { println!("{} discarding finished_pattern because one of its args has negative vars: {}", "[TRACK]".red().bold(), finished_pattern.to_expr(&shared)); }
                    continue;
                }
                if pattern.tracked {
                    println!("{} pushed {} to donelist", "----->".yellow().bold(), finished_pattern.to_expr(&shared));
                }
                donelist_buf.push(finished_pattern);
                
                continue; 
            } else {
                // worklist case - note that at this point we know we wont be able to
                // completely prune this worklist item, so we can break early if we want
                // OR more likely we should just keep going to improve our `pruned_assignment_prefixes`
                // since if we dont do that now our children might have to do repeat that work separately
                // for each child.
                unfinished_pattern_succeeded = true;
                if shared.cfg.break_early_assignment {
                    break
                }
                continue; 
            }
        }

    }


    if !is_finished_pattern {
        if unfinished_pattern_succeeded {
            if pattern.tracked { println!("{} pushed {} to worklist", "[TRACK]".green().bold(), pattern.to_expr(&shared)); }
            worklist_buf.push(HeapItem::new(pattern))
        } else {
            if pattern.tracked { println!("{} discarding because failed to find an assignment: {}", "[TRACK]".red().bold(), pattern.to_expr(&shared)); }
        }
    }


}

/// figure out all the N^2 zippers from choosing any given node and then choosing a descendant and returning the zipper from
/// the node to the descendant. We also collect a bunch of other useful stuff like the argument you would get if you abstracted
/// the descendant and introduced an invention rooted at the ancestor node.
fn get_zippers(
    treenodes: &[Id],
    cost_of_node_once: &HashMap<Id,i32>,
    no_cache: bool,
    egraph: &mut EGraph,
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
            Arg::new(*treenode, *treenode, cost_of_node_once[treenode], node_type_of_node(&node).unwrap()));
        
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
        let rewritten: Expr = rewrite_with_invention_egraph(programs_node, &inv, &mut shared.egraph);

        let expected_cost = initial_cost - done.compressive_utility;
        let final_cost = rewritten.cost();
        if expected_cost != final_cost {
            println!("*** expected cost {} != final cost {}", expected_cost, final_cost);
        }
        let multiplier = initial_cost as f64 / final_cost as f64;
        let multiplier_wrt_orig = very_first_cost as f64 / final_cost as f64;
        let uses = done.usages;
        let use_exprs: Vec<Expr> = done.match_locations.iter().map(|node| extract(*node, &shared.egraph)).collect();
        let use_args: Vec<Vec<Expr>> = done.match_locations.iter().map(|node|
            done.first_zid_of_ivar.iter().map(|zid|
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



/// This utility directly corresponds to decrease in program cost of the
/// final program tree once it has been rewritten with the invention. Program
/// cost is leaf_nodes * 100 + non_leaf_nodes * 1, so dividing a cost (or a utility)
/// by 100 will give approximately the number of leaf nodes.
/// 
/// At a very high level, we can calculate this utility as:
///     (num places the invention is useful) * (size of invention body)
/// However it's a little more complicated due to inventions re-using their variables
/// and the slight cost of using the invention primitive itself.
fn compressive_utility(
    labelled_zids: &Vec<LabelledZId>,
    first_zid_of_ivar: &Vec<ZId>, // [i] gives the zid for the first instance of #i
    arity: usize,
    body_utility: i32,
    match_locations: &Vec<Id>,
    shared: &SharedData,
) -> i32 {

    // get a list of (ivar,usages-1) filtering out things that are only used once
    let ivar_multiuses: Vec<(usize,i32)> = labelled_zids.iter().map(|labelled|labelled.ivar).counts()
        .iter().filter_map(|(ivar,count)| if *count > 1 { Some((*ivar, (*count-1) as i32)) } else { None }).collect();

    // removed: ctx_thread_penalty

    // it costs a tiny bit to apply the invention, for example (app (app inv0 x) y) incurs a cost
    // of COST_TERMINAL for the `inv0` primitive and 2 * COST_NONTERMINAL for the two `app`s.
    let app_penalty = - (COST_TERMINAL + COST_NONTERMINAL * arity as i32);

    let compressive_utility = match_locations.iter().map(|loc|{
        // compressivity of body minus slight penalty from the application
        let base_utility = body_utility + app_penalty;
        // for each extra usage of an argument, we gain the cost of that argument as
        // extra utility. Note we use `first_zid_of_ivar` since it doesn't matter which
        // of the zids we use as long as it corresponds to the right ivar
        let multiuse_utility = ivar_multiuses.iter().map(|(ivar,count)|
            count * shared.arg_of_zid_node[first_zid_of_ivar[*ivar]][loc].cost
        ).sum::<i32>();
        // multiply all this utility by the number of times this node shows up
        (base_utility + multiuse_utility) * shared.num_paths_to_node[loc]
    }).sum::<i32>();

    compressive_utility
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

/// Multistep compression. See `compression_step` if you'd just like to do a single step of compression.
pub fn compression(
    programs_expr: &Expr,
    iterations: usize,
    cfg: &CompressionStepConfig,
    tasks: &Vec<String>,
    num_prior_inventions: usize,
) -> Vec<CompressionStepResult> {
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
    let tstart = std::time::Instant::now();


    // set up tracking if any
    let tracking: Option<Tracking> = cfg.track.as_ref().map(|s|{
        let mut s = s.clone();
        for i in 0..30 {
            // un-assign any ivars from #i back to ?#
            s = replace_prim_with(&s, &format!("#{}",i),&format!("?#"));
        }
        let expr: Expr = s.parse().unwrap();
        Tracking { expr }
    });

    // build the egraph. We'll just be using this as a structural hasher we don't use rewrites at all. All eclasses will always only have one node.
    let mut egraph: EGraph = Default::default();
    let programs_node = egraph.add_expr(programs_expr.into());
    egraph.rebuild();

    let roots: Vec<Id> = egraph[programs_node].nodes[0].children().iter().cloned().collect();

    // all nodes in child-first order except for the Programs node
    let mut treenodes: Vec<Id> = topological_ordering(programs_node,&egraph);
    treenodes.retain(|id| *id != programs_node);
    // assert!(usize::from(*treenodes.iter().max().unwrap()) == treenodes.len() - 1); // ensures we can safely just use Vecs of length treenodes.len() to store various nodewise things
    // let treenodes_no_programs_node: Vec<Id> = treenodes.iter().filter(|&&id| id != programs_node).cloned().collect();

    // populate num_paths_to_node so we know how many different parts of the programs tree
    // a node participates in (ie multiple uses within a single program or among programs)
    let num_paths_to_node: HashMap<Id,i32> = num_paths_to_node(&roots, &treenodes, &egraph);
    let tasks_of_node: HashMap<Id, HashSet<usize>> = associate_tasks(programs_node, &egraph, tasks);
    // cost of a single usage of a node (same as inventionless_cost)
    let cost_of_node_once: HashMap<Id,i32> = treenodes.iter().map(|node| (*node,egraph[*node].data.inventionless_cost)).collect();
    // cost of a single usage times number of paths to node
    let cost_of_node_all: HashMap<Id,i32> = treenodes.iter().map(|node| (*node,cost_of_node_once[node] * num_paths_to_node[node])).collect();

    println!("set up low cost data structs: {:?}ms", tstart.elapsed().as_millis());

    let tstart = std::time::Instant::now();
    let (_zid_of_zip,
        zip_of_zid,
        arg_of_zid_node,
        zids_of_node,
        extensions_of_zid,
        descendants_of_node) = get_zippers(&treenodes, &cost_of_node_once, cfg.no_cache, &mut egraph, cfg);
    println!("get_zippers: {:?}ms", tstart.elapsed().as_millis());


    println!("{} zips", zip_of_zid.len());
    println!("arg_of_zid_node size: {}", arg_of_zid_node.len());

    // define all the important data structures for compression
    let mut donelist: Vec<FinishedPattern> = Default::default(); // completed inventions will go here    

    let tstart = std::time::Instant::now();

    // arity 0 inventions
    for node in treenodes.iter() {
        if !egraph[*node].data.free_vars.is_empty() { continue; }
        if tasks_of_node[&node].len() < 2 { continue; }
        // Note that "single use" pruning is intentionally not done here,
        // since any invention specific to a node will by definition only
        // be useful at that node

        let match_locations = vec![*node];
        let body_utility = cost_of_node_once[node];
        // compressive_utility for arity-0 is cost_of_node_all[node] minus the penalty of using the new prim
        let compressive_utility = cost_of_node_all[node] - num_paths_to_node[node] * COST_TERMINAL;
        let utility = compressive_utility + noncompressive_utility(body_utility, cfg);
        if utility <= 0 { continue; }

        donelist.push(FinishedPattern { labelled_zids: vec![], match_locations, first_zid_of_ivar: vec![], utility, compressive_utility, arity: 0, usages: num_paths_to_node[node] });
    }
    println!("got {} arity zero inventions in {:?}ms", donelist.len(), tstart.elapsed().as_millis());


    // sort and truncate

    let stats: Stats = Default::default();

    println!("total prep: {:?}ms", tstart_total.elapsed().as_millis());

    println!("running pattern search...");
    let tstart = std::time::Instant::now();


    let crit = CriticalMultithreadData::new(donelist, &treenodes, &cost_of_node_all, &num_paths_to_node, &egraph, &cfg);
    let shared = Arc::new(SharedData {
        crit: Mutex::new(crit),
        arg_of_zid_node,
        treenodes: treenodes.clone(),
        zids_of_node,
        zip_of_zid,
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

    if cfg.verbose_best {
        let mut crit = shared.crit.lock();
        let best_util = crit.deref_mut().donelist.first().unwrap().utility;
        let best_expr: Expr = crit.deref_mut().donelist.first().unwrap().to_expr(&shared);
        println!("{} @ step=0 util={} for {}", "[new best utility]".blue(), best_util, best_expr);
    }
    
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
            
            // launch thread to just call derive_inventions()
            handles.push(thread::spawn(move || {
                stitch_search(shared);
            }));
        }
        // wait for all threads to finish (when all have empty worklists)
        for handle in handles {
            handle.join().unwrap();
        }
    }

    // at this point we hold the only reference so we can get rid of the Arc
    let mut shared: SharedData = Arc::try_unwrap(shared).unwrap();

    // one last .update()
    shared.crit.lock().deref_mut().update(&cfg);

    println!("{:?}", shared.stats.lock().deref_mut());
    assert!(shared.crit.lock().deref_mut().worklist.is_empty());

    let donelist: Vec<FinishedPattern> = shared.crit.lock().deref_mut().donelist.clone();

    let elapsed_derive_inventions = tstart.elapsed().as_millis();

    println!("\nstitch_search() done: {:?}ms\n", elapsed_derive_inventions);
    println!("total everything: {:?}ms", tstart_total.elapsed().as_millis());    

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


    println!("Final donelist length: {}",donelist.len());
    println!("derive_inventions() took: {}ms ***\n", elapsed_derive_inventions);

    results
}
