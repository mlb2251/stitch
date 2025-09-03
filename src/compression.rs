use crate::*;
use lambdas::*;
use rand::seq::SliceRandom;
use rustc_hash::{FxHashMap,FxHashSet};
use core::panic;
use std::convert::TryInto;
use std::fmt::{self, Formatter, Display};
use std::hash::{Hash, Hasher};
use std::str::FromStr;
use itertools::Itertools;
use serde_json::{json, Value};
use clap::{Parser};
use serde::Serialize;
use std::thread;
use std::sync::Arc;
use parking_lot::Mutex;
use std::ops::DerefMut;
use std::collections::{BinaryHeap, HashMap};
use rand::Rng;

pub type Cost = i64;

/// Multistep Compression
#[derive(Parser, Debug, Serialize, Clone)]
#[clap(name = "Multistep Compression")]
pub struct MultistepCompressionConfig {

    /// Maximum number of iterations to run compression for (number of inventions to find, though
    /// stitch will stop early if no compressive abstraction exists)
    #[clap(short, long, default_value = "3")]
    pub iterations: usize,

    /// Prefix used to generate names of new abstractions, by default we will name our
    /// abstractions fn_0, fn_1, fn_2, etc
    #[clap(long, default_value = "fn_")]
    pub abstraction_prefix: String,

    /// Number of previous abstractions that have been found before this round of compression - this
    /// is used to calculate what the next abstraction name should be - for example if 2 abstractions have
    /// been found previously then the next abstraction will be fn_2
    #[clap(long, default_value = "0")]
    pub previous_abstractions: usize,

    /// Shuffles the order of the programs passed in
    #[clap(long)]
    pub shuffle: bool,

    /// Truncate the list of programs (happens after shuffle if shuffle is also specified)
    #[clap(long)]
    pub truncate: Option<usize>,

    /// Disable all optimizations
    #[clap(long)]
    pub no_opt: bool,

    /// Disables all prinouts except in the case of a panic. See also `quiet` to just silence internal printouts during each compression step
    /// In Python this defaults to True.
    #[clap(long)]
    pub silent: bool,

    /// Very verbose when rewriting happens - turns off --silent and --quiet which are usually forced on
    /// in rewriting
    #[clap(long)]
    pub verbose_rewrite: bool,

    #[clap(flatten)]
    pub step: CompressionStepConfig,
}

/// Args for compression step
#[derive(Parser, Debug, Serialize, Clone)]
#[clap(name = "Stitch")]
pub struct CompressionStepConfig {
    /// Max arity of abstractions to find (will find arities from 0 to this number inclusive).
    /// Note that scaling with arity can be very expensive
    #[clap(short='a', long, default_value = "2")]
    pub max_arity: usize,

    /// Number of threads to use for compression (no parallelism if set to 1)
    #[clap(short='t', long, default_value = "1")]
    pub threads: usize,

    /// Disable stat logging - note that stat logging in multithreading requires taking a mutex
    /// so it can be a source of slowdown in the massively multithreaded case, hence this flag to disable it.
    #[clap(long)]
    pub no_stats: bool,

    /// How many worklist items a thread will take at once
    #[clap(short='b', long, default_value = "1")]
    pub batch: usize,

    /// Threads will autoadjust how large their batches are based on the worklist size
    #[clap(long)]
    pub dynamic_batch: bool,

    /// Puts result into eta-long form when rewriting (also requires beta-normal form). This
    /// can be useful for programs that will be used to train top down synthesizers, but it also
    /// restricts what abstractions can be found a bit (i.e. only those that can be put in beta-normal
    /// eta-long form are allowed).
    #[clap(long)]
    pub eta_long: bool,

    /// Forbid metavariables to the left of an app
    #[clap(long)]
    pub no_curried_metavars: bool,

    /// Forbid abstraction bodies rooted to the left of an app (see also no_curried_metavars and eta_long)
    #[clap(long)]
    pub no_curried_bodies: bool,

    /// [currently not used] Number of invention candidates compression_step should return in a *single* step. Note that
    /// these will be the top n optimal candidates modulo subsumption pruning (and the top-1 is guaranteed
    /// to be globally optimal)
    #[clap(short='n', long, default_value = "1")]
    pub inv_candidates: usize,

    /// Method for choosing hole to expand at each step. Doesn't have a huge effect.
    #[clap(long, arg_enum, default_value = "depth-first")]
    pub hole_choice: HoleChoice,

    #[clap(flatten)]
    pub cost: CostConfig,

    /// Disables the safety check for the utility being correct; you only want
    /// to do this if you truly dont mind unsoundness for a minute
    #[clap(long)]
    pub no_mismatch_check: bool,

    /// Pattern or abstraction to follow and give prinouts about. If `follow_prune=True` we will aggressively prune to
    /// only follow this pattern, otherwise we will just verbosely print when ancestors of this pattern
    /// are encountered.
    #[clap(long)]
    pub follow: Option<String>,

    /// Variable types for the `follow` invention. Use M for metavariable and S for symbolic variable.
    #[clap(long, value_parser, value_delimiter = ' ')]
    pub follow_types: Option<Vec<VariableType>>,

    /// For use with `follow`, enables aggressive pruning. Useful for ensuring that it is *possible* to find a particular
    /// abstraction by guiding the search directly towards it.
    #[clap(long)]
    pub follow_prune: bool,

    /// Prints every worklist item as it is processed (will slow things down a ton due to rendering out expressions).
    #[clap(long)]
    pub verbose_worklist: bool,
    
    /// Prints whenever a new best abstraction is found
    #[clap(long)]
    pub verbose_best: bool,

    /// Print stats this often (0 means never)
    #[clap(long, default_value = "0")]
    pub print_stats: usize,

    /// Print out programs rewritten under abstraction
    #[clap(long,short='r')]
    pub show_rewritten: bool,

    /// Include the dreamcoder-format rewritten programs in the output
    #[clap(long)]
    pub rewritten_dreamcoder: bool,

    /// For each abstraction learned, includes the rewritten programs right after learning that
    /// abstraction in the output. If `rewritten_dreamcoder` is also specified, then the rewritten
    /// programs in dreamcoder format will also be included.
    #[clap(long)]
    pub rewritten_intermediates: bool,    

    /// Enables edge case handling where inverting the argument capture subsumption pruning is needed for optimality.
    /// Generally not relevant just included for completeness, see the footnoted section
    /// of Section 4.3 of the Stitch paper https://arxiv.org/abs/2211.16605
    #[clap(long)]
    pub inv_arg_cap: bool,

    /// Allow for abstractions that are only useful in a single task (defaults to False like DreamCoder)
    #[clap(long)]
    pub allow_single_task: bool,

    /// Disable the single structurally hashed subtree match pruning. This is a very minor optimization that allows
    /// discarding certain abstractions that only match at a single unique subtree as long as that subtree lacks free
    /// variables, because arity zero abstractions are always superior in this case
    #[clap(long)]
    pub no_opt_single_use: bool,

    /// Disable *upper bound* based pruning. Section 4.2 of Stitch paper https://arxiv.org/abs/2211.16605
    /// This is an extremely important optimization (ablation study in Section 6.4 of Stitch paper)
    #[clap(long)]
    pub no_opt_upper_bound: bool,

    /// Disable *redundant argument elimination* pruning (aka "force multiuse"). Section 4.3 of Stitch paper https://arxiv.org/abs/2211.16605
    /// This is a fairly important optimization (ablation study in Section 6.4 of Stitch paper)
    #[clap(long)]
    pub no_opt_force_multiuse: bool,

    /// Disable *argument capture* pruning (aka "useless abstraction pruning"). Section 4.3 of Stitch paper https://arxiv.org/abs/2211.16605
    /// This is an extremely important optimization (ablation study in Section 6.4 of Stitch paper)
    #[clap(long)]
    pub no_opt_useless_abstract: bool,

    /// Disable the arity zero optimization, which searches first for the most compressive arity-zero abstraction since this
    /// is extremely fast to find and provides a good starting point for our upper bound pruning. In practice this isn't a very
    /// important optimization
    #[clap(long)]
    pub no_opt_arity_zero: bool,

    /// Switch to utility based purely on program size without adding
    /// in the abstraction size (aka the "structure penalty" in DreamCoder)
    #[clap(long)]
    pub no_other_util: bool,

    /// DreamCoder style structure penalty - must be positive. Overall utility is this
    /// difference in corpus size minus structure_penalty * abstraction_size
    #[clap(long, default_value = "1.0")]
    pub structure_penalty: f32,

    /// Used for soundness testing. Whenever you finish an invention do a full rewrite to check
    /// that rewriting doesnt raise a cost mismatch exception. 
    #[clap(long)]
    pub rewrite_check: bool,

    /// Calculate utility exhaustively by performing a full rewrite. Used for debugging when cost mismatch exceptions
    /// are happening and we need something slow but accurate as a temporary solution.
    #[clap(long)]
    pub utility_by_rewrite: bool,

    /// Extra printouts related to running a dreamcoder comparison. Section 6.1 of Stitch paper https://arxiv.org/abs/2211.16605
    #[clap(long)]
    pub dreamcoder_comparison: bool,

    /// Silence all printing within a compression step. See `silent` to silence all outputs between compression steps as well.
    #[clap(long)]
    pub quiet: bool,

    // Fused lambda tags
    #[clap(long, value_parser = clap::value_parser!(FusedLambdaTags), default_value="")]
    pub fused_lambda_tags: FusedLambdaTags,

    /// TDFA settings
    #[clap(flatten)]
    pub tdfa: TDFAConfig,

    /// Symvar settings
    #[clap(flatten)]
    pub symvar: SymvarConfig,
}

#[derive(Debug, Clone, Serialize)]
pub struct FusedLambdaTags {
    tags: Option<FxHashSet<Tag>>,
}

// parse from a string like "1,2,3"
impl std::str::FromStr for FusedLambdaTags {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Ok(FusedLambdaTags { tags: None })
        }
        let tags = s.split(',').map(|s| s.parse::<Tag>().unwrap()).collect();
        Ok(FusedLambdaTags { tags: Some(tags) })
    }
}

impl CompressionStepConfig {
    pub fn no_opt(&mut self) {
        self.no_opt_upper_bound = true;
        self.no_opt_force_multiuse = true;
        self.no_opt_useless_abstract = true;
        self.no_opt_arity_zero = true;
    }
    pub fn new() -> Self {
        Self::parse_from("compress".split_whitespace())
    }
}
impl MultistepCompressionConfig {
    pub fn new() -> Self {
        Self::parse_from("compress".split_whitespace())
    }
}

// we use these manual implementations - deriving would set things to zero instead of
// to their clap defaults i think
impl Default for MultistepCompressionConfig {
    fn default() -> Self {
        Self::new()
    }
}
impl Default for CompressionStepConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// A Pattern is a partial abstraction which may have holes
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pattern {
    pub holes: Vec<ZId>, // zipper to hole in order of when theyre added NOT left to right
    pub pattern_args: PatternArgs,
    pub match_locations: Vec<Idx>, // places where it applies
    pub utility_upper_bound: Cost,
    pub body_utility: Cost, // the size (in `cost`) of a single use of the pattern body so far
    pub tracked: bool, // for debugging
}

impl Hash for Pattern {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Only hash the structural components of the pattern
        self.holes.hash(state);
        self.pattern_args.hash(state);
        // Intentionally skip:
        // - match_locations (computed)
        // - utility_upper_bound (computed)
        // - body_utility (computed)
        // - tracked (debug flag)
    }
}

/// only used during tracking - gets the zippers to args of a pattern
fn zids_of_ivar_of_expr(expr: &ExprOwned, zid_of_zip: &FxHashMap<Vec<ZNode>,ZId>) -> Option<Vec<Vec<ZId>>> {

    // quickly determine arity
    let mut arity = 0;
    for node in expr.set.iter() {
        if let Node::IVar(ivar) = expr.set[node] {
            if ivar + 1 > arity {
                arity = ivar + 1;
            }
        }
    }

    let mut curr_zip: Vec<ZNode> = vec![];
    let mut zids_of_ivar = vec![vec![]; arity as usize];

    fn helper(expr: Expr, curr_zip: &mut Vec<ZNode>, zids_of_ivar: &mut Vec<Vec<ZId>>, zid_of_zip: &FxHashMap<Vec<ZNode>,ZId>) -> Result<(), ()> {
        match expr.node() {
            Node::Prim(_) => {},
            Node::Var(_, _) => {},
            Node::IVar(i) => {
                zids_of_ivar[*i as usize].push(zid_of_zip.get(curr_zip).cloned().ok_or(())?);
            },
            Node::Lam(b, _) => {
                curr_zip.push(ZNode::Body);
                helper(expr.get(*b), curr_zip, zids_of_ivar, zid_of_zip)?;
                curr_zip.pop();
            }
            Node::App(f,x) => {
                curr_zip.push(ZNode::Func);
                helper(expr.get(*f), curr_zip, zids_of_ivar, zid_of_zip)?;
                curr_zip.pop();
                curr_zip.push(ZNode::Arg);
                helper(expr.get(*x), curr_zip, zids_of_ivar, zid_of_zip)?;
                curr_zip.pop();
            }
        }
        Ok(())
    }
    // we can pick any match location
    if helper(expr.immut(), &mut curr_zip, &mut zids_of_ivar, zid_of_zip).is_err() {
        return None
    };

    Some(zids_of_ivar)
}


/// Args for cost function used
#[derive(Parser, Debug, Serialize, Clone)]
#[clap(name = "Cost Config")]
pub struct CostConfig {
    /// Sets cost for lambdas
    #[clap(long, default_value = "1")]
    pub cost_lam: usize,
    
    /// Sets cost for applications in the lambda calculus
    #[clap(long, default_value = "1")]
    pub cost_app: usize,

    /// Sets cost for `$i` variables
    #[clap(long, default_value = "100")]
    pub cost_var: usize,

    /// Sets cost for `#i` abstraction variables
    #[clap(long, default_value = "100")]
    pub cost_ivar: usize,

    /// Sets cost for primitives like `+` and `*`
    #[clap(long, default_value = "100")]
    pub cost_prim_default: usize,

    /// Sets cost for primitives like `+` and `*` in the form of a dictionary, json encoded
    #[clap(long, default_value = "{}")]
    pub cost_prim: String,
}

impl CostConfig {

    fn compute_cost_prim(&self) -> HashMap<Symbol, i32> {
        let cost_prim: serde_json::Value = serde_json::from_str(&self.cost_prim).unwrap_or_else(|_| {
            panic!("Invalid JSON for argument --cost-prim: {}", self.cost_prim);
        });
        let serde_json::Value::Object(map_obj) = cost_prim else {
            panic!("Expected a JSON object for --cost-prim, got: {}", self.cost_prim);
        };
        let mut map: HashMap<Symbol, i32> = HashMap::default();
        for (k, v) in map_obj {
            if let serde_json::Value::Number(ref num) = v {
                let cost_opt: Option<i32> = num.as_i64().and_then(|x| TryInto::try_into(x).ok());
                let cost = cost_opt.unwrap_or_else(|| {
                    panic!("Expected a number for cost of primitive '{}', got: {}", k, v);
                });
                map.insert(Symbol::from(k), cost);
            } else {
                panic!("Expected a number for cost of primitive '{}', got: {}", k, v);
            }
        }
        map
    }

    pub fn expr_cost(&self) -> ExprCost {
        ExprCost::new(
            self.cost_lam.try_into().unwrap(),
            self.cost_app.try_into().unwrap(),
            self.cost_var.try_into().unwrap(),
            self.cost_ivar.try_into().unwrap(),
            self.compute_cost_prim(),
            self.cost_prim_default.try_into().unwrap(),
        )
    }
}


impl Pattern {
    /// create a single hole pattern `??`
    //#[inline(never)]
    #[allow(clippy::too_many_arguments)]
    fn single_hole(corpus_span: &Span, cost_fn: &ExprCost, cost_of_node_all: &[Cost], num_paths_to_node: &[Cost], tdfa_global_annotations: &Option<TDFAGlobalAnnotations>, set: &ExprSet, cfg: &CompressionStepConfig, follow: &Option<Invention>) -> Self {
        let body_utility = 0;
        let mut match_locations: Vec<Idx> = corpus_span.clone().collect();
        match_locations.sort(); // we assume match_locations is always sorted

        let match_locations_before = match_locations.clone();

        if cfg.no_curried_bodies {
            for node in corpus_span.clone() {
                if let Node::App(f,_) = &set[node] {
                    // similar to eta_long, no appzipper bodies are allowed to be rooted to the left of an App
                    match_locations.retain(|node| node != f);
                }
            }
        }

        match_locations.retain(|node|
            !invalid_match_location(set,
                                    &cfg.fused_lambda_tags.tags,
                                    tdfa_global_annotations,
                                    *node)
        );
        
        if cfg.eta_long {

            assert!(cfg.utility_by_rewrite || cfg.no_mismatch_check, "eta long form requires utility_by_rewrite or no_mismatch_check");

            for node in corpus_span.clone() {
                if let Node::App(f,_) = &set[node] {
                    // this for eta long form / dreamcoder compatability: no appzipper bodies can be rooted to the left of an App
                    // because that means the body is a function type, which isnt allowed. For example an arity 2 invention with a
                    // function type body would be effectively arity 3 and dreamcoder doesnt support this sort of thing.
                    match_locations.retain(|node| node != f);

                    if let Node::Lam(_, _) = &set[*f] {
                        panic!("corpus was not in beta-normal form")
                    }
                }
            }
            // check that original corpus was in eta long form, or at least that if a term appears to the left of an
            // app it never also appears to the right of an app. If it's to the left of an app it must be a function type
            for node in match_locations_before {
                match &set[node] {
                    Node::App(_,x) => {
                        if !AnalyzedExpr::new(FreeVarAnalysis).analyze_get(set.get(*x)).is_empty() {
                            // continue if there are free vars in the expression - eg $0 can validly appear to either the left or right of an app
                            continue 
                        }
                        assert!(match_locations.contains(x), "corpus was not in eta long form (?). This appeared both to the left and right of an app: {}; for example it is to the right in: {}", set.get(*x), set.get(node));
                    },
                    Node::Lam(b, _) => {
                        if !AnalyzedExpr::new(FreeVarAnalysis).analyze_get(set.get(*b)).is_empty() {
                            continue
                        }
                        assert!(match_locations.contains(b), "corpus was not in eta long form (?)");
                    },
                    _ => {}
                }
            }
        }

        // to guarantee eta long we cant allow abstractions to start with a lambda at the top
        if cfg.eta_long {
            match_locations.retain(|node| expands_to_of_node(&set[*node]).is_lam());
        }

        let utility_upper_bound = utility_upper_bound(&match_locations, body_utility, cost_fn, cost_of_node_all, num_paths_to_node, cfg);
        Pattern {
            holes: vec![EMPTY_ZID], // (zid 0 is the empty zipper)
            pattern_args: PatternArgs::default(),
            match_locations, // single hole matches everywhere
            utility_upper_bound,
            body_utility, // 0 body utility
            tracked: follow.is_some(),
        }
    }
    /// convert pattern to an Expr
    fn to_expr(&self, shared: &SharedData) -> ExprOwned {
        let mut set = ExprSet::empty(Order::ChildFirst, false, false);

        let mut curr_zip: Vec<ZNode> = vec![];
        // map zids to zips with a bool thats true if this is a hole and false if its a future ivar
        let zips: Vec<(Vec<ZNode>,Node)> = self.holes.iter().map(|zid| (shared.zip_of_zid[*zid].clone(), Node::Prim(HOLE_SYM.clone())))
            .chain(self.pattern_args.iterate_arguments()
            .map(|labelled_zid| (shared.zip_of_zid[labelled_zid.zid].clone(), Node::IVar(labelled_zid.ivar as i32)))).collect();

        fn helper(set: &mut ExprSet, curr_node: Idx, curr_zip: &mut Vec<ZNode>, zips: &[(Vec<ZNode>,Node)], shared: &SharedData) -> Idx {
            if let Some((_,e)) = zips.iter().find(|(zip,_)| zip == curr_zip) {
                return set.add(e.clone()); // current zip matches a hole
            }
            // no ivar zip match, so recurse
            match &shared.set[curr_node] {
                Node::Prim(p) => set.add(Node::Prim(p.clone())),
                Node::Var(v, tag) => set.add(Node::Var(*v, *tag)),
                Node::Lam(b, tag) => {
                    curr_zip.push(ZNode::Body);
                    let b_idx = helper(set, *b, curr_zip, zips, shared);
                    curr_zip.pop();
                    set.add(Node::Lam(b_idx, *tag))
                }
                Node::App(f,x) => {
                    curr_zip.push(ZNode::Func);
                    let f_idx = helper(set, *f, curr_zip, zips, shared);
                    curr_zip.pop();
                    curr_zip.push(ZNode::Arg);
                    let x_idx = helper(set, *x, curr_zip, zips, shared);
                    curr_zip.pop();
                    set.add(Node::App(f_idx,x_idx))
                }
                _ => unreachable!(),
            }
        }
        // we can pick any match location
        let idx = helper(&mut set, self.match_locations[0], &mut curr_zip, &zips, shared);
        ExprOwned { set, idx }
    }

    /// convert pattern to an Expr then with `hole_zid` highlighted in color with what we would expect it to expand to
    fn show_track_expansion(&self, hole_zid: ZId, shared: &SharedData) -> String {
        let mut expr = self.to_expr(shared);
        let expands_to = format!("{}",tracked_expands_to(self, hole_zid, shared)).magenta().bold().to_string();
        let replace_sentinel = Node::Prim("<REPLACE>".into());
        let idx = expr.immut().zip(&shared.zip_of_zid[hole_zid]).idx;
        expr.set[idx] = replace_sentinel;
        expr.to_string().replace("<REPLACE>", &expands_to)
    }
    pub fn info(&self, shared: &SharedData) -> String {
        format!("{}: utility_upper_bound={}, body_utility={}, match_locations={}, usages={}",self.to_expr(shared), self.utility_upper_bound, self.body_utility, self.match_locations.len(), self.match_locations.iter().map(|loc|shared.num_paths_to_node[*loc]).sum::<Cost>())
    }
}


/// the index of the empty zipper `[]` in the list of zippers
pub const EMPTY_ZID: ZId = 0;

/// an argument to an abstraction.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Arg {
    pub shifted_id: Idx, // post-shifting node - this tells you what the actual argument is
    pub unshifted_id: Idx, // tells you which node in the original corpus this was before it was possibly shifted
    pub shift: i32, // how much was it shifted?
    pub cost: Cost,
    pub expands_to: ExpandsTo,
}

/// The heap item used for heap-based worklists. Holds a pattern
#[derive(Debug,Clone, Eq, PartialEq)]
pub struct HeapItem {
    key: Cost,
    pattern: Pattern,
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
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
            // key: pattern.body_utility * pattern.match_locations.iter().map(|loc|num_paths_to_node[loc]).sum::<Cost>(),
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
    utility_pruning_cutoff: Cost,
    active_threads: FxHashSet<std::thread::ThreadId>, // list of threads currently holding worklist items
}

/// All the data shared among threads, mostly read-only
/// except for the mutexes
#[derive(Debug)]
pub struct SharedData {
    pub crit: Mutex<CriticalMultithreadData>,
    pub programs: Vec<ExprOwned>,
    pub arg_of_zid_node: Vec<FxHashMap<Idx,Arg>>,
    pub cost_fn: ExprCost,
    pub analyzed_free_vars: AnalyzedExpr<FreeVarAnalysis>,
    pub sym_var_info: Option<SymvarInfo>,
    pub analyzed_ivars: AnalyzedExpr<IVarAnalysis>,
    pub analyzed_cost: AnalyzedExpr<ExprCost>,
    pub corpus_span: Span,
    pub roots: Vec<Idx>,
    pub zids_of_node: FxHashMap<Idx,Vec<ZId>>,
    pub zip_of_zid: Vec<Vec<ZNode>>,
    pub zid_of_zip: FxHashMap<Vec<ZNode>, ZId>,
    pub extensions_of_zid: Vec<ZIdExtension>,
    pub set: ExprSet,
    pub num_paths_to_node: Vec<Cost>,
    pub num_paths_to_node_by_root_idx: Vec<Vec<Cost>>,
    num_paths_to_node_by_root_idx_sparse: Vec<Vec<(usize, Cost)>>,
    pub tdfa_global_annotations: Option<TDFAGlobalAnnotations>,
    pub tasks_of_node: Vec<FxHashSet<usize>>,
    pub task_name_of_task: Vec<String>,
    pub task_of_root_idx: Vec<usize>,
    pub root_idxs_of_task: Vec<Vec<usize>>,
    pub cost_of_node_all: Vec<Cost>,
    pub init_cost: Cost,
    pub init_cost_weighted: Cost,
    pub init_cost_by_root_idx: Vec<Cost>,
    pub init_cost_by_root_idx_weighted: Vec<f32>,
    pub weight_by_root_idx: Vec<f32>,
    pub first_train_cost: Cost,
    pub stats: Mutex<Stats>,
    pub cfg: CompressionStepConfig,
    pub multistep_cfg: MultistepCompressionConfig,
    pub tracking: Option<Tracking>,
    pub fused_lambda_tags: Option<FxHashSet<Tag>>,
}

pub fn invalid_metavar_location(shared : &SharedData, node: Idx) -> bool {
    tdfa_invalid_metavar(&shared.tdfa_global_annotations, node) || fused_lambda_location(&shared.set, &shared.fused_lambda_tags, node)
}

fn invalid_match_location(set : &ExprSet, fused_lambda_tags: &Option<FxHashSet<Tag>>, tdfa_global_annotations: &Option<TDFAGlobalAnnotations>, node: Idx) -> bool {
    tdfa_invalid_root(tdfa_global_annotations, node) || fused_lambda_location(set, fused_lambda_tags, node)
}

fn fused_lambda_location(set : &ExprSet, fused_lambda_tags: &Option<FxHashSet<Tag>>, node: Idx) -> bool {
    if let Some(fused_lambda_tags) = fused_lambda_tags {
        if let Node::Lam(_, tag) = &set[node] {
            if fused_lambda_tags.contains(tag) {
                return true
            }
        }
    }
    false
}

/// Used for debugging tracking information
#[derive(Debug)]
pub struct Tracking {
    pub expr: ExprOwned,
    pub zids_of_ivar: Vec<Vec<ZId>>,
}

impl CriticalMultithreadData {
    /// Create a new mutable multithread data struct with
    /// a worklist that just has a single hole on it
    fn new(donelist: Vec<FinishedPattern>, worklist: BinaryHeap<HeapItem>, cfg: &CompressionStepConfig) -> Self {        
        let mut res = CriticalMultithreadData {
            donelist,
            worklist,
            // we allow negative utilities in follow_prune case
            utility_pruning_cutoff: if !cfg.follow_prune { 0 } else { Cost::MIN },
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
        self.donelist.sort_unstable_by(|a,b| (b.utility,&b.pattern.pattern_args).cmp(&(a.utility,&a.pattern.pattern_args)));
        self.donelist.truncate(cfg.inv_candidates);
        // the cutoff is the lowest utility
        // we allow negative utilities in follow_prune case
        let default_bound = if !cfg.follow_prune { 0 } else { Cost::MIN };
        self.utility_pruning_cutoff = if cfg.no_opt_upper_bound { default_bound } else { std::cmp::max(0,self.donelist.last().map(|x|x.utility).unwrap_or(0)) };
    }
}



/// At the end of the day we convert our Inventions into InventionExprs to make
/// them standalone without needing to carry the EGraph around to figure out what
/// the body Idx points to.
#[derive(Debug, Clone)]
pub struct Invention {
    pub body: ExprOwned, // invention body (not wrapped in lambdas)
    pub arity: usize,
    pub name: String,
    pub variable_types: Vec<VariableType>, // type of each variable
}

impl Invention {
    pub fn new(body: ExprOwned, arity: usize, name: &str, variable_types: Vec<VariableType>) -> Self {
        Self { body, arity, name: String::from(name), variable_types }
    }
    pub fn from_string(name: &str, body: &str, variable_types: Option<Vec<VariableType>>) -> Self {
        let mut set = ExprSet::empty(Order::ChildFirst, false, false);
        let idx = set.parse_extend(body).unwrap();
        let body = ExprOwned { set, idx };
        let arity = AnalyzedExpr::new(IVarAnalysis).analyze_get(body.immut()).iter().max().map(|x|*x as usize + 1).unwrap_or(0);
        Self { body, arity, name: String::from(name), variable_types: variable_types.unwrap_or_else(|| vec![VariableType::Metavar; arity]) }
    }

    pub fn to_tracking(self, zid_of_zip: &FxHashMap<Vec<ZNode>, ZId>) -> Option<Tracking> {
        let zids_of_ivar = zids_of_ivar_of_expr(&self.body, zid_of_zip)?;
        Some(Tracking { expr: self.body, zids_of_ivar })
    }

    pub fn from_compression_output(output: &Value) -> Invention {
        let arity = output["arity"].as_u64().unwrap() as usize;
        Invention {
            body: {
                let mut set = ExprSet::empty(Order::ChildFirst, false, false);
                let idx = set.parse_extend(output["body"].as_str().unwrap()).unwrap();
                ExprOwned::new(set, idx)
            },
            arity,
            name: output["name"].as_str().unwrap().parse().unwrap(),
            variable_types: if let Some(var_types) = output.get("variable_types") {
                var_types.as_array().unwrap().iter().map(|v| VariableType::from_str(v.as_str().unwrap()).unwrap()).collect()
            } else {
                vec![VariableType::Metavar; arity]
            }
        }
    }
}

impl Display for Invention {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "[{} arity={}: {}", self.name, self.arity, self.body.immut())?;
        if self.variable_types.iter().any(|x| *x != VariableType::Metavar) {
            write!(f, ", variable_types={}", self.variable_types.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" "))?;
        }
        write!(f, "]")
    }
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
    pub useless_abstract_fired: usize,
    pub force_multiuse_fired: usize,
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
                    (hole_idx, pattern.match_locations.iter().filter(|loc|shared.arg_of_zid_node[*hole_zid][loc].expands_to.is_app()).count()))
                        .min_by_key(|x|x.1).unwrap().0
            }
            HoleChoice::MaxCost => {
                pattern.holes.iter().enumerate().map(|(hole_idx,hole_zid)|
                    (hole_idx, pattern.match_locations.iter().map(|loc|shared.arg_of_zid_node[*hole_zid][loc].cost).sum::<Cost>()))
                        .max_by_key(|x|x.1).unwrap().0
            }
            HoleChoice::MinCost => {
                pattern.holes.iter().enumerate().map(|(hole_idx,hole_zid)|
                    (hole_idx, pattern.match_locations.iter().map(|loc|shared.arg_of_zid_node[*hole_zid][loc].cost).sum::<Cost>()))
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

/// tells you which zipper if any you would get if you extended the depth
/// of whatever the current zipper is in any of these directions.
#[derive(Clone,Debug)]
pub struct ZIdExtension {
    pub body: Option<ZId>,
    pub arg: Option<ZId>,
    pub func: Option<ZId>,
}

/// empties worklist_buf and donelist_buf into the shared worklist while holding the mutex, updates
/// the donelist and cutoffs, and grabs and returns a new worklist item along with new cutoff bounds.
//#[inline(never)]
fn get_worklist_item(
    worklist_buf: &mut Vec<HeapItem>,
    donelist_buf: &mut Vec<FinishedPattern>,
    shared: &Arc<SharedData>,
) -> Option<(Vec<Pattern>,Cost)> {

    // * MULTITHREADING: CRITICAL SECTION START *
    // take the lock, which will be released immediately when this scope exits
    let mut shared_guard = shared.crit.lock();
    let mut crit: &mut CriticalMultithreadData = shared_guard.deref_mut();
    let old_best_utility = crit.donelist.first().map(|x|x.utility).unwrap_or(0);
    let old_donelist_len: usize = crit.donelist.len();
    let old_utility_pruning_cutoff = crit.utility_pruning_cutoff;
    // drain from donelist_buf into the actual donelist
    crit.donelist.extend(donelist_buf.drain(..).filter(|done| done.utility > old_utility_pruning_cutoff));
    if !shared.cfg.no_stats { shared.stats.lock().deref_mut().finished += crit.donelist.len() - old_donelist_len; };
    // sort + truncate + update utility_pruning_cutoff
    crit.update(&shared.cfg); // this also updates utility_pruning_cutoff

    if shared.cfg.verbose_best && crit.donelist.first().map(|x|x.utility).unwrap_or(0) > old_best_utility {

        let new_expected_cost = shared.first_train_cost - crit.donelist.first().unwrap().compressive_utility + crit.donelist.first().unwrap().to_expr(shared).cost(&shared.cost_fn) as Cost;
        let trainratio = shared.first_train_cost as f64 / new_expected_cost as f64;
        if !shared.cfg.quiet { println!("{} @ step={} util={} trainratio={:.2} for {}", "[new best utility]".blue(), shared.stats.lock().deref_mut().worklist_steps, crit.donelist.first().unwrap().utility, trainratio, crit.donelist.first().unwrap().info(shared)) }
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
    // if !shared.cfg.quiet { println!("worklist len: {}", crit.worklist.len()) }

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
            if !shared.cfg.no_stats && shared.cfg.print_stats > 0 && shared.stats.lock().deref_mut().worklist_steps % shared.cfg.print_stats == 0 && !shared.cfg.quiet { println!("{:?} \n\t@ [bound={}; uses={}] chose: {}",shared.stats.lock().deref_mut(),   original_pattern.utility_upper_bound, original_pattern.match_locations.iter().map(|loc| shared.num_paths_to_node[*loc]).sum::<Cost>(), original_pattern.to_expr(&shared)) };

            if shared.cfg.verbose_worklist && !shared.cfg.quiet { println!("[bound={}; uses={}] chose: {}", original_pattern.utility_upper_bound, original_pattern.match_locations.iter().map(|loc| shared.num_paths_to_node[*loc]).sum::<Cost>(), original_pattern.to_expr(&shared)) }

            // choose which hole we're going to expand
            let hole_idx: usize = shared.cfg.hole_choice.choose_hole(&original_pattern, &shared);

            // pop that hole from the list of holes
            let mut holes_after_pop: Vec<ZId> = original_pattern.holes.clone();
            let hole_zid: ZId = holes_after_pop.remove(hole_idx);

            // get the hashmap for looking up the Arg struct for this hole based on match location. The Arg
            // struct has a bunch of info about the hole, including what it expands into at each match location
            let arg_of_loc = &shared.arg_of_zid_node[hole_zid];

            // sort the match locations by node type (ie what theyll expand into) so that we can do a group_by() on
            // node type in order to iterate over all the different expansions
            // We also sort secondarily by `loc` to ensure each groupby subsequence has the locations in sorted order
            let mut match_locations = original_pattern.match_locations.clone();
            match_locations.sort_by_cached_key(|loc| (&arg_of_loc[loc].expands_to, *loc));

            let syntactic_expansions = get_syntactic_expansions(arg_of_loc, match_locations, &shared.sym_var_info);
            let ivars_expansions = get_ivars_expansions(&original_pattern, arg_of_loc, hole_zid, &shared);

            let mut found_tracked = false;
            // for each way of expanding the hole...

            'expansion:
                for (expands_to, locs) in syntactic_expansions.into_iter().chain(ivars_expansions.into_iter())
            {
                // for debugging
                let tracked = original_pattern.tracked && expands_to == tracked_expands_to(&original_pattern, hole_zid, &shared);
                if tracked { found_tracked = true; }
                if shared.cfg.follow_prune && !tracked { continue 'expansion; }


                if should_prune_single_use(&shared, &locs) {
                    if !shared.cfg.no_stats { shared.stats.lock().deref_mut().single_use_fired += 1; }
                    continue 'expansion;
                }

                if should_prune_single_task(&shared, &locs) {
                    if !shared.cfg.no_stats { shared.stats.lock().deref_mut().single_task_fired += 1; }
                    if tracked && !shared.cfg.quiet { println!("{} single task pruned when expanding {} to {}", "[TRACK]".red().bold(), original_pattern.to_expr(&shared), zipper_replace(original_pattern.to_expr(&shared), &shared.zip_of_zid[hole_zid], Node::Prim(format!("<{expands_to}>").into()))) }
                    continue 'expansion;
                }

                // Pruning (FREE VARS): if an invention has free variables in the body then it's not a real function and we can discard it
                // Here we just check if our expansion just yielded a variable, and if that is bound based on how many lambdas there are above it.
                if expands_to.free_variable(shared.zip_of_zid[hole_zid].iter().filter(|znode|**znode == ZNode::Body).count()) {
                    if !shared.cfg.no_stats { shared.stats.lock().deref_mut().free_vars_fired += 1; };
                    if tracked && !shared.cfg.quiet { println!("{} pruned by free var in body when expanding {} to {}", "[TRACK]".red().bold(), original_pattern.to_expr(&shared), original_pattern.show_track_expansion(hole_zid, &shared)) }
                    continue 'expansion; // free var
                }

                // update the body utility
                let body_utility = original_pattern.body_utility + expands_to.local_expansion_utility(&shared);

                // update the upper bound
                let util_upper_bound: Cost = utility_upper_bound(&locs, body_utility, &shared.cost_fn, &shared.cost_of_node_all, &shared.num_paths_to_node, &shared.cfg);
                assert!(util_upper_bound <= original_pattern.utility_upper_bound, "{} > {}", util_upper_bound, original_pattern.utility_upper_bound);

                // Pruning (UPPER BOUND): if the upper bound is less than the best invention we've found so far (our cutoff), we can discard this pattern
                if !shared.cfg.no_opt_upper_bound && util_upper_bound <= weak_utility_pruning_cutoff {
                    if !shared.cfg.no_stats { shared.stats.lock().deref_mut().upper_bound_fired += 1; };
                    if tracked && !shared.cfg.quiet { println!("{} upper bound ({} < {}) pruned when expanding {} to {}", "[TRACK]".red().bold(), util_upper_bound, weak_utility_pruning_cutoff, original_pattern.to_expr(&shared), original_pattern.show_track_expansion(hole_zid, &shared)) }
                    continue 'expansion; // too low utility
                }

                // assert!(shared.cfg.no_opt_upper_bound || !holes_after_pop.is_empty() || !original_pattern.arg_choices.is_empty() || expands_to.has_holes() || expands_to.is_var(),
                        // "unexpected arity 0 invention: upper bounds + priming with arity 0 inventions should have prevented this");
                // assert!(shared.cfg.no_opt_upper_bound || (locs.len() > 1 || !shared.egraph[locs[0]].data.free_vars.is_empty()),
                //         "single-use pruning doesn't seem to be happening, it should be an automatic side effect of upper bounds + priming with arity zero inventions (as long as they dont have free vars)\n{}\n{}\n{}\n{}\n{}", original_pattern.to_expr(&shared), extract(locs[0], &shared.egraph), expands_to,  util_upper_bound, weak_utility_pruning_cutoff);

                // add any new holes to the list of holes
                let mut holes = holes_after_pop.clone();
                expands_to.syntactic_expansion(&shared.extensions_of_zid[hole_zid], |zid, _| holes.push(zid));

                // update arg_choices and possibly variables if a new ivar was added
                let mut pattern_args = original_pattern.pattern_args.clone();
                expands_to.add_variables(hole_zid, &mut pattern_args);

                if original_pattern.pattern_args.is_useless_abstract(&shared, &locs) {
                    continue 'expansion;
                }
                
                if  original_pattern.pattern_args.is_redundant_argument(&shared, &locs) {
                    if tracked && !shared.cfg.quiet { println!("{} force multiuse pruned when expanding {} to {}", "[TRACK]".red().bold(), original_pattern.to_expr(&shared), original_pattern.show_track_expansion(hole_zid, &shared)) }
                    continue 'expansion;
                }


                // build our new pattern with all the variables we've just defined. Copy in the argchoices and prefixes
                // from the old pattern.
                let new_pattern = Pattern {
                    holes,
                    pattern_args,
                    match_locations: locs,
                    utility_upper_bound: util_upper_bound,
                    body_utility,
                    tracked
                };

                // new_pattern.utility_upper_bound = utility_upper_bound_with_conflicts(&new_pattern, body_utility_no_refinement + refinement_body_utility, &shared);
                // // branch and bound again
                // if !shared.cfg.no_opt_upper_bound && new_pattern.utility_upper_bound <= weak_utility_pruning_cutoff {
                //     if !shared.cfg.no_stats { shared.stats.lock().deref_mut().conflict_upper_bound_fired += 1; };
                //     if tracked { if !shared.cfg.quiet { println!("{} upper bound ({} < {}) pruned when expanding {} to {}", "[TRACK]".red().bold(), util_upper_bound, weak_utility_pruning_cutoff, original_pattern.to_expr(&shared), original_pattern.show_track_expansion(hole_zid, &shared)) } }
                //     continue 'expansion; // too low utility
                // }

                if new_pattern.holes.is_empty() {
                    // it's a finished pattern

                    let mut finished_pattern = FinishedPattern::new(new_pattern, &shared);

                    if !shared.cfg.no_stats { shared.stats.lock().calc_final_utility += 1; };

                    // Pruning (UPPER BOUND): here we use just compressive_utility to prune before calling the expensive
                    // inverse_argument_capture(). Note that this pruning is okay because compressive utility itself is an upper bound
                    // on total utility.
                    if finished_pattern.compressive_utility <= weak_utility_pruning_cutoff {
                        continue 'expansion // todo could add a tracked{} printing thing here
                    }

                    if !shared.cfg.no_stats { shared.stats.lock().calc_unargcap += 1; };
                    inverse_argument_capture(&mut finished_pattern, &shared.cfg, &shared.zip_of_zid, &shared.arg_of_zid_node, &shared.extensions_of_zid, &shared.set, &shared.analyzed_ivars, &shared.cost_fn);

                    // Pruning (UPPER BOUND)
                    if finished_pattern.utility <= weak_utility_pruning_cutoff {
                        continue 'expansion // todo could add a tracked{} printing thing here
                    }

                    if !shared.cfg.no_stats { shared.stats.lock().donelist_push += 1; };

                    if shared.cfg.rewrite_check {
                        // run rewriting just to make sure the assert in it passes
                        let rw_fast = rewrite_fast(&finished_pattern, &shared, &Node::Prim("fake_inv".into()), &shared.cost_fn);
                        let (rw_slow, _, _) = rewrite_with_inventions(&shared.programs.iter().map(|p|p.to_string()).collect::<Vec<_>>(), &[finished_pattern.clone().to_invention("fake_inv", &shared)], &shared.multistep_cfg);
                        for (fast,slow) in rw_fast.iter().zip(rw_slow.iter()) {
                            assert_eq!(fast.to_string(), slow.to_string());
                        }
                    }

                    if tracked && !shared.cfg.quiet { println!("{} pushed {} to donelist (util: {})", "[TRACK:DONE]".green().bold(), finished_pattern.to_expr(&shared), finished_pattern.utility) }

                    if shared.cfg.inv_candidates == 1 && finished_pattern.utility > weak_utility_pruning_cutoff {
                        // if we're only looking for one invention, we can directly update our cutoff here
                        weak_utility_pruning_cutoff = finished_pattern.utility;
                    }

                    donelist_buf.push(finished_pattern);

                } else {
                    // it's a partial pattern so just add it to the worklist
                    if tracked && !shared.cfg.quiet { println!("{} pushed {} to work list (bound: {})", "[TRACK]".green().bold(), original_pattern.show_track_expansion(hole_zid, &shared), new_pattern.utility_upper_bound) }
                    worklist_buf.push(HeapItem::new(new_pattern))
                }
            }

            if original_pattern.tracked && !found_tracked {
                // let new = format!("<{}>",tracked_expands_to(&original_pattern, hole_zid, &shared));
                // let mut s = original_pattern.to_expr(&shared).zipper_replace(&shared.zip_of_zid[hole_zid], &new ).to_string();
                // s = s.replace(&new, &new.clone().magenta().bold().to_string());
            if !shared.cfg.quiet { println!("{} pruned when expanding because there were no match locations for the target expansion of {} to {}", "[TRACK]".red().bold(), original_pattern.to_expr(&shared), original_pattern.show_track_expansion(hole_zid, &shared)) }
            }
        
        }
    }

}

fn should_prune_single_use(shared: &SharedData, locs: &[Idx]) -> bool {
    // Pruning (SINGLE USE): prune inventions that only match at a single unique (structurally hashed) subtree. This only applies if we
    // also are priming with arity 0 inventions. Basically if something only matches at one subtree then the best you can
    // do is the arity zero invention which is the whole subtree, and since we already primed with arity 0 inventions we can
    // prune here. The exception is when there are free variables so arity 0 wouldn't have applied.
    // Also, note that upper bounding + arity 0 priming does nearly perfectly handle this already, but there are cases where
    // you can't improve your structure penalty bound enough to catch everything hence this separate single_use thing.
    !shared.cfg.no_opt_single_use && !shared.cfg.no_opt_arity_zero && locs.len()  == 1 && shared.analyzed_free_vars[locs[0]].is_empty() && shared.sym_var_info.as_ref().is_none_or(|svi| !svi.contains_symbols(locs[0]))
}

fn should_prune_single_task(shared: &SharedData, locs: &[usize]) -> bool {
    // Pruning (SINGLE TASK): prune inventions that are only used in one task
    !shared.cfg.allow_single_task
            && locs.iter().all(|node| shared.tasks_of_node[*node].len() == 1)
            && locs.iter().all(|node| shared.tasks_of_node[locs[0]].iter().next() == shared.tasks_of_node[*node].iter().next())
}

pub fn should_prune_single_use_or_task(shared: &SharedData, locs: &[Idx]) -> bool {
    should_prune_single_use(shared, locs) || should_prune_single_task(shared, locs)
}


/// A finished abstraction
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FinishedPattern {
    pub pattern: Pattern,
    pub utility: Cost,
    pub compressive_utility: Cost,
    pub util_calc: UtilityCalculation,
    pub arity: usize,
    pub usages: Cost,
}

impl FinishedPattern {
    //#[inline(never)]
    fn new(pattern: Pattern, shared: &SharedData) -> Self {
        let arity = pattern.pattern_args.arity();
        let usages = pattern.match_locations.iter().map(|loc| shared.num_paths_to_node[*loc]).sum();
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
            let rewritten: Vec<ExprOwned> = rewrite_fast(&res, shared, &Node::Prim("fake_inv".into()), &shared.cost_fn);
            res.compressive_utility = shared.init_cost - shared.root_idxs_of_task.iter().map(|root_idxs|
                root_idxs.iter().map(|idx| rewritten[*idx].cost(&shared.cost_fn) as Cost).min().unwrap()
            ).sum::<Cost>();
            // res.compressive_utility = shared.init_cost - rewritten.iter().map(|e|e.cost()).sum::<Cost>();
            res.util_calc.util = res.compressive_utility;
            res.utility = res.compressive_utility + noncompressive_utility;
        }
        res
    }
    // convert finished invention to an Expr
    pub fn to_expr(&self, shared: &SharedData) -> ExprOwned {
        self.pattern.to_expr(shared)
    }
    pub fn to_invention(&self, name: &str, shared: &SharedData) -> Invention {
        Invention::new(self.to_expr(shared), self.arity, name, self.pattern.pattern_args.variable_types())
    }
    pub fn info(&self, shared: &SharedData) -> String {
        format!("{} -> finished: utility={}, compressive_utility={}, arity={}, usages={}",self.pattern.info(shared), self.utility, self.compressive_utility, self.arity, self.usages)
    }

}
// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
// struct Refinement {
//     refined_subtree: Idx, // the thing you can refine out
//     uses: HashMap<Idx,i32>, // map from loc to number of times it's used
//     refined_subtree_cost: i32, // the compressive utility gained by refining it
// }


/// figure out all the N^2 zippers from choosing any given node and then choosing a descendant and returning the zipper from
/// the node to the descendant. We also collect a bunch of other useful stuff like the argument you would get if you abstracted
/// the descendant and introduced an invention rooted at the ancestor node.
#[allow(clippy::type_complexity)]
//#[inline(never)]
fn get_zippers(
    corpus_span: &Span,
    analyzed_cost: &AnalyzedExpr<ExprCost>,
    set: &mut ExprSet,
    analyzed_free_vars: &mut AnalyzedExpr<FreeVarAnalysis>,
) -> (FxHashMap<Vec<ZNode>, ZId>, Vec<Vec<ZNode>>, Vec<FxHashMap<Idx,Arg>>, FxHashMap<Idx,Vec<ZId>>,  Vec<ZIdExtension>) {

    let mut zid_of_zip: FxHashMap<Vec<ZNode>, ZId> = Default::default();
    let mut zip_of_zid: Vec<Vec<ZNode>> = Default::default();
    let mut arg_of_zid_node: Vec<FxHashMap<Idx,Arg>> = Default::default();
    let mut zids_of_node: FxHashMap<Idx,Vec<ZId>> = Default::default();

    zid_of_zip.insert(vec![], EMPTY_ZID);
    zip_of_zid.push(vec![]);
    arg_of_zid_node.push(FxHashMap::default());
    
    // loop over all nodes in all programs in bottom up order
    for idx in corpus_span.clone() {
        // if !shared.cfg.quiet { println!("processing Idx={}: {}", treenode, extract(*treenode, egraph) ) }

        
        // any node can become the identity function (the empty zipper with itself as the arg)
        let mut zids: Vec<ZId> = vec![EMPTY_ZID];

        // clone to appease the borrow checker
        let node = set.get(idx).node().clone();

        arg_of_zid_node[EMPTY_ZID].insert(idx,
            Arg { shifted_id: idx, unshifted_id: idx, shift: 0, cost: analyzed_cost[idx] as Cost, expands_to: expands_to_of_node(&node) });

        match node {
            Node::IVar(_) => { unreachable!() }
            Node::Var(_, _) | Node::Prim(_) => {},
            Node::App(f,x) => {
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
                    arg_of_zid_node[*zid].insert(idx, arg);
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
                    arg_of_zid_node[*zid].insert(idx, arg);

                }
            },
            Node::Lam(b, _) => {
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

                    if !analyzed_free_vars.analyze_get(set.get(arg.shifted_id)).is_empty() {
                        // the arg has free vars so we should actually downshift it by 1
                        if analyzed_free_vars[arg.shifted_id].contains(&0) {
                            // furthermore one of those vars is a 0 then it will get shifted to -1, so we handle that slightly specially
                            // by inserting an IVar to indicate this

                            // how many lambdas are along this zipper? (including most recent one)
                            let depth_root_to_arg = zip.iter().filter(|x| **x == ZNode::Body).count() as i32;

                            // find all pointers to $0 (this is the `init_depth` parameter) and replace then with #(num_lams - 1) that is
                            // point past all lambdas except the newly added one. For example if there were no lambdas other than the
                            // newly added one this would be num_lams=1 so it'd be #0.
                            arg.shifted_id = insert_arg_ivars(&mut set.get_mut(arg.shifted_id), depth_root_to_arg-1, 0, analyzed_free_vars);
                        }
                        arg.shifted_id = set.get_mut(arg.shifted_id).shift(-1, 0, analyzed_free_vars);
                        arg.shift -= 1;
                    }
                    arg_of_zid_node[*zid].insert(idx, arg);
                }
            },
        }
        zids_of_node.insert(idx, zids);
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
    pub set: ExprSet,
    pub inv: Invention,
    pub rewritten: Vec<ExprOwned>,
    pub rewritten_dreamcoder: Option<Vec<String>>,
    pub done: FinishedPattern,
    pub expected_cost: Cost,
    pub final_cost: Cost,
    pub multiplier: f64,
    pub multiplier_wrt_orig: f64,
    pub uses: Cost,
    pub use_exprs: Vec<Idx>,
    pub use_args: Vec<Vec<Idx>>,
    pub dc_inv_str: String,
    pub initial_cost: Cost,
    pub name_mapping: Vec<(String,String)>,
    pub dc_comparison_millis: Option<usize>,
    pub tdfa_annotation: Option<TDFAInventionAnnotation>,
}

impl CompressionStepResult {
    fn new(done: FinishedPattern, inv_name: &str, shared: &mut SharedData, very_first_cost: Cost, name_mapping: &[(String,String)], dc_comparison_millis: Option<usize>) -> Self {

        let inv = done.to_invention(inv_name, shared);
        let rewritten = rewrite_fast(&done, shared, &Node::Prim(inv.name.clone().into()), &shared.cost_fn);

        let expected_cost = shared.init_cost_weighted - done.compressive_utility;
        // let final_cost = rewritten.cost();
        let final_cost = shared.root_idxs_of_task.iter().map(|root_idxs|
            root_idxs.iter().map(|idx| (rewritten[*idx].cost(&shared.cost_fn) as f32 * shared.weight_by_root_idx[*idx]).round() as Cost).min().unwrap()
        ).sum::<Cost>();
        if expected_cost != final_cost && !shared.cfg.quiet { println!("*** expected cost {expected_cost} != final cost {final_cost}") }
        let multiplier = shared.init_cost_weighted as f64 / final_cost as f64;
        let multiplier_wrt_orig = very_first_cost as f64 / final_cost as f64;
        let uses = done.usages;
        let use_exprs: Vec<Idx> = done.pattern.match_locations.clone();
        let use_args: Vec<Vec<Idx>> = done.pattern.match_locations.iter().map(|node|
            done.pattern.pattern_args.use_args(shared, node)).collect();
        

        // dreamcoder compatability
        let dc_inv_str: String = dc_inv_str(&inv, name_mapping);
        // Rewrite to dreamcoder syntax with all past invention
        // we rewrite "inv1)" and "inv1 " instead of just "inv1" because we dont want to match on "inv10"

        // Combine the past_invs with the existing dreamcoder inventions.
        let mut name_mapping = name_mapping.to_vec();
        name_mapping.push((inv.name.clone(), dc_inv_str.clone()));
        

        let rewritten_dreamcoder: Option<Vec<String>> = if !shared.cfg.rewritten_dreamcoder { None } else {
            Some(rewritten.iter().map(|p|{
            let mut res: String = p.to_string();
            for (name, anonymous) in &name_mapping {
                res = replace_prim_with(&res, name, anonymous);
            }

            // Now go ahead and replace the current invention.
            res = replace_prim_with(&res, inv_name, &dc_inv_str);
            res = res.replace("(lam ","(lambda ");
            res
        }).collect())};

        let tdfa_annotation = TDFAInventionAnnotation::from_pattern(&done.pattern, shared);

        CompressionStepResult {
            set: shared.set.clone(),
            inv,
            rewritten,
            rewritten_dreamcoder,
            done,
            expected_cost,
            final_cost,
            multiplier,
            multiplier_wrt_orig,
            uses,
            use_exprs,
            use_args,
            dc_inv_str,
            initial_cost: shared.init_cost,
            name_mapping,
            dc_comparison_millis,
            tdfa_annotation,
        }
    }
    pub fn json(&self, cfg: &CompressionStepConfig) -> serde_json::Value {        
        let all_uses: Vec<serde_json::Value> = {
            let use_exprs: Vec<String> = self.use_exprs.iter().map(|expr| self.set.get(*expr).to_string()).collect();
            let use_args: Vec<String> = self.use_args.iter().map(|args| format!("{} {}", self.inv.name, args.iter().map(|expr| self.set.get(*expr).to_string()).collect::<Vec<String>>().join(" "))).collect();
            use_exprs.iter().zip(use_args.iter()).sorted().map(|(expr,args)| json!({args: expr})).collect()
        };

        let rewritten = if !cfg.rewritten_intermediates { None } else { Some(self.rewritten.iter().map(|p| p.to_string()).collect::<Vec<String>>()) };
        let rewritten_dreamcoder = if !cfg.rewritten_intermediates { &None } else { &self.rewritten_dreamcoder };

        let variable_types = self.inv.variable_types.iter().map(|ty| ty.to_string()).collect::<Vec<_>>();

        json!({            
            "body": self.inv.body.to_string(),
            "dreamcoder": self.dc_inv_str,
            "arity": self.inv.arity,
            "name": self.inv.name,
            "utility": self.done.utility,
            "final_cost": self.final_cost,
            "compression_ratio": self.multiplier,
            "cumulative_compression_ratio": self.multiplier_wrt_orig,
            "num_uses": self.uses,
            "rewritten": rewritten,
            "rewritten_dreamcoder": rewritten_dreamcoder,
            "uses": all_uses,
            "dc_comparison_millis": self.dc_comparison_millis,
            "tdfa_annotation": self.tdfa_annotation,
            "variable_types": variable_types,
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
    match_locations: &[Idx],
    body_utility_lower_bound: Cost,
    cost_fn: &ExprCost,
    cost_of_node_all: &[Cost],
    num_paths_to_node: &[Cost],
    cfg: &CompressionStepConfig,
) -> Cost {
    compressive_utility_upper_bound(match_locations, cost_fn, cost_of_node_all, num_paths_to_node)
        + noncompressive_utility_upper_bound(body_utility_lower_bound, cfg)
}

/// This utility is just for any utility terms that we care about that don't directly correspond
/// to changes in size that come from rewriting with an invention. Currently this is just the
/// size of the abstraction itself
//#[inline(never)]
fn noncompressive_utility(
    body_utility: Cost,
    cfg: &CompressionStepConfig,
) -> Cost {
    if cfg.no_other_util { return 0; }
    // this is a bit like the structure penalty from dreamcoder except that
    // that penalty uses inlined versions of nested inventions.
    // 0
    - (body_utility as f32 * cfg.structure_penalty) as Cost
}

/// This takes a partial invention and gives an upper bound on the maximum
/// compressive_utility() that any completed offspring of this partial invention could have.
//#[inline(never)]
fn compressive_utility_upper_bound(
    match_locations: &[Idx],
    cost_fn: &ExprCost,
    cost_of_node_all: &[Cost],
    num_paths_to_node: &[Cost],
) -> Cost {
    match_locations.iter().map(|node|
        std::cmp::max(0, cost_of_node_all[*node] - num_paths_to_node[*node] * cost_fn.compute_cost_new_prim() as Cost)
    ).sum::<Cost>()
    
    // shared.init_cost - shared.root_idxs_of_task.iter().map(|root_idxs|
    //     root_idxs.iter().map(|idx| shared.init_cost_by_root_idx[*idx] - adjusted_util_by_root_idx[*idx]).min().unwrap()
    // ).sum::<Cost>()
}


/// This takes a partial invention and gives an upper bound on the maximum
/// other_utility() that any completed offspring of this partial invention could have.
//#[inline(never)]
fn noncompressive_utility_upper_bound(
    _body_utility_lower_bound: Cost,
    cfg: &CompressionStepConfig,
) -> Cost {
    // 0
    if cfg.no_other_util { return 0; }
    // - body_utility_lower_bound
    0
    // safe bound: since structure_penalty is negative an upper bound is anything less negative or exact. Since
    // left_utility < body_utility we know that this will be a less negative bound.
    
}

pub fn compressive_utility_from_marginals(
    pattern: &Pattern,
    shared: &SharedData,
    marginal_util: Vec<Cost>
) -> UtilityCalculation {
    assert!(marginal_util.len() == pattern.match_locations.len(),
        "utility_of_loc_once should have the same length as match_locations, but got {} and {}",
        marginal_util.len(), pattern.match_locations.len());
    let unused_locations: FxHashSet<Idx> = marginal_util.iter().enumerate().filter_map(|(idx, util)|
        if *util > 0 {None} else {Some(pattern.match_locations[idx])}
    ).collect();
    let mut util_by_root = vec![0; shared.roots.len()];
    for (loc_idx, loc) in pattern.match_locations.iter().enumerate() {
        // for each match location, we add the utility of that location to the util_by_root for each root
        let util = marginal_util[loc_idx];
        if util > 0 {
            for (root_idx, num_paths) in shared.num_paths_to_node_by_root_idx_sparse[*loc].iter() {
                util_by_root[*root_idx] += util * num_paths;
            }
        }
    }

    let util = shared.init_cost_weighted - shared.root_idxs_of_task.iter().map(|root_idxs|
        root_idxs.iter().map(|idx| (shared.init_cost_by_root_idx_weighted[*idx] - util_by_root[*idx] as f32 * shared.weight_by_root_idx[*idx]).round() as Cost).min().unwrap()
    ).sum::<Cost>();

    UtilityCalculation {util, unused_locations}
}

//#[inline(never)]
fn compressive_utility(pattern: &Pattern, shared: &SharedData) -> UtilityCalculation {

    // * BASIC CALCULATION
    // Roughly speaking compressive utility is num_usages(invention) * size(invention), however there are a few extra
    // terms we need to take care of too.

    let Some(marginal_utilities) = get_utility_of_loc_once(pattern, shared) else {
        // All utilities were 0 or negative, so we should autoreject this pattern
        return UtilityCalculation { util: 0, unused_locations: Default::default() };
    };
    let self_intersects = can_self_unify(&pattern.pattern_args, shared, pattern.match_locations[0]);
    if self_intersects.is_empty() {
        // no conflicts, so we can just return the utility assuming no corrections
        return compressive_utility_from_marginals(pattern, shared, marginal_utilities);
    }
    let loc_to_idx = pattern.match_locations.iter().enumerate().map(|(idx, loc)| (*loc, idx)).collect::<FxHashMap<_,_>>();
    let mut marginal_utilities = marginal_utilities;
    // match locations are sorted by location in ExprSet, so are in bottom up order
    for (i, loc) in pattern.match_locations.iter().enumerate() {
        let mut alternate_utility = 0;
        for zid in &self_intersects {
            let child = shared.arg_of_zid_node[*zid][loc].unshifted_id;
            if loc_to_idx.contains_key(&child) {
                let idx = loc_to_idx[&child];
                alternate_utility += marginal_utilities[idx];
            }
        }
        marginal_utilities[i] -= alternate_utility;
        if marginal_utilities[i] < 0 {
            marginal_utilities[i] = 0; // we don't want to count negative utilities
        }
    }
    compressive_utility_from_marginals(pattern, shared, marginal_utilities)
}

//#[inline(never)]
fn get_utility_of_loc_once(pattern: &Pattern, shared: &SharedData) -> Option<Vec<Cost>> {
    // it costs a tiny bit to apply the invention, for example (app (app inv0 x) y) incurs a cost
    // of COST_TERMINAL for the `inv0` primitive and 2 * COST_NONTERMINAL for the two `app`s.
    // Also an extra COST_NONTERMINAL for each argument that is refined (for the lambda).
    // Returns None if all utilities are 0 or negative, in which case the pattern should be autorejected.
    let app_penalty = - (shared.cost_fn.compute_cost_new_prim() as Cost + shared.cost_fn.cost_app as Cost * pattern.pattern_args.arity() as Cost);

    // get a list of (ivar,usages-1) filtering out things that are only used once, this will come in handy for adding multi-use utility later
    let ivar_multiuses: Vec<(usize,Cost)> = pattern.pattern_args.multiuses();
    let mut at_least_one_util_valid = false;

    let results = pattern.match_locations.iter().map(|loc| {

        if pattern.pattern_args.has_free_ivars(shared, loc) {
            return 0; // set whole util to 0 for this loc, causing an autoreject
        }

        // if !shared.cfg.quiet { println!("calculating util of {}", extract(*loc, &shared.egraph)) }
        // compressivity of body (no refinement) minus slight penalty from the application
        let base_utility = pattern.body_utility + app_penalty;
        // if !shared.cfg.quiet { println!("base {}", base_utility) }

        // for each extra usage of an argument, we gain the cost of that argument as
        // extra utility. Note that it doesn't matter which
        // of the zids we use as long as it corresponds to the right ivar
        let multiuse_utility = ivar_multiuses.iter().map(|(zid,count)|
            count * shared.arg_of_zid_node[*zid][loc].cost as Cost
        ).sum::<Cost>();
        // if !shared.cfg.quiet { println!("multiuse {}", multiuse_utility) }

        let util = base_utility + multiuse_utility;
        if util > 0 {
            at_least_one_util_valid = true;
        }
        util
    }).collect();
    if !at_least_one_util_valid {
        // if all utilities are 0 or negative, we should autoreject this pattern
        return None;
    }
    Some(results)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UtilityCalculation {
    pub util: Cost,
    pub unused_locations: FxHashSet<Idx>, // if present, do not accept
}

// (not used in popl code - experimental)
pub fn inverse_delta(cost_once: Cost, usages: Cost, arg_uses: usize, cost_fn: &ExprCost) -> (Cost, Cost, Cost) {
    let compressive_delta = - (cost_once + cost_fn.cost_app as Cost) * usages;
    let noncompressive_delta = arg_uses as Cost * (cost_once - cost_fn.compute_cost_new_prim() as Cost) ;
    (compressive_delta,noncompressive_delta, compressive_delta+noncompressive_delta)
}

// (not used in popl code - experimental; always exists at the first return statement unless --inv-arg-cap is turned on)
#[allow(clippy::too_many_arguments)]
pub fn inverse_argument_capture(finished: &mut FinishedPattern, cfg: &CompressionStepConfig, zip_of_zid: &[Vec<ZNode>], arg_of_zid_node: &[FxHashMap<Idx,Arg>], extensions_of_zid: &[ZIdExtension], set: &ExprSet, analyzed_ivars: &AnalyzedExpr<IVarAnalysis>, cost_fn: &ExprCost) {
    if !cfg.inv_arg_cap || cfg.no_other_util {
        return
    }
    // panic!("inverse_argument_capture is disabled");
    if finished.arity >= cfg.max_arity {
        return
    }
    let _max_num_to_add = cfg.max_arity - finished.arity;

    while finished.arity < cfg.max_arity {
    let counts = use_counts(&finished.pattern, zip_of_zid, arg_of_zid_node, extensions_of_zid, set, analyzed_ivars);
    let possible_to_uninline = possible_to_uninline(counts, finished.usages, cost_fn);
    
    let best = possible_to_uninline.into_iter().max_by_key(|(delta, _compressive_delta, _noncompressive_delta, _cost, _zids)| *delta);
    
    if let Some((delta, compressive_delta, _noncompressive_delta, _cost, zids)) = best {
        let ivar = finished.arity;
        for zid in &zids {
            finished.pattern.pattern_args.add_var(ivar, *zid, VariableType::Metavar);
        }
        finished.compressive_utility += compressive_delta;
        finished.util_calc.util += compressive_delta;
        finished.utility += delta;
        finished.arity +=1;
        // println!("UNARG")
    } else {
        return
    }
    }
}

/// not used in popl code - experimental
fn possible_to_uninline(counts: FxHashMap<Idx, (Cost, Vec<usize>)>, finished_usages: Cost, cost_fn: &ExprCost) -> Vec<(Cost,Cost,Cost,Cost,Vec<ZId>)> {
    let possible_to_uninline = counts.values()
    // can only have a positive delta to compression if used more times within the abstraction than
    // there are usages of the abstraction in the corpus
    .filter(|(_,zids)| zids.len() > finished_usages as usize)
    // argument must be larger than the cost of adding the terminal for the new abstraction variable
    .filter(|(cost,_zids)| *cost > cost_fn.compute_cost_new_prim() as Cost)
    .filter_map(|(cost,zids)| {
        let (compressive_delta,noncompressive_delta, delta) = inverse_delta(*cost, finished_usages, zids.len(), cost_fn);
        if delta > 0 {
            Some((delta, compressive_delta, noncompressive_delta, *cost, zids.clone()))
        } else {
            None
        }
    });
    possible_to_uninline.collect()
}

/// not used in popl code - experimental
fn use_counts(pattern: &Pattern, zip_of_zid: &[Vec<ZNode>], arg_of_zid_node: &[FxHashMap<Idx,Arg>], extensions_of_zid: &[ZIdExtension], set: &ExprSet, analyzed_ivars: &AnalyzedExpr<IVarAnalysis>) -> FxHashMap<Idx,(Cost,Vec<ZId>)> {
    let mut curr_zip: Vec<ZNode> = vec![];
    let curr_zid: ZId = EMPTY_ZID;
    let zids = &pattern.pattern_args.iterate_arguments().cloned().collect::<Vec<LabelledZId>>();

    // map zids to zips with a bool thats true if this is a hole and false if its a future ivar
    let zips: Vec<Vec<ZNode>> = zids.iter()
        .map(|labelled_zid| zip_of_zid[labelled_zid.zid].clone()).collect();

    let mut counts: FxHashMap<Idx,(Cost,Vec<ZId>)> = Default::default();

    #[allow(clippy::too_many_arguments)]
    fn helper(curr_node: Idx, match_loc: Idx, curr_zip: &mut Vec<ZNode>, curr_zid: ZId, zips: &[Vec<ZNode>], zids: &[LabelledZId], arg_of_zid_node: &[FxHashMap<Idx,Arg>], extensions_of_zid: &[ZIdExtension], set: &ExprSet,  counts: &mut FxHashMap<Idx,(Cost,Vec<ZId>)>, analyzed_ivars: &AnalyzedExpr<IVarAnalysis>) {
        if zids.iter().any(|labelled| labelled.zid == curr_zid){
            return // current zip matches an arg
        }
        // if curr_zip is not a prefix of any arg zipper, then increment its count
        if zips.iter().all(|zip| !zip.starts_with(curr_zip)) {
            // also make sure its valid ie doesnt have any free ivars as ew do during normal checks
            let arg = arg_of_zid_node[curr_zid].get(&match_loc).unwrap();
            if analyzed_ivars[arg.shifted_id].is_empty() {
                counts.entry(arg.shifted_id)
                    .or_insert_with(||(arg.cost as Cost, vec![]))
                    .1.push(curr_zid);
            }
        }
        match &set[curr_node] {
            Node::Prim(_) => {},
            Node::Var(_, _) => {},
            Node::Lam(b, _) => {
                curr_zip.push(ZNode::Body);
                let new_zid = extensions_of_zid[curr_zid].body.unwrap();
                helper(*b, match_loc, curr_zip, new_zid, zips, zids,  arg_of_zid_node, extensions_of_zid, set, counts, analyzed_ivars);
                curr_zip.pop();
            }
            Node::App(f,x) => {
                curr_zip.push(ZNode::Func);
                let new_zid = extensions_of_zid[curr_zid].func.unwrap();
                helper(*f, match_loc, curr_zip, new_zid, zips, zids,  arg_of_zid_node, extensions_of_zid, set, counts, analyzed_ivars);
                curr_zip.pop();
                curr_zip.push(ZNode::Arg);
                let new_zid = extensions_of_zid[curr_zid].arg.unwrap();
                helper(*x, match_loc, curr_zip, new_zid, zips, zids,  arg_of_zid_node, extensions_of_zid, set, counts, analyzed_ivars);
                curr_zip.pop();
            }
            _ => unreachable!(),
        }
    }
    // we can pick any match location
    helper(pattern.match_locations[0], pattern.match_locations[0], &mut curr_zip, curr_zid, &zips, zids, arg_of_zid_node, extensions_of_zid, set, &mut counts, analyzed_ivars);
    counts
}

/// Multistep compression
pub fn multistep_compression_internal(
    train_programs: &[ExprOwned],
    tasks: Option<Vec<String>>,
    weights: Option<Vec<f32>>,
    name_mapping: Option<Vec<(String, String)>>,
    follow: Option<Vec<Invention>>,
    cfg: &MultistepCompressionConfig
) -> Vec<CompressionStepResult> {

    let mut rewritten: Vec<ExprOwned> = train_programs.to_vec();
    let mut step_results: Vec<CompressionStepResult> = Default::default();
    let cost_fn = &cfg.step.cost.expr_cost();

    let tstart = std::time::Instant::now();

    let mut cfg = cfg.clone();

    if let Some(follow) = &follow {
        assert_eq!(follow.len(), cfg.iterations);
        cfg.step.follow_prune = true;
        cfg.step.rewrite_check = false; // this will cause a loop
        if !cfg.verbose_rewrite{
            cfg.step.quiet = true;
        }
        cfg.step.no_opt();
    }

    let very_first_cost = min_cost(train_programs, &weights, &tasks, cost_fn);

    let tasks: Vec<String> = tasks.unwrap_or_else(|| {
        (0..train_programs.len())
            .map(|i| i.to_string())
            .collect()
    });

    let weights: Vec<f32> = weights.unwrap_or_else(|| vec![1.0; train_programs.len()]);

    let mut name_mapping = name_mapping.unwrap_or_default();




    for i in 0..cfg.iterations {
        if !cfg.step.quiet { println!("{}",format!("\n=======Iteration {i}=======").blue().bold()) }
        let (inv_name, follow_iter) = if let Some(follow) = &follow {
            (follow[i].name.clone(), Some(follow[i].clone()))
        } else {
            (format!("{}{}", cfg.abstraction_prefix, cfg.previous_abstractions + step_results.len()), cfg.step.follow.as_ref().map(|x| Invention::from_string("inv", x, cfg.step.follow_types.clone())))
        };

        // call actual compression
        let res: Vec<CompressionStepResult> = compression_step(
            &rewritten,
            &inv_name,
            &cfg,
            &tasks,
            &weights,
            very_first_cost,
            &name_mapping,
            &step_results,
            &follow_iter,
        );

        if !res.is_empty() {
            // rewrite with the invention
            let res: CompressionStepResult = res[0].clone();
            rewritten = res.rewritten.clone();
            name_mapping = res.name_mapping.clone();
            if !cfg.step.quiet { println!("Chose Invention {}: {}", res.inv.name, res) }
            step_results.push(res);
        } else if follow.is_some() {
            // if `follow` was given then we will keep going for the full set of iterations
            if !cfg.step.quiet { println!("Invention not found: {:?}", follow.as_ref().unwrap() ) }
        } else {
            if !cfg.step.quiet { println!("No inventions found at iteration {i}") }
            break;    
        }
    }

    if cfg.step.show_rewritten {
        println!("rewritten:\n{}", rewritten.iter().map(|p|p.to_string()).collect::<Vec<_>>().join("\n"));
    }

    if !cfg.step.quiet { println!("{}","\n=======Compression Summary=======".blue().bold()) }
    if !cfg.step.quiet { println!("Found {} inventions", step_results.len()) }
    let rewritten_cost = min_cost(&rewritten, &Some(weights.clone()), &Some(tasks.clone()), cost_fn);
    if !cfg.step.quiet { println!("Cost Improvement: ({:.2}x better) {} -> {}", compression_factor(very_first_cost, rewritten_cost), very_first_cost, rewritten_cost) }
    for res in step_results.iter() {
        let rewritten_cost = min_cost(&res.rewritten, &Some(weights.clone()), &Some(tasks.clone()), cost_fn);
        if !cfg.step.quiet { println!("{} ({:.2}x wrt orig): {}" , res.inv.name.clone().blue(), compression_factor(very_first_cost, rewritten_cost), res) }
    }
    if !cfg.step.quiet { println!("Time: {}ms", tstart.elapsed().as_millis()) }
    if cfg.step.follow_prune && !(
        cfg.step.no_opt_upper_bound
        && cfg.step.no_opt_force_multiuse
        && cfg.step.no_opt_useless_abstract
        && cfg.step.no_opt_arity_zero) && !cfg.step.quiet { println!("{} you often want to run --follow-track with --no-opt otherwise your target may get pruned", "[WARNING]".yellow()) }

    step_results
}

pub fn construct_shared(
    programs: &[ExprOwned],
    multistep_cfg: &MultistepCompressionConfig,
    tasks: &[String],
    weights: &[f32],
    prev_results: &[CompressionStepResult],
    follow: &Option<Invention>,
) -> Option<Arc<SharedData>> {
    let cfg = &multistep_cfg.step.clone();

    let cost_fn = &cfg.cost.expr_cost();

    let mut tstart = std::time::Instant::now();

    // structurally hashed exprset
    let mut set = ExprSet::empty(Order::ChildFirst, false, true);
    let roots: Vec<Idx> = programs.iter().map(|e| e.immut().copy_rec(&mut set)).collect();
    let corpus_span: Span = 0..set.len();

    let mut analyzed_cost = AnalyzedExpr::new(cost_fn.clone());
    analyzed_cost.analyze(&set);

    // populate num_paths_to_node so we know how many different parts of the programs tree
    // a node participates in (ie multiple uses within a single program or among programs)
    let (num_paths_to_node, num_paths_to_node_by_root_idx, num_paths_to_node_by_root_idx_sparse) = num_paths_to_node(&roots, &corpus_span, &set);

    if !cfg.quiet { println!("num_paths_to_node(): {:?}ms", tstart.elapsed().as_millis()) }
    tstart = std::time::Instant::now();

    let mut task_name_of_task: Vec<String> = vec![];
    let mut task_of_root_idx: Vec<usize> = vec![];
    let mut root_idxs_of_task: Vec<Vec<usize>> = vec![];
    for (root_idx,task_name) in tasks.iter().enumerate() {
        let task = task_name_of_task.iter().position(|name| name == task_name)
            .unwrap_or_else(||{
                task_name_of_task.push(task_name.clone());
                root_idxs_of_task.push(vec![]);
                task_name_of_task.len() - 1
            });
        task_of_root_idx.push(task);
        root_idxs_of_task[task].push(root_idx);
    }
    let tasks_of_node: Vec<FxHashSet<usize>> = associate_tasks(&roots, &set, &corpus_span, &task_of_root_idx);

    let init_cost_by_root_idx: Vec<Cost> = roots.iter().map(|idx| analyzed_cost[*idx] as Cost).collect();
    let init_cost_by_root_idx_weighted: Vec<f32> = init_cost_by_root_idx.iter().zip(weights.iter()).map(|(cost,weight)| (*cost as f32 * weight)).collect();
    let init_cost: Cost = root_idxs_of_task.iter().map(|root_idxs|
        root_idxs.iter().map(|idx| init_cost_by_root_idx[*idx]).min().unwrap()
    ).sum();
    let init_cost_weighted: Cost = root_idxs_of_task.iter().map(|root_idxs|
        root_idxs.iter().map(|idx| init_cost_by_root_idx_weighted[*idx].round() as Cost).min().unwrap()
    ).sum();
    let first_train_cost = roots.iter().map(|idx| analyzed_cost[*idx] as Cost).sum(); // This is used for --verbose-print

    if !cfg.quiet { println!("associate_tasks() and other task stuff: {:?}ms", tstart.elapsed().as_millis()) }
    if !cfg.quiet { println!("num unique tasks: {}", task_name_of_task.len()) }
    if !cfg.quiet { println!("num unique programs: {}", roots.len()) }
    tstart = std::time::Instant::now();
    
    // cost of a single usage times number of paths to node
    let cost_of_node_all: Vec<Cost> = corpus_span.clone().map(|node| analyzed_cost[node] as Cost * num_paths_to_node[node]).collect();

    let mut analyzed_free_vars = AnalyzedExpr::new(FreeVarAnalysis);

    if !cfg.quiet { println!("cost_of_node structs: {:?}ms", tstart.elapsed().as_millis()) }
    tstart = std::time::Instant::now();

    let (zid_of_zip,
        zip_of_zid,
        arg_of_zid_node,
        zids_of_node,
        extensions_of_zid) = get_zippers(&corpus_span, &analyzed_cost, &mut set, &mut analyzed_free_vars);
    
    if !cfg.quiet { println!("get_zippers(): {:?}ms", tstart.elapsed().as_millis()) }
    tstart = std::time::Instant::now();
    
    if !cfg.quiet { println!("{} zips", zip_of_zid.len()) }
    if !cfg.quiet { println!("arg_of_zid_node size: {}", arg_of_zid_node.len()) }

    let tracking = match follow {
        Some(follow) => if let Some(x) = follow.clone().to_tracking(&zid_of_zip) {
            Some(x)
        } else {
            if !cfg.quiet { println!("Tracking: can't possibly find a match for this in corpus because one if the necessary zippers ZIDs doesnt exist in corpus")}
            return None
        }
        None => None
    };


    if !cfg.quiet { println!("Tracking setup: {:?}ms", tstart.elapsed().as_millis()) }
    tstart = std::time::Instant::now();

    let mut analyzed_ivars = AnalyzedExpr::new(IVarAnalysis);
    analyzed_free_vars.analyze(&set);
    analyzed_cost.analyze(&set);
    analyzed_ivars.analyze(&set);

    let sym_var_info = SymvarInfo::new(&set, &cfg.symvar);


    if !cfg.quiet { println!("ran analyses: {:?}ms", tstart.elapsed().as_millis()) }
    tstart = std::time::Instant::now();


    let mut stats: Stats = Default::default();
    
    // define all the important data structures for compression
    let mut donelist: Vec<FinishedPattern> = Default::default(); // completed inventions will go here    

    let tdfa_global_annotations = TDFAGlobalAnnotations::new(cfg, &set, &roots, prev_results, &sym_var_info);

    let single_hole = Pattern::single_hole(&corpus_span, cost_fn, &cost_of_node_all, &num_paths_to_node, &tdfa_global_annotations, &set, cfg, follow);

    let mut azero_pruning_cutoff = 0;

    // arity 0 inventions
    if !cfg.no_opt_arity_zero {

        // we use single hole match locations in case they were pruned down from corpus_span by single_hole()
        for node in single_hole.match_locations.clone() {

            // Pruning (FREE VARS): inventions with free vars in the body are not well-defined functions
            // and should thus be discarded
            if !analyzed_free_vars[node].is_empty() {
                if !cfg.no_stats { stats.free_vars_fired += 1; };
                continue;
            }

            // PRUNING (SYMBOLIC VARS): inventions with symbolic variables in the body are not 0-arity inventions
            if sym_var_info.as_ref().is_some_and(|info| info.contains_symbols(node)) {
                continue;
            }

            // Pruning (SINGLE TASK): prune if used in only one task
            if !cfg.allow_single_task && tasks_of_node[node].len() < 2 {
                if !cfg.no_stats { stats.single_task_fired += 1; };
                continue;
            }

            // Note that "single use" pruning is intentionally not done here,
            // since any invention specific to a node will by definition only
            // be useful at that node

            let match_locations = vec![node];
            let body_utility = analyzed_cost[node] as Cost;

            // compressive_utility for arity-0 is cost_of_node_all[node] minus the penalty of using the new prim
            let compressive_utility: Cost = init_cost_weighted - root_idxs_of_task.iter().map(|root_idxs|
                root_idxs.iter().map(|idx|
                    (init_cost_by_root_idx_weighted[*idx]
                    - weights[*idx] * (num_paths_to_node_by_root_idx[*idx][node] * (analyzed_cost[node] - cost_fn.compute_cost_new_prim()) as Cost) as f32).round() as Cost)
                    .min().unwrap()
            ).sum::<Cost>();
            
            let utility = compressive_utility + noncompressive_utility(body_utility, cfg);
            if utility <= 0 { continue; }


            if !cfg.no_stats { stats.azero_calc_util += 1; };

            // Pruning (UPPER BOUND): Here we use compressive_utility which itself is an upper bound on total utility
            // so it is sound to prune based on it, as if we pruned based on compressive_utility then we would definitely
            // prune based on total utility. The pruning here before total utility is done because inverse_argument_capture()
            // is expensive.
            if compressive_utility <= azero_pruning_cutoff {
                continue // upper bound pruning
            }

            let pattern = Pattern {
                holes: vec![],
                pattern_args: PatternArgs::default(),
                match_locations,
                utility_upper_bound: utility,
                body_utility,
                tracked: false,
            };
            let mut finished_pattern = FinishedPattern {
                pattern,
                utility,
                compressive_utility,
                util_calc: UtilityCalculation { util: compressive_utility, unused_locations: Default::default()},
                arity: 0,
                usages: num_paths_to_node[node]
            };

            // This handle the case covered by Appendix B in the paper
            inverse_argument_capture(&mut finished_pattern, cfg, &zip_of_zid, &arg_of_zid_node, &extensions_of_zid, &set, &analyzed_ivars, cost_fn);
            if !cfg.no_stats { stats.azero_calc_unargcap += 1; };

            // Pruning (UPPER BOUND): This is the full upper bound pruning
            if finished_pattern.utility <= azero_pruning_cutoff {
                continue // upper bound pruning
            }
            // Pruning (UPPER BOUND): Here we update the upper bound (only in the inv_candidates=1 case for now but would
            // be good to handle other cases more generally by pushing to donelist and doing donelist.update())
            if cfg.inv_candidates == 1 && finished_pattern.utility > azero_pruning_cutoff {
                // if we're only looking for one invention, we can directly update our cutoff here
                azero_pruning_cutoff = finished_pattern.utility
            }
            donelist.push(finished_pattern);
        }
    }

    if !cfg.quiet { println!("arity 0: {:?}ms", tstart.elapsed().as_millis()) }
    tstart = std::time::Instant::now();

    if !cfg.quiet { println!("got {} arity zero inventions", donelist.len()) }

    let mut worklist = BinaryHeap::new();
    worklist.push(HeapItem::new(single_hole));

    let crit = CriticalMultithreadData::new(donelist, worklist, cfg);
    
    let fused_copy: Option<FxHashSet<Tag>> = if let Some(fused_tags) = &cfg.fused_lambda_tags.tags {
        let mut fused_copy = FxHashSet::default();
        fused_copy.extend(fused_tags.iter().cloned());
        Some(fused_copy)
    } else {
        None
    };
    
    let shared = Arc::new(SharedData {
        crit: Mutex::new(crit),
        programs: programs.to_vec(),
        arg_of_zid_node,
        cost_fn: cost_fn.clone(),
        analyzed_free_vars,
        analyzed_ivars,
        sym_var_info,
        analyzed_cost,    
        corpus_span: corpus_span.clone(),
        roots: roots.to_vec(),
        zids_of_node,
        zip_of_zid,
        zid_of_zip,
        extensions_of_zid,
        set,
        num_paths_to_node,
        num_paths_to_node_by_root_idx,
        num_paths_to_node_by_root_idx_sparse,
        tdfa_global_annotations,
        tasks_of_node,
        task_name_of_task,
        task_of_root_idx,
        root_idxs_of_task,
        cost_of_node_all,
        init_cost,
        init_cost_weighted,
        init_cost_by_root_idx,
        init_cost_by_root_idx_weighted,
        weight_by_root_idx: weights.to_vec(),
        first_train_cost,
        stats: Mutex::new(stats),
        cfg: cfg.clone(),
        multistep_cfg: multistep_cfg.clone(),
        tracking,
        fused_lambda_tags: fused_copy,
    });

    if !shared.cfg.quiet { println!("built SharedData: {:?}ms", tstart.elapsed().as_millis()) }

    Some(shared)
}

/// Takes a set of programs and does one full step of compresison.
#[allow(clippy::too_many_arguments)]
pub fn compression_step(
    programs: &[ExprOwned],
    new_inv_name: &str, // name of the new invention, like "inv4"
    multistep_cfg: &MultistepCompressionConfig,
    tasks: &[String],
    weights: &[f32],
    very_first_cost: Cost,
    name_mapping: &[(String, String)],
    prev_results: &[CompressionStepResult],
    follow: &Option<Invention>,
) -> Vec<CompressionStepResult> {

    let tstart_total = std::time::Instant::now();
    let tstart_prep = std::time::Instant::now();

    let Some(shared) = construct_shared(
        programs,
        multistep_cfg,
        tasks,
        weights,
        prev_results,
        follow,
    ) else {
        return vec![];
    };

    let mut tstart = std::time::Instant::now();

    if shared.cfg.verbose_best {
        let mut crit = shared.crit.lock();
        if !crit.deref_mut().donelist.is_empty() {
            let best_util = crit.deref_mut().donelist.first().unwrap().utility;
            let best_expr: String = crit.deref_mut().donelist.first().unwrap().info(&shared);
            let new_expected_cost =
                shared.first_train_cost
                - crit.donelist.first().unwrap().compressive_utility
                + crit.donelist.first().unwrap().to_expr(&shared).cost(&shared.cost_fn) as Cost;
            let trainratio = shared.first_train_cost as f64/new_expected_cost as f64;
            if !shared.cfg.quiet { println!("{} @ step=0 util={} trainratio={:.2} for {}", "[new best utility]".blue(), best_util, trainratio, best_expr) }
        }
    }

    if !shared.cfg.quiet { println!("TOTAL PREP: {:?}ms", tstart_prep.elapsed().as_millis()) }

    if !shared.cfg.quiet { println!("running pattern search...") }

    // *****************
    // * STITCH SEARCH *
    // *****************
    if shared.cfg.threads == 1 {
        // Single threaded
        stitch_search(Arc::clone(&shared));
    } else {
        // Multithreaded
        let mut handles = vec![];
        for _ in 0..shared.cfg.threads {
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

    if !shared.cfg.quiet { println!("TOTAL SEARCH: {:?}ms", tstart.elapsed().as_millis()) }
    if !shared.cfg.quiet { println!("TOTAL PREP + SEARCH: {:?}ms", tstart_total.elapsed().as_millis()) }


    tstart = std::time::Instant::now();

    // at this point we hold the only reference so we can get rid of the Arc
    let mut shared: SharedData = Arc::try_unwrap(shared).unwrap();

    // one last .update()
    shared.crit.lock().deref_mut().update(&shared.cfg);

    if !shared.cfg.quiet { println!("{:?}", shared.stats.lock().deref_mut()) }
    assert!(shared.crit.lock().deref_mut().worklist.is_empty());

    let donelist: Vec<FinishedPattern> = shared.crit.lock().deref_mut().donelist.clone();

    let dc_comparison_millis = if shared.cfg.dreamcoder_comparison {
        if !shared.cfg.quiet { println!("Timing point 1 (from the start of compression_step to final donelist): {:?}ms", tstart_total.elapsed().as_millis()) }
        if !shared.cfg.quiet { println!("Timing Comparison Point A (search) (millis): {}", tstart_total.elapsed().as_millis()) }
        let tstart_rewrite = std::time::Instant::now();
        rewrite_fast(&donelist[0], &shared, &Node::Prim(new_inv_name.into()), &shared.cost_fn);
        if !shared.cfg.quiet { println!("Timing point 2 (rewriting the candidate): {:?}ms", tstart_rewrite.elapsed().as_millis()) }
        if !shared.cfg.quiet { println!("Timing Comparison Point B (search+rewrite) (millis): {}", tstart_total.elapsed().as_millis()) }
        Some(tstart_total.elapsed().as_millis() as usize)
    } else {
        None
    };

    let mut results: Vec<CompressionStepResult> = vec![];

    // construct CompressionStepResults and print some info about them)
    if !shared.cfg.quiet { println!("Cost before: {}", shared.init_cost) }
    for (i,done) in donelist.iter().enumerate() {
        let res = CompressionStepResult::new(done.clone(), new_inv_name, &mut shared, very_first_cost, name_mapping, dc_comparison_millis);
        if !shared.cfg.quiet { println!("{i}: {res}") }
        results.push(res);
    }

    if shared.cfg.follow_prune && !results.is_empty() {
        if let Some(follow) = follow {
            assert_eq!(follow.body.to_string(), results[0].inv.body.to_string(), "found something other than the followed abstraction somehow");
        }
    }

    if !shared.cfg.quiet { println!("post processing: {:?}ms", tstart.elapsed().as_millis()) }

    results
}

/// toplevel entrypoint to compression used by most apis
pub fn multistep_compression(
    programs: &[String],
    tasks: Option<Vec<String>>,
    weights: Option<Vec<f32>>,
    name_mapping: Option<Vec<(String,String)>>,
    follow: Option<Vec<Invention>>,
    cfg: &MultistepCompressionConfig
)-> (Vec<CompressionStepResult>, serde_json::Value) {
    let mut programs = programs.to_vec();
    let mut cfg = cfg.clone();

    if let Some(tasks) = &tasks {
        assert_eq!(tasks.len(), programs.len());
    }

    if cfg.silent {
        cfg.step.quiet = true
    }
    
    if cfg.no_opt {
        cfg.step.no_opt();
    }
    
    if cfg.shuffle {
        programs.shuffle(&mut rand::thread_rng());
    }
    if let Some(n) = cfg.truncate {
        programs.truncate(n);
    }
    
    // parse the program strings into expressions
    let train_programs: Vec<ExprOwned> = programs.iter().map(|p|{
        let mut set = ExprSet::empty(Order::ChildFirst, false, false);
        let idx = set.parse_extend(p).unwrap();
        ExprOwned::new(set,idx)
    }).collect();

    let cost_fn = cfg.step.cost.expr_cost();

    if !cfg.silent {
        println!("{}","**********".blue().bold());
        println!("{}","* Stitch *".blue().bold());
        println!("{}","**********".blue().bold());
        programs_info(&train_programs, &cost_fn);
    }

    let step_results = multistep_compression_internal(
        &train_programs, 
        tasks.clone(),
        weights.clone(),
        name_mapping, 
        follow,
        &cfg, 
    );

    // write everything to json
    let json_res = json_of_step_results(&step_results, &train_programs, weights, tasks, &cost_fn, &cfg);

    (step_results, json_res)
}

pub fn json_of_step_results(step_results: &[CompressionStepResult], train_programs: &Vec<ExprOwned>, weights: Option<Vec<f32>>, tasks: Option<Vec<String>>, cost_fn: &ExprCost, cfg: &MultistepCompressionConfig) -> serde_json::Value {
    let rewritten: &Vec<ExprOwned> = step_results.iter().last().map(|res| &res.rewritten).unwrap_or(train_programs);
    let original_cost = min_cost(train_programs, &weights, &tasks, cost_fn);
    let final_cost = min_cost(rewritten, &weights, &tasks, cost_fn);
    let rewritten = step_results.iter().last().map(|res| &res.rewritten).unwrap_or(train_programs).iter().map(|p| p.to_string()).collect::<Vec<String>>();
    let rewritten_dreamcoder = if !cfg.step.rewritten_dreamcoder { None } else {
        let rewritten_dreamcoder = step_results.iter().last().map(|res| res.rewritten_dreamcoder.clone().unwrap()).unwrap_or_else(||train_programs.iter().map(
            |p| p.to_string().replace("(lam ", "(lambda ").replace("(lam_", "(lambda_")
        ).collect::<Vec<String>>());
        Some(rewritten_dreamcoder)
    };
    json!({
        "cmd": std::env::args().join(" "),
        "args": cfg,
        "original_cost": original_cost,
        "final_cost": final_cost,
        "compression_ratio": compression_factor(original_cost,final_cost),
        "num_abstractions": step_results.len(),
        "original": train_programs.iter().map(|p| p.to_string()).collect::<Vec<String>>(),
        "rewritten": rewritten.iter().map(|p| p.to_string()).collect::<Vec<String>>(),
        "rewritten_dreamcoder": rewritten_dreamcoder,
        "abstractions": step_results.iter().map(|inv| inv.json(&cfg.step)).collect::<Vec<serde_json::Value>>(),
    })
}
