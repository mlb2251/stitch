use egg::{rewrite as rw, *};
use std::collections::{HashSet,HashMap};
use std::path::PathBuf;
use chrono;
use serde_json;
extern crate log;
use clap::Parser;
use serde_json::de::from_reader;
use std::fs::File;
use std::fmt::{self, Formatter, Display};
pub mod expr;
use expr::*;
pub mod eval;
mod simple_domain;
use simple_domain::Simple;

/// egg dream
#[derive(Parser, Debug)]
#[clap(name = "Dream Egg")]
struct Args {
    /// json file to read compression input programs from
    #[clap(short, long, parse(from_os_str), default_value = "data/train_19.json")]
    file: PathBuf,

    /// Number of iterations to run compression for
    #[clap(short, long, default_value = "20")]
    iterations: usize,

    /// max arity of inventions
    #[clap(short='a', long, default_value = "2")]
    max_arity: usize,

    /// beam size
    #[clap(short, long, default_value = "10000000")]
    beam_size: usize,

    /// disable caching
    #[clap(long)]
    no_cache: bool,

    /// whether to render the inventions
    #[clap(long)]
    render_inventions: bool,

    /// render the final egraph
    #[clap(long)]
    render_final: bool,

    /// render initial egraph
    #[clap(long)]
    render_initial: bool,

    /// number of inventions to print - set to 0 if you dont want to print inventions at all
    #[clap(long, default_value="0")]
    print_inventions: usize,
}

const COST_NONTERMINAL: i32 = 1;
const COST_TERMINAL: i32 = 100;

type EGraph = egg::EGraph<Lambda, LambdaAnalysis>;

#[derive(Default)]
struct LambdaAnalysis;

#[derive(Debug)]
struct Data {
    upward_refs: HashSet<i32>, // $i vars. "how much higher"
    upward_refs_ivar: HashSet<i32>, // for #i ivars
    inventionless_cost: i32,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
struct Invention {
    body:Id, // this will be a subtree which can have IVars
    arity: usize // also equivalent to max ivar in subtree + 1
}

#[derive(Debug, Clone)]
struct InventionExpr {
    lam_expr: Expr, // invention body but wrapped in lambdas
    arity: usize
}

impl Display for InventionExpr {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "[arity {}]: {} ", self.arity, self.lam_expr)
    }
}

impl Invention {
    fn new(body:Id, arity: usize) -> Invention {
        Invention { body, arity }
    }
    fn canonicalize(&mut self, egraph: &EGraph) {
        self.body = egraph.find(self.body);
    }
    fn is_canonical(&self, egraph: &mut EGraph) -> bool {
        self.body == egraph.find(self.body)
    }
    fn valid_invention(&self, egraph: &EGraph) -> bool {
        // even invalid Inventions are important as parts of AppLams that will propagate recursively upward,
        // This checks that there aren't any upward refs that go beyond the args of the AppLam itself
        // egraph[self.body].data.upward_refs.iter().all(|i| *i < (self.arity as i32))
        egraph[self.body].data.upward_refs.is_empty()
    }
    fn to_expr(&self, egraph: &EGraph) -> InventionExpr {
        // wrap body in lambdas
        let mut expr = extract(self.body, &egraph);
        for _ in 0..self.arity {
            expr = Expr::lam(expr);
        }
        InventionExpr {lam_expr: expr, arity:self.arity}
    }
}

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
    fn canonicalize(&mut self, egraph: &mut EGraph) {
        self.inv.canonicalize(egraph);
        for arg in &mut self.args {
            if !canonical(arg, egraph) {
                *arg = egraph.find(*arg);
            }
        }
    }
    fn is_canonical(&self, egraph: &mut EGraph) -> bool {
        self.inv.is_canonical(egraph) &&
        self.args.iter().all(|arg| canonical(arg, egraph))
    }
    fn upward_refs(&self, egraph: &mut EGraph) -> HashSet<i32> {
        // union together all the upward refs of body + args
        let mut upward_refs: HashSet<i32> = egraph[self.inv.body].data.upward_refs.clone();
        for arg in self.args.iter() {
            upward_refs.extend(egraph[*arg].data.upward_refs.clone());
        }
        upward_refs
    }
    fn to_string(&self, egraph: &EGraph) -> String {
        format!("inv:{}\narg:{}",
            self.inv.to_expr(egraph),
            self.args.iter().map(|arg| extract(*arg, egraph).to_string()).collect::<Vec<_>>().join("\narg:")
        )
    }

}

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
    fn cost_under_inv(&self, inv: &Invention) -> i32 {
        // cost under invention else default cost
        // invention is (applam_body_eclass,arity)
        if self.inventionful_cost.contains_key(inv) {
            self.inventionful_cost[inv]
        } else {
            self.inventionless_cost
        }
    }
    fn new_cost_under_inv(&mut self, inv: Invention, cost:i32) {
        // in this algorithm I don't think we should ever insert a key
        // we've already inserted
        if cost < self.inventionless_cost {
            if !self.inventionful_cost.contains_key(&inv)
               || cost < self.inventionful_cost[&inv]  {
                self.inventionful_cost.insert(inv, cost);
            }
        }
    }
    fn top_inventions(&self) -> Vec<Invention> {
        let mut top_inventions: Vec<Invention> = self.inventionful_cost.keys().cloned().collect();
        top_inventions.sort_by(|a,b| self.inventionful_cost[a].cmp(&self.inventionful_cost[b]));
        top_inventions
    }
}


fn extract(eclass: Id, egraph: &EGraph) -> Expr {
    let mut extractor = Extractor::new(&egraph, ProgramCost{});
    let (_,p) = extractor.find_best(eclass);
    p.into()
}

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

// todo definitely possible to rewrite much simpler and such that we dont need applams_of_treenode and best_inventions_of_treenode and it works generically for rewriting on things we havent seen before. See my own notes.
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
        assert_eq!(target_cost,cost_rec(&expr));
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
        Lambda::IVar(i) => {
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

    assert_eq!(target_cost,cost_rec(&expr));
    expr
}


#[inline]
fn canonical(id:&Id, egraph: &EGraph) -> bool {
    egraph.find(*id) == *id
}

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

impl Analysis<Lambda> for LambdaAnalysis {
    type Data = Data;
    fn merge(&self, to: &mut Data, from: Data) -> bool {

        assert_eq!(to.upward_refs,from.upward_refs);
        assert_eq!(to.upward_refs_ivar,from.upward_refs_ivar);
        // we really shouldnt be merging anyone ever rn, but later we'll want to do this
        assert_eq!(to.inventionless_cost,from.inventionless_cost);

        // keep the lowest inventionless cost
        // modified |= merge_inventionless(&mut to.inventionless_cost_any, &from.inventionless_cost_any);
        
        false // didnt modify anything
    }

    fn make(egraph: &EGraph, enode: &Lambda) -> Data {
        let mut upward_refs: HashSet<i32> = HashSet::new();
        let mut upward_refs_ivar: HashSet<i32> = HashSet::new();
        match enode {
            Lambda::Var(i) => {
                upward_refs.insert(*i);
            }
            Lambda::IVar(i) => {
                upward_refs_ivar.insert(*i);
            }
            Lambda::Prim(_) => {
            }
            Lambda::App([f, x]) => {
                // union of f and x
                upward_refs.extend(egraph[*f].data.upward_refs.iter());
                upward_refs.extend(egraph[*x].data.upward_refs.iter());
                upward_refs_ivar.extend(egraph[*f].data.upward_refs_ivar.iter());
                upward_refs_ivar.extend(egraph[*x].data.upward_refs_ivar.iter());
            }
            Lambda::Lam([b]) => {
                // body, subtract 1 from all values, remove the -1 if its in there
                upward_refs.extend(egraph[*b].data.upward_refs.iter()
                    .map(|x| x-1)
                    .filter(|x| *x >= 0));
                upward_refs_ivar.extend(egraph[*b].data.upward_refs_ivar.iter());
            }
            Lambda::Programs(programs) => {
                // assert no free variables in programs
                assert!(programs.iter().all(|p| egraph[*p].data.upward_refs.is_empty()));
                assert!(programs.iter().all(|p| egraph[*p].data.upward_refs_ivar.is_empty()));
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
               upward_refs: upward_refs,
               upward_refs_ivar: upward_refs_ivar,
               inventionless_cost: inventionless_cost
            }
    }

    fn modify(egraph: &mut EGraph, id: Id) {
    }
}

fn var(s: &str) -> Var {
    s.parse().unwrap()
}

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

fn recursive_var_mod(
    var_mod: impl Fn(i32, i32, i32, &mut EGraph) -> Option<Id>,
    ivars: bool, // whether to run this on vars or ivars
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

        if  (ivars && egraph[eclass].data.upward_refs_ivar.is_empty())
        || (!ivars && egraph[eclass].data.upward_refs.iter().all(|i| *i < depth)) {
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


struct ProgramCost {}

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

fn cost_rec(expr: &Expr) -> i32 {
    ProgramCost{}.cost_rec(expr.into())
}

struct ProgramDepth {}

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
                .min().unwrap()
            }
        }
    }
}

fn depth_rec(expr: &Expr) -> i32 {
    ProgramDepth{}.cost_rec(expr.into())
}





fn timestamp() -> String {
    format!("{}", chrono::Local::now().format("%Y-%m-%d_%H-%M-%S"))
}

/// finds everywhere the rewrite rules matches and applies it to each of them
/// and rebuilds the egraph. Will only apply to matches that are visible before
/// any rewriting occurs. This is the same as running a runner with an iter limit of 1.
/// I guess I'm not using this in the code right now bc I like the runner's report.
fn apply_everywhere_once(rules_: &[&str], egraph: &mut EGraph) {
    let rules: Vec<Rewrite<Lambda,LambdaAnalysis>> = rules(rules_);
    let matches: Vec<Vec<SearchMatches>> = rules.iter().map(|r| r.search(egraph)).collect();
    for (r,m) in rules.iter().zip(matches) {
        let hits = r.apply(egraph, &m).len();
        println!("(applied {} {} times out of {} matches)",r.name(),hits, m.len());
    }
    egraph.rebuild();
}

fn saturate(rules_: &[&str], render: bool, out_dir: String, egraph: EGraph) -> EGraph {
    let rules: Vec<Rewrite<Lambda,LambdaAnalysis>> = rules(rules_);
    let mut runner = Runner::default()
        .with_egraph(egraph)
        .with_iter_limit(400)
        .with_scheduler(SimpleScheduler)
        .with_time_limit(core::time::Duration::from_secs(200))
        .with_node_limit(3000000);
    
    if render {
        runner = runner.with_hook(
        {
            let out_dir = out_dir.clone(); // silly thing to clone into the closure
            move |runner|{
                let iter = runner.iterations.len();
                println!("Iter {}: {}", iter, egraph_info(&runner.egraph));
                save(&runner.egraph, format!("3_propagate_{}",iter).as_str(), &out_dir);
                Ok(())
            }
        });
    }

    let runner = runner.run(rules.iter());
    runner.print_report();
    runner.egraph
}

fn run_pretty(rule_: &str, name:&str, egraph: &mut EGraph) {
    let rule: Rewrite<Lambda,LambdaAnalysis> = rule(rule_);
    let matches = rule.search(egraph);
    egraph.dot().to_png(format!("target/match_{}_0pre.png",name)).unwrap();
    rule.apply(egraph, &matches).len();
    egraph.dot().to_png(format!("target/match_{}_1post.png",name)).unwrap();
    egraph.rebuild();
    egraph.dot().to_png(format!("target/match_{}_2rebuild.png",name)).unwrap();
}

fn search(pat: &str, egraph: &EGraph) -> Vec<SearchMatches>{
    let applam:Pattern<Lambda> = pat.parse().unwrap();
    applam.search(&egraph)
}

fn save(egraph: &EGraph, name: &str, outdir: &str) {
    egraph.dot().to_png(format!("{}/{}.png",outdir,name)).unwrap();
}

fn save_expr(expr: &Expr, name: &str, outdir: &str) {
    let mut egraph: EGraph = Default::default();
    egraph.add_expr(expr.into());
    egraph.dot().to_png(format!("{}/{}.png",outdir,name)).unwrap();
}

fn rule_map() -> HashMap<String,Rewrite<Lambda, LambdaAnalysis>> {
    vec![
    ].into_iter().map(|r:Rewrite<Lambda, LambdaAnalysis>| (r.name().to_string(),r)).collect()
}

// ownership is a pain so this is a helper
fn rule(name: &str) -> Rewrite<Lambda, LambdaAnalysis> {
    rule_map().remove(name).expect(format!("rule {} not found",name).as_str())
}

fn rules(names: &[&str]) -> Vec<Rewrite<Lambda, LambdaAnalysis>> {
    names.iter().map(|name|rule(name)).collect()
}

fn egraph_info(egraph: &EGraph) -> String {
    format!("{} nodes, {} classes, {} memo", egraph.total_number_of_nodes(), egraph.number_of_classes(), egraph.total_size())
}

fn toplogical_ordering(root: Id, egraph: &EGraph) -> Vec<Id> {
    // returns a Vec of Ids ending in the root Id (ie child first traversal)
    // and notably an Id never shows up twice (if it was already there earlier it wont be added again)
    //todo  assumes no cycles!! AND assumes each eclass only has one enode at this point, though you could relax the latter
    let mut vec = Vec::new();
    toplogical_ordering_rec(root, egraph, &mut vec);
    vec
}

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


type RecVarModCache = HashMap<(Id,i32),Option<Id>>;
#[derive(Debug,Clone,Eq,PartialEq,Hash)]
enum Shift {
    ShiftVar(i32), // shift $i to be $(i+incr_by)
    ShiftIVar(i32), // shift #i to be #(i+incr_by)
    TableShiftIVar(Vec<i32>), // shift #i to be #(i+table[#i]) ie look up the shift amount in the table
}
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

struct InversionResult {
    applams_of_treenode: HashMap<Id,Vec<AppLam>>,
    best_inventions_of_treenode: HashMap<Id,BestInventions>
}

/// Does all the work
#[inline(never)] // for flamegraph debugging
fn run_inversions(
    treenodes: &Vec<Id>,
    max_arity: usize,
    beam_size: usize,
    no_cache: bool,
    egraph: &mut EGraph
) -> InversionResult {
    // one vector of applams per tree node
    let mut applams_of_treenode: HashMap<Id,Vec<AppLam>> = Default::default();
    let mut best_inventions_of_treenode: HashMap<Id,BestInventions> = Default::default();
    
    let ivar0: Id = egraph.add(Lambda::IVar(0));

    // Caches - these give us considerable speedup (26s -> 18s)
    let caches = &mut CacheGenerator::new(!no_cache);

    for treenode in treenodes.iter() {
        // println!("processing id={}: {}", treenode, extract(*treenode, egraph) );

        // im essentially using the egraph just for its structural hashing rn
        assert!(egraph[*treenode].nodes.len() == 1);
        // clone to appease the borrow checker
        let node = egraph[*treenode].nodes[0].clone();

        // todo maybe should just straight up call canoncialize here to make sure instead of just asserting it
        // its very very very important that these are all canonical
        debug_assert!(node.children().iter().all(|c| applams_of_treenode[c].iter().all(|applam| applam.is_canonical(egraph))));
        debug_assert!(node.children().iter().all(|c| best_inventions_of_treenode[c].inventionful_cost.keys().all(|inv| inv.is_canonical(egraph))));

        //==================================//
        // *** PROPAGATE/CREATE APPLAMS *** //
        //==================================//
        
        let mut applams: Vec<AppLam> = Vec::new();
        let mut downshifted_applam_args: Vec<(Id,Id)> = Vec::new(); // minor impl detail
        
        // any node can become the identity applam
        applams.push(AppLam::new(ivar0, vec![*treenode]));

        match node {
            Lambda::IVar(_) => {
                panic!("attempted to abstract an IVar"); // could adapt to handle this sort of situation, just not what I'm aiming for right now
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
                    debug_assert_eq!(applams.last().unwrap().upward_refs(egraph),egraph[*treenode].data.upward_refs);                        
                }

                // bubbling from the right:
                // (app f x) == (app f (applam body arg)) => (applam (app upshift(f) body) arg)
                // note no shifting is needed thanks to IVars
                for x_applam in x_applams.iter() {
                    let new_applam_body = egraph.add(Lambda::App([f, x_applam.inv.body]));
                    applams.push(AppLam::new(new_applam_body, x_applam.args.clone()));
                    debug_assert_eq!(applams.last().unwrap().upward_refs(egraph), egraph[*treenode].data.upward_refs);
                }

                // println!("f_applam x_applam pairwise product size: {} x {} -> {}",f_applams.len(), x_applams.len(), f_applams.len() * x_applams.len());

                for f_applam in f_applams.iter() {
                    for x_applam in x_applams.iter() {
                        // making a higher arity applam out of two diff applams
                        // and merging any shared arguments. Higher arity applam looks a bit like:
                        // (app f x) == (app (applam body1 arg1) (applam body2 arg2)) => (applam (app body1 upshift(body2)) arg1 arg2)
                        // note that (applam body arg0 arg1) means arg0 will fill upward refs to $0 and arg1 will fill upward refs to $1
                        // so somewhat confusingly (applam body arg0 arg1) == (app (app (lam (lam body)) arg1) arg0) - but hopefully you dont need to think about that
                        // Merging: when f_applam and x_applam have identical args we can merge them
                        // (app f x) == (app (applam body1 arg) (applam body2 arg)) => (applam (app body1 body2) arg)
                        // here we do that for partial overlap between the two as well!

                        let overlap: usize = f_applam.args.iter().filter(|farg| x_applam.args.contains(farg)).count();
                        if f_applam.inv.arity + x_applam.inv.arity - overlap > max_arity {
                            continue;
                        }

                        if overlap > 0 {
                            // merging is needed

                            // x_shift_table[1] tells us how much to shift an upward ref to $1 in x_applam.body
                            // (note without merging this would be the arity of f_applam)
                            let mut x_shift_table = vec![]; // just gonna assume nobody wants an arity greater than 10 (for static speed)
                            let mut to_remove = vec![];
                            let mut shift_rest_by = f_applam.inv.arity as i32; // normal amt we shift x by, except if there are merges to be done. If a merge happens all the higher x vars get shifted less, and the specific x var gets shifted a very specific amount
                            for (x_idx,xarg) in x_applam.args.iter().enumerate() {
                                if let Some(f_idx) = f_applam.args.iter().position(|farg| farg == xarg) {
                                    // we found a match! $x_idx should map to the same thing as $f_idx.
                                    // remember, our body currently has $x_idx at the toplevel so now
                                    // we want to shift it by $(f_idx-x_idx) so that it ends up as f_idx.
                                    x_shift_table.push((f_idx as i32) - (x_idx as i32));
                                    to_remove.push(true);
                                    shift_rest_by -= 1; // effectively downshifts all the higher args now that this one is gone
                                } else {
                                    // shift fully without merging
                                    x_shift_table.push(shift_rest_by);
                                    to_remove.push(false);
                                }
                            }

                            // remove the args from xargs that we can merge into fargs
                            let new_x_applam_args: Vec<Id> = x_applam.args.iter()
                                .zip(to_remove)
                                .filter(|(_,remove)| !*remove)
                                .map(|(xarg,_)| xarg)
                                .cloned().collect();

                            let shifted_x_applam_body = shift(x_applam.inv.body, Shift::TableShiftIVar(x_shift_table), egraph, Some(caches)).unwrap();

                            let new_applam_body = egraph.add(Lambda::App([f_applam.inv.body,shifted_x_applam_body]));
                            let mut new_applam_args = f_applam.args.clone();
                            new_applam_args.extend(new_x_applam_args);
                            applams.push(AppLam::new(new_applam_body, new_applam_args));
                            debug_assert_eq!(applams.last().unwrap().upward_refs(egraph),egraph[*treenode].data.upward_refs);                        
                        } else {
                            // no overlap so no merging
                            // We will use the lower indices from f_applam and will upshift x_applam
                            let shifted_x_applam_body = shift(x_applam.inv.body, Shift::ShiftIVar(f_applam.inv.arity as i32), egraph, Some(caches)).unwrap();

                            let new_applam_body = egraph.add(Lambda::App([f_applam.inv.body,shifted_x_applam_body]));
                            let mut new_applam_args = f_applam.args.clone();
                            new_applam_args.extend(x_applam.args.clone());
                            applams.push(AppLam::new(new_applam_body, new_applam_args));
                            debug_assert_eq!(applams.last().unwrap().upward_refs(egraph),egraph[*treenode].data.upward_refs);                        
                        };
                    }
                    
                }

            },
            Lambda::Lam([b]) => {
                let ref b_applams = applams_of_treenode[&b];
                // bubbling up over the lambda:
                // (lam b) == (lam (applam body arg)) => (applam (lam careful_shift(body)) arg)
                // where:
                //  - arg must not have any upward refs to $0 in it since we cant jump over a lambda we point to
                //    > (in the multiarg applam case, none of them can have $0)
                //  - we need to shift the body in a very specific way. Say the applam was arity 3. Then
                //    any outgoing refs to $0 $1 $2 in the original body point to these args, and $3 points to the lam
                //    we're about to jump over. Now the lam is 3 levels deeper so pointers to $3 at the top
                //    level should now point to $0. Meanwhile pointers to $0 $1 $2 should be incremented by 1 since
                //    theres one more lambda in the way now.
                // *** note: all that painful shifting above is no longer needed thanks to IVars! Just a little downshift remains.

                for b_applam in b_applams.iter() {
                    // can't bubble an applam over a lambda if its arg refers to the lambda!
                    if b_applam.args.iter().any(|arg| egraph[*arg].data.upward_refs.contains(&0)) {
                        continue;
                    }
                    
                    // downshift the args since the lambda above them moved below them (earlier we made sure none of them had pointers to it)
                    let new_args: Vec<Id> = b_applam.args.iter().map(|arg| shift(*arg, Shift::ShiftVar(-1), egraph, Some(caches)).unwrap()).collect();

                    // add the downshifted args to a worklist discussed below. Worklist needed bc of borrow checker.
                    downshifted_applam_args.extend(new_args.iter().cloned().zip(b_applam.args.iter().cloned())); // (new,old)

                    let new_applam_body = egraph.add(Lambda::Lam([b_applam.inv.body]));
                    applams.push(AppLam::new(new_applam_body, new_args));
                    debug_assert_eq!(applams.last().unwrap().upward_refs(egraph),egraph[*treenode].data.upward_refs);                        
                }
            },
        }

        // let child_applams: Vec<usize> = node.children().iter()
        //     .map(|id| applams_of_treenode[id].len())
        //     .collect();

        // println!("{:?}:\n\t{:?} -> {}\n\t{}",node,child_applams, applams.len(), egraph.total_size());

        // Processing downshifted_applam_args:
        // downshifting args just is sortof a big deal because other than this moment we have had the invariant that args
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

        let mut best_inventions = BestInventions::new(egraph[*treenode].data.inventionless_cost);

        // replacing this node with a call to an invention
        for applam in applams.iter() {
            if applam.inv.valid_invention(egraph) && applam.inv.body != ivar0 {
                let cost: i32 =
                    COST_TERMINAL // the new primitive for this invention
                    + COST_NONTERMINAL * applam.inv.arity as i32 // the chain of app()s needed to apply the new primitive
                    + applam.args.iter()
                        .map(|id| best_inventions_of_treenode[&id]
                        .cost_under_inv(&applam.inv))
                        .sum::<i32>(); // sum costs of actual args
                best_inventions.new_cost_under_inv(applam.inv, cost);
            }
        }
                
        // which inventions helped our children?
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
                    best_inventions.new_cost_under_inv(inv, cost);
                }
            }
            Lambda::Lam([b]) => {
                // just map +1 over the costs
                let ref b_best_inventions = best_inventions_of_treenode[&b];
                for (inv,cost) in b_best_inventions.inventionful_cost.iter() {
                    best_inventions.new_cost_under_inv(*inv, cost + COST_NONTERMINAL);
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
                    best_inventions.new_cost_under_inv(inv, cost);
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
fn run_compression_step(
    programs_expr: &Expr,
    args: &Args,
    out_dir: &str,
    new_inv_name: &str,
) -> Option<CompressionResult> {

    // build the egraph. We'll just be using this as a structural hasher we don't use rewrites at all. All eclasses will always only have one node.
    let mut egraph: EGraph = Default::default();
    let programs_id = egraph.add_expr(programs_expr.into());
    egraph.rebuild(); // this is VERY important to run before you try applying any searches or rewrites

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

    let inversion_result
        = run_inversions(
            &treenodes,
            args.max_arity,
            args.beam_size,
            args.no_cache,
            &mut egraph
        );
    let (best_inventions_of_treenode,applams_of_treenode) = (inversion_result.best_inventions_of_treenode,inversion_result.applams_of_treenode);

    egraph.rebuild(); // hopefully doesnt matter at all anyways, not sure if we needed to do this thruout inversions

    let elapsed = tstart.elapsed().as_millis();

    println!("Inventionless (cost={:?}):\n{}\n",
        egraph[programs_id].data.inventionless_cost,
        extract(programs_id, &egraph)
    );

    let top_invs: Vec<Invention> = best_inventions_of_treenode[&programs_id].top_inventions();
    println!("Found {} Inventions that helped at the top level", top_invs.len());
    println!("\n*** Core stuff took: {}ms ***\n", elapsed);

    for (i,inv) in top_invs.iter().take(args.print_inventions).enumerate() {
        let inv_expr = inv.to_expr(&egraph).lam_expr;
        let inv_str = &format!("inv{}_{}",inv.body,inv.arity);
        let rewritten = extract_under_inv(programs_id, *inv, inv_str, &applams_of_treenode, &best_inventions_of_treenode, &egraph);
        println!("\nInvention {} {:?} (inv_cost={:?}; rewritten_cost={:?}):\n{}\n Rewritten:\n{}",
            i,
            inv,
            cost_rec(&inv_expr),
            cost_rec(&rewritten),
            inv_expr,
            rewritten,
        );
        if args.render_inventions {
            save_expr(&inv_expr, &format!("inv{}",i), &out_dir);
        }
    }

    println!("Final egraph: {}",egraph_info(&egraph));
    // a terrible way of printing the max/min that i should change i just havent gotten around to it
    println!("Variables used:");
    for i in -2..10 {
        println!("{}: {}", i, search(format!("(${})",i).as_str(),&egraph).len());
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

    let top_inv = top_invs[0].clone();
    let top_inv_expr = top_inv.to_expr(&egraph);
    let top_inv_rewritten = extract_under_inv(programs_id, top_inv.clone(), new_inv_name, &applams_of_treenode, &best_inventions_of_treenode, &egraph);

    Some(CompressionResult {
        inv: top_inv_expr,
        rewritten: top_inv_rewritten,
    })
}

fn compression(
    programs_expr: &Expr,
    args: &Args,
    out_dir: &str,
) -> (Vec<InventionExpr>,Expr) {
    let mut rewritten: Expr = programs_expr.clone();
    let mut invs: Vec<InventionExpr> = Default::default();
    
    for i in 0..args.iterations {
        println!("***Iteration {}",i);
        let inv_name = &format!("inv{}",invs.len());
        if let Some(res) = run_compression_step(&rewritten, args, out_dir, inv_name) {
            rewritten = res.rewritten.clone();
            println!("***Found Invention {}: {}\n***Rewritten:{}", inv_name, res.inv, res.rewritten);
            invs.push(res.inv);
        } else {
            println!("***No inventions found at iteration {}",i);
            break;
        }
        
    }
    (invs,rewritten)
}

fn programs_info(programs: &Vec<String>) {
    let ps: Vec<Expr> = programs.iter().map(|p| p.parse::<Expr>().unwrap()).collect();
    let max_cost = ps.iter().map(|p| cost_rec(p)).max().unwrap();
    let max_depth = ps.iter().map(|p| depth_rec(p)).max().unwrap();
    println!("Programs:");
    println!("\t num: {}",ps.len());
    println!("\t max cost: {}",max_cost);
    println!("\t max depth: {}",max_depth);
}


fn simple_test() {
    let e: Expr = "(app (app + 1) 2)".parse().unwrap();
    let mut e: DomExpr<Simple> = e.into();
    println!("{}",e);
    let res = e.eval(&[]);
    println!("{:?}",res);

    let e: Expr = "(app (app map (lam (app (app + 1) $0))) $0)".parse().unwrap();
    let mut e: DomExpr<Simple> = e.into();
    println!("{}",e);
    let res = e.eval(&[Simple::val_of_prim("[1,2,3]".into()).unwrap()]);
    println!("{:?}",res);
    println!("{}",e.pretty_evals());


    // let res = eval::run_with_timeout(
    //     |(e,env)|  e.eval::<simple_domain::SimpleVal>(env),
    //     (e,&[]),
    //     std::time::Duration::from_millis(100)
    // );
    println!("{:?}",res);
    panic!("done")
}

fn main() {
    procspawn::init();
    // eval::test_run();

    env_logger::init();

    simple_test();

    let args: Args = Args::parse();

    // create a new directory for logging outputs
    let out_dir: String = format!("target/{}",timestamp());
    let out_dir_p = std::path::Path::new(out_dir.as_str());
    assert!(!out_dir_p.exists());
    std::fs::create_dir(out_dir_p).unwrap();

    let programs: Vec<String> = from_reader(File::open(&args.file).expect("file not found")).expect("json deserializing error");
    programs_info(&programs);
    let programs: String = format!("(programs {})",programs.join(" "));

    assert!(!programs.to_string().contains("(app (lam"),
        "Normal dreamcoder programs never have unapplied lambdas in them! 
         Who knows what might happen if you run this. Side note you can probably
         inline them and it'd be fine (we've got a function for that!) and also
         who knows maybe it wouldnt be an issue in the first place");

    let programs_expr: Expr = programs.parse().unwrap();

    compression(&programs_expr, &args, &out_dir);

}

