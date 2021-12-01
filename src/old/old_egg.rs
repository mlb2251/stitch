/// This is an old egg based implementation of compression!

use egg::{rewrite as rw, *};
use std::collections::{HashSet,HashMap};
use chrono;


extern crate log;

const ARGC: i32 = 6;
const BEAM_SIZE: usize = 1000000;
const COST_NONTERMINAL: i32 = 1;
const COST_TERMINAL: i32 = 100;

// const SAVE: &str = "all";


struct AppLam {
    
}



define_language! {
    enum Lambda {
        Var(i32), // db index
        "app" = App([Id; 2]), // f, x
        "lam" = Lam([Id; 1]), // body
        Prim(egg::Symbol), // fallback, parses prims
        "programs" = Programs(Vec<Id>),
    }
}

impl Lambda {
}

type EGraph = egg::EGraph<Lambda, LambdaAnalysis>;

#[derive(Default)]
struct LambdaAnalysis;

#[derive(Debug)]
struct Data {
    upward_refs: HashSet<i32>, // "how much higher"
    inventionless_cost_any: Option<i32>,
    inventionless_cost_nolambda: Option<i32>,
    is_invention: bool, // true if theres a lambda in the eclass
    inventionful_cost_any: HashMap<Id, i32>,
    inventionful_cost_nolambda: HashMap<Id, i32>,
}

fn extract(eclass: Id, egraph: &EGraph) -> RecExpr<Lambda> {
    // expensively extracts a small program from the eclass
    let mut extractor = Extractor::new(&egraph, NaiveCost);
    let (_,p) = extractor.find_best(eclass);
    p // this is printable
}

fn extract_enode(enode: &Lambda, egraph: &EGraph) -> String {
    match enode {
        Lambda::Prim(p) => {format!("{}",p)},
        Lambda::Var(i) => {format!("{}",i)},
        Lambda::App([f,x]) => {format!("(app {} {})",extract(*f,egraph),extract(*x,egraph))},
        Lambda::Lam([b]) => {format!("(lam {})",extract(*b,egraph))},
        _ => {format!("not rendered")},
    }
}

fn min_cost(eclass:Id, inv: Option<Id>, nolambda:bool, egraph: &EGraph) -> Option<i32>{
    let ref data = egraph[eclass].data;
    match (inv,nolambda) {
        (Some(inv),true) =>  data.inventionful_cost_nolambda
            .get(&inv).cloned()
            .or(data.inventionless_cost_nolambda),
        (Some(inv),false) => data.inventionful_cost_any
            .get(&inv).cloned()
            .or(data.inventionless_cost_any),
        (None,true) => data.inventionless_cost_nolambda,
        (None,false) => data.inventionless_cost_any,
    }
}

fn beam_extract(
    root: Id,
    inv: Option<Id>, // invention to use when extracting. None means no invention.
    egraph: &EGraph,
) -> RecExpr<Lambda> {
    let root = egraph.find(root);
    let mut expr = RecExpr::default();
    beam_extract_rec(root, inv, false, egraph, &mut expr);
    expr
}

fn beam_extract_rec(
    root: Id,
    inv: Option<Id>, // invention to use when extracting. None means no invention.
    nolambda: bool, // whether or not a toplevel lambda is okay
    egraph: &EGraph,
    into_expr: &mut RecExpr<Lambda> // expr we're extracting into
) -> Id {
    let root = egraph.find(root);

    let target_cost:i32 = min_cost(root, inv, nolambda, egraph)
        .expect("attempting to extract something with infinite cost");

    if Some(root) == inv {
        assert!(target_cost == COST_TERMINAL);
        return into_expr.add(Lambda::Prim(format!("inv{}",root).into()));
    }

    for enode in egraph[root].iter() {
        match enode {
            Lambda::Prim(prim) => {
                assert!(target_cost == COST_TERMINAL);
                return into_expr.add(Lambda::Prim(*prim))
            }
            Lambda::Var(var) => {
                assert!(target_cost == COST_TERMINAL);
                return into_expr.add(Lambda::Var(*var))
            }
            Lambda::App([f,x]) => {
                let fcost = min_cost(*f, inv, true, egraph);
                let xcost = min_cost(*x, inv, false, egraph);
                if let (Some(fcost),Some(xcost)) = (fcost,xcost) {
                    if target_cost == COST_NONTERMINAL + fcost + xcost {
                        let f_id = beam_extract_rec(*f, inv, true, egraph, into_expr);
                        let x_id = beam_extract_rec(*x, inv, false, egraph, into_expr);
                        return into_expr.add(Lambda::App([f_id,x_id]))    
                    }
                }
            }
            Lambda::Lam([b]) => {
                if !nolambda {
                    let bcost = min_cost(*b, inv, false, egraph);
                    if let Some(bcost) = bcost {
                        if target_cost == COST_NONTERMINAL + bcost {
                            let b_id = beam_extract_rec(*b, inv, false, egraph, into_expr);
                            return into_expr.add(Lambda::Lam([b_id]))   
                        }
                    }
                }
            }
            Lambda::Programs(roots) => {
                let costs: Option<Vec<i32>> = roots.iter()
                    .map(|r| min_cost(*r,inv,false,egraph))
                    .collect();
                let total_cost = costs.map(|xs| xs.iter().sum());
                if total_cost == Some(target_cost) {
                    let root_ids: Vec<Id> = roots.iter()
                        .map(|r| beam_extract_rec(*r, inv, false, egraph, into_expr))
                        .collect();
                    return into_expr.add(Lambda::Programs(root_ids))
                }
            }
        }
    }
    panic!("Couldn't find the mincost node in this eclass");
}


fn best_inventions(beam: &HashMap<Id,i32>) -> Vec<Id> {
    let mut costs: Vec<(Id,i32)> = beam.iter().map(|(id,cost)|(*id,*cost)).collect();
    // INCREASING order of cost (lowest first)
    costs.sort_by(|(_,cost1),(_,cost2)| cost1.cmp(cost2));
    costs.iter().map(|(id,_)| id).cloned().collect()
}

fn noncanonical_inventions(beam: &HashMap<Id,i32>, egraph: &EGraph) -> Vec<Id> {
    beam.keys().cloned().filter(|id| canonical(id,egraph)).collect()
}

fn safe_to_remove_noncanonical(beam: &HashMap<Id,i32>, egraph: &EGraph) -> bool {
    noncanonical_inventions(beam,egraph).iter().all(|id|{
        let canonical_id = egraph.find(*id);
        beam.get(&id) == beam.get(&canonical_id)
    })
}

#[inline]
fn canonical(id:&Id, egraph: &EGraph) -> bool {
    egraph.find(*id) == *id
}

fn merge_inventionless(to: &mut Option<i32>, from: &Option<i32>) -> bool {
    match (*to,*from) {
        (Some(to_cost), Some(from_cost)) => {
            if to_cost > from_cost {
                // from is cheaper so we replace ourselves with it
                *to = Some(from_cost);
                true
            } else {
                false
            }
        },
        (None, Some(from_cost)) => {
            // we were None so we replace ourselves with from
            *to = Some(from_cost);
            true
        },
        // merging a `None` into ourselves so no change
        (_, None) => false,
    }
}

fn merge_inventionful(to: &mut HashMap<Id, i32>, from: &HashMap<Id, i32>) -> bool {
    if from.is_empty() {
        return false;
    }
    if to.is_empty() {
        *to = from.clone();
        return true;
    }
    let mut modified = false;
    for (k,v) in from.iter() {
        if to.contains_key(k) {
            if to[k] > *v {
                // from is cheaper so we replace ourselves with it
                modified = true;
                to.insert(*k,*v);
            }
        } else {
            to.insert(*k,*v);
            modified = true;
        }
    }
    modified
}

fn prune_inventionful(beam: &mut HashMap<Id, i32>, inventionless: Option<i32>) {
    // remove anything with a cost greater than the inventionless cost
    if let Some(inventionless) = inventionless {
        let mut to_remove = Vec::new();
        for (k,v) in beam.iter() {
            if *v > inventionless {
                to_remove.push(*k);
            }
        }
        for k in to_remove {
            beam.remove(&k);
        }
    }
}

fn narrow_beam(beam: &mut HashMap<Id,i32>) {
    if beam.len() < BEAM_SIZE {
        // beam.shrink_to_fit(); // todo idk if this is expensive but prob fine - does it duplicate anything?
        return
    }
    println!("Narrowing beam!");
    let num_to_drop = BEAM_SIZE - beam.len();
    let mut costs: Vec<(Id,i32)> = beam.iter().map(|(id,cost)|(*id,*cost)).collect();
    // DECREASING order of cost (since i do cost2.cmp(cost1))
    costs.sort_by(|(_,cost1),(_,cost2)| cost2.cmp(cost1));
    for (id,_) in costs.iter().take(num_to_drop) {
        beam.remove(id);
    }
    // beam.shrink_to_fit(); // todo idk if this is expensive but prob fine - does it duplicate anything?
}

impl Analysis<Lambda> for LambdaAnalysis {
    type Data = Data;
    fn merge(&self, to: &mut Data, from: Data) -> bool {

        let mut modified = false;
        assert_eq!(to.upward_refs,from.upward_refs);

        // keep the lowest inventionless cost
        modified |= merge_inventionless(&mut to.inventionless_cost_any, &from.inventionless_cost_any);
        modified |= merge_inventionless(&mut to.inventionless_cost_nolambda, &from.inventionless_cost_nolambda);
        
        // merge the inventionful costs
        modified |= merge_inventionful(&mut to.inventionful_cost_any, &from.inventionful_cost_any);
        modified |= merge_inventionful(&mut to.inventionful_cost_nolambda, &from.inventionful_cost_nolambda);

        // prune ones smaller than inventionless cost (since merging may have affected this)
        prune_inventionful(&mut to.inventionful_cost_any, to.inventionless_cost_any);
        prune_inventionful(&mut to.inventionful_cost_nolambda, to.inventionless_cost_nolambda);

        // narrow beam if needed
        narrow_beam(&mut to.inventionful_cost_any);
        narrow_beam(&mut to.inventionful_cost_nolambda);
        modified
    }

    fn make(egraph: &EGraph, enode: &Lambda) -> Data {
        let mut upward_refs: HashSet<i32> = HashSet::new();
        match enode {
            Lambda::Var(i) => {
                upward_refs.insert(*i);
            }
            Lambda::Prim(_) => {
            }
            Lambda::App([f, x]) => {
                // union of f and x
                upward_refs.extend(egraph[*f].data.upward_refs.iter());
                upward_refs.extend(egraph[*x].data.upward_refs.iter());
            }
            Lambda::Lam([b]) => {
                // body, subtract 1 from all values, remove the -1 if its in there
                upward_refs.extend(egraph[*b].data.upward_refs.iter()
                    .map(|x| x-1)
                    .filter(|x| *x >= 0));
            }
            Lambda::Programs(programs) => {
                // assert no free variables in programs
                assert!(programs.iter().all(|p| egraph[*p].data.upward_refs.is_empty()));
            }
        }
        let inventionless_cost_any = match enode {
            Lambda::Var(_) | Lambda::Prim(_) => Some(COST_TERMINAL),
            Lambda::App([f,x]) => {
                match (egraph[*f].data.inventionless_cost_nolambda, egraph[*x].data.inventionless_cost_any) {
                    (Some(f), Some(x)) => Some(COST_NONTERMINAL+f+x),
                    _ => None,
                }
            }
            Lambda::Lam([b]) => {
                egraph[*b].data.inventionless_cost_any.map(|x| x+COST_NONTERMINAL)
            }
            Lambda::Programs(ps) => {
                // type annotate to make collect() turn a vec<opt<>> into an opt<vec<>>
                let costs: Option<Vec<i32>> = ps.iter().map(|p| egraph[*p].data.inventionless_cost_any).collect();
                costs.map(|xs| xs.iter().sum())
            }
        };
        let inventionless_cost_nolambda = match enode {
            Lambda::Lam([_]) => None,
            _ => inventionless_cost_any
        };
        let mut inventionful_cost_any: HashMap<Id,i32> = match enode {
            Lambda::Var(_) | Lambda::Prim(_) => HashMap::new(),
            Lambda::App([f,x]) => {

                let ref f_nolambda = egraph[*f].data.inventionful_cost_nolambda;
                let ref x_any = egraph[*x].data.inventionful_cost_any;
                let f_inventionless = egraph[*f].data.inventionless_cost_nolambda;
                let x_inventionless = egraph[*x].data.inventionless_cost_any;
                
                debug_assert!(safe_to_remove_noncanonical(f_nolambda, egraph));
                debug_assert!(safe_to_remove_noncanonical(x_any,      egraph));

                // figure out which (canonical) inventions helped `f` and `x` and union them
                let mut inventions: HashSet<Id> = HashSet::new();
                inventions.extend(f_nolambda.keys().cloned().filter(|id|canonical(id,egraph)));
                inventions.extend(x_any     .keys().cloned().filter(|id|canonical(id,egraph)));
                
                // costs with inventions as 1 + fcost + xcost. Use inventionless cost as a default.
                // if either fcost or xcost is None (ie infinite)
                inventions.iter().filter_map(|invention| {
                    let fcost = f_nolambda.get(invention).cloned()
                        .or(f_inventionless);
                    let xcost = x_any.get(invention).cloned()
                        .or(x_inventionless);
                    if let (Some(fcost), Some(xcost)) = (fcost, xcost) {
                        let cost = COST_NONTERMINAL+fcost+xcost;
                        debug_assert!(cost <= inventionless_cost_any.unwrap_or(i32::MAX), "cost {} is greater than inventionless cost {} for {} and inv {}", cost, inventionless_cost_any.unwrap_or(i32::MAX), extract_enode(enode, egraph), extract(*invention, egraph));
                        Some((*invention, COST_NONTERMINAL+fcost+xcost))
                    } else {
                        None
                    }
                }).collect()
            }
            Lambda::Lam([b]) => {
                // just map +1 over the costs
                let ref b_any = egraph[*b].data.inventionful_cost_any;
                debug_assert!(safe_to_remove_noncanonical(b_any, egraph));
                b_any.iter().clone()
                    .filter(|(k,_)|canonical(k, egraph))
                    .map(|(k,v)| (*k,*v+COST_NONTERMINAL)).collect()
            }
            Lambda::Programs(roots) => {
                // note we only add Programs node after running the egraph so it doesnt matter how expensive this is
                debug_assert!(roots.iter().all(|r|safe_to_remove_noncanonical(&egraph[*r].data.inventionful_cost_any,egraph)));
                // union together all the useful inventions of diff programs
                let inventions: Vec<Id> = roots.iter()
                    .map(|r| egraph[*r].data.inventionful_cost_any.keys().cloned())
                    .flatten()
                    .filter(|id| canonical(id, egraph))
                    .collect();
                // count num occurences of each invention
                let mut counts: HashMap<Id,i32> = inventions.iter().map(|i| (*i,0)).collect();
                for inv in inventions {
                    counts.insert(inv, counts[&inv] + 1);
                }

                let inventions: Vec<Id> = counts.iter()
                    .filter_map(|(i,c)| if *c > 1 { Some(*i) } else { None }).collect();
                
                // get the inventionless costs (turns a vector of options into an option of vectors via type annotation)
                let costs_inventionless: Option<Vec<i32>> = roots.iter()
                    .map(|r| egraph[*r].data.inventionless_cost_any).collect();
                assert_ne!(costs_inventionless, None); // something is wrong if we cant write a toplevel program inventionlessly...
                let costs_inventionless = costs_inventionless.unwrap();

                inventions.iter().map(|invention| {
                    let cost = roots.iter().enumerate()
                        .map(|(idx,root)| {
                            egraph[*root].data.inventionful_cost_any.get(invention).cloned()
                            .unwrap_or(costs_inventionless[idx])
                        })
                        .sum();
                    (*invention, cost)
                }).collect()
            }
        };
        narrow_beam(&mut inventionful_cost_any);
        let inventionful_cost_nolambda = match enode {
            Lambda::Lam([_]) => HashMap::new(),
            _ => inventionful_cost_any.clone()
        };
        let is_invention = upward_refs.is_empty() && match enode { Lambda::Lam(_) => true, _ => false };

        Data { upward_refs: upward_refs,
               inventionless_cost_any: inventionless_cost_any,
               inventionless_cost_nolambda: inventionless_cost_nolambda,
               inventionful_cost_any: inventionful_cost_any,
               inventionful_cost_nolambda: inventionful_cost_nolambda,
               is_invention: is_invention }
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        if egraph[id].data.is_invention {
            debug_assert_eq!(id,egraph.find(id)); // just wanna make sure modify always gets called w canonicals, else we want a find() call here

            // if we just merged two is_invention eclasses we wanna remove any leftover noncanonical inventions
            if egraph[id].data.inventionful_cost_any.values().any(|&v| v == 0) {
                let mut to_remove = Vec::new();
                for (k,v) in egraph[id].data.inventionful_cost_any.iter() {
                    if *v == COST_TERMINAL {
                        debug_assert_eq!(id,egraph.find(*k));
                        to_remove.push(*k);
                    }
                }
                for k in to_remove {
                    egraph[id].data.inventionful_cost_any.remove(&k);
                }
            }
            // add our new invention!
            egraph[id].data.inventionful_cost_any.insert(id, COST_TERMINAL);
            egraph[id].data.inventionful_cost_nolambda.insert(id, COST_TERMINAL);
    }
    }
}

fn var(s: &str) -> Var {
    s.parse().unwrap()
}

// outward references are $0 or $1 but not greater
fn upward_refs_lt_2(v: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _, subst| {
        match egraph[subst[v]].data.upward_refs.iter().min() {
            Some(min) => *min < 2,
            None => true
        }
    }
}

// Theres no $0 upward ref
fn zero_not_in_upward_refs(v: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _, subst| !egraph[subst[v]].data.upward_refs.contains(&0)
}

// here I copied the ConditionEqual code to make my own
pub struct ConditionNotEqual<A1, A2>(pub A1, pub A2);

impl<L: Language> ConditionNotEqual<Pattern<L>, Pattern<L>> {
    pub fn parse(a1: &str, a2: &str) -> Self {
        Self(a1.parse().unwrap(), a2.parse().unwrap())
    }
}

impl<L, N, A1, A2> Condition<L, N> for ConditionNotEqual<A1, A2>
where
    L: Language,
    N: Analysis<L>,
    A1: Applier<L, N>,
    A2: Applier<L, N>,
{
    fn check(&self, egraph: &mut egg::EGraph<L, N>, eclass: Id, subst: &Subst) -> bool {
        let a1 = self.0.apply_one(egraph, eclass, subst);
        let a2 = self.1.apply_one(egraph, eclass, subst);
        assert_eq!(a1.len(), 1);
        assert_eq!(a2.len(), 1);
        a1[0] != a2[0]
    }

    fn vars(&self) -> Vec<Var> {
        let mut vars = self.0.vars();
        vars.extend(self.1.vars());
        vars
    }
}

fn shift(e: Id, incr_by: i32, egraph: &mut EGraph) -> Option<Id> {
    recursive_var_mod(
        |actual_idx, depth, which_upward_ref, egraph| {
            if actual_idx + incr_by >= ARGC {
                return None // $3+ get pruned
            } 
            Some(egraph.add(Lambda::Var(actual_idx + incr_by)))
        },
        e,egraph
    )
}

struct Shifter {
    incr_by: i32, // how much to increment by eg +1 or -1
    to_shift: Var, // expression to shift
    rhs: Pattern<Lambda>, // expr to be unified with original LHS - but with to_shift modified!
}

impl Applier<Lambda, LambdaAnalysis> for Shifter {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        eclass: Id,
        subst: &Subst) -> Vec<Id> 
        {
            // println!("initial: left={} right={}", extract(subst[var("?left")],egraph),extract(subst[var("?right")],egraph));
            let e_new = shift(subst[self.to_shift], self.incr_by, egraph);
            if e_new.is_none() { return vec![]; }
            let mut subst = subst.clone(); // they do this in the example
            subst.insert(self.to_shift, e_new.unwrap()); // overwrites the e with shifted_e
            let res = self.rhs.apply_one(egraph, eclass, &subst);
            // println!("{} | left={} right={}", self.rhs.ast, extract(subst[var("?left")],egraph),extract(subst[var("?right")],egraph));
            // println!("{} {}", extract(eclass,egraph), extract(res[0],egraph));
            // egraph.union(res[0],eclass);
            // println!("safe");
            res

            // warning: there are unions that happen during class_shift internally which arent reported
            // to apply_matches. That seems totally okay though from reading the source code (which only
            // uses the Ids you return from apply_one to figure out how many places were modified)
    }
}

fn inline(e: Id, replace_with: Id, egraph: &mut EGraph) -> Option<Id> {
    recursive_var_mod(
        |actual_idx, depth, which_upward_ref, egraph| {
            if which_upward_ref == 0 {
                // ShifterVM { incr_by: depth }.recursive_var_mod(replace_with, egraph)
                shift(replace_with, depth, egraph)
            } else {
                // we need to decrement this by 1 since its a pointer above the lambda we removed
                Some(egraph.add(Lambda::Var(actual_idx - 1)))
            }
        },
        e,egraph
    )
}

struct Inliner {
    replace_with: Var, // what to inline
    inline_into: Var, // what to inline it into
    rhs: Pattern<Lambda>, // expr to be unified with original LHS - but with inline_into modified!
}


impl Applier<Lambda, LambdaAnalysis> for Inliner {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        eclass: Id,
        subst: &Subst) -> Vec<Id> 
        {
            let e_new = inline(subst[self.inline_into], subst[self.replace_with], egraph);
            if e_new.is_none() { return vec![]; }
            let mut subst = subst.clone(); // they do this in the example
            subst.insert(self.inline_into, e_new.unwrap()); // overwrites the e with shifted_e
            self.rhs.apply_one(egraph, eclass, &subst)
            // warning: there are unions that happen during inline internally which arent reported
            // to apply_matches. That seems totally okay though from reading the source code (which only
            // uses the Ids you return from apply_one to figure out how many places were modified)
    }
}

struct ApplamBubbleOverLam {
    arg: Var,
    body: Var,
    rhs: Pattern<Lambda>,
}
impl Applier<Lambda, LambdaAnalysis> for ApplamBubbleOverLam {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        eclass: Id,
        subst: &Subst) -> Vec<Id> 
        {
            // downshift arg
            let arg_new = shift(subst[self.arg], -1, egraph);

            // increment upward_refs to 0 and decrement upward_refs to 1 in the body
            let body_new = recursive_var_mod(
                |actual_idx, depth, which_upward_ref, egraph| {
                    if which_upward_ref == 0 {
                        Some(egraph.add(Lambda::Var(actual_idx + 1)))
                    } else if which_upward_ref == 1 {
                        Some(egraph.add(Lambda::Var(actual_idx - 1)))
                    }
                    else {
                        Some(egraph.add(Lambda::Var(actual_idx)))
                    }
                },
                subst[self.body], egraph
            );

            if body_new.is_none() || arg_new.is_none() { return vec![]; }

            // construct the rhs an return it!
            let mut subst = subst.clone(); // they do this in the example
            subst.insert(self.arg, arg_new.unwrap());
            subst.insert(self.body, body_new.unwrap());
            self.rhs.apply_one(egraph, eclass, &subst)
    }
}




fn recursive_var_mod(
    var_mod: impl Fn(i32, i32, i32, &mut EGraph) -> Option<Id>,
    eclass:Id,
    egraph: &mut EGraph
    ) -> Option<Id>
    {
        recursive_var_mod_helper(
            &var_mod,
            eclass,
            0,
            egraph,
            &mut HashMap::new(),
        )
}

fn recursive_var_mod_helper(
    var_mod: &impl Fn(i32, i32, i32, &mut EGraph) -> Option<Id>,
    eclass:Id,
    depth: i32,
    egraph: &mut EGraph,
    seen : &mut HashMap<(Id,i32),Option<Id>>,
    ) -> Option<Id>
    {
        // important invariant: a $i with i==depth would be a $0 pointer at the top level
        // meaning i<depth is an internal pointer that doesnt break the top level

        let key = (eclass,depth);
        // check if we've seen this before (ie we're looping). If so return whatever we got last time we calculated it.
        if seen.contains_key(&key) {
            return seen[&key];
        }
        if egraph[eclass].data.upward_refs.iter().all(|i| *i < depth) {
            // from our invariant (above) we know i<depth is an internal pointer that doesnt point out of the top level
            seen.insert(key, Some(eclass));
            return Some(eclass)
        }
        // we temporarily insert None to break any loops (ie if a recursive call asks us to compute the same thing). Note that at the end of this function we insert the real result in the cache
        seen.insert(key, None);

        // ALL children need modification (since they all contain us in some form or need decrementing)
        // we need to fully clone all the ENodes so we can let go of the borrow
        // of `egraph` (which is happening bc of the iter()) so we can use `egraph` in the body of the loop
        let enodes: Vec<Lambda> = egraph[eclass].iter().cloned().collect();
        let eclasses_to_union : Vec<Id> = enodes.iter().cloned().filter_map(|enode| {
            match enode {
                Lambda::Var(i) => {
                    assert!(i >= depth); // otherwise we should have returned earlier
                    // by our invariant be have i-depth as the toplevel version of this index
                    var_mod(i, depth, i-depth, egraph)
                }
                Lambda::Prim(_) => {
                    panic!("attempted to shift Prim, which shouldnt be attempted since Prim never has free vars")
                }
                Lambda::App([f, x]) => {
                    // recurse in each (class shift will return early if no shifting is needed) and build a new App
                    let fnew_opt = recursive_var_mod_helper(var_mod, f, depth, egraph, seen);
                    let xnew_opt = recursive_var_mod_helper(var_mod, x, depth, egraph, seen);
                    match (fnew_opt,xnew_opt) {
                        (Some(fnew),Some(xnew)) => Some(egraph.add(Lambda::App([fnew, xnew]))),
                        _ => None,
                    }
                }
                Lambda::Lam([b]) => {
                    // increment depth
                    recursive_var_mod_helper(var_mod, b, depth+1, egraph, seen)
                    .map(|bnew| egraph.add(Lambda::Lam([bnew])))
                }
                Lambda::Programs(_) => {
                    panic!("attempted to shift a Programs node")
                }
            }
        }).collect();
        
        // todo figure out why this fires
        // assert!(!eclasses_to_union.contains(&eclass)); // should pass else why didnt we return early
        // union the eclasses
        if eclasses_to_union.is_empty() {
            seen.insert(key, None); // redundant since we did this above to prevent loops anyways
            return None
        }
        let new_eclass = egraph.find(eclasses_to_union[0]); // dont need to canonicalize like this, but will prob speed up the unionfind later
        eclasses_to_union.iter().skip(1).for_each(|id| {egraph.union(*id, new_eclass);});
        seen.insert(key, Some(new_eclass));
        Some(new_eclass)
}



// this is the cost where applams are allowed - just a very naive cost
struct NaiveCost;
impl CostFunction<Lambda> for NaiveCost {
    type Cost = i32;
    fn cost<C>(&mut self, enode: &Lambda, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost
    {
        match enode {
            Lambda::Var(_) | Lambda::Prim(_) => COST_TERMINAL,
            Lambda::App([f, x]) => COST_NONTERMINAL + costs(*f) + costs(*x),
            Lambda::Lam([b]) => COST_NONTERMINAL + costs(*b),
            Lambda::Programs(ps) => ps.iter().map(|p| costs(*p)).sum(),
        }
    }
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
    let runner = Runner::default()
        .with_egraph(egraph)
        .with_iter_limit(400)
        .with_scheduler(SimpleScheduler)
        .with_time_limit(core::time::Duration::from_secs(200))
        .with_node_limit(3000000);
    
    // i know this is awful but its the best i can do that makes rust happy
    let runner = if !render {runner} else {runner.with_hook(
        {
            let out_dir = out_dir.clone(); // silly thing to clone into the closure
            move |runner|{
                let iter = runner.iterations.len();
                println!("Iter {}: {}", iter, egraph_info(&runner.egraph));
                save(&runner.egraph, format!("3_propagate_{}",iter).as_str(), &out_dir);
                Ok(())
            }
    })};

    
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


fn rule_map() -> HashMap<String,Rewrite<Lambda, LambdaAnalysis>> {
    vec![

        // // applam-intro: this rule matches any node and rewrites it to be an applam with
        // // $0 in the body and the subtree in the arg. Applies to all nodes
        // // not just leaves. This rule necessarily introduces a self loop.
        // rw!("applam-intro"; "(?subtree)" => "(app (lam 0) ?subtree)"
        // // conditions: abstracting the identity out just leads to insane blowups everywhere as
        // // a result of (app (lam 0) (lam 0)) == (lam 0) which lets you build infinite
        // // trees of things and it just gets messy.
        // if ConditionNotEqual::parse("(?subtree)", "(lam 0)")
        // if ConditionNotEqual::parse("(?subtree)", "(0)") // todo unclear if this does anything -- and why dont we do it for the other vars? worth considering
        // ),

        rw!("alt-intro-left"; "(app ?left ?right)" =>
        {Shifter {
            incr_by: 1, // how much to increment by eg +1 or -1
            to_shift: var("?right"), // expression to shift
            rhs: "(app (lam (app 0 ?right)) ?left)".parse().unwrap(), // expr to be unified with original LHS - but with to_shift modified!
        }}
        ),

        rw!("alt-intro-right"; "(app ?left ?right)" =>
        {Shifter {
            incr_by: 1, // how much to increment by eg +1 or -1
            to_shift: var("?left"), // expression to shift
            rhs: "(app (lam (app ?left 0)) ?right)".parse().unwrap(), // expr to be unified with original LHS - but with to_shift modified!
        }}
        ),

        //todo need to add an alt-intro for jumping a lambda right away as in `(lam y)` when you abstract y


        // // these `-unrestrained` rules are version of `applam-bubble-from-left` and `applam-bubble-from-right` that are
        // // have less restrictions on where they can apply. In particular theyre allowed to bubble up an identity applam.
        // // which causes huge blowups if used repeatedly but is important to use once at the start.
        // rw!("applam-bubble-from-left-unrestrained"; "(app (app (lam ?body) ?arginner) ?argouter)"
        // => {Shifter {
        //     incr_by: 1, // how much to increment by eg +1 or -1
        //     to_shift: var("?argouter"), // expression to shift
        //     rhs: "(app (lam (app ?body ?argouter)) ?arginner)".parse().unwrap(), // expr to be unified with original LHS - but with to_shift modified!
        // }}
        // // condition accounts for avoiding blowup from bubbling out of self-loop. If the two apps are the same eclass already it doesnt bubble the lower one up. Not sure if this will limit anything, its just my quick fix.
        // // todo should be able to safely remove condition
        // if ConditionNotEqual::parse("(app (app (lam ?body) ?arginner) ?argouter)", "(app (lam ?body) ?arginner)")
        // ),
        // rw!("applam-bubble-from-right-unrestrained"; "(app ?f (app (lam ?body) ?arg))"
        //     => {Shifter {
        //         incr_by: 1, // how much to increment by eg +1 or -1
        //         to_shift: var("?f"), // expression to shift
        //         rhs: "(app (lam (app ?f ?body)) ?arg)".parse().unwrap(), // expr to be unified with original LHS - but with to_shift modified!
        //     }}
        //     // condition accounts for avoiding blowup from bubbling out of self-loop. If the two apps are the same eclass already it doesnt bubble the lower one up. Not sure if this will limit anything, its just my quick fix.
        //     // todo should be able to safely remove condition
        //     if ConditionNotEqual::parse("(app ?f (app (lam ?body) ?arg))", "(app (lam ?body) ?arg)")
        // ),

        // applam-bubble-from-left and applam-bubble-from-right:
        // these are the rules for bubbling an applam up out of the left and right sides
        // of an app respectively. In the left case, the `arg` of the above app will be dropped
        // below the lambda meaning any pointers in it that point above its own root need
        // incrementing. In the right case its the same but with the `body` of the above app.
        rw!("applam-bubble-from-left"; "(app (app (lam ?body) ?arginner) ?argouter)"
            => {Shifter {
                incr_by: 1, // how much to increment by eg +1 or -1
                to_shift: var("?argouter"), // expression to shift
                rhs: "(app (lam (app ?body ?argouter)) ?arginner)".parse().unwrap(), // expr to be unified with original LHS - but with to_shift modified!
            }}
            // condition accounts for avoiding blowup from bubbling out of self-loop. If the two apps are the same eclass already it doesnt bubble the lower one up. Not sure if this will limit anything, its just my quick fix.
            // todo maybe only the second condition is needed, worth seeing
            // if ConditionNotEqual::parse("(app (app (lam ?body) ?arginner) ?argouter)", "(app (lam ?body) ?arginner)")
            // if ConditionNotEqual::parse("(?body)", "(0)")
        ),
        rw!("applam-bubble-from-right"; "(app ?f (app (lam ?body) ?arg))"
            => {Shifter {
                incr_by: 1, // how much to increment by eg +1 or -1
                to_shift: var("?f"), // expression to shift
                rhs: "(app (lam (app ?f ?body)) ?arg)".parse().unwrap(), // expr to be unified with original LHS - but with to_shift modified!
            }}
            // condition accounts for avoiding blowup from bubbling out of self-loop. If the two apps are the same eclass already it doesnt bubble the lower one up. Not sure if this will limit anything, its just my quick fix.
            // if ConditionNotEqual::parse("(app ?f (app (lam ?body) ?arg))", "(app (lam ?body) ?arg)")
            // todo maybe only the second condition is needed, worth seeing
            // if ConditionNotEqual::parse("(?body)", "(0)")
        ),

        // applam-merge: this rule says when you have an app of two applams
        // that have a shared arg, you can bubble the lambda up above the inner
        // applications and merge them (body of left gets applied to body of right).
        rw!("applam-merge"; "(app (app (lam ?body1) ?argshared) (app (lam ?body2) ?argshared))"
            => "(app (lam (app ?body1 ?body2)) ?argshared)"
        ),
        
        // this is a subset of the applam-inline rule which catches the same immediately
        // anyways without deep analysis. Yes good to turn this on if you turn that one off.
        // simple-inline: this rule does inlining in the special case where no shifting is needed,
        // which turns out to be really useful for proving equivalences that avoid blowups
        // (notice the RHS introduces no new terms so its purely compressive!)
        // rw!("simple-inline"; "(lam (app ?f 0))" => "(?f)"
        //     if no_upward_refs(var("?f"))),
        
        // applam-inline: this inlines an applam to destroy it. I have a feeling itll help kill
        // some infinities by proving equivalences. But I also fear it. Though it doesnt introduce new
        // lambdas so it seems like it might not blow things up.
        // todo its unclear if this rule is actually useful. It used to avoid certain blowups but I've fixed those in other ways now.
        // todo ...And it still decreases the egraph size but at massive compute cost so worth seeing if I can fix the blowups
        // todo ...in alternative ways and not need this inlining
        rw!("applam-inline"; "(app (lam ?body) ?arg)"
            => {Inliner {
                replace_with: var("?arg"), // what to inline
                inline_into: var("?body"), // what to inline it into
                rhs: "(?body)".parse().unwrap(), // expr to be unified with original LHS - but with inline_into modified!
                // abort_if_equal: "(lam 0)".parse().unwrap(),
            }}
        ),

        // applam-multiarg: this is just the transformation that takes an applam applam setup and moves it so
        // its appapplamlam. Basically it lets you stack two lambdas on top of each other.
        // todo we need to make a 3arg version of this
        // todo this might be replacable by an unrestrained version of applam-bubble-over-lam which is worth experimenting with in case it doesnt cause a blowup
        rw!("applam-multiarg"; "(app (lam (app (lam ?body) ?arginner)) ?argouter)"
            => {Shifter {
                incr_by: -1, // how much to increment by eg +1 or -1
                to_shift: var("?arginner"), // expression to shift
                rhs: "(app (app (lam (lam (?body))) ?argouter) ?arginner)".parse().unwrap(), // expr to be unified with original LHS - but with to_shift modified!
            }}
            // condition: cant raise arginner above argouter if arginner points to argouter
            if zero_not_in_upward_refs(var("?arginner"))
            // condition: since we're using this to expose inventions, they must not have any upward refs
            if upward_refs_lt_2(var("?body"))
        ),

        // applam-bubble-over-lam: this is the transformation that lets you bubble an applam up over a lambda.
        rw!("applam-bubble-over-lam-unrestrained"; "(lam (app (lam ?body) ?arg))"
            // details:
            // - downshift ?arg
            // - since the lambdas swapped places, the body still has just as many lambda over it
            //     as before, but we need to increment all the pointers to the previously-lower lambda
            //     and decrement all the pointers to the previously-higher lambda
            => {ApplamBubbleOverLam {
                arg: var("?arg"),
                body: var("?body"),
                rhs: "(app (lam (lam ?body)) ?arg)".parse().unwrap(),
            }}
            // condition: cant raise arg above a lambda that it points to
            if zero_not_in_upward_refs(var("?arg"))
            // if ConditionNotEqual::parse("(?body)", "(0)")
        ),

        // this `-if-under-lam` version is the same as the `-unrestrained` version but only
        // fires when theres a lambda wrapping everything
        rw!("applam-bubble-over-lam-if-under-lam"; "(lam (lam (app (lam ?body) ?arg)))"
            => {ApplamBubbleOverLam {
                arg: var("?arg"),
                body: var("?body"),
                rhs: "(lam (app (lam (lam ?body)) ?arg))".parse().unwrap(),
            }}
            // condition: cant raise arg above a lambda that it points to
            if zero_not_in_upward_refs(var("?arg"))
            // if ConditionNotEqual::parse("(?body)", "(0)")
        ),

        // this `-if-arg-of-app` version is the same as the `-unrestrained` version but only
        // fires when the lam is the right hand argument of an application
        rw!("applam-bubble-over-lam-if-arg-of-app"; "(app ?f (lam (app (lam ?body) ?arg)))"
            => {ApplamBubbleOverLam {
                arg: var("?arg"),
                body: var("?body"),
                rhs: "(app ?f (app (lam (lam ?body)) ?arg))".parse().unwrap(),
            }}
            // condition: cant raise arg above a lambda that it points to
            if zero_not_in_upward_refs(var("?arg"))
            if ConditionNotEqual::parse("(?body)", "(0)") // todo explore removing this not sure if its limiting us
            // if ConditionNotEqual::parse("(?f)", "(lam 0)")
        ),

    ].into_iter().map(|r| (r.name().to_string(),r)).collect()
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

fn main() {
    env_logger::init();

    let x: RecExpr<Lambda> = "(app x x)".parse().unwrap();
    println!("{}: {:?}", x, x);
    
    panic!("done");

    // create a new directory for logging outputs
    let out_dir: String = format!("target/{}",timestamp());
    let out_dir_p = std::path::Path::new(out_dir.as_str());
    assert!(!out_dir_p.exists());
    std::fs::create_dir(out_dir_p).unwrap();


    let mut egraph: EGraph = Default::default();

    // first dreamcoder program
    let programs: Vec<RecExpr<Lambda>> = vec![
        // "(lam (app - 0))",
        "(lam (app (app (app + x) y) z))",
        "(lam (app (app + x) (app - y)) )",

        // "(lam (lam (app (app - x) y)))",

        // "(app - (lam (lam (app + 0))))",

        // "(app - (lam (app + y)))",
        // "(lam (app - (lam (lam (y)))))",


        // 116/74 (no-inline: 136/94)
        // "(lam (app - (app + y)))",

        // 55/34 (no-inline: 63/42)
        // "(app - (app + y))",

        // "(lam (app x y))",




        // "(lam (app - y))",

        // first:
        // "(lam (app (app (app logo_forLoop t3) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t3)) 0)))) 0))",
        // second:
        // "(lam (app (app (app logo_forLoop t3) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t3)) 0)))) 0))",

        // "(lam (app (app (app logo_forLoop t8) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t8)) 0)))) 0))",
        // "(lam (app (app (app logo_forLoop t8) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t8)) 0)))) 0))",
        // "(lam (app (app (app logo_forLoop t9) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t9)) 0)))) 0))",
        // "(lam (app (app (app logo_forLoop t9) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t9)) 0)))) 0))",
        // "(lam (app (app (app logo_forLoop logo_IFTY) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_epsL) t1)) logo_epsA) 0)))) 0))",
        // "(lam (app (app (app logo_forLoop logo_IFTY) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_epsL) t2)) logo_epsA) 0)))) 0))",
        // "(lam (app (app (app logo_forLoop logo_IFTY) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_epsL) t5)) logo_epsA) 0)))) 0))",
        // "(lam (app (app (app logo_forLoop t4) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t4)) 0)))) 0))",
        // "(lam (app (app (app logo_FWRT logo_UL) logo_ZA) 0))",
        // "(lam (app (app (app logo_FWRT logo_ZL) (app (app logo_DIVA logo_UA) t4)) (app (app (app logo_FWRT logo_UL) logo_ZA) 0)))",
        // "(lam (app (app (app logo_forLoop t4) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t4)) 0)))) 0))",
        // "(lam (app (app (app logo_forLoop t5) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t5)) 0)))) 0))",
        // "(lam (app (app (app logo_forLoop t5) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t5)) 0)))) 0))",
        // "(lam (app (app (app logo_forLoop t6) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t6)) 0)))) 0))",
        // "(lam (app (app (app logo_forLoop t9) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) 1)) (app (app logo_DIVA logo_UA) t4)) 0)))) 0))",
        // "(lam (app (app (app logo_forLoop t6) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t6)) 0)))) 0))",
        // "(lam (app (app (app logo_forLoop t7) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t7)) 0)))) 0))",
            ].iter().map(|p| p.parse().unwrap()).collect();
    let roots: Vec<Id> = programs.iter().map(|p| egraph.add_expr(&p)).collect();
    egraph.rebuild(); // this is VERY important to run before you try applying any searches or rewrites

    let applam:Pattern<Lambda> = "(app (lam ?body) ?arg)".parse().unwrap();
    assert!(applam.search(&egraph).is_empty(),
        "Normal dreamcoder programs never have unapplied lambdas in them. 
        it's important to avoid this because if we abstract this term with
        applam-intro then prove its equivalence to the identity function (lam 0),
        the search will explode (which is why we forbid lam 0 in the first place).
        If you really want to use this program, run applam-inline on it until it no
        longer has an applam (assuming its possible to put it in normal form without
        looping infinitely)");

    // let rules_ = rules();

    println!("Available rules:");
    rule_map().keys().for_each(|r| println!("\t{}", r));


    // *** ACTUAL EGRAPH RUNNING ***


    println!("Initial egraph:\n\t{}\n", egraph_info(&egraph));
    save(&egraph, "0_init", &out_dir);

    // apply_everywhere_once(&["applam-intro"], &mut egraph);
    // println!("After applam-intro:\n\t{}\n", egraph_info(&egraph));
    // save(&egraph, "1_applam-intro", &out_dir);

    // apply_everywhere_once(&["applam-bubble-from-left-unrestrained",
    //                         "applam-bubble-from-right-unrestrained"], &mut egraph);
    // println!("After unrestrained bubble:\n\t{}\n", egraph_info(&egraph));
    // save(&egraph, "2_applam-bubble-unrestrained", &out_dir);

    // println!("hits {}",search("(lam ?a)", &egraph).len());

    apply_everywhere_once(&["alt-intro-left","alt-intro-right"], &mut egraph);
    println!("After alt intro:\n\t{}\n", egraph_info(&egraph));


    // run propagation rules until saturation
    println!("*** Propagation");
    let mut egraph = saturate(&[
                     "applam-bubble-from-left",
                     "applam-bubble-from-right",
                    //  "applam-bubble-over-lam-if-under-lam",
                    //  "applam-bubble-over-lam-if-arg-of-app",
                    //  "applam-bubble-over-lam-unrestrained",
                     "applam-merge",
                    //  "applam-inline",
                     ], false, out_dir.to_string(), egraph);

    // save(&egraph, "4_propagate", &out_dir);

    // todo i just put an inline in here
    apply_everywhere_once(&["applam-inline"], &mut egraph);
    println!("After inline:\n\t{}\n", egraph_info(&egraph));

    apply_everywhere_once(&["applam-multiarg"], &mut egraph);
    println!("After multiarg:\n\t{}\n", egraph_info(&egraph));
    // save(&egraph, "5_multiarg", &out_dir);
    apply_everywhere_once(&["applam-inline"], &mut egraph);
    println!("After inline:\n\t{}\n", egraph_info(&egraph));

    save(&egraph, "final", &out_dir);

    // *** END OF ACTUAL EGRAPH RUNNING ***





    if false {

    // add a parent Programs node
    let programs_id = egraph.add(Lambda::Programs(roots.clone()));
    // rebuild the invariants bc we're totally done!
    egraph.rebuild();

    println!("Inventionless (cost={:?}):\n{}\n",
        min_cost(programs_id, None, false, &egraph),
        beam_extract(programs_id, None, &egraph)
    );

    let top_invs: Vec<Id> = best_inventions(&egraph[programs_id].data.inventionful_cost_any);
    assert!(top_invs.iter().all(|id| canonical(id,&egraph)));

    println!("Found {} Inventions that helped at the top level", top_invs.len());

    for (i,inv) in top_invs.iter().take(5).enumerate() {
        println!("\nInvention {} (id={}) (inv_cost={:?}; rewritten_cost={:?}):\n{}\n Rewritten:\n{}",
            i,
            inv,
            min_cost(*inv, None, false, &egraph),
            min_cost(programs_id, Some(*inv), false, &egraph),
            beam_extract(*inv, None, &egraph),
            beam_extract(programs_id, Some(*inv), &egraph),
        );
    }

    for inv in top_invs.iter(){
        let expr = beam_extract(*inv, None, &egraph);
        if expr.to_string().contains("logo_forLoop") {
            println!("Found!: {}", expr);
        }
    }


    for i in 0..10 {
        println!("{}: {}", i, search(format!("({})",i).as_str(),&egraph).len());
    }
    }


    
}
