use egg::{rewrite as rw, *};
use std::collections::{HashSet,HashMap};

extern crate log;

const ARGC: i32 = 2;
const BEAM_SIZE: usize = 1000000;


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
    // free: HashSet<Id>,
    // constant: Option<(Lambda, PatternAst<Lambda>)>,
}

fn extract(eclass: Id, egraph: &EGraph) -> RecExpr<Lambda> {
    // expensively extracts a small program from the eclass
    let mut extractor = Extractor::new(&egraph, ProgramCost);
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

    let ref data = egraph[root].data;


    let target_cost:i32 = min_cost(root, inv, nolambda, egraph)
        .expect("attempting to extract something with infinite cost");

    if Some(root) == inv {
        assert!(target_cost == 100);
        return into_expr.add(Lambda::Prim(format!("inv{}",root).into()));
    }

    for enode in egraph[root].iter() {
        match enode {
            Lambda::Prim(prim) => {
                assert!(target_cost == 100);
                return into_expr.add(Lambda::Prim(*prim))
            }
            Lambda::Var(var) => {
                assert!(target_cost == 100);
                return into_expr.add(Lambda::Var(*var))
            }
            Lambda::App([f,x]) => {
                let fcost = min_cost(*f, inv, true, egraph);
                let xcost = min_cost(*x, inv, false, egraph);
                if let (Some(fcost),Some(xcost)) = (fcost,xcost) {
                    if target_cost == 1 + fcost + xcost {
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
                        if target_cost == 1 + bcost {
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







// // min cost among two options, and Some beats None (unlike normal Option.min())
// fn mincost_discard_none(a: &Option<i32>, b: &Option<i32>) -> Option<i32> {
//     match (*a,*b) {
//         (Some(a), Some(b)) => {Some(a.min(b))},
//         (Some(a), None) => {Some(a)},
//         (None, Some(b)) => {Some(b)},
//         (None, None) => {None},
//     }
// }


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
                // upward_refs: singleton set
                upward_refs.insert(*i);
                
            }
            Lambda::Prim(_) => {
                // upward_refs: empty set
            }
            Lambda::App([f, x]) => {
                // upward_refs: union of f and x
                upward_refs.extend(egraph[*f].data.upward_refs.iter());
                upward_refs.extend(egraph[*x].data.upward_refs.iter());
            }
            Lambda::Lam([b]) => {
                // upward_refs: body, subtract 1 from all values, remove the -1 if its in there
                upward_refs.extend(egraph[*b].data.upward_refs.iter()
                    .map(|x| x-1)
                    .filter(|x| *x >= 0));
            }
            Lambda::Programs(programs) => {
                // upward_refs: empty set
                // assert no free variables in programs
                assert!(programs.iter().all(|p| egraph[*p].data.upward_refs.is_empty()));
            }
        }
        let inventionless_cost_any = match enode {
            Lambda::Var(_) | Lambda::Prim(_) => Some(100),
            Lambda::App([f,x]) => {
                match (egraph[*f].data.inventionless_cost_nolambda, egraph[*x].data.inventionless_cost_any) {
                    (Some(f), Some(x)) => Some(1+f+x),
                    _ => None,
                }
            }
            Lambda::Lam([b]) => {
                egraph[*b].data.inventionless_cost_any.map(|x| x+1)
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
                // if egraph[*f].data.is_invention {
                //     inventions.insert(*f);
                // }
                // if egraph[*x].data.is_invention {
                //     inventions.insert(*x);
                // }
                
                // costs with inventions as 1 + fcost + xcost. Use inventionless cost as a default.
                // if either fcost or xcost is None (ie infinite)
                inventions.iter().filter_map(|invention| {
                    let fcost = f_nolambda.get(invention).cloned()
                        .or(f_inventionless);
                    let xcost = x_any.get(invention).cloned()
                        .or(x_inventionless);
                    if let (Some(fcost), Some(xcost)) = (fcost, xcost) {
                        let cost = 1+fcost+xcost;
                        debug_assert!(cost <= inventionless_cost_any.unwrap_or(i32::MAX), "cost {} is greater than inventionless cost {} for {} and inv {}", cost, inventionless_cost_any.unwrap_or(i32::MAX), extract_enode(enode, egraph), extract(*invention, egraph));
                        Some((*invention, 1+fcost+xcost))
                    } else {
                        None
                    }
                }).collect()
                // if egraph[*f].data.is_invention {
                //     // child is an invention so we can add it
                //     inventionful.insert(egraph[*b].id, 100);
                // }
                // inventionful
            }
            Lambda::Lam([b]) => {
                // just map +1 over the costs
                // let mut inventionful: HashMap<Id,i32> = egraph[*b].data.inventionful_cost_any.clone().iter().map(|(k,v)| (*k,*v+1)).collect();
                // if egraph[*b].data.is_invention {
                //     // child is an invention so we can add it
                //     inventionful.insert(egraph[*b].id, 100);
                // }
                // inventionful
                let ref b_any = egraph[*b].data.inventionful_cost_any;
                debug_assert!(safe_to_remove_noncanonical(b_any, egraph));

                b_any.iter().clone()
                    .filter(|(k,_)|canonical(k, egraph))
                    .map(|(k,v)| (*k,*v+1)).collect()
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

            // if we just merged two is_invention eclasses to make this one
            if egraph[id].data.inventionful_cost_any.values().any(|&v| v == 0) {
                let mut to_remove = Vec::new();
                for (k,v) in egraph[id].data.inventionful_cost_any.iter() {
                    if *v == 100 {
                        debug_assert_eq!(id,egraph.find(*k));
                        to_remove.push(*k);
                    }
                }
                for k in to_remove {
                    egraph[id].data.inventionful_cost_any.remove(&k);
                }
            }
            egraph[id].data.inventionful_cost_any.insert(id, 100);
            egraph[id].data.inventionful_cost_nolambda.insert(id, 100);
    }

    // let to_remove = noncanonical_inventions(&egraph[id].data.inventionful_cost_any,egraph);
    // for k in to_remove {
    //     assert_eq!(egraph[id].data.inventionful_cost_any[&k], egraph[id].data.inventionful_cost_any[&egraph.find(k)]);
    //     // println!("removing {} from {}", k, id);
    //     // egraph[id].data.inventionful_cost_any.remove(&k);
    // }
    

        // important: modify() must be idempotent



        // if let Some(c) = egraph[id].data.constant.clone() {
        //     if egraph.are_explanations_enabled() {
        //         egraph.union_instantiations(
        //             &c.0.to_string().parse().unwrap(),
        //             &c.1,
        //             &Default::default(),
        //             "analysis".to_string(),
        //         );
        //     } else {
        //         let const_id = egraph.add(c.0);
        //         egraph.union(id, const_id);
        //     }
        // }
    }
}

fn var(s: &str) -> Var {
    s.parse().unwrap()
}

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
        // println!("{:?} {:?}", a1, a2);
        a1[0] != a2[0]
    }

    fn vars(&self) -> Vec<Var> {
        let mut vars = self.0.vars();
        vars.extend(self.1.vars());
        vars
    }
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
            let e = subst[self.to_shift];
            // println!("Shifter on {}", extract(eclass, egraph));
            let e_new = class_shift(e, self.incr_by, 0, egraph, &mut HashMap::new());
            if e_new.is_none() { return vec![]; }
            let mut subst = subst.clone(); // they do this in the example
            subst.insert(self.to_shift, e_new.unwrap()); // overwrites the e with shifted_e
            let res = self.rhs.apply_one(egraph, eclass, &subst);
            // assert!(res.len() == 1);
            // // println!("Shifter {} == {}", extract(res[0], egraph), extract(eclass, egraph));
            // egraph.union(eclass,res[0]);
            res
            
            // warning: there are unions that happen during class_shift internally which arent reported
            // to apply_matches. That seems totally okay though from reading the source code (which only
            // uses the Ids you return from apply_one to figure out how many places were modified)
    }
}

fn class_shift(
    eclass:Id,
    incr_by:i32,
    shift_refs_geq: i32,
    egraph: &mut EGraph,
    seen : &mut HashMap<(Id,i32),Option<Id>>,
    ) -> Option<Id>
    {
        // println!("class_shift[>={}] {}", shift_refs_geq, extract(eclass, egraph));
        // println!("seen: {:?}", seen);
        let key = (eclass,shift_refs_geq); // for caching
        // check if we've seen this before (ie we're looping). If so return our shifted value for it.
        if seen.contains_key(&key) {
            // println!("was seen!: {:?}", seen[&key]);
            return seen[&key];
        }
        if egraph[eclass].data.upward_refs.iter().all(|i| *i < shift_refs_geq) {
            // no refs inside need modification, so the shifted eclass == original eclass
            seen.insert(key, Some(eclass));
            // println!("2 ez");
            return Some(eclass)
        }
        // println!("not simple");
        // we temporarily insert None to break any loops (ie if a recursive call asks us to compute the same thing). Note that at the end of this function we insert the real result in the cache
        seen.insert(key, None);
        // ALL children need modification (since ofc they all have the same free vars)
        // we need to fully clone all the ENodes so we can let go of the borrow
        // of `egraph` (which is happening bc of the iter()) so we can use `egraph` in the body of the loop
        let enodes: Vec<Lambda> = egraph[eclass].iter().cloned().collect();
        let eclasses_to_union : Vec<Id> = enodes.iter().cloned().filter_map(|enode| {
            // println!("[change if >= {}] entering: {}", shift_refs_geq, extract_enode(enode.clone(), egraph));
            match enode {
                Lambda::Var(i) => {
                    // since we didnt return early, this must be a variable that needs shifting
                    assert!(i >= shift_refs_geq);
                    if i + incr_by >= ARGC { seen.insert(key, None); return None }; // $3+ get pruned
                    Some(egraph.add(Lambda::Var(i + incr_by)))
                }
                Lambda::Prim(_) => {
                    panic!("attempted to shift Prim, which shouldnt be attempted since Prim never has free vars")
                }
                Lambda::App([f, x]) => {
                    // recurse in each (class shift will return early if no shifting is needed) and build a new App
                    let fnew_opt = class_shift(f, incr_by, shift_refs_geq, egraph, seen);
                    let xnew_opt = class_shift(x, incr_by, shift_refs_geq, egraph, seen);
                    match (fnew_opt,xnew_opt) {
                        (Some(fnew),Some(xnew)) => Some(egraph.add(Lambda::App([fnew, xnew]))),
                        _ => None,
                    }
                }
                Lambda::Lam([b]) => {
                    // increment shift_refs_geq since refs must point even FURTHER to point out of the shifted region now
                    // println!("entering lam with {:?}", seen);
                    let res = class_shift(b, incr_by, shift_refs_geq + 1, egraph, seen)
                        .map(|bnew| egraph.add(Lambda::Lam([bnew])));
                    // println!("exited lam with {:?}", seen);
                    res
                }
                Lambda::Programs(_) => {
                    panic!("attempted to shift a Programs node")
                }
            }
        }).collect();
        // println!("original eclass: {}", extract(eclass,egraph));
        // print("{:?}",enodes)
        // todo figure out why this fires
        // assert!(!eclasses_to_union.contains(&eclass)); // implies shifting wasnt needed, so why didnt we return early
        // union the eclasses
        if eclasses_to_union.is_empty() {
            seen.insert(key, None);
            return None
        }
        let new_eclass = egraph.find(eclasses_to_union[0]); // dont need to canonicalize like this, but will prob speed up the unionfind later
        eclasses_to_union.iter().skip(1).for_each(|id| {egraph.union(*id, new_eclass);});
        seen.insert(key, Some(new_eclass));
        Some(new_eclass)
}



struct Inliner {
    replace_with: Var, // what to inline
    inline_into: Var, // what to inline it into
    rhs: Pattern<Lambda>, // expr to be unified with original LHS - but with inline_into modified!
    // abort_if_equal: Pattern<Lambda>,
}

impl Applier<Lambda, LambdaAnalysis> for Inliner {
    fn apply_one(
        &self,
        egraph: &mut EGraph,
        eclass: Id,
        subst: &Subst) -> Vec<Id> 
        {
            let e = subst[self.inline_into];
            let e_new = inline(e, subst[self.replace_with], 0, egraph, &mut HashMap::new());
            if e_new.is_none() { return vec![]; }
            let mut subst = subst.clone(); // they do this in the example
            subst.insert(self.inline_into, e_new.unwrap()); // overwrites the e with shifted_e
            let res = self.rhs.apply_one(egraph, eclass, &subst);
            // assert!(res.len() == 1);
            // // println!("Inliner {} == {}", extract(res[0], egraph), extract(eclass, egraph));
            // egraph.union(eclass,res[0]);
            res
            // warning: there are unions that happen during inline internally which arent reported
            // to apply_matches. That seems totally okay though from reading the source code (which only
            // uses the Ids you return from apply_one to figure out how many places were modified)
    }
}


// inline() is shockingly annoying to do bc you gotta
// a) downshift indices that point above whatever ur inlining
// b) replace indices that point to what ur inlining w the new contents
// c) actually modify that new contents for the specific location ur putting
//    it by upshifting all the indices that point outside of it
fn inline(
    eclass:Id,
    replace_with:Id,
    arg_idx: i32, // starts at 0
    egraph: &mut EGraph,
    seen : &mut HashMap<(Id,i32),Option<Id>>,
    ) -> Option<Id>
    {
        let key = (eclass,arg_idx);
        // check if we've seen this before (ie we're looping). If so return whatever we got last time we calculated it.
        if seen.contains_key(&key) {
            return seen[&key];
        }
        if egraph[eclass].data.upward_refs.iter().all(|i| *i < arg_idx) {
            // theres no ref to us or any parent of us in here so we dont need to modify anything
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
                    if i == arg_idx {
                        // we need to replace this with whatever we're inlining
                        // and sadly we actually need to add +arg_idx to all outgoing indices
                        // in replace_with bc we've moved it from its home down deeper.
                        // dont worry the new `seen` is a) needed and b) cant form a loop since
                        // inline isnt mutually recursive with class_shift, its just one direction of inline calling class shift.
                        class_shift(replace_with, arg_idx, 0, egraph, &mut HashMap::new())
                    } else if i > arg_idx {
                        // we need to decrement this by 1 since its a pointer above the lambda we removed
                        Some(egraph.add(Lambda::Var(i - 1)))
                    } else {
                        panic!("should have returned earlier")
                    }
                }
                Lambda::Prim(_) => {
                    panic!("attempted to shift Prim, which shouldnt be attempted since Prim never has free vars")
                }
                Lambda::App([f, x]) => {
                    // recurse in each (class shift will return early if no shifting is needed) and build a new App
                    let fnew_opt = inline(f, replace_with, arg_idx, egraph, seen);
                    let xnew_opt = inline(x, replace_with, arg_idx, egraph, seen);
                    match (fnew_opt,xnew_opt) {
                        (Some(fnew),Some(xnew)) => Some(egraph.add(Lambda::App([fnew, xnew]))),
                        _ => None,
                    }
                }
                Lambda::Lam([b]) => {
                    // increment arg_idx since refs must point even FURTHER to point out of the shifted region now
                    inline(b, replace_with, arg_idx + 1, egraph, seen)
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
            seen.insert(key, None);
            return None
        }
        let new_eclass = egraph.find(eclasses_to_union[0]); // dont need to canonicalize like this, but will prob speed up the unionfind later
        eclasses_to_union.iter().skip(1).for_each(|id| {egraph.union(*id, new_eclass);});
        seen.insert(key, Some(new_eclass));
        Some(new_eclass)
}

struct BeamSearch {
    beam_size: usize,
    inventions: HashSet<Id>,
    seen: HashMap<Id,Option<Beam>>,
}

#[derive(Clone,Debug)]
struct Beam {
    cost_nolambda_inventionless: (f64,usize),
    cost_any_inventionless: (f64,usize),
    cost_nolambda_under_invention: HashMap<Id,(f64,usize)>,
    cost_any_under_invention: HashMap<Id,(f64,usize)>,
}

fn update_if_better(old: &mut (f64,usize), new: (f64,usize)) {
    if new.0 < old.0 {
        *old = new;
    }
}

fn update_if_better_inv(cost_under_inv: &mut HashMap<Id,(f64,usize)>, inv: Id, new: (f64,usize)) {
    if !cost_under_inv.contains_key(&inv) || new.0 < cost_under_inv[&inv].0 {
        cost_under_inv.insert(inv, new);
    }
}

const EPSILON_COST: f64 = 0.01;

// #[derive(Debug)]
// struct CostPair {any:f64, nolambda:f64}
// struct ProgramCost;
// impl CostFunction<Lambda> for ProgramCost {
//     type Cost = CostPair;
//     fn cost<C>(&mut self, enode: &Lambda, mut costs: C) -> Self::Cost
//     where
//         C: FnMut(Id) -> Self::Cost
//     {
//         match enode {
//             Lambda::Var(_) | Lambda::Prim(_) => CostPair { any: 1., nolambda: 1.},
//             Lambda::App([f, x]) => {
//                 let cost = EPSILON_COST + costs(*f).nolambda + costs(*x).any;
//                 CostPair { any: cost, nolambda: cost}
//             }
//             Lambda::Lam([b]) => {
//                 CostPair { any: EPSILON_COST + costs(*b).any, nolambda: f64::INFINITY }
//             }
//             Lambda::Programs(ps) => {
//                 let cost = ps.iter().map(|p| costs(*p).any).sum();
//                 CostPair { any: cost, nolambda: cost}
//             }
//         }
//     }
// }

// struct CostPair {
//     any: Option<i32>,
//     nolambda: Option<i32>,
// }
// struct CleverCost<'a> { inv:Option<Id>, egraph: &'a EGraph }
// impl<'a> CostFunction<Lambda> for CleverCost<'a> {
//     type Cost = CostPair;
//     fn cost<C>(&mut self, enode: &Lambda, mut costs: C) -> Self::Cost
//     where
//         C: FnMut(Id) -> Self::Cost
//     {
//         let cost_any: Option<i32> = match enode {
//             Lambda::Var(_) | Lambda::Prim(_) => Some(100),
//             Lambda::App([f, x]) => {
//                 let fcost = costs(*f).nolambda;
//                 let xcost = costs(*x).any;
//                 match (fcost,xcost) {
//                     (Some(fcost),Some(xcost)) => Some(1 + fcost + xcost),
//                     _ => None,
//                 }
//             }
//             Lambda::Lam([b]) => {
//                 if self.inv == Some(self.egraph.find(*b)) {
//                     Some(1 + 100)
//                 } else {
//                     costs(*b).any.map(|c| 1 + c)
//                 }
//             }
//             Lambda::Programs(roots) => {
//                 let costs: Option<Vec<i32>> = roots.iter().map(|p| costs(*p).any).collect();
//                 costs.map(|cs| cs.iter().sum())
//             }
//         };
//         let cost_nolambda = match enode {
//             Lambda::Lam(_) => None,
//             _ => cost_any.clone()
//         };
//         CostPair { any: cost_any, nolambda: cost_nolambda }
//         // match enode {
//         //     Lambda::Var(_) | Lambda::Prim(_) => 100,
//         //     Lambda::App([f, x]) => 1 + costs(*f) + costs(*x),
//         //     Lambda::Lam([b]) => 1 + costs(*b),
//         //     Lambda::Programs(ps) => ps.iter().map(|p| costs(*p)).sum(),
//         // }
//     }
// }

// this is the cost where applams are allowed
struct NaiveCost;
impl CostFunction<Lambda> for NaiveCost {
    type Cost = i32;
    fn cost<C>(&mut self, enode: &Lambda, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost
    {
        match enode {
            Lambda::Var(_) | Lambda::Prim(_) => 100,
            Lambda::App([f, x]) => 1 + costs(*f) + costs(*x),
            Lambda::Lam([b]) => 1 + costs(*b),
            Lambda::Programs(ps) => ps.iter().map(|p| costs(*p)).sum(),
        }
    }
}

impl BeamSearch {

    fn best_inventions(
        &self,
        eclass: Id,
        nolambda: bool,
        egraph: &EGraph,
    ) -> (f64, Vec<(f64,Id)>) {
        let eclass = egraph.find(eclass);
        // gives a sorted list of the best inventions (and costs under them) along with the inventionless cost
        let beam: &Beam = self.seen[&eclass].as_ref().unwrap();
        let inventionless_cost: f64 = if nolambda {beam.cost_nolambda_inventionless.0} else {beam.cost_any_inventionless.0};
        let hashmap = if nolambda {&beam.cost_nolambda_under_invention} else {&beam.cost_any_under_invention};
        let mut invention_costs: Vec<_> = hashmap.iter().map(|(inv,(cost,_))| (*cost,*inv)).collect();
        invention_costs.sort_by(|(cost1,_),(cost2,_)| cost1.partial_cmp(cost2).unwrap());
        (inventionless_cost, invention_costs)
    }

    fn extract_cheapest(
        &mut self,
        root: Id,
        inv: Option<Id>, // invention to use when extracting. None means no invention.
        egraph: &EGraph,
    ) -> RecExpr<Lambda> {
        let root = egraph.find(root);
        // runs beam search in case we haven't (cached if we already have)
        self.eclass_cost(root, egraph);

        let mut expr = RecExpr::default();
        self.extract_cheapest_rec(root, inv, false, egraph, &mut expr);

        // ProgramCost.rec_cost(&expr) - 

        // let beam: &Beam = self.seen[&root].as_ref().unwrap();
        // beam.cost_any_under_invention.get(invention)
        //                         .map(|(cst,_)|*cst)
        //                         .unwrap_or(xcost_inventionless);

        expr
    }

    fn extract_cheapest_rec(
        &mut self,
        root: Id,
        inv: Option<Id>, // invention to use when extracting. None means no invention.
        nolambda: bool, // whether or not a toplevel lambda is okay
        egraph: &EGraph,
        into_expr: &mut RecExpr<Lambda> // expr we're extracting into
    ) -> Id { // returns the recexpr id of this node in into_expr
        let root = egraph.find(root);

        if let Some(inv) = inv {
            if inv == root {
                // this node is the we're extracting with
                return into_expr.add(Lambda::Prim(format!("inv{}",inv).into()));
            }
        }

        let beam: &Beam = self.seen[&root].as_ref().unwrap();

        let inventionless_cost: (f64,usize) = if nolambda {beam.cost_nolambda_inventionless} else {beam.cost_any_inventionless};

        let (cost,node_id) = match inv {
            Some(inv) => {
                let hashmap = if nolambda {&beam.cost_nolambda_under_invention} else {&beam.cost_any_under_invention};
                hashmap.get(&inv).unwrap_or(&inventionless_cost).clone()
            }
            None => {inventionless_cost.clone()}
        };
        if node_id == usize::MAX {
            println!("{:?} {} {}", inv, extract(root, &egraph), cost);
            println!("{:?}",beam);
        }
        match &egraph[root].nodes[node_id] {
            Lambda::Prim(prim) => {
                into_expr.add(Lambda::Prim(*prim))
            }
            Lambda::Var(var) => {
                into_expr.add(Lambda::Var(*var))
            }
            Lambda::App([f,x]) => {
                let f_id = self.extract_cheapest_rec(*f, inv, true, egraph, into_expr);
                let x_id = self.extract_cheapest_rec(*x, inv, false, egraph, into_expr);
                into_expr.add(Lambda::App([f_id,x_id]))
            }
            Lambda::Lam([b]) => {
                let b_id = self.extract_cheapest_rec(*b, inv, false, egraph, into_expr);
                into_expr.add(Lambda::Lam([b_id]))
            }
            Lambda::Programs(programs) => {
                let p_ids: Vec<Id> = programs.iter().map(|p| self.extract_cheapest_rec(*p, inv, false, egraph, into_expr)).collect();
                into_expr.add(Lambda::Programs(p_ids))
            }
        }
    }


    fn eclass_cost(&mut self, eclass: Id, egraph: &EGraph) {
        let eclass = egraph.find(eclass);
        // compute the cost of an eclass and add a Beam to self.seen for it. Recursively calculate children as needed.
        if self.seen.contains_key(&eclass) {
            return
        }
        // sentinel so we know if we're trying to self loop
        // todo question: what if its a self loop from nolambda to any or vis versa tho is that ever a thing?
        self.seen.insert(eclass, None);

        let mut beam = Beam {
            cost_nolambda_inventionless: (f64::INFINITY,usize::MAX),
            cost_any_inventionless: (f64::INFINITY,usize::MAX),
            cost_nolambda_under_invention: HashMap::new(),
            cost_any_under_invention: HashMap::new(),
        };

        if self.inventions.contains(&eclass) {
            //todo using cost=1 and child=usize::MAX to indicate invention...
            update_if_better_inv(&mut beam.cost_any_under_invention, eclass, (1.,usize::MAX));
            update_if_better_inv(&mut beam.cost_nolambda_under_invention, eclass, (1.,usize::MAX));
        }

        let enodes: Vec<Lambda> = egraph[eclass].iter().cloned().collect();
        for (node_id,enode) in enodes.iter().enumerate() {
            // recurse on children
            enode.for_each(|eclass| self.eclass_cost(eclass, egraph));

            // get inventionless and inventionful costs
            let (cost_inventionless,cost_inventions) = self.enode_cost(enode,&egraph);
            // if !(cost_inventionless < f64::INFINITY) {
            //     println!("enode cost invless is inf: {}",extract_enode(enode, &egraph));
            //     // println!("len:{:?} cost[0]:{:?}",enodes.len(), self.enode_cost(&enodes[0],&egraph).0);
            //     panic!("enode cost invless is infinite");
            // }
            // cost may very well be infinite thats ok

            // update the beam if these costs are better than what we have
            if let Lambda::Lam(_) = enode {
                // lambda case: since we're in a lambda, only update the `any` costs and not the `nolambda` costs
                update_if_better(&mut beam.cost_any_inventionless, (cost_inventionless,node_id));
                for (inv,cost) in cost_inventions {
                    update_if_better_inv(&mut beam.cost_any_under_invention, inv, (cost,node_id));
                }
            } else {
                // not a lambda case: update the `any` and `nolambda` costs
                update_if_better(&mut beam.cost_any_inventionless, (cost_inventionless,node_id));
                update_if_better(&mut beam.cost_nolambda_inventionless, (cost_inventionless,node_id));
                for (inv,cost) in cost_inventions {
                    update_if_better_inv(&mut beam.cost_any_under_invention, inv, (cost,node_id));
                    update_if_better_inv(&mut beam.cost_nolambda_under_invention, inv, (cost,node_id));
                }
            }
        };

        // if !(beam.cost_any_inventionless.0 < f64::INFINITY) {
        //     println!("{}", extract(eclass, &egraph));
        //     panic!("aaa");
        // }

        // no point keeping around inventionful values that are worse than inventionless ones
        let best_cost_inventionless = beam.cost_any_inventionless.0.clone();
        beam.cost_any_under_invention.retain(|_,(cost,_)| *cost < best_cost_inventionless);
        // assert!(best_cost_inventionless < f64::INFINITY);
        // if !(best_cost_inventionless < f64::INFINITY) {
        //     println!("best cost inventionaless is inf: {}",extract(eclass, &egraph));
        //     println!("len:{:?} cost[0]:{:?}",enodes.len(), self.enode_cost(&enodes[0],&egraph).0);
        //     panic!("best cost inventionless is infinite");
        // }
        // todo not certain if its ok to not have the above assert. I think maybe its ok and certain terms that get marked as infinite have necessary self loops or something. idk multiarg is the only think that intros them it seems.

        // same for no_lambda
        let best_cost_inventionless = beam.cost_nolambda_inventionless.0.clone();
        beam.cost_nolambda_under_invention.retain(|_,(cost,_)| *cost < best_cost_inventionless);
        // no assertion about infinity for no_lambda since sometimes that will be impossible to construct

        // now lets narrow the beams to the beam_size
        let mut invention_costs: Vec<_> = beam.cost_any_under_invention.into_iter().collect();
        invention_costs.sort_by(|(_,(cost1,_)),(_,(cost2,_))| cost1.partial_cmp(cost2).unwrap());
        invention_costs.truncate(self.beam_size);
        beam.cost_any_under_invention = invention_costs.into_iter().collect();

        //narrow the nolambda beam
        let mut invention_costs: Vec<_> = beam.cost_nolambda_under_invention.into_iter().collect();
        invention_costs.sort_by(|(_,(cost1,_)),(_,(cost2,_))| cost1.partial_cmp(cost2).unwrap());
        invention_costs.truncate(self.beam_size);
        beam.cost_nolambda_under_invention = invention_costs.into_iter().collect();

        self.seen.insert(eclass, Some(beam)); 
    }

    fn enode_cost(&self, enode: &Lambda, egraph: &EGraph) -> (f64,Vec<(Id,f64)>) {
        // calculate the cost of an enode and return the
        // inventionless cost as well as the cost under any inventions that
        // improve it

        match enode {
            Lambda::Var(_) | Lambda::Prim(_) => {
                (1.,vec![])
            }
            Lambda::App([f, x]) => {
                let f = &egraph.find(*f);
                let x = &egraph.find(*x);
                match (&self.seen[f],&self.seen[x]) {
                    (Some(fbeam),Some(xbeam)) => {
                        // inventionless costs
                        let fcost_inventionless = fbeam.cost_nolambda_inventionless.0;
                        let xcost_inventionless = xbeam.cost_any_inventionless.0;
                        let cost_inventionless = fcost_inventionless + xcost_inventionless + EPSILON_COST;
                        
                        // figure out which inventions helped `f` and `x` and union them
                        let mut inventions: HashSet<Id> = fbeam.cost_nolambda_under_invention.keys().cloned().collect();
                        inventions.extend(xbeam.cost_any_under_invention.keys().cloned());
                        
                        // costs with inventions
                        let cost_inventions: Vec<(Id,f64)> = inventions.iter().map(|invention| {
                            let fcost_invention = fbeam.cost_nolambda_under_invention.get(invention)
                                .map(|(cst,_)|*cst) // extract out cost from cost,node_id tuple
                                .unwrap_or(fcost_inventionless); // default to inventionless
                            let xcost_invention = xbeam.cost_any_under_invention.get(invention)
                                .map(|(cst,_)|*cst)
                                .unwrap_or(xcost_inventionless);
                            (*invention, fcost_invention + xcost_invention + EPSILON_COST)
                        }).collect();
                        (cost_inventionless,cost_inventions)
                    }
                    _ => {
                        // one of the pointers caused a loop so not worth pursuing
                        (f64::INFINITY,vec![])
                    } 
                }
            }
            Lambda::Lam([b]) => {
                let b = &egraph.find(*b);
                match &self.seen[b] {
                    Some(bbeam) => {
                        // inventionless cost
                        let bcost_inventionless = bbeam.cost_any_inventionless.0;
                        let cost_inventionless = bcost_inventionless + EPSILON_COST;

                        // cost with inventions
                        let cost_inventions: Vec<(Id,f64)> = bbeam.cost_any_under_invention.iter()
                            .map(|(invention,(cst,_))| {
                                (*invention, *cst + EPSILON_COST)
                            }).collect();
                        (cost_inventionless,cost_inventions)
                    }
                    None => {
                        // println!("hit a self loop on {}", extract(*b, &egraph));
                        // looped back to something we've seen so not worth pursuing
                        (f64::INFINITY,vec![])
                    }
                }
            }
            Lambda::Programs(ps) => {
                let ps: Vec<Id> = ps.iter().map(|p| egraph.find(*p)).collect();
                if ps.iter().any(|p| self.seen[p].is_none()) {
                    (f64::INFINITY,vec![]) // self loop
                } else {
                    let costs_inventionless: Vec<f64> = ps.iter().map(|p| self.seen[p].as_ref().unwrap().cost_any_inventionless.0).collect();
                    let cost_inventionless = costs_inventionless.iter().sum();

                    // union together all the useful inventions of diff programs
                    let mut inventions: HashSet<Id> = HashSet::new();
                    for p in ps.iter().map(|p| self.seen[p].as_ref().unwrap()){
                        inventions.extend(p.cost_any_under_invention.keys().cloned())
                    }

                    // cost with inventions
                    let cost_inventions: Vec<(Id,f64)> = inventions.iter().map(|invention|{
                        let cost = ps.iter().enumerate().map(|(i,p)| self.seen[p].as_ref().unwrap()
                            .cost_any_under_invention.get(invention)
                            .map(|(cst,_)|*cst)
                            .unwrap_or(costs_inventionless[i])
                        ).sum();
                        (*invention,cost)
                    }).collect();
                    (cost_inventionless,cost_inventions)
                }
            }
        }
    }
}



fn main() {
    env_logger::init();

    let intro_rules: &[Rewrite<Lambda, LambdaAnalysis>] = &[
        // applam-intro: this rule matches any node and rewrites it to be an applam with
        // $0 in the body and the subtree in the arg. Applies to all nodes
        // not just leaves. This rule necessarily introduces a self loop.
        rw!("applam-intro"; "(?subtree)" => "(app (lam 0) ?subtree)"
        // abstracting the identity out just leads to insane blowups everywhere...
        if ConditionNotEqual::parse("(?subtree)", "(lam 0)")
        ),
    ];
    let propagate_rules: &[Rewrite<Lambda, LambdaAnalysis>] = &[
        // applam-bubble-from-left and applam-bubble-from-right:
        // these are the rules for bubbling an applam up out of the left and right sides
        // of an app respectively In the left case, the `arg` of the above app will be dropped
        // below the lambda meaning any pointers in it that point above its own root need
        // incrementing. In the right case its the same but with the `body` of the above app.
        rw!("applam-bubble-from-left"; "(app (app (lam ?body) ?arginner) ?argouter)"
            => {Shifter {
                incr_by: 1, // how much to increment by eg +1 or -1
                to_shift: var("?argouter"), // expression to shift
                rhs: "(app (lam (app ?body ?argouter)) ?arginner)".parse().unwrap(), // expr to be unified with original LHS - but with to_shift modified!
            }}
            // condition accounts for avoiding blowup from bubbling out of self-loop. If the two apps are the same eclass already it doesnt bubble the lower one up. Not sure if this will limit anything, its just my quick fix.
            if ConditionNotEqual::parse("(app (app (lam ?body) ?arginner) ?argouter)", "(app (lam ?body) ?arginner)")
            // dont do this if itll create a $3 or more
            // if large_upward_refs(var("?f"))
        ),
        rw!("applam-bubble-from-right"; "(app ?f (app (lam ?body) ?arg))"
            => {Shifter {
                incr_by: 1, // how much to increment by eg +1 or -1
                to_shift: var("?f"), // expression to shift
                rhs: "(app (lam (app ?f ?body)) ?arg)".parse().unwrap(), // expr to be unified with original LHS - but with to_shift modified!
            }}
            // condition accounts for avoiding blowup from bubbling out of self-loop. If the two apps are the same eclass already it doesnt bubble the lower one up. Not sure if this will limit anything, its just my quick fix.
            if ConditionNotEqual::parse("(app ?f (app (lam ?body) ?arg))", "(app (lam ?body) ?arg)")
            // dont do this if itll create a $3 or more
            // if large_upward_refs(var("?f"))
        ),

        // applam-merge: this rule says when you have an app of two applams
        // that have a shared arg, you can bubble the lambda up above the inner
        // applications and merge them (body of left gets applied to body of right).
        rw!("applam-merge"; "(app (app (lam ?body1) ?argshared) (app (lam ?body2) ?argshared))"
            => "(app (lam (app ?body1 ?body2)) ?argshared)"
            // todo this is overly strict but its hard to do better. detecting self application
            // doesnt help because technically when looking in isolation at (app 0 0) w no
            // lambdas you cant actually prove that it equals 0.
        // todo this is a p important thing to fix bc for example you cant abstract across same branches like (app x x) or (app (-y) (-y)) etc
        // the "if is_not_same_var(var("?body1"), var("?body2"))" is all i settled on btw
            // if is_not_same_var(var("?body1"), var("?body2"))
            // if not_all_same(var("?body1"), var("?body2"), "(app ?body1 ?body2)".parse().unwrap())
            // if ConditionNotEqual::parse("(?body1)", "(app ?body1 ?body2)")
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
        rw!("applam-inline"; "(app (lam ?body) ?arg)"
            => {Inliner {
                replace_with: var("?arg"), // what to inline
                inline_into: var("?body"), // what to inline it into
                rhs: "(?body)".parse().unwrap(), // expr to be unified with original LHS - but with inline_into modified!
                // abort_if_equal: "(lam 0)".parse().unwrap(),
            }}
            // we dont inline the identity function ??? idk this is just me trying ot fix things
            // if ConditionNotEqual::parse("?arg", "(lam 0)")
        ),
    ];

    let final_rules: &[Rewrite<Lambda, LambdaAnalysis>] = &[
        //todo multiarg is an insane slowdown. Ofc you could investigate why this is, but also i think its just fine to run it at the END bc if we do all possible refactorings then everything will already be in a perfect position for multiarg abstracting?
        // applam-multiarg: this is just the transformation that takes an applam applam setup and moves it so
        // its appapplamlam. Theres a tradeoff here bc this feels superficial and I worry about adding rewrite rules,
        // but a) this feels relatively safe and b) structural hashing wise you actually REALLY want the two lambdas on
        // top of each other.
        // btw we do need to downshift the outgoing refs of arginner since it gets hoisted up,
        // and furthermore we need to make sure its not pointing to argouter (by making sure $0 isnt an
        // upward ref)
        rw!("applam-multiarg"; "(app (lam (app (lam ?body) ?arginner)) ?argouter)"
            => {Shifter {
                incr_by: -1, // how much to increment by eg +1 or -1
                to_shift: var("?arginner"), // expression to shift
                rhs: "(app (app (lam (lam (?body))) ?argouter) ?arginner)".parse().unwrap(), // expr to be unified with original LHS - but with to_shift modified!
            }}
            // condition: cant raise arginner above argouter if arginner points to argouter
            if zero_not_in_upward_refs(var("?arginner"))
        ),
    ];

    let mut egraph: EGraph = Default::default();
    // egraph.add_expr(&"(x)".parse().unwrap());
    // let root = egraph.add_expr(&"(programs (app - y))".parse().unwrap());
    // let root = egraph.add_expr(&"(programs  (app - y) (app (app - x) x))".parse().unwrap());
    // egraph.add_expr(&"(programs (app (app x x) (app y y)))".parse().unwrap());
    // egraph.add_expr(&"(programs (app (app (app + x) z) (app (app + x) y)))".parse().unwrap());
    // egraph.add_expr(&"(programs (app (app (app + x) z) (app (app + x) y)) (app (app (app + x) z) (app (app + x) y)))".parse().unwrap());

    // first dreamcoder program
    let programs: Vec<RecExpr<Lambda>> = vec![
        // "(app - y)",

        "(lam (app (app (app logo_forLoop t3) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t3)) 0)))) 0))",
        "(lam (app (app (app logo_forLoop t3) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t3)) 0)))) 0))",

        "(lam (app (app (app logo_forLoop t8) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t8)) 0)))) 0))",
        "(lam (app (app (app logo_forLoop t8) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t8)) 0)))) 0))",
        "(lam (app (app (app logo_forLoop t9) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t9)) 0)))) 0))",
        "(lam (app (app (app logo_forLoop t9) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t9)) 0)))) 0))",
        "(lam (app (app (app logo_forLoop logo_IFTY) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_epsL) t1)) logo_epsA) 0)))) 0))",
        "(lam (app (app (app logo_forLoop logo_IFTY) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_epsL) t2)) logo_epsA) 0)))) 0))",
        "(lam (app (app (app logo_forLoop logo_IFTY) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_epsL) t5)) logo_epsA) 0)))) 0))",
        "(lam (app (app (app logo_forLoop t4) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t4)) 0)))) 0))",
        "(lam (app (app (app logo_FWRT logo_UL) logo_ZA) 0))",
        "(lam (app (app (app logo_FWRT logo_ZL) (app (app logo_DIVA logo_UA) t4)) (app (app (app logo_FWRT logo_UL) logo_ZA) 0)))",
        "(lam (app (app (app logo_forLoop t4) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t4)) 0)))) 0))",
        "(lam (app (app (app logo_forLoop t5) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t5)) 0)))) 0))",
        "(lam (app (app (app logo_forLoop t5) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t5)) 0)))) 0))",
        "(lam (app (app (app logo_forLoop t6) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t6)) 0)))) 0))",
        "(lam (app (app (app logo_forLoop t9) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) 1)) (app (app logo_DIVA logo_UA) t4)) 0)))) 0))",
        "(lam (app (app (app logo_forLoop t6) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t6)) 0)))) 0))",
        "(lam (app (app (app logo_forLoop t7) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t7)) 0)))) 0))",
            ].iter().map(|p| p.parse().unwrap()).collect();
    let roots: Vec<Id> = programs.iter().map(|p| egraph.add_expr(&p)).collect();
    // second dreamcoder program
    // egraph.add_expr(&"(lam (app (app (app logo_forLoop t3) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t3)) 0)))) 0))".parse().unwrap());

    
    // 19 dreamcoder programs w heavy overlap
    // egraph.add_expr(&"(programs (lam (app (app (app logo_forLoop t3) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t3)) 0)))) 0)) (lam (app (app (app logo_forLoop t8) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t8)) 0)))) 0)) (lam (app (app (app logo_forLoop t8) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t8)) 0)))) 0)) (lam (app (app (app logo_forLoop t9) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t9)) 0)))) 0)) (lam (app (app (app logo_forLoop t9) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t9)) 0)))) 0)) (lam (app (app (app logo_forLoop logo_IFTY) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_epsL) t1)) logo_epsA) 0)))) 0)) (lam (app (app (app logo_forLoop logo_IFTY) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_epsL) t2)) logo_epsA) 0)))) 0)) (lam (app (app (app logo_forLoop logo_IFTY) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_epsL) t5)) logo_epsA) 0)))) 0)) (lam (app (app (app logo_forLoop t4) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t4)) 0)))) 0)) (lam (app (app (app logo_FWRT logo_UL) logo_ZA) 0)) (lam (app (app (app logo_FWRT logo_ZL) (app (app logo_DIVA logo_UA) t4)) (app (app (app logo_FWRT logo_UL) logo_ZA) 0))) (lam (app (app (app logo_forLoop t4) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t4)) 0)))) 0)) (lam (app (app (app logo_forLoop t5) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t5)) 0)))) 0)) (lam (app (app (app logo_forLoop t5) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t5)) 0)))) 0)) (lam (app (app (app logo_forLoop t6) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t6)) 0)))) 0)) (lam (app (app (app logo_forLoop t9) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) 1)) (app (app logo_DIVA logo_UA) t4)) 0)))) 0)) (lam (app (app (app logo_forLoop t6) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t6)) 0)))) 0)) (lam (app (app (app logo_forLoop t7) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t7)) 0)))) 0)))".parse().unwrap());



    // egraph.add_expr(&"(programs (app y (lam (app y 0))) (app x (lam (app 0 0))))".parse().unwrap());

    // egraph.add_expr(&"(app x (lam (app 0 0)))".parse().unwrap());
    // egraph.add_expr(&"(app y (lam (app y 0)))".parse().unwrap());

    egraph.dot().to_png("target/0.png").unwrap();


    let fast:bool = true;
    if fast {
        let runner = Runner::default().with_egraph(egraph);
        let runner = Runner::default().with_egraph(runner.egraph).with_iter_limit(1).run(intro_rules);
        let runner = Runner::default().with_egraph(runner.egraph).with_iter_limit(400).with_time_limit(core::time::Duration::from_secs(200)).with_node_limit(3000000).run(propagate_rules);
        runner.print_report();
        let runner = Runner::default().with_egraph(runner.egraph).with_iter_limit(400).with_time_limit(core::time::Duration::from_secs(200)).with_node_limit(3000000).run(final_rules);
        runner.print_report();
        let mut egraph = runner.egraph;
        egraph.rebuild();

        // egraph.dot().to_png("target/zzz.png").unwrap();


        let find_cand: Pattern<Lambda> = "(lam ?b)".parse().unwrap();
        let matches = find_cand.search(&egraph);
        let cands: HashSet<Id> = matches.iter().map(|m| egraph.find(m.eclass))
            .filter(|eclass|egraph[*eclass].data.upward_refs.is_empty()) // cant have free vars in an invention!
            .collect();

        println!("candidates: {:?}", cands.len());
        let mut beam_search = BeamSearch {
            beam_size: 1000000,
            inventions: cands,
            seen: HashMap::new(),
        };
        // println!("beam searchin\n");
        let programs_id = egraph.add(Lambda::Programs(roots.clone()));
        egraph.rebuild();


        // println!("\n***DOING IT THE OLD WAY\n");

        // beam_search.eclass_cost(programs_id, &egraph);

        // // get the top inventions used by at least two programs
        // let top_inventions_all: Vec<Id> = roots.iter().map(|r| beam_search.best_inventions(*r, false, &egraph).1).flatten().map(|(_,inv)| inv).collect();
        // let mut top_inv_uniq: Vec<Id> = top_inventions_all.clone();
        // top_inv_uniq.sort();
        // top_inv_uniq.dedup();
        // let top_inventions: Vec<Id> = top_inv_uniq.into_iter().filter(|i| top_inventions_all.iter().filter(|j|i==*j).count() > 1).collect();

        // let (inventionless_cost, top_individual_inventions) = beam_search.best_inventions(programs_id, false, &egraph);

        // let expr = extract(programs_id,&egraph);
        // println!("Inventionless ({} | {} | {:?}):\n{}\n", inventionless_cost, ProgramCost.cost_rec(&expr), egraph[programs_id].data.inventionless_cost_any, expr);

        // let mut top_inventions: Vec<(i32,Id,RecExpr<Lambda>)> = top_inventions.iter().map(|inv| {
        //     let p = beam_search.extract_cheapest(programs_id, Some(*inv), &egraph);
        //     (ProgramCost.cost_rec(&p), *inv, p)
        // }).collect();
        // top_inventions.sort_by(|a,b| a.0.partial_cmp(&b.0).unwrap());

        // for (i,(cost,inv,rewritten)) in top_inventions.iter().take(5).enumerate() {
        //     let inv_expr = extract(*inv,&egraph);            
        //     println!("\nInvention {} (id={}) ({} | {:?}): {}", i, inv, ProgramCost.cost_rec(&inv_expr), egraph[*inv].data.inventionless_cost_any, inv_expr);
        //     let rwid = egraph.add_expr(&rewritten);
        //     println!("Rewritten({} | {:?}):\n{}", cost, egraph[rwid].data.inventionless_cost_any, rewritten);
        // }

        // println!("\n***DOING IT THE NEW WAY\n");

        println!("Inventionless (cost={:?}):\n{}\n",
            min_cost(programs_id, None, false, &egraph),
            beam_extract(programs_id, None, &egraph)
        );

        let top_invs: Vec<Id> = best_inventions(&egraph[programs_id].data.inventionful_cost_any)
            .into_iter()
            .take(5).collect();
        assert!(top_invs.iter().all(|id| canonical(id,&egraph)));

        for (i,inv) in top_invs.iter().enumerate() {
            println!("\nInvention {} (id={}) (inv_cost={:?}; rewritten_cost={:?}): {}\n Rewritten:\n{}",
                i,
                inv,
                min_cost(*inv, None, false, &egraph),
                min_cost(programs_id, Some(*inv), false, &egraph),
                beam_extract(*inv, None, &egraph),
                beam_extract(programs_id, Some(*inv), &egraph),
            );
        }



        // println!("\n");
        // let top_invs: Vec<Id> = best_inventions(&egraph[programs_id].data.inventionful_cost_any)
        //     .into_iter()
        //     .take(10).collect();
        // assert!(top_invs.iter().all(|id| canonical(id,&egraph)));
        // for (i,inv) in top_invs.iter().enumerate() {
        //     println!("Invention {} (id={} -> {}) (inv_cost={:?}) (rewritten cost = {:?}): ", i, inv, egraph.find(*inv), egraph[*inv].data.inventionless_cost_any, egraph[programs_id].data.inventionful_cost_any[&inv]);
        // }

        // runner.egraph.dot().to_png("target/final.png").unwrap();
    }
    else { 
        egraph.dot().to_png("target/0.png").unwrap();

        // let limit = 1;
        let runner = Runner::default().with_egraph(egraph).with_iter_limit(1).run(intro_rules);
        egraph = runner.egraph;
        egraph.dot().to_png("target/1.png").unwrap();

        let runner = Runner::default().with_egraph(egraph).with_iter_limit(1).run(propagate_rules);
        egraph = runner.egraph;
        egraph.dot().to_png("target/2.png").unwrap();

        let runner = Runner::default().with_egraph(egraph).with_iter_limit(1).run(propagate_rules);
        egraph = runner.egraph;
        egraph.dot().to_png("target/3.png").unwrap();

        let runner = Runner::default().with_egraph(egraph).with_iter_limit(1).run(propagate_rules);
        egraph = runner.egraph;
        egraph.dot().to_png("target/4.png").unwrap();

        let runner = Runner::default().with_egraph(egraph).with_iter_limit(5).run(propagate_rules);
        runner.print_report();
        egraph = runner.egraph;
        egraph.dot().to_png("target/5.png").unwrap();
        println!("nodes: {}; classes: {}", egraph.total_size(), egraph.number_of_classes());
        println!("stop reason: {:?}", runner.stop_reason);
    }

    
}
