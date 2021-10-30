use egg::{rewrite as rw, *};
use std::collections::{HashSet,HashMap};

#[macro_use]
extern crate log;



const ARGC: i32 = 2;

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
    // fn num(&self) -> Option<i32> {
    //     match self {
    //         Lambda::Num(n) => Some(*n),
    //         _ => None,
    //     }
    // }
}

type EGraph = egg::EGraph<Lambda, LambdaAnalysis>;

#[derive(Default)]
struct LambdaAnalysis;

#[derive(Debug)]
struct Data {
    upward_refs: HashSet<i32>, // "how much higher"
    // free: HashSet<Id>,
    // constant: Option<(Lambda, PatternAst<Lambda>)>,
}

fn extract(eclass: Id, egraph: &EGraph) -> RecExpr<Lambda> {
    // expensively extracts a small program from the eclass
    let mut extractor = Extractor::new(&egraph, AstSize);
    let (_,p) = extractor.find_best(eclass);
    p // this is printable
}

fn extract_enode(enode: Lambda, egraph: &EGraph) -> String {
    // let mut expr : RecExpr<Lambda> = RecExpr::default();
    // enode.children().iter().for_each(|c| {
    //     expr.add(extract(*c, egraph).to_string());
    // });
    match enode {
        Lambda::Prim(p) => {format!("{}",p)},
        Lambda::Var(i) => {format!("{}",i)},
        Lambda::App([f,x]) => {format!("(app {} {})",extract(f,egraph),extract(x,egraph))},
        Lambda::Lam([b]) => {format!("(lam {})",extract(b,egraph))},
        _ => {format!("not rendered")},
    }
}


// fn eval(egraph: &EGraph, enode: &Lambda) -> Option<(Lambda, PatternAst<Lambda>)> {
//     let x = |i: &Id| egraph[*i].data.constant.as_ref().map(|c| &c.0);
//     match enode {
//         Lambda::Num(n) => Some((enode.clone(), format!("{}", n).parse().unwrap())),
//         Lambda::Bool(b) => Some((enode.clone(), format!("{}", b).parse().unwrap())),
//         Lambda::Add([a, b]) => Some((
//             Lambda::Num(x(a)?.num()? + x(b)?.num()?),
//             format!("(+ {} {})", x(a)?, x(b)?).parse().unwrap(),
//         )),
//         Lambda::Eq([a, b]) => Some((
//             Lambda::Bool(x(a)? == x(b)?),
//             format!("(= {} {})", x(a)?, x(b)?).parse().unwrap(),
//         )),
//         _ => None,
//     }
// }

impl Analysis<Lambda> for LambdaAnalysis {
    type Data = Data;
    fn merge(&self, to: &mut Data, from: Data) -> bool {
        // println!("merge {:?} {:?}", to, from);
        assert_eq!(to.upward_refs,from.upward_refs);
        false // says we did not modify `to` data i think
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
        Data { upward_refs: upward_refs }
    }

    fn modify(egraph: &mut EGraph, id: Id) {
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

fn is_not_same_var(v1: Var, v2: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _, subst| egraph.find(subst[v1]) != egraph.find(subst[v2])
}

// fn not_all_same(a: Var, b: Var, c: Pattern<Lambda>) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
//     move |egraph, eclass, subst|{
//         println!("checking sameness");
//         if egraph.find(subst[a]) != egraph.find(subst[b]) { return true; }
//         println!("passed first");
//         // should be rare to get this far
//         let cc = c.apply_one(egraph, eclass, subst);
//         assert_eq!(cc.len(), 1);
//         let res = egraph.find(subst[a]) != egraph.find(cc[0]);
//         println!("passed second? {}", res);
//         println!("extracted: cc: {} a: {}",extract(cc[0],egraph), extract(egraph.find(subst[a]),egraph));
//         res
//     }
// }



// egg::test_fn! {
//     lambda_under, rules(),
//     "(lam x (+ 4
//                (app (lam y (var y))
//                     4)))"
//     =>
//     // "(lam x (+ 4 (let y 4 (var y))))",
//     // "(lam x (+ 4 4))",
//     "(lam x 8))",
// }

// egg::test_fn! {
//     lambda_if_elim, rules(),
//     "(if (= (var a) (var b))
//          (+ (var a) (var a))
//          (+ (var a) (var b)))"
//     =>
//     "(+ (var a) (var b))"
// }

// egg::test_fn! {
//     lambda_let_simple, rules(),
//     "(let x 0
//      (let y 1
//      (+ (var x) (var y))))"
//     =>
//     // "(let ?a 0
//     //  (+ (var ?a) 1))",
//     // "(+ 0 1)",
//     "1",
// }

// egg::test_fn! {
//     #[should_panic(expected = "Could not prove goal 0")]
//     lambda_capture, rules(),
//     "(let x 1 (lam x (var x)))" => "(lam x 1)"
// }

// egg::test_fn! {
//     #[should_panic(expected = "Could not prove goal 0")]
//     lambda_capture_free, rules(),
//     "(let y (+ (var x) (var x)) (lam x (var y)))" => "(lam x (+ (var x) (var x)))"
// }

// egg::test_fn! {
//     #[should_panic(expected = "Could not prove goal 0")]
//     lambda_closure_not_seven, rules(),
//     "(let five 5
//      (let add-five (lam x (+ (var x) (var five)))
//      (let five 6
//      (app (var add-five) 1))))"
//     =>
//     "7"
// }

// egg::test_fn! {
//     lambda_compose, rules(),
//     "(let compose (lam f (lam g (lam x (app (var f)
//                                        (app (var g) (var x))))))
//      (let add1 (lam y (+ (var y) 1))
//      (app (app (var compose) (var add1)) (var add1))))"
//     =>
//     "(lam ?x (+ 1
//                 (app (lam ?y (+ 1 (var ?y)))
//                      (var ?x))))",
//     "(lam ?x (+ (var ?x) 2))"
// }

// egg::test_fn! {
//     lambda_if_simple, rules(),
//     "(if (= 1 1) 7 9)" => "7"
// }

// egg::test_fn! {
//     lambda_compose_many, rules(),
//     "(let compose (lam f (lam g (lam x (app (var f)
//                                        (app (var g) (var x))))))
//      (let add1 (lam y (+ (var y) 1))
//      (app (app (var compose) (var add1))
//           (app (app (var compose) (var add1))
//                (app (app (var compose) (var add1))
//                     (app (app (var compose) (var add1))
//                          (app (app (var compose) (var add1))
//                               (app (app (var compose) (var add1))
//                                    (var add1)))))))))"
//     =>
//     "(lam ?x (+ (var ?x) 7))"
// }

// egg::test_fn! {
//     #[cfg(not(debug_assertions))]
//     lambda_function_repeat, rules(),
//     runner = Runner::default()
//         .with_time_limit(std::time::Duration::from_secs(20))
//         .with_node_limit(150_000)
//         .with_iter_limit(60),
//     "(let compose (lam f (lam g (lam x (app (var f)
//                                        (app (var g) (var x))))))
//      (let repeat (fix repeat (lam fun (lam n
//         (if (= (var n) 0)
//             (lam i (var i))
//             (app (app (var compose) (var fun))
//                  (app (app (var repeat)
//                            (var fun))
//                       (+ (var n) -1)))))))
//      (let add1 (lam y (+ (var y) 1))
//      (app (app (var repeat)
//                (var add1))
//           2))))"
//     =>
//     "(lam ?x (+ (var ?x) 2))"
// }

// egg::test_fn! {
//     lambda_if, rules(),
//     "(let zeroone (lam x
//         (if (= (var x) 0)
//             0
//             1))
//         (+ (app (var zeroone) 0)
//         (app (var zeroone) 10)))"
//     =>
//     // "(+ (if false 0 1) (if true 0 1))",
//     // "(+ 1 0)",
//     "1",
// }

// egg::test_fn! {
//     #[cfg(not(debug_assertions))]
//     lambda_fib, rules(),
//     runner = Runner::default()
//         .with_iter_limit(60)
//         .with_node_limit(50_000),
//     "(let fib (fix fib (lam n
//         (if (= (var n) 0)
//             0
//         (if (= (var n) 1)
//             1
//         (+ (app (var fib)
//                 (+ (var n) -1))
//             (app (var fib)
//                 (+ (var n) -2)))))))
//         (app (var fib) 4))"
//     => "3"
// }


fn diff_eclasses(v1: Var, v2: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _, subst| egraph.find(subst[v1]) != egraph.find(subst[v2])
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

fn no_upward_refs(v: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _, subst| egraph[subst[v]].data.upward_refs.is_empty()
}
fn zero_not_in_upward_refs(v: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    move |egraph, _, subst| !egraph[subst[v]].data.upward_refs.contains(&0)
}

// fn large_upward_refs(v: Var) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
//     // true if has an outgoing ref of 2 or more (meaning theres a )
//     move |egraph, _, subst| {
//         match egraph[subst[v]].data.upward_refs.iter().max() {
//             Some(max) => ,
//             None => ,
//         }
//         }
//     }
// }


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





struct BeamCostFn {
    epsilon: f64,
}
impl CostFunction<Lambda> for BeamCostFn {
    type Cost = f64;
    fn cost<C>(&mut self, enode: &Lambda, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost
    {
        match enode {
            Lambda::Var(_) => 1.,
            Lambda::Prim(_) => 1.,
            Lambda::App([f, x]) => {
                let fcost = costs(*f);
                let xcost = costs(*x);
                fcost + xcost + self.epsilon
            }
            Lambda::Lam([b]) => {
                let bcost = costs(*b);
                bcost + self.epsilon
            }
            Lambda::Programs(ps) => {
                ps.iter().map(|p| costs(*p)).sum()
            }
        }
    }
}


// #[derive(Debug,Clone)]
// struct BeamCost {
//     cost_nolambda: f64,
//     child_nolambda: Lambda,
//     cost_any: f64,
//     child_any: Lambda,
// }
struct Beam {
    cost_nolambda_inventionless: (f64,usize),
    cost_any_inventionless: (f64,usize),
    cost_nolambda_under_invention: HashMap<Id,(f64,usize)>,
    cost_any_under_invention: HashMap<Id,(f64,usize)>,
}

struct BeamSearch {
    epsilon: f64,
    beam_size: usize,
    seen: HashMap<Id,Option<Beam>>,
}

fn update_if_better(old: &mut (f64,usize), new: (f64,usize)) {
    if new.0 < old.0 {
        *old = new;
    }
}

impl BeamSearch {
    fn eclass_cost(&mut self, eclass: Id, egraph: &EGraph) {
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

        let enodes: Vec<Lambda> = egraph[eclass].iter().cloned().collect();
        enodes.iter().enumerate().for_each(|(node_id,enode)| {
            match enode {
                Lambda::Var(_) | Lambda::Prim(_) => {
                    let cost_inventionless = 1.;
                    update_if_better(&mut beam.cost_any_inventionless, (cost_inventionless,node_id));
                    update_if_better(&mut beam.cost_nolambda_inventionless, (cost_inventionless,node_id));
                }
                Lambda::App([f, x]) => {
                    self.eclass_cost(*f, egraph);
                    self.eclass_cost(*x, egraph);
                    match (self.seen[f],self.seen[x]) {
                        (Some(fbeam),Some(xbeam)) => {
                            let fcost_inventionless = fbeam.cost_nolambda_inventionless.0;
                            let xcost_inventionless = xbeam.cost_any_inventionless.0;
                            let cost_inventionless = fcost_inventionless + xcost_inventionless + self.epsilon;
                            update_if_better(&mut beam.cost_any_inventionless, (cost_inventionless,node_id));
                            update_if_better(&mut beam.cost_nolambda_inventionless, (cost_inventionless,node_id));
                            let mut inventions: HashSet<Id> = fbeam.cost_nolambda_under_invention.keys().cloned().collect();
                            inventions.extend(xbeam.cost_any_under_invention.keys().cloned());
                            inventions.iter().for_each(|invention| {
                                let fcost_invention = fbeam.cost_nolambda_under_invention.get(invention)
                                    .map(|(cst,nid)|*cst) // extract out cost from cost,node_id tuple
                                    .unwrap_or(fcost_inventionless); // default to inventionless
                                let xcost_invention = xbeam.cost_any_under_invention.get(invention)
                                    .map(|(cst,nid)|*cst)
                                    .unwrap_or(xcost_inventionless);
                                let cost_invention = fcost_invention + xcost_invention + self.epsilon;
                                let mut entry = beam.cost_any_under_invention.entry(*invention).or_insert((f64::INFINITY,usize::MAX));
                                update_if_better(entry, (cost_invention,node_id));
                                let mut entry = beam.cost_nolambda_under_invention.entry(*invention).or_insert((f64::INFINITY,usize::MAX));
                                update_if_better(entry, (cost_invention,node_id));
                            });
                        }
                        _ => {} // one of the pointers caused a loop so not worth pursuing
                    }
                }
                Lambda::Lam([b]) => {
                    self.eclass_cost(*b, egraph);
                    let bcost = self.seen[b].unwrap().cost_any_inventionless.0;
                    let cost = bcost + self.epsilon;
                    beam.cost_any_inventionless = (cost,node_id);
                    for (inv,(inv_bcost,_)) in self.seen[b].unwrap().cost_any_under_invention.iter() {
                        let inv_cost = inv_bcost + self.epsilon;
                        assert!(inv_cost <= cost); // it would be nonsensical if this invention was chosen for a child yet didnt improve the cost of the parent above the baseline
                        beam.cost_any_under_invention.insert(*inv, (inv_cost,node_id));
                    }
                }
                Lambda::Programs(ps) => {
                    ps.iter().for_each(|p| self.eclass_cost(*p, egraph));
                    if ps.iter().any(|p| self.seen[p].is_none()) {
                        beam.cost_any_inventionless = (f64::INFINITY,node_id);
                        beam.cost_nolambda_inventionless = (f64::INFINITY,node_id);
                    }
                    let cost = ps.iter().map(|p| self.seen[p].unwrap().cost_any_inventionless.0).sum();
                    beam.cost_any_inventionless = (cost,node_id);
                    beam.cost_nolambda_inventionless = (cost,node_id);
                }
            }
        });

        let mut beamcost = BeamCost {
            cost_nolambda: f64::INFINITY,
            child_nolambda: Lambda::Var(100), // dummy
            cost_any: f64::INFINITY,
            child_any: Lambda::Var(100), // dummy
        };

        for (node,cost) in enodes.iter().zip(costs.iter()) {
            if *cost < beamcost.cost_any {
                beamcost.cost_any = *cost;
                beamcost.child_any = node.clone();
            }
            match node {
                Lambda::Lam(_) => { }
                _ => {
                    if *cost < beamcost.cost_nolambda {
                        beamcost.cost_nolambda = *cost;
                        beamcost.child_nolambda = node.clone();
                    }
                }
            }
        }

        // all terms should have a legit noninfininte cost
        assert!(beamcost.cost_any < f64::INFINITY);
        assert!(beamcost.cost_nolambda < f64::INFINITY);

        self.seen.insert(eclass, beamcost.clone());
        return beamcost
    }





    // fn enode_cost(&mut self, enode: &Lambda) -> BeamCost
    // {
    //     match enode {
    //         Lambda::Var(_) => BeamCost {cost_nolambda: 1., cost_any: 1. },
    //         Lambda::Prim(_) => BeamCost {cost_nolambda: 1., cost_any: 1. },
    //         Lambda::App([f, x]) => {
    //             let fcost = costs(*f);
    //             let xcost = costs(*x);
    //             fcost + xcost + self.epsilon
    //         }
    //         Lambda::Lam([b]) => {
    //             let bcost = costs(*b);
    //             bcost + self.epsilon
    //         }
    //         Lambda::Programs(ps) => {
    //             ps.iter().map(|p| costs(*p)).sum()
    //         }
    //     }
    // }
    // fn eclass_cost(&mut self, eclass: &EClass<Lambda,Data>) -> BeamCost
    // {
    //     eclass.min(|enode| self.enode_cost(enode))
    //     let mut cost_nolambda = 0.;
    //     let mut cost_any = 0.;
    //     for enode in eclass.iter() {
    //         let cost = self.enode_cost(enode);
    //         cost_nolambda += cost.cost_nolambda;
    //         cost_any += cost.cost_any;
    //     }
    //     BeamCost {cost_nolambda, cost_any}
    // }
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

        //todo multiarg is an insane slowdown. Ofc you could investigate why this is, but also i think its just fine to run it at the END bc if we do all possible refactorings then everything will already be in a perfect position for multiarg abstracting?
        // applam-multiarg: this is just the transformation that takes an applam applam setup and moves it so
        // its appapplamlam. Theres a tradeoff here bc this feels superficial and I worry about adding rewrite rules,
        // but a) this feels relatively safe and b) structural hashing wise you actually REALLY want the two lambdas on
        // top of each other.
        // btw we do need to downshift the outgoing refs of arginner since it gets hoisted up,
        // and furthermore we need to make sure its not pointing to argouter (by making sure $0 isnt an
        // upward ref)
        // rw!("applam-multiarg"; "(app (lam (app (lam ?body) ?arginner)) ?argouter)"
        //     => {Shifter {
        //         incr_by: -1, // how much to increment by eg +1 or -1
        //         to_shift: var("?arginner"), // expression to shift
        //         rhs: "(app (app (lam (lam (?body))) ?argouter) ?arginner)".parse().unwrap(), // expr to be unified with original LHS - but with to_shift modified!
        //     }}
        //     // condition: cant raise arginner above argouter if arginner points to argouter
        //     if zero_not_in_upward_refs(var("?arginner"))
        // ),

        
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

    let mut egraph: EGraph = Default::default();
    // egraph.add_expr(&"(x)".parse().unwrap());
    // egraph.add_expr(&"(programs (app - y))".parse().unwrap());
    // egraph.add_expr(&"(programs  (app - y) (app (app - x) x))".parse().unwrap());
    // egraph.add_expr(&"(programs (app (app x x) (app y y)))".parse().unwrap());
    // egraph.add_expr(&"(programs (app (app (app + x) z) (app (app + x) y)))".parse().unwrap());
    // egraph.add_expr(&"(programs (app (app (app + x) z) (app (app + x) y)) (app (app (app + x) z) (app (app + x) y)))".parse().unwrap());

    // first dreamcoder program
    // egraph.add_expr(&"(lam (app (app (app logo_forLoop t3) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t3)) 0)))) 0))".parse().unwrap());
    // second dreamcoder program
    // egraph.add_expr(&"(lam (app (app (app logo_forLoop t3) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t3)) 0)))) 0))".parse().unwrap());

    
    // 19 dreamcoder programs w heavy overlap
    egraph.add_expr(&"(programs (lam (app (app (app logo_forLoop t3) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t3)) 0)))) 0)) (lam (app (app (app logo_forLoop t8) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t8)) 0)))) 0)) (lam (app (app (app logo_forLoop t8) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t8)) 0)))) 0)) (lam (app (app (app logo_forLoop t9) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t9)) 0)))) 0)) (lam (app (app (app logo_forLoop t9) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t9)) 0)))) 0)) (lam (app (app (app logo_forLoop logo_IFTY) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_epsL) t1)) logo_epsA) 0)))) 0)) (lam (app (app (app logo_forLoop logo_IFTY) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_epsL) t2)) logo_epsA) 0)))) 0)) (lam (app (app (app logo_forLoop logo_IFTY) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_epsL) t5)) logo_epsA) 0)))) 0)) (lam (app (app (app logo_forLoop t4) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t4)) 0)))) 0)) (lam (app (app (app logo_FWRT logo_UL) logo_ZA) 0)) (lam (app (app (app logo_FWRT logo_ZL) (app (app logo_DIVA logo_UA) t4)) (app (app (app logo_FWRT logo_UL) logo_ZA) 0))) (lam (app (app (app logo_forLoop t4) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t4)) 0)))) 0)) (lam (app (app (app logo_forLoop t5) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t5)) 0)))) 0)) (lam (app (app (app logo_forLoop t5) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t5)) 0)))) 0)) (lam (app (app (app logo_forLoop t6) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t6)) 0)))) 0)) (lam (app (app (app logo_forLoop t9) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) 1)) (app (app logo_DIVA logo_UA) t4)) 0)))) 0)) (lam (app (app (app logo_forLoop t6) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t2)) (app (app logo_DIVA logo_UA) t6)) 0)))) 0)) (lam (app (app (app logo_forLoop t7) (lam (lam (app (app (app logo_FWRT (app (app logo_MULL logo_UL) t1)) (app (app logo_DIVA logo_UA) t7)) 0)))) 0)))".parse().unwrap());



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
        let mut egraph = runner.egraph;
        let find_cand: Pattern<Lambda> = "(lam ?b)".parse().unwrap();
        let matches = find_cand.search(&egraph);
        let cands: Vec<Id> = matches.iter().map(|m| m.eclass).collect();
        println!("{:?}", cands.len());
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
