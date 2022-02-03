// The primitive list domain from Josh Rule's thesis, p.170.

use crate::*;
use std::collections::HashMap;

#[derive(Clone,Debug, PartialEq, Eq, Hash)]
pub enum ListVal {
    Int(i32),
    // Nan,  // TODO to model NAN or not...
    Bool(bool),
    List(Vec<Val>),
}

type Val = domain::Val<ListVal>;
type Executable = domain::Executable<ListVal>;
type VResult = domain::VResult<ListVal>;
type DSLFn = domain::DSLFn<ListVal>;

use ListVal::*;
use domain::Val::*;

// this macro generates two global lazy_static constants: PRIM and FUNCS
// which get used by `val_of_prim` and `fn_of_prim` below. In short they simply
// associate the strings on the left with the rust function and arity on the right.
define_semantics! {
    ListVal;
    "cons" = (cons, 2),
    "+" = (add, 2),
    "-" = (sub, 2),
    ">" = (gt, 2),
    "if" = (branch, 3),
    "==" = (eq, 2),
    "is_empty" = (is_empty, 1),
    "head" = (head, 1),
    "tail" = (tail, 1),
    "fix" = (fix, 2)
}


// From<Val> impls are needed for unwrapping values. We can assume the program
// has been type checked so it's okay to panic if the type is wrong. Each val variant
// must map to exactly one unwrapped type (though it doesnt need to be one to one in the
// other direction)
impl From<Val> for i32 {
    fn from(v: Val) -> Self {
        match v {
            Dom(Int(i)) => i,
            _ => panic!("from_val_to_list: not an int")
        }
    }
}
impl From<Val> for bool {
    fn from(v: Val) -> Self {
        match v {
            Dom(Bool(b)) => b,
            _ => panic!("from_val_to_bool: not a bool")
        }
    }
}
impl<T: From<Val>> From<Val> for Vec<T> {
    fn from(v: Val) -> Self {
        match v {
            Dom(List(v)) => v.into_iter().map(|v| v.into()).collect(),
            _ => panic!("from_val_to_vec: not a list")
        }
    }
}

// These Into<Val>s are convenience functions. It's okay if theres not a one to one mapping
// like this in all domains - it just makes .into() save us a lot of work if there is.
impl Into<Val> for i32 {
    fn into(self) -> Val {
        Dom(Int(self))
    }
}
impl Into<Val> for bool {
    fn into(self) -> Val {
        Dom(Bool(self))
    }
}
impl<T: Into<Val>> Into<Val> for Vec<T> {
    fn into(self) -> Val {
        Dom(List(self.into_iter().map(|v| v.into()).collect()))
    }
}

fn parse_vec(vec: & Vec<serde_json::value::Value>) -> Vec<Val> {
    let valvec: Vec<Val> = vec.into_iter().map(|v| {
        if let Some(i) = v.as_i64() {
            Dom(Int(i as i32))
        } else if let Some(b) = v.as_bool() {
            Dom(Bool(b))
        } else {
            // not int, not bool -> must be array. If not, we error.
            // TODO make this spit out a more useful error than panic
            let arr = v.as_array().unwrap();
            Dom(List(parse_vec(arr)))
        }
    }).collect();
    return valvec;
}

impl Domain for ListVal {
    // we dont use Data here
    // TODO ask Matt why this is
    type Data = ();

    const TRUST_LEVEL: TrustLevel = TrustLevel::WontLoopMayPanic;

    // val_of_prim takes a symbol like "+" or "0" and returns the corresponding Val.
    // Note that it can largely just be a call to the global hashmap PRIMS that define_semantics generated
    // however you're also free to do any sort of generic parsing you want, allowing for domains with
    // infinite sets of values or dynamically generated values. For example here we support all integers
    // and all integer lists.
    fn val_of_prim(p: Symbol) -> Option<Val> {
        PRIMS.get(&p).cloned().or_else(||
            // starts with digit -> Int
            if p.as_str().chars().next().unwrap().is_digit(10) {
                let i: i32 = p.as_str().parse().ok()?;
                Some(Int(i).into())
            }
            // starts with "f" or "t" -> must be a bool (if not found in PRIMS)
            else if p.as_str().chars().next().unwrap() == 'f' || p.as_str().chars().next().unwrap() == 't' {
                let s: String = p.as_str().parse().ok()?;
                if s == "false" {
                    Some(Dom(Bool(false)))
                } else if s == "true" {
                    Some(Dom(Bool(true)))
                } else {
                    None
                }
            }
            // starts with `[` -> List
            // Note lists may contain ints, bools, or other lists in this domain
            else if p.as_str().chars().next().unwrap() == '[' {
                let elems: Vec<serde_json::value::Value> = serde_json::from_str(p.as_str()).ok()?;
                let valvec: Vec<Val> = parse_vec(&elems);
                Some(List(valvec).into())
            } else {
                None
            }
        )
    }

    // fn_of_prim takes a symbol and returns the corresponding DSL function. Again this is quite easy
    // with the global hashmap FUNCS created by the define_semantics macro.
    fn fn_of_prim(p: Symbol) -> Option<DSLFn> {
        FUNCS.get(&p).cloned()
    }
}


// *** DSL FUNCTIONS ***

fn cons(mut args: Vec<Val>, _handle: &Executable) -> VResult {
    load_args!(args, x:Val, xs:Vec<Val>); 
    let mut rxs = xs.clone();
    rxs.insert(0, x);
    ok(rxs)
}

fn add(mut args: Vec<Val>, _handle: &Executable) -> VResult {
    load_args!(args, x:i32, y:i32); 
    ok(x+y)
}

fn sub(mut args: Vec<Val>, _handle: &Executable) -> VResult {
    load_args!(args, x:i32, y:i32); 
    ok(x-y)
}

fn gt(mut args: Vec<Val>, _handle: &Executable) -> VResult {
    load_args!(args, x:i32, y:i32); 
    ok(x>y)
}

fn branch(mut args: Vec<Val>, _handle: &Executable) -> VResult {
    load_args!(args, b: bool, tbranch: Val, fbranch: Val); 
    ok(if b { tbranch } else { fbranch })
}

fn eq(mut args: Vec<Val>, handle: &Executable) -> VResult {
    load_args!(args, a:Val, b:Val); 
    match (a, b) {
        (Dom(Int(i)), Dom(Int(j))) => { return ok(i==j); },
        (Dom(Bool(b1)), Dom(Bool(b2))) => { return ok(b1==b2); },
        (Dom(List(l1)), Dom(List(l2))) => {
            let l1_len = l1.len();
            if l1_len != l2.len() {
                return ok(false);
            } else {
                let mut all_elems_equal = true;
                for i in 0..l1_len {
                    let elems_equal = eq(vec![l1[i].clone(), l2[i].clone()], handle);
                    match elems_equal {
                        VResult::Ok(Dom(Bool(b))) => { all_elems_equal = b; },
                        VResult::Err(s) => { return Err(s) }
                        _       => {
                            all_elems_equal = false;
                            break;
                        }
                    }
                }
                return ok(all_elems_equal);
            }
        }
        _ => { return ok(false); }
    }
}

fn is_empty(mut args: Vec<Val>, _handle: &Executable) -> VResult {
    load_args!(args, xs: Vec<Val>);
    ok(xs.is_empty())
}

fn head(mut args: Vec<Val>, _handle: &Executable) -> VResult {
    load_args!(args, xs: Vec<Val>);
    if xs.is_empty() {
        Err(String::from("head called on empty list"))
    } else {
        ok(xs[0].clone())
    }
}

fn tail(mut args: Vec<Val>, _handle: &Executable) -> VResult {
    load_args!(args, xs: Vec<Val>);
    if xs.is_empty() {
        Err(String::from("tail called on empty list"))
    } else {
        ok(xs[1..].to_vec())
    }
}

fn fix(mut args: Vec<Val>, handle: &Executable) -> VResult {
    load_args!(args, x: Val, fn_val: Val);
    println!("Running fix with x={:?}, fn_val={:?}", x, fn_val);

    // TODO at what level am I implementing fix here?
    // Should I replicate the y-combinator, or can I take some shortcuts due to
    // working in the meta language?

    if let VResult::Ok(fx) = handle.apply(&fn_val, x.clone()) {
        // we successfully applied the fn to x once and got some result fx
        // If x = fx, we have arrived at a fixpoint, so start unwrapping recursion
        // otherwise, recurse to get f(f(x))
        ok(fx)
        //if fx == x {
        //    ok(fx)
        //} else if let VResult::Ok(ffx) = fix(vec![fx, fn_val], handle) {
        //    ok(ffx)
        //} else {
        //    Err(String::from("recursive call gave error"))
        //}
    } else {
        Err(String::from("failed to apply fn to arg"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    

    #[test]
    fn eval_test() {

        // test cons
        let arg = ListVal::val_of_prim("[1,2,3]".into()).unwrap();
        assert_execution("(cons 0 $0)", &[arg], vec![0,1,2,3]);
        // test +
        assert_execution::<domains::prim_lists::ListVal, i32>("(+ 1 2)", &[], 3);
        // test -
        assert_execution::<domains::prim_lists::ListVal, i32>("(- 22 1)", &[], 21);
        // test >
        assert_execution::<domains::prim_lists::ListVal, bool>("(> 22 1)", &[], true);
        assert_execution::<domains::prim_lists::ListVal, bool>("(> 2 11)", &[], false);
        // test if
        assert_execution::<domains::prim_lists::ListVal, i32>("(if true 5 50)", &[], 5);
        assert_execution::<domains::prim_lists::ListVal, i32>("(if false 5 50)", &[], 50);
        // test ==
        assert_execution::<domains::prim_lists::ListVal, bool>("(== 5 5)", &[], true);
        assert_execution::<domains::prim_lists::ListVal, bool>("(== 5 50)", &[], false);
        // test is_empty
        let arg = ListVal::val_of_prim("[[],[3],[4,5]]".into()).unwrap();
        assert_execution("(is_empty $0)", &[arg], false);
        let arg = ListVal::val_of_prim("[]".into()).unwrap();
        assert_execution("(is_empty $0)", &[arg], true);
        // test head
        let arg = ListVal::val_of_prim("[[1,2],[3],[4,5]]".into()).unwrap();
        assert_execution("(head $0)", &[arg], vec![1,2]);
        // test tail
        let arg = ListVal::val_of_prim("[[1,2],[3],[4,5]]".into()).unwrap();
        assert_execution("(tail $0)", &[arg], vec![vec![3], vec![4, 5]]);
        let arg = ListVal::val_of_prim("[[1,2]]".into()).unwrap();
        assert_execution::<domains::prim_lists::ListVal, Vec<Val>>("(tail $0)", &[arg], vec![]);
        // test fix
        //let arg = ListVal::val_of_prim("[1,2,3,4,5]".into()).unwrap();
        //assert_execution("(fix $0 (lam (lam (if (is_empty $0) 0 (+ 1 ($1 (tail $0)))))))", &[arg], 5);
        //let arg = ListVal::val_of_prim("[1,2,3,4,5]".into()).unwrap();
        //assert_execution("(fix $0 (lam (lam (if (is_empty $0) $0 (cons (+ 1 (head $0)) ($1 (tail $0)))))))", &[arg], vec![2, 3, 4, 5, 6]);
    }
}