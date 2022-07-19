// The primitive list domain from Josh Rule's thesis, p.170.

use crate::*;
use std::collections::HashMap;

#[derive(Clone,Debug, PartialEq, Eq, Hash)]
pub enum ListVal {
    Int(i32),
    // Nan,  // TODO to model NAN or not...(josh rule dsl does it for things outside of 0-99)
    Bool(bool),
    List(Vec<Val>),
}

// In this domain, we limit how many times "fix" can be invoked.
// This is a crude way of finding infinitely looping programs.
const MAX_FIX_INVOCATIONS: u32 = 100;

type Val = domain::Val<ListVal>;
type LazyVal = domain::LazyVal<ListVal>;
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
impl From<i32> for Val {
    fn from(i: i32) -> Val {
        Dom(Int(i))
    }
}
impl From<bool> for Val {
    fn from(b: bool) -> Val {
        Dom(Bool(b))
    }
}
impl<T: Into<Val>> From<Vec<T>> for Val {
    fn from(vec: Vec<T>) -> Val {
        Dom(List(vec.into_iter().map(|v| v.into()).collect()))
    }
}

fn parse_vec(vec: &[serde_json::value::Value]) -> Vec<Val> {
    let valvec: Vec<Val> = vec.iter().map(|v| {
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
    valvec
}

impl Domain for ListVal {

    type Data = u32;  // Use Data as fix-point invocation counter

    const TRUST_LEVEL: TrustLevel = TrustLevel::WontLoopMayPanic;

    // val_of_prim takes a symbol like "+" or "0" and returns the corresponding Val.
    // Note that it can largely just be a call to the global hashmap PRIMS that define_semantics generated
    // however you're also free to do any sort of generic parsing you want, allowing for domains with
    // infinite sets of values or dynamically generated values. For example here we support all integers
    // and all integer lists.
    fn val_of_prim(p: Symbol) -> Option<Val> {
        PRIMS.get(&p).cloned().or_else(||
            // starts with digit -> Int
            if p.as_str().chars().next().unwrap().is_ascii_digit() {
                let i: i32 = p.as_str().parse().ok()?;
                Some(Int(i).into())
            }
            // starts with "f" or "t" -> must be a bool (if not found in PRIMS)
            else if p.as_str().starts_with('f') || p.as_str().starts_with('t') {
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
            else if p.as_str().starts_with('[') {
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

    fn get_fns() -> HashMap<egg::Symbol, DSLFn> {
        FUNCS.clone()
    }
}


// *** DSL FUNCTIONS ***

fn cons(mut args: Vec<LazyVal>, handle: &Executable) -> VResult {
    load_args!(handle, args, x:Val, xs:Vec<Val>); 
    let mut rxs = xs;
    rxs.insert(0, x);
    ok(rxs)
}

fn add(mut args: Vec<LazyVal>, handle: &Executable) -> VResult {
    load_args!(handle, args, x:i32, y:i32); 
    ok(x+y)
}

fn sub(mut args: Vec<LazyVal>, handle: &Executable) -> VResult {
    load_args!(handle, args, x:i32, y:i32); 
    ok(x-y)
}

fn gt(mut args: Vec<LazyVal>, handle: &Executable) -> VResult {
    load_args!(handle, args, x:i32, y:i32); 
    ok(x>y)
}

fn branch(mut args: Vec<LazyVal>, handle: &Executable) -> VResult {
    load_args!(handle, args, b: bool);
    load_args_lazy!(args, tbranch: LazyVal, fbranch: LazyVal); 
    if b { 
        tbranch.eval(handle)
    } else { 
        fbranch.eval(handle)
    }
}

fn eq(mut args: Vec<LazyVal>, handle: &Executable) -> VResult {
    load_args!(handle, args, x:Val, y:Val); 
    match (x, y) {
        (Dom(Int(i)),  Dom(Int(j)))  => { ok(i==j) },
        (Dom(Bool(a)), Dom(Bool(b))) => { ok(a==b) },
        (Dom(List(l)), Dom(List(k))) => {
            if l.len() != k.len() {
                ok(false)
            } else {
                for (a,b) in l.iter().zip(k.iter()) {
                    match eq(vec![LazyVal::new_strict(a.clone()), LazyVal::new_strict(b.clone())], handle)? {
                        Dom(Bool(b)) => if !b { return ok(false) },
                        _ => unreachable!() // eq should never return a non-bool
                        }
                    }
                ok(true)
            }
        }
        _ => { ok(false) } // todo: or type error?
    }
}

fn is_empty(mut args: Vec<LazyVal>, handle: &Executable) -> VResult {
    load_args!(handle, args, xs: Vec<Val>);
    ok(xs.is_empty())
}

fn head(mut args: Vec<LazyVal>, handle: &Executable) -> VResult {
    load_args!(handle, args, xs: Vec<Val>);
    if xs.is_empty() {
        Err(String::from("head called on empty list"))
    } else {
        ok(xs[0].clone())
    }
}

fn tail(mut args: Vec<LazyVal>, handle: &Executable) -> VResult {
    load_args!(handle, args, xs: Vec<Val>);
    if xs.is_empty() {
        Err(String::from("tail called on empty list"))
    } else {
        ok(xs[1..].to_vec())
    }
}

fn fix(mut args: Vec<LazyVal>, handle: &Executable) -> VResult {
    *handle.data.borrow_mut() += 1;
    if *handle.data.borrow() > MAX_FIX_INVOCATIONS {
        return Err(format!("Exceeded max number of fix invocations. Max was {}", MAX_FIX_INVOCATIONS));
    }
    load_args!(handle, args, fn_val: Val, x: Val);

    // fix f x = f(fix f)(x)
    let fixf = PrimFun(CurriedFn::new_with_args(Symbol::from("fix"), 2, vec![LazyVal::new_strict(fn_val.clone())]));
    if let VResult::Ok(ffixf) = handle.apply(&fn_val, fixf) {
        handle.apply(&ffixf, x)
    } else {
        Err(String::from("Could not apply fixf to f"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    

    #[test]
    fn eval_test() {

        let arg = ListVal::val_of_prim("[]".into()).unwrap();
        assert_execution::<ListVal, Vec<Val>>("(if (is_empty $0) $0 (tail $0))", &[arg], vec![]);

        // test cons
        let arg = ListVal::val_of_prim("[1,2,3]".into()).unwrap();
        assert_execution("(cons 0 $0)", &[arg], vec![0,1,2,3]);

        // test +
        assert_execution::<ListVal, i32>("(+ 1 2)", &[], 3);

        // test -
        assert_execution::<ListVal, i32>("(- 22 1)", &[], 21);

        // test >
        assert_execution::<ListVal, bool>("(> 22 1)", &[], true);
        assert_execution::<ListVal, bool>("(> 2 11)", &[], false);

        // test if
        assert_execution::<ListVal, i32>("(if true 5 50)", &[], 5);
        assert_execution::<ListVal, i32>("(if false 5 50)", &[], 50);

        // test ==
        assert_execution::<ListVal, bool>("(== 5 5)", &[], true);
        assert_execution::<ListVal, bool>("(== 5 50)", &[], false);
        let arg1 = ListVal::val_of_prim("[[],[3],[4,5]]".into()).unwrap();
        let arg2 = ListVal::val_of_prim("[[],[3],[4,5]]".into()).unwrap();
        assert_execution::<ListVal, bool>("(== $0 $1)", &[arg1, arg2], true);
        let arg1 = ListVal::val_of_prim("[[],[3],[4,5]]".into()).unwrap();
        let arg2 = ListVal::val_of_prim("[[3],[4,5]]".into()).unwrap();
        assert_execution::<ListVal, bool>("(== $0 $1)", &[arg1, arg2], false);
        let arg1 = ListVal::val_of_prim("[[]]".into()).unwrap();
        let arg2 = ListVal::val_of_prim("[]".into()).unwrap();
        assert_execution::<ListVal, bool>("(== $0 $1)", &[arg1, arg2], false);
        let arg1 = ListVal::val_of_prim("[]".into()).unwrap();
        let arg2 = ListVal::val_of_prim("[]".into()).unwrap();
        assert_execution::<ListVal, bool>("(== $0 $1)", &[arg1, arg2], true);

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
        assert_execution::<ListVal, Vec<Val>>("(tail $0)", &[arg], vec![]);

        // test fix
        let arg = ListVal::val_of_prim("[]".into()).unwrap();
        assert_execution("(fix (lam (lam (if (is_empty $0) 0 (+ 1 ($1 (tail $0)))))) $0)", &[arg], 0);
        let arg = ListVal::val_of_prim("[1,2,3,2,1]".into()).unwrap();
        assert_execution("(fix (lam (lam (if (is_empty $0) 0 (+ 1 ($1 (tail $0)))))) $0)", &[arg], 5);
        let arg = ListVal::val_of_prim("[1,2,3,4,5]".into()).unwrap();
        assert_execution("(fix (lam (lam (if (is_empty $0) $0 (cons (+ 1 (head $0)) ($1 (tail $0)))))) $0)", &[arg], vec![2, 3, 4, 5, 6]);
        let arg = ListVal::val_of_prim("[1,2,3,4,5]".into()).unwrap();
        assert_error::<ListVal, Val>(
            "(fix (lam (lam (if (is_empty $0) $0 (cons (+ 1 (head $0)) ($1 $0))))) $0)",
            &[arg],
            format!("Exceeded max number of fix invocations. Max was {}", MAX_FIX_INVOCATIONS));
    }
}