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

fn parse_vec(vec: & Vec<serde_json::value::Value>) -> Option<Vec<Val>> {
    // TODO this will SILENTLY fail to parse elems of the vector which are not
    // themselves Vals
    let valvec: Vec<Val> = vec.into_iter().filter_map(|v| {
        if let Some(i) = v.as_i64() {
            Some(Dom(Int(i as i32)))
        } else if let Some(b) = v.as_bool() {
            Some(Dom(Bool(b)))
        } else if let Some(a) = v.as_array() {
            Some(Dom(List(parse_vec(a).unwrap())))
        } else {
            None
        }
    }).collect();
    Some(valvec)
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
                let valvec: Vec<Val> = parse_vec(&elems).unwrap();
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
        (Dom(Int(i)), Dom(Int(j))) => ok(i==j),
        (Dom(Bool(b1)), Dom(Bool(b2))) => ok(b1==b2),
        (Dom(List(l1)), Dom(List(l2))) => {
            let l1_len = l1.len();
            if l1_len != l2.len() {
                ok(false)
            } else {
                let mut all_elems_equal = true;
                for i in 0..l1_len {
                    let elems_equal = eq(vec![l1[i].clone(), l2[i].clone()], handle);
                    match elems_equal {
                        VResult::Ok(Dom(Bool(b))) => continue,
                        _       => {
                            all_elems_equal = false;
                            break;
                        }
                    }
                }
                ok(all_elems_equal)
            }
        }
        _ => ok(false)
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
    // TODO how to handle the result being either a Val or an Error?

    if let VResult::Ok(fx) = handle.apply(&fn_val, x) {
        // we successfully applied the fn to x once and got some result fx
        // try doing it recursively
        if let VResult::Ok(rfx) = fix(vec![fx, fn_val], handle) {
            ok(rfx)
        } else {
            Err(String::from("recursive call gave error"))
        }
    } else {
        Err(String::from("failed to apply fn to arg"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    

    #[test]
    fn eval_test() {

        assert_execution::<domains::prim_lists::ListVal, i32>("(+ 1 2)", &[], 3);
        assert_execution::<domains::prim_lists::ListVal, i32>("(- 22 1)", &[], 21);
        assert_execution::<domains::prim_lists::ListVal, bool>("(> 22 1)", &[], true);
        assert_execution::<domains::prim_lists::ListVal, bool>("(> 2 11)", &[], false);

        let arg = ListVal::val_of_prim("[1,2,3]".into()).unwrap();
        assert_execution("(cons 0 $0)", &[arg], vec![0,1,2,3]);

        //let arg = ListVal::val_of_prim("[1,2,3]".into()).unwrap();
        //assert_execution("(sum (map (lam (+ 1 $0)) $0))", &[arg], 9);

        //let arg = ListVal::val_of_prim("[1,2,3]".into()).unwrap();
        //assert_execution("(map (lam (* $0 $0)) (map (lam (+ 1 $0)) $0))", &[arg], vec![4,9,16]);

        //let arg = ListVal::val_of_prim("[1,2,3]".into()).unwrap();
        //assert_execution("(map (lam (* $0 $0)) (map (lam (+ (sum $1) $0)) $0))", &[arg], vec![49,64,81]);

    }
}