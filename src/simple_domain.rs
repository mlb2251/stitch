use super::dom_expr::{*, Val::*};
use std::collections::HashMap;
use egg::*;

#[derive(Clone,Debug, PartialEq, Eq, Hash)]
pub enum Simple {
    Int(i32),
    List(Vec<Val>),
}

pub enum TSimple {
    Int,
    List(Box<Type>)
}

use Simple::*;
type Val = super::dom_expr::Val<Simple>;
type DomExpr = super::dom_expr::DomExpr<Simple>;
type VResult = super::dom_expr::VResult<Simple>;
type Type = super::dom_expr::Type<Simple>;
type DSLFn = fn(Vec<Val>, &DomExpr) -> VResult;


/// From<Val> impls are needed for unwrapping values. We can assume the program
/// has been type checked so it's okay to panic if the type is wrong. Each val variant
/// must map to exactly one unwrapped type (though it doesnt need to be one to one in the
/// other direction)
impl From<Val> for i32 {
    fn from(v: Val) -> Self {
        match v {
            Dom(Simple::Int(i)) => i,
            _ => panic!("from_val_to_i32: not an int")
        }
    }
}
impl<T: From<Val>> From<Val> for Vec<T> {
    fn from(v: Val) -> Self {
        match v {
            Dom(Simple::List(v)) => v.into_iter().map(|v| v.into()).collect(),
            _ => panic!("from_val_to_vec: not a list")
        }
    }
}

/// These Into<Val>s are convenience functions. It's okay if theres not a one to one mapping
/// like this in all domains - it just makes .into() save us a lot of work if there is.
impl Into<Val> for i32 {
    fn into(self) -> Val {
        Dom(Simple::Int(self))
    }
}
impl<T: Into<Val>> Into<Val> for Vec<T> {
    fn into(self) -> Val {
        Dom(Simple::List(self.into_iter().map(|v| v.into()).collect()))
    }
}

define_semantics! {
    type Val = Val;
    type DSLFn = DSLFn;
    "+" = (add, 2), // Type::Fun(Type::Dom(Int), Type::Dom(Int), Type::Dom(Int))
    "*" = (mul, 2),
    "map" = (map, 2), // Fun(Fun(Dom(Int), Dom(Int)), Dom(List(Int)), Dom(List(Int)))
    "sum" = (sum, 1)
}

impl Domain for Simple {
    type Data = ();
    type DomType = TSimple;
    fn val_of_prim(p: Symbol) -> Option<Val> {
        PRIMS.get(&p).cloned().or_else(||
            // starts with digit -> Int
            if p.as_str().chars().next().unwrap().is_digit(10) {
                let i: i32 = p.as_str().parse().ok()?;
                Some(Int(i).into())
            }
            // starts with `[` -> List (must be all ints)
            else if p.as_str().chars().next().unwrap() == '[' {
                let intvec: Vec<i32> = serde_json::from_str(p.as_str()).ok()?;
                let valvec: Vec<Val> = intvec.into_iter().map(|v|Dom(Int(v))).collect();
                Some(List(valvec).into())
            } else {
                None
            }
        )
    }
    fn fn_of_prim(p: Symbol) -> DSLFn {
        FUNCS.get(&p).cloned().unwrap_or_else(|| panic!("unknown function primitive: {}", p))
    }
    // fn type_of_dom_val(v: &Self) -> Type {
    //     match v {
    //         Int(_) => Type::Int,
    //         List(l) => Type::List(Box::new(l.iter().map(|v| type_of_val(v)).collect())),
    //     }
    // }
}

fn add(mut args: Vec<Val>, _handle: &DomExpr) -> VResult {
    load_args!(args, x:i32, y:i32);
    Ok((x+y).into())
}

fn mul(mut args: Vec<Val>, _handle: &DomExpr) -> VResult {
    load_args!(args, x:i32, y:i32);
    Ok((x*y).into())
}

fn map(mut args: Vec<Val>, handle: &DomExpr) -> VResult {
    load_args!(args, fn_val: Val, xs: Vec<Val>);
    Ok(xs.into_iter()
        .map(|x| handle.apply(&fn_val, x))
        .collect::<Result<Vec<Val>,_>>()?
        .into())
}

fn sum(mut args: Vec<Val>, _handle: &DomExpr) -> VResult {
    load_args!(args, xs: Vec<i32>);
    Ok(xs.iter().sum::<i32>().into())
}


