use super::dom_expr::{*, Val::*};
use std::collections::HashMap;
use egg::*;

#[derive(Clone,Debug, PartialEq, Eq, Hash)]
pub enum Simple {
    Int(i32),
    List(Vec<Simple>),
}

pub enum TSimple {
    Int,
    List(Box<TSimple>)
}

impl Simple {
    pub fn unwrap_int(self) -> Result<i32,VError> {
        match self {
            Simple::Int(i) => Ok(i),
            _ => Err("Simple::unwrap_int: expected Int".into()),
        }
    }
    pub fn unwrap_list(self) -> Result<Vec<Simple>,VError> {
        match self {
            Simple::List(l) => Ok(l),
            _ => Err("Simple::unwrap_list: expected List".into()),
        }
    }
}

use Simple::*;
type Val = super::dom_expr::Val<Simple>;
type DomExpr = super::dom_expr::DomExpr<Simple>;
type VResult = super::dom_expr::VResult<Simple>;
type Type = super::dom_expr::Type<Simple>;

type DSLFn = fn(Vec<Val>, &mut DomExpr) -> VResult;

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
    fn val_of_prim(p: egg::Symbol) -> Option<Val> {
        PRIMS.get(&p).cloned().or_else(||
            // starts with digit -> Int
            if p.as_str().chars().next().unwrap().is_digit(10) {
                let i: i32 = p.as_str().parse().ok()?;
                Some(Int(i).into())
            }
            // starts with `[` -> List (must be all ints)
            else if p.as_str().chars().next().unwrap() == '[' {
                let intvec: Vec<i32> = serde_json::from_str(p.as_str()).ok()?;
                let valvec: Vec<Simple> = intvec.into_iter().map(Int).collect();
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




fn add(mut args: Vec<Val>, _handle: &mut DomExpr) -> VResult {
    let x = args.remove(0).unwrap_dom()?.unwrap_int()?;
    let y = args.remove(0).unwrap_dom()?.unwrap_int()?;
    Ok(Int(x+y).into())
}

fn mul(mut args: Vec<Val>, _handle: &mut DomExpr) -> VResult {
    let x = args.remove(0).unwrap_dom()?.unwrap_int()?;
    let y = args.remove(0).unwrap_dom()?.unwrap_int()?;
    Ok(Int(x*y).into())
}

fn map(mut args: Vec<Val>, handle: &mut DomExpr) -> VResult {
    let fn_val = args.remove(0);
    let xs = args.remove(0).unwrap_dom()?.unwrap_list()?;
    Ok(List(
        xs.into_iter()
            .map(|x| handle.apply(fn_val.clone(), x.into()).and_then(|v| v.unwrap_dom()))
            .collect::<Result<_,_>>()?
    ).into())
}

fn sum(mut args: Vec<Val>, _handle: &mut DomExpr) -> VResult {
    let xs = args.remove(0).unwrap_dom()?.unwrap_list()?;
    Ok(Int(xs.into_iter()
            .map(|x| x.unwrap_int())
            .sum::<Result<_,_>>()? // result implements sum
            ).into())
}


