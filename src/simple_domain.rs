use super::dom_expr::{*, Val::*};
use std::collections::HashMap;
use egg::*;

#[derive(Clone,Debug, PartialEq, Eq, Hash)]
pub enum Simple {
    Int(i32),
    List(Vec<Simple>),
}
impl Simple {
    pub fn unwrap_int(self) -> i32 {
        match self {
            Simple::Int(i) => i,
            _ => panic!("Simple::unwrap_int: expected Int"),
        }
    }
    pub fn unwrap_list(self) -> Vec<Simple> {
        match self {
            Simple::List(l) => l,
            _ => panic!("Simple::unwrap_list: expected List"),
        }
    }
}

use Simple::*;
type Val = super::dom_expr::Val<Simple>;
type DomExpr = super::dom_expr::DomExpr<Simple>;
// type Result = super::dom_expr::Result<Simple>;

type DSLFn = fn(Vec<Val>, &mut DomExpr) -> Val;

define_semantics! {
    type Val = Val;
    type DSLFn = DSLFn;
    "+" = (add, 2),
    "*" = (mul, 2),
    "map" = (map, 2)
}

impl Domain for Simple {
    type Data = ();
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
}




fn add(mut args: Vec<Val>, _handle: &mut DomExpr) -> Val {
    let x = args.remove(0).unwrap_dom().unwrap_int();
    let y = args.remove(0).unwrap_dom().unwrap_int();
    Int(x+y).into()
}

fn mul(mut args: Vec<Val>, _handle: &mut DomExpr) -> Val {
    let x = args.remove(0).unwrap_dom().unwrap_int();
    let y = args.remove(0).unwrap_dom().unwrap_int();
    Int(x*y).into()
}

fn map(mut args: Vec<Val>, handle: &mut DomExpr) -> Val {
    let fn_val = args.remove(0);
    let xs = args.remove(0).unwrap_dom().unwrap_list();
    List(
        xs.into_iter()
            .map(|x| handle.apply(fn_val.clone(), x.into()).unwrap_dom())
            .collect()
    ).into()
}


