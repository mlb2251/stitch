use super::dom_expr::{*, Val::*};
use std::collections::HashMap;


#[derive(Clone,Debug, PartialEq, Eq, Hash)]
pub enum Simple {
    Int(i32),
    List(Vec<Simple>),
}
use Simple::*;
type Val = super::dom_expr::Val<Simple>;
type DomExpr = super::dom_expr::DomExpr<Simple>;

type DSLFn = fn(&[Val], &mut DomExpr) -> Val;

lazy_static::lazy_static! {
    static ref PRIMS: HashMap<egg::Symbol, Val> = vec![
            ("+".into(), PrimFun(CurriedFn::new("+".into(), 2))),
            ("*".into(), PrimFun(CurriedFn::new("*".into(), 2))),
            ("map".into(), PrimFun(CurriedFn::new("map".into(), 2))),
        ].into_iter().collect();
    
    static ref FNS: HashMap<egg::Symbol, DSLFn> = vec![
        ("+".into(), add as DSLFn),
        ("*".into(), mul as DSLFn),
        ("map".into(), map as DSLFn),
    ].into_iter().collect();
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
    fn fn_of_prim(p: egg::Symbol) -> fn(&[Val], &mut DomExpr) -> Val {
        FNS.get(&p).cloned().unwrap_or_else(|| panic!("unknown function primitive: {}", p))
    }
}




fn add(args: &[Val], _handle: &mut DomExpr) -> Val {
    if let [Dom(Int(x)), Dom(Int(y))] = args {
        Int(x+y).into()
    } else { panic!("type error: {:?}",args) }
}

fn mul(args: &[Val], _handle: &mut DomExpr) -> Val {
    if let [Dom(Int(x)), Dom(Int(y))] = args {
        Int(x*y).into()
    } else { panic!("type error: {:?}",args) }
}

fn map(args: &[Val], handle: &mut DomExpr) -> Val {
    if let [fn_val, Dom(List(xs))] = args {
        List(
            xs.iter()
                .map(|x| handle.apply(fn_val, &Dom(x.clone())))
                .map(|val| val.unwrap_dom())
                .collect()
        ).into()
    } else { panic!("type error: {:?}",args) }
}


