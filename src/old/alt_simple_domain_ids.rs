use super::dom_expr::{*, Val::*};
use std::collections::HashMap;
use egg::*;

#[derive(Clone,Debug, PartialEq, Eq, Hash)]
pub enum Simple {
    Int(i32),
    List(Vec<Simple>),
}
use Simple::*;
type Val = super::dom_expr::Val<Simple>;
type DomExpr = super::dom_expr::DomExpr<Simple>;

type DSLFn = fn(&[Id], &mut DomExpr) -> Id;

lazy_static::lazy_static! {
    static ref PRIMS: HashMap<Symbol, Val> = vec![
            ("+".into(), PrimFun(CurriedFn::new("+".into(), 2))),
            ("*".into(), PrimFun(CurriedFn::new("*".into(), 2))),
            ("map".into(), PrimFun(CurriedFn::new("map".into(), 2))),
        ].into_iter().collect();
    
    static ref FNS: HashMap<Symbol, DSLFn> = vec![
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
    fn fn_of_prim(p: Symbol) -> DSLFn {
        FNS.get(&p).cloned().unwrap_or_else(|| panic!("unknown function primitive: {}", p))
    }
}




fn add(args: &[Id], handle: &mut DomExpr) -> Id {
    if let [Dom(Int(x)), Dom(Int(y))] = handle.get_many(args).as_slice() {
        handle.add_dom(Int(x+y))
    } else { panic!("type error: {:?}",args) }
}




fn mul(args: &[Id], handle: &mut DomExpr) -> Id {
    if let [Dom(Int(x)), Dom(Int(y))] = handle.get_many(args).as_slice() {
        handle.add_dom(Int(x*y))
    } else { panic!("type error: {:?}",args) }
}

fn map(args: &[Id], handle: &mut DomExpr) -> Id {
    if let [fn_id, xs_id] = args {
        if let Dom(List(xs)) = handle[*xs_id].clone() {
        let res = List(
            xs.into_iter()
                .map(|x| handle.add_dom(x)).collect::<Vec<_>>().into_iter()
                .map(|x| handle.apply(*fn_id, x)).collect::<Vec<_>>().into_iter()
                .map(|id| handle[id].unwrap_dom())
                .collect()
        );
        handle.add_dom(res)
    } else { panic!("type error: {:?}",args) }
    } else { panic!("type error: {:?}",args) }
}


