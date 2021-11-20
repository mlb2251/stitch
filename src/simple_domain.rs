use super::expr::{*, Val::Dom, Val::PrimFun};
use std::collections::HashMap;


#[derive(Clone,Debug, PartialEq, Eq, Hash)]
pub enum Simple {
    Int(i32),
    List(Vec<Simple>),
}
use Simple::*;
type Val = super::expr::Val<Simple>;
type DomExpr = super::expr::DomExpr<Simple>;


impl Domain for Simple {
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
}

lazy_static::lazy_static! {
    static ref PRIMS: HashMap<egg::Symbol, Val> = vec![
            ("+".into(), PrimFun(CurriedFn::new("+".into(), add, 2))),
            ("*".into(), PrimFun(CurriedFn::new("*".into(), mul, 2))),
            ("map".into(), PrimFun(CurriedFn::new("map".into(), map, 2))),
        ].into_iter().collect();
}




fn add(args: &[Val], handle_: &mut DomExpr) -> Val {
    if let [Dom(Int(x)), Dom(Int(y))] = args {
        Int(x+y).into()
    } else { panic!("type error: {:?}",args) }
}

fn mul(args: &[Val], handle_: &mut DomExpr) -> Val {
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






// Simple example language
// 
// + :: int -> int -> int
// * :: int -> int -> int
// 0 :: int
// 1 :: int
// 2 :: int
// 3 :: int
// nil :: [int]
// cons :: int -> [int] -> [int]
// map :: (int -> int) -> [int] -> [int]



// enum Type {
//     TBase,// base type like int or bool
//     TCon, // type constructor like List
//     TFun
// }

// enum ExampleUserVal {
//     Int(i32),
//     Bool(bool),
//     List(Vec<ExampleUserVal>),
// }

// fn plus(a: i32, b: i32) -> i32 {
//     a + b
// }

// fn times(a: i32, b: i32) -> i32 {
//     a + b
// }