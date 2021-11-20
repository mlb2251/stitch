use super::expr::{*, Val as SuperVal, Val::Domain, Val::PrimFun};
use std::collections::HashMap;


#[derive(Clone,Debug)]
pub enum SimpleVal {
    Int(i32),
    List(Vec<SimpleVal>),
}
use SimpleVal::*;
type Val = SuperVal<SimpleVal>;


impl DomainVal for SimpleVal {
    fn val_of_prim(sym: &egg::Symbol) -> Option<Val> {
        PRIMS.get(sym).cloned().or_else(||
            sym.as_str().parse::<i32>().ok().map(|i| Domain(Int(i)))
        )
    }
}

lazy_static::lazy_static! {
    static ref PRIMS: HashMap<egg::Symbol, Val> = vec![
            ("+".into(), PrimFun(CurriedFn::new("+".into(), add, 2))),
            ("*".into(), PrimFun(CurriedFn::new("*".into(), add, 2)))
        ].into_iter().collect();
}




fn add(args: &[Val]) -> Val {
    if let [Domain(Int(x)), Domain(Int(y))] = args {
        Domain(Int(x+y))
    } else { panic!("type error") }
}

fn mul(args: &[Val]) -> Val {
    if let [Domain(Int(x)), Domain(Int(y))] = args {
        Domain(Int(x*y))
    } else { panic!("type error") }
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