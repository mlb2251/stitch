use super::expr::{*, Val as SuperVal, Val::Domain};

#[derive(Clone)]
pub enum SimpleVal {
    Int(i32),
    List(Vec<SimpleVal>),
}
use SimpleVal::*;
type Val = SuperVal<SimpleVal>;


impl DomainVal for SimpleVal {

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