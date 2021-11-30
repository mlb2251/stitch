use super::dom_expr::{*, Val::*};
use std::collections::HashMap;
use egg::*;

#[derive(Clone,Debug, PartialEq, Eq, Hash)]
pub enum Simple {
    Int(i32),
    List(Vec<Val>), // todo change to Val since thats how foralls work
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




impl Simple {
    // pub fn int(self) -> Result<i32,VError> {
    //     match self {
    //         Simple::Int(i) => Ok(i),
    //         _ => Err("Simple::unwrap_int: expected Int".into()),
    //     }
    // }
    // pub fn list(self) -> Result<Vec<Simple>,VError> {
    //     match self {
    //         Simple::List(l) => Ok(l),
    //         _ => Err("Simple::unwrap_list: expected List".into()),
    //     }
    // }
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

// option 1: macros for loading
// must handle stuff like Simple::Int Simple::List (ie polymorphic) and Val::Func
// fn add(mut args: Vec<Val>, _handle: &DomExpr) -> VResult {
//     let x = load_arg!(Simple::Int, args); // macro auto derefs the Dom bit via case analysis
//     let y = load_arg!(Type::Func, args);
//     Ok(Int(x+y).into())
// }

// option 2: move the type sig into the macro allowing for a more natural signature.
// #[dslfn(Simple::Int, Type::List)]
// fn add(x:i32, y:i32, _handle: &DomExpr) -> VResult {
//     Ok(Int(x+y).into())
// }




fn add(mut args: Vec<Val>, _handle: &DomExpr) -> VResult {
    let x:i32 = args.remove(0).into();
    let y:i32 = args.remove(0).into();
    Ok(Int(x+y).into())
}

fn mul(mut args: Vec<Val>, _handle: &DomExpr) -> VResult {
    let x: i32 = args.remove(0).into();
    let y: i32 = args.remove(0).into();
    Ok(Int(x*y).into())
}

fn map(mut args: Vec<Val>, handle: &DomExpr) -> VResult {
    let fn_val = args.remove(0);
    let xs: Vec<Val> = args.remove(0).into();
    Ok(List(
        xs.into_iter()
            .map(|x| handle.apply(&fn_val, x.into()))
            .collect::<Result<_,_>>()?
    ).into())
}

fn sum(mut args: Vec<Val>, _handle: &DomExpr) -> VResult {
    let xs: Vec<i32> = args.remove(0).into();
    Ok(Int(xs.iter().sum::<i32>().into()).into())
}


