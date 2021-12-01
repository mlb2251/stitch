use dreamegg::*;
use dreamegg::domains::simple::*;
type DomExpr = dreamegg::DomExpr<SimpleVal>;

fn main() {
    // let e: DomExpr = Expr::from_uncurried("(+ 1 2)").into();
    // println!("{}",e);
    // let res = e.eval(&[]).unwrap().0.unwrap();
    // println!("-> {:?}",res);
    // assert_eq!(i32::from(res),3);

    // let e: DomExpr = Expr::from_uncurried("(map (lam (+ 1 $0)) $0)").into();
    // println!("{}",e);
    // let arg = SimpleVal::val_of_prim("[1,2,3]".into()).unwrap();
    // let res = e.eval(&[arg]).unwrap().0.unwrap();
    // println!("-> {:?}",res);
    // assert_eq!(<Vec<i32>>::from(res),vec![2,3,4]);
    // // println!("{}",e.pretty_evals());

    // let e: DomExpr = Expr::from_uncurried("(sum (map (lam (+ 1 $0)) $0))").into();
    // println!("{}",e);
    // let arg = SimpleVal::val_of_prim("[1,2,3]".into()).unwrap();
    // let res = e.eval(&[arg]).unwrap().0.unwrap();
    // println!("-> {:?}",res);
    // assert_eq!(i32::from(res),9);

    // let e: DomExpr = Expr::from_uncurried("(map (lam (* $0 $0)) (map (lam (+ 1 $0)) $0))").into();
    // println!("{}",e);
    // let arg = SimpleVal::val_of_prim("[1,2,3]".into()).unwrap();
    // let res = e.eval(&[arg]).unwrap().0.unwrap();
    // println!("-> {:?}",res);
    // assert_eq!(<Vec<i32>>::from(res),vec![4,9,16]);
    

    // let e: DomExpr = Expr::from_uncurried("(map (lam (* $0 $0)) (map (lam (+ (sum $1) $0)) $0))").into();
    // println!("{}",e);
    // let arg = SimpleVal::val_of_prim("[1,2,3]".into()).unwrap();
    // let res = e.eval(&[arg]).unwrap().0.unwrap();
    // println!("-> {:?}",res);
    // assert_eq!(<Vec<i32>>::from(res),vec![49,64,81]);
}
