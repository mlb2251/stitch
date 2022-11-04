use crate::expr::expr::Expr;
use sexp::{Sexp,Atom};

/// Uncurries an s expression. For example: (app (app foo x) y) -> (foo x y)
/// panics if sexp is already uncurried.
pub fn uncurry_sexp(e: &Sexp) -> Sexp {
    match e {
        Sexp::List(orig_list) => {
            assert!(orig_list.len() > 1);
            // recurse on children
            let uncurried_children: Vec<Sexp> = orig_list.iter().map(uncurry_sexp).collect();
            match uncurried_children[0].to_string().as_str() {
                "lam" => {
                    Sexp::List(uncurried_children)
                },
                "app" =>  {
                    // (app (app foo x) y) -> (foo x y)
                    assert_eq!(uncurried_children.len(), 3);

                    // see if `f` is also an app in which case we'll have to flatten
                    let has_inner_app = if let Sexp::List(innerlist) = &orig_list[1] {
                        innerlist[0].to_string().as_str() == "app"
                    } else { false };

                    let f = uncurried_children[1].clone();
                    let x = uncurried_children[2].clone();
                    
                    let mut res = vec![];

                    if has_inner_app {
                        match f {
                            Sexp::List(list) => { res.extend(list) }
                            _ => panic!("expected list, got {}", f)
                        }
                    } else {
                        res.push(f);
                    }
                    res.push(x);
                    Sexp::List(res)
                },
                "programs" => {
                    Sexp::List(uncurried_children)
                }
                _ => {
                    panic!("not curried {}",e);
                }
            }
        },
        Sexp::Atom(atom) => Sexp::Atom(atom.clone())
    }
}

/// Currys an s expression. For example: (foo x y) -> (app (app foo x) y)
/// panics if sexp is already curried.
pub fn curry_sexp(e: &Sexp) -> Sexp {
    let app:Sexp = Sexp::Atom(Atom::S("app".into()));
    match e {
        Sexp::List(list) => {
            assert!(list.len() > 1);
            // recurse on children
            let list: Vec<Sexp> = list.iter().map(curry_sexp).collect();
            match list[0].to_string().as_str() {
                "lam" => {
                    Sexp::List(list)
                },
                "app" => panic!("already curried: {}",e),
                "programs" => {
                    Sexp::List(list)
                }
                _ => {
                    // (foo x y) -> (app (app foo x) y)
                    let mut res = Sexp::List(vec![app.clone(), list[0].clone(), list[1].clone()]);
                    for item in list.iter().skip(2) {
                        res = Sexp::List(vec![app.clone(), res, item.clone()])
                    }
                    res
                }
            }
        },
        Sexp::Atom(atom) => Sexp::Atom(atom.clone())
    }
}

impl std::str::FromStr for Expr {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.contains("(app ") {
            // this is curried
            Self::from_curried(s)
        } else {
            // this is uncurried. Note that even if it's curried and just lacks
            // a "(app " then that means that it's identical to an uncurried one. 
            Self::from_uncurried(s)
        }
    }
}