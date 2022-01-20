use crate::*;
use sexp::{Sexp,Atom};
use std::fmt::Debug;
use std::rc::Rc;
use std::cell::RefCell;

/// Uncurries an s expression. For example: (app (app foo x) y) -> (foo x y)
/// panics if sexp is already uncurried.
pub fn uncurry_sexp(e: &Sexp) -> Sexp {
    match e {
        Sexp::List(orig_list) => {
            assert!(orig_list.len() > 1);
            // recurse on children
            let uncurried_children: Vec<Sexp> = orig_list.iter().map(|e| uncurry_sexp(e)).collect();
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
            let list: Vec<Sexp> = list.iter().map(|e| curry_sexp(e)).collect();
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

/// print some info about a Vec of programs
pub fn programs_info(programs: &Vec<Expr>) {
    let max_cost = programs.iter().map(|p| p.cost()).max().unwrap();
    let max_depth = programs.iter().map(|p| p.depth()).max().unwrap();
    println!("Programs:");
    println!("\t num: {}",programs.len());
    println!("\t max cost: {}",max_cost);
    println!("\t max depth: {}",max_depth);
}

/// provides a timestamp as a string in a format you can use for file/folder names: YYYY-MM-DD_HH-MM-SS
pub fn timestamp() -> String {
    format!("{}", chrono::Local::now().format("%Y-%m-%d_%H-%M-%S"))
}

pub fn search<L,A>(pat: &str, egraph: &EGraph<L,A>) -> Vec<SearchMatches>
where 
    L: Language,
    A: Analysis<L>
{
    let applam:Pattern<L> = pat.parse().unwrap();
    applam.search(&egraph)
}

pub fn save<L,A>(egraph: &EGraph<L,A>, name: &str, outdir: &str) 
where 
    L: Language,
    A: Analysis<L>
{
    egraph.dot().to_png(format!("{}/{}.png",outdir,name)).unwrap();
}

pub fn egraph_info<L,A>(egraph: &EGraph<L,A>) -> String 
where 
    L: Language,
    A: Analysis<L>
{
    format!("{} nodes, {} classes, {} memo", egraph.total_number_of_nodes(), egraph.number_of_classes(), egraph.total_size())
}

/// convenience function for returning arguments from a DSL function
pub fn ok<T: Into<Val<D>> , D:Domain>(v: T) -> VResult<D> {
    Ok(v.into())
}

/// convenience function for equality assertions
pub fn assert_eq_val<D:Domain, T>(v: &Val<D>, o: T)
where T: From<Val<D>>+ Debug + PartialEq
{
    assert_eq!(T::from(v.clone()), o);
}

/// convenience function for asserting that something executes to what you'd expect
pub fn assert_execution<D: Domain, T>(expr: &str, args: &[Val<D>], expected: T)
where T: From<Val<D>>+ Debug + PartialEq
{
    let e: Executable<D> = expr.parse().unwrap();
    let res = e.eval(&args).unwrap();
    assert_eq_val(&res,expected);
}

pub fn compression_factor(original: &Expr, compressed: &Expr) -> f64 {
    f64::from(original.cost())/f64::from(compressed.cost())
}

// An imbalanced tree structure with multiple reference counting
// and interior mutability
// Used to represent programs within the program sampler
#[derive(Debug)]
pub struct Node<'a> {
    data: Option<String>,
    children: Vec<Rc<RefCell<Node<'a>>>>,
}

impl<'a> Node<'a> {
    pub fn new_leaf(data: String) -> Rc<RefCell<Node<'a>>> {
        Rc::new(RefCell::new(Node {
            data: Some(data),
            children: vec![],
        }))
    }

    pub fn new_internal_node(data: String, children: Vec<Rc<RefCell<Node<'a>>>>)
        -> Rc<RefCell<Node<'a>>> {
        Rc::new(RefCell::new(Node {
            data: Some(data),
            children: children,
        }))
    }

    pub fn new_hole() -> Rc<RefCell<Node<'a>>> {
        Rc::new(RefCell::new(Node {
            data: None,
            children: vec![],
        }))
    }

    pub fn insert(&mut self, data: String, children: Vec<Rc<RefCell<Node<'a>>>>) {
        self.data = Some(data);
        self.children = children;
    }

    pub fn to_string(&self) -> String {
        let mut res = String::from("(");
        match &self.data {
            Some(s) => res.push_str(&s.clone()),
            None    => res.push_str("??")
        }
        for i in 0..self.children.len() {
            res.push_str(" ");
            res.push_str(&self.children[i].borrow().to_string());
        }
        res.push_str(")");
        return res;
    }
}