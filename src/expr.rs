use crate::*;
use std::fmt::{self, Formatter, Display, Debug};
use std::hash::Hash;
use sexp::Sexp;
use serde::{Serialize, Deserialize};

/// A node of an untyped lambda calculus expression compatible with `egg` but also used more widely throughout this crate.
/// Note that there is no domain associated with this object. This makes it easy to run compression on
/// domains that don't have semantics yet, makes it easy to add new prims (eg learned functions), etc.
/// You'll have to specify a domain when you execute the expression, type check it, etc, but you can easily do
/// that at the time through generics on those functions.
/// 
/// Variants:
/// * Var(i): "$i" a debruijn index variable
/// * IVar(i): "#i" a debruijn index variable used by inventions (advantage: readability of inventions + less shifting required)
/// * App([f, x]): f applied to x
/// * Lam([body]): lambda abstraction referred to through $i Vars
/// * Prim(Symbol): primitive (eg functions, constants, all nonvariable leaf nodes)
/// * Programs(Vec<Id>): list of root nodes of the programs. There's just one of these at the top of the program tree
/// 
/// Note there is no AppLam construct. This is because AppLams are represented through the `AppLam` struct when it comes
/// to invention-finding, and they don't belong in Lambda because they never actually show up within programs (theyre only
/// ever used in passing at the top level when constructing inventions) 
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Lambda {
    Var(i32), // db index ($i)
    IVar(i32), // db index used by inventions (#i)
    Prim(Symbol), // primitive (eg functions, constants, all nonvariable leaf nodes)
    App([Id; 2]), // f, x
    Lam([Id; 1]), // body
    Programs(Vec<Id>), // root node at the very top of the tree
}

/// An untyped lambda calculus expression, much like `egg::RecExpr` but with a public `nodes` field
/// and many attached functions. See `Lambda` for details on the individual nodes.
/// 
/// Creation:
/// * From<RecExpr> is implemented (and vis versa) for interop with `egg`
/// * Expr::new() directly constructs an Expr from a Vec<Lambda>
/// * Expr::prim(Symbol), Expr::app(Expr,Expr), etc let you construct Exprs from other Exprs
/// * Expr::from_curried(&str) parses from a curried string like (app (app + 3) 4)
/// * Expr::from_uncurried(&str) parses from an uncurried string like (+ 3 4)
/// 
/// Displaying an expression or subexpression:
/// * fmt::Display is implemented to return an uncurried string like (+ 3 4)
/// * Expr::to_curried_string(Option<Id>) returns a curried string like (app (app + 3) 4) rooted at the Id if given
/// * Expr::to_uncurried_string(Option<Id>) returns an uncurried string like (+ 3 4) rooted at the Id if given
/// * Expr::save() lets you save an image of the expr to a file
/// 
/// Creating a subexpression:
/// * Expr::cloned_subexpr(Id) returns the subexpression rooted at the Id. Generally you want to avoid this because
///   most methods can get by just fine by taking a parent Expr and a child Id without the need for all this cloning.
///   Importantly all Id indexing should be preserved just fine since this is implemented through truncating the underlying vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Expr {
    pub nodes: Vec<Lambda>, // just like in a RecExpr but public
}

/// printing a single node prints the operator - this is needed for `egg`.
/// If you want to print the whole expression, use `Expr` not `Lambda`.
impl Display for Lambda {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Var(i) => write!(f, "${}", i),
            Self::IVar(i) => write!(f, "#{}", i),
            Self::Prim(p) => write!(f,"{}",p),
            Self::App(_) => write!(f,"app"),
            Self::Lam(_) => write!(f,"lam"),
            Self::Programs(_) => write!(f,"programs"),
        }
    }
}

/// implement egg-compatability for Lambda
impl Language for Lambda {
    fn matches(&self, other: &Self) -> bool {
        // consider only operator, not children. I believe we do want to consider number of children in the `Programs` case based on the macro code.
        match (self,other) {
            (Self::Var(i), Self::Var(j)) => i == j,
            (Self::IVar(i), Self::IVar(j)) => i == j,
            (Self::Prim(p1), Self::Prim(p2)) => p1 == p2,
            (Self::App(_), Self::App(_)) => true,
            (Self::Lam(_), Self::Lam(_)) => true,
            (Self::Programs(p1), Self::Programs(p2)) => p1.len() == p2.len(),
            (_,_) => false,
        }
    }

    fn children(&self) -> &[Id] {
        match self {
            Self::App(ids) => ids,
            Self::Lam(ids) => ids,
            Self::Programs(ids) => ids,
            _ => &[],
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Self::App(ids) => ids,
            Self::Lam(ids) => ids,
            Self::Programs(ids) => ids,
            _ => &mut [],
        }
    }
}
impl FromOp for Lambda {
    type Error = String;

    /// Parse an e-node with operator `op` and children `children`.
    fn from_op(op: &str, children: Vec<Id>) -> Result<Self, Self::Error> {
        match op {
            "app" => {
                if children.len() != 2 {
                    return Err(format!("app needs 2 children, got {}", children.len()));
                }
                Ok(Self::App([children[0], children[1]]))
            },
            "lam" => {
                if children.len() != 1 {
                    return Err(format!("lam needs 1 child, got {}", children.len()));
                }
                Ok(Self::Lam([children[0]]))
            }
            "programs" => Ok(Self::Programs(children)),
            _ => {
                if !children.is_empty() {
                    return Err(format!("{} needs 0 children, got {}", op, children.len()))
                }
                if op.starts_with('$') {
                    let i = op.chars().skip(1).collect::<String>().parse::<i32>().unwrap();
                    Ok(Self::Var(i))
                } else if op.starts_with('#') {
                    let i = op.chars().skip(1).collect::<String>().parse::<i32>().unwrap();
                    Ok(Self::IVar(i))
                } else {
                    Ok(Self::Prim(egg::Symbol::from(op)))
                }
            },
        }
    }

}



/// Expr <-> RecExpr
impl From<RecExpr<Lambda>> for Expr {
    fn from(e: RecExpr<Lambda>) -> Self {
        // todo you could (and should) actually grab it recursively, this is just some unsafe cheating during experimenting
        let nodes: Vec<Lambda> = unsafe{ std::mem::transmute(e) };
        Expr::new(nodes)
    }
}
/// Expr <-> RecExpr
impl From<Expr> for RecExpr<Lambda> {
    fn from(e: Expr) -> Self {
        // todo you could (and should) actually grab it recursively, this is just some unsafe cheating during experimenting
        unsafe{ std::mem::transmute(e.nodes) }
    }
}
/// Expr <-> RecExpr
impl From<&Expr> for &RecExpr<Lambda> {
    fn from(e: &Expr) -> Self {
        // todo you could (and should) actually grab it recursively, this is just some unsafe cheating during experimenting
        let nodes: &Vec<Lambda> = &e.nodes;
        unsafe{ std::mem::transmute(nodes) }
    }
}

impl Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_uncurried(None))
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


impl Expr {
    /// Construct a new Expr
    pub fn new(nodes: Vec<Lambda>) -> Self {
        Self { nodes }
    }

    /// Returns the root
    pub fn root(&self) -> Id {
        Id::from(self.nodes.len()-1)
    }

    /// Returns the root
    pub fn get_root(&self) -> &Lambda {
        self.get(self.root())
    }

    /// Returns the root
    pub fn get(&self, child:Id) -> &Lambda {
        &self.nodes[usize::from(child)]
    }

    /// construct an Expr with a single Var node
    pub fn var(i: i32) -> Self {
        Self::new(vec![Lambda::Var(i)])
    }
    /// construct an Expr with a single IVar node
    pub fn ivar(i: i32) -> Self {
        Self::new(vec![Lambda::IVar(i)])
    }
    /// construct an Expr with a single Prim node
    pub fn prim(p: Symbol) -> Self {
        Self::new(vec![Lambda::Prim(p)])
    }
    /// construct an Expr with a toplevel App node
    pub fn app(f: Expr, mut x: Expr) -> Self {
        let mut nodes = f.nodes;
        let f_id = Id::from(nodes.len()-1);
        x.shift_nodes(nodes.len() as i32);
        nodes.extend(x.nodes);
        let x_id = Id::from(nodes.len()-1);
        nodes.push(Lambda::App([f_id, x_id]));
        Self::new(nodes)
    }
    /// construct an Expr with a toplevel Lam node
    pub fn lam(b: Expr) -> Self{
        let mut nodes = b.nodes.clone();
        let b_id = Id::from(b.nodes.len()-1);
        nodes.push(Lambda::Lam([b_id]));
        Self::new(nodes)
    }
    /// construct an Expr with a toplevel Programs node
    pub fn programs(programs: Vec<Expr>) -> Self {
        let mut nodes = vec![];
        let mut root_ids = vec![];
        for mut p in programs.into_iter() {
            p.shift_nodes(nodes.len() as i32);
            nodes.extend(p.nodes);
            root_ids.push(Id::from(nodes.len() - 1));
        }
        nodes.push(Lambda::Programs(root_ids));
        Self::new(nodes)
    }

    /// split a Programs node into a vector of the programs.
    /// (This does not consume `self` because you cant split a single Vec allocation
    /// into multiple (allocator restriction) so we need to make clones anyways)
    pub fn split_programs(&self) -> Vec<Expr> {
        match self.get_root() {
            Lambda::Programs(roots) => {
                // we know the separate programs are in non-overlapping contiguous
                // chunks so this is all safe
                let mut res: Vec<Expr> = vec![];
                let mut start: usize = 0;
                for root in roots.iter() {
                    let end = usize::from(*root)+1;
                    let mut e = Expr::new(self.nodes[start..end].to_vec());
                    e.shift_nodes(-(start as i32));
                    res.push(e);
                    start = end;
                }
                res
                // roots.iter().map(|root| self.to_string_uncurried(Some(*root)).parse().unwrap()).collect()
            },
            _ => unreachable!()
        }
    }

    /// helper fn to shift add the Ids by a certain amount
    pub fn shift_nodes(&mut self, shift: i32) {
        for node in &mut self.nodes {
            node.update_children(|id| Id::from((usize::from(id) as i32 + shift) as usize));
        }
    }

    /// returns expr depth as per `ProgramDepth`
    pub fn depth(&self) -> i32 {
        ProgramDepth{}.cost_rec(self.into())
    }
    /// returns expr cost as per `ProgramCost`
    pub fn cost(&self) -> i32 {
        ProgramCost{}.cost_rec(self.into())
    }

    /// returns expr length as per `ProgramLength`
    pub fn length(&self) -> i32 {
        ProgramLength{}.cost_rec(self.into())
    }

    pub fn executable<D: Domain>(&self) -> Executable<D> {
        self.clone().into()
    }

    /// Returns a subexpression cloned out of this one with new root `child`.
    /// Generally you want to avoid this because
    /// most methods can get by just fine by taking a parent Expr and a child Id without the need for all this cloning.
    /// Importantly all Id indexing should be preserved just fine since this is implemented through truncating the underlying vector.
    pub fn cloned_subexpr(&self, child:Id) -> Self {
        assert!(self.nodes.len() > child.into());
        Self::new(self.nodes.iter().take(usize::from(child)+1).cloned().collect())
    }
    /// Consumes an expr and returns a subexpr.
    /// Importantly all Id indexing should be preserved just fine since this is implemented through truncating the underlying vector.
    pub fn into_subexpr(mut self, child:Id) -> Self {
        assert!(self.nodes.len() > child.into());
        self.nodes.truncate(usize::from(child)+1);
        self
    }

    /// Go from a curried string to an Expr
    /// Uncurried: (foo x y)
    /// Curried: (app (app foo x) y)
    pub fn from_curried(s: &str) -> Result<Self,String> {
        let recexpr: RecExpr<Lambda> = s.parse().map_err(|e|format!("{:?}",e))?;
        Ok(recexpr.into())
    }

    /// Go from an uncurried string to an Expr
    /// Uncurried: (foo x y)
    /// Curried: (app (app foo x) y)
    pub fn from_uncurried(s: &str) -> Result<Self,String> {
        let mut sexpr: Sexp = sexp::parse(s).map_err(|e|e.to_string())?;
        sexpr = curry_sexp(&sexpr);
        Self::from_curried(&sexpr.to_string())
    }

    /// Print Expr as a curried string
    /// Uncurried: (foo x y)
    /// Curried: (app (app foo x) y)
    pub fn to_string_curried(&self, child: Option<Id>) -> String {
        let expr = match child {
            None => self.clone(),
            Some(id) => self.cloned_subexpr(id)
        };
        expr.to_sexp(self.root()).to_string()
    }

    /// Print Expr as an uncurried string
    /// Uncurried: (foo x y)
    /// Curried: (app (app foo x) y)
    pub fn to_string_uncurried(&self, child:Option<Id>) -> String {
        uncurry_sexp(&self.to_sexp(child.unwrap_or_else(|| self.root()))).to_string()
    }

    /// convert to an s expression. Useful for printing / parsing purposes
    pub fn to_sexp(&self, child: Id) -> Sexp {
        let node = &self.nodes[usize::from(child)];
        match node {
            Lambda::Var(_) | Lambda::IVar(_) | Lambda::Prim(_) => sexp::parse(&node.to_string()).unwrap(),
            Lambda::App([f,x]) => {
                let f = self.to_sexp(*f);
                let x = self.to_sexp(*x);
                let app = Sexp::Atom(sexp::Atom::S("app".to_string()));
                Sexp::List(vec![app,f,x])
            },
            Lambda::Lam([b]) => {
                let b = self.to_sexp(*b);
                let lam = Sexp::Atom(sexp::Atom::S("lam".to_string()));
                Sexp::List(vec![lam,b])
            },
            Lambda::Programs(root_ids) => {
                let mut res = vec![Sexp::Atom(sexp::Atom::S("programs".to_string()))];
                root_ids.iter().for_each(|id| res.push(self.to_sexp(*id)));
                Sexp::List(res)
            }
        }
    }

    /// write the Expr to a file (includes structural hashing sharing)
    /// writes to `outdir/name.png` (no need to provide the extension)
    pub fn save(&self, name: &str, outdir: &str) {
        let mut egraph: EGraph = Default::default();
        egraph.add_expr(self.into());
        egraph.dot().to_png(format!("{}/{}.png",outdir,name)).unwrap();
    }
}