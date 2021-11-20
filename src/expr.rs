use egg::*;
use std::fmt::{self, Formatter, Display};
use std::collections::HashMap;


#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Lambda {
    Var(i32), // db index ($i)
    IVar(i32), // db index used by inventions (#i)
    App([Id; 2]), // f, x
    Lam([Id; 1]), // body
    Prim(egg::Symbol), // primitive (eg functions, constants, all nonvariable leaf nodes)
    Programs(Vec<Id>), // root node at the very top of the tree
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub nodes: Vec<Lambda>, // just like in a RecExpr but public
}

#[derive(Clone)]
pub struct CurriedFn<D: DomainVal> {
    name: egg::Symbol, // name included really just for debugging/clarity
    func: fn(&[Val<D>]) -> Val<D>,
    arity: usize,
    partial_args: Vec<Val<D>>,
}
impl<D: DomainVal> fmt::Debug for CurriedFn<D> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "CurriedFn(name={:?}, arity={:?}, partial_args={:?})", self.name, self.arity, self.partial_args)
    }
}

impl<D: DomainVal> CurriedFn<D> {
    pub fn new(name: egg::Symbol, func: fn(&[Val<D>]) -> Val<D>, arity: usize) -> Self {
        Self {
            name,
            func,
            arity,
            partial_args: Vec::new(),
        }
    }
    pub fn apply(&self, arg: &Val<D>) -> Val<D> {
        let mut new_dslfn = self.clone();
        new_dslfn.partial_args.push(arg.clone());
        if new_dslfn.partial_args.len() == new_dslfn.arity {
            (new_dslfn.func)(&new_dslfn.partial_args)
        } else {
            Val::PrimFun(new_dslfn)
        }
    }
}

#[derive(Clone, Debug)]
pub enum Val<D: DomainVal> {
    Domain(D),
    PrimFun(CurriedFn<D>), // function ptr, arity, any args that have been partially filled in
    LamClosure(Id, Vec<Val<D>>) // body, captured env
}

/// I'm sure this will come in handy
pub trait DomainVal: Clone + fmt::Debug {

}

// todo theres def a better way to do this that doesnt have fallback as a functin pointer. Like
// todo somehow it should be part of the DomainVal trait. Except how can that trait carry around a big
// todo hashmap? Then again the fallback will be called almost never so optimizing for prims() makes
// a fair bit of sense. Hmm I think lazy_static!{} would let you init a hashmap so that you can ref it
// from fallback() and just not have prims at all.
// hmm ya thatd be nice and then you could use D::get_val(sym) or something
pub struct DSL<D: DomainVal> {
    pub prims: HashMap<egg::Symbol, Val<D>>,
    pub fallback: fn(&egg::Symbol) -> Option<Val<D>>,
}

impl<D: DomainVal> DSL<D> {
    pub fn new(prims: HashMap<egg::Symbol, Val<D>>, fallback: fn(&egg::Symbol) -> Option<Val<D>>) -> Self {
        Self { prims, fallback }
    }
    pub fn get(&self, sym: &egg::Symbol) -> Option<Val<D>> {
        self.prims.get(&sym).cloned().or_else(|| (self.fallback)(sym))
    }
}

impl Expr {

    pub fn root(&self) -> Id {
        Id::from(self.nodes.len()-1)
    }

    pub fn apply<D>(&self, f: &Val<D>, x: &Val<D>, dsl: &DSL<D>) -> Val<D>
    where
        D: DomainVal,
    {
        match f {
            Val::PrimFun(f) => f.apply(x),
            Val::LamClosure(f, env) => {
                let mut env = env.clone();
                env.push(x.clone());
                self.eval_child(*f, &env, dsl)
            }
            _ => panic!("Expected function or closure"),
        }
    }
    pub fn eval<D: DomainVal>(
        &self,
        env: &[Val<D>],
        dsl: &DSL<D>,
    ) -> Val<D>
    {
        self.eval_child(self.root(), env, dsl)
    }

    pub fn eval_child<D>(
        &self,
        child: Id,
        env: &[Val<D>],
        dsl: &DSL<D>,
    ) -> Val<D>
    where
        D: DomainVal,
    {
        match &self.nodes[usize::from(child)] {
            Lambda::Var(i) => {
                env[*i as usize].clone()
            }
            Lambda::IVar(i) => {
                panic!("attempting to execute a #i ivar")
            }
            Lambda::App([f,x]) => {
                let f_val = self.eval_child(*f, env, dsl);
                let x_val = self.eval_child(*x, env, dsl);
                self.apply(&f_val, &x_val, dsl)
            }
            Lambda::Prim(p) => {
                match dsl.get(p) {
                    Some(v) => v.clone(),
                    None => panic!("Prim {} not found",p),
                }
            }
            Lambda::Lam([b]) => {
                Val::LamClosure(*b, env.to_vec())
            }
            Lambda::Programs(_) => {
                panic!("not implemented")
            }
        }

    }
}



type RecExpr = egg::RecExpr<Lambda>;

impl Display for Lambda {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Var(i) => write!(f, "${}", i),
            Self::IVar(i) => write!(f, "#{}", i),
            Self::App(_) => write!(f,"app"),
            Self::Lam(_) => write!(f,"lam"),
            Self::Prim(p) => write!(f,"{}",p),
            Self::Programs(_) => write!(f,"programs"),
        }
    }
}

impl Language for Lambda {
    fn matches(&self, other: &Self) -> bool {
        // consider only operator, not children. I believe (?) we do want to consider number of children based on the macro code.
        match (self,other) {
            (Self::Var(i), Self::Var(j)) => i == j,
            (Self::IVar(i), Self::IVar(j)) => i == j,
            (Self::App(_), Self::App(_)) => true,
            (Self::Lam(_), Self::Lam(_)) => true,
            (Self::Prim(p1), Self::Prim(p2)) => p1 == p2,
            (Self::Programs(p1), Self::Programs(p2)) => p1.len() == p2.len(),
            (_,_) => false,
        }
    }

    fn children(&self) -> &[Id] {
        match self {
            Self::Lam(ids) => ids,
            Self::App(ids) => ids,
            Self::Programs(ids) => ids,
            _ => &[],
        }
    }

    fn children_mut(&mut self) -> &mut [Id] {
        match self {
            Self::Lam(ids) => ids,
            Self::App(ids) => ids,
            Self::Programs(ids) => ids,
            _ => &mut [],
        }
    }

    fn display_op(&self) -> &dyn Display {
        unimplemented!("use Expr() not RecExpr for printing. from::() is implemented. This is because egg 0.6.0 hasnt fixed issue #83 so displaying things like $5 is not valid")
    }

    fn from_op_str(op_str: &str, children: Vec<Id>) -> Result<Self, String> {
        match op_str {
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
                if children.len() != 0 {
                    return Err(format!("{} needs 0 children, got {}", op_str, children.len()))
                }
                if op_str.starts_with("$") {
                    let i = op_str.chars().skip(1).collect::<String>().parse::<i32>().unwrap();
                    Ok(Self::Var(i))
                } else if op_str.starts_with("#") {
                    let i = op_str.chars().skip(1).collect::<String>().parse::<i32>().unwrap();
                    Ok(Self::IVar(i))
                } else {
                    Ok(Self::Prim(egg::Symbol::from(op_str)))
                }
            },
        }
    }
}

impl From<RecExpr> for Expr {
    fn from(e: RecExpr) -> Self {
        // todo you could (and should) actually grab it recursively, this is just some unsafe cheating during experimenting
        unsafe{ std::mem::transmute(e) }
    }
}
impl From<Expr> for RecExpr {
    fn from(e: Expr) -> Self {
        // todo you could (and should) actually grab it recursively, this is just some unsafe cheating during experimenting
        unsafe{ std::mem::transmute(e) }
    }
}
impl From<&Expr> for &RecExpr {
    fn from(e: &Expr) -> Self {
        // todo you could (and should) actually grab it recursively, this is just some unsafe cheating during experimenting
        unsafe{ std::mem::transmute(e) }
    }
}

impl std::str::FromStr for Expr {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let e: RecExpr = s.parse()?;
        Ok(e.into())
    }
}

impl Expr {
    pub fn new(nodes: Vec<Lambda>) -> Self {
        Self { nodes: nodes}
    }
    pub fn var(i: i32) -> Self {
        Self { nodes: vec![Lambda::Var(i)] }
    }
    pub fn ivar(i: i32) -> Self {
        Self { nodes: vec![Lambda::IVar(i)] }
    }
    pub fn prim(p: egg::Symbol) -> Self {
        Self { nodes: vec![Lambda::Prim(p)] }
    }
    pub fn app(f: Expr, mut x: Expr) -> Self {
        let mut nodes = f.nodes;
        let f_id = Id::from(nodes.len()-1);
        x.shift_nodes(nodes.len());
        nodes.extend(x.nodes);
        let x_id = Id::from(nodes.len()-1);
        nodes.push(Lambda::App([f_id, x_id]));
        Self::new(nodes)
    }
    pub fn lam(b: Expr) -> Self{
        let mut nodes = b.nodes.clone();
        let b_id = Id::from(b.nodes.len()-1);
        nodes.push(Lambda::Lam([b_id]));
        Self::new(nodes)
    }
    pub fn programs(programs: Vec<Expr>) -> Self {
        let mut nodes = vec![];
        let mut root_ids = vec![];
        for mut p in programs.into_iter() {
            p.shift_nodes(nodes.len());
            nodes.extend(p.nodes);
            root_ids.push(Id::from(nodes.len() - 1));
        }
        nodes.push(Lambda::Programs(root_ids));
        Self::new(nodes)
    }

    fn shift_nodes(&mut self, shift: usize) {
        for node in &mut self.nodes {
            node.update_children(|id| Id::from(usize::from(id) + shift))
        }
    }

    fn write_child(&self, child:Id, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let node = &self.nodes[usize::from(child)];
        if node.is_leaf() {
            write!(f, "{}", node)?;
        } else {
            write!(f,"(")?;
            write!(f, "{}", node)?;
            for child in node.children() {
                write!(f," ")?;
                self.write_child(*child,f)?;
            };
            write!(f,")")?;
        }
        Ok(())
    }
}


impl Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.nodes.is_empty() {
            write!(f, "()")
        } else {
            self.write_child(self.root(), f)
        }
    }
}
// impl std::ops::Index<Id> for Expr {
//     type Output = Lambda;
//     fn index(&self, id: Id) -> &Self::Output {
//         self.nodes[usize::from(id)]
//             .as_ref()
//             .unwrap_or_else(|| panic!("Invalid id {}", id))
//     }
// }

// impl std::ops::IndexMut<Id> for Expr {
//     fn index_mut(&mut self, id: Id) -> &mut Self::Output {
//         let id = self.find(id);
//         self.nodes[usize::from(id)]
//             .as_mut()
//             .unwrap_or_else(|| panic!("Invalid id {}", id))
//     }
// }