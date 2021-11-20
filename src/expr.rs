use egg::*;
use std::fmt::{self, Formatter, Display, Debug};
use std::collections::HashMap;
use std::hash::Hash;


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

#[derive(Debug, Clone)]
pub struct DomExpr<D: Domain> {
    pub expr: Expr,
    pub evals: HashMap<(Id,Vec<Val<D>>), Val<D>>, // from (node,env) to result
}

impl<D: Domain> From<Expr> for DomExpr<D> {
    fn from(expr: Expr) -> Self {
        DomExpr {
            expr,
            evals: HashMap::new(),
        }
    }
}

impl<D: Domain> Display for DomExpr<D> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

/// Wraps a DSL function in a struct that manages currying of the arguments
/// which are fed in one at a time through .apply(). Example: the "+" primitive
/// evaluates to a CurriedFn with arity 2 and empty partial_args. The expression
/// (app + 3) evals to a CurriedFn with vec![3] as the partial_args. The expression
/// (app (app + 3) 4) will evaluate to 7 (since .apply() will fill the last argument,
/// notice that all arguments are filled, and return the result).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CurriedFn<D: Domain> {
    name: egg::Symbol,
    arity: usize,
    partial_args: Vec<Val<D>>,
}

impl<D: Domain> CurriedFn<D> {
    pub fn new(name: egg::Symbol, arity: usize) -> Self {
        Self {
            name,
            arity,
            partial_args: Vec::new(),
        }
    }
    /// Feed one more argument into the function, returning a new CurriedFn if
    /// still not all the arguments have been received. Evaluate the function
    /// if all arguments have been received.
    pub fn apply(&self, arg: &Val<D>, handle: &mut DomExpr<D>) -> Val<D> {
        let mut new_dslfn = self.clone();
        new_dslfn.partial_args.push(arg.clone());
        if new_dslfn.partial_args.len() == new_dslfn.arity {
            D::fn_of_prim(new_dslfn.name) (&new_dslfn.partial_args, handle)
        } else {
            Val::PrimFun(new_dslfn)
        }
    }
}


#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Val<D: Domain> {
    Dom(D),
    PrimFun(CurriedFn<D>), // function ptr, arity, any args that have been partially filled in
    LamClosure(Id, Vec<Val<D>>) // body, captured env
}

impl<D: Domain> Val<D> {
    pub fn unwrap_dom(&self) -> D {
        match self {
            Val::Dom(d) => d.clone(),
            _ => panic!("Val::unwrap_dom: not a domain value")
        }
    }
}
impl<D: Domain> From<D> for Val<D> {
    fn from(d: D) -> Self {
        Val::Dom(d)
    }
}


/// todo unsure if we should require PartialEq Eq Hash
/// in particular Eq is the big one since f64 doesnt implement it. But in order
/// to use HashMaps (or even search for elements in Vecs) we need it. I think
/// it's fair to require this since itll let us do a LOT more generic algorithms that
/// work with the domain semantics. And if someone really wants f64s and know that
/// they wont be NaN they can make a f64_noNaN type with From<f64> that panics on NaN
/// and impl Eq for that, then whenever constructing a value variant that uses floats
/// just do a .into() on it. So they can use f64s within their logic and just need to
/// wrap them when passing things back out to our system.
pub trait Domain: Clone + Debug + PartialEq + Eq + Hash {
    fn val_of_prim(_p: egg::Symbol) -> Option<Val<Self>> {
        unimplemented!()
    }
    fn fn_of_prim(_p: egg::Symbol) -> fn(&[Val<Self>], &mut DomExpr<Self>) -> Val<Self> {
        unimplemented!()
    }
}

impl<D: Domain> DomExpr<D> {
    /// pretty prints the env->result pairs organized by node
    pub fn pretty_evals(&self) -> String {
        let mut s = format!("Evals for {}:",self);
        for id in 1..self.expr.nodes.len() {
            let id = Id::from(id);
            s.push_str(&format!(
                "\n\t{}:\n\t\t{}",
                self.expr.to_string_child(id),
                self.evals_of_node(id).iter().map(|(env,res)| format!("{:?} -> {:?}",env,res)).collect::<Vec<_>>().join("\n\t\t")
            ))
        }
        s
    }

    /// gets vec of (env,result) pairs for all the envs this node has been evaluated under
    pub fn evals_of_node(&self, node: Id) -> Vec<(Vec<Val<D>>,Val<D>)> {
        let flat: Vec<(&(Id,_),_)> = self.evals.iter().collect();
        flat.iter()
            .filter(|((id,_env),_res)| *id == node).map(|((_id,env),res)| (env.clone(),(*res).clone())).collect()
    }

    /// apply a function (Val) to an argument (Val)
    pub fn apply(&mut self, f: &Val<D>, x: &Val<D>) -> Val<D> {
        match f {
            Val::PrimFun(f) => f.apply(x, self),
            Val::LamClosure(f, env) => {
                let mut new_env = vec![x.clone()];
                new_env.extend(env.iter().cloned());
                self.eval_child(*f, &new_env)
            }
            _ => panic!("Expected function or closure"),
        }
    }
    pub fn eval(
        &mut self,
        env: &[Val<D>],
    ) -> Val<D> {
        self.eval_child(self.expr.root(), env)
    }

    pub fn eval_child(
        &mut self,
        child: Id,
        env: &[Val<D>],
    ) -> Val<D> {
        let key = (child, env.to_vec());
        if let Some(val) = self.evals.get(&key).cloned() {
            return val;
        }
        let val = match self.expr.nodes[usize::from(child)] {
            Lambda::Var(i) => {
                env[i as usize].clone()
            }
            Lambda::IVar(i) => {
                panic!("attempting to execute a #i ivar")
            }
            Lambda::App([f,x]) => {
                let f_val = self.eval_child(f, env);
                let x_val = self.eval_child(x, env);
                self.apply(&f_val, &x_val)
            }
            Lambda::Prim(p) => {
                match D::val_of_prim(p) {
                    Some(v) => v.clone(),
                    None => panic!("Prim {} not found",p),
                }
            }
            Lambda::Lam([b]) => {
                Val::LamClosure(b, env.to_vec())
            }
            Lambda::Programs(_) => {
                panic!("not implemented")
            }
        };
        self.evals.insert(key, val.clone());
        val
    }

    // pub fn get_evals(&self) -> &HashMap<(Id,Vec<Val<D>>), Val<D>> {
    //     &self.evals
    // }
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
        let nodes: Vec<Lambda> = unsafe{ std::mem::transmute(e) };
        Expr::new(nodes)
    }
}
impl From<Expr> for RecExpr {
    fn from(e: Expr) -> Self {
        // todo you could (and should) actually grab it recursively, this is just some unsafe cheating during experimenting
        unsafe{ std::mem::transmute(e.nodes) }
    }
}
impl From<&Expr> for &RecExpr {
    fn from(e: &Expr) -> Self {
        // todo you could (and should) actually grab it recursively, this is just some unsafe cheating during experimenting
        let nodes: &Vec<Lambda> = &e.nodes;
        unsafe{ std::mem::transmute(nodes) }
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
        Self { nodes: nodes }
    }
    pub fn root(&self) -> Id {
        Id::from(self.nodes.len()-1)
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

    fn write_child(&self, child:Id, f: &mut impl std::fmt::Write) -> fmt::Result {
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
    fn to_string_child(&self, child:Id) -> String {
        let mut s = String::new();
        // write!(&mut s, "hello");
        self.write_child(child, &mut s).unwrap();
        s
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