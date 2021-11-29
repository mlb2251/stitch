use super::expr::*;
use std::collections::HashMap;
use egg::*;
use std::fmt::{self, Formatter, Display, Debug};
use std::hash::Hash;
use std::cell::RefCell;

pub type VResult<D> = Result<Val<D>,VError>;
pub type VError = String;

/// this macros defines two lazy static variables PRIMS
/// and FUNCS 
#[macro_export]
macro_rules! define_semantics {
    (   type Val = $val_type:ty;
        type DSLFn = $dsl_fn_type:ty;
        $($string:literal = ($fname:ident,$arity:literal) ),*
    ) => { 
        lazy_static::lazy_static! {
        static ref PRIMS: HashMap<egg::Symbol, $val_type> = vec![
            $(($string.into(), PrimFun(CurriedFn::new($string.into(), $arity)))),*
            ].into_iter().collect();
        
        static ref FUNCS: HashMap<egg::Symbol, $dsl_fn_type> = vec![
            $(($string.into(), $fname as $dsl_fn_type)),*
        ].into_iter().collect();
        }
    }
}


#[derive(Debug, Clone)]
pub struct DomExpr<D: Domain> {
    pub expr: Expr,
    pub evals: RefCell<HashMap<(Id,Vec<Val<D>>), Val<D>>>, // from (node,env) to result
    pub data: RefCell<D::Data>,
}

impl<D: Domain> From<Expr> for DomExpr<D> {
    fn from(expr: Expr) -> Self {
        DomExpr {
            expr,
            evals: HashMap::new().into(),
            data: D::Data::default().into(),
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
    pub fn apply(&self, arg: Val<D>, handle: &DomExpr<D>) -> VResult<D> {
        let mut new_dslfn = self.clone();
        new_dslfn.partial_args.push(arg);
        if new_dslfn.partial_args.len() == new_dslfn.arity {
            D::fn_of_prim(new_dslfn.name) (new_dslfn.partial_args, handle)
        } else {
            Ok(Val::PrimFun(new_dslfn))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Val<D: Domain> {
    Dom(D),
    PrimFun(CurriedFn<D>), // function ptr, arity, any args that have been partially filled in
    LamClosure(Id, Vec<Val<D>>) // body, captured env
}

pub enum Type<D: Domain> {
    Dom(D::DomType),
    Fun(Vec<Type<D>>,Box<Type<D>>),
}

impl<D: Domain> Val<D> {
    pub fn unwrap_dom(self) -> Result<D,VError> {
        match self {
            Val::Dom(d) => Ok(d),
            _ => Err("Val::unwrap_dom: not a domain value".into())
        }
    }
}
impl<D: Domain> From<D> for Val<D> {
    fn from(d: D) -> Self {
        Val::Dom(d)
    }
}



/// The key trait that defines a domain
pub trait Domain: Clone + Debug + PartialEq + Eq + Hash {
    /// Domain::Data is attached to the DomExpr so all DSL functions will have a
    /// mut ref to it (through the handle argument). Feel free to make it the empty
    /// tuple if you dont need it.
    /// Motivation: For some complicated domains you could leave Ids as pointers and
    /// have your domaindata be a system to lookup the actual value from the pointer
    /// (and guarantee no value has multiple pointers so that comparison works by Ids).
    /// Btw, I briefly implemented it so that the whole system worked by these pointers
    /// and it was absolutely miserable, see my notes. But this is here if someone finds
    /// a use for it. Ofc be careful not to break function purity with this but otherwise
    /// be creative :)
    type Data: Debug + Default;
    /// Domain::DomType is the type of Domain values.
    type DomType; // todo not yet used for anything
    /// given a primitive's symbol return a runtime Val object. For function primitives
    /// this should return a PrimFun(CurriedFn) object.
    fn val_of_prim(_p: egg::Symbol) -> Option<Val<Self>> {
        unimplemented!()
    }
    /// given a function primitive's symbol return the function pointer
    /// you can use to call the function.
    /// Breakdown of the function type: it takes a slice of values as input (the args)
    /// along with a mut ref to an Expr (I'll refer to as a "handle") which is necessary
    /// for calling .apply(f,x). This setup with a handle guarantees we can always track
    /// when applys happen and log them in our Expr.evals, and also it's necessary for
    /// executing LamClosures in order to look up their body Id (and we wouldn't want
    /// LamClosures to carry around full Exprs because that would break the Expr.evals tracking)
    fn fn_of_prim(_p: egg::Symbol) -> fn(Vec<Val<Self>>, &DomExpr<Self>) -> Result<Val<Self>,String> {
        unimplemented!()
    }
    // fn type_of_dom_val(_v: &Self) -> Type<Self> {
    //     unimplemented!()
    // }
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
        self.evals.borrow().iter()
            .filter(|((id,_env),_res)| *id == node)
            .map(|((_id,env),res)| (env.clone(),(*res).clone()))
            .collect()
    }

    /// apply a function (Val) to an argument (Val)
    pub fn apply(&self, f: Val<D>, x: Val<D>) -> VResult<D> {
        match f {
            Val::PrimFun(f) => f.apply(x.clone(), self),
            Val::LamClosure(f, env) => {
                let mut new_env = vec![x.clone()];
                new_env.extend(env.iter().cloned());
                self.eval_child(f, &new_env)
            }
            _ => panic!("Expected function or closure"),
        }
    }
    /// eval the Expr in an environment
    pub fn eval(
        &self,
        env: &[Val<D>],
    ) -> VResult<D> {
        self.eval_child(self.expr.root(), env)
    }

    /// eval a subexpression in an environment
    pub fn eval_child(
        &self,
        child: Id,
        env: &[Val<D>],
    ) -> VResult<D> {
        let key = (child, env.to_vec());
        if let Some(val) = self.evals.borrow().get(&key).cloned() {
            return Ok(val);
        }
        let val = match self.expr.nodes[usize::from(child)] {
            Lambda::Var(i) => {
                env[i as usize].clone()
            }
            Lambda::IVar(_) => {
                panic!("attempting to execute a #i ivar")
            }
            Lambda::App([f,x]) => {
                let f_val = self.eval_child(f, env)?;
                let x_val = self.eval_child(x, env)?;
                self.apply(f_val, x_val)?
            }
            Lambda::Prim(p) => {
                match D::val_of_prim(p) {
                    Some(v) => v.clone(),
                    None => panic!("Prim `{}` not found",p),
                }
            }
            Lambda::Lam([b]) => {
                Val::LamClosure(b, env.to_vec())
            }
            Lambda::Programs(_) => {
                panic!("todo implement")
            }
        };
        self.evals.borrow_mut().insert(key, val.clone());
        Ok(val)
    }
}