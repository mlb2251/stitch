use crate::expr::*;

use std::fmt::{Debug};
use std::hash::Hash;
use std::cell::RefCell;
use std::time::{Instant,Duration};
use serde::{Serialize, Deserialize};

/// env[i] is the value at $i
pub type Env<D> = Vec<LazyVal<D>>;

/// a value can either be some domain specific value Dom(D) like an Int,
/// or it can be a primitive function or partially applied primitive function like + or (+ 2)
/// or it can be a lambda function with some captured env like (lam (* $1 $0)) where $1 may have been captured from
/// the surrounding code and this whole object may now be passed around
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Val<D: Domain> {
    Dom(D),
    PrimFun(CurriedFn<D>), // function ptr, arity, any args that have been partially filled in
    LamClosure(Id, Env<D>) // body, captured env
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LazyValSource<D: Domain> {
    Lazy(Id, Env<D>),
    Strict(Val<D>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LazyVal<D: Domain> {
    pub val: Option<Val<D>>,
    pub source: LazyValSource<D>
}

impl<D: Domain> LazyVal<D> {
    pub fn new_lazy(child: Id, env: Env<D>) -> Self {
        LazyVal {
            val: None,
            source: LazyValSource::Lazy(child, env)
        }
    }
    pub fn new_strict(val: Val<D>) -> Self {
        LazyVal {
            val: None,
            source: LazyValSource::Strict(val)
        }
    }
    pub fn eval(&mut self, handle: &Evaluator<D>) -> VResult<D> {
        if self.val.is_none() {
            match &mut self.source {
                LazyValSource::Lazy(child, env) => {
                    self.val = Some(handle.eval_child(*child, env.as_mut_slice())?)
                }
                LazyValSource::Strict(val) => {
                    self.val = Some(val.clone());
                }
            }
        }
        Ok(self.val.clone().unwrap())
    }
}

pub type VResult<D> = Result<Val<D>,VError>;
pub type VError = String;


#[derive(Debug)]
pub struct Evaluator<'a, D: Domain> {
    pub expr: &'a Expr,
    pub data: RefCell<D::Data>,
    pub start_and_timelimit: Option<(Instant, Duration)>,
}

impl Expr {
    pub fn eval<D:Domain>(&self, child: Id, env: &mut [LazyVal<D>], timelimit: Option<Duration>) -> VResult<D> {
        self.as_eval(timelimit).eval_child(child, env)
    }
    pub fn as_eval<D:Domain>(&self, timelimit: Option<Duration>) -> Evaluator<D> {
        let start_and_timelimit = timelimit.map(|d| (Instant::now(),d));
        Evaluator {
            expr: self,
            data: Default::default(),
            start_and_timelimit
        }
    }
}


/// Wraps a DSL function in a struct that manages currying of the arguments
/// which are fed in one at a time through .apply(). Example: the "+" primitive
/// evaluates to a CurriedFn with arity 2 and empty partial_args. The expression
/// (app + 3) evals to a CurriedFn with vec![3] as the partial_args. The expression
/// (app (app + 3) 4) will evaluate to 7 (since .apply() will fill the last argument,
/// notice that all arguments are filled, and return the result).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CurriedFn<D: Domain> {
    name: egg::Symbol,
    arity: usize,
    partial_args: Env<D>,
}

impl<D: Domain> CurriedFn<D> {
    pub fn new(name: egg::Symbol, arity: usize) -> Self {
        Self {
            name,
            arity,
            partial_args: Vec::new(),
        }
    }
    pub fn new_with_args(name: egg::Symbol, arity: usize, partial_args: Env<D>) -> Self {
        Self {
            name,
            arity,
            partial_args,
        }
    }
    /// Feed one more argument into the function, returning a new CurriedFn if
    /// still not all the arguments have been received. Evaluate the function
    /// if all arguments have been received. Does not mutate the original.
    pub fn apply(&self, arg: LazyVal<D>, handle: &Evaluator<D>) -> VResult<D> {
        let mut new_dslfn = self.clone();
        new_dslfn.partial_args.push(arg);
        if new_dslfn.partial_args.len() == new_dslfn.arity {
            (D::lookup_fn_ptr(new_dslfn.name)) (new_dslfn.partial_args, handle)
        } else {
            Ok(Val::PrimFun(new_dslfn))
        }
    }
}

impl<D: Domain> Val<D> {
    pub fn dom(self) -> Result<D,VError> {
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

pub trait FromVal<D: Domain>: Sized {
    fn from_val(val: Val<D>) -> Result<Self,VError>;
}

impl<D: Domain> FromVal<D> for Val<D> {
    fn from_val(val: Val<D>) -> Result<Self,VError> {
        Ok(val)
    }
}


impl<'a, D: Domain> Evaluator<'a,D> {
    /// apply a function (Val) to an argument (Val)
    pub fn apply(&self, f: &Val<D>, x: Val<D>) -> VResult<D> {
        self.apply_lazy(f, LazyVal::new_strict(x))
    }
    // apply a function (Val) to an argument (LazyVal)
    pub fn apply_lazy(&self, f: &Val<D>, x: LazyVal<D>) -> VResult<D> {
        match f {
            Val::PrimFun(f) => f.apply(x, self),
            Val::LamClosure(f, env) => {
                let mut new_env = vec![x];
                new_env.extend(env.iter().cloned());
                self.eval_child(*f, &mut new_env)
            }
            _ => Err("Expected function or closure".into()),
        }
    }

    pub fn set_timeout(&mut self, timeout: Duration) {
        self.start_and_timelimit = Some((Instant::now(), timeout))
    }

    /// eval a subexpression in an environment
    pub fn eval_child(&self, child: Id, env: &mut [LazyVal<D>]) -> VResult<D> {
        if let Some((start_time, duration)) = &self.start_and_timelimit {
            if start_time.elapsed() >= *duration {
                return Err(format!("Eval Timeout"));
            }
        }
        let val = match self.expr.nodes[usize::from(child)] {
            Lambda::Var(i) => {
                env[i as usize].eval(self)?
            }
            Lambda::IVar(_) => {
                panic!("attempting to execute a #i ivar")
            }
            Lambda::App([f,x]) => {
                let f_val = self.eval_child(f, env)?;
                let x_val = LazyVal::new_lazy(x, env.to_vec());
                self.apply_lazy(&f_val, x_val)?
            }
            Lambda::Prim(p) => {
                match D::val_of_prim(p) {
                    Some(v) => v,
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
        Ok(val)
    }
}
