use crate::*;
use std::collections::HashMap;
use std::fmt::{self, Formatter, Display, Debug};
use std::hash::Hash;
use std::cell::RefCell;
use serde::{Serialize, Deserialize};


#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Val<D: Domain> {
    Dom(D),
    PrimFun(CurriedFn<D>), // function ptr, arity, any args that have been partially filled in
    LamClosure(Id, Vec<LazyVal<D>>) // body, captured env
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LazyValSource<D: Domain> {
    Lazy(Id, Vec<LazyVal<D>>),
    Strict(Val<D>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LazyVal<D: Domain> {
    pub val: Option<Val<D>>,
    pub source: LazyValSource<D>
}

impl<D: Domain> LazyVal<D> {
    pub fn new_lazy(child: Id, env: Vec<LazyVal<D>>) -> Self {
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
    pub fn eval(&mut self, handle: &Executable<D>) -> VResult<D> {
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
pub type DSLFn<D> = fn(Vec<LazyVal<D>>, &Executable<D>) -> VResult<D>;

pub enum TrustLevel {
    MayLoopMayPanic,
    WontLoopMayPanic,
    WontLoopWontPanic,
}

#[derive(Debug, Clone)]
pub struct Executable<D: Domain> {
    pub expr: Expr,
    pub evals: RefCell<HashMap<(Id,Vec<LazyVal<D>>), Val<D>>>, // from (node,env) to result
    pub data: RefCell<D::Data>,
}

impl<D: Domain> From<Expr> for Executable<D> {
    fn from(expr: Expr) -> Self {
        Executable {
            expr,
            evals: HashMap::new().into(),
            data: D::Data::default().into(),
        }
    }
}

impl<D:Domain> Display for Executable<D> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.expr)
    }
}

impl<D: Domain> std::str::FromStr for Executable<D> {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let expr: Expr = s.parse()?;
        Ok(expr.into())
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
    partial_args: Vec<LazyVal<D>>,
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
    /// if all arguments have been received. Does not mutate the original.
    pub fn apply(&self, arg: LazyVal<D>, handle: &Executable<D>) -> VResult<D> {
        let mut new_dslfn = self.clone();
        new_dslfn.partial_args.push(arg);
        if new_dslfn.partial_args.len() == new_dslfn.arity {
            D::fn_of_prim(new_dslfn.name).unwrap() (new_dslfn.partial_args, handle)
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



/// The key trait that defines a domain
pub trait Domain: Clone + Debug + PartialEq + Eq + Hash {
    /// Domain::Data is attached to the Executable so all DSL functions will have a
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

    const TRUST_LEVEL: TrustLevel;

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
    fn fn_of_prim(_p: egg::Symbol) -> Option<DSLFn<Self>> {
        unimplemented!()
    }
    // fn type_of_dom_val(_v: &Self) -> Type<Self> {
    //     unimplemented!()
    // }
}

impl<D: Domain> Executable<D> {
    /// pretty prints the env->result pairs organized by node
    pub fn pretty_evals(&self) -> String {
        let mut s = format!("Evals for {}:",self);
        for id in 1..self.expr.nodes.len() {
            let id = Id::from(id);
            s.push_str(&format!(
                "\n\t{}:\n\t\t{}",
                self.expr.to_string_uncurried(Some(id)),
                self.evals_of_node(id).iter().map(|(env,res)| format!("{:?} -> {:?}",env,res)).collect::<Vec<_>>().join("\n\t\t")
            ))
        }
        s
    }

    /// gets vec of (env,result) pairs for all the envs this node has been evaluated under
    pub fn evals_of_node(&self, node: Id) -> Vec<(Vec<Val<D>>,Val<D>)> {
        unimplemented!()
        // self.evals.borrow().iter()
        //     .filter(|((id,_env),_res)| *id == node)
        //     .map(|((_id,env),res)| (env.clone(),(*res).clone()))
        //     .collect()
    }

    /// apply a function (Val) to an argument (Val)
    pub fn apply(&self, f: &Val<D>, x: Val<D>) -> VResult<D> {
        self.apply_lazy(f, LazyVal::new_strict(x))
    }
    // apply a function (Val) to an argument (LazyVal)
    pub fn apply_lazy(&self, f: &Val<D>, x: LazyVal<D>) -> VResult<D> {
        match f {
            Val::PrimFun(f) => f.apply(x.clone(), self),
            Val::LamClosure(f, env) => {
                let mut new_env = vec![x.clone()];
                new_env.extend(env.iter().cloned());
                self.eval_child(*f, &mut new_env)
            }
            _ => panic!("Expected function or closure"),
        }
    }

    /// Evaluate the expression with the given environment. If this returns
    /// an error the whole Executable should be considered poisoned and you
    /// should not inspect partial results in it. This is because the error
    /// may have come from a panic that could leave Domain::Data in an invalid
    /// state depending on how you implement it. Read about `UnwindSafe` in Rust
    /// for more info. In practice it's probably actually generally safe to do
    /// whatever you want with the expression.
    /// 
    /// You're free to call eval() on nodes multiple times (it caches results), and
    /// to call it on new environments (itll union all the results together naturally). Just
    /// beware of the Error case.
    /// 
    /// todo I can probably add some refcell bool flag that gets flipped when a panic is caught. Or
    /// we could have people mark their stuff as UnwindSafe and really understand the risks (which are
    /// probably nonexistent in most cases and only ever really introduced by Data)
    pub fn eval(&self, env: &mut [LazyVal<D>]) -> VResult<D> {
        match D::TRUST_LEVEL {
            TrustLevel::WontLoopWontPanic => {
                self.eval_child(self.expr.root(), env)
            }
            TrustLevel::WontLoopMayPanic => {
                std::panic::catch_unwind( // todo note if running in a separate thread eg when parallelizing you dont need catch_unwind i guess
                    std::panic::AssertUnwindSafe( // todo see my notes unsure what to do about this
                        ||self.eval_child(self.expr.root(), env)
                    )
                ).map_err(|e| format!("panic: {:?}",e))
                 .and_then(|res| res) // flattens Result<Result<T,E>,E> -> Result<T,E>
            }
            TrustLevel::MayLoopMayPanic => {
                // todo implement this using run_with_timeout(). Will need to do some Serde stuff but shouldnt be terrible
                unimplemented!()
            }
        }
    }

    /// eval a subexpression in an environment
    pub fn eval_child(&self, child: Id, env: &mut [LazyVal<D>]) -> VResult<D> {
        let key = (child, env.to_vec());
        if let Some(val) = self.evals.borrow().get(&key).cloned() {
            return Ok(val);
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