use crate::expr::*;

use std::collections::HashMap;
use std::fmt::{Debug};
use std::hash::Hash;
use std::collections::hash_map::Values;

pub type DSLFn<D> = fn(Env<D>, &Evaluator<D>) -> VResult<D>;

#[derive(Clone)]
pub struct DSLEntry<D: Domain> {
    pub name: Symbol, // eg "map" or "0" or "[1,2,3]"
    pub val: Val<D>,
    pub tp: Type,
    pub arity: usize,
}

#[derive(Clone)]
pub struct DSL<D:Domain> {
    pub entries: HashMap<Symbol,DSLEntry<D>>,
}

impl<D: Domain> DSLEntry<D> {
    pub fn new(name: Symbol, val: Val<D>, tp: Type) -> Self {
        let arity = tp.arity();
        DSLEntry {
            name,
            val,
            tp,
            arity,
        }
    }
}
impl<D: Domain> DSL<D> {
    pub fn new( entries: Vec<DSLEntry<D>> ) -> Self {
        DSL {
            entries: entries.into_iter().map(|entry| (entry.name, entry)).collect()
        }
    }
}


/// The key trait that defines a domain
pub trait Domain: Clone + Debug + PartialEq + Eq + Hash + 'static {
    /// Domain::Data is attached to the Evaluator so all DSL functions will have a
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
    // type Type: Debug + Clone + PartialEq + Eq + Hash;

    /// given a primitive's symbol return a runtime Val object. For function primitives
    /// this should return a PrimFun(CurriedFn) object.
    fn val_of_prim(p: egg::Symbol) -> Option<Val<Self>> {
        Self::dsl_entry(p).map(|entry| entry.val.clone()).or_else(||
            Self::val_of_prim_fallback(p))
    }

    fn val_of_prim_fallback(p: egg::Symbol) -> Option<Val<Self>>;

    /// given a function primitive's symbol return the function pointer
    /// you can use to call the function.
    /// Breakdown of the function type: it takes a slice of values as input (the args)
    /// along with a mut ref to an Expr (I'll refer to as a "handle") which is necessary
    /// for calling .apply(f,x). This setup with a handle guarantees we can always track
    /// when applys happen and log them in our Expr.evals, and also it's necessary for
    /// executing LamClosures in order to look up their body Id (and we wouldn't want
    /// LamClosures to carry around full Exprs because that would break the Expr.evals tracking)
    fn lookup_fn_ptr(p: Symbol) -> DSLFn<Self>;

    fn dsl_entry(p: Symbol) -> Option<&'static DSLEntry<Self>>;

    fn dsl_entries() -> Values<'static, Symbol, DSLEntry<Self>>;

    fn type_of_dom_val(&self) -> Type;

    fn type_of_prim(p: Symbol) -> Type {
        Self::dsl_entry(p).map(|entry| entry.tp.clone()).unwrap_or_else(|| {
            Self::type_of_dom_val(&Self::val_of_prim(p).unwrap().dom().unwrap())
        })
    }

}

