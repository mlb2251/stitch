use std::{collections::VecDeque};

use crate::*;
use egg::Symbol;



#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnifyErr {
    Occurs,
    ConcreteSubtree,
    Production
}
pub type UnifyResult = Result<(), UnifyErr>;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Var(usize), // type variable like t0 t1 etc
    Term(Symbol, Vec<Type>), // symbol is the name like "int" or "list" or "->" and Vec<Type> is the args which is empty list for things like int etc
    // Arrow(Box<Type>,Box<Type>)
}

impl Type {
    pub const ARROW: &'static str = "->";

    pub fn base(name: Symbol) -> Type {
        Type::Term(name, vec![])
    }

    pub fn arrow(left: Type, right: Type) -> Type {
        Type::Term(Type::ARROW.into(), vec![left, right])
    }

    pub fn is_arrow(&self) -> bool {
        match self {
            Type::Var(_) => false,
            Type::Term(name, _) => name.as_str() == Type::ARROW,
        }
    }

    pub fn as_arrow(&self) -> Option<(&Type, &Type)> {
        match self {
            Type::Term(name,args) => {
                if name.as_str() != Type::ARROW {
                    return None
                }
                assert_eq!(args.len(),2);
                Some((&args[0], &args[1]))
            },
            _ => None
        }
    }

    /// iterates over all (left_type,right_type) pairs for the chain of arrows
    /// starting here. Empty iterator if this is not an arrow.
    // pub fn iter_nodes(&self) -> impl Iterator<Item=&Type> {
    //     return NodeIter { curr: self }
    // }

    /// iterates over all nodes in the term of this type
    pub fn iter_arrows(&self) -> ArrowIter {
        ArrowIter { curr: self }
    }

    /// iterates over uncurried argument types of this arrow type
    pub fn iter_args(&self) -> impl Iterator<Item=&Type> {
        self.iter_arrows().map(|(left,_right)| left)
    }

    /// arity of this arrow type (zero if not an arrow type)
    pub fn arity(&self) -> usize {
        self.iter_args().count()
    }

    /// return type of this arrow types *after* uncurrying. For a non arrow type
    /// this just returns the type itself.
    pub fn return_type(&self) -> &Type {
        self.iter_arrows().last().map(|(_left,right)| right).unwrap_or(self)
    }

    /// true if there are no type vars in this type
    pub fn is_concrete(&self) -> bool {
        match self {
            Type::Var(_) => false,
            Type::Term(_, args) => args.iter().all(|ty| ty.is_concrete())
        }
    }

    /// true if type var i occurs in this type
    pub fn occurs(&self, i: usize) -> bool {
        match self {
            Type::Var(j)  => i == *j,
            Type::Term(_, args) => args.iter().any(|ty| ty.occurs(i))
        }
    }

    pub fn apply(&self, ctx: &mut Context) -> Type {
        if self.is_concrete() {
            return self.clone();
        }
        match self {
            Type::Var(i) => {
                // look up the type var in the ctx to see if its bound
                if let Some(tp) = ctx.get(*i).cloned() {
                    // in case it's bound to something that ALSO has variables, we want to track those down too
                    let tp_applied = tp.apply(ctx);
                    if tp != tp_applied {
                        // and to save our work for the future, lets amortize it (union-find style) by saving what we
                        // found things were bound to. Since bindings will never change this is okay.
                        ctx.set(*i, tp_applied.clone())
                    }
                    tp_applied
                } else {
                    self.clone() // t0 is not bound by ctx so we leave it unbound
                }
            },
            Type::Term(name, args) => Type::Term(*name, args.iter().map(|ty| ty.apply(ctx)).collect())
        }
    }

    /// same as apply() but doesnt do the unionfind style caching of results, so there's no need to mutate the ctx
    pub fn apply_immut(&self, ctx: &Context) -> Type {
        if self.is_concrete() {
            return self.clone();
        }
        match self {
            Type::Var(i) => {
                // look up the type var in the ctx to see if its bound
                if let Some(tp) = ctx.get(*i).cloned() {
                    // in case it's bound to something that ALSO has variables, we want to track those down too
                    tp.apply_immut(ctx)
                } else {
                    self.clone() // t0 is not bound by ctx so we leave it unbound
                }
            },
            Type::Term(name, args) => Type::Term(*name, args.iter().map(|ty| ty.apply_immut(ctx)).collect())
        }
    }


    /// shifts all variables in a type such that they are fresh variables in the context, returning a new type
    pub fn instantiate(&self, ctx: &mut Context) -> Type {
        if self.is_concrete() {
            return self.clone()
        }
        fn instantiate_aux(ty: &Type, ctx: &mut Context, shift_by: usize) -> Type {
            match ty {
                Type::Var(i) => {
                    let new = i + shift_by;
                    ctx.ensure_capacity(new);
                    Type::Var(new)
                },
                Type::Term(name, args) => Type::Term(*name, args.iter().map(|t| instantiate_aux(t, ctx, shift_by)).collect()),
            }
        }
        // shift by the highest var that already exists, so that theres no conflict
        instantiate_aux(self, ctx, ctx.subst.len())
    }
}

pub struct ArrowIter<'a> {
    curr: &'a Type
}

impl<'a> Iterator for ArrowIter<'a> {
    type Item = (&'a Type, &'a Type);

    fn next(&mut self) -> Option<Self::Item> {
        if let Some((left,right)) = self.curr.as_arrow() {
            self.curr = right;
            Some((left,right))
        } else {
            None
        }
    }
}



impl std::str::FromStr for Type {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        crate::parse_types::parse(s)
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn helper(ty: &Type, f: &mut std::fmt::Formatter<'_>, arrow_parens: bool) -> std::fmt::Result {
            match ty {
                Type::Var(i) => write!(f,"t{}", i),
                Type::Term(name, args) => {
                    if args.is_empty() {
                        write!(f, "{}", name)
                    } else if name.as_str() == Type::ARROW {
                        assert_eq!(args.len(), 2);
                        // write!(f, "({} {} {})", &args[0], name, &args[1])
                        if arrow_parens {
                            write!(f, "(")?;
                        }
                        helper(&args[0], f, true)?;
                        write!(f, " {} ", Type::ARROW)?;
                        helper(&args[1], f, false)?;
                        if arrow_parens {
                            write!(f, ")")?;
                        }
                        Ok(())
                    } else {
                        write!(f, "({}", name)?;
                        for arg in args.iter() {
                            write!(f, " ")?;
                            helper(arg, f, true)?;
                        }
                        write!(f, ")")
                    }
                },
            }
        }
        helper(self, f, true)
    }
}




#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Context {
    // next_var: usize,
    subst: Vec<Option<Type>> // todo also try ahashmap tho i just wanted to avoid the allocations
}

impl Context {
    pub fn empty() -> Context {
        Context::default()
    }

    fn fresh_type_var(&mut self) -> Type {
        self.subst.push(None);
        Type::Var(self.subst.len() - 1)
    }

    /// a very quick non-allocating check that returns false if it's
    /// obvious that these types won't unify. First this checks if t1 and t2 have the same constructors
    /// and if theres an obvious mismatch there it gives up. Then it goes and looks up the types in the ctx
    /// in case they were typevars, and then again checks if they have th same constructor. It uses apply_immut() to
    /// avoid mutating the context for this lookup.
    pub fn might_unify(&self, t1: &Type, t2: &Type) -> bool {
        self.might_unify_(t1,t2) && self.might_unify_(&t1.apply_immut(self), &t2.apply_immut(self))
    }

    pub fn might_unify_(&self, t1: &Type, t2: &Type) -> bool {
        match (t1,t2) {
            (Type::Var(_), Type::Var(_)) => true,
            (Type::Var(_), Type::Term(_, _)) => true,
            (Type::Term(_, _), Type::Var(_)) => true,
            (Type::Term(x, xs), Type::Term(y, ys)) => {
                x == y && xs.len() == ys.len() && xs.iter().zip(ys.iter()).all(|(x,y)| self.might_unify(x,y))
            },
        }
    }

    pub fn unify(&mut self, t1: &Type,  t2: &Type) -> UnifyResult {
        // println!("unify({},{}) {}", t1, t2, self);
        let t1: Type = t1.apply(self);
        let t2: Type = t2.apply(self);
        // println!("  ...({},{}) {}", t1, t2, self);
        if t1.is_concrete() && t2.is_concrete() {
            // if both types are concrete, simple equality works because we dont need to do any fancy variable binding
            if t1 == t2 {
                return Ok(())
            } else {
                return Err(UnifyErr::ConcreteSubtree)
            }
        }
        match (t1, t2) {
            (Type::Var(i), ty) | (ty, Type::Var(i)) => {
                if ty == Type::Var(i) { return Ok(()) } // unify(t0, t0) -> true
                if ty.occurs(i) { return Err(UnifyErr::Occurs) } // unify(t0, (t0 -> int)) -> false
                // *** Above is the "occurs" check, which prevents recursive definitions of types. Removing it would allow them.

                // todo is it really ok to just set this to this? what if that overwrites some important equality thats already there?
                self.set(i, ty);
                Ok(())
            },
            (Type::Term(x, xs), Type::Term(y, ys)) => {
                // simply recurse
                if x != y || xs.len() != ys.len() {
                    return Err(UnifyErr::Production)
                }
                xs.iter().zip(ys.iter()).try_for_each(|(x,y)| self.unify(x,y))
            }
        }
    }

    #[inline(always)]
    fn get(&self, var: usize) -> Option<&Type> { // todo written in a silly way, rewrite
        if var < self.subst.len() { 
            self.subst[var].as_ref()
        } else {
            None
        }
    }

    #[inline(always)]
    fn set(&mut self, var: usize, ty: Type) {
        self.ensure_capacity(var);
        self.subst[var] = Some(ty);
    }

    /// adds new fresh type vars as necessary such that variable Var exists
    #[inline(always)]
    fn ensure_capacity(&mut self, var: usize) {
        while var >= self.subst.len() {
            self.fresh_type_var();
        }
    }

}

impl std::fmt::Display for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,"{{")?;
        let mut first: bool = true;
        for (i, item) in self.subst.iter().enumerate() {
            if let Some(ty) = item {
                if !first { write!(f, ", ")? } else { first = false }
                write!(f, "{}:{}", i, ty)?
            }
        }
        write!(f,"}}")
    }
}


impl Expr {
    pub fn infer<D: Domain>(&self, child: Option<Id>, ctx: &mut Context, env: &mut VecDeque<Type>) -> Result<Type,UnifyErr> {
        // println!("infer({})", self.to_string_uncurried(child));
        let child = child.unwrap_or(self.root());
        match &self.nodes[usize::from(child)] {
            Lambda::App([f,x]) => {
                let return_tp = ctx.fresh_type_var();
                let x_tp = self.infer::<D>(Some(*x), ctx, env)?;
                let f_tp = self.infer::<D>(Some(*f), ctx, env)?;
                ctx.unify(&f_tp, &Type::arrow(x_tp, return_tp.clone()))?;
                Ok(return_tp.apply(ctx))
            },
            Lambda::Lam([b]) => {
                let var_tp = ctx.fresh_type_var();
                // todo maybe optimize by making this a vecdeque for faster insert/remove at the zero index
                env.push_front(var_tp.clone());
                let body_tp = self.infer::<D>(Some(*b), ctx, env)?;
                env.pop_front();
                Ok(Type::arrow(var_tp, body_tp).apply(ctx))
            },
            Lambda::Var(i) => {
                if (*i as usize) >= env.len() {
                    panic!("unbound variable encountered during infer(): ${}", i)
                }
                Ok(env[*i as usize].apply(ctx))
            },
            Lambda::IVar(_i) => {
                // interesting, I guess we can have this and it'd probably be easy to do
                unimplemented!();
            }
            Lambda::Prim(p) => {
                Ok(D::type_of_prim(*p).instantiate(ctx))
            },
            Lambda::Programs(_) => panic!("trying to infer() type of Programs() node"),
        }
    }
}


#[test]
fn test_types() {
    use domains::simple::SimpleVal;

    fn assert_unify(t1: &str, t2: &str, expected: UnifyResult) {
        let mut ctx = Context::empty();
        let res = ctx.unify(&t1.parse::<Type>().unwrap(),
                     &t2.parse::<Type>().unwrap());
        assert_eq!(res, expected);
    }

    fn assert_infer(p: &str, expected: Result<&str, UnifyErr>) {
        let res = p.parse::<Expr>().unwrap().infer::<SimpleVal>(None, &mut Default::default(), &mut Default::default());
        assert_eq!(res, expected.map(|ty| ty.parse::<Type>().unwrap()));
    }

    assert_unify("int", "int", Ok(()));
    assert_unify("int", "t0", Ok(()));
    assert_unify("int", "t1", Ok(()));
    assert_unify("(list int)", "(list t1)", Ok(()));
    assert_unify("(int -> bool)", "(int -> t0)", Ok(()));
    assert_unify("t0", "t1", Ok(()));

    assert_infer("3", Ok("int"));
    assert_infer("[1,2,3]", Ok("list int"));
    assert_infer("(+ 2 3)", Ok("int"));
    assert_infer("(lam $0)", Ok("t0 -> t0"));
    assert_infer("(lam (+ $0 1))", Ok("int -> int"));
    assert_infer("map", Ok("((t0 -> t1) -> (list t0) -> (list t1))"));
    assert_infer("(map (lam (+ $0 1)))", Ok("list int -> list int"));
    // assert_infer("(map (lam (+ 1)))", Err(UnifyErr::Production));
    // assert_infer("map 3");

    let mut ctx = Context::empty();

    let map_tp: Type = "((t0 -> t1) -> (list t0) -> (list t1))".parse().unwrap();

    let args: Vec<&Type> = map_tp.iter_args().collect();

    ctx.unify(args[0], &"int -> bool".parse().unwrap());
    println!("{}", ctx);
    println!("{}", args[1].apply(&mut ctx));


}