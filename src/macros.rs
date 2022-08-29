
/// this macros defines two lazy static variables PRIMS and FUNCS 
#[macro_export]
macro_rules! define_semantics {
    (   $domain_val:ty;
        $($string:literal = ($fname:ident, $ty:literal) ),*
    ) => { 
        lazy_static::lazy_static! {
        static ref PRIMS: HashMap<Symbol, $crate::Val<$domain_val>> = vec![
            $(($string.into(), PrimFun(CurriedFn::new($string.into(), $ty.parse::<Type>().unwrap().arity())))),*
            ].into_iter().collect();
        
        static ref FUNCS: $crate::DSL<$domain_val> = DSL::new(vec![
            $($crate::DSLEntry::new(
                $string.into(), // name
                $ty.parse().unwrap(), // type
                $fname as $crate::DSLFn<$domain_val> // dsl_fn ptr
            )),*
        ].into_iter().collect());
        }
    }
}


/// this macro is used at the start of a DSL function to load arguments out of their args vec
#[macro_export]
macro_rules! load_args {
    (   $handle: expr,
        $args:expr,
        $($name:ident : $type:ty ),*
    ) => { 
        use crate::domain::FromVal;
        $(let $name:$type = <$type>::from_val($args.remove(0).eval($handle)?)?;)*
    }
}

/// this macro is used at the start of a DSL function to load arguments out of their args vec
#[macro_export]
macro_rules! load_args_lazy {
    (   $args:expr,
        $($name:ident : $type:ty ),*
    ) => { 
        $(let mut $name:$type = $args.remove(0).into();)*
    }
}