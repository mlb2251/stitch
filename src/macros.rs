
/// this macros defines two lazy static variables PRIMS and FUNCS 
#[macro_export]
macro_rules! define_semantics {
    (   $domain_val:ty;
        $($rest:tt)*
        // $($string:literal = ($fname:ident, $ty:literal) ),*
    ) => {
        lazy_static::lazy_static! {

        static ref DSL_ENTRIES: $crate::DSL<$domain_val> = DSL::new(vec![
            dsl_entries!($domain_val; $($rest)*)
            // $($crate::DSLEntry::new(
            //     $string.into(), // name
            //     PrimFun(CurriedFn::new($string.into(), $ty.parse::<Type>().unwrap().arity())), // val
            //     $ty.parse().unwrap() // type
            // )),*
        ].into_iter().collect());


        static ref LOOKUP_FN_PTR: HashMap<Symbol,$crate::DSLFn<$domain_val>> = vec![
            // $((
            //     $string.into(), // name
            //     $fname as $crate::DSLFn<$domain_val> // dsl_fn ptr
            // )),*
            fn_ptr_lookup!($domain_val; $($rest)*)
        ].into_iter().collect();

        }
    }
}

#[macro_export]
macro_rules! dsl_entries {
    // case like: "head" = (head, "list t0 -> t0"),
    ( $domain_val:ty; ) => {
        // base case, do nothing
    };

    (   $domain_val:ty;
        $string:literal = ($fname:ident, $ty:literal),
        $($rest:tt)*
    ) => {
        // add entry
        $crate::DSLEntry::new(
            $string.into(), // name
            PrimFun(CurriedFn::new($string.into(), $ty.parse::<Type>().unwrap().arity())), // val
            $ty.parse().unwrap() // type
        ), // <-- add comma
        // recurse
        dsl_entries!{$domain_val; $($rest)*};
    };

    // case like: "[1,2,3]" = "list int"),
    (   $domain_val:ty;
        $string:literal = $ty:literal,
        $($rest:tt)*
    ) => {
        // add entry
        $crate::DSLEntry::new(
            $string.into(), // name
            <$domain_val>::val_of_prim_fallback($string.into()), // val
            $ty.parse().unwrap() // type
        ), // <-- add comma
        // recurse
        dsl_entries!($domain_val; $($rest)*);
    }
}

#[macro_export]
macro_rules! fn_ptr_lookup {

    ( $domain_val:ty; ) => {
        // base case, do nothing
    };

    // case like: "head" = (head, "list t0 -> t0"),
    (   $domain_val:ty;
        $string:literal = ($fname:ident, $ty:literal),
        $($rest:tt)*
    ) => {
        // add entry
        (
            $string.into(), // name
            $fname as $crate::DSLFn<$domain_val> // dsl_fn ptr
        ), // <-- add comma
        // recurse
        fn_ptr_lookup!($domain_val; $($rest)*);
    };

    // case like: "[1,2,3]" = "list int"),
    (   $domain_val:ty;
        $string:literal = $ty:literal,
        $($rest:tt)*
    ) => {
        // no entry since this is not a
        // recurse
        fn_ptr_lookup!($domain_val; $($rest)*);
    }
}




#[macro_export]
macro_rules! dsl_entries_lookup_gen {
    (  
    ) => { 
        fn lookup_fn_ptr(p: Symbol) -> DSLFn {
            *LOOKUP_FN_PTR.get(&p).unwrap()
        }
        fn dsl_entry(p: Symbol) -> Option<&'static DSLEntry<Self>> {
            DSL_ENTRIES.entries.get(&p)
        }
        fn dsl_entries() -> std::collections::hash_map::Values<'static, Symbol, DSLEntry<Self>> {
            DSL_ENTRIES.entries.values()
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
        use $crate::domain::FromVal;
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