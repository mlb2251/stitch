
/// this macros defines two lazy static variables PRIMS and FUNCS 
#[macro_export]
macro_rules! define_semantics {
    (   $domain:ty;
        $($rest:tt)*
        // $($string:literal = ($fname:ident, $ty:literal) ),*
    ) => {
        lazy_static::lazy_static! {

        static ref DSL_ENTRIES: $crate::DSL<$domain> = {
            let mut entries = vec![];
            dsl_entries!{$domain; entries; $($rest)*};
            DSL::new(entries)
        };

        static ref LOOKUP_FN_PTR: HashMap<Symbol,$crate::DSLFn<$domain>> = {
            let mut entries = HashMap::new();
            fn_ptr_entries!{$domain; entries; $($rest)*};
            entries
        };

        }
    }
}

#[macro_export]
macro_rules! dsl_entries {
    ( $domain:ty; $entries:ident; ) => {
        // base case, do nothing
    };

    // case like: "head" = (head, "list t0 -> t0"),
    (   $domain:ty; $entries:ident;
        $string:literal = ($fname:ident, $ty:literal),
        $($rest:tt)*
    ) => { 
        // add entry
        $entries.push($crate::DSLEntry::new(
            $string.into(), // name
            PrimFun(CurriedFn::<$domain>::new($string.into(), $ty.parse::<Type>().unwrap().arity())), // val
            $ty.parse().unwrap() // type
        ));
        // recurse
        dsl_entries!{$domain; $entries; $($rest)*};
    };

    // case like: "[1,2,3]" = "list int"),
    (   $domain:ty; $entries:ident;
        $string:literal = $ty:literal,
        $($rest:tt)*
    ) => {
        // add entry
        $entries.push($crate::DSLEntry::new(
            $string.into(), // name
            <$domain>::val_of_prim_fallback($string.into()).unwrap(), // val
            $ty.parse().unwrap() // type
        ));
        // recurse
        dsl_entries!($domain; $entries; $($rest)*);
    }
}

#[macro_export]
macro_rules! fn_ptr_entries {

    ( $domain:ty; $entries:ident; ) => {
        // base case, do nothing
    };

    // case like: "head" = (head, "list t0 -> t0"),
    (   $domain:ty; $entries:ident;
        $string:literal = ($fname:ident, $ty:literal),
        $($rest:tt)*
    ) => {
        // add entry
        $entries.insert(
            $string.into(), // name
            $fname as $crate::DSLFn<$domain> // dsl_fn ptr
        );
        // recurse
        fn_ptr_entries!($domain; $entries; $($rest)*);
    };

    // case like: "[1,2,3]" = "list int"),
    (   $domain:ty; $entries:ident;
        $string:literal = $ty:literal,
        $($rest:tt)*
    ) => {
        // no entry since this is not a function
        // recurse
        fn_ptr_entries!($domain; $entries; $($rest)*);
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
        use $crate::expr::eval::FromVal;
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