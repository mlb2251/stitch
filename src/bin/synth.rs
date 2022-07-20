use stitch::*;
use stitch::domains::prim_lists::*;
use std::iter::once;

fn main() {

    // todo more or less nums than 0 to 9?
    let initial_strs: Vec<String> = (0..10).map(|i| i.to_string())
        .chain(once(String::from("[]"))).collect();

    let initial: Vec<FoundExpr<ListVal>> = initial_strs.iter().map(|s| {
        let expr = Expr::prim(s.into());
        FoundExpr::new(ListVal::val_of_prim(s.into()).unwrap(), expr, 1)
    }).collect();    

    let fns: Vec<(DSLEntry<ListVal>,usize)> = ListVal::get_dsl().entries.values()
        .map(|entry| (entry.clone(),1)).collect();

    bottom_up(&initial, &fns, 10, 1)

}
