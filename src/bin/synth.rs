use stitch::*;
use stitch::domains::prim_lists::*;

fn main() {

    // todo more or less nums than 0 to 9?
    let nums: Vec<String> = (0..10).map(|i| i.to_string()).collect();
    let terminals: Vec<Val<ListVal>> =
        [String::from("[]")].iter()
        .chain(nums.iter())
        .map(|s| ListVal::val_of_prim(s.into()).unwrap()).collect();
    
    bottom_up::<ListVal>( unimplemented!(), unimplemented!(), unimplemented!());

}
