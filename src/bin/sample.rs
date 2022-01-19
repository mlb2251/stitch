use dreamegg::*;

fn main() {
    let num_samples = 20000; // TODO expose as command-line arg 
    for _ in 0..num_samples {
        println!("{}", sample_program());
    }
    //for (key, val) in &*d {
    //    match val {
    //        PrimFun(f) => println!("{}", f.name()),
    //        _ => panic!("")
    //    }
    //}
    //println!("hi");
}