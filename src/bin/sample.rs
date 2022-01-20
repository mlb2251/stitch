use dreamegg::*;

fn main() {
    let num_samples = 20000; // TODO expose as command-line arg 
    println!("{}", sample_n_programs::<domains::simple::SimpleVal>(num_samples).join("\n"));
    // ^ which domain to use should also be plugged in dynamically
}