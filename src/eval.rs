use nix::unistd::Pid;
use nix::sys::signal;
use std::time::Duration;
use std::process;
use serde::{Serialize, Deserialize};
use serde::de::DeserializeOwned;

const TIMEOUT: u64 = 1000; // ms timeout

pub fn run_with_timeout<T,A>(f: fn(A) -> T, args:A) -> Option<T>
where
    T: Serialize + DeserializeOwned + std::fmt::Debug,
    A: Serialize + DeserializeOwned + std::fmt::Debug,
{

    let handle = procspawn::spawn(args, f);
    let pid = Pid::from_raw(handle.pid().unwrap() as i32);
    let result = handle.join_timeout(Duration::from_millis(TIMEOUT));
    // handle.kill would be nice except that .join_timeout takes ownership so we cant do that

    match result {
        Ok(r) => Some(r),
        Err(e) => if e.is_timeout() {
                match signal::kill(pid,signal::Signal::SIGKILL) {
                    Ok(_) => {},
                    Err(e) => println!("Possible leak: Could not kill pid={:?}: {:?}",pid,e),
                }
                None
            } else {
                None
            }
    }
}



pub fn test_run() {

    println!("***Panic example***");
    let args = vec![1, 2, 3, 4];
    let res = run_with_timeout(|args| {
        std::panic::set_hook(Box::new(|_| ())); // disable printing of panic messages
        panic!("aaaaa");
    },args);
    println!("Example Returned: {:?}", res);

    println!("***Infinite loop example***");
    let args = vec![1, 2, 3, 4];
    let res = run_with_timeout(|args| {
        println!("starting loop...");
        loop{};
    },args);
    println!("Example Returned: {:?}", res);


    println!("***Normal example***");
    let args = vec![1, 2, 3, 4];
    let res = run_with_timeout(|args| {
        println!("Received data {:?}", &args);
        args.into_iter().sum::<i64>()
    },args);
    println!("Example Returned: {:?}", res);
}
    