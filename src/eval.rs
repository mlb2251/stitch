use nix::unistd::Pid;
use nix::sys::signal;
use std::time::Duration;
use serde::Serialize;
use serde::de::DeserializeOwned;

/// This will call a function with a timeout. It was a pain to make and I tried
/// 4+ different libraries but this is what we're going with.
/// Note that since this is a fn() you cant closure in any variables. So you can
/// pass in a closure but it must take all inputs thru `args`.
pub fn run_with_timeout<T,A>(f: fn(A) -> T, args:A, timeout:Duration) -> Option<T>
where
    T: Serialize + DeserializeOwned + std::fmt::Debug,
    A: Serialize + DeserializeOwned + std::fmt::Debug,
{

    let handle = procspawn::spawn(args, f);
    let pid = Pid::from_raw(handle.pid().unwrap() as i32);
    let result = handle.join_timeout(timeout);

    match result {
        Ok(r) => Some(r),
        Err(e) => if e.is_timeout() {
                // note that handle.kill() would be nice except that .join_timeout() causes a `move` on `handle` so we can't touch it after that
                // There's a tiny race condition here but I think itll be okay
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
    let timeout = Duration::from_millis(1000);

    println!("***Panic example***");
    let args = vec![1, 2, 3, 4];
    let res = run_with_timeout(|args| {
        std::panic::set_hook(Box::new(|_| ())); // disable printing of panic messages
        panic!("aaaaa");
    },args,timeout);
    println!("Example Returned: {:?}", res);

    println!("***Infinite loop example***");
    let args = vec![1, 2, 3, 4];
    let res = run_with_timeout(|args| {
        println!("starting loop...");
        loop{};
    },args,timeout);
    println!("Example Returned: {:?}", res);


    println!("***Normal example***");
    let args = vec![1, 2, 3, 4];
    let res = run_with_timeout(|args| {
        println!("Received data {:?}", &args);
        args.into_iter().sum::<i64>()
    },args,timeout);
    println!("Example Returned: {:?}", res);
}
    