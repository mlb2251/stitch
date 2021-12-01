use nix::unistd::Pid;
use nix::sys::signal;
use std::time::Duration;
use serde::Serialize;
use serde::de::DeserializeOwned;
use procspawn::SpawnError;

/// This will call a function with a timeout. It was a pain to make and I tried
/// 4+ different libraries but this is what we're going with.
/// Note that since this is a fn() you cant closure in any variables. So you can
/// pass in a closure but it must take all inputs thru `args`.
/// *** IMPORTANT USAGE DETAILS:
///     * `procspawn::init();` must be called at the start of your main() function
///     * use `std::panic::set_hook(Box::new(|_| ()));` at the start of your function if you want to suppress printing panics. This
///       shouldn't affect the main process so no need to unset it.
/// todo there's a tiny race condition that could technically attempt to kill a random process.
/// todo That's ok for my purposes but not if this becomes a real public library
pub fn run_with_timeout<T,A>(f: fn(A) -> T, args:A, timeout:Duration) -> Result<T,SpawnError>
where
    T: Serialize + DeserializeOwned,
    A: Serialize + DeserializeOwned,
{
    let handle = procspawn::spawn(args, f);
    let pid = Pid::from_raw(handle.pid().unwrap() as i32);
    let result: Result<T,SpawnError> = handle.join_timeout(timeout);

    if let Err(e) = &result {
        if e.is_timeout() {
            // kill the process
            // note that handle.kill() would be nice except that .join_timeout() causes a `move` on `handle` so we can't touch it after that
            // There's a tiny race condition here but I think itll be okay
            if let Err(e) = signal::kill(pid,signal::Signal::SIGKILL) {
                println!("Possible leak: Could not kill pid={:?}: {:?}",pid,e);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal() {
        procspawn::init();
        let timeout = Duration::from_millis(10000);
        let args = vec![1, 2, 3, 4];
        let res = run_with_timeout(|args| {
            args.into_iter().sum::<i64>()
        },args,timeout);
        assert_eq!(res.unwrap(), 10);
    }

    #[test]
    fn panic() {
        procspawn::init();
        let timeout = Duration::from_millis(10000);
        let args = vec![1, 2, 3, 4];
        let res = run_with_timeout(|_args| {
            std::panic::set_hook(Box::new(|_| ())); // disable printing of panic messages
            panic!("aaaaa");
        },args,timeout);
        assert!(res.unwrap_err().is_panic());
    }

    #[test]
    fn infinite_loop() {
        procspawn::init();
        let timeout = Duration::from_millis(2000);
        let args = vec![1, 2, 3, 4];
        let res = run_with_timeout(|_args| {
            loop{};
        },args,timeout);
        assert!(res.unwrap_err().is_timeout());
    }
}