// use nix::unistd::Pid;
// use nix::sys::signal::{self, Signal};
// use std::thread;
// use nix::unistd::{fork,ForkResult};
// use std::time::Duration;
// use nix::unistd::alarm;

// // use std::sync::mpsc::channel;
// use std::process;

// use std::sync::Arc;
// use std::sync::atomic::{AtomicBool};
// use serde::{Serialize, Deserialize};

// use signal_hook::{consts::{SIGINT,SIGALRM}, iterator::Signals};
// use std::{error::Error};

// use nix::sys::signal::Signal::SIGKILL;
// use ipc_channel::ipc;

// const TIMEOUT: u128 = 3000;

// pub fn run_with_timeout<F, T>(f: F)
// where
//     // F: FnOnce() -> T,
//     // F: Send + 'static,
//     // T: Send + 'static + Serialize,
//     F: FnOnce() -> T,
//     T: Serialize + for<'de> serde::Deserialize<'de> + std::fmt::Debug,
// {


//     let (server, server_name) = ipc::IpcOneShotServer::<ipc::IpcReceiver<T>>::new().unwrap();

//     // sender.send(f()).unwrap();
//     // receiver.try_recv().unwrap();

//     let res = match unsafe{ fork() } {
//         Ok(ForkResult::Child) => {
//             println!("child start");
//             let (sender, receiver) = ipc::channel::<T>().unwrap();
//             let tx0 = ipc::IpcSender::connect(server_name).unwrap(); 
//             tx0.send(receiver).unwrap();
//             println!("child sends receiver");

//             let res = f();
//             println!("child send");
//             sender.send(res).unwrap();
//             // sender.send(res).unwrap();
//             println!("child exit");
//             thread::sleep(Duration::from_millis(4000));
//             process::exit(0);
//         },
//         Ok(ForkResult::Parent{child}) => {
//             let tstart = std::time::Instant::now();
//             println!("parent waits");
//             let (receiver, _) = server.accept().unwrap();
//             println!("parent gets receiver");
//             loop {
//                 match receiver.try_recv() {
//                     Ok(res) => {
//                         println!("parent got result");
//                         break Some(res);
//                     },
//                     Err(e) => match e {
//                         ipc::TryRecvError::Empty => {
//                             if std::time::Instant::now().duration_since(tstart).as_millis() > TIMEOUT {
//                                 println!("timeout!!!");
//                                 signal::kill(child,SIGKILL).unwrap();
//                                 break None;
//                             }
//                             // println!("retry...");
//                         }
//                         _ => {
//                             panic!("Error during recv(): {:?}", e);
//                         }
//                     }
//                 }
//             }
//         },
//         Err(e) => {
//             panic!("fork error: {}", e);
//         }
//     };

//     println!("parent got result: {:?}",res);

//     // signal::kill(Pid::from_raw(process::id() as i32), SIGABRT);
    
//     // as soon as this line executes, ctrl-c will start doing nothing
//     // let mut signals = Signals::new(&[SIGINT]).unwrap();

//     // let flag = Arc::new(AtomicBool::new(false));
//     // let _ = signal_hook::flag::register(SIGALRM, Arc::clone(&flag));



//     // let handle = thread::spawn(move || {
//         //  let signal = unsafe {
//         //     signal_hook::low_level::register(signal_hook::consts::SIGINT, || {
//         //         vec![1][1];
//         //     })
//         //  };
//     //     loop {println!("aaaaa")}
//     // });


//     // let handle = thread::spawn(move || {
//     //     for sig in signals.forever() {
//     //         match sig {
//     //             SIGINT => {
//     //                 println!("SIGINT received");
//     //                 panic!("aaaa")
//     //             },
//     //             _ => {
//     //                 println!("other signal received");
//     //             }
//     //         }
//     //     }
//     // });

//     // Following code does the actual work, and can be interrupted by pressing
//     // Ctrl-C. As an example: Let's wait a few seconds.
//     // println!("interrupt me");
//     // while !flag.load(std::sync::atomic::Ordering::Relaxed) {
//         // Do some time-limited stuff here
//         // (if this could block forever, then there's no guarantee the signal will have any
//         // effect).
//         // println!("doin stufffff");
//     // }
//     // thread::sleep(Duration::from_millis(4000));

//     // println!("aaaaayyy!!");


//     // Create a simple streaming channel
//     // let (tx, rx) = channel();
//     // let handle = thread::spawn(move|| {
//     //     println!("Spawned thread");
//     //     thread::spawn(move|| {
//     //         thread::sleep(Duration::from_millis(2000));
//     //         panic!("aaaaa")
//     //     });


//     //     println!("Child: My pid is {}", process::id());
//     //     tx.send(10).unwrap();
//     //     panic!("aaaa!")
//     // });
//     // println!("Parent: My pid is {}", process::id());

//     // assert_eq!(rx.recv().unwrap(), 10);

//     // let handle = thread::spawn(f);

//     // return handle.join()
// }


// pub fn test_run() {
//     run_with_timeout(|| {
//         println!("Hello from child");
//         45
//     });

//     println!("done!");
//     process::exit(0);

//     // Send SIGTERM to child process.
//     // signal::kill(Pid::from_raw(child.id()), Signal::SIGTERM).unwrap();
// }


// // use signal_hook::{iterator, consts::{SIGINT};
// // use std::{process, thread, error::Error};
// // use nix::sys::signal::{self, Signal};

// // Registers UNIX system signals
// // fn register_signal_handlers() -> Result<(), Box<dyn Error>>  {
// //     let mut signals = iterator::Signals::new(&[SIGINT])?;

// //     // signal execution is forwarded to the child process
// //     thread::spawn(move || {
// //         for sig in signals.forever() {
// //             match sig {
// //                 SIGINT => assert_ne!(0, sig), // assert that the signal is sent
// //                 _ => continue,
// //             }
// //         }
// //     });

// //     Ok(())
// // }



