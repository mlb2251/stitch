use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[test]
fn utility_by_rewrite_rejects_identity_without_recursing() {
    // Run in a child: before the fix the rejected identity candidate #0 causes
    // unbounded recursion, which aborts rather than producing a catchable panic.
    let dir = std::env::temp_dir().join(format!("stitch-rejected-rewrite-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("input.json");
    let output = dir.join("output.json");
    std::fs::write(&input, r#"["(f (a a a))", "(g (b b b))"]"#).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_compress"))
        .arg(&input)
        .args([
            "-i1",
            "-a3",
            "--utility-by-rewrite",
            "--rewrite-check",
            "--silent",
            "-o",
        ])
        .arg(&output)
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let start = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if start.elapsed() > Duration::from_secs(30) {
            child.kill().unwrap();
            child.wait().unwrap();
            std::fs::remove_dir_all(&dir).unwrap();
            panic!("utility_by_rewrite did not terminate within 30 seconds");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    if !status.success() {
        std::fs::remove_dir_all(&dir).unwrap();
        panic!("utility_by_rewrite failed: {}", status);
    }
    let result: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(result["final_cost"], 604);
    assert_eq!(
        result["rewritten"],
        serde_json::json!(["(f (fn_0 a))", "(g (fn_0 b))"])
    );
}
