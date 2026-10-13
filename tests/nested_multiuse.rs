use clap::Parser;
use stitch_core::*;

fn rewrite(program: &str, body: &str) -> (String, serde_json::Value) {
    let cfg = MultistepCompressionConfig::parse_from(["compress", "--silent", "--rewrite-check"]);
    let inv = Invention::from_string("fn_0", body, None);
    // The normal mismatch assertion compares predicted savings with an actual
    // rewrite; keep it enabled for all cases, including the parameterized ones.
    let (rewritten, _, output) = rewrite_with_inventions(&[program.to_owned()], &[inv], &cfg);
    (rewritten[0].clone(), output)
}

#[test]
fn nested_multiuse_counts_retained_argument_once() {
    let (rewritten, output) = rewrite("(f (g x (L x y)) (L (g x (L x y)) z))", "(#2 #1 (L #1 #0))");
    assert_eq!(rewritten, "(fn_0 z (fn_0 y x g) f)");
    assert_eq!(output["original_cost"], 1312);
    assert_eq!(output["final_cost"], 706);
    assert_eq!(output["abstractions"][0]["utility"], 502);
}

#[test]
fn nested_multiuse_handles_nonmatching_wrappers_and_depth() {
    for copies in 2..=3 {
        let body = format!("(L {})", vec!["#0"; copies].join(" "));
        let mut program = "x".to_owned();
        let mut expected = "x".to_owned();
        for _ in 0..5 {
            let argument = format!("(w {})", program);
            program = format!("(L {})", vec![argument; copies].join(" "));
            expected = format!("(fn_0 (w {}))", expected);
            assert_eq!(rewrite(&program, &body).0, expected);
        }
    }
}

#[test]
fn nested_multiuse_preserves_lambda_scopes() {
    assert_eq!(
        rewrite(
            "(lam (f (g $0 (L $0 y)) (L (g $0 (L $0 y)) z)))",
            "(#2 #1 (L #1 #0))",
        )
        .0,
        "(lam (fn_0 z (fn_0 y $0 g) f))",
    );
    assert_eq!(
        rewrite(
            "(lam (L (lam ($1 $0)) (lam ($1 $0))))",
            "(L (lam (#0 $0)) (lam (#0 $0)))",
        )
        .0,
        "(lam (fn_0 $0))",
    );
}

#[test]
fn nested_multiuse_handles_overlapping_matches() {
    // Replacing (h x x) would save zero, so leave the innermost match alone.
    assert_eq!(
        rewrite("(f (g (h x x) (h x x)) (g (h x x) (h x x)))", "(#1 #0 #0)",).0,
        "(fn_0 (fn_0 (h x x) g) f)",
    );
}

#[test]
fn nested_multiuse_preserves_weighted_task_costs() {
    let programs = vec![
        "(f (g x (L x y)) (L (g x (L x y)) z))".to_owned(),
        "(u (f (g x (L x y)) (L (g x (L x y)) z)))".to_owned(),
    ];
    for tasks in [vec!["a", "b"], vec!["a", "a"]] {
        for weights in [vec![1.0, 1.0], vec![0.5, 2.0]] {
            let cfg = MultistepCompressionConfig::parse_from([
                "compress",
                "-i1",
                "-a3",
                "--silent",
                "--no-opt",
                "--allow-single-task",
            ]);
            let inv = Invention::from_string("fn_0", "(#2 #1 (L #1 #0))", None);
            let expected = if tasks[0] == tasks[1] {
                (706.0_f32 * weights[0])
                    .round()
                    .min((807.0_f32 * weights[1]).round()) as i64
            } else {
                (706.0_f32 * weights[0]).round() as i64 + (807.0_f32 * weights[1]).round() as i64
            };
            let (_, result) = multistep_compression(
                &programs,
                Some(tasks.iter().map(|t| (*t).to_owned()).collect()),
                Some(weights),
                None,
                Some(vec![inv]),
                &cfg,
            );
            assert_eq!(result["final_cost"], expected);
        }
    }
}
