use stitch::{compression, CompressionStepConfig, InputFormat, Input, Expr, CompressionStepResult};
use clap::Parser;
use serde_json::{json,Value};
// use stitch::format


fn load_programs(file: &str, input_format: InputFormat) -> (Input,Expr) {
    let input_file = std::path::Path::new(file);
    let input = input_format.load_programs_and_tasks(input_file).unwrap();
    let train_programs: Vec<Expr> = input.train_programs.iter().map(|p| p.parse().unwrap()).collect();
    (input,Expr::programs(train_programs))
}

fn out_json(train_programs: &Expr, step_results: &Vec<CompressionStepResult>) -> serde_json::Value {
    json!({
        "cmd": Value::Null,
        "args": Value::Null,
        "original_cost": train_programs.cost(),
        "original": train_programs.split_programs().iter().map(|p| p.to_string()).collect::<Vec<String>>(),
        "invs": step_results.iter().map(|inv| inv.json()).collect::<Vec<serde_json::Value>>(),
    })
}

fn run_compression(train_programs: &Expr, input: &Input, iterations: usize, args: &str) -> Vec<CompressionStepResult> {
    compression(
        &train_programs,
        &None,
        iterations,
        &CompressionStepConfig::parse_from(format!("compress {}",args).split_whitespace()),
        &input.tasks,
        &input.prev_dc_inv_to_inv_strs)
}

fn compare_out_jsons(file: &str, expected_out_file: &str, args: &str, iterations: usize, input_format: InputFormat) {
    let (input,train_programs) = load_programs(file, input_format);
    let step_results = run_compression(&train_programs, &input, iterations, args);

    let output: Value = out_json(&train_programs, &step_results);
    let expected_output: Value = serde_json::from_str(&std::fs::read_to_string(std::path::Path::new(expected_out_file)).unwrap()).unwrap();

    check_eq(&output["original"], &expected_output["original"], vec!["original".into()]);
    check_eq(&output["original_cost"], &expected_output["original_cost"], vec!["original_cost".into()]);
    check_eq(&output["invs"], &expected_output["invs"], vec!["invs".into()]);

}

fn check_eq(actual: &Value, expected: &Value, path: Vec<String>) {
    match (actual,expected) {
        (Value::Null,Value::Null) => {}
        (Value::Bool(actual),Value::Bool(expected)) => {assert_eq!(actual,expected, "\nmismatch at {}:\nactual:{}\nexpected:{}\n", path.join("."), actual, expected);}
        (Value::String(actual),Value::String(expected)) => {assert_eq!(actual,expected, "\nmismatch at {}:\nactual:{}\nexpected:{}\n", path.join("."), actual, expected);}
        (Value::Number(actual),Value::Number(expected)) => {assert_eq!(actual,expected, "\nmismatch at {}:\nactual:{}\nexpected:{}\n", path.join("."), actual, expected);}

        (Value::Array(actual),Value::Array(expected)) => {
            if actual.len() != expected.len() {
                panic!("\nmismatched array lengths at at {}:\nactual:{}\nexpected:{}\n", path.join("."), actual.len(), expected.len());
            }
            for (i,(a,e)) in actual.iter().zip(expected.iter()).enumerate() {
                let mut new_path = path.clone();
                new_path.push(format!("{}",i));
                check_eq(a,e, new_path);
            }
        }
        (Value::Object(actual),Value::Object(expected)) => {

            // iterate over Expected keys only bc its fine to add NEW things to the spec in Actual, we just need to ensure everything from Expected is still there.
            for k in expected.keys() {
                if !actual.contains_key(k) {
                    panic!("\nmissing json map key at {}: {}\n", path.join("."), k);
                }

                let mut new_path = path.clone();
                new_path.push(k.clone());
                check_eq(&actual[k],&expected[k], new_path);
            }

        }
        _ => {
            panic!("mismatch between types in json at {}:\nactual:{}\nexpected:{}\n", path.join("."), actual, expected);
        }
    }
}

#[test]
fn nuts_bolts() {
    compare_out_jsons("data/cogsci/nuts-bolts.json", "data/expected_outputs/nuts-bolts-a3-i10.json", "-a3", 10, InputFormat::ProgramsList);
}