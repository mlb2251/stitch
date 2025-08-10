use serde_json::Value;
use clap::Parser;
use std::path::Path;

use crate::{multistep_compression, Input, MultistepCompressionConfig, InputFormat};
use crate::util::timestamp;

pub fn run_compression_testing(inputs: &Input, cfg: &MultistepCompressionConfig) -> Value {
    multistep_compression(
        &inputs.train_programs,
        inputs.tasks.clone(),
        None,
        inputs.name_mapping.clone(),
        None,
        cfg,
        ).1
}


fn write_json_for_diff(out: &Value, expected_out_path: &str) {
    let path = format!("out/test_outputs/{}_{}.json",timestamp(), Path::new(expected_out_path).file_stem().unwrap().to_str().unwrap());
    let out_path = std::path::Path::new(&path);
    if let Some(out_path_dir) = out_path.parent() {
        if !out_path_dir.exists() {
            std::fs::create_dir_all(out_path_dir).unwrap();
        }
    }
    std::fs::write(out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("Wrote test output to {out_path:?} diff with expected out path {expected_out_path:?}");
    println!("Command to replace:");
    println!("cp {out_path:?} {expected_out_path:?}");
}

pub fn compare_out_jsons_testing(file: &str, expected_out_file: &str, args: &str, input_format: InputFormat) {
    let input = input_format.load_programs_and_tasks(std::path::Path::new(file)).unwrap();

    let mut cfg = MultistepCompressionConfig::parse_from(format!("compress {args}").split_whitespace());

    cfg.previous_abstractions = input.name_mapping.clone().unwrap_or_default().len();

    let output = run_compression_testing(&input, &cfg);

    println!("{}", serde_json::to_string(&output).unwrap());

    let expected_output: Value = serde_json::from_str(&std::fs::read_to_string(std::path::Path::new(expected_out_file)).unwrap()).unwrap();

    check_eq(&output["original"], &expected_output["original"], vec!["original".into()], &output, expected_out_file);
    check_eq(&output["original_cost"], &expected_output["original_cost"], vec!["original_cost".into()], &output, expected_out_file);
    check_eq(&output["final_cost"], &expected_output["final_cost"], vec!["final_cost".into()], &output, expected_out_file);
    check_eq(&output["compression_ratio"], &expected_output["compression_ratio"], vec!["compression_ratio".into()], &output, expected_out_file);
    check_eq(&output["num_abstractions"], &expected_output["num_abstractions"], vec!["num_abstractions".into()], &output, expected_out_file);
    check_eq(&output["abstractions"], &expected_output["abstractions"], vec!["abstractions".into()], &output, expected_out_file);
    check_eq(&output["rewritten"], &expected_output["rewritten"], vec!["rewritten".into()], &output, expected_out_file);

}

//todo add write_json_for_diff calls and also make it add a random suffix too

fn check_eq(actual: &Value, expected: &Value, path: Vec<String>, out: &Value, expected_out_path: &str) {
    match (actual,expected) {
        (Value::Null,Value::Null) => {}
        (Value::Bool(actual),Value::Bool(expected)) => {
            if actual != expected {
                write_json_for_diff(out, expected_out_path);
                panic!("\nmismatch at {}:\nactual:  {}\nexpected:{}\n", path.join("."), actual, expected);
            }
        }
        (Value::String(actual),Value::String(expected)) => {
            if actual != expected {
                write_json_for_diff(out, expected_out_path);
                panic!("\nmismatch at {}:\nactual:  {}\nexpected:{}\n", path.join("."), actual, expected);
            }
        }
        (Value::Number(actual),Value::Number(expected)) => {
            if actual.is_f64() {
                if (actual.as_f64().unwrap() - expected.as_f64().unwrap()).abs() > 1e-2  {
                    write_json_for_diff(out, expected_out_path);
                    panic!("\nmismatch at {}:\nactual:  {}\nexpected:{}\n", path.join("."), actual, expected);
                }
            } else if actual != expected {
                write_json_for_diff(out, expected_out_path);
                panic!("\nmismatch at {}:\nactual:  {}\nexpected:{}\n", path.join("."), actual, expected);
            }
        }

        (Value::Array(actual),Value::Array(expected)) => {
            if actual.len() != expected.len() {
                write_json_for_diff(out, expected_out_path);
                panic!("\nmismatched array lengths at at {}:\nactual:  {}\nexpected:{}\n", path.join("."), actual.len(), expected.len());
            }
            for (i,(a,e)) in actual.iter().zip(expected.iter()).enumerate() {
                let mut new_path = path.clone();
                new_path.push(format!("{i}"));
                check_eq(a,e, new_path, out, expected_out_path);
            }
        }
        (Value::Object(actual),Value::Object(expected)) => {

            // iterate over Expected keys only bc its fine to add NEW things to the spec in Actual, we just need to ensure everything from Expected is still there.
            for k in expected.keys() {
                if !actual.contains_key(k) {
                    write_json_for_diff(out, expected_out_path);
                    panic!("\nmissing json map key at {}: {}\n", path.join("."), k);
                }

                let mut new_path = path.clone();
                new_path.push(k.clone());
                check_eq(&actual[k],&expected[k], new_path, out, expected_out_path);
            }

        }
        _ => {
            write_json_for_diff(out, expected_out_path);
            panic!("mismatch between types in json at {}:\nactual:  {}\nexpected:{}\n", path.join("."), actual, expected);
        }
    }
}
