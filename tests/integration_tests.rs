use stitch_core::*;
use clap::Parser;
use serde_json::{json,Value};


fn load_programs(file: &str, input_format: InputFormat) -> (Input,Vec<ExprOwned>) {
    let input_file = std::path::Path::new(file);
    let input = input_format.load_programs_and_tasks(input_file).unwrap();
    let train_programs: Vec<ExprOwned> = input.train_programs.iter().map(|p|{
        let mut set = ExprSet::empty(Order::ChildFirst, false, false);
        let idx = set.parse_extend(p).unwrap();
        ExprOwned::new(set,idx)
    }).collect();
    (input,train_programs)
}

fn out_json(train_programs: &[ExprOwned], step_results: &Vec<CompressionStepResult>, cost_fn: &ExprCost) -> serde_json::Value {
    json!({
        "cmd": Value::Null,
        "args": Value::Null,
        "original_cost": train_programs.iter().map(|p|p.cost(cost_fn)).sum::<i32>(),
        "original": train_programs.iter().map(|p| p.to_string()).collect::<Vec<String>>(),
        "invs": step_results.iter().map(|inv| inv.json()).collect::<Vec<serde_json::Value>>(),
    })
}

fn write_json_for_diff(out: &Value, expected_out_path: &str) {
    let path = format!("out/test_outputs/{}.json",timestamp());
    let out_path = std::path::Path::new(&path);
    if let Some(out_path_dir) = out_path.parent() {
        if !out_path_dir.exists() {
            std::fs::create_dir_all(out_path_dir).unwrap();
        }
    }
    std::fs::write(out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    println!("Wrote test output to {:?} diff with expected out path {:?}", out_path, expected_out_path);
}

fn run_compression(train_programs: &[ExprOwned], input: &Input, iterations: usize, args: &str, cost_fn: &ExprCost) -> Vec<CompressionStepResult> {
    compression(
        train_programs,
        &None,
        iterations,
        &CompressionStepConfig::parse_from(format!("compress {}",args).split_whitespace()),
        &input.tasks,
        &input.prev_dc_inv_to_inv_strs,
        cost_fn)
}

fn compare_out_jsons(file: &str, expected_out_file: &str, args: &str, iterations: usize, input_format: InputFormat) {
    let (input,train_programs) = load_programs(file, input_format);
    let cost_fn = ExprCost::dreamcoder();

    let step_results = run_compression(&train_programs, &input, iterations, args, &cost_fn);

    let output: Value = out_json(&train_programs, &step_results, &cost_fn);
    let expected_output: Value = serde_json::from_str(&std::fs::read_to_string(std::path::Path::new(expected_out_file)).unwrap()).unwrap();

    check_eq(&output["original"], &expected_output["original"], vec!["original".into()], &output, expected_out_file);
    check_eq(&output["original_cost"], &expected_output["original_cost"], vec!["original_cost".into()], &output, expected_out_file);
    check_eq(&output["invs"], &expected_output["invs"], vec!["invs".into()], &output, expected_out_file);

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
                new_path.push(format!("{}",i));
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



#[test]
fn simple1_a1_i1() {
    compare_out_jsons("data/basic/simple1.json", "data/expected_outputs/simple1-a1-i1.json", "-a1", 1, InputFormat::ProgramsList);
}

#[test]
fn simple2_a1_i1() {
    compare_out_jsons("data/basic/simple2.json", "data/expected_outputs/simple2-a1-i1.json", "-a1", 1, InputFormat::ProgramsList);
}

#[test]
fn nuts_bolts_a3_i10() {
    compare_out_jsons("data/cogsci/nuts-bolts.json", "data/expected_outputs/nuts-bolts-a3-i10.json", "-a3", 10, InputFormat::ProgramsList);
}
#[test]
fn furniture_a2_i10() {
    compare_out_jsons("data/cogsci/furniture.json", "data/expected_outputs/furniture-a2-i10.json", "-a2", 10, InputFormat::ProgramsList);
}
#[test]
fn wheels_a2_i10() {
    compare_out_jsons("data/cogsci/wheels.json", "data/expected_outputs/wheels-a2-i10.json", "-a2", 10, InputFormat::ProgramsList);
}

#[test]
fn dials_a2_i10() {
    compare_out_jsons("data/cogsci/dials.json", "data/expected_outputs/dials-a2-i10.json", "-a2", 10, InputFormat::ProgramsList);
}

#[test]
fn city_a1_i1() {
    compare_out_jsons("data/cogsci/city.json", "data/expected_outputs/city-a1-i1.json", "-a1", 1, InputFormat::ProgramsList);
}

#[test]
fn bridge_a2_i10() {
    compare_out_jsons("data/cogsci/bridge.json", "data/expected_outputs/bridge-a2-i10.json", "-a2", 10, InputFormat::ProgramsList);
}

#[test]
fn castle_a1_i1() {
    compare_out_jsons("data/cogsci/castle.json", "data/expected_outputs/castle-a1-i1.json", "-a1", 1, InputFormat::ProgramsList);
}

#[test]
fn house_a1_i1() {
    compare_out_jsons("data/cogsci/house.json", "data/expected_outputs/house-a1-i1.json", "-a1", 1, InputFormat::ProgramsList);
}

#[test]
fn logo_iteration_1_a3_i10() {
    compare_out_jsons("data/dc/logo_iteration_1.json", "data/expected_outputs/logo_iteration_1-a3-i10.json", "-a3", 10, InputFormat::Dreamcoder);
}

#[test]
fn origami_0_a3_i10() {
    compare_out_jsons("data/dc/origami/iteration_0_3.json", "data/expected_outputs/origami_0-a3-i10.json", "-a3", 10, InputFormat::Dreamcoder);
}

#[test]
fn origami_1_a3_i10() {
    compare_out_jsons("data/dc/origami/iteration_1_6.json", "data/expected_outputs/origami_1-a3-i10.json", "-a3", 10, InputFormat::Dreamcoder);
}

#[test]
fn origami_2_a3_i10() {
    compare_out_jsons("data/dc/origami/iteration_2_1.json", "data/expected_outputs/origami_2-a3-i10.json", "-a3", 10, InputFormat::Dreamcoder);
}

// todo disabled bc nondeterminism with 2 equal things on the first invention (usually threading prevents that, but here for some reason you always get the same result when running from commandline and a diff result when running from test)
// #[test]
// fn origami_3_a3_i10() {
//     compare_out_jsons("data/dc/origami/iteration_3_1.json", "data/expected_outputs/origami_3-a3-i10.json", "-a3", 10, InputFormat::Dreamcoder);
// }