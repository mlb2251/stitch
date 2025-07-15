use std::path::Path;

use stitch_core::*;
use clap::Parser;
use serde_json::Value;

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

fn compare_out_jsons(file: &str, expected_out_file: &str, args: &str, input_format: InputFormat) {
    let input = input_format.load_programs_and_tasks(std::path::Path::new(file)).unwrap();

    let mut cfg = MultistepCompressionConfig::parse_from(format!("compress {args}").split_whitespace());

    cfg.previous_abstractions = input.name_mapping.clone().unwrap_or_default().len();

    let output = run_compression_testing(&input, &cfg);

    println!("{}", serde_json::to_string(&output).unwrap());

    let expected_output: Value = serde_json::from_str(&std::fs::read_to_string(std::path::Path::new(expected_out_file)).unwrap_or("{}".to_owned())).unwrap();

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

#[test]
fn simple1_a1_i1() {
    compare_out_jsons("data/basic/simple1.json", "data/expected_outputs/simple1-a1-i1.json", "-i1 -a1 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn simple2_a1_i1() {
    compare_out_jsons("data/basic/simple2.json", "data/expected_outputs/simple2-a1-i1.json", "-i1 -a1 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn simple3_a1_i1() {
    compare_out_jsons("data/basic/simple3.json", "data/expected_outputs/simple3-a1-i1.json", "-i1 -a1 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn simple4_a1_i1() {
    compare_out_jsons("data/basic/simple4.json", "data/expected_outputs/simple4-a1-i1.json", "-i1 -a1 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn simple5_a1_i1() {
    compare_out_jsons("data/basic/simple5.json", "data/expected_outputs/simple5-a1-i1.json", "-i1 -a1 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn nuts_bolts_a3_i10() {
    compare_out_jsons("data/cogsci/nuts-bolts.json", "data/expected_outputs/nuts-bolts-a3-i10.json", "-i10 -a3 --rewrite-check", InputFormat::ProgramsList);
}
#[test]
fn furniture_a2_i10() {
    compare_out_jsons("data/cogsci/furniture.json", "data/expected_outputs/furniture-a2-i10.json", "-i10 -a2 --rewrite-check", InputFormat::ProgramsList);
}
#[test]
fn wheels_a2_i10() {
    compare_out_jsons("data/cogsci/wheels.json", "data/expected_outputs/wheels-a2-i10.json", "-i10 -a2 --rewrite-check",  InputFormat::ProgramsList);
}

#[test]
fn dials_a2_i10() {
    compare_out_jsons("data/cogsci/dials.json", "data/expected_outputs/dials-a2-i10.json", "-i10 -a2 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn city_a1_i1() {
    compare_out_jsons("data/cogsci/city.json", "data/expected_outputs/city-a1-i1.json", "-i1 -a1 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn bridge_a2_i10() {
    compare_out_jsons("data/cogsci/bridge.json", "data/expected_outputs/bridge-a2-i10.json", "-i10 -a2 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn castle_a1_i1() {
    compare_out_jsons("data/cogsci/castle.json", "data/expected_outputs/castle-a1-i1.json", "-i1 -a1 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn house_a1_i1() {
    compare_out_jsons("data/cogsci/house.json", "data/expected_outputs/house-a1-i1.json", "-i1 -a1 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn logo_iteration_1_a3_i10() {
    compare_out_jsons("data/dc/logo_iteration_1.json", "data/expected_outputs/logo_iteration_1-a3-i10.json", "-i10 -a3 --rewrite-check", InputFormat::Dreamcoder);
}

#[test]
fn origami_0_a3_i10() {
    compare_out_jsons("data/dc/origami/iteration_0_3.json", "data/expected_outputs/origami_0-a3-i10.json", "-i10 -a3 --rewrite-check", InputFormat::Dreamcoder);
}

#[test]
fn origami_1_a3_i10() {
    compare_out_jsons("data/dc/origami/iteration_1_6.json", "data/expected_outputs/origami_1-a3-i10.json", "-i10 -a3 --rewrite-check", InputFormat::Dreamcoder);
}

#[test]
fn origami_2_a3_i10() {
    compare_out_jsons("data/dc/origami/iteration_2_1.json", "data/expected_outputs/origami_2-a3-i10.json", "-i10 -a3 --rewrite-check", InputFormat::Dreamcoder);
}

#[test]
fn neurosym_match_at_tag() {
    compare_out_jsons("data/neurosym/match_at_tag.json", "data/expected_outputs/neurosym_match_at_tag.json", "", InputFormat::ProgramsList);
    compare_out_jsons("data/neurosym/match_at_tag.json", "data/expected_outputs/neurosym_match_at_tag_excluded.json", "--fused-lambda-tags 2", InputFormat::ProgramsList);
}

#[test]
fn neurosym_metavariable_with_tag() {
    compare_out_jsons("data/neurosym/metavariable_with_tag.json", "data/expected_outputs/neurosym_metavariable_with_tag.json", "", InputFormat::ProgramsList);
    compare_out_jsons("data/neurosym/metavariable_with_tag.json", "data/expected_outputs/neurosym_metavariable_with_tag_excluded.json", "--fused-lambda-tags 2", InputFormat::ProgramsList);
}

#[test]
fn symbol_weighting_test_higher_weight() {
    compare_out_jsons("data/basic/symbol_weighting_test_1.json", "data/expected_outputs/symbol_weighting_1_default.json", "-i1 -a3", InputFormat::ProgramsList);
    compare_out_jsons("data/basic/symbol_weighting_test_1.json", "data/expected_outputs/symbol_weighting_1_h_200.json", "-i1 -a3 --cost-prim {\"H\":200}", InputFormat::ProgramsList);
    compare_out_jsons("data/basic/symbol_weighting_test_1.json", "data/expected_outputs/symbol_weighting_1_h_202.json", "-i1 -a3 --cost-prim {\"H\":202}", InputFormat::ProgramsList);
}

#[test]
fn symbol_weighting_test_lower_weight() {
    compare_out_jsons("data/basic/symbol_weighting_test_2.json", "data/expected_outputs/symbol_weighting_2_default.json", "-i1 -a3", InputFormat::ProgramsList);
    // l1,l2,l3 all should have value 60 in the following test
    compare_out_jsons("data/basic/symbol_weighting_test_2.json", "data/expected_outputs/symbol_weighting_2_l_60.json", "-i1 -a3 --cost-prim {\"L1\":60,\"L2\":60,\"L3\":60}", InputFormat::ProgramsList);
    compare_out_jsons("data/basic/symbol_weighting_test_2.json", "data/expected_outputs/symbol_weighting_2_l_67.json", "-i1 -a3 --cost-prim {\"L1\":67,\"L2\":67,\"L3\":67}", InputFormat::ProgramsList);
}

const DFA_ARGS: &str = r#"--tdfa-json-path test_data/dfa.json --tdfa-root M --valid-metavars ["S","E","seqS"] --valid-roots ["S","E","seqS"] --tdfa-non-eta-long-states {"seqS":"S"}  --tdfa-split ~"#;

#[test]
fn tdfa_multi_arg_function() {
    compare_out_jsons("data/python/multi-arg-function.json", "data/expected_outputs/multi-arg-function-basic.json", "-i2 -a3", InputFormat::ProgramsList);
    compare_out_jsons("data/python/multi-arg-function.json", "data/expected_outputs/multi-arg-function-with-dfa.json", &("-i2 -a3 ".to_owned() + DFA_ARGS), InputFormat::ProgramsList);
}

#[test]
fn tdfa_sequence() {
    compare_out_jsons("data/python/front-of-sequence.json", "data/expected_outputs/front-of-sequence.json", &("-i2 -a3 ".to_owned() + DFA_ARGS), InputFormat::ProgramsList);
    compare_out_jsons("data/python/back-of-sequence.json", "data/expected_outputs/back-of-sequence.json", &("-i2 -a3 ".to_owned() + DFA_ARGS), InputFormat::ProgramsList);
}

#[test]
fn smc_regression_tests() {
    let args = "-i10 --smc --smc-particles 1000 --smc-extra-steps 40".to_owned();
    compare_out_jsons("data/cogsci/nuts-bolts.json", "data/expected_outputs/smc-nuts-bolts.json", &args, InputFormat::ProgramsList);
    compare_out_jsons("data/cogsci/wheels.json", "data/expected_outputs/smc-wheels.json", &args, InputFormat::ProgramsList);
    compare_out_jsons("data/cogsci/furniture.json", "data/expected_outputs/smc-furniture.json", &args, InputFormat::ProgramsList);
    compare_out_jsons("data/cogsci/dials.json", "data/expected_outputs/smc-dials.json", &args, InputFormat::ProgramsList);
    // compare_out_jsons("data/cogsci/city.json", "data/expected_outputs/smc-city.json", &args, InputFormat::ProgramsList);
}

#[test]
#[should_panic(expected = "Inconsistent symbols: \"NameStr\" and \"Name\" for expr &os:0")]
fn symbols_basic_inconsistent_symbols() {
    compare_out_jsons("data/python/non-working-import-and-number-in-same-spot.json", "data/expected_outputs/non-working-import-and-number-in-same-spot.json", &("-i3 -a0 ".to_owned() + DFA_ARGS), InputFormat::ProgramsList);
}


// todo disabled bc nondeterminism with 2 equal things on the first invention (usually threading prevents that, but here for some reason you always get the same result when running from commandline and a diff result when running from test)
// #[test]
// fn origami_3_a3_i10() {
//     compare_out_jsons("data/dc/origami/iteration_3_1.json", "data/expected_outputs/origami_3-a3-i10.json", "-a3", 10, InputFormat::Dreamcoder);
// }
