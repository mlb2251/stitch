use std::collections::HashMap;

use lambdas::{ExprSet, Order};
use stitch_core::TDFA;


fn assert_tdfa(json: String, root: String, valid_metavar: String, eta_long: (String, String), code: &'static str, split: Option<String>, mut expected: Vec<&'static str>) {
    let eta_long_map: HashMap<String, String> = HashMap::from([(eta_long.0, eta_long.1)]);
    let tdfa: TDFA = TDFA::new(root, json, vec![valid_metavar], vec![], eta_long_map, vec![], split);
    println!("TDFA created with valid metavars: {:?}", tdfa);
    let mut set = ExprSet::empty(Order::ChildFirst, false, false);
    let node = set.parse_extend(code).unwrap();
    
    
    let result = tdfa.annotate(&set, &[node], &None);
    
    let mut result_str = result.iter()
        .map(|(k, v)| format!("{}: {}", set.get(*k), v))
        .collect::<Vec<_>>();
    println!("Result: {:?}", result_str);
    result_str.sort();

    
    expected.sort();
    
    assert_eq!(
        result_str,
        expected
    );
}


#[test]
fn test_basic_tdfa() {
    let json = r#"
    {
        "T": {
            "Assign": ["V", "E"]
        },
        "E": {
            "list": ["E"]
        }
    }
    "#.to_string();
    let expected = vec![
            "(Assign x (list 3 4 5)): T",
            "x: V",
            "(list 3 4 5): E",
            "3: E",
            "4: E",
            "5: E"
        ];
    assert_tdfa(json,  "T".to_string(),  "E".to_string(), ("listV".to_string(), "V".to_string()), "(Assign x (list 3 4 5))", None, expected);
}

#[test]
fn test_basic_with_some_annotations_tdfa() {
    let json = r#"
    {
        "T": {
            "Assign": ["V", "E"]
        },
        "E": {
            "list": ["E"]
        }
    }
    "#.to_string();
    let expected = vec![
            "(Assign x (list~E 3 4 5)): T",
            "x: V",
            "(list~E 3 4 5): E",
            "3: E",
            "4: E",
            "5: E"
        ];
    assert_tdfa(json,  "T".to_string(),  "E".to_string(), ("listV".to_string(), "V".to_string()), "(Assign x (list~E 3 4 5))", Some("~".to_owned()), expected);
}

#[test]
fn test_non_eta_long() {
    let json = r#"
    {
        "T": {
            "Assign": ["listV", "E"]
        },
        "listV": {
            "vars": []
        },
        "E": {
            "list": ["E"]
        }
    }
    "#.to_string();
    let expected = vec![
            "(Assign (vars x y z) (list 3 4 5)): T",
            "(vars x y z): listV",
            "(vars x y): listV",
            "(vars x): listV",
            "vars: listV",
            "x: V",
            "y: V",
            "z: V",
            "(list 3 4 5): E",
            "3: E",
            "4: E",
            "5: E"
        ];
    assert_tdfa(json,  "T".to_string(),  "E".to_string(), ("listV".to_string(), "V".to_string()), "(Assign (vars x y z) (list 3 4 5))", None, expected);
}

#[test]
fn test_non_eta_long_with_args() {
    let json = r#"
    {
        "T": {
            "Assign": ["listV", "E"]
        },
        "listV": {
            "vars": [],
            "fn_1": ["A"]
        },
        "E": {
            "list": ["E"]
        }
    }
    "#.to_string();
    let expected = vec![
            "(Assign (fn_1 x y z) (list 3 4 5)): T",
            "(fn_1 x y z): listV",
            "(fn_1 x y): listV",
            "(fn_1 x): listV",
            "x: A",
            "y: V",
            "z: V",
            "(list 3 4 5): E",
            "3: E",
            "4: E",
            "5: E"
        ];
    assert_tdfa(json,  "T".to_string(),  "E".to_string(), ("listV".to_string(), "V".to_string()), "(Assign (fn_1 x y z) (list 3 4 5))", None, expected);
}
