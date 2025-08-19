use stitch_core::*;


#[test]
fn test_do_not_clobber() {
    assert!(!CLOBBER, "CLOBBER is true, so the tests will clobber the expected outputs. This should not be committed.");
}

#[test]
fn simple1_a1_i1() {
    compare_out_jsons_testing("data/basic/simple1.json", "data/expected_outputs/simple1-a1-i1.json", "-i1 -a1 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn simple2_a1_i1() {
    compare_out_jsons_testing("data/basic/simple2.json", "data/expected_outputs/simple2-a1-i1.json", "-i1 -a1 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn simple3_a1_i1() {
    compare_out_jsons_testing("data/basic/simple3.json", "data/expected_outputs/simple3-a1-i1.json", "-i1 -a1 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn simple4_a1_i1() {
    compare_out_jsons_testing("data/basic/simple4.json", "data/expected_outputs/simple4-a1-i1.json", "-i1 -a1 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn simple5_a1_i1() {
    compare_out_jsons_testing("data/basic/simple5.json", "data/expected_outputs/simple5-a1-i1.json", "-i1 -a1 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn nuts_bolts_a3_i10() {
    compare_out_jsons_testing("data/cogsci/nuts-bolts.json", "data/expected_outputs/nuts-bolts-a3-i10.json", "-i10 -a3 --rewrite-check", InputFormat::ProgramsList);
}
#[test]
fn furniture_a2_i10() {
    compare_out_jsons_testing("data/cogsci/furniture.json", "data/expected_outputs/furniture-a2-i10.json", "-i10 -a2 --rewrite-check", InputFormat::ProgramsList);
}
#[test]
fn wheels_a2_i10() {
    compare_out_jsons_testing("data/cogsci/wheels.json", "data/expected_outputs/wheels-a2-i10.json", "-i10 -a2 --rewrite-check",  InputFormat::ProgramsList);
}

#[test]
fn dials_a2_i10() {
    compare_out_jsons_testing("data/cogsci/dials.json", "data/expected_outputs/dials-a2-i10.json", "-i10 -a2 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn city_a1_i1() {
    compare_out_jsons_testing("data/cogsci/city.json", "data/expected_outputs/city-a1-i1.json", "-i1 -a1 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn bridge_a2_i10() {
    compare_out_jsons_testing("data/cogsci/bridge.json", "data/expected_outputs/bridge-a2-i10.json", "-i10 -a2 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn castle_a1_i1() {
    compare_out_jsons_testing("data/cogsci/castle.json", "data/expected_outputs/castle-a1-i1.json", "-i1 -a1 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn house_a1_i1() {
    compare_out_jsons_testing("data/cogsci/house.json", "data/expected_outputs/house-a1-i1.json", "-i1 -a1 --rewrite-check", InputFormat::ProgramsList);
}

#[test]
fn logo_iteration_1_a3_i10() {
    compare_out_jsons_testing("data/dc/logo_iteration_1.json", "data/expected_outputs/logo_iteration_1-a3-i10.json", "-i10 -a3 --rewrite-check", InputFormat::Dreamcoder);
}

#[test]
fn origami_0_a3_i10() {
    compare_out_jsons_testing("data/dc/origami/iteration_0_3.json", "data/expected_outputs/origami_0-a3-i10.json", "-i10 -a3 --rewrite-check", InputFormat::Dreamcoder);
}

#[test]
fn origami_1_a3_i10() {
    compare_out_jsons_testing("data/dc/origami/iteration_1_6.json", "data/expected_outputs/origami_1-a3-i10.json", "-i10 -a3 --rewrite-check", InputFormat::Dreamcoder);
}

#[test]
fn origami_2_a3_i10() {
    compare_out_jsons_testing("data/dc/origami/iteration_2_1.json", "data/expected_outputs/origami_2-a3-i10.json", "-i10 -a3 --rewrite-check", InputFormat::Dreamcoder);
}

#[test]
fn neurosym_match_at_tag() {
    compare_out_jsons_testing("data/neurosym/match_at_tag.json", "data/expected_outputs/neurosym_match_at_tag.json", "", InputFormat::ProgramsList);
    compare_out_jsons_testing("data/neurosym/match_at_tag.json", "data/expected_outputs/neurosym_match_at_tag_excluded.json", "--fused-lambda-tags 2", InputFormat::ProgramsList);
}

#[test]
fn neurosym_metavariable_with_tag() {
    compare_out_jsons_testing("data/neurosym/metavariable_with_tag.json", "data/expected_outputs/neurosym_metavariable_with_tag.json", "", InputFormat::ProgramsList);
    compare_out_jsons_testing("data/neurosym/metavariable_with_tag.json", "data/expected_outputs/neurosym_metavariable_with_tag_excluded.json", "--fused-lambda-tags 2", InputFormat::ProgramsList);
}

#[test]
fn symbol_weighting_test_higher_weight() {
    compare_out_jsons_testing("data/basic/symbol_weighting_test_1.json", "data/expected_outputs/symbol_weighting_1_default.json", "-i1 -a3", InputFormat::ProgramsList);
    compare_out_jsons_testing("data/basic/symbol_weighting_test_1.json", "data/expected_outputs/symbol_weighting_1_h_200.json", "-i1 -a3 --cost-prim '{\"H\":200}'", InputFormat::ProgramsList);
    compare_out_jsons_testing("data/basic/symbol_weighting_test_1.json", "data/expected_outputs/symbol_weighting_1_h_202.json", "-i1 -a3 --cost-prim '{\"H\":202}'", InputFormat::ProgramsList);
}

#[test]
fn symbol_weighting_test_lower_weight() {
    compare_out_jsons_testing("data/basic/symbol_weighting_test_2.json", "data/expected_outputs/symbol_weighting_2_default.json", "-i1 -a3", InputFormat::ProgramsList);
    // l1,l2,l3 all should have value 60 in the following test
    compare_out_jsons_testing("data/basic/symbol_weighting_test_2.json", "data/expected_outputs/symbol_weighting_2_l_60.json", "-i1 -a3 --cost-prim '{\"L1\":60,\"L2\":60,\"L3\":60}'", InputFormat::ProgramsList);
    compare_out_jsons_testing("data/basic/symbol_weighting_test_2.json", "data/expected_outputs/symbol_weighting_2_l_67.json", "-i1 -a3 --cost-prim '{\"L1\":67,\"L2\":67,\"L3\":67}'", InputFormat::ProgramsList);
}

const DFA_ARGS: &str = r#"--tdfa-json-path test_data/dfa.json --tdfa-root M --valid-metavars '["S","E","seqS"]' --valid-roots '["S","E","seqS"]' --tdfa-non-eta-long-states '{"seqS":"S"}'  --tdfa-split ~"#;

#[test]
fn tdfa_multi_arg_function() {
    compare_out_jsons_testing("data/python/multi-arg-function.json", "data/expected_outputs/multi-arg-function-basic.json", "-i2 -a3", InputFormat::ProgramsList);
    compare_out_jsons_testing("data/python/multi-arg-function.json", "data/expected_outputs/multi-arg-function-with-dfa.json", &("-i2 -a3 ".to_owned() + DFA_ARGS), InputFormat::ProgramsList);
}

#[test]
fn tdfa_sequence() {
    compare_out_jsons_testing("data/python/front-of-sequence.json", "data/expected_outputs/front-of-sequence.json", &("-i2 -a3 ".to_owned() + DFA_ARGS), InputFormat::ProgramsList);
    compare_out_jsons_testing("data/python/back-of-sequence.json", "data/expected_outputs/back-of-sequence.json", &("-i2 -a3 ".to_owned() + DFA_ARGS), InputFormat::ProgramsList);
}

fn python_args() -> String {
    DFA_ARGS.to_owned() + " --symvar-prefix &"
}

#[test]
fn python_symbols_regression() {
    compare_out_jsons_testing("data/python/10.json", "data/expected_outputs/10.json", &("-i10 -a2 --symvar-prefix & ".to_owned() + DFA_ARGS), InputFormat::ProgramsList);
}


#[test]
fn symbols_basic() {
    compare_out_jsons_testing("data/python/symbols-alignment.json", "data/expected_outputs/symbols-alignment.json", "-i2 -a3 --symvar-prefix & ", InputFormat::ProgramsList);
    compare_out_jsons_testing("data/python/symbols-cannot-be-literal.json", "data/expected_outputs/symbols-cannot-be-literal.json", "-i2 -a3 --symvar-prefix & ", InputFormat::ProgramsList);
    compare_out_jsons_testing("data/python/symbols-cannot-be-literal-0-arity.json", "data/expected_outputs/symbols-cannot-be-literal-0-arity.json", "-i2 -a3 --symvar-prefix & ", InputFormat::ProgramsList);
    compare_out_jsons_testing("data/python/symbol-reuse.json", "data/expected_outputs/symbol-reuse.json", "-i1 -a0 --symvar-prefix & ", InputFormat::ProgramsList);
    compare_out_jsons_testing("data/python/symbol-reuse-dfa.json", "data/expected_outputs/symbol-reuse-dfa.json", &("-i1 -a0 ".to_owned() + &python_args()), InputFormat::ProgramsList);
    compare_out_jsons_testing("data/python/pick-up-on-abstractions-0-arity.json", "data/expected_outputs/pick-up-on-abstractions-0-arity.json", &("-i1 -a0 ".to_owned() + &python_args()), InputFormat::ProgramsList);
}

#[test]
#[should_panic(expected = "Inconsistent symbols: \"NameStr\" and \"Name\" for expr &os:0")]
fn symbols_basic_inconsistent_symbols() {
    compare_out_jsons_testing("data/python/non-working-import-and-number-in-same-spot.json", "data/expected_outputs/non-working-import-and-number-in-same-spot.json", &("-i3 -a0 ".to_owned() + &python_args()), InputFormat::ProgramsList);
}


// todo disabled bc nondeterminism with 2 equal things on the first invention (usually threading prevents that, but here for some reason you always get the same result when running from commandline and a diff result when running from test)
// #[test]
// fn origami_3_a3_i10() {
//     compare_out_jsons("data/dc/origami/iteration_3_1.json", "data/expected_outputs/origami_3-a3-i10.json", "-a3", 10, InputFormat::Dreamcoder);
// }
