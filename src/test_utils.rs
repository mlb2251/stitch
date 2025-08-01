use serde_json::Value;

use crate::{multistep_compression, Input, MultistepCompressionConfig};


pub fn run_compression_testing(inputs: &Input, cfg: &MultistepCompressionConfig) -> Value {
    run_compression_testing_weighted(inputs, cfg, None)
}

pub fn run_compression_testing_weighted(inputs: &Input, cfg: &MultistepCompressionConfig, weights: Option<Vec<f32>>) -> Value {
    multistep_compression(
        &inputs.train_programs,
        inputs.tasks.clone(),
        weights,
        inputs.name_mapping.clone(),
        None,
        cfg,
    ).1
}
