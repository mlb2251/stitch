use serde_json::Value;

use crate::{multistep_compression, Input, MultistepCompressionConfig};


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
