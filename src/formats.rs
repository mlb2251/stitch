use std::io::Error;
use std::path::Path;
use std::fs::File;
use clap::ArgEnum;
use serde::Serialize;
use serde_json::de::from_reader;

#[derive(Debug, Clone, ArgEnum, Serialize)]
pub enum InputFormat {
    Dreamcoder,
    ProgramsList
}

impl InputFormat {
    pub fn load_programs(&self, path: &Path) -> Result<Vec<String>, String> {
        match self {
            &InputFormat::Dreamcoder => {
                // read dreamcoder format
                let json: serde_json::Value = from_reader(File::open(path).expect("file not found")).expect("json deserializing error");
                let mut programs: Vec<String> = json["frontiers"].as_array().unwrap_or_else(||panic!("json parse error, are you sure you wanted format {:?}?", self)).iter().map(|f| f["programs"].as_array().unwrap().iter().map(|p|p["program"].as_str().unwrap().to_string())).flatten().collect();
                programs = programs.iter().map(|p| p.replace("(lambda ","(lam ")).collect();
                Ok(programs)
            }
            &InputFormat::ProgramsList => {
                from_reader(File::open(path).map_err(|e| format!("file not found, error code {:?}", e))?).map_err(|e| format!("json parser error, are you sure you wanted format {:?}? Error code was {:?}", self, e))
            }
        }
    }
}