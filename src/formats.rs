use std::io::Error;
use std::path::Path;
use std::fs::File;
use clap::ArgEnum;
use serde::Serialize;
use serde_json::de::from_reader;

pub trait InputLoader {
    fn load_program(&self, path: &Path) -> Result<Vec<String>, Error>;
}

#[derive(Debug, Clone, ArgEnum, Serialize)]
pub enum InputFormat {
    Dreamcoder,
    Default
}

impl InputLoader for InputFormat {
    fn load_program(&self, path: &Path) -> Result<Vec<String>, Error> {
        match self {
            &InputFormat::Dreamcoder => {
                // read dreamcoder format
                let json: serde_json::Value = from_reader(File::open(path).expect("file not found")).expect("json deserializing error");
                let mut programs: Vec<String> = json["frontiers"].as_array().unwrap_or_else(||panic!("json parse error, are you sure you wanted format {:?}?", self)).iter().map(|f| f["programs"].as_array().unwrap().iter().map(|p|p["program"].as_str().unwrap().to_string())).flatten().collect();
                programs = programs.iter().map(|p| p.replace("(lambda ","(lam ")).collect();
                Ok(programs)
            }
            &InputFormat::Default => {
                Ok(from_reader(File::open(path).expect("file not found")).unwrap_or_else(|_|panic!("json parse error, are you sure you wanted format {:?}?", self)))
            }
        }
    }
}