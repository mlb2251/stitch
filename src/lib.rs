
pub mod compression;
pub mod rewriting;
pub mod egraphs;
pub mod util;
pub mod formats;
pub mod smc;
pub mod expand_variable;

pub use {
    compression::*,
    rewriting::*,
    egraphs::*,
    util::*,
    formats::*,
    lambdas::*,
    smc::*,
    expand_variable::*,
};

pub use colorful::{Color,Colorful,RGB};















