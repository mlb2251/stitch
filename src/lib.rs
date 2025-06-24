
pub mod compression;
pub mod rewriting;
pub mod egraphs;
pub mod util;
pub mod formats;
pub mod smc;
pub mod expand_variable;
pub mod tdfa;
pub mod test_utils;

pub use {
    compression::*,
    rewriting::*,
    egraphs::*,
    util::*,
    formats::*,
    lambdas::*,
    smc::*,
    expand_variable::*,
    tdfa::*,
    test_utils::*,
};

pub use colorful::{Color,Colorful,RGB};















