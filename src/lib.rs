

#[macro_use]
mod macros;

mod domain;
mod expr;
mod run_with_timeout;
mod compression;
mod util;
mod extraction;
mod egraphs;

pub use {
    egg::*,
    util::*,
    expr::*,
    compression::*,
    domain::*,
    macros::*,
    extraction::*,
    egraphs::{*,EGraph},
};

pub use colorful::{Color,Colorful,RGB};

pub mod domains;

/// nonterminals ("app" and "lam") cost 1/100th of a terminal ("var", "ivar", "prim"). This is because nonterminals
/// can be autofilled based on the type of the hole you're filling during most search methods.
pub const COST_NONTERMINAL: i32 = 1;
pub const COST_TERMINAL: i32 = 100;

















