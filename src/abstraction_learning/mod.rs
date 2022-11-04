pub mod compression;
pub mod rewriting;
pub mod egraphs;
pub mod util;
pub mod formats;

pub use {
    egg::*,
    compression::*,
    rewriting::*,
    egraphs::*,
    util::*,
    formats::*,
};

pub use colorful::{Color,Colorful,RGB};

/// nonterminals ("app" and "lam") cost 1/100th of a terminal ("var", "ivar", "prim"). This is because nonterminals
/// can be autofilled based on the type of the hole you're filling during most search methods.
pub const COST_NONTERMINAL: i32 = 1;
pub const COST_TERMINAL: i32 = 100;