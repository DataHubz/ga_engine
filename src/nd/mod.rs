// src/nd/mod.rs
pub mod ga;
pub mod gp;
pub mod multivector;
pub mod types;
pub mod vecn;

// re-exports for easy import:
pub use gp::{gp_table_2, gp_table_3, gp_table_4, make_gp_table};
pub use multivector::Multivector;
pub use vecn::VecN;
