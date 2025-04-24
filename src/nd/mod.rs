// src/nd/mod.rs
pub mod types;
pub mod vecn;
pub mod multivector;
pub mod gp;
pub mod ga;

// re-exports for easy import:
pub use vecn::VecN;
pub use multivector::Multivector;
pub use gp::{ make_gp_table, gp_table_2, gp_table_3, gp_table_4 };
