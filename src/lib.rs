//! tensor-forge.rs
//!
//! A minimal compute graph runtime in Rust for executing tensor operations using a pluggable kernel registry and deterministic execution engine.
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![deny(missing_docs)]

pub mod graph;
pub mod kernel;
pub mod node;
pub mod op;
pub mod registry;
pub mod tensor;
