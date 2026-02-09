//! tensor-forge.rs
//!
//! A minimal compute graph runtime in Rust for executing tensor operations using a pluggable kernel registry and deterministic execution engine.
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![deny(missing_docs)]

pub mod tensor;
pub mod node;
pub mod op;
