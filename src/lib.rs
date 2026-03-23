//! `tensor_forge` is a minimal compute graph runtime for tensor operations.
//!
//! The crate provides:
//! - [`Graph`], a directed acyclic compute graph of tensor operations,
//! - [`Executor`], a deterministic execution engine for evaluating graphs,
//! - [`KernelRegistry`], a pluggable registry mapping [`OpKind`] values to kernels,
//! - [`Tensor`], the runtime tensor value type used for inputs and outputs.
//!
//! Graphs are constructed by adding input and operation nodes, marking one or more
//! nodes as outputs, and then executing the graph with runtime input bindings.
//!
//! # Core workflow
//!
//! A typical workflow is:
//! 1. create a [`Graph`],
//! 2. add input and operation nodes,
//! 3. mark output nodes,
//! 4. construct an [`Executor`] with a [`KernelRegistry`],
//! 5. execute the graph with `(NodeId, Tensor)` input bindings.
//!
//! # Examples
//!
//! ```
//! use tensor_forge::{Executor, Graph, KernelRegistry, Tensor};
//!
//! let mut g = Graph::new();
//! let a = g.input_node(vec![2, 2]);
//! let b = g.input_node(vec![2, 2]);
//! let out = g.add(a, b).expect("Valid add operation should succeed");
//! g.set_output_node(out)
//!     .expect("Setting output node should succeed");
//!
//! let a_tensor = Tensor::zeros(vec![2, 2]).expect("Tensor allocation should succeed");
//! let b_tensor = Tensor::zeros(vec![2, 2]).expect("Tensor allocation should succeed");
//! let expected = Tensor::from_vec(vec![2, 2], vec![0_f64, 0_f64, 0_f64, 0_f64]).expect("Tensor allocation should
//! succeed");
//!
//! let exec = Executor::new(KernelRegistry::default()); // imports default kernel operation mappings
//! let outputs = exec
//!     .execute(&g, vec![(a, a_tensor), (b, b_tensor)])
//!     .expect("Execution should succeed");
//!
//! // `outputs` now contains the resulting tensors of nodes marked as outputs
//! assert!(outputs.contains_key(&out));
//!
//! let output_tensor: &Tensor = &outputs[&out];
//! assert_eq!(output_tensor.shape(), expected.shape());
//! assert_eq!(output_tensor.data().len(), expected.data().len());
//! assert_eq!(output_tensor.data(), expected.data());
//! ```
//! See the `examples/` directory for larger runnable examples, including:
//! - `add_graph.rs`              # introductory example
//! - `branching_graph.rs`        # complex chain example with `Add`, `ReLU`, `MatMul`
//! - `feedforward_neural_net.rs` # programmatic neural network generation
//! - `custom_kernel.rs`          # defining custom kernels
//!
//! # Module overview
//!
//! - [`executor`] contains graph execution and execution-time errors.
//! - [`graph`] contains graph construction, validation, and topology utilities.
//! - [`kernel`] defines the kernel trait and kernel-level errors.
//! - [`node`] defines graph node identifiers and node metadata.
//! - [`op`] defines supported operation kinds.
//! - [`registry`] contains the kernel registry.
//! - [`tensor`] defines the tensor value type.
#![deny(clippy::all)]
#![deny(clippy::pedantic)]
#![deny(missing_docs)]

pub mod executor;
pub mod graph;
pub mod kernel;
pub mod node;
pub mod op;
pub mod registry;
pub mod tensor;

pub use executor::{ExecutionError, Executor};
pub use graph::{Graph, GraphError};
pub use kernel::{Kernel, KernelError};
pub use node::{Node, NodeId};
pub use op::OpKind;
pub use registry::KernelRegistry;
pub use tensor::Tensor;
