//! Demonstrates how to define and register a custom kernel.
//!
//! This example is aimed at users who want to extend the runtime with their own
//! kernel implementations.
//!
//! The important architectural pieces are:
//!
//! - `Graph` describes *what* should be computed.
//! - `Executor` decides *when* each node is executed.
//! - `KernelRegistry` decides *which kernel* handles each `OpKind`.
//! - A `Kernel` implementation computes the output for one node from its inputs.
//!
//! In this example, we replace the default `Add` kernel with a custom one.
//!
//! ## The kernel contract
//!
//! A kernel implementation receives:
//! - `inputs`: already-computed input tensors for the current node
//! - `output`: a mutable tensor already allocated by the executor
//!
//! A kernel is expected to:
//! 1. validate its input count and any assumptions it cares about,
//! 2. read from `inputs`,
//! 3. write the result into `output`,
//! 4. return `Ok(())` on success or `Err(KernelError)` on failure.
//!
//! The executor handles:
//! - topological ordering,
//! - dependency resolution,
//! - output allocation,
//! - error attribution back to the graph node.
//!
//! So kernel authors do **not** need to traverse the graph themselves.

use tensor_forge::executor::Executor;
use tensor_forge::graph::Graph;
use tensor_forge::kernel::{Kernel, KernelError};
use tensor_forge::op::OpKind;
use tensor_forge::registry::KernelRegistry;
use tensor_forge::tensor::Tensor;

/// A minimal custom kernel for `OpKind::Add`.
///
/// This example focuses on the *shape* of a kernel implementation and the
/// registry/executor integration points.
struct CustomAddKernel;

impl Kernel for CustomAddKernel {
    fn compute(&self, inputs: &[&Tensor], output: &mut Tensor) -> Result<(), KernelError> {
        // Validate arity.
        //
        // The Add operation expects exactly two input tensors.
        if inputs.len() != 2 {
            return Err(KernelError::InvalidArguments);
        }

        // Validate shape agreement.
        //
        // Graph construction should already guarantee this for well-formed Add nodes,
        // but kernels may still validate assumptions defensively.
        let left = inputs[0];
        let right = inputs[1];

        if left.shape() != right.shape() || left.shape() != output.shape() {
            return Err(KernelError::InvalidArguments);
        }

        let left_data = left.data();
        let right_data = right.data();
        let output_data = output.data_mut();
        for i in 0..output_data.len() {
            output_data[i] = left_data[i] + right_data[i];
        }
        Ok(())
    }
}

fn main() {
    // Build a tiny graph:
    //
    //   out = add(a, b)
    //
    // The graph describes *what* should be computed, not how.
    let mut graph = Graph::new();

    let a = graph.input_node(vec![2, 2]);
    let b = graph.input_node(vec![2, 2]);
    let out = graph
        .add(a, b)
        .expect("Adding valid input nodes should succeed");

    graph
        .set_output_node(out)
        .expect("Setting output node should succeed");

    // Create a custom registry.
    //
    // Start from an empty registry and explicitly register only the kernel(s)
    // needed by this graph.
    let mut registry = KernelRegistry::new();

    // Register our custom Add kernel.
    //
    // `register(...)` returns the previous mapping if one existed.
    let old = registry.register(OpKind::Add, Box::new(CustomAddKernel));
    assert!(
        old.is_none(),
        "First Add registration should not replace an existing kernel"
    );

    // Construct the executor with the custom registry.
    let exec = Executor::new(registry);

    // Bind runtime inputs.
    //
    // These are ordinary tensors supplied for the graph input nodes.
    let a_tensor = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
        .expect("Tensor construction should succeed");
    let b_tensor = Tensor::from_vec(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0])
        .expect("Tensor construction should succeed");

    // Execute the graph.
    //
    // During execution:
    // - the executor validates input bindings,
    // - walks the graph in topological order,
    // - sees an `OpKind::Add` node,
    // - looks up `OpKind::Add` in the registry,
    // - dispatches to `CustomAddKernel::compute(...)`.
    let outputs = exec
        .execute(&graph, vec![(a, a_tensor), (b, b_tensor)])
        .expect("Execution should succeed");

    let result = outputs
        .get(&out)
        .expect("Declared output should be present in executor results");

    println!("Computed output for node {:?}: {:?}", out, result);
}
