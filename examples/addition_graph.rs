//! Minimal end-to-end example: build a graph, bind inputs, execute it, and read outputs.
//!
//! This example is the best starting point for understanding the crate's execution model.
//!
//! Key ideas:
//! - A `Graph` is a DAG of tensor operations.
//! - Input nodes declare required runtime inputs by `NodeId` and shape.
//! - Operation nodes declare computation to perform later.
//! - `Executor` evaluates the graph using kernels from a `KernelRegistry`.
//! - The executor returns a map from output `NodeId` to computed `Tensor`.
//!
//! For kernel authors, this example shows the contract at the executor boundary:
//! - graph construction performs structural validation,
//! - execution provides concrete `Tensor` inputs,
//! - kernels are responsible for producing the tensor value for a single op node.
//!
//! The Add kernel used here conceptually receives:
//! - the Add node metadata,
//! - the already-computed tensors for its input nodes,
//! - enough context to produce a new output tensor.
//!
//! A custom kernel for Add would usually:
//! - confirm input count and shape assumptions,
//! - read element values from both input tensors,
//! - allocate or construct the output tensor,
//! - return the computed tensor or a `KernelError`.
use tensor_forge::{Executor, Graph, KernelRegistry, Tensor};

fn main() {
    // Build the graph:
    //
    //   a ----\
    //          add ---> out
    //   b ----/
    //
    // Here `a` and `b` are graph input nodes. They do not yet have runtime values;
    // they only declare that the graph expects tensors of shape [2, 2].
    let mut graph = Graph::new();

    let a = graph.input_node(vec![2, 2]);
    let b = graph.input_node(vec![2, 2]);

    // Add an operation node.
    //
    // `graph.add(a, b)` does not perform arithmetic immediately. It adds a new node to
    // the graph describing a future Add operation whose inputs are `a` and `b`.
    //
    // Shape validation happens here at graph-construction time. Since both inputs are
    // [2, 2], the resulting Add node is also [2, 2].
    let out = graph
        .add(a, b)
        .expect("Adding valid input nodes should succeed");

    // Mark the node as an output. The executor will return a tensor for every node
    // designated as a graph output.
    graph
        .set_output_node(out)
        .expect("Setting output node should succeed");

    // Create runtime tensors for the graph input nodes.
    //
    // These must match the shapes declared by the corresponding input nodes.
    let a_tensor = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])
        .expect("Tensor construction should succeed");
    let b_tensor = Tensor::from_vec(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0])
        .expect("Tensor construction should succeed");

    // Construct an executor with the default kernel registry.
    //
    // The registry determines which kernel implementation is used for each `OpKind`.
    // In this example, the default registry is expected to contain an Add kernel.
    let exec = Executor::new(KernelRegistry::default());

    // Execute the graph.
    //
    // The bindings are `(NodeId, Tensor)` pairs. Each input node in the graph must be
    // bound exactly once at runtime.
    //
    // Internally, the executor:
    // 1. validates the bindings,
    // 2. topologically orders the graph,
    // 3. executes non-input nodes using registered kernels,
    // 4. returns tensors for all declared output nodes.
    let outputs = exec
        .execute(&graph, vec![(a, a_tensor), (b, b_tensor)])
        .expect("Execution should succeed");

    let result = outputs
        .get(&out)
        .expect("Declared output should be present in executor results");

    println!("Computed output for node {:?}: {:?}", out, result);
}
