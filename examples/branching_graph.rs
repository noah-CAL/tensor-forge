//! Slightly richer example: a branching graph with ReLU, MatMul, and Add.
//!
//! Graph:
//!   ra = relu(a)
//!   rb = relu(b)
//!   mm = matmul(ra, rb)
//!   out = add(mm, c)
//!                             c ---- \
//!   a ---- relu(a) ---- \              out
//!                         matmul --- /
//!   b ---- relu(b) ---- /
//!
//! This demonstrates:
//! - multiple input nodes with different shapes,
//! - shape propagation through several operations,
//! - intermediate nodes feeding later computation,
//! - how the executor relies on registered kernels for each op kind.
//!
//! This example is useful because it shows that kernels are not
//! responsible for graph traversal. The executor handles dependency ordering; each
//! kernel only computes one node once its dependencies are available.
//!
//! For kernel authors:
//!
//! The executor will evaluate this graph in dependency order roughly like:
//!   1. read input bindings for `a`, `b`, `c`
//!   2. execute ReLU kernel for `ra`
//!   3. execute ReLU kernel for `rb`
//!   4. execute MatMul kernel for `mm`
//!   5. execute Add kernel for `out`
//!
//! Important consequence:
//! - a kernel does not need to recursively evaluate upstream nodes,
//! - a kernel only consumes already-available input tensors,
//! - shape compatibility should already be structurally valid if the graph builder
//!   enforces shape rules, though kernels may still defensively validate runtime inputs.
//!
//! A custom MatMul kernel, for example, would typically:
//! - expect exactly 2 input tensors,
//! - verify rank/shape assumptions,
//! - compute matrix multiplication,
//! - return a new tensor with the node's declared output shape.

use tensor_forge::{Executor, Graph, KernelRegistry, Tensor};

fn main() {
    let mut graph = Graph::new();

    // Declare graph inputs.
    //
    // The shapes here establish the legal runtime tensor shapes:
    //   a: [2, 3]
    //   b: [3, 2]
    //   c: [2, 2]
    let a = graph.input_node(vec![2, 3]);
    let b = graph.input_node(vec![3, 2]);
    let c = graph.input_node(vec![2, 2]);

    // Build intermediate operations.
    //
    // `relu(a)` preserves shape [2, 3].
    let ra = graph.relu(a).expect("Valid ReLU operation should succeed");

    // `relu(b)` preserves shape [3, 2].
    let rb = graph.relu(b).expect("Valid ReLU operation should succeed");

    // `matmul(ra, rb)` combines [2, 3] x [3, 2] -> [2, 2].
    //
    // This is a good example of graph-level validation preventing malformed graphs
    // before execution ever begins.
    let mm = graph
        .matmul(ra, rb)
        .expect("Valid matmul operation should succeed");

    // `add(mm, c)` adds two [2, 2] tensors and also yields [2, 2].
    let out = graph
        .add(mm, c)
        .expect("Valid add operation should succeed");

    graph
        .set_output_node(out)
        .expect("Setting output node should succeed");

    // Bind concrete runtime values.
    //
    // These values are only examples; the graph structure is independent of them.
    // The same graph can be executed many times with different input tensors.
    let a_tensor = Tensor::from_vec(vec![2, 3], vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
        .expect("Tensor construction should succeed");

    let b_tensor = Tensor::from_vec(vec![3, 2], vec![-7.0, 8.0, 9.0, -10.0, 11.0, 12.0])
        .expect("Tensor construction should succeed");

    let c_tensor = Tensor::from_vec(vec![2, 2], vec![0.5, 1.5, 2.5, 3.5])
        .expect("Tensor construction should succeed");

    let exec = Executor::new(KernelRegistry::default());

    let outputs = exec
        .execute(&graph, vec![(a, a_tensor), (b, b_tensor), (c, c_tensor)])
        .expect("Execution should succeed");

    let result = outputs
        .get(&out)
        .expect("Declared output should be present in executor results");

    println!("Computed output for node {:?}: {:?}", out, result);
}
