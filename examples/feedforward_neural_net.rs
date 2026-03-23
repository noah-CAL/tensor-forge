//! Construct a small feedforward-style network by building layers in a loop.
//!
//! This is not a full neural-network framework, but it shows how larger graphs can be
//! assembled programmatically rather than node-by-node.
//!
//! Layout:
//! - input layer:  3 nodes
//! - hidden layer: 2 nodes
//! - hidden layer: 2 nodes
//! - output layer: 2 nodes
//!
//! Each node in the next layer is computed as:
//!   relu(add(add(prev[0], prev[1]), prev[2]...))
//!
//! All nodes use the same tensor shape so that the example can be expressed using
//! only `Add` and `ReLU`.
//!
//! For kernel authors, this example is the most useful one for understanding scale:
//! - kernels remain small and local,
//! - graph construction can be arbitrarily rich,
//! - executor behavior is unchanged regardless of graph size.
//!
//! This example also shows an important architectural boundary:
//!
//! Graph construction:
//! - decides how many nodes exist,
//! - decides which ops connect which dependencies,
//! - enforces shape / structural constraints,
//! - decides which nodes are final outputs.
//!
//! Executor:
//! - validates runtime input bindings,
//! - computes nodes in dependency order,
//! - looks up the registered kernel for each op kind,
//! - stores intermediate results for later nodes.
//!
//! Kernel:
//! - computes exactly one node,
//! - reads already-computed input tensors,
//! - returns the output tensor or a kernel-level error.
//!
//! That separation is what allows users to design custom kernels without needing
//! to reimplement graph traversal or scheduling.
use tensor_forge::{Executor, Graph, KernelRegistry, NodeId, Tensor};

fn main() {
    let mut graph = Graph::new();

    let input_width = 3;
    let hidden_widths = [2, 2];
    let output_width = 2;
    let shape = vec![1, 4];

    // Create the input layer.
    //
    // Each input node declares that execution must provide a [1, 4] tensor for it.
    let mut current_layer: Vec<NodeId> = (0..input_width)
        .map(|_| graph.input_node(shape.clone()))
        .collect();

    let input_ids = current_layer.clone();

    // Build hidden layers iteratively.
    //
    // This loop is only in graph construction. The resulting graph is still a
    // feedforward DAG with no cycles.
    for &layer_width in &hidden_widths {
        let mut next_layer = Vec::with_capacity(layer_width);

        for _ in 0..layer_width {
            // Start accumulation with the first node of the current layer.
            let mut acc = current_layer[0];

            // Repeatedly add the remaining nodes from the current layer.
            //
            // Each `graph.add(...)` creates a new intermediate node.
            for &node in &current_layer[1..] {
                acc = graph
                    .add(acc, node)
                    .expect("Adding nodes in a layer should succeed");
            }

            // Apply a nonlinearity at the end of the layer computation.
            let out = graph
                .relu(acc)
                .expect("Applying ReLU after accumulation should succeed");

            next_layer.push(out);
        }

        current_layer = next_layer;
    }

    // Build the output layer the same way.
    let mut output_ids = Vec::with_capacity(output_width);

    for _ in 0..output_width {
        let mut acc = current_layer[0];

        for &node in &current_layer[1..] {
            acc = graph
                .add(acc, node)
                .expect("Adding nodes in the output layer should succeed");
        }

        let out = graph
            .relu(acc)
            .expect("Applying ReLU in the output layer should succeed");

        graph
            .set_output_node(out)
            .expect("Setting output node should succeed");

        output_ids.push(out);
    }

    // Bind concrete input tensors.
    //
    // As in the smaller examples, bindings are keyed by input-node `NodeId`.
    // The graph can be re-used with different runtime values.
    let bindings = vec![
        (
            input_ids[0],
            Tensor::from_vec(vec![1, 4], vec![1.0, -2.0, 3.0, -4.0])
                .expect("Tensor construction should succeed"),
        ),
        (
            input_ids[1],
            Tensor::from_vec(vec![1, 4], vec![0.5, 1.5, -2.5, 3.5])
                .expect("Tensor construction should succeed"),
        ),
        (
            input_ids[2],
            Tensor::from_vec(vec![1, 4], vec![10.0, -20.0, 30.0, -40.0])
                .expect("Tensor construction should succeed"),
        ),
    ];

    let exec = Executor::new(KernelRegistry::default());
    let outputs = exec
        .execute(&graph, bindings)
        .expect("Execution should succeed");

    for out in output_ids {
        let tensor = outputs
            .get(&out)
            .expect("Declared output should be present in executor results");

        println!("Computed output for node {:?}: {:?}", out, tensor);
    }
}
