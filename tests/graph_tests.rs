use tensor_forge::graph::{Graph, GraphError};
use tensor_forge::op::OpKind;

use std::collections::HashSet;

#[test]
fn graph_creation_valid() {
    let graph = Graph::new();
    assert!(graph.nodes().is_empty());
}

#[test]
fn graph_node_creation() {
    let mut graph = Graph::new();
    assert!(graph.nodes().is_empty());

    let (shape_a, shape_b) = (vec![4, 1], vec![1, 4]);
    let node_id_a = graph.input_node(shape_a);
    let node_id_b = graph.input_node(shape_b);
    assert_ne!(node_id_a, node_id_b);

    let node_a = graph
        .node(node_id_a)
        .expect("Node A should be added and present in graph");
    let node_b = graph
        .node(node_id_b)
        .expect("Node B should be added and present in graph");

    assert_ne!(node_a.id, node_b.id);
    assert_eq!(node_a.op, OpKind::Input);
    assert_eq!(node_b.op, OpKind::Input);
}

#[test]
fn graph_add_matrices() {
    let mut graph = Graph::new();
    assert!(graph.nodes().is_empty());

    let shape = vec![4, 1];
    // 1. Test addition of nodes with valid dimensions.
    let node_a = graph.input_node(shape.clone());
    let node_b = graph.input_node(shape.clone());
    assert_ne!(node_a, node_b);

    let output = graph
        .add(node_a, node_b)
        .expect("Valid addition should succeed");
    assert!(output != node_a && output != node_b);

    let output_node = graph.node(output).expect("Addition node should be present");
    assert_eq!(output_node.op, OpKind::Add);
    assert_eq!(output_node.shape, shape);
    assert_eq!(output_node.inputs.len(), 2);
    assert_eq!(output_node.inputs[0], node_a);
    assert_eq!(output_node.inputs[1], node_b);

    // 2. Test addition of nodes with invalid dimensions.
    let mut graph = Graph::new();
    assert!(graph.nodes().is_empty());

    let node_a = graph.input_node(vec![4, 1]);
    let node_b = graph.input_node(vec![4, 2]);
    assert_ne!(node_a, node_b);

    let output_id = graph.add(node_a, node_b);
    assert!(matches!(output_id.unwrap_err(), GraphError::ShapeMismatch));
}

#[test]
fn graph_relu() {
    let mut graph = Graph::new();
    assert!(graph.nodes().is_empty());

    let shape = vec![4, 1, 8];
    let input = graph.input_node(shape.clone());
    let output = graph
        .relu(input)
        .expect("Adding valid relu input should succeed");

    let relu_node = graph
        .node(output)
        .expect("relu output note should be present");
    assert_eq!(relu_node.op, OpKind::ReLU);
    assert_eq!(relu_node.shape, shape);
    assert_eq!(relu_node.inputs.len(), 1);
    assert_eq!(relu_node.inputs[0], input);
}

#[test]
fn graph_add_matmul_node() {
    let mut graph = Graph::new();
    assert!(graph.nodes().is_empty());

    let node_a_id = graph.input_node(vec![4, 1]);
    let node_b_id = graph.input_node(vec![1, 4]);
    assert_ne!(node_a_id, node_b_id);

    let output_id = graph
        .matmul(node_a_id, node_b_id)
        .expect("Valid matmul dimensions should succeed");
    assert!(output_id != node_a_id && output_id != node_b_id);

    let output_node = graph
        .node(output_id)
        .expect("Matmul output node should be present");
    assert_eq!(output_node.op, OpKind::MatMul);
    assert_eq!(output_node.shape, vec![4, 4]);
    assert_eq!(output_node.inputs.len(), 2);
    assert_eq!(output_node.inputs[0], node_a_id);
    assert_eq!(output_node.inputs[1], node_b_id);
}

#[test]
fn graph_matmul_invalid_dimensions() {
    let mut graph = Graph::new();
    assert!(graph.nodes().is_empty());

    let node_a = graph.input_node(vec![4, 1]);
    let node_b = graph.input_node(vec![4, 1]);
    assert_ne!(node_a, node_b);

    let output = graph.matmul(node_a, node_b);
    assert!(matches!(output.unwrap_err(), GraphError::ShapeMismatch));
}

#[test]
fn graph_missing_input_nodes() {
    // Test missing inputs by using Node IDs from a 
    // second graph instantiation
    let mut fake_graph = Graph::new();
    for _ in 0..10 {
        fake_graph.input_node(vec![1, 1]);
    }
    let fake_node = fake_graph.input_node(vec![1, 1]);

    let mut graph = Graph::new();
    assert!(graph.nodes().is_empty());
    let node_a = graph.input_node(vec![4, 1]);

    // 1) Node
    let output = graph.node(fake_node);
    assert!(matches!(output.unwrap_err(), GraphError::InvalidNodeId));
    // 2) ReLU
    let output = graph.relu(fake_node);
    assert!(matches!(output.unwrap_err(), GraphError::InvalidNodeId));
    // 3) Add
    let output = graph.add(node_a, fake_node);
    assert!(matches!(output.unwrap_err(), GraphError::InvalidNodeId));
    // 4) Matmul
    let output = graph.matmul(node_a, fake_node);
    assert!(matches!(output.unwrap_err(), GraphError::InvalidNodeId));

    assert_eq!(graph.inputs().len(), 1);
    assert_eq!(graph.outputs().len(), 0);
}

///////////////////////////////
/// Large integration tests ///
///////////////////////////////
#[test]
fn graph_chained_addition() {
    // Add a large number of nodes of arbitrary shape and inspect the output.
    // Ensure there are no errors.
    let mut graph = Graph::new();
    assert!(graph.nodes().is_empty());

    // 1) Create large number of input nodes.
    const NUM_INPUTS: usize = 1_000;
    const NUM_OUTPUTS: usize = NUM_INPUTS / 2;
    let shape = vec![20, 10];

    let mut nodes = Vec::with_capacity(NUM_INPUTS);
    for _ in 0..NUM_INPUTS {
        let node = graph.input_node(shape.clone());
        nodes.push(node);
    }

    // 2) Link each pair of input nodes in an Add operation and declare them
    //    all outputs. For this reason, NUM_NODES must be a multiple of two.
    //
    //    (and NUM_NODES must be greater than 2, of course)
    assert_eq!(
        NUM_INPUTS % 2,
        0,
        "NUM_NODES must be divisible by 2 for testing. (NUM_NODES={})",
        NUM_INPUTS
    );
    for i in (1..NUM_INPUTS).step_by(2) {
        let result = graph
            .add(nodes[i], nodes[i - 1])
            .expect("Adding valid Addition node should succeed");
        graph.set_output_node(result);
        // Add to total nodes
        nodes.push(result);
    }

    // 3) Ensure that all of our input and output nodes are present.
    assert_eq!(nodes.len(), NUM_INPUTS + NUM_OUTPUTS);
    for (i, &node_id) in nodes.iter().enumerate() {
        // Ensure our ID is present in our graph
        let node = graph
            .node(node_id)
            .expect("Expected ID to be present in graph");

        // Check its input/output node selection.
        if i < NUM_INPUTS {
            assert_eq!(node.op, OpKind::Input);
        } else {
            assert_eq!(node.op, OpKind::Add);
        }
    }
    // Avoid specifically checking that the nodes exist in the graph's
    // input/output node list due to O(N^2) time complexity.
    //
    // Solutions exist in O(N) time, but valid checks will already occur in other
    // integration tests.
}

#[test]
fn graph_chained_matmul() {
    // Tests:
    // (10x2) x (2x8) x (8x4) x (4x61) x (61x3)
    let mut graph = Graph::new();
    assert!(graph.nodes().is_empty());

    let shapes: Vec<Vec<usize>> = vec![
        vec![10, 2],
        vec![2, 8],
        vec![8, 4],
        vec![4, 61],
        vec![61, 3],
    ];
    let mut inputs = Vec::new();
    for shape in shapes {
        inputs.push(graph.input_node(shape));
    }
    let mut prev = inputs[0];
    for &input_node in &inputs[1..] {
        prev = graph
            .matmul(prev, input_node)
            .expect("Matmul dimensions are valid and should not fail");
    }
}

#[test]
fn graph_chain_full_implementation() {
    // Tests:
    // Input A -- ReLU     Input C
    //                 \          \
    //                  -- Matmul - Add -> Output
    //                 /
    // Input B -- ReLU
    //
    // Layers:
    //    (1)     (2)    (3)      (4)      (5)
    let mut graph = Graph::new();
    assert!(graph.nodes().is_empty());

    let mut node_set = HashSet::new();

    let shape_a = vec![10, 5];
    let shape_b = vec![5, 10];
    let shape_c = vec![shape_a[0], shape_b[1]];

    // (1) Create input nodes A and B
    let node_a = graph.input_node(shape_a.clone());
    let node_b = graph.input_node(shape_b.clone());
    node_set.insert(node_a.clone());
    node_set.insert(node_b.clone());

    // (2) Create nodes relu(A) and relu(B)
    let relu_a = graph
        .relu(node_a)
        .expect("Valid relu operation should succeed");
    let relu_b = graph
        .relu(node_b)
        .expect("Valid relu operation should succeed");

    // Verify relu(A)
    let relu_node = graph
        .node(relu_a)
        .expect("ReLU node A should have been added");
    assert_eq!(relu_node.shape, shape_a.clone());
    assert_eq!(relu_node.inputs.len(), 1);
    assert_eq!(relu_node.inputs[0], node_a);

    node_set.insert(relu_a.clone());

    // Verify relu(B)
    let relu_node = graph
        .node(relu_b)
        .expect("ReLU node A should have been added");
    assert_eq!(relu_node.shape, shape_b);
    assert_eq!(relu_node.inputs.len(), 1);
    assert_eq!(relu_node.inputs[0], node_b);

    node_set.insert(relu_b.clone());

    // (3) Feed into Matmul operation
    let matmul = graph
        .matmul(relu_a.clone(), relu_b.clone())
        .expect("Valid matmul after relu should succeed");
    let matmul_node = graph.node(matmul).expect("Matmul node should be added");
    assert_eq!(matmul_node.shape, shape_c);
    assert_eq!(matmul_node.inputs.len(), 2);
    assert_eq!(matmul_node.inputs[0], relu_a.clone());
    assert_eq!(matmul_node.inputs[1], relu_b.clone());

    node_set.insert(matmul.clone());

    // (4) Compute the addition of the matmul result with an addition
    //     C of shape dim(A)[0] + dim(B)[1]
    let node_c = graph.input_node(shape_c.clone());
    let add = graph
        .add(matmul, node_c)
        .expect("Adding Add node should succeed");
    node_set.insert(add.clone());

    let add_node = graph.node(add).expect("Addition node should have be added");
    assert_eq!(add_node.shape, shape_c.clone());
    assert_eq!(add_node.inputs.len(), 2);
    assert!(add_node.inputs.contains(&node_c.clone()));
    assert!(add_node.inputs.contains(&matmul.clone()));

    // (5) Create final output node and bind to add.
    graph.set_output_node(add);

    // Check that we allocated 6 unique nodes throughout the test.
    // Any missing nodes or any node_id collisions will be detected.
    assert_eq!(node_set.len(), 6);
    assert_eq!(graph.nodes().len(), node_set.len());
}
