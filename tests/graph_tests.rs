use tensor_forge::graph::{Graph, GraphError};

#[test]
fn graph_creation_valid() {
    let graph = Graph::new();
    assert!(graph.nodes().is_empty());
}

#[test]
fn graph_add_nodes() {
    unimplemented!()
}

#[test]
fn graph_shape_validation() {
    unimplemented!()
}

#[test]
fn graph_invalid_shape_errors() {
    unimplemented!()
}

#[test]
fn graph_add_input() {
    unimplemented!()
}

#[test]
fn graph_add_matmul_node() {
    unimplemented!()
}

#[test]
fn graph_matmul_invalid_dimensions() {
    unimplemented!()
}


#[test]
fn graph_add_matrices() {
    unimplemented!()
}

#[test]
fn graph_relu() {
    unimplemented!()
}

#[test]
fn graph_nodes() {
    unimplemented!()
}

#[test]
fn graph_set_output() {
    unimplemented!()
}

#[test]
fn graph_missing_input_nodes() {
}
