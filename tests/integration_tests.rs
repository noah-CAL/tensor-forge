use std::collections::HashMap;

use tensor_forge::executor::{ExecutionError, Executor};
use tensor_forge::graph::Graph;
use tensor_forge::node::NodeId;
use tensor_forge::op::OpKind;
use tensor_forge::registry::KernelRegistry;
use tensor_forge::tensor::Tensor;

/////////////
// Helpers //
/////////////
mod common;
use common::{add_ref, assert_tensor_eq, matmul_ref, relu_ref};

////////////////////////////////
// Executor Integration Tests //
////////////////////////////////
#[test]
fn executor_simple_relu_execution_bindings() {
    // Graph: x -> relu(x) -> out
    let mut g = Graph::new();
    let x_id = g.input_node(vec![1, 6]);
    let y_id = g.relu(x_id).expect("Adding ReLU node should succeed");
    g.set_output_node(y_id)
        .expect("Setting output should succeed");

    let x_data = vec![-3.0, -0.0, 0.0, 1.5, 2.0, -7.25];
    let x =
        Tensor::from_vec(vec![1, 6], x_data.clone()).expect("Tensor construction should succeed");

    let expected = relu_ref(&x_data);

    let exec = Executor::new(KernelRegistry::default());
    let outs = exec
        .execute(&g, vec![(x_id, x)])
        .expect("Execution should succeed");

    assert_eq!(outs.len(), 1);
    let output_tensor: &Tensor = &outs[&y_id];

    assert_eq!(output_tensor.shape(), &[1, 6]);
    assert_tensor_eq(output_tensor, &expected);
}

#[test]
fn executor_add_graph_execution_bindings_order_independent() {
    // Graph: a, b -> add(a,b) -> out
    let mut g = Graph::new();
    let a_id = g.input_node(vec![2, 3]);
    let b_id = g.input_node(vec![2, 3]);
    let c_id = g.add(a_id, b_id).expect("Add should succeed");
    g.set_output_node(c_id)
        .expect("Setting output should succeed");

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let expected = add_ref(&a_data, &b_data);

    let a = Tensor::from_vec(vec![2, 3], a_data).expect("Tensor construction should succeed");
    let b = Tensor::from_vec(vec![2, 3], b_data).expect("Tensor construction should succeed");

    let exec = Executor::new(KernelRegistry::default());

    // Swap binding order deliberately.
    let outs = exec
        .execute(&g, vec![(b_id, b), (a_id, a)])
        .expect("Execution should succeed");

    assert_eq!(outs.len(), 1);
    let output_tensor: &Tensor = &outs[&c_id];

    assert_eq!(output_tensor.shape(), &[2, 3]);
    assert_tensor_eq(output_tensor, &expected);
}

#[test]
fn executor_matmul_graph_execution_bindings() {
    // Graph: a, b -> matmul(a,b) -> out
    // A: 2x3, B: 3x2 => C: 2x2
    let mut g = Graph::new();
    let a_id = g.input_node(vec![2, 3]);
    let b_id = g.input_node(vec![3, 2]);
    let c_id = g.matmul(a_id, b_id).expect("Matmul should succeed");
    g.set_output_node(c_id)
        .expect("Setting output should succeed");

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let expected = matmul_ref(&a_data, &b_data, 2, 3, 2);

    let a = Tensor::from_vec(vec![2, 3], a_data).expect("Tensor construction should succeed");
    let b = Tensor::from_vec(vec![3, 2], b_data).expect("Tensor construction should succeed");

    let exec = Executor::new(KernelRegistry::default());
    let outs = exec
        .execute(&g, vec![(a_id, a), (b_id, b)])
        .expect("Execution should succeed");

    assert_eq!(outs.len(), 1);
    let output_tensor: &Tensor = &outs[&c_id];

    assert_eq!(output_tensor.shape(), &[2, 2]);
    assert_tensor_eq(output_tensor, &expected);
}

#[test]
fn executor_multi_op_matmul_add_relu_execution_bindings() {
    // Graph: relu(matmul(A,B) + C)
    let mut g = Graph::new();
    let a_id = g.input_node(vec![2, 3]);
    let b_id = g.input_node(vec![3, 2]);
    let c_id = g.input_node(vec![2, 2]);

    let mm_id = g.matmul(a_id, b_id).expect("Matmul should succeed");
    let add_id = g.add(mm_id, c_id).expect("Add should succeed");
    let relu_id = g.relu(add_id).expect("ReLU should succeed");
    g.set_output_node(relu_id)
        .expect("Setting output should succeed");

    let a_data = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0];
    let b_data = vec![1.0, -2.0, 3.0, 4.0, -5.0, 6.0];
    let c_data = vec![10.0, 20.0, 30.0, 40.0];

    let mm = matmul_ref(&a_data, &b_data, 2, 3, 2);
    let add = add_ref(&mm, &c_data);
    let expected = relu_ref(&add);

    let a = Tensor::from_vec(vec![2, 3], a_data).expect("Tensor construction should succeed");
    let b = Tensor::from_vec(vec![3, 2], b_data).expect("Tensor construction should succeed");
    let c = Tensor::from_vec(vec![2, 2], c_data).expect("Tensor construction should succeed");

    let exec = Executor::new(KernelRegistry::default());

    // Scramble binding order to ensure NodeId binding works.
    let outs = exec
        .execute(&g, vec![(c_id, c), (a_id, a), (b_id, b)])
        .expect("Execution should succeed");

    assert_eq!(outs.len(), 1);
    let output_tensor: &Tensor = &outs[&relu_id];

    assert_eq!(output_tensor.shape(), &[2, 2]);
    assert_tensor_eq(output_tensor, &expected);
}

//////////////////////////////
// Binding / Error Coverage //
//////////////////////////////

#[test]
fn executor_rejects_missing_input_binding() {
    // Graph: a, b -> add(a,b) -> out; provide only a
    let mut g = Graph::new();
    let a_id = g.input_node(vec![2, 2]);
    let b_id = g.input_node(vec![2, 2]);
    let c_id = g.add(a_id, b_id).expect("Add should succeed");
    g.set_output_node(c_id)
        .expect("Setting output should succeed");

    let a = Tensor::zeros(vec![2, 2]).expect("Tensor construction should succeed");

    let exec = Executor::new(KernelRegistry::default());
    let err = exec.execute(&g, vec![(a_id, a)]).unwrap_err();

    assert!(matches!(err, ExecutionError::MissingInput { node } if node == b_id));
}

#[test]
fn executor_rejects_invalid_binding_node_foreign_graph() {
    let mut g = Graph::new();
    let x_id = g.input_node(vec![1, 1]);
    g.set_output_node(x_id)
        .expect("Setting output should succeed");

    let x = Tensor::from_vec(vec![1, 1], vec![1.0]).expect("Tensor construction should succeed");

    // Foreign id from another graph
    let mut g2 = Graph::new();
    let foreign_id = g2.input_node(vec![1, 1]);
    let foreign_t =
        Tensor::from_vec(vec![1, 1], vec![2.0]).expect("Tensor construction should succeed");

    let exec = Executor::new(KernelRegistry::default());
    let err = exec
        .execute(&g, vec![(x_id, x), (foreign_id, foreign_t)])
        .unwrap_err();

    assert!(matches!(err, ExecutionError::InvalidBindingNode { node } if node == foreign_id));
}

#[test]
fn executor_rejects_duplicate_binding() {
    let mut g = Graph::new();
    let x_id = g.input_node(vec![1, 2]);
    g.set_output_node(x_id)
        .expect("Setting output should succeed");

    let t1 =
        Tensor::from_vec(vec![1, 2], vec![1.0, 2.0]).expect("Tensor construction should succeed");
    let t2 =
        Tensor::from_vec(vec![1, 2], vec![3.0, 4.0]).expect("Tensor construction should succeed");

    let exec = Executor::new(KernelRegistry::default());
    let err = exec.execute(&g, vec![(x_id, t1), (x_id, t2)]).unwrap_err();

    assert!(matches!(err, ExecutionError::DuplicateBinding { node } if node == x_id));
}

#[test]
fn executor_rejects_binding_to_non_input_node() {
    // Graph: a, b -> add(a,b) -> out ; try to bind to add node
    let mut g = Graph::new();
    let a_id = g.input_node(vec![1, 2]);
    let b_id = g.input_node(vec![1, 2]);
    let add_id = g.add(a_id, b_id).expect("Add should succeed");
    g.set_output_node(add_id)
        .expect("Setting output should succeed");

    let bogus = Tensor::zeros(vec![1, 2]).expect("Tensor construction should succeed");

    let exec = Executor::new(KernelRegistry::default());
    let err = exec.execute(&g, vec![(add_id, bogus)]).unwrap_err();

    assert!(matches!(
        err,
        ExecutionError::BindingToNonInputNode { node, op }
            if node == add_id && op == OpKind::Add
    ));
}

#[test]
fn executor_rejects_input_shape_mismatch() {
    let mut g = Graph::new();
    let x_id = g.input_node(vec![2, 3]);
    g.set_output_node(x_id)
        .expect("Setting output should succeed");

    // Same numel, wrong shape.
    let bad =
        Tensor::from_vec(vec![3, 2], vec![1.0; 6]).expect("Tensor construction should succeed");

    let exec = Executor::new(KernelRegistry::default());
    let err = exec.execute(&g, vec![(x_id, bad)]).unwrap_err();

    assert!(matches!(
        err,
        ExecutionError::InputShapeMismatch { node, expected, actual }
            if node == x_id && expected == vec![2, 3] && actual == vec![3, 2]
    ));
}

#[test]
fn executor_rejects_missing_kernel() {
    // Graph: a, b -> add(a,b) -> out
    let mut g = Graph::new();
    let a_id = g.input_node(vec![1, 2]);
    let b_id = g.input_node(vec![1, 2]);
    let add_id = g.add(a_id, b_id).expect("Add should succeed");
    g.set_output_node(add_id)
        .expect("Setting output should succeed");

    let a =
        Tensor::from_vec(vec![1, 2], vec![1.0, 2.0]).expect("Tensor construction should succeed");
    let b =
        Tensor::from_vec(vec![1, 2], vec![3.0, 4.0]).expect("Tensor construction should succeed");

    // Registry intentionally has a missing Add kernel.
    let reg = KernelRegistry::new();
    let exec = Executor::new(reg);

    let err = exec.execute(&g, vec![(a_id, a), (b_id, b)]).unwrap_err();

    assert!(matches!(err, ExecutionError::KernelNotFound { op } if op == OpKind::Add));

    let _ = add_id;
}

#[test]
fn executor_reports_kernel_failure_with_node_context() {
    // For a kernel that fails, the intent is not just "execution fails" but
    // the executor reports that failure and annotates it with the failing node.
    use tensor_forge::kernel::{Kernel, KernelError};

    // A kernel that always fails. Useful for verifying error propagation + node tagging.
    struct FailingKernel;

    impl Kernel for FailingKernel {
        fn compute(&self, _inputs: &[&Tensor], _output: &mut Tensor) -> Result<(), KernelError> {
            Err(KernelError::InvalidArguments)
        }
    }

    // Graph: a, b -> add(a,b) -> out
    let mut g = Graph::new();
    let a_id = g.input_node(vec![1, 2]);
    let b_id = g.input_node(vec![1, 2]);
    let add_id = g.add(a_id, b_id).expect("Add should succeed");
    g.set_output_node(add_id)
        .expect("Setting output should succeed");

    let a =
        Tensor::from_vec(vec![1, 2], vec![1.0, 2.0]).expect("Tensor construction should succeed");
    let b =
        Tensor::from_vec(vec![1, 2], vec![3.0, 4.0]).expect("Tensor construction should succeed");

    // Registry: provide only Add -> FailingKernel so the executor reaches this node and fails there.
    let mut reg = KernelRegistry::new();
    assert!(
        reg.register(OpKind::Add, Box::new(FailingKernel)).is_none(),
        "Registering failing kernel should not replace an existing kernel"
    );

    let exec = Executor::new(reg);
    let err = exec.execute(&g, vec![(a_id, a), (b_id, b)]).unwrap_err();

    assert!(matches!(
        err,
        ExecutionError::KernelExecutionFailed { node, op, source }
            if node == add_id && op == OpKind::Add && matches!(source, KernelError::InvalidArguments)
    ));
}

/////////////////////////////////////////
// Additional Execution Stress-Tests   //
/////////////////////////////////////////

#[test]
fn executor_multiple_outputs() {
    // Graph:
    //   u = relu(x)
    //   v = relu(u)
    // outputs = [v, u]
    let mut g = Graph::new();
    let x_id = g.input_node(vec![1, 4]);
    let u_id = g.relu(x_id).expect("Operation should succeed");
    let v_id = g.relu(u_id).expect("Operation should succeed");

    g.set_output_node(v_id).expect("Operation should succeed");
    g.set_output_node(u_id).expect("Operation should succeed");

    let x_data = vec![-1.0, 2.0, -3.0, 4.0];
    let x = Tensor::from_vec(vec![1, 4], x_data.clone()).expect("Operation should succeed");

    let u_expected = relu_ref(&x_data);
    let v_expected = relu_ref(&u_expected);

    let exec = Executor::new(KernelRegistry::default());
    let outs = exec
        .execute(&g, vec![(x_id, x)])
        .expect("Operation should succeed");

    assert_eq!(outs.len(), 2);
    let (v_out, u_out) = (&outs[&v_id], &outs[&u_id]);

    // Must match output declaration order: [v, u]
    assert_eq!(v_out.shape(), &[1, 4]);
    assert_tensor_eq(v_out, &v_expected);

    assert_eq!(u_out.shape(), &[1, 4]);
    assert_tensor_eq(u_out, &u_expected);
}

#[test]
fn executor_output_is_input_identity_graph() {
    // Graph: output = x
    let mut g = Graph::new();
    let x_id = g.input_node(vec![2, 2]);
    g.set_output_node(x_id).expect("Operation should succeed");

    let x_data = vec![1.0, 2.0, 3.0, 4.0];
    let x = Tensor::from_vec(vec![2, 2], x_data.clone()).expect("Operation should succeed");

    let exec = Executor::new(KernelRegistry::default());
    let outs = exec
        .execute(&g, vec![(x_id, x)])
        .expect("Operation should succeed");

    assert_eq!(outs.len(), 1);
    let output_tensor: &Tensor = &outs[&x_id];

    assert_eq!(output_tensor.shape(), &[2, 2]);
    assert_tensor_eq(output_tensor, &x_data);
}

#[test]
fn executor_fanout_diamond_graph_shared_dependency() {
    // Graph:
    //   u = relu(x)
    //   v = relu(x)
    //   y = add(u, v)
    // output = y
    let mut g = Graph::new();
    let x_id = g.input_node(vec![1, 6]);

    let u_id = g.relu(x_id).expect("Operation should succeed");
    let v_id = g.relu(x_id).expect("Operation should succeed");
    let y_id = g.add(u_id, v_id).expect("Operation should succeed");
    g.set_output_node(y_id).expect("Operation should succeed");

    let x_data = vec![-3.0, -0.0, 0.0, 1.5, 2.0, -7.25];
    let x = Tensor::from_vec(vec![1, 6], x_data.clone()).expect("Operation should succeed");

    let relu = relu_ref(&x_data);
    let expected = add_ref(&relu, &relu);

    let exec = Executor::new(KernelRegistry::default());
    let outs = exec
        .execute(&g, vec![(x_id, x)])
        .expect("Operation should succeed");

    assert_eq!(outs.len(), 1);
    let output_tensor: &Tensor = &outs[&y_id];

    assert_eq!(output_tensor.shape(), &[1, 6]);
    assert_tensor_eq(output_tensor, &expected);
}

#[test]
fn executor_same_tensor_used_twice_add_x_x() {
    // Graph: y = add(x, x)
    let mut g = Graph::new();
    let x_id = g.input_node(vec![2, 3]);
    let y_id = g.add(x_id, x_id).expect("Operation should succeed");
    g.set_output_node(y_id).expect("Operation should succeed");

    let x_data = vec![1.0, -2.0, 3.5, 0.0, -1.25, 6.0];
    let x = Tensor::from_vec(vec![2, 3], x_data.clone()).expect("Operation should succeed");

    let expected = add_ref(&x_data, &x_data);

    let exec = Executor::new(KernelRegistry::default());
    let outs = exec
        .execute(&g, vec![(x_id, x)])
        .expect("Operation should succeed");

    assert_eq!(outs.len(), 1);
    let output_tensor: &Tensor = &outs[&y_id];

    assert_eq!(output_tensor.shape(), &[2, 3]);
    assert_tensor_eq(output_tensor, &expected);
}

#[test]
fn executor_chain_longer_than_one_op() {
    // Graph: relu(relu(relu(x)))
    let mut g = Graph::new();
    let x_id = g.input_node(vec![1, 5]);

    let a_id = g.relu(x_id).expect("Operation should succeed");
    let b_id = g.relu(a_id).expect("Operation should succeed");
    let c_id = g.relu(b_id).expect("Operation should succeed");
    g.set_output_node(c_id).expect("Operation should succeed");

    let x_data = vec![-1.0, -2.0, 0.0, 3.0, 4.0];
    let x = Tensor::from_vec(vec![1, 5], x_data.clone()).expect("Operation should succeed");

    let expected = relu_ref(&relu_ref(&relu_ref(&x_data)));

    let exec = Executor::new(KernelRegistry::default());
    let outs = exec
        .execute(&g, vec![(x_id, x)])
        .expect("Operation should succeed");

    assert_eq!(outs.len(), 1);
    let output_tensor: &Tensor = &outs[&c_id];

    assert_eq!(output_tensor.shape(), &[1, 5]);
    assert_tensor_eq(output_tensor, &expected);
}

#[test]
fn executor_unused_nodes_do_not_break_execution() {
    // Graph:
    //   main: y = relu(x) -> output
    //   dead subgraph: d = relu(z) (never marked as output)
    let mut g = Graph::new();
    let x_id = g.input_node(vec![1, 4]);
    let y_id = g.relu(x_id).expect("Operation should succeed");
    g.set_output_node(y_id).expect("Operation should succeed");

    let z_id = g.input_node(vec![1, 4]);
    let _dead = g.relu(z_id).expect("Operation should succeed");

    let x_data = vec![-1.0, 2.0, -3.0, 4.0];
    let z_data = vec![9.0, 9.0, 9.0, 9.0];

    let x = Tensor::from_vec(vec![1, 4], x_data.clone()).expect("Operation should succeed");
    let z = Tensor::from_vec(vec![1, 4], z_data).expect("Operation should succeed");

    let expected = relu_ref(&x_data);

    let exec = Executor::new(KernelRegistry::default());
    // Provide bindings for both input nodes. Executor requires all graph inputs to be bound.
    let outs = exec
        .execute(&g, vec![(z_id, z), (x_id, x)])
        .expect("Operation should succeed");

    assert_eq!(outs.len(), 1);
    let output_tensor: &Tensor = &outs[&y_id];

    assert_eq!(output_tensor.shape(), &[1, 4]);
    assert_tensor_eq(output_tensor, &expected);
}

////////////////////////////
// Additional Error Cases //
////////////////////////////

#[test]
fn executor_rejects_graph_with_no_outputs() {
    // Graph has an input but no outputs designated.
    // This will run and then fail when collecting outputs
    let mut g = Graph::new();
    let x_id = g.input_node(vec![1, 1]);

    let x = Tensor::from_vec(vec![1, 1], vec![1.0]).expect("Operation should succeed");
    let exec = Executor::new(KernelRegistry::default());

    let outs = exec
        .execute(&g, vec![(x_id, x)])
        .expect("Operation should succeed");
    assert!(outs.is_empty());
}

#[test]
fn executor_rejects_extra_binding_for_nonexistent_node_in_graph() {
    // Like the "foreign graph" test, but tighter: graph has one input x, output x.
    // Provide x plus another binding from a different graph.
    let mut g = Graph::new();
    let x_id = g.input_node(vec![1, 1]);
    g.set_output_node(x_id).expect("Operation should succeed");

    let x = Tensor::from_vec(vec![1, 1], vec![1.0]).expect("Operation should succeed");

    let mut other = Graph::new();
    let foreign_id = other.input_node(vec![1, 1]);
    let foreign = Tensor::from_vec(vec![1, 1], vec![2.0]).expect("Operation should succeed");

    let exec = Executor::new(KernelRegistry::default());
    let err = exec
        .execute(&g, vec![(x_id, x), (foreign_id, foreign)])
        .unwrap_err();

    assert!(matches!(
        err,
        ExecutionError::InvalidBindingNode { node } if node == foreign_id
    ));
}

#[test]
fn executor_rejects_binding_node_that_exists_but_is_not_an_input() {
    // Graph: x -> relu(x) -> out
    // Bind to relu node id (exists but is non-input)
    let mut g = Graph::new();
    let x_id = g.input_node(vec![1, 3]);
    let y_id = g.relu(x_id).expect("Operation should succeed");
    g.set_output_node(y_id).expect("Operation should succeed");

    let bogus = Tensor::zeros(vec![1, 3]).expect("Operation should succeed");

    let exec = Executor::new(KernelRegistry::default());
    let err = exec.execute(&g, vec![(y_id, bogus)]).unwrap_err();

    assert!(matches!(
        err,
        ExecutionError::BindingToNonInputNode { node, op }
            if node == y_id && op == OpKind::ReLU
    ));
}

#[test]
fn executor_missing_irrelevant_input() {
    // This locks down current executor behavior:
    // it requires all Graph::inputs() to be bound, even if some are not reachable
    // from the output(s).
    //
    // Graph:
    //   y = relu(x) -> output
    //   z is a second input but unused.
    let mut g = Graph::new();
    let x_id = g.input_node(vec![1, 2]);
    let y_id = g.relu(x_id).expect("Operation should succeed");
    g.set_output_node(y_id).expect("Operation should succeed");

    let z_id = g.input_node(vec![1, 2]); // unused input node

    let x = Tensor::from_vec(vec![1, 2], vec![1.0, -2.0]).expect("Operation should succeed");

    let exec = Executor::new(KernelRegistry::default());
    let err = exec.execute(&g, vec![(x_id, x)]).unwrap_err();

    assert!(matches!(err, ExecutionError::MissingInput { node } if node == z_id));
}

#[test]
fn executor_output_node_set_twice_returns_one_output() {
    // Graph: output list will not include duplicates; set_output_node() does not dedupe.
    // This test locks down executor behavior for duplicate outputs:
    // it should return 1 tensor despite marking duplicate output nodes.
    let mut g = Graph::new();
    let x_id = g.input_node(vec![1, 3]);
    let y_id = g.relu(x_id).expect("Operation should succeed");

    g.set_output_node(y_id).expect("Operation should succeed");
    g.set_output_node(y_id).expect("Operation should succeed"); // duplicate on purpose

    let x_data = vec![-1.0, 2.0, -3.0];
    let x = Tensor::from_vec(vec![1, 3], x_data.clone()).expect("Operation should succeed");
    let expected = relu_ref(&x_data);

    let exec = Executor::new(KernelRegistry::default());
    let outs = exec
        .execute(&g, vec![(x_id, x)])
        .expect("Operation should succeed");

    assert_eq!(outs.len(), 1);
    let output_tensor: &Tensor = &outs[&y_id];

    assert_eq!(output_tensor.shape(), &[1, 3]);
    assert_tensor_eq(output_tensor, &expected);
}

#[test]
fn executor_multiple_outputs_from_disjoint_subgraphs() {
    // Graph:
    //   y = relu(a)
    //   z = relu(b)
    // outputs = [y, z]
    let mut g = Graph::new();
    let a_id = g.input_node(vec![1, 4]);
    let b_id = g.input_node(vec![1, 4]);

    let y_id: NodeId = g.relu(a_id).expect("Tensor creation should not fail");
    let z_id: NodeId = g.relu(b_id).expect("Tensor creation should not fail");

    g.set_output_node(y_id).expect("Operation should succeed");
    g.set_output_node(z_id).expect("Operation should succeed");

    let a_data = vec![-1.0, 2.0, -3.0, 4.0];
    let b_data = vec![5.0, -6.0, 7.0, -8.0];

    let a = Tensor::from_vec(vec![1, 4], a_data.clone()).expect("Operation should succeed");
    let b = Tensor::from_vec(vec![1, 4], b_data.clone()).expect("Operation should succeed");

    let y_expected = relu_ref(&a_data);
    let z_expected = relu_ref(&b_data);

    let exec = Executor::new(KernelRegistry::default());
    let outs: HashMap<NodeId, Tensor> = exec
        .execute(&g, vec![(b_id, b), (a_id, a)])
        .expect("Execution should not fail");

    assert_eq!(outs.len(), 2);

    assert_eq!(outs[&y_id].shape(), &[1, 4]);
    assert_tensor_eq(&outs[&y_id], &y_expected);

    assert_eq!(outs[&z_id].shape(), &[1, 4]);
    assert_tensor_eq(&outs[&z_id], &z_expected);
}

#[test]
fn executor_matmul_then_add_then_relu_more_shapes() {
    // A slightly different shape combo than the existing test, to catch indexing bugs:
    // A: 1x3, B: 3x4 => mm: 1x4, then add with C: 1x4, then relu
    let mut g = Graph::new();
    let a_id = g.input_node(vec![1, 3]);
    let b_id = g.input_node(vec![3, 4]);
    let c_id = g.input_node(vec![1, 4]);

    let mm_id = g.matmul(a_id, b_id).expect("Operation should succeed");
    let add_id = g.add(mm_id, c_id).expect("Operation should succeed");
    let out_id = g.relu(add_id).expect("Operation should succeed");
    g.set_output_node(out_id).expect("Operation should succeed");

    let a_data = vec![1.0, -2.0, 3.0];
    let b_data = vec![
        1.0, 2.0, 3.0, 4.0, //
        -1.0, -2.0, -3.0, -4.0, //
        0.5, 0.25, -0.5, -0.25,
    ];
    let c_data = vec![-10.0, 0.0, 10.0, -1.0];

    let mm = matmul_ref(&a_data, &b_data, 1, 3, 4);
    let add = add_ref(&mm, &c_data);
    let expected = relu_ref(&add);

    let a = Tensor::from_vec(vec![1, 3], a_data).expect("Operation should succeed");
    let b = Tensor::from_vec(vec![3, 4], b_data).expect("Operation should succeed");
    let c = Tensor::from_vec(vec![1, 4], c_data).expect("Operation should succeed");

    let exec = Executor::new(KernelRegistry::default());
    let outs = exec
        .execute(&g, vec![(c_id, c), (a_id, a), (b_id, b)])
        .expect("Operation should succeed");

    assert_eq!(outs.len(), 1);
    let output_tensor: &Tensor = &outs[&out_id];

    assert_eq!(output_tensor.shape(), &[1, 4]);
    assert_tensor_eq(output_tensor, &expected);
}

#[test]
fn executor_add_chain() {
    // Graph:
    //   s1 = add(a, b)
    //   s2 = add(s1, c)
    //   out = s2
    let mut g = Graph::new();
    let a_id = g.input_node(vec![2, 2]);
    let b_id = g.input_node(vec![2, 2]);
    let c_id = g.input_node(vec![2, 2]);

    let s1_id = g.add(a_id, b_id).expect("Add node s1 should be added");
    let s2_id = g.add(s1_id, c_id).expect("Add node s2 should be added");
    g.set_output_node(s2_id)
        .expect("Setting output should succeed");

    let a_data = vec![1.0, 2.0, 3.0, 4.0];
    let b_data = vec![10.0, 20.0, 30.0, 40.0];
    let c_data = vec![100.0, 200.0, 300.0, 400.0];

    let s1_expected = add_ref(&a_data, &b_data);
    let expected = add_ref(&s1_expected, &c_data);

    let a = Tensor::from_vec(vec![2, 2], a_data).expect("Tensor construction should succeed");
    let b = Tensor::from_vec(vec![2, 2], b_data).expect("Tensor construction should succeed");
    let c = Tensor::from_vec(vec![2, 2], c_data).expect("Tensor construction should succeed");

    let exec = Executor::new(KernelRegistry::default());
    let outs = exec
        .execute(&g, vec![(c_id, c), (a_id, a), (b_id, b)])
        .expect("Execution should succeed");

    assert_eq!(outs.len(), 1);
    let output_tensor: &Tensor = &outs[&s2_id];

    assert_eq!(output_tensor.shape(), &[2, 2]);
    assert_tensor_eq(output_tensor, &expected);
}

#[test]
fn executor_chain_relu_matmul_add() {
    // Graph:
    //   relu_a = relu(a)
    //   relu_b = relu(b)
    //   mm     = matmul(relu_a, relu_b)
    //   out    = add(mm, c)
    //
    // Shapes:
    //   a: [2, 3]
    //   b: [3, 2]
    //   c: [2, 2]
    let mut g = Graph::new();
    let a_id = g.input_node(vec![2, 3]);
    let b_id = g.input_node(vec![3, 2]);
    let c_id = g.input_node(vec![2, 2]);

    let relu_a_id = g.relu(a_id).expect("ReLU(A) should succeed");
    let relu_b_id = g.relu(b_id).expect("ReLU(B) should succeed");
    let mm_id = g
        .matmul(relu_a_id, relu_b_id)
        .expect("Matmul after ReLUs should succeed");
    let add_id = g.add(mm_id, c_id).expect("Final add should succeed");
    g.set_output_node(add_id)
        .expect("Setting output should succeed");

    let a_data = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0];
    let b_data = vec![-7.0, 8.0, 9.0, -10.0, 11.0, 12.0];
    let c_data = vec![0.5, 1.5, 2.5, 3.5];

    let relu_a = relu_ref(&a_data);
    let relu_b = relu_ref(&b_data);
    let mm = matmul_ref(&relu_a, &relu_b, 2, 3, 2);
    let expected = add_ref(&mm, &c_data);

    let a = Tensor::from_vec(vec![2, 3], a_data).expect("Tensor construction should succeed");
    let b = Tensor::from_vec(vec![3, 2], b_data).expect("Tensor construction should succeed");
    let c = Tensor::from_vec(vec![2, 2], c_data).expect("Tensor construction should succeed");

    let exec = Executor::new(KernelRegistry::default());
    let outs = exec
        .execute(&g, vec![(b_id, b), (c_id, c), (a_id, a)])
        .expect("Execution should succeed");

    assert_eq!(outs.len(), 1);
    let output_tensor: &Tensor = &outs[&add_id];

    assert_eq!(output_tensor.shape(), &[2, 2]);
    assert_tensor_eq(output_tensor, &expected);
}

#[test]
fn executor_feedforward_neural_network() {
    // Simulate a small feedforward network using repeated layer construction.
    //
    // Layout:
    //   input layer:  3 nodes
    //   hidden layer: 2 nodes
    //   hidden layer: 2 nodes
    //   output layer: 2 nodes
    //
    // Each node in the next layer is computed as:
    //   relu(add(add(prev[0], prev[1]), prev[2]...))
    //
    // All tensors use the same shape so we can express the whole network with Add + ReLU.

    let mut g = Graph::new();

    let input_width = 3;
    let hidden_widths = [2, 2];
    let output_width = 2;
    let shape = vec![1, 4];

    // Input layer
    let mut current_layer: Vec<NodeId> = (0..input_width)
        .map(|_| g.input_node(shape.clone()))
        .collect();

    let input_ids = current_layer.clone();

    // Concrete input data
    let input_data = vec![
        vec![1.0, -2.0, 3.0, -4.0],
        vec![0.5, 1.5, -2.5, 3.5],
        vec![10.0, -20.0, 30.0, -40.0],
    ];

    // Reference values for the current layer
    let mut current_expected = input_data.clone();

    // Build hidden layers in a loop
    for &layer_width in &hidden_widths {
        let mut next_layer = Vec::new();
        let mut next_expected = Vec::new();

        for _ in 0..layer_width {
            let mut acc_id = current_layer[0];
            let mut acc_expected = current_expected[0].clone();

            for i in 1..current_layer.len() {
                acc_id = g
                    .add(acc_id, current_layer[i])
                    .expect("Add node should be added");
                acc_expected = add_ref(&acc_expected, &current_expected[i]);
            }

            let out_id = g.relu(acc_id).expect("ReLU node should be added");
            let out_expected = relu_ref(&acc_expected);

            next_layer.push(out_id);
            next_expected.push(out_expected);
        }

        current_layer = next_layer;
        current_expected = next_expected;
    }

    // Output layer
    let mut output_ids = Vec::new();
    let mut output_expected = Vec::new();

    for _ in 0..output_width {
        let mut acc_id = current_layer[0];
        let mut acc_expected = current_expected[0].clone();

        for i in 1..current_layer.len() {
            acc_id = g
                .add(acc_id, current_layer[i])
                .expect("Add node should be added");
            acc_expected = add_ref(&acc_expected, &current_expected[i]);
        }

        let out_id = g.relu(acc_id).expect("ReLU node should be added");
        let out_expected = relu_ref(&acc_expected);

        g.set_output_node(out_id)
            .expect("Setting output should succeed");

        output_ids.push(out_id);
        output_expected.push(out_expected);
    }

    let bindings: Vec<(NodeId, Tensor)> = input_ids
        .iter()
        .zip(input_data)
        .map(|(&id, data)| {
            (
                id,
                Tensor::from_vec(shape.clone(), data).expect("Tensor construction should succeed"),
            )
        })
        .collect();

    let exec = Executor::new(KernelRegistry::default());
    let outs = exec
        .execute(&g, bindings)
        .expect("Execution should succeed");

    assert_eq!(outs.len(), output_width);

    for (out_id, expected) in output_ids.iter().zip(output_expected.iter()) {
        let output_tensor: &Tensor = &outs[out_id];
        assert_eq!(output_tensor.shape(), &[1, 4]);
        assert_tensor_eq(output_tensor, expected);
    }
}

#[test]
fn executor_diamond_graph() {
    //       relu(A)
    //     /        \
    //    A          matmul(relu(A), relu(B)) -> add(..., C)
    //    B          /
    //     \        /
    //      relu(B)
    // and C is another input to add

    let mut g = Graph::new();

    let a_id = g.input_node(vec![2, 3]);
    let b_id = g.input_node(vec![3, 2]);
    let c_id = g.input_node(vec![2, 2]);

    let ra_id = g.relu(a_id).expect("Valid ReLU operation should succeed");
    let rb_id = g.relu(b_id).expect("Valid ReLU operation should succeed");
    let mm_id = g
        .matmul(ra_id, rb_id)
        .expect("Valid Matmul operation should succeed");
    let add_id = g
        .add(mm_id, c_id)
        .expect("Valid addition operation should succeed");

    g.set_output_node(add_id)
        .expect("Setting output node should succeed");

    let a_data = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0];
    let b_data = vec![-7.0, 8.0, 9.0, -10.0, 11.0, 12.0];
    let c_data = vec![0.5, 1.5, 2.5, 3.5];

    let ra_expected = relu_ref(&a_data);
    let rb_expected = relu_ref(&b_data);
    let mm_expected = matmul_ref(&ra_expected, &rb_expected, 2, 3, 2);
    let add_expected = add_ref(&mm_expected, &c_data);

    let a = Tensor::from_vec(vec![2, 3], a_data).expect("Tensor construction should succeed");
    let b = Tensor::from_vec(vec![3, 2], b_data).expect("Tensor construction should succeed");
    let c = Tensor::from_vec(vec![2, 2], c_data).expect("Tensor construction should succeed");

    let exec = Executor::new(KernelRegistry::default());
    let outs = exec
        .execute(&g, vec![(a_id, a), (b_id, b), (c_id, c)])
        .expect("Diamond graph execution should succeed");

    assert_eq!(
        outs.len(),
        1,
        "Executor should return exactly one declared output"
    );

    let output = outs
        .get(&add_id)
        .expect("Declared output node should be present in executor results");

    assert_eq!(output.shape(), &[2, 2]);
    assert_tensor_eq(output, &add_expected);
}
