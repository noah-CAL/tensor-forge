//! Graph integration tests specifically for topological sorting.
use tensor_forge::graph::Graph;
use tensor_forge::node::NodeId;

//////////////////////////
// Topological Ordering //
//////////////////////////
fn pos(order: &[NodeId], id: NodeId) -> usize {
    order
        .iter()
        .position(|&x| x == id)
        .expect("NodeId must appear in topo_sort output")
}

#[test]
fn topo_sort_linear_orders_dependencies_first() {
    // A -> relu(A) -> relu(relu(A))
    let mut g = Graph::new();

    let a = g.input_node(vec![2, 3]);
    let b = g.relu(a).unwrap();
    let c = g.relu(b).unwrap();

    let order = g
        .topo_sort()
        .expect("Valid DAG should have a valid topo_sort");

    assert!(pos(&order, a) < pos(&order, b));
    assert!(pos(&order, b) < pos(&order, c));
    assert_eq!(
        order.len(),
        g.num_nodes(),
        "topo_sort must include every node exactly once"
    );
}

#[test]
fn topo_sort_diamond_orders_all_dependencies_first() {
    //       relu(A)
    //     /        \
    //    A          matmul(relu(A), relu(B)) -> add(..., C)
    //    B          /
    //     \        /
    //      relu(B)
    // and C is another input to add

    let mut g = Graph::new();

    let a = g.input_node(vec![2, 3]);
    let b = g.input_node(vec![3, 2]);
    let c = g.input_node(vec![2, 2]);

    let ra = g.relu(a).expect("Valid ReLU operation"); // shape [2,3]
    let rb = g.relu(b).expect("Valid ReLU operation"); // shape [3,2]
    let mm = g.matmul(ra, rb).expect("Valid Matmul operation"); // shape [2,2]
    let add = g.add(mm, c).expect("Valid addition operation"); // shape [2,2]

    let order = g.topo_sort().expect("Valid DAG should topo_sort");

    // Every edge u -> v must satisfy pos(u) < pos(v)
    assert!(pos(&order, a) < pos(&order, ra));
    assert!(pos(&order, b) < pos(&order, rb));
    assert!(pos(&order, ra) < pos(&order, mm));
    assert!(pos(&order, rb) < pos(&order, mm));
    assert!(pos(&order, mm) < pos(&order, add));
    assert!(pos(&order, c) < pos(&order, add));

    assert_eq!(
        order.len(),
        g.num_nodes(),
        "topo_sort must include every node exactly once"
    );
}

#[test]
fn topo_sort_is_deterministic_for_independent_nodes() {
    // Topological sort should be deterministic for a specific graph creation.
    //
    // Easiest way to exercise this without relying on HashMap iteration:
    // create 3 independent inputs and ensure topo_sort returns them in ascending NodeId.
    let mut g = Graph::new();

    let n0 = g.input_node(vec![1, 1]);
    let n1 = g.input_node(vec![1, 1]);
    let n2 = g.input_node(vec![1, 1]);

    let order = g.topo_sort().expect("Valid DAG should topo_sort");

    // Since they have no deps, tie-break should be NodeId ascending.
    let p0 = pos(&order, n0);
    let p1 = pos(&order, n1);
    let p2 = pos(&order, n2);

    assert!(
        p0 < p1 && p1 < p2,
        "independent nodes must appear in ascending NodeId order"
    );
}

#[test]
fn topo_sort_is_deterministic_for_multiple_ready_ops() {
    // Build:
    //   a, b are inputs
    //   r0 = relu(a)
    //   r1 = relu(b)
    //
    // Once a and b are done, both relus are ready.
    // Determinism rule should order r0 and r1 by ascending NodeId.
    let mut g = Graph::new();

    let a = g.input_node(vec![2, 2]);
    let b = g.input_node(vec![2, 2]);

    let r0 = g.relu(a).unwrap();
    let r1 = g.relu(b).unwrap();

    let order = g.topo_sort().expect("Valid DAG should topo_sort");

    // deps first
    assert!(pos(&order, a) < pos(&order, r0));
    assert!(pos(&order, b) < pos(&order, r1));

    // tie-break between r0 and r1 (both become ready after their respective inputs)
    let pr0 = pos(&order, r0);
    let pr1 = pos(&order, r1);

    // We don't assume which relu corresponds to smaller NodeId beyond "ascending NodeId".
    // Since NodeId is monotonic, r0 was created before r1, so:
    assert!(
        pr0 < pr1,
        "ready ops must be processed in ascending NodeId order"
    );
}
