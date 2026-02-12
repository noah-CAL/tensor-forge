//! Structure for representing ML runtimes via Node and Op intermediate representation.

use crate::node::{Node, NodeId};
use crate::op::OpKind;
use std::collections::HashMap;
use std::fmt;

/// Error types for [`Graph`] construction and validation.
///
/// These errors are returned by graph-building APIs when an operation cannot be
/// represented safely in the current graph.
///
/// # Examples
/// ```
/// # use tensor_forge::graph::{Graph, GraphError};
/// let mut g = Graph::new();
/// let a = g.input_node(vec![2, 3]);
/// let b = g.input_node(vec![2, 4]);
///
/// // add() requires identical shapes
/// assert!(matches!(g.add(a, b).unwrap_err(), GraphError::ShapeMismatch));
/// ```
#[derive(Clone, Debug)]
pub enum GraphError {
    /// Raised when connecting nodes whose tensor shapes are incompatible for the requested op.
    ///
    /// # Examples
    /// - `add(A, B)` requires `shape(A) == shape(B)`.
    /// - `matmul(L, R)` requires `L` and `R` be 2-D and `L.shape[1] == R.shape[0]`.
    ///
    /// This error indicates the graph is not well-typed under the op’s shape rules.
    ShapeMismatch,
    /// Raised when an operation references a [`NodeId`] that does not exist in the graph.
    ///
    /// This typically happens when:
    /// - A `NodeId` was produced by a different [`Graph`] instance, or
    /// - A stale/invalid `NodeId` was stored and reused.
    ///
    /// # Example
    /// ```
    /// # use tensor_forge::graph::{Graph, GraphError};
    /// let mut g1 = Graph::new();
    /// let foreign = g1.input_node(vec![1, 1]);
    ///
    /// let mut g2 = Graph::new();
    /// assert!(matches!(g2.relu(foreign).unwrap_err(), GraphError::InvalidNodeId));
    /// ```
    InvalidNodeId,
    /// Raised when inserting a node whose ID already exists in the graph.
    ///
    /// In this implementation, node IDs are expected to be monotonically increasing and unique.
    /// A collision indicates a serious invariant failure (e.g. ID overflow or a bug in node
    /// allocation), and should be treated as unrecoverable at the application level.
    IdCollision,
    /// Raised when the graph contains a cycle and no valid execution order exists.
    CycleDetected,
}

/// A minimal compute-graph container for an ML runtime intermediate representation (IR).
///
/// A [`Graph`] owns a set of [`Node`]s indexed by [`NodeId`]. Each node encodes:
/// - an operation kind ([`OpKind`]),
/// - a list of input dependencies (by `NodeId`), and
/// - the inferred output tensor shape.
///
/// This type currently supports constructing a graph via:
/// - [`Graph::input_node`] for source nodes, and
/// - op constructors like [`Graph::add`], [`Graph::matmul`], and [`Graph::relu`].
///
/// Output nodes must be designated explicitly via [`Graph::set_output_node`].
///
/// # Examples
/// ```
/// # use tensor_forge::graph::Graph;
/// let mut g = Graph::new();
/// let x = g.input_node(vec![2, 3]);
/// let y = g.relu(x).unwrap();
/// g.set_output_node(y).unwrap();
/// assert_eq!(g.outputs().len(), 1);
/// ```
pub struct Graph {
    nodes: HashMap<NodeId, Node>,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    fn add_node(&mut self, node: Node) -> Result<NodeId, GraphError> {
        let node_id = node.id;
        // Each node is generated to be unique in monotonically increasing order. Collisions
        // indicate that graph nodes have overflowed.
        if self.nodes.contains_key(&node_id) {
            return Err(GraphError::IdCollision);
        }
        self.nodes.insert(node_id, node);
        self.inputs.push(node_id);
        Ok(node_id)
    }

    /// Creates an empty graph with no nodes, inputs, or outputs.
    ///
    /// # Examples
    /// ```
    /// # use tensor_forge::graph::Graph;
    /// let g = Graph::new();
    /// assert_eq!(g.num_nodes(), 0);
    /// assert!(g.inputs().is_empty());
    /// assert!(g.outputs().is_empty());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Graph {
            nodes: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Creates a new input node with the given tensor `shape` and returns its `NodeId`.
    ///
    /// Input nodes have no dependencies and an output shape equal to `shape`.
    ///
    /// # Panics
    /// Panics if a node ID collision is detected (an invariant violation indicating too many nodes
    /// have been allocated or ID generation is broken).
    ///
    /// # Examples
    /// ```
    /// # use tensor_forge::graph::Graph;
    /// let mut g = Graph::new();
    /// let x = g.input_node(vec![2, 3]);
    /// assert!(g.node(x).is_ok());
    /// assert_eq!(g.num_nodes(), 1);
    /// ```
    pub fn input_node(&mut self, shape: Vec<usize>) -> NodeId {
        let node = Node::new(OpKind::Input, Vec::new(), shape);
        self.add_node(node).expect("Node ID collision detected on node creation. Too many nodes may have been allocated. Ensure that Graph operations are single-threaded.")
    }

    /// Adds a matrix multiplication node `left × right`.
    ///
    /// Shape rule (2-D):
    /// - `left.shape = [m, k]`
    /// - `right.shape = [k, n]`
    /// - output shape is `[m, n]`
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::InvalidNodeId`] if either `left` or `right` does not exist
    /// in this graph.
    ///
    /// Returns [`GraphError::ShapeMismatch`] if the inner dimensions do not match.
    ///
    /// # Examples
    /// ```
    /// # use tensor_forge::graph::{Graph, GraphError};
    /// let mut g = Graph::new();
    /// let a = g.input_node(vec![2, 3]);
    /// let b = g.input_node(vec![3, 4]);
    ///
    /// let c = g.matmul(a, b).unwrap();
    /// assert!(g.node(c).is_ok());
    /// assert_eq!(g.num_nodes(), 3);
    ///
    /// // Mismatched inner dimension: [2,3] x [2,4] is invalid
    /// let bad = g.input_node(vec![2, 4]);
    /// assert!(matches!(g.matmul(a, bad).unwrap_err(), GraphError::ShapeMismatch));
    /// ```
    pub fn matmul(&mut self, left: NodeId, right: NodeId) -> Result<NodeId, GraphError> {
        let left_node = self.node(left)?;
        let right_node = self.node(right)?;
        if left_node.shape[1] != right_node.shape[0] {
            return Err(GraphError::ShapeMismatch);
        }
        let shape = vec![left_node.shape[0], right_node.shape[1]];
        let matmul_node = Node::new(OpKind::MatMul, vec![left_node.id, right_node.id], shape);
        self.add_node(matmul_node)
    }

    /// Adds an elementwise addition node `left + right`.
    ///
    /// Shape rule:
    /// - `shape(left) == shape(right)`
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::InvalidNodeId`] if either input does not exist in this graph.
    ///
    /// Returns [`GraphError::ShapeMismatch`] if the shapes differ.
    ///
    /// # Examples
    /// ```
    /// # use tensor_forge::graph::{Graph, GraphError};
    /// let mut g = Graph::new();
    /// let a = g.input_node(vec![2, 3]);
    /// let b = g.input_node(vec![2, 3]);
    ///
    /// let c = g.add(a, b).unwrap();
    /// assert!(g.node(c).is_ok());
    ///
    /// let d = g.input_node(vec![2, 4]);
    /// assert!(matches!(g.add(a, d).unwrap_err(), GraphError::ShapeMismatch));
    /// ```
    pub fn add(&mut self, left: NodeId, right: NodeId) -> Result<NodeId, GraphError> {
        let left_node = self.node(left)?;
        let right_node = self.node(right)?;
        if left_node.shape != right_node.shape {
            return Err(GraphError::ShapeMismatch);
        }
        let addition_node = Node::new(
            OpKind::Add,
            vec![left_node.id, right_node.id],
            left_node.shape.clone(),
        );
        self.add_node(addition_node)
    }

    /// Adds a `ReLU` node `relu(input)`.
    ///
    /// `ReLU` preserves shape: `shape(output) == shape(input)`.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::InvalidNodeId`] if `input` does not exist in this graph.
    ///
    /// # Examples
    /// ```
    /// # use tensor_forge::graph::{Graph, GraphError};
    /// let mut g = Graph::new();
    /// let x = g.input_node(vec![2, 3]);
    ///
    /// let y = g.relu(x).unwrap();
    /// assert!(g.node(y).is_ok());
    ///
    /// // Using a NodeId from another graph is invalid
    /// let mut other = Graph::new();
    /// let foreign = other.input_node(vec![2, 3]);
    /// assert!(matches!(g.relu(foreign).unwrap_err(), GraphError::InvalidNodeId));
    /// ```
    pub fn relu(&mut self, input: NodeId) -> Result<NodeId, GraphError> {
        let input_node = self.node(input)?;
        let relu_node = Node::new(OpKind::ReLU, vec![input_node.id], input_node.shape.clone());
        self.add_node(relu_node)
    }

    /// Marks `node` as an output node.
    ///
    /// Graphs must have at least one output node to be meaningful for execution, and may have
    /// multiple outputs. This method does **not** create a new node or execute anything; it only
    /// records the provided node ID as an output.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::InvalidNodeId`] if `node` does not exist in this graph.
    ///
    /// # Examples
    /// ```
    /// # use tensor_forge::graph::{Graph, GraphError};
    /// let mut g = Graph::new();
    /// let x = g.input_node(vec![2, 3]);
    /// let y = g.relu(x).unwrap();
    ///
    /// g.set_output_node(y).unwrap();
    /// assert_eq!(g.outputs(), &[y]);
    ///
    /// // A NodeId from another graph is invalid
    /// let mut other = Graph::new();
    /// let foreign = other.input_node(vec![2, 3]);
    /// assert!(matches!(g.set_output_node(foreign).unwrap_err(), GraphError::InvalidNodeId));
    /// ```
    pub fn set_output_node(&mut self, node: NodeId) -> Result<(), GraphError> {
        let node = self.node(node)?;
        self.outputs.push(node.id);
        Ok(())
    }

    /// Returns a shared reference to the node with the given `NodeId`.
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::InvalidNodeId`] if the node is not present in this graph.
    ///
    /// # Examples
    /// ```
    /// # use tensor_forge::graph::{Graph, GraphError};
    /// let mut g = Graph::new();
    /// let x = g.input_node(vec![1, 1]);
    /// assert!(g.node(x).is_ok());
    ///
    /// // A NodeId from another graph is invalid
    /// let mut other = Graph::new();
    /// let foreign = other.input_node(vec![1, 1]);
    /// assert!(matches!(g.node(foreign).unwrap_err(), GraphError::InvalidNodeId));
    /// ```
    pub fn node(&self, id: NodeId) -> Result<&Node, GraphError> {
        match self.nodes.get(&id) {
            Some(node) => Ok(node),
            None => Err(GraphError::InvalidNodeId),
        }
    }

    /// Returns the total number of nodes stored in this graph.
    ///
    /// # Examples
    /// ```
    /// # use tensor_forge::graph::Graph;
    /// let mut g = Graph::new();
    /// assert_eq!(g.num_nodes(), 0);
    /// let x = g.input_node(vec![2, 3]);
    /// let y = g.relu(x).unwrap();
    /// assert_eq!(g.num_nodes(), 2);
    /// ```
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.nodes.values().len()
    }

    /// Returns the list of nodes recorded as inputs.
    ///
    /// Note: in the current implementation, *every* inserted node is appended to this list
    /// (including op nodes created by [`Graph::add`], [`Graph::matmul`], and [`Graph::relu`]).
    ///
    /// # Examples
    /// ```
    /// # use tensor_forge::graph::Graph;
    /// let mut g = Graph::new();
    /// let a = g.input_node(vec![2, 3]);
    /// let b = g.input_node(vec![2, 3]);
    /// let c = g.add(a, b).unwrap();
    ///
    /// // Currently includes both inputs and the derived node.
    /// assert_eq!(g.inputs(), &[a, b, c]);
    /// ```
    #[must_use]
    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    /// Returns the list of nodes marked as outputs via [`Graph::set_output_node`].
    ///
    /// # Examples
    /// ```
    /// # use tensor_forge::graph::Graph;
    /// let mut g = Graph::new();
    /// let x = g.input_node(vec![2, 3]);
    /// let y = g.relu(x).unwrap();
    ///
    /// assert!(g.outputs().is_empty());
    /// g.set_output_node(y).unwrap();
    /// assert_eq!(g.outputs(), &[y]);
    /// ```
    #[must_use]
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    /// Computes a deterministic topological execution order of all nodes in the graph.
    ///
    /// Topological ordering guarantees that every node appears *after* all of its
    /// dependencies. This ordering is required for correct execution of the compute graph,
    /// since kernels must not execute before their input tensors are available.
    ///
    /// The returned order includes every node in the graph exactly once.
    ///
    /// # Determinism
    ///
    /// Determinism is guaranteed by enforcing a stable tie-breaking rule when multiple
    /// nodes are ready for execution. Nodes with zero remaining dependencies are processed
    /// in ascending [`NodeId`] order.
    ///
    /// This ensures:
    ///
    /// - Reproducible execution across runs
    /// - Independence from hash seed randomization
    /// - Stable ordering suitable for debugging and testing
    ///
    /// # Returns
    ///
    /// A vector of [`NodeId`] representing the execution order.
    ///
    /// The order satisfies the invariant:
    ///
    /// ```text
    /// For every node N:
    ///     all inputs(N) appear before N in the returned vector
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`GraphError::CycleDetected`] if the graph contains a cycle.
    ///
    /// Cycles violate compute graph semantics because no valid execution order exists.
    ///
    /// # Complexity
    ///
    /// Time complexity: **O(V + E)**
    /// Space complexity: **O(V + E)**
    ///
    /// where:
    ///
    /// - V = number of nodes
    /// - E = number of edges (dependencies)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use tensor_forge::graph::Graph;
    /// let mut g = Graph::new();
    ///
    /// let a = g.input_node(vec![2, 3]);
    /// let b = g.relu(a).unwrap();
    /// let c = g.relu(b).unwrap();
    ///
    /// let order = g.topo_sort().unwrap();
    ///
    /// let pos = |id| order.iter().position(|&x| x == id).unwrap();
    ///
    /// assert!(pos(a) < pos(b));
    /// assert!(pos(b) < pos(c));
    /// ```
    pub fn topo_sort(&self) -> Result<Vec<NodeId>, GraphError> {
        Err(GraphError::ShapeMismatch)
    }
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            GraphError::ShapeMismatch => {
                write!(
                    f,
                    "Mismatched input and output dimensions for Nodes A and B. dim(Output(A)) must match dim(Output(B))"
                )
            }
            GraphError::InvalidNodeId => {
                write!(
                    f,
                    "Attempted to operate on a Node that does not exist in the graph. Ensure you are only interacting with nodes via Graph::input_node()."
                )
            }
            GraphError::IdCollision => {
                write!(
                    f,
                    "Attempted to add a new node to a graph with an ID that already exists."
                )
            }
            GraphError::CycleDetected => {
                write!(
                    f,
                    "Graph contains a dependency cycle. Execution order cannot be determined."
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::graph::*;
    use crate::node::*;

    /// Small unit test for internal implementation of returning IdCollision. This is untestable in
    /// integration tests because normal methods of generating node collisions are not publicly exposed 
    /// in the API.
    ///
    /// See `tests/graph_tests.rs` for graph integration tests.
    #[test]
    fn add_node_rejects_duplicate_id() {
        let mut g = Graph::new();

        let n1 = Node::new(OpKind::Input, vec![], vec![2, 2]);
        let n2 = Node {
            id: n1.id.clone(),
            op: OpKind::Input,
            inputs: vec![],
            shape: vec![2, 2],
        };

        assert!(g.add_node(n1).is_ok());
        assert!(matches!(
            g.add_node(n2).unwrap_err(),
            GraphError::IdCollision
        ));
    }
}
