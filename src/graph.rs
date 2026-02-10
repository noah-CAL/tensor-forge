//! Structure for representing ML runtimes via Node and Op intermediate representation.

use crate::node::{Node, NodeId};
use crate::op::OpKind;
use std::collections::HashMap;
use std::fmt;

/// Error types for Graph construction and execution.
#[derive(Clone, Debug)]
pub enum GraphError {
    /// Raised if attempting to concatenate input/output nodes
    /// with incorrect tensor dimensions.
    ///
    /// In general for two nodes A and B:
    /// Output(A) == Input(B)
    ShapeMismatch,
    /// Raised if attempting to connect nodes that do not exist in the graph.
    ///
    /// # Example
    /// ```
    /// # use tensor_forge::graph::{Graph, GraphError};
    /// let mut alt_graph = Graph::new();
    /// let fake_node = alt_graph.input_node(vec![1, 1]);
    ///
    /// let mut graph = Graph::new();
    /// let result = graph.relu(fake_node);
    /// assert!(matches!(result.unwrap_err(), GraphError::InvalidNodeId));
    ///
    /// let result = graph.add(fake_node, fake_node);
    /// assert!(matches!(result.unwrap_err(), GraphError::InvalidNodeId));
    ///
    /// let result = graph.matmul(fake_node, fake_node);
    /// assert!(matches!(result.unwrap_err(), GraphError::InvalidNodeId));
    /// ```
    InvalidNodeId,
    /// Raised if attempting to add a new node with the same ID.
    ///
    /// Note that this is unlikely to be raised except in the case of overflowing
    /// the graph by failing a new node allocation
    IdCollision,
}

/// Graph stores the nodes for the ML runtime.
///
/// # Examples
/// #todo
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

    #[must_use]
    pub fn new() -> Self {
        Graph {
            nodes: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn input_node(&mut self, shape: Vec<usize>) -> NodeId {
        let node = Node::new(OpKind::Input, Vec::new(), shape);
        self.add_node(node).expect("Node ID collision detected on node creation. Too many nodes may have been allocated. Ensure that Graph operations are single-threaded.")
    }

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

    pub fn relu(&mut self, input: NodeId) -> Result<NodeId, GraphError> {
        let input_node = self.node(input)?;
        let relu_node = Node::new(OpKind::ReLU, vec![input_node.id], input_node.shape.clone());
        self.add_node(relu_node)
    }

    /// Marks `node` as an output node. Graphs must have at least one output mode, and may have
    /// multiple.
    ///
    /// It does not create a node or execute anything; it simply marks
    /// an existing node as the graph output.
    pub fn set_output_node(&mut self, node: NodeId) -> Result<(), GraphError> {
        let node = self.node(node)?;
        self.outputs.push(node.id);
        Ok(())
    }

    pub fn node(&self, id: NodeId) -> Result<&Node, GraphError> {
        match self.nodes.get(&id) {
            Some(node) => Ok(node),
            None => Err(GraphError::InvalidNodeId),
        }
    }

    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.nodes.values().len()
    }

    #[must_use]
    pub fn inputs(&self) -> &[NodeId] {
        &self.inputs
    }

    #[must_use]
    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
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
        }
    }
}
