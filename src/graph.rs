//! Structure for representing ML runtimes via Node and Op intermediate representation.

use crate::node::{Node, NodeId};
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
    /// let graph = Graph::new();
    /// let fake_node = NodeId { 10 };
    /// let result = graph.relu(fake_node);
    /// assert!(matches!(result.unwrap_err(), GraphError::MissingNode));
    /// ```
    MissingNode,
}

/// Graph stores the nodes for the ML runtime.
///
/// # Examples
/// #todo
pub struct Graph {
    nodes: Vec<Node>,
    inputs: Vec<NodeId>,
    outputs: Vec<NodeId>,
}

impl Graph {
    pub fn new() -> Graph {
        unimplemented!()
    }

    pub fn input_node(&mut self, shape: Vec<usize>) -> NodeId {
        unimplemented!()
    }

    pub fn matmul(&mut self, left: NodeId, right: NodeId) -> Result<NodeId, GraphError> {
        unimplemented!()
    }

    pub fn add(&mut self, left: NodeId, right: NodeId) -> Result<NodeId, GraphError> {
        unimplemented!()
    }

    pub fn relu(&mut self, input: NodeId) -> Result<NodeId, GraphError> {
        unimplemented!()
    }

    /// Marks `node` as an output node. Graphs must have at least one output mode, and may have
    /// multiple.
    ///
    /// It does not create a node or execute anything; it simply marks
    /// an existing node as the graph output.
    pub fn set_output_node(&mut self, node: NodeId) -> () {
        unimplemented!()
    }

    pub fn node(&self, id: NodeId) -> &Node {
        unimplemented!()
    }

    pub fn nodes(&self) -> &[Node] {
        unimplemented!()
    }

    /// Validates a graph before execution.
    ///
    /// Graphs must have at least one input node and one node
    /// marked as output.
    fn validate(&self) -> bool {
        unimplemented!()
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
            GraphError::MissingNode => {
                write!(
                    f,
                    "Attempted to operate on a Node that does not exist in the graph. Ensrue you are only interacting with nodes via Graph::input_node()."
                )
            }
        }
    }
}
