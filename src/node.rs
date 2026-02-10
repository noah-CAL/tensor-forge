//! Representations one operation instance in an ML graph.

use crate::op::OpKind;

/// Represents the `NodeId` for a specific node in the graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

/// Struct for a graph node in the ML operations sequence.
///
/// They are constructed
/// automatically by interacting with operations in a [`Graph`] struct.
///
/// Each node represents a particular action as determined by the
/// [`Node.op`] field (See [`OpKind`]).
///
/// # Examples
/// #TODO
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Node {
    /// ID of the current node.
    ///
    /// This is automatically generated on Node creation.
    pub id: NodeId,
    /// Operation of the current node in the ML pipeline. See [`OpKind`].
    pub op: OpKind,
    /// Node IDs of the inputs to this operation.
    pub inputs: Vec<NodeId>,
    /// Tensor dimensions (shape) of the output tensor produced by this node.
    pub shape: Vec<usize>,
}

use std::sync::atomic::{AtomicU32, Ordering};

static ID_COUNTER: AtomicU32 = AtomicU32::new(0);

impl Node {
    /// Rather than use reference-counted pointers, `graph` contains
    /// a list of all the valid nodes. Input-output pairs are generated
    /// by examining the indices and forming a DAG. See [`graph.rs`] for more information.
    pub(crate) fn new(op: OpKind, inputs: Vec<NodeId>, shape: Vec<usize>) -> Self {
        let node_id = NodeId(ID_COUNTER.fetch_add(1, Ordering::SeqCst) as usize);
        Self {
            id: node_id,
            op,
            inputs,
            shape,
        }
    }
}
