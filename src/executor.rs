//! Execution engine for evaluating compute graphs against a [`KernelRegistry`].
//!
//! An [`Executor`] is responsible for:
//! - validating runtime input bindings,
//! - traversing a [`Graph`] in topological order,
//! - dispatching each non-input node to its registered kernel, and
//! - returning the tensors for all nodes marked as graph outputs.
//!
//! Execution is deterministic with respect to graph topology because it relies on
//! [`Graph::topo_sort`], which returns a stable topological order.
//!
//! # Input Binding Model
//!
//! Execution requires one tensor binding for every [`OpKind::Input`] node in the graph.
//! Bindings are passed as a `Vec<(NodeId, Tensor)>`, where:
//! - the [`NodeId`] identifies an input node in the graph, and
//! - the [`Tensor`] is the runtime value to supply for that input.
//!
//! Bindings are validated before execution begins:
//! - each bound node must exist in the graph,
//! - each bound node must be an input node,
//! - each input node must be bound exactly once, and
//! - each bound tensor must match the input node’s declared shape.
//!
//! # Output Model
//!
//! The executor returns a `HashMap<NodeId, Tensor>` containing the computed tensors
//! for all nodes designated as outputs via [`Graph::set_output_node`].
//!
//! Output values are keyed by the output node’s [`NodeId`]. Output ordering is not
//! part of the API contract.
//!
//! # Errors
//!
//! Execution may fail for several classes of reasons:
//! - graph-level failures (for example, invalid topology),
//! - invalid runtime bindings,
//! - missing kernel implementations, or
//! - kernel execution failures at a specific node.
//!
//! These are reported via [`ExecutionError`].
//!
//! # Examples
//! ```
//! # use tensor_forge::executor::Executor;
//! # use tensor_forge::graph::Graph;
//! # use tensor_forge::registry::KernelRegistry;
//! # use tensor_forge::tensor::Tensor;
//! let mut g = Graph::new();
//! let a = g.input_node(vec![2, 2]);
//! let b = g.input_node(vec![2, 2]);
//! let c = g.add(a, b).expect("Valid add operation should succeed");
//! g.set_output_node(c).expect("Valid output node should succeed");
//!
//! let a_tensor = Tensor::zeros(vec![2, 2]).expect("Tensor allocation should succeed");
//! let b_tensor = Tensor::zeros(vec![2, 2]).expect("Tensor allocation should succeed");
//!
//! let exec = Executor::new(KernelRegistry::default());
//! let outputs = exec
//!     .execute(&g, vec![(a, a_tensor), (b, b_tensor)])
//!     .expect("Execution should succeed");
//!
//! assert!(outputs.contains_key(&c));
//! ```
use crate::graph::{Graph, GraphError};
use crate::kernel::KernelError;
use crate::node::NodeId;
use crate::op::OpKind;
use crate::registry::KernelRegistry;
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

/// Executes graphs using kernels registered in a [`KernelRegistry`].
///
/// An [`Executor`] owns a kernel registry and uses it to evaluate graph nodes
/// according to each node’s [`OpKind`]. Non-input nodes are executed in
/// deterministic topological order, and intermediate tensors are stored internally
/// until all requested graph outputs have been produced.
///
/// # Examples
/// ```
/// # use tensor_forge::executor::Executor;
/// # use tensor_forge::registry::KernelRegistry;
/// let exec = Executor::new(KernelRegistry::default());
/// ```
pub struct Executor {
    registry: KernelRegistry,
}

impl Executor {
    /// Creates a new executor backed by the provided kernel `registry`.
    ///
    /// The registry determines which kernel implementation will be used for each
    /// operation kind encountered during execution.
    ///
    /// # Examples
    /// ```
    /// # use tensor_forge::executor::Executor;
    /// # use tensor_forge::registry::KernelRegistry;
    /// let registry = KernelRegistry::default();
    /// let exec = Executor::new(registry);
    /// ```
    #[must_use]
    pub fn new(registry: KernelRegistry) -> Self {
        Self { registry }
    }

    /// Executes `graph` using the provided input `bindings`.
    ///
    /// Each binding is a `(NodeId, Tensor)` pair supplying the runtime value for a
    /// graph input node. Execution proceeds in deterministic topological order:
    ///
    /// 1. Validate the graph topology.
    /// 2. Validate input bindings.
    /// 3. Execute every non-input node using the corresponding registered kernel.
    /// 4. Return a map containing the tensors for all graph output nodes.
    ///
    /// The returned map is keyed by output [`NodeId`]. Output order is not part of
    /// the contract.
    ///
    /// # Binding Rules
    ///
    /// The `inputs` vector must satisfy all of the following:
    /// - every bound node must exist in `graph`,
    /// - every bound node must be an [`OpKind::Input`] node,
    /// - every graph input node must appear exactly once, and
    /// - every bound tensor shape must match the input node’s declared shape.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`ExecutionError::GraphError`] if topological traversal or graph lookup fails,
    /// - [`ExecutionError::DuplicateBinding`] if the same input node is bound more than once,
    /// - [`ExecutionError::InvalidBindingNode`] if a binding references a node not in the graph,
    /// - [`ExecutionError::BindingToNonInputNode`] if a binding targets a non-input node,
    /// - [`ExecutionError::InputShapeMismatch`] if a bound tensor has the wrong shape,
    /// - [`ExecutionError::MissingInput`] if any graph input node is not bound,
    /// - [`ExecutionError::KernelNotFound`] if no kernel is registered for an op,
    /// - [`ExecutionError::KernelExecutionFailed`] if a kernel returns an error during execution,
    /// - [`ExecutionError::InternalError`] if an internal invariant is violated.
    ///
    /// # Examples
    /// ```
    /// # use tensor_forge::executor::Executor;
    /// # use tensor_forge::graph::Graph;
    /// # use tensor_forge::registry::KernelRegistry;
    /// # use tensor_forge::tensor::Tensor;
    /// let mut g = Graph::new();
    /// let x = g.input_node(vec![2, 2]);
    /// let y = g.relu(x).expect("Valid ReLU operation should succeed");
    /// g.set_output_node(y).expect("Valid output node should succeed");
    ///
    /// let x_tensor = Tensor::zeros(vec![2, 2]).expect("Tensor allocation should succeed");
    ///
    /// let exec = Executor::new(KernelRegistry::default());
    /// let outputs = exec
    ///     .execute(&g, vec![(x, x_tensor)])
    ///     .expect("Execution should succeed");
    ///
    /// assert!(outputs.contains_key(&y));
    /// ```
    pub fn execute(
        &self,
        graph: &Graph,
        inputs: Vec<(NodeId, Tensor)>,
    ) -> Result<HashMap<NodeId, Tensor>, ExecutionError> {
        // 0) Discover input nodes (OpKind::Input) and validate bindings.
        //
        // NOTE: This uses topo_sort() to traverse all nodes deterministically.
        // If the graph is cyclic/malformed, this returns a GraphError.
        let topo = graph.topo_sort().map_err(ExecutionError::GraphError)?;

        let mut input_nodes: Vec<NodeId> = Vec::new();
        for &id in &topo {
            let node = graph.node(id).map_err(ExecutionError::GraphError)?;
            if node.op == OpKind::Input {
                input_nodes.push(id);
            }
        }

        // 1) Validate: duplicate bindings
        let mut seen: HashSet<NodeId> = HashSet::with_capacity(inputs.len());
        for (id, _) in &inputs {
            if !seen.insert(*id) {
                return Err(ExecutionError::DuplicateBinding { node: *id });
            }
        }

        // 2) Validate: binding node exists, is input node, and shape matches
        //
        // Also build the runtime value table with the provided inputs.
        let mut values: HashMap<NodeId, Tensor> =
            HashMap::with_capacity(graph.num_nodes().max(inputs.len()));

        for (id, t) in inputs {
            let node = graph
                .node(id)
                .map_err(|_| ExecutionError::InvalidBindingNode { node: id })?;

            if node.op != OpKind::Input {
                return Err(ExecutionError::BindingToNonInputNode {
                    node: id,
                    op: node.op.clone(),
                });
            }

            let expected = node.shape.clone();
            let actual = t.shape().to_vec();
            if actual != expected {
                return Err(ExecutionError::InputShapeMismatch {
                    node: id,
                    expected,
                    actual,
                });
            }

            // Move the owned tensor into the value table.
            values.insert(id, t);
        }

        // 3) Validate: all graph inputs are present in bindings
        for &input_id in &input_nodes {
            if !values.contains_key(&input_id) {
                return Err(ExecutionError::MissingInput { node: input_id });
            }
        }

        // 4) Execute in topological order.
        for &node_id in &topo {
            let node = graph.node(node_id).map_err(ExecutionError::GraphError)?;

            if node.op == OpKind::Input {
                // Inputs were already populated by bindings validation.
                continue;
            }

            // 4a) Fetch kernel
            let kernel =
                self.registry
                    .get(&node.op)
                    .ok_or_else(|| ExecutionError::KernelNotFound {
                        op: node.op.clone(),
                    })?;

            // 4b) Fetch input tensors (by NodeId)
            let mut input_tensors: Vec<&Tensor> = Vec::with_capacity(node.inputs.len());
            for &dep in &node.inputs {
                let t = values.get(&dep).ok_or({
                    // This indicates a bug in topo_sort/executor bookkeeping, or a malformed graph.
                    ExecutionError::InternalError("missing dependency tensor during execution")
                })?;
                input_tensors.push(t);
            }

            // 4c) Allocate output tensor
            let mut out = Tensor::zeros(node.shape.clone())
                .map_err(|_| ExecutionError::InternalError("failed to allocate output tensor"))?;

            // 4d) Execute kernel
            kernel
                .compute(&input_tensors, &mut out)
                .map_err(|e: KernelError| ExecutionError::KernelExecutionFailed {
                    node: node_id,
                    op: node.op.clone(),
                    source: e,
                })?;

            // Store produced tensor for downstream consumers.
            values.insert(node_id, out);
        }

        // 5) Collect outputs by output NodeId.
        let mut outputs: HashMap<NodeId, Tensor> = HashMap::with_capacity(graph.outputs().len());
        for &out_id in graph.outputs() {
            let t = values.remove(&out_id).ok_or({
                ExecutionError::InternalError("output tensor missing after execution")
            })?;
            outputs.insert(out_id, t);
        }

        Ok(outputs)
    }
}

impl Default for Executor {
    /// Creates an executor using [`KernelRegistry::default`].
    ///
    /// This is a convenience constructor for the common case where the default
    /// kernel set is sufficient.
    ///
    /// # Examples
    /// ```
    /// # use tensor_forge::executor::Executor;
    /// let exec = Executor::default();
    /// ```
    fn default() -> Self {
        Self {
            registry: KernelRegistry::default(),
        }
    }
}

/// Errors that may occur while validating bindings or executing a graph.
///
/// These errors distinguish between:
/// - invalid caller-supplied bindings,
/// - graph-level failures,
/// - missing kernel implementations, and
/// - runtime kernel failures.
///
/// # Examples
/// ```
/// # use tensor_forge::executor::{ExecutionError, Executor};
/// # use tensor_forge::graph::Graph;
/// # use tensor_forge::registry::KernelRegistry;
/// # use tensor_forge::tensor::Tensor;
/// let mut g = Graph::new();
/// let x = g.input_node(vec![2, 2]);
/// g.set_output_node(x).expect("Valid output node should succeed");
///
/// let wrong = Tensor::zeros(vec![3, 3]).expect("Tensor allocation should succeed");
/// let exec = Executor::new(KernelRegistry::default());
///
/// let err = exec.execute(&g, vec![(x, wrong)]).unwrap_err();
/// assert!(matches!(err, ExecutionError::InputShapeMismatch { .. }));
/// ```
#[derive(Debug, Clone)]
pub enum ExecutionError {
    /// A required input node was not provided in the bindings.
    ///
    /// Every [`OpKind::Input`] node in the graph must have exactly one runtime binding.
    MissingInput {
        /// The input node that was not bound at execution time.
        node: NodeId,
    },

    /// A binding was provided for a node that exists in the graph but is not an input node.
    ///
    /// Only input nodes may be bound directly by the caller.
    BindingToNonInputNode {
        /// The node that was incorrectly bound.
        node: NodeId,
        /// The operation kind of `node`.
        op: OpKind,
    },

    /// A binding was provided for a node that does not exist in the graph.
    ///
    /// This typically indicates that:
    /// - a stale [`NodeId`] was reused, or
    /// - a [`NodeId`] from another graph was supplied.
    InvalidBindingNode {
        /// The invalid node identifier supplied in the bindings.
        node: NodeId,
    },

    /// Multiple bindings were provided for the same input node.
    ///
    /// Input bindings must be unique by [`NodeId`].
    DuplicateBinding {
        /// The input node that was bound more than once.
        node: NodeId,
    },

    /// A runtime tensor shape did not match the graph input node’s declared shape.
    ///
    /// The `expected` shape is taken from the graph node, while `actual` is the
    /// shape of the caller-provided tensor.
    InputShapeMismatch {
        /// The input node whose binding had the wrong shape.
        node: NodeId,
        /// The shape declared by the graph for `node`.
        expected: Vec<usize>,
        /// The shape of the tensor supplied by the caller.
        actual: Vec<usize>,
    },

    /// No kernel implementation was registered for the requested operation.
    ///
    /// This prevents execution of any node with the given [`OpKind`].
    KernelNotFound {
        /// The operation kind for which no kernel was registered.
        op: OpKind,
    },

    /// A kernel returned an error while executing a specific graph node.
    ///
    /// This variant preserves:
    /// - the failing node ID,
    /// - the operation kind being executed, and
    /// - the underlying [`KernelError`].
    KernelExecutionFailed {
        /// The graph node whose kernel execution failed.
        node: NodeId,
        /// The operation kind being executed at `node`.
        op: OpKind,
        /// The underlying kernel error returned by the registered kernel.
        source: KernelError,
    },

    /// A graph-level failure occurred while preparing execution.
    ///
    /// This wraps errors originating from graph traversal or graph validation,
    /// such as cycle detection or invalid node references.
    GraphError(GraphError),

    /// An internal executor invariant was violated.
    ///
    /// This variant indicates a bug or malformed internal state rather than a
    /// user-facing validation issue. Under normal operation, callers should not
    /// be able to trigger this error through the public API alone.
    InternalError(&'static str),
}

impl std::fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            ExecutionError::MissingInput { node } => {
                write!(f, "Missing input binding for node {node:?}")
            }

            ExecutionError::BindingToNonInputNode { node, op } => {
                write!(f, "Cannot bind tensor to non-input node {node:?} ({op:?})")
            }

            ExecutionError::InvalidBindingNode { node } => {
                write!(f, "Binding provided for invalid node {node:?}")
            }

            ExecutionError::DuplicateBinding { node } => {
                write!(f, "Duplicate binding for node {node:?}")
            }

            ExecutionError::InputShapeMismatch {
                node,
                expected,
                actual,
            } => write!(
                f,
                "Shape mismatch for node {node:?}: expected {expected:?}, got {actual:?}",
            ),

            ExecutionError::KernelNotFound { op } => {
                write!(f, "No kernel registered for operation {op:?}")
            }

            ExecutionError::KernelExecutionFailed { node, op, source } => write!(
                f,
                "Kernel execution failed at node {node:?} ({op:?}): {source}",
            ),

            ExecutionError::GraphError(e) => write!(f, "Graph error during execution: {e}"),

            ExecutionError::InternalError(msg) => write!(f, "Executor internal error: {msg}"),
        }
    }
}
