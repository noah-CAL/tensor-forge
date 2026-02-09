//! Defines support ML operations.

/// This enum represents graph-level operation identity.
///
/// # Examples
/// # TODO add examples
pub enum OpKind {
    /// Defines node as an input.  
    Input,
    /// Matrix Multiplication operation.  
    MatMul,
    /// Matrix Addition operation.  
    Add,
    /// Matrix `ReLU` operation.  
    ReLU
}
