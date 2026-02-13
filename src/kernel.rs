//! Defines runtime-executable compute kernels.
//!
//! A *kernel* is a pure compute primitive that reads one or more input [`Tensor`]s and writes
//! results into a pre-allocated output [`Tensor`]. Kernels do not allocate output storage and
//! are expected to validate basic argument invariants (arity, shape, rank) before any compute.
//!
//! To allocate [`Tensor`]s, see the [`crate::graph::Graph`] API.

use crate::tensor::Tensor;
use std::fmt;
use std::iter::zip;

/// A runtime-executable compute primitive.
///
/// A kernel computes `output = f(inputs...)` for a particular operation. The caller is
/// responsible for allocating `output` with the correct shape.
///
/// # Errors
///
/// Implementations return [`KernelError`] if:
/// - The number of `inputs` does not match the kernel contract,
/// - Shapes are incompatible for the operation, or
/// - The operation requires a specific rank (e.g., 2-D matrices for matmul) and the input rank
///   is unsupported.
///
/// # Examples
/// ```
/// # use tensor_forge::tensor::Tensor;
/// # use tensor_forge::kernel::{Kernel, AddKernel};
/// let shape = vec![2, 2];
/// let a = Tensor::from_vec(shape.clone(), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let b = Tensor::from_vec(shape.clone(), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
///
/// let mut out = Tensor::zeros(shape).unwrap();
/// AddKernel.compute(&[&a, &b], &mut out).unwrap();
/// assert_eq!(out.data(), &[11.0, 22.0, 33.0, 44.0]);
/// ```
pub trait Kernel {
    /// Computes the kernel output in-place.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError`] on invalid input arity, shape incompatibility, or unsupported rank.
    fn compute(&self, inputs: &[&Tensor], output: &mut Tensor) -> Result<(), KernelError>;
}

/// Validates argument invariants for 2-D matrix multiplication.
///
/// Shape rule (rank-2 only):
/// - `left.shape = [m, n]`
/// - `right.shape = [n, d]`
/// - `output.shape` must be `[m, d]`
///
/// # Errors
///
/// Returns [`KernelError::InvalidRank`] if either input is not rank-2.
///
/// Returns [`KernelError::ShapeMismatch`] if inner dimensions do not match or `output` has the
/// wrong shape.
fn verify_matmul_arguments(
    left: &Tensor,
    right: &Tensor,
    output: &Tensor,
) -> Result<(), KernelError> {
    // N-dimensional matrix multiplications are not supported right now.
    if left.shape().len() != 2 || right.shape().len() != 2 {
        return Err(KernelError::InvalidRank);
    }

    // Input/Output connections should already be verified in Graph construction, but good
    // to sanity check here.
    let exp_output_shape = vec![left.shape()[0], right.shape()[1]];
    if left.shape()[1] != right.shape()[0] || output.shape() != exp_output_shape {
        return Err(KernelError::ShapeMismatch);
    }
    Ok(())
}

/// Naive 2-D matrix multiplication kernel.
///
/// Computes `output = left × right` for rank-2 matrices.
///
/// - `left` shape: `[m, n]`
/// - `right` shape: `[n, d]`
/// - `output` shape: `[m, d]`
///
/// Data is interpreted as row-major contiguous storage.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `inputs.len() != 2`.
///
/// Returns [`KernelError::InvalidRank`] if either input is not rank-2.
///
/// Returns [`KernelError::ShapeMismatch`] if inner dimensions do not match or `output` has the
/// wrong shape.
///
/// # Examples
/// ```
/// # use tensor_forge::tensor::Tensor;
/// # use tensor_forge::kernel::{Kernel, MatMulKernel};
/// let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0,
///                                   4.0, 5.0, 6.0]).unwrap();
/// let b = Tensor::from_vec(vec![3, 2], vec![7.0,  8.0,
///                                   9.0, 10.0,
///                                  11.0, 12.0]).unwrap();
///
/// let mut out = Tensor::zeros(vec![2, 2]).unwrap();
/// MatMulKernel.compute(&[&a, &b], &mut out).unwrap();
/// assert_eq!(out.data(), &[58.0, 64.0, 139.0, 154.0]);
/// ```
pub struct MatMulKernel;

impl Kernel for MatMulKernel {
    fn compute(&self, inputs: &[&Tensor], output: &mut Tensor) -> Result<(), KernelError> {
        if inputs.len() != 2 {
            return Err(KernelError::InvalidArguments);
        }
        let (left, right) = (inputs[0], inputs[1]);
        verify_matmul_arguments(left, right, output)?;
        // Perform a naive matrix multiplication for alpha testing purposes.
        //
        // For an (MxN) x (NxD) Matmul: we compute successive dot products with each row and
        // column.
        let n = left.shape()[1];
        let d = right.shape()[1];
        // Tensor data is stored in row-major order. Therefore, we can access a full column with an iterator stride
        let l_data = left.data();
        let r_data = right.data();
        for (i, element) in output.data_mut().iter_mut().enumerate() {
            let row_offset = (i / d) * n;
            let col_offset = i % d;
            let row = l_data.iter().skip(row_offset).take(n);
            let col = r_data.iter().skip(col_offset).step_by(d).take(n);
            *element = zip(row, col).fold(0.0, |acc, (r, c)| acc + r * c);
        }
        Ok(())
    }
}

/// Elementwise addition kernel.
///
/// Computes `output = left + right` (binary add).
///
/// All tensors must have identical shapes. Addition is performed elementwise in row-major order.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `inputs.len() != 2`.
///
/// Returns [`KernelError::ShapeMismatch`] if `left`, `right`, and `output` do not all share the
/// same shape.
///
/// # Examples
/// ```
/// # use tensor_forge::tensor::Tensor;
/// # use tensor_forge::kernel::{Kernel, AddKernel};
/// let shape = vec![1, 4];
/// let a = Tensor::from_vec(shape.clone(), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let b = Tensor::from_vec(shape.clone(), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
///
/// let mut out = Tensor::zeros(shape).unwrap();
/// AddKernel.compute(&[&a, &b], &mut out).unwrap();
/// assert_eq!(out.data(), &[11.0, 22.0, 33.0, 44.0]);
/// ```
pub struct AddKernel;

impl Kernel for AddKernel {
    fn compute(&self, inputs: &[&Tensor], output: &mut Tensor) -> Result<(), KernelError> {
        if inputs.len() != 2 {
            return Err(KernelError::InvalidArguments);
        }
        // Input connections should already be verified in Graph construction, but good
        // to sanity check here.
        let (left, right) = (inputs[0], inputs[1]);
        if left.shape() != right.shape() || output.shape() != right.shape() {
            return Err(KernelError::ShapeMismatch);
        }
        zip(output.data_mut().iter_mut(), zip(left.data(), right.data()))
            .for_each(|(out, (l, r))| *out = l + r);
        Ok(())
    }
}

/// Rectified Linear Unit (`ReLU`) activation kernel.
///
/// Computes `output[i] = max(0, input[i])` elementwise.
///
/// NaN handling: if an input element is NaN, the corresponding output element is set to NaN.
///
/// # Errors
///
/// Returns [`KernelError::InvalidArguments`] if `inputs.len() != 1`.
///
/// Returns [`KernelError::ShapeMismatch`] if `input.shape() != output.shape()`.
///
/// # Examples
/// ```
/// # use tensor_forge::tensor::Tensor;
/// # use tensor_forge::kernel::{Kernel, ReluKernel};
/// let shape = vec![1, 5];
/// let x = Tensor::from_vec(shape.clone(), vec![-2.0, -0.0, 0.0, 1.5, 3.0]).unwrap();
///
/// let mut out = Tensor::zeros(shape).unwrap();
/// ReluKernel.compute(&[&x], &mut out).unwrap();
/// assert_eq!(out.data(), &[0.0, 0.0, 0.0, 1.5, 3.0]);
/// ```
pub struct ReluKernel;

impl Kernel for ReluKernel {
    fn compute(&self, inputs: &[&Tensor], output: &mut Tensor) -> Result<(), KernelError> {
        if inputs.len() != 1 {
            return Err(KernelError::InvalidArguments);
        }
        let input = inputs[0];
        // Input connections should already be verified in Graph construction, but good
        // to sanity check here.
        if input.shape() != output.shape() {
            return Err(KernelError::ShapeMismatch);
        }
        for (output, &input) in zip(output.data_mut().iter_mut(), input.data().iter()) {
            if input.is_nan() {
                *output = input;
            } else {
                *output = input.max(0_f64);
            }
        }
        Ok(())
    }
}

/// Kernel-level errors raised during argument validation or execution.
#[derive(Clone, Debug)]
pub enum KernelError {
    /// Wrong number of inputs were provided for the kernel.
    InvalidArguments,
    /// Input/output shapes are incompatible for the requested operation.
    ShapeMismatch,
    /// Input rank is unsupported for the requested operation (e.g., non-2-D matmul).
    InvalidRank,
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            KernelError::InvalidArguments => write!(
                f,
                "Invalid tensor dimensions. Tensor shape must not contain a zero."
            ),
            KernelError::ShapeMismatch => {
                write!(
                    f,
                    "Input/Output tensors have mismatched data size for the selected operation."
                )
            }
            KernelError::InvalidRank => {
                write!(
                    f,
                    "Invalid/Unsupported matrix rank for supported operation."
                )
            }
        }
    }
}
