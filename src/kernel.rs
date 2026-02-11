//! Defines runtime-executable compute kernels.

use crate::tensor::Tensor;
use std::iter::zip;

pub trait Kernel {
    fn compute(&self, inputs: &[&Tensor], output: &mut Tensor) -> Result<(), KernelError>;
}

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

#[derive(Clone, Debug)]
pub enum KernelError {
    InvalidArguments,
    ShapeMismatch,
    InvalidRank,
}
