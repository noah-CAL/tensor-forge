use tensor_forge::kernel::{AddKernel, Kernel, KernelError, MatMulKernel, ReluKernel};
use tensor_forge::tensor::Tensor;

use std::iter::zip;

mod common;

/////////////
// Helpers //
/////////////

fn assert_tensor_eq(actual: &Tensor, expected: &[f64]) {
    assert_eq!(
        expected,
        actual.data(),
        "tensor data mismatch (len expected={}, actual={})",
        expected.len(),
        actual.data().len(),
    );
}

fn run_kernel<T>(
    kernel: T,
    inputs: &[&Tensor],
    output_shape: &[usize],
) -> Result<Tensor, KernelError>
where
    T: Kernel,
{
    let mut output = Tensor::zeros(output_shape).expect("Valid tensor shape");
    kernel.compute(inputs, &mut output)?;
    Ok(output)
}

/////////////////////////////
// Golden Model References //
/////////////////////////////
fn relu_ref(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| if v < 0.0 { 0.0 } else { v }).collect()
}

fn add_ref(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

fn matmul_ref(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    // Naive matmul approach
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

///////////////////////
//     AddKernel     //
///////////////////////
#[test]
fn add_kernel_basic() {
    let shape = vec![4, 4, 8];
    let numel: usize = shape.iter().product();

    let a_data: Vec<f64> = (0..numel).map(|x| x as f64).collect();
    let b_data: Vec<f64> = (0..numel).map(|x| (x as f64) * 4.35).collect();

    let expected: Vec<f64> = zip(a_data.iter().copied(), b_data.iter().copied())
        .map(|(x, y)| x + y)
        .collect();

    let a = Tensor::from_vec(shape.clone(), a_data).unwrap();
    let b = Tensor::from_vec(shape.clone(), b_data).unwrap();

    let inputs = vec![&a, &b];
    let output = run_kernel(AddKernel, &inputs, &shape).expect("Valid addition should occur");
    assert_tensor_eq(&output, &expected);
}

#[test]
fn add_kernel_complex() {
    let shape = vec![2, 3];
    let numel: usize = shape.iter().product();

    let a_data: Vec<f64> = vec![-1.0, 2.5, 3.25, 0.0, -7.75, 9.0];
    let b_data: Vec<f64> = vec![4.0, -2.5, 0.75, -3.0, 7.25, -10.5];

    assert_eq!(a_data.len(), numel);
    assert_eq!(b_data.len(), numel);

    let expected: Vec<f64> = zip(a_data.iter().copied(), b_data.iter().copied())
        .map(|(x, y)| x + y)
        .collect();

    let a = Tensor::from_vec(shape.clone(), a_data).unwrap();
    let b = Tensor::from_vec(shape.clone(), b_data).unwrap();

    let inputs = vec![&a, &b];
    let output = run_kernel(AddKernel, &inputs, &shape).unwrap();
    assert_tensor_eq(&output, &expected);
}

#[test]
fn add_kernel_invalid_arity_zero_inputs() {
    let shape = vec![2, 2];
    let inputs: Vec<&Tensor> = Vec::new();

    let err = run_kernel(AddKernel, &inputs, &shape).unwrap_err();
    assert!(matches!(err, KernelError::InvalidArguments));
}

#[test]
fn add_kernel_invalid_arity_one_input() {
    let shape = vec![2, 2];
    let numel: usize = shape.iter().product();

    let a = Tensor::from_vec(shape.clone(), vec![1.0; numel]).unwrap();
    let inputs = vec![&a];

    let err = run_kernel(AddKernel, &inputs, &shape).unwrap_err();
    assert!(matches!(err, KernelError::InvalidArguments));
}

#[test]
fn add_kernel_invalid_arity_three_inputs() {
    let shape = vec![2, 2];
    let numel: usize = shape.iter().product();

    let a = Tensor::from_vec(shape.clone(), vec![1.0; numel]).unwrap();
    let b = Tensor::from_vec(shape.clone(), vec![2.0; numel]).unwrap();
    let c = Tensor::from_vec(shape.clone(), vec![3.0; numel]).unwrap();
    let inputs = vec![&a, &b, &c];

    let err = run_kernel(AddKernel, &inputs, &shape).unwrap_err();
    assert!(matches!(err, KernelError::InvalidArguments));
}

#[test]
fn add_kernel_input_shape_mismatch() {
    let shape_a = vec![2, 3];
    let shape_b = vec![3, 2];
    let numel_a: usize = shape_a.iter().product();
    let numel_b: usize = shape_b.iter().product();

    let a = Tensor::from_vec(shape_a.clone(), vec![1.0; numel_a]).unwrap();
    let b = Tensor::from_vec(shape_b.clone(), vec![2.0; numel_b]).unwrap();
    let inputs = vec![&a, &b];

    let err = run_kernel(AddKernel, &inputs, &shape_a).unwrap_err();
    assert!(matches!(err, KernelError::ShapeMismatch));
}

#[test]
fn add_kernel_output_shape_mismatch() {
    // Inputs are [2,3], but output shape is [3,2] (same numel, different shape).
    let in_shape = vec![2, 3];
    let out_shape = vec![3, 2];
    let numel: usize = in_shape.iter().product();

    let a = Tensor::from_vec(in_shape.clone(), vec![1.0; numel]).unwrap();
    let b = Tensor::from_vec(in_shape.clone(), vec![2.0; numel]).unwrap();
    let inputs = vec![&a, &b];

    let err = run_kernel(AddKernel, &inputs, &out_shape).unwrap_err();
    assert!(matches!(err, KernelError::ShapeMismatch));
}

#[test]
fn add_kernel_overflow_f64_to_infinity() {
    let shape = vec![1, 8];
    let numel: usize = shape.iter().product();

    // For f64, overflow does not wrap; it becomes +infinity.
    let a = Tensor::from_vec(shape.clone(), vec![f64::MAX; numel]).unwrap();
    let b = Tensor::from_vec(shape.clone(), vec![f64::MAX; numel]).unwrap();
    let inputs = vec![&a, &b];

    let out = run_kernel(AddKernel, &inputs, &shape).unwrap();

    assert!(
        out.data()
            .iter()
            .all(|x| x.is_infinite() && x.is_sign_positive())
    );
}

#[test]
fn add_kernel_infinity_and_nan_semantics() {
    let shape = vec![1, 6];

    let a_data = vec![
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        1.0,
        f64::NAN,
        0.0,
    ];
    let b_data = vec![
        1.0,
        1.0,
        f64::NEG_INFINITY, // inf + -inf => NaN
        f64::INFINITY,     // 1 + inf => inf
        2.0,               // NaN + 2 => NaN
        f64::NAN,          // 0 + NaN => NaN
    ];

    let a = Tensor::from_vec(shape.clone(), a_data).unwrap();
    let b = Tensor::from_vec(shape.clone(), b_data).unwrap();
    let inputs = vec![&a, &b];

    let out = run_kernel(AddKernel, &inputs, &shape).unwrap();
    let got = out.data();

    assert!(got[0].is_infinite() && got[0].is_sign_positive()); // inf + 1 => inf
    assert!(got[1].is_infinite() && got[1].is_sign_negative()); // -inf + 1 => -inf
    assert!(got[2].is_nan()); // inf + -inf => NaN
    assert!(got[3].is_infinite() && got[3].is_sign_positive()); // 1 + inf => inf
    assert!(got[4].is_nan()); // NaN + 2 => NaN
    assert!(got[5].is_nan()); // 0 + NaN => NaN
}

#[test]
fn add_kernel_signed_zero_behavior() {
    // Signed zero is a real IEEE-754 thing; addition should follow f64 semantics.
    let shape = vec![1, 2];

    let a_data = vec![0.0, -0.0];
    let b_data = vec![-0.0, 0.0];

    let a = Tensor::from_vec(shape.clone(), a_data).unwrap();
    let b = Tensor::from_vec(shape.clone(), b_data).unwrap();
    let inputs = vec![&a, &b];

    let out = run_kernel(AddKernel, &inputs, &shape).unwrap();
    let got = out.data();

    // Numerically both are 0.0; sign of zero is subtle and not always worth pinning down.
    // We assert value equality and that the results are zero.
    assert_eq!(got[0], 0.0);
    assert_eq!(got[1], 0.0);
}

///////////////////////
//     ReluKernel    //
///////////////////////
#[test]
fn relu_kernel_basic() {
    let shape = vec![2, 3];
    let numel: usize = shape.iter().product();

    let x_data: Vec<f64> = vec![-3.0, -0.0, 0.0, 1.5, 2.0, -7.25];
    assert_eq!(x_data.len(), numel);

    // ReLU: max(0, x). Note: -0.0 becomes +0.0 in IEEE math via max.
    let expected: Vec<f64> = x_data
        .iter()
        .copied()
        .map(|x| if x.is_nan() { x } else { x.max(0.0) })
        .collect();

    let x = Tensor::from_vec(shape.clone(), x_data).unwrap();
    let inputs = vec![&x];

    let out = run_kernel(ReluKernel, &inputs, &shape).expect("Valid ReLU should occur");
    assert_tensor_eq(&out, &expected);
}

#[test]
fn relu_kernel_all_negative() {
    let shape = vec![4, 4];
    let numel: usize = shape.iter().product();

    let x_data: Vec<f64> = (0..numel).map(|i| -(i as f64) - 1.0).collect();
    let expected: Vec<f64> = vec![0.0; numel];

    let x = Tensor::from_vec(shape.clone(), x_data).unwrap();
    let inputs = vec![&x];

    let out = run_kernel(ReluKernel, &inputs, &shape).unwrap();
    assert_tensor_eq(&out, &expected);
}

#[test]
fn relu_kernel_all_positive() {
    let shape = vec![3, 5];
    let numel: usize = shape.iter().product();

    let x_data: Vec<f64> = (0..numel).map(|i| (i as f64) + 0.25).collect();
    let expected = x_data.clone();

    let x = Tensor::from_vec(shape.clone(), x_data).unwrap();
    let inputs = vec![&x];

    let out = run_kernel(ReluKernel, &inputs, &shape).unwrap();
    assert_tensor_eq(&out, &expected);
}

#[test]
fn relu_kernel_preserves_shape() {
    let shape = vec![1, 2, 3, 4];
    let numel: usize = shape.iter().product();

    let x_data: Vec<f64> = (0..numel).map(|i| (i as f64) - 10.0).collect();

    let x = Tensor::from_vec(shape.clone(), x_data).unwrap();
    let inputs = vec![&x];

    let out = run_kernel(ReluKernel, &inputs, &shape).unwrap();
    assert_eq!(out.shape(), shape.as_slice());
}

#[test]
fn relu_kernel_invalid_arity_zero_inputs() {
    let shape = vec![2, 2];
    let inputs: Vec<&Tensor> = vec![];

    let err = run_kernel(ReluKernel, &inputs, &shape).unwrap_err();
    assert!(matches!(err, KernelError::InvalidArguments));
}

#[test]
fn relu_kernel_invalid_arity_two_inputs() {
    let shape = vec![2, 2];
    let numel: usize = shape.iter().product();

    let a = Tensor::from_vec(shape.clone(), vec![1.0; numel]).unwrap();
    let b = Tensor::from_vec(shape.clone(), vec![-1.0; numel]).unwrap();
    let inputs = vec![&a, &b];

    let err = run_kernel(ReluKernel, &inputs, &shape).unwrap_err();
    assert!(matches!(err, KernelError::InvalidArguments));
}

#[test]
fn relu_kernel_invalid_output_shape() {
    let in_shape = vec![2, 3];
    let out_shape = vec![3, 2]; // same numel but different shape => should still fail
    let numel: usize = in_shape.iter().product();

    let x = Tensor::from_vec(in_shape.clone(), vec![1.0; numel]).unwrap();
    let inputs = vec![&x];

    let err = run_kernel(ReluKernel, &inputs, &out_shape).unwrap_err();
    assert!(matches!(err, KernelError::ShapeMismatch));
}

#[test]
fn relu_kernel_nan_propagation() {
    let shape = vec![2, 3];
    let x_data = vec![f64::NAN, -1.0, 0.0, 1.0, f64::NAN, -0.0];

    let expected: Vec<f64> = x_data
        .iter()
        .copied()
        .map(|x| if x.is_nan() { x } else { x.max(0.0) })
        .collect();

    let x = Tensor::from_vec(shape.clone(), x_data).unwrap();
    let inputs = vec![&x];

    let out = run_kernel(ReluKernel, &inputs, &shape).unwrap();

    // Can't use assert_tensor_eq if it does exact equality, because NaN != NaN.
    // So we do a per-element check:
    for (got, exp) in zip(out.data(), expected.iter()) {
        if exp.is_nan() {
            assert!(
                got.is_nan(),
                "Expected output to carry NAN through ReLU. Got {}",
                got
            );
        } else {
            assert_eq!(*got, *exp);
        }
    }
}

#[test]
fn relu_kernel_infinity_handling() {
    let shape = vec![1, 6];
    let x_data = vec![f64::INFINITY, f64::NEG_INFINITY, 1.0, -1.0, 0.0, -0.0];

    let expected = vec![f64::INFINITY, 0.0, 1.0, 0.0, 0.0, 0.0];

    let x = Tensor::from_vec(shape.clone(), x_data).unwrap();
    let inputs = vec![&x];

    let out = run_kernel(ReluKernel, &inputs, &shape).unwrap();

    // Infinity comparisons are well-defined (inf == inf).
    assert_tensor_eq(&out, &expected);
}

///////////////////////
//    MatmulKernel   //
///////////////////////
#[test]
fn matmul_kernel_basic_rectangular() {
    // A: 2x3, B: 3x2 => C: 2x2
    let a_shape = vec![2, 3];
    let b_shape = vec![3, 2];
    let out_shape = vec![2, 2];

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    let expected = matmul_ref(&a_data, &b_data, 2, 3, 2);

    let a = Tensor::from_vec(a_shape, a_data).unwrap();
    let b = Tensor::from_vec(b_shape, b_data).unwrap();
    let inputs = vec![&a, &b];

    let out = run_kernel(MatMulKernel, &inputs, &out_shape).unwrap();
    assert_tensor_eq(&out, &expected);
}

#[test]
fn matmul_kernel_identity_right() {
    // A: 3x3, I: 3x3 => A
    let shape = vec![3, 3];
    let out_shape = vec![3, 3];

    let a_data = vec![2.0, -1.0, 3.0, 0.0, 4.0, 5.0, 7.0, 8.0, 9.0];
    let i_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let a = Tensor::from_vec(shape.clone(), a_data.clone()).unwrap();
    let i = Tensor::from_vec(shape, i_data).unwrap();

    let inputs = vec![&a, &i];
    let out = run_kernel(MatMulKernel, &inputs, &out_shape).unwrap();

    assert_tensor_eq(&out, &a_data);
}

#[test]
fn matmul_kernel_zeros() {
    // A: 2x3, B: 3x4 all zeros => C all zeros
    let a_shape = vec![2, 3];
    let b_shape = vec![3, 4];
    let out_shape = vec![2, 4];

    let a_data = vec![0.0; 2 * 3];
    let b_data = vec![0.0; 3 * 4];
    let expected = vec![0.0; 2 * 4];

    let a = Tensor::from_vec(a_shape, a_data).unwrap();
    let b = Tensor::from_vec(b_shape, b_data).unwrap();
    let inputs = vec![&a, &b];

    let out = run_kernel(MatMulKernel, &inputs, &out_shape).unwrap();
    assert_tensor_eq(&out, &expected);
}

#[test]
fn matmul_kernel_invalid_arity() {
    let a = Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap();
    let b = Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap();

    // 0 inputs
    let err = run_kernel(MatMulKernel, &Vec::<&Tensor>::new(), &[1, 1]).unwrap_err();
    assert!(matches!(err, KernelError::InvalidArguments));

    // 1 input
    let err = run_kernel(MatMulKernel, &[&a], &[1, 1]).unwrap_err();
    assert!(matches!(err, KernelError::InvalidArguments));

    // 3 inputs
    let err = run_kernel(MatMulKernel, &[&a, &b, &a], &[1, 1]).unwrap_err();
    assert!(matches!(err, KernelError::InvalidArguments));
}

#[test]
fn matmul_kernel_shape_mismatch() {
    // A: 2x3, B: 2x4 => inner dims mismatch (3 != 2)
    let a = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).unwrap();
    let b = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).unwrap();

    let err = run_kernel(MatMulKernel, &[&a, &b], &[2, 4]).unwrap_err();
    assert!(matches!(err, KernelError::ShapeMismatch));
}

#[test]
fn matmul_kernel_invalid_rank_one() {
    let shape = vec![4];
    let numel: usize = shape.iter().product();

    let a = Tensor::from_vec(shape.clone(), vec![1.0; numel]).unwrap();
    let b = Tensor::from_vec(shape.clone(), vec![2.0; numel]).unwrap();
    let inputs = vec![&a, &b];

    let err = run_kernel(MatMulKernel, &inputs, &shape).unwrap_err();
    assert!(matches!(err, KernelError::InvalidRank));
}

#[test]
fn matmul_kernel_invalid_rank_three() {
    let shape = vec![4, 4, 4];
    let numel: usize = shape.iter().product();

    let a = Tensor::from_vec(shape.clone(), vec![1.0; numel]).unwrap();
    let b = Tensor::from_vec(shape.clone(), vec![2.0; numel]).unwrap();
    let inputs = vec![&a, &b];

    let err = run_kernel(MatMulKernel, &inputs, &shape).unwrap_err();
    assert!(matches!(err, KernelError::InvalidRank));
}

#[test]
fn matmul_kernel_output_shape_mismatch() {
    // Valid A*B would be 2x2, but we ask for 2x3 output => should error
    let a = Tensor::from_vec(vec![2, 3], vec![1.0; 6]).unwrap();
    let b = Tensor::from_vec(vec![3, 2], vec![1.0; 6]).unwrap();

    let err = run_kernel(MatMulKernel, &[&a, &b], &[2, 3]).unwrap_err();
    assert!(matches!(err, KernelError::ShapeMismatch));
}

#[test]
fn matmul_kernel_chain() {
    // (A * B) * C
    // A: 2x3, B: 3x2 => D: 2x2
    // D: 2x2, C: 2x4 => E: 2x4
    let a_shape = vec![2, 3];
    let b_shape = vec![3, 2];
    let d_shape = vec![2, 2];
    let c_shape = vec![2, 4];
    let e_shape = vec![2, 4];

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let c_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let a = Tensor::from_vec(a_shape, a_data.clone()).unwrap();
    let b = Tensor::from_vec(b_shape, b_data.clone()).unwrap();
    let c = Tensor::from_vec(c_shape, c_data.clone()).unwrap();

    let d_expected = matmul_ref(&a_data, &b_data, 2, 3, 2);
    let d = run_kernel(MatMulKernel, &[&a, &b], &d_shape).unwrap();
    assert_tensor_eq(&d, &d_expected);

    let e_expected = matmul_ref(d.data(), &c_data, 2, 2, 4);
    let e = run_kernel(MatMulKernel, &[&d, &c], &e_shape).unwrap();
    assert_tensor_eq(&e, &e_expected);
}

/////////////////////////////////
//  Kernel Integration Tests   //
/////////////////////////////////
#[test]
fn kernel_integration_relu_matmul_add_graph() {
    // Tests the same graph structure in graph_tests.rs:
    // Input A -- ReLU     Input C
    //                 \          \
    //                  -- Matmul - Add -> Output
    //                 /
    // Input B -- ReLU
    //
    // Layers:
    //    (1)     (2)    (3)      (4)      (5)
    let a_shape = vec![2, 3];
    let b_shape = vec![3, 2];
    let mm_shape = vec![2, 2];
    let c_shape = vec![2, 2];
    let out_shape = vec![2, 2];

    let a_data = vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0];
    let b_data = vec![1.0, -2.0, 3.0, 4.0, -5.0, 6.0];
    let c_data = vec![10.0, 20.0, 30.0, 40.0];

    // Reference expected:
    let a_relu = relu_ref(&a_data);
    let b_relu = relu_ref(&b_data);
    let mm = matmul_ref(&a_relu, &b_relu, 2, 3, 2);
    let expected = add_ref(&mm, &c_data);

    // Run via kernels:
    let a = Tensor::from_vec(a_shape.clone(), a_data).unwrap();
    let b = Tensor::from_vec(b_shape.clone(), b_data).unwrap();
    let c = Tensor::from_vec(c_shape.clone(), c_data).unwrap();

    let a1 = run_kernel(ReluKernel, &[&a], &a_shape).unwrap();
    let b1 = run_kernel(ReluKernel, &[&b], &b_shape).unwrap();

    let mm_out = run_kernel(MatMulKernel, &[&a1, &b1], &mm_shape).unwrap();
    let out = run_kernel(AddKernel, &[&mm_out, &c], &out_shape).unwrap();

    assert_tensor_eq(&out, &expected);
}

#[test]
fn kernel_integration_matmul_add_relu_pipeline() {
    // Tests: relu(matmul(A,B) + D)
    //
    // Input A     Input D
    //         \          \
    //          --- Matmul - Add -> Output
    //         /
    // Input B
    //
    // Layers:
    //   (1)         (2)     (3)
    let a_shape = vec![2, 3];
    let b_shape = vec![3, 2];
    let mm_shape = vec![2, 2];
    let d_shape = vec![2, 2];
    let out_shape = vec![2, 2];

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let d_data = vec![-10.0, 0.0, 10.0, -1.0];

    // Reference expected:
    let mm = matmul_ref(&a_data, &b_data, 2, 3, 2);
    let add = add_ref(&mm, &d_data);
    let expected = relu_ref(&add);

    let a = Tensor::from_vec(a_shape, a_data).unwrap();
    let b = Tensor::from_vec(b_shape, b_data).unwrap();
    let d = Tensor::from_vec(d_shape, d_data).unwrap();

    let mm_out = run_kernel(MatMulKernel, &[&a, &b], &mm_shape).unwrap();
    let add_out = run_kernel(AddKernel, &[&mm_out, &d], &out_shape).unwrap();
    let out = run_kernel(ReluKernel, &[&add_out], &out_shape).unwrap();

    assert_tensor_eq(&out, &expected);
}

#[test]
fn kernel_error_display_implemented() {
    // Test important errors which are most likely to occur
    let errors = [
        KernelError::InvalidRank,
        KernelError::ShapeMismatch,
        KernelError::InvalidArguments,
    ];
    common::validate_error_messages(&errors);
}
