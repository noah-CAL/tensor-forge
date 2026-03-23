#![allow(dead_code)]
use std::fmt::{Debug, Display};
use tensor_forge::tensor::Tensor;

pub fn validate_error_messages(errors: &[impl Display + Debug]) {
    for err in errors {
        assert!(
            err.to_string().len() >= 5,
            "Display not implemented properly for {:?}",
            err
        );
    }
}

pub fn assert_tensor_eq(actual: &Tensor, expected: &[f64]) {
    assert_eq!(
        expected,
        actual.data(),
        "tensor data mismatch (len expected={}, actual={})",
        expected.len(),
        actual.data().len(),
    );
}

/////////////////////////////
// Golden Model References //
/////////////////////////////
pub fn relu_ref(x: &[f64]) -> Vec<f64> {
    x.iter().map(|&v| if v < 0.0 { 0.0 } else { v }).collect()
}

pub fn add_ref(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

pub fn matmul_ref(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
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
