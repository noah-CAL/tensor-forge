
use tensor_forge::tensor::{Tensor, TensorError};

#[test]
fn tensor_creation_valid() {
    let shape = vec![4, 4];
    let data: Vec<f64> = (0..16).map(|x| x as f64).collect();

    let tensor = Tensor::from_vec(shape.clone(), data.clone());
    assert!(tensor.is_ok());

    let mut tensor = tensor.unwrap();
    assert_eq!(tensor.shape(), shape);
    assert_eq!(tensor.data(), data);
    assert_eq!(tensor.numel(), 4 * 4);
    assert_eq!(tensor.data_mut(), data);
}

#[test]
fn tensor_creation_invalid_shape() {
    // First, test mismatch between shape and data
    let shape = vec![1, 4];
    let data: Vec<f64> = (0..16).map(|x| x as f64).collect();

    let tensor = Tensor::from_vec(shape.clone(), data.clone());
    assert!(matches!(tensor.unwrap_err(), TensorError::ShapeMismatch));

    // Second, test illegal shape (dimension of zero)
    let shape = vec![0];
    let data: Vec<f64> = Vec::new();

    let tensor = Tensor::from_vec(shape.clone(), data.clone());
    assert!(matches!(tensor.unwrap_err(), TensorError::InvalidShape));
}

#[test]
fn tensor_zeros_correct_size() {
    // 1) Test with 4x4 tensor
    let shape = vec![4, 4];
    let tensor = Tensor::zeros(shape.clone());
    assert!(tensor.is_ok());

    let mut tensor = tensor.unwrap();
    assert_eq!(tensor.shape(), shape);
    assert_eq!(tensor.data(), vec![0_f64; 4 * 4]);
    assert_eq!(tensor.numel(), 4 * 4);
    assert_eq!(tensor.data_mut(), vec![0_f64; 4 * 4]);

    // 2) Test with 2x16x9 tensor
    let shape = vec![2, 16, 9];
    let tensor = Tensor::zeros(shape.clone());
    assert!(tensor.is_ok());

    let mut tensor = tensor.unwrap();
    assert_eq!(tensor.shape(), shape);
    assert_eq!(tensor.data(), vec![0_f64; 2 * 16 * 9]);
    assert_eq!(tensor.numel(), 2 * 16 * 9);
    assert_eq!(tensor.data_mut(), vec![0_f64; 2 * 16 * 9]);
}

#[test]
fn tensor_mutability_test() {
    let shape = vec![4, 4];
    let mut tensor = Tensor::zeros(shape.clone()).unwrap();
    assert_eq!(tensor.data(), vec![0_f64; 4 * 4]);

    let numel = tensor.numel();
    let data: &mut [f64] = tensor.data_mut();
    for i in 0..numel {
        data[i] = i as f64;
    }
    let expected_data: Vec<f64> = (0..16).map(|x| x as f64).collect();
    assert_eq!(tensor.data(), expected_data);
}
