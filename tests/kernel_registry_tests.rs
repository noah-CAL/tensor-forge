// tests/kernel_registry_tests.rs

use tensor_forge::kernel::{AddKernel, MatMulKernel, ReluKernel};
use tensor_forge::op::OpKind;
use tensor_forge::tensor::Tensor;

use tensor_forge::registry::KernelRegistry;

#[test]
fn registry_get_missing_returns_none() {
    let reg = KernelRegistry::new();
    assert!(reg.get(&OpKind::Add).is_none());
    assert!(reg.get(&OpKind::MatMul).is_none());
    assert!(reg.get(&OpKind::ReLU).is_none());
}

#[test]
fn registry_register_then_get_returns_kernel() {
    let mut reg = KernelRegistry::new();

    assert!(reg.register(OpKind::Add, Box::new(AddKernel)).is_none());

    let k = reg.get(&OpKind::Add).expect("Should return Add kernel.");

    // Ensure we can call through the trait object (object safety / dyn dispatch works).
    let shape = vec![1, 4];
    let a = Tensor::from_vec(shape.clone(), vec![1.0, 2.0, 3.0, 4.0])
        .expect("Tensor creation should not fail.");
    let b = Tensor::from_vec(shape.clone(), vec![10.0, 20.0, 30.0, 40.0])
        .expect("Tensor creation should not fail.");
    let mut out = Tensor::zeros(shape).expect("Tensor creation should not fail.");

    k.compute(&[&a, &b], &mut out)
        .expect("Addition compute should pass. All tensors and dimensions are valid.");
    assert_eq!(out.data(), &[11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn registry_default_has_expected_kernels() {
    let reg = KernelRegistry::default();

    // Default must register MatMul, Add, ReLU.
    assert!(reg.get(&OpKind::MatMul).is_some());
    assert!(reg.get(&OpKind::Add).is_some());
    assert!(reg.get(&OpKind::ReLU).is_some());

    // Input is a graph-level op. Registry does not provide a compute kernel for it.
    assert!(reg.get(&OpKind::Input).is_none());
}

#[test]
fn registry_can_register_multiple_ops() {
    let mut reg = KernelRegistry::new();
    assert!(
        reg.register(OpKind::MatMul, Box::new(MatMulKernel))
            .is_none()
    );
    assert!(reg.register(OpKind::Add, Box::new(AddKernel)).is_none());
    assert!(reg.register(OpKind::ReLU, Box::new(ReluKernel)).is_none());

    assert!(reg.get(&OpKind::MatMul).is_some());
    assert!(reg.get(&OpKind::Add).is_some());
    assert!(reg.get(&OpKind::ReLU).is_some());
}

#[test]
fn registry_dispatch_matmul_add_relu_works() {
    let reg = KernelRegistry::default();

    // MatMul: (2x3) * (3x2) -> (2x2)
    let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("Tensor creation should not fail");
    let b = Tensor::from_vec(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        .expect("Tensor creation should not fail");
    let mut mm_out = Tensor::zeros(vec![2, 2]).expect("Tensor creation should not fail");
    reg.get(&OpKind::MatMul)
        .unwrap()
        .compute(&[&a, &b], &mut mm_out)
        .expect("MatMul operation should succeed");
    assert_eq!(mm_out.data(), &[58.0, 64.0, 139.0, 154.0]);

    // Add: (2x2) + (2x2)
    let d = Tensor::from_vec(vec![2, 2], vec![1.0, -100.0, 2.0, -200.0])
        .expect("Tensor creation should not fail");
    let mut add_out = Tensor::zeros(vec![2, 2]).expect("Tensor creation should not fail");
    reg.get(&OpKind::Add)
        .unwrap()
        .compute(&[&mm_out, &d], &mut add_out)
        .expect("Addition operation should succeed");
    assert_eq!(add_out.data(), &[59.0, -36.0, 141.0, -46.0]);

    // ReLU: elementwise max(0, x)
    let mut relu_out = Tensor::zeros(vec![2, 2]).unwrap();
    reg.get(&OpKind::ReLU)
        .unwrap()
        .compute(&[&add_out], &mut relu_out)
        .expect("ReLU operation should succeed");
    assert_eq!(relu_out.data(), &[59.0, 0.0, 141.0, 0.0]);
}

#[test]
fn registry_overwrite_returns_old() {
    let mut reg = KernelRegistry::new();

    // 1) Register AddKernel under OpKind::Add (first insert -> None)
    assert!(reg.register(OpKind::Add, Box::new(AddKernel)).is_none());

    // 2) Overwrite OpKind::Add with a different kernel type (MatMulKernel)
    // This is intentionally incorrect use
    assert!(reg.register(OpKind::Add, Box::new(MatMulKernel)).is_some());

    // 3) Prove registry changed: calling `OpKind::Add` now runs MatMulKernel.
    let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("Tensor creation should not fail");
    let b = Tensor::from_vec(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
        .expect("Tensor creation should not fail");
    let mut out = Tensor::zeros(vec![2, 2]).expect("Tensor creation should not fail");

    reg.get(&OpKind::Add)
        .unwrap()
        .compute(&[&a, &b], &mut out)
        .unwrap();

    assert_eq!(out.data(), &[58.0, 64.0, 139.0, 154.0]);

    // 4) Overwrite back to AddKernel and prove dispatch changes again.
    assert!(reg.register(OpKind::Add, Box::new(AddKernel)).is_some());

    let x = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0])
        .expect("Tensor creation should not fail");
    let y = Tensor::from_vec(vec![1, 4], vec![10.0, 20.0, 30.0, 40.0])
        .expect("Tensor creation should not fail");
    let mut out2 = Tensor::zeros(vec![1, 4]).expect("Tensor creation should not fail");

    reg.get(&OpKind::Add)
        .unwrap()
        .compute(&[&x, &y], &mut out2)
        .unwrap();

    assert_eq!(out2.data(), &[11.0, 22.0, 33.0, 44.0]);
}
