//! Kernel registry for runtime dispatch of compute operations.
//!
//! A [`KernelRegistry`] maps graph-level operations ([`OpKind`]) to concrete
//! compute implementations ([`Kernel`]). This allows the execution engine to
//! resolve an operation at runtime without hardcoding kernel types.
//!
//! Kernels are stored as trait objects (`Box<dyn Kernel>`), enabling different
//! kernel implementations to be registered and overridden dynamically.
//!
//! # Examples
//!
//! Register and retrieve a kernel:
//!
//! ```
//! use tensor_forge::kernel::AddKernel;
//! use tensor_forge::op::OpKind;
//! use tensor_forge::registry::KernelRegistry;
//! use tensor_forge::tensor::Tensor;
//!
//! let mut reg = KernelRegistry::new();
//! assert!(reg.register(OpKind::Add, Box::new(AddKernel)).is_none());
//!
//! let kernel = reg.get(&OpKind::Add).unwrap();
//!
//! let shape = vec![1, 4];
//! let a = Tensor::from_vec(shape.clone(), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
//! let b = Tensor::from_vec(shape.clone(), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
//! let mut out = Tensor::zeros(shape).unwrap();
//!
//! kernel.compute(&[&a, &b], &mut out).unwrap();
//! assert_eq!(out.data(), &[11.0, 22.0, 33.0, 44.0]);
//! ```
//!
//! Overwrite an existing kernel:
//!
//! ```
//! use tensor_forge::kernel::{AddKernel, MatMulKernel};
//! use tensor_forge::op::OpKind;
//! use tensor_forge::registry::KernelRegistry;
//!
//! let mut reg = KernelRegistry::new();
//!
//! assert!(reg.register(OpKind::Add, Box::new(AddKernel)).is_none());
//! assert!(reg.register(OpKind::Add, Box::new(MatMulKernel)).is_some());
//! ```
use std::collections::HashMap;

use crate::kernel::{AddKernel, Kernel, MatMulKernel, ReluKernel};
use crate::op::OpKind;

/// Registry mapping [`OpKind`] to runtime-executable [`Kernel`] implementations.
///
/// This type is used by the execution engine to dispatch operations to the
/// correct kernel at runtime.
pub struct KernelRegistry {
    kernels: HashMap<OpKind, Box<dyn Kernel>>,
}

impl Default for KernelRegistry {
    /// Creates a registry populated with the built-in kernels.
    ///
    /// Registers:
    /// - [`OpKind::Add`] → [`crate::kernel::AddKernel`]
    /// - [`OpKind::MatMul`] → [`crate::kernel::MatMulKernel`]
    /// - [`OpKind::ReLU`] → [`crate::kernel::ReluKernel`]
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_forge::op::OpKind;
    /// use tensor_forge::registry::KernelRegistry;
    ///
    /// let reg = KernelRegistry::default();
    /// assert!(reg.get(&OpKind::Add).is_some());
    /// assert!(reg.get(&OpKind::MatMul).is_some());
    /// assert!(reg.get(&OpKind::ReLU).is_some());
    /// ```
    fn default() -> Self {
        let mut registry = Self::new();
        let _ = registry.register(OpKind::Add, Box::new(AddKernel));
        let _ = registry.register(OpKind::MatMul, Box::new(MatMulKernel));
        let _ = registry.register(OpKind::ReLU, Box::new(ReluKernel));
        registry
    }
}

impl KernelRegistry {
    /// Creates an empty kernel registry.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_forge::registry::KernelRegistry;
    /// use tensor_forge::op::OpKind;
    ///
    /// let reg = KernelRegistry::new();
    ///
    /// assert!(reg.get(&OpKind::Add).is_none());
    /// assert!(reg.get(&OpKind::MatMul).is_none());
    /// assert!(reg.get(&OpKind::ReLU).is_none());
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            kernels: HashMap::new(),
        }
    }

    /// Registers `kernel` as the implementation for `op`.
    ///
    /// Returns the previously registered kernel if one existed.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_forge::kernel::{AddKernel, MatMulKernel};
    /// use tensor_forge::op::OpKind;
    /// use tensor_forge::registry::KernelRegistry;
    ///
    /// let mut reg = KernelRegistry::new();
    ///
    /// let old_kernel = reg.register(OpKind::Add, Box::new(AddKernel));
    /// assert!(old_kernel.is_none());
    /// assert!(reg.get(&OpKind::Add).is_some());
    ///
    /// // Add conflicting mapping
    /// let old_kernel = reg.register(OpKind::Add, Box::new(MatMulKernel));
    /// assert!(old_kernel.is_some());            // returns old AddKernel Mapping.
    /// assert!(reg.get(&OpKind::Add).is_some());  // `OpKind::Add` now maps to MatMulKernel.
    ///
    /// ```
    #[must_use]
    pub fn register(&mut self, op: OpKind, kernel: Box<dyn Kernel>) -> Option<Box<dyn Kernel>> {
        self.kernels.insert(op, kernel)
    }

    /// Returns the kernel registered for `op`, if present.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_forge::kernel::AddKernel;
    /// use tensor_forge::op::OpKind;
    /// use tensor_forge::registry::KernelRegistry;
    ///
    /// let mut reg = KernelRegistry::new();
    /// reg.register(OpKind::Add, Box::new(AddKernel));
    ///
    /// assert!(reg.get(&OpKind::Add).is_some());
    /// assert!(reg.get(&OpKind::MatMul).is_none());
    /// ```
    #[must_use]
    pub fn get(&self, op: &OpKind) -> Option<&dyn Kernel> {
        self.kernels.get(op).map(Box::as_ref)
    }
}
