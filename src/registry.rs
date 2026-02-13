use std::collections::HashMap;

use crate::kernel::{AddKernel, Kernel, MatMulKernel, ReluKernel};
use crate::op::OpKind;

pub struct KernelRegistry {
    kernels: HashMap<OpKind, Box<dyn Kernel>>,
}

impl Default for KernelRegistry {
    fn default() -> Self {
        let mut kernels: HashMap<OpKind, Box<dyn Kernel>> = HashMap::new();
        kernels.insert(OpKind::Add, Box::new(AddKernel));
        kernels.insert(OpKind::MatMul, Box::new(MatMulKernel));
        kernels.insert(OpKind::ReLU, Box::new(ReluKernel));
        Self { kernels }
    }
}

impl KernelRegistry {
    pub fn new() -> Self {
        Self {
            kernels: HashMap::new(),
        }
    }

    pub fn register(&mut self, op: OpKind, kernel: Box<dyn Kernel>) -> Option<Box<dyn Kernel>> {
        self.kernels.insert(op, kernel)
    }

    pub fn get(&self, op: OpKind) -> Option<&dyn Kernel> {
        self.kernels.get(&op).map(|k| k.as_ref())
    }
}
