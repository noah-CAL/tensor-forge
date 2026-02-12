
# Tensor-Forge
<div align="center">

<!--[![Crates.io](https://img.shields.io/crates/v/tensor-forge.svg)]-->
<!-- [![Docs.rs](https://docs.rs/tensor-forge/badge.svg)] -->
![coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/noah-CAL/tensor-forge/auto-badges/coverage-badge.json)
[![Build Status](https://github.com/noah-CAL/tensor-forge/actions/workflows/rust.yml/badge.svg)](https://github.com/noah-CAL/tensor-forge/actions/workflows/rust.yml)
![GitHub License](https://img.shields.io/github/license/noah-CAL/tensor-forge)

</div>

A minimal compute-graph runtime library in Rust for executing tensor operations with a pluggable kernel model and deterministic scheduling.

This project is a readable “runtime infrastructure” sandbox inspired by ML runtimes (e.g., dispatcher/IR/execution layers), focused on correctness, reproducible execution, and clean module boundaries.

**Design/roadmap:** see [`docs/outline.md`](docs/outline.md)

## Status

In active development. Current focus: Tensor + Graph IR + deterministic topological ordering; kernels/executor are next.

## What’s implemented

- **Tensor**: contiguous row-major storage (`Vec<f64>`) with shape invariants and validation
- **Graph IR**: nodes (`NodeId`, `OpKind`) stored for O(1) lookup, shape-checked op construction (`add`, `matmul`, `relu`)
- **Determinism**: execution order is intended to be independent of `HashMap` iteration

## Repository layout

- `src/tensor.rs` — tensor storage + shape validation
- `src/op.rs` — graph-level ops (`Input`, `MatMul`, `Add`, `ReLU`)
- `src/node.rs` — `Node` / `NodeId`
- `src/graph.rs` — graph construction, validation, and topological sort
- `tests/*` — integration tests
- `docs/outline.md` — living spec / roadmap

## Quick start

```bash
cargo test
```

## Library Status
- deterministic topo_sort + determinism tests
- kernel trait + registry
- executor to run graphs end-to-end

## License
This project is licensed under the MIT license. 

See [LICENSE](LICENSE) for details.

---

> MIT License
> 
> Copyright (c) 2026 Noah Sedlik
> 
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
> 
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
> 
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
