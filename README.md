# tensor-forge
<div align="center">

[![Crates.io](https://img.shields.io/crates/v/tensor-forge.svg)](https://crates.io/crates/tensor-forge)
[![Docs.rs](https://docs.rs/tensor-forge/badge.svg)](https://docs.rs/tensor-forge)
![coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/noah-CAL/tensor-forge/auto-badges/coverage-badge.json)
[![Build Status](https://github.com/noah-CAL/tensor-forge/actions/workflows/rust.yml/badge.svg)](https://github.com/noah-CAL/tensor-forge/actions/workflows/rust.yml)
![GitHub License](https://img.shields.io/github/license/noah-CAL/tensor-forge)

</div>

A minimal, deterministic compute graph runtime for tensor operations in Rust.

`tensor-forge` is a small, focused runtime project for building and executing tensor compute graphs with deterministic scheduling, graph-level validation, and pluggable kernel dispatch. It is designed to be readable, well-tested, and extensible rather than maximally optimized.

It provides:

- **`Graph`** for constructing directed acyclic tensor compute graphs
- **`Executor`** for deterministic graph execution
- **`KernelRegistry`** for pluggable operation dispatch
- **`Tensor`** as the runtime value type used for inputs and outputs

## Highlights

- **Deterministic execution** — graph execution order is stable and independent of map iteration order
- **Validated graph construction** — operations are shape-checked before execution
- **Pluggable kernels** — operation implementations are dispatched through a registry
- **Reusable graphs** — graph structure is defined once and executed with runtime input bindings
- **Well-tested and documented** — includes unit tests, integration tests, doctests, CI, and runnable examples

## Installation

Add this to your project's `Cargo.toml`:

```toml
[dependencies]
tensor-forge = "1.0.0"
```

In Rust code, import the crate as:

```rust
use tensor_forge::{Executor, Graph, KernelRegistry, Tensor};
```

## Quick Example

```rust
use tensor_forge::{Executor, Graph, KernelRegistry, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut g = Graph::new();

    let a = g.input_node(vec![2, 2]);
    let b = g.input_node(vec![2, 2]);

    let out = g.add(a, b)?;
    g.set_output_node(out)?;

    let a_tensor = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    let b_tensor = Tensor::from_vec(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0])?;

    let exec = Executor::new(KernelRegistry::default());
    let outputs = exec.execute(&g, vec![(a, a_tensor), (b, b_tensor)])?;

    let output = &outputs[&out];
    assert_eq!(output.shape(), &[2, 2]);
    assert_eq!(output.data(), &[11.0, 22.0, 33.0, 44.0]);

    Ok(())
}
```

## Runnable Examples

The `examples/` directory includes:

- `addition_graph.rs` — smallest complete graph execution example
- `branching_graph.rs` — branching graph with multiple operations
- `custom_kernel.rs` — defining and registering a custom kernel
- `feedforward_neural_net.rs` — programmatic construction of a small feedforward network

Run them with:

```rust
cargo run --example addition_graph
cargo run --example branching_graph
cargo run --example custom_kernel
cargo run --example feedforward_neural_net
```

## Current capabilities

`tensor-forge` currently supports:

- contiguous row-major tensor storage with `Vec<f64>`
- graph construction with explicit input and output nodes
- shape-checked graph operations
- deterministic topological ordering
- kernel dispatch through a default or user-provided registry
- end-to-end graph execution through `Executor`
- custom kernel definition and registration

## Scope and tradeoffs

`tensor-forge` is intentionally narrow in scope. Current tradeoffs include:

- serial execution only
- `f64` tensors only
- no GPU backend
- no graph-level optimization passes yet

These constraints keep the runtime core compact and make the implementation easier to reason about.

## Planned improvements

Planned next steps include:

- parallel execution support
- additional tensor operations
- graph-level optimization passes such as dead-node elimination
- improved evaluation strategies for memory usage and performance

## Development

Common commands:

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## Contributing

Contributions, bug reports, and suggestions are welcome. For substantial changes, please open an issue first to discuss the design.

Before submitting a pull request, please make sure formatting, linting, tests, and doctests all pass.

## License

This project is licensed under the MIT License.

See `LICENSE` for details.
