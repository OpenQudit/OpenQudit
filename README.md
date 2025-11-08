# OpenQudit
[![Build, Lint, and Test](https://github.com/OpenQudit/openqudit/actions/workflows/build_lint_test.yml/badge.svg)](https://github.com/OpenQudit/openqudit/actions/workflows/build_lint_test.yml) [![Build Wheels](https://github.com/OpenQudit/openqudit/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/OpenQudit/openqudit/actions/workflows/build_wheels.yml) ![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg) [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](CONTRIBUTING.md) [![PyPI version](https://badge.fury.io/py/openqudit.svg)](https://pypi.org/project/openqudit)


A high-performance Rust library providing accelerated and extensible representation for quantum programs as circuits. OpenQudit fully supports qubit-based, qudit-based, and even many-radix based quantum circuits.

## 📦 Installation

### ![python-logo](https://s3.dualstack.us-east-2.amazonaws.com/pythondotorg-assets/media/community/logos/python-logo-only.png =18x) For Python Users

The `openqudit` package is available on PyPI:

```bash
pip install openqudit
```

### 🦀 For Rust Developers

You can add the individual crates you need to your `Cargo.toml`.

```toml
[dependencies]
qudit-core = "*"
qudit-expr = "*"
qudit-circuit = "*"
# … Add others as needed
```

## ⚡️ Quick Start

### Python Example

```python
# TODO
```

### Rust Example

```rust
# TODO
```

## 🏗️ Workspace Crate Overview

This workspace contains several related crates:

* **`crates/qudit-circuit`**: Data structures for representing extensible quantum circuits.
* **`crates/qudit-inst`**: Accelerated circuit instantiation and optimization subroutines.
* **`crates/qudit-python`**: The Python wrapper crate that builds the `openqudit` wheel.
* **`crates/qudit-tensor`**: Accelerated dense exact tensor network simulation.
* **`crates/qudit-expr`**: Symbolic expression engine and LLVM JIT compiler.
* **`crates/qudit-core`**: Core data structures, traits, and error types.
* **`crates/qudit-macros`**: Procedural macros used by other crates.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to get started.

## 📜 License

This project is licensed under a **BSD-3-Clause License**. See [LICENSE](LICENSE) for the full text.

*** Copyright Notice ***

OpenQudit Copyright (c) 2024, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to do so.

