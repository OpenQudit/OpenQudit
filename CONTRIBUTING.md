# Contributing to OpenQudit

First off, thank you for considering contributing! Help is always welcome, from bug reports and feature requests to documentation improvements and code contributions.

This document provides guidelines to help you get started.

## Table of Contents

- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
- [Development Setup](#development-setup)
  - [Python Development Setup](#python-development-setup)
    - [Python Tests](#python-tests)
  - [Pre-commit Hooks (recommended)](#pre-commit-hooks-recommended)
- [Updating Third-Party Licenses](#updating-third-party-licenses)
- [Pull Request Process](#pull-request-process)
- [Licensing](#licensing)

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report, please check the [existing issues](https://github.com/OpenQudit/openqudit/issues) to see if the bug has already been reported.

If you're opening a new issue, please include:

1.  A clear, descriptive title.
2.  A detailed description of the problem.
3.  Steps to reproduce the bug.
4.  The version of Rust used, operating system, and if using Python, the Python version.
5.  A **minimal reproducible example**, if possible. This is the most helpful thing you can do!

### Suggesting Enhancements

If you have an idea for a new feature or an improvement:

1.  Check the [existing issues](https://github.com/OpenQudit/openqudit/issues) to see if your idea has already been discussed.
2.  Open a new issue, clearly describing the feature you'd like to see, why it's needed/wanted, and any other valuable information regarding design or implementation.

### Your First Code Contribution

Unsure where to begin? Look for issues tagged `good first issue`. These are great starting points.

## Development Setup

We use a Rust workspace to manage multiple crates.

1. **Install** necessary dependencies. LLVM development libraries will need to be accessible to build OpenQudit.
2. **Fork** the repository on GitHub.
3. **Clone** your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/openqudit.git
   cd openqudit
   ```
4. **Build** the project to make sure everything is working:
   ```bash
   cargo build --workspace
   ```
5. **Run Tests** to confirm your local setup is correct:
   ```bash
   cargo test --workspace
   ```

### Python Development Setup

OpenQudit uses [Maturin](https://github.com/PyO3/maturin) and [uv](https://docs.astral.sh/uv) to build the python library from the qudit-python crate. This sets the `python` feature flag on all of the other rust crates. If you would like to do python-driven development, after ensuring you have installed `uv` and you have forked and cloned the repo:

Setup (in [crates/qudit-python](./crates/qudit-python)) a virtual environment:

```bash
uv sync --group=dev
```

This may take a second to build the package the first time.

#### Python Tests

To run Python tests, in the [qudit-python](./crates/qudit-python) crate, run:

```bash
uv run pytest tests
```

#### Python Stubs

Python stubs will automatically be built by pre-commit (see the [next section](#pre-commit-hooks-recommended)).

If you'd like to build the stubs manually to test Python additions, you can run `cargo run --bin stub_gen` in [./crates/qudit-python](./crates/qudit-python), and it will automatically generate new stubs in the [openqudit](./crates/qudit-python/openqudit) folder.

Stubs are built from the associated Rust/pyo3 implementation and doc comments.

### Pre-commit Hooks (recommended)

This repo ships a `.pre-commit-config.yaml` that runs the checks from the [Pull Request Process](#pull-request-process) below automatically:

**Note**: You will need `uv` installed (see [docs.astral.sh/uv](https://docs.astral.sh/uv) to run all the pre-commit hooks, as one of them runs Python tests.

1.  **Install** [pre-commit](https://pre-commit.com/) and [cargo-deny](https://github.com/EmbarkStudios/cargo-deny):

```bash
pip install pre-commit
cargo install cargo-deny
```

2.  **Set up the hooks** from the repo root:

```bash
pre-commit install
```

This installs two git hooks:

- **pre-commit** (every commit): runs `cargo fmt`, reformatting files in place. If it changes anything, the commit is aborted so you can review and re-stage.
- **pre-push** (every push): runs `cargo deny`, `cargo clippy`, `cargo test`, `cargo doc`, `pytest`, and `cargo run --bin stub_gen`.

You can also run all the push-stage checks manually at any time with `pre-commit run --all-files --hook-stage pre-push`.

## Updating Third-Party Licenses

Updating the `THIRD-PARTY-LICENSES.json` file requires [cargo-bundle-licenses](https://github.com/sstadick/cargo-bundle-licenses) to be installed.

1. In the qudit-python crate, regenerate to a temporary JSON file:

```bash
cargo bundle-licenses --format json --previous THIRD-PARTY-LICENSES.json --output tmp.json
```

2. Format the JSON to remove duplicate licenses if an MIT license is already present:

```bash
jq '.third_party_libraries |= map(
  if (.licenses | any(.license == "MIT")) then
    .license = "MIT" |
    .licenses |= map(select(.license == "MIT"))
  else
    .
  end
)' tmp.json > THIRD-PARTY-LICENSES.json
```

3. Remove `tmp.json`

4. Review the output for any licenses listed as `"NOT FOUND"`.

If there are any, find the license online, copy the file contents, format (for example on macOS: `pbpaste | jq -Rs | pbcopy`), and replace `"NOT FOUND"`.

## Pull Request Process

> [!TIP]
> Almost all of these steps can be run automatically or in one command with [pre-commit](#pre-commit-hooks-recommended)

When you're ready to submit your contribution:

1.  Create a new branch for your changes:

```bash
git checkout -b my-feature-branch
```

2.  Make your changes and commit them with a clear message.
3.  **Format your code:**

```bash
cargo fmt
```

4.  **Check licenses and advisories:**

```bash
cargo deny --all-features check -c .cargo/deny.toml
```

5.  **Check for lint warnings:**

```bash
cargo clippy --workspace --features python --all-targets -- -D warnings
```

6.  **Check for doc warnings:**

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
```

7.  **Run Rust tests:**

```bash
cargo test --workspace --features python
```

8.  **Generate Python stubs:**

```bash
cd crates/qudit-python
cargo run --bin stub_gen
```

9.  **Run Python tests:**

```bash
# In crates/qudit-python
uv run pytest tests/
```

10. **Push** your branch to your fork on GitHub:

```bash
# Back in project root
git add .
git commit -m "added my feature"
git push origin my-feature-branch
```

11. **Open a Pull Request** (PR) on the main repository.

12. In your PR description:

- Clearly describe the problem and your solution.
- If your PR fixes an existing issue, link it (e.g., "Fixes #123").
- Ensure all CI checks (GitHub Actions) are passing.

A maintainer will review your PR, provide feedback, and merge it when it's ready.

## Licensing

By contributing to this project, you agree that your contributions will be licensed under its [BSD-3-Clause](LICENSE) license.
