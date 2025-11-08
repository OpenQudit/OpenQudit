# Contributing to OpenQudit

First off, thank you for considering contributing! Help is always welcome, from bug reports and feature requests to documentation improvements and code contributions.

This document provides guidelines to help you get started.

## Table of Contents

* [How Can I Contribute?](#how-can-i-contribute)
    * [Reporting Bugs](#reporting-bugs)
    * [Suggesting Enhancements](#suggesting-enhancements)
    * [Your First Code Contribution](#your-first-code-contribution)
* [Development Setup](#development-setup)
* [Pull Request Process](#pull-request-process)
* [Licensing](#licensing)

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
2.  **Fork** the repository on GitHub.
3.  **Clone** your fork locally:
    ```bash
    git clone https://github.com/YOUR_USERNAME/openqudit.git
    cd openqudit
    ```
4.  **Build** the project to make sure everything is working:
    ```bash
    cargo build --workspace
    ```
5.  **Run Tests** to confirm your local setup is correct:
    ```bash
    cargo test --workspace
    ```

## Pull Request Process

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
4.  **Check for lint warnings:**
    ```bash
    cargo clippy --workspace --features python --all-targets -- -D warnings
    ```
5.  **Check for doc warnings:**
    ```bash
    RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
    ```
5.  **Run all tests:**
    ```bash
    cargo test --workspace --features python
    ```
6.  **Push** your branch to your fork on GitHub:
    ```bash
    git push origin my-feature-branch
    ```
7.  **Open a Pull Request** (PR) on the main repository.
8.  In your PR description:
    * Clearly describe the problem and your solution.
    * If your PR fixes an existing issue, link it (e.g., "Fixes #123").
    * Ensure all CI checks (GitHub Actions) are passing.

A maintainer will review your PR, provide feedback, and merge it when it's ready.

## Licensing

By contributing to this project, you agree that your contributions will be licensed under its [BSD-3-Clause](LICENSE) license.

