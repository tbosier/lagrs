# GitHub Actions CI/CD

This directory contains GitHub Actions workflows for continuous integration.

## Workflows

### `ci.yml`

Runs on every push and pull request to main/master/develop branches:

1. **Rust Tests**: Runs `cargo test` on all workspace crates
2. **Format Check**: Ensures code is properly formatted with `cargo fmt`
3. **Clippy**: Runs Rust linter with `cargo clippy`
4. **Python Tests**: Tests Python bindings on Python 3.10, 3.11, and 3.12
5. **Build Check**: Verifies the project builds successfully in release mode

## Requirements

All tests must pass before merging pull requests. The workflow will:
- ✅ Run all Rust unit tests
- ✅ Check code formatting
- ✅ Run clippy linter
- ✅ Test Python bindings on multiple Python versions
- ✅ Verify release builds work

## Local Testing

Before pushing, you can run the same checks locally:

```bash
# Run tests
cargo test --workspace

# Check formatting
cargo fmt --all -- --check

# Run clippy
cargo clippy --workspace -- -D warnings

# Build
cargo build --workspace --release

# Test Python bindings
cd lagrs-python
maturin develop
pytest tests/
```

