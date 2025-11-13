# Setting Up GitHub Repository

## Initial Setup (Already Done)

✅ Repository created: `gh repo create lagrs --public`
✅ Remote added: `origin` points to your GitHub repo

## Next Steps

### 1. Add All Files and Commit

```bash
# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: lagrs time series engine with MLE and parallel processing"

# Push to GitHub
git push -u origin main
```

If your default branch is `master` instead of `main`:
```bash
git push -u origin master
```

### 2. Verify GitHub Actions

After pushing, check:
- Go to: https://github.com/YOUR_USERNAME/lagrs/actions
- You should see the CI workflow running
- All tests should pass ✅

## GitHub Actions CI/CD

The repository includes automated testing that runs on every push:

### What Gets Tested

1. **Rust Tests** (`cargo test`)
   - All unit tests in `lagrs-core`
   - All unit tests in `lagrs-python`
   - All unit tests in `lagrs-bench`

2. **Code Quality**
   - Formatting check (`cargo fmt`)
   - Linting (`cargo clippy`)

3. **Python Tests** (Python 3.10, 3.11, 3.12)
   - Import tests
   - Rolling mean tests
   - ARIMA model tests
   - Batch processing tests

4. **Build Verification**
   - Release build check
   - Python extension build

### Running Tests Locally

Before pushing, run these to ensure CI will pass:

```bash
# Rust tests
cargo test --workspace

# Format check
cargo fmt --all -- --check

# Clippy
cargo clippy --workspace -- -D warnings

# Python tests
source .venv/bin/activate
pip install pytest
pytest tests/ -v

# Build check
cargo build --workspace --release
cd lagrs-python && maturin build --release
```

## Branch Protection (Optional)

To require tests to pass before merging:

1. Go to: Settings → Branches
2. Add rule for `main` branch
3. Enable: "Require status checks to pass before merging"
4. Select: `test`, `python-tests`, `build`

## Workflow Files

- `.github/workflows/ci.yml` - Main CI workflow
- `tests/test_lagrs.py` - Python unit tests

## Troubleshooting

### If push fails:
```bash
# Check remote
git remote -v

# If wrong URL, fix it:
git remote set-url origin https://github.com/YOUR_USERNAME/lagrs.git
```

### If tests fail locally:
- Make sure Rust is installed: `rustc --version`
- Make sure Python 3.10+ is available
- Install dependencies: `pip install maturin pytest numpy`

