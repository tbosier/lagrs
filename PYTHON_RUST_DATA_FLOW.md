# Python-Rust Data Structure Connection

This document explains how data flows between Python and Rust in the lagrs library, including the underlying data structures and memory management.

## Overview

lagrs uses **PyO3** (Python-O3 bindings) to create a bridge between Python and Rust. The key principle is **zero-copy data transfer** where possible, using numpy arrays as the common data structure.

## Data Flow Architecture

```
Python (numpy array) 
    ↓
PyO3 Binding Layer (PyReadonlyArray1 / PyArray1)
    ↓
Rust Slice (&[f64]) - ZERO COPY
    ↓
Rust Processing (Vec<f64> for results)
    ↓
PyO3 Binding Layer (PyArray1)
    ↓
Python (numpy array)
```

## Key Data Structures

### 1. Python Side: `numpy.ndarray`

```python
import numpy as np
data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
```

- **Memory Layout**: Contiguous array of f64 values in memory
- **Ownership**: Python manages memory via reference counting
- **Type**: Must be `dtype=np.float64` for compatibility

### 2. PyO3 Binding: `PyReadonlyArray1<f64>`

```rust
fn rolling_mean_py(
    series: PyReadonlyArray1<f64>,  // <-- Read-only view
    window: usize,
) -> PyResult<Bound<PyArray1<f64>>>
```

**What it is:**
- A **read-only wrapper** around a numpy array
- Provides **zero-copy access** to the underlying memory
- Does NOT copy data from Python to Rust
- Maintains Python's GIL (Global Interpreter Lock) safety

**Memory:**
- Points directly to Python's numpy array memory
- No allocation in Rust for input data
- Lifetime tied to Python object

### 3. Rust Slice: `&[f64]`

```rust
let array = series.as_array();
let data = array.as_slice().unwrap();  // Zero-copy slice
```

**What it is:**
- A **reference** to the contiguous memory from numpy
- Zero-copy: points to same memory as Python array
- Can be used directly in Rust code
- Read-only (immutable reference)

**Memory:**
- No copy occurs
- Points to Python-managed memory
- Must respect Rust's borrowing rules

### 4. Rust Processing: `Vec<f64>`

```rust
let result = rolling_mean(data, window);  // Returns Vec<f64>
```

**What it is:**
- Owned Rust vector for computation results
- Allocated in Rust's heap
- Can be modified, resized, etc.
- Independent of Python memory

**Memory:**
- Allocated by Rust
- Owned by Rust
- Will be converted back to Python array

### 5. PyO3 Output: `PyArray1<f64>`

```rust
Ok(PyArray1::from_vec_bound(series.py(), result))
```

**What it is:**
- A **writable** numpy array wrapper
- Created from Rust `Vec<f64>`
- Will be converted to Python numpy array
- Owned by Python after return

**Memory:**
- Initially owned by Rust
- Transferred to Python on return
- Python manages memory via reference counting

## Zero-Copy Mechanism

### Input (Python → Rust)

```rust
// Python side
data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
result = lagrs.rolling_mean(data, 50)

// Rust side (what happens internally)
fn rolling_mean_py(series: PyReadonlyArray1<f64>, ...) {
    let array = series.as_array();           // Get numpy array view
    let data = array.as_slice().unwrap();    // Get Rust slice - ZERO COPY!
    // data now points to same memory as Python's numpy array
}
```

**Key Points:**
- `PyReadonlyArray1` doesn't copy data
- `as_slice()` returns a reference to Python's memory
- No memory allocation for input
- Python retains ownership

### Output (Rust → Python)

```rust
// Rust side
let result: Vec<f64> = rolling_mean(data, window);  // Allocated in Rust
Ok(PyArray1::from_vec_bound(py, result))           // Convert to Python array

// Python side
result = lagrs.rolling_mean(data, 50)  # Now a numpy array
```

**Key Points:**
- Rust `Vec<f64>` is allocated in Rust
- `PyArray1::from_vec_bound()` creates a Python numpy array
- Memory is transferred to Python (not copied if possible)
- Python manages the memory after return

## Interior Mutability Pattern

### Why RefCell?

```rust
pub struct ARIMA {
    ar_params: RefCell<Option<Vec<f64>>>,  // Why RefCell?
    // ...
}
```

**Problem:**
- PyO3 methods must take `&self` (immutable reference)
- But we need to mutate internal state after `fit()`
- Rust's borrow checker prevents `&mut self` in PyO3

**Solution:**
- `RefCell` provides **interior mutability**
- Allows mutation through immutable reference
- Runtime borrow checking (panics if violated)
- Safe for single-threaded Python GIL context

**Usage:**
```rust
// In fit() method (takes &self, not &mut self)
*self.ar_params.borrow_mut() = Some(params);  // Mutate through RefCell

// In forecast() method
let params = self.ar_params.borrow();  // Read through RefCell
```

## Parallel Processing with Rayon

### How Parallelism Works

```rust
// Sequential (old way)
for sku in sku_data {
    model.fit(sku);  // One at a time
}

// Parallel (new way)
sku_data.par_iter()  // Rayon parallel iterator
    .map(|y| {
        let model = ARIMA::new(p, d, q);
        model.fit(y)  // Multiple SKUs processed simultaneously
    })
    .collect()
```

**Memory Safety:**
- Each parallel task gets its own copy of data
- No shared mutable state between threads
- Rust's ownership system prevents data races
- Rayon manages thread pool automatically

**Python Interaction:**
- Python passes data to Rust (zero-copy)
- Rust processes in parallel (copies for each thread)
- Results collected and returned to Python
- Python receives results as numpy arrays

## Complete Example: Data Flow

```python
# Python
import numpy as np
import lagrs

# Create numpy array (Python manages memory)
data = np.random.randn(1000).astype(np.float64)

# Call Rust function
result = lagrs.rolling_mean(data, 50)
```

**What happens step-by-step:**

1. **Python → PyO3**: numpy array wrapped in `PyReadonlyArray1`
   - No copy, just a view

2. **PyO3 → Rust Slice**: `as_slice()` gets `&[f64]`
   - Points to Python's memory
   - Zero-copy reference

3. **Rust Processing**: `rolling_mean()` processes data
   - Reads from Python's memory (zero-copy)
   - Allocates new `Vec<f64>` for results

4. **Rust → PyO3**: `PyArray1::from_vec_bound()` creates numpy array
   - Transfers ownership to Python
   - Python manages memory

5. **PyO3 → Python**: Returns numpy array
   - Python receives result
   - Can use immediately

## Memory Management

### Who Owns What?

| Stage | Data Structure | Owner | Memory Location |
|-------|---------------|-------|----------------|
| Python input | `numpy.ndarray` | Python | Python heap |
| PyO3 input | `PyReadonlyArray1` | Python (view) | Python heap (same memory) |
| Rust processing | `&[f64]` | Python (borrowed) | Python heap (same memory) |
| Rust result | `Vec<f64>` | Rust | Rust heap |
| PyO3 output | `PyArray1` | Python | Python heap (transferred) |
| Python output | `numpy.ndarray` | Python | Python heap |

### Garbage Collection

- **Python**: Uses reference counting + cycle detector
- **Rust**: Uses ownership system (no GC needed)
- **PyO3**: Manages Python object lifetimes automatically
- **Result**: No memory leaks, automatic cleanup

## Performance Implications

### Zero-Copy Benefits

1. **No Memory Allocation** for input data
2. **No Data Copying** from Python to Rust
3. **Fast Access** - direct memory pointer
4. **Low Overhead** - just pointer dereference

### When Copies Happen

1. **Input**: Never (zero-copy via slice)
2. **Processing**: Only for results (necessary)
3. **Parallel Processing**: Each thread gets copy (necessary for safety)
4. **Output**: Transfer to Python (not a copy, ownership transfer)

## Best Practices

### For Python Users

```python
# ✅ Good: Use numpy arrays
data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
result = lagrs.rolling_mean(data, 50)

# ❌ Bad: Python lists (will be converted, slower)
data = [1.0, 2.0, 3.0]  # Slower conversion
result = lagrs.rolling_mean(data, 50)
```

### For Rust Developers

```rust
// ✅ Good: Use slices for input (zero-copy)
fn process(data: &[f64]) -> Vec<f64> {
    // Process without copying input
}

// ❌ Bad: Taking ownership (unnecessary copy)
fn process(data: Vec<f64>) -> Vec<f64> {
    // Unnecessary allocation
}
```

## Summary

The Python-Rust connection in lagrs uses:

1. **Zero-copy input**: numpy arrays accessed via Rust slices
2. **Interior mutability**: `RefCell` for stateful Python objects
3. **Ownership transfer**: Results transferred (not copied) to Python
4. **Parallel safety**: Rayon ensures thread-safe parallel processing
5. **Automatic memory management**: PyO3 handles Python object lifetimes

This design provides **maximum performance** (zero-copy where possible) while maintaining **safety** (Rust's type system + Python's GC).

