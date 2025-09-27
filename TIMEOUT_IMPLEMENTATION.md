# Timeout Implementation for Structure Learning Algorithms

## Overview

This implementation adds timeout functionality to structure learning algorithms, starting with causal-learn (Python-based) algorithms. The timeout is controlled via the existing `--maxtime` command line parameter.

## Implementation Details

### 1. Command Line Interface

The existing `--maxtime` parameter now serves dual purposes:
- **Analysis timeouts** (original): limits analysis execution time  
- **Algorithm timeouts** (new): limits individual algorithm execution time

Usage:
```bash
# Set 60-second timeout for algorithm execution
python -m pytest experiments/run_learn.py --action=replace --series=CAUSAL_DEF --networks=abc --N=100 --maxtime=60

# No timeout (existing behavior)
python -m pytest experiments/run_learn.py --action=replace --series=CAUSAL_DEF --networks=abc --N=100
```

### 2. Core Timeout Utility (`core/timing.py`)

- **`run_with_timeout()`**: Executes any function with a timeout using threading
- **`with_timeout()`**: Decorator version for algorithm functions
- **`TimeoutError`**: Custom exception for timeout conditions
- **Thread-based approach**: Compatible with both direct Python calls and subprocess calls
- **Integrated with existing timing utilities**: Added to the existing timing module for better organization

### 3. Causal-Learn Integration (`call/causal.py`)

- Added `maxtime` parameter to `causal_learn()` function
- Wrapped GES algorithm execution in timeout functionality
- Converts `TimeoutError` to `RuntimeError` for consistency with existing error handling
- Maintains backward compatibility (maxtime=None means no timeout)

### 4. Experiment Framework Integration (`experiments/run_learn.py`)

- Updated `do_experiment()` to accept and pass through `maxtime` parameter
- Integrated timeout into causal-learn algorithm calls
- Maintains existing error handling patterns

## Architecture Benefits

1. **Consistent Interface**: Uses existing `--maxtime` parameter
2. **Extensible Design**: `run_with_timeout()` can be applied to any algorithm type
3. **Backward Compatible**: No timeout when `maxtime=None`
4. **Small Steps**: Started with Python-based algorithms as requested

## Next Steps for Complete Implementation

### R Algorithms (bnlearn)
- Enhance `dispatch_r()` in `call/r.py` to support subprocess timeouts
- Add process termination logic for R subprocesses
- Update `bnlearn_learn()` to use timeout wrapper

### Java Algorithms (TETRAD)  
- Enhance `dispatch_cmd()` in `call/cmd.py` to support subprocess timeouts
- Add process termination logic for Java subprocesses
- Update `tetrad_learn()` to use timeout wrapper

### Implementation Pattern
```python
# For subprocess-based algorithms
def enhanced_dispatch_r(package, method, params, timeout_seconds=None):
    if timeout_seconds:
        # Use subprocess.run() with timeout parameter
        # Add process termination logic
    else:
        # Use existing implementation
```

## Testing

Created test file `call/test/test_causal_timeout.py` to verify:
- Timeout functionality works correctly
- Normal operation unaffected when no timeout specified
- Proper error handling and conversion

## Usage Examples

```python
# Direct API usage
from call.causal import causal_learn

# With timeout
graph, trace = causal_learn("ges", data, maxtime=300)  # 5 minutes

# Without timeout  
graph, trace = causal_learn("ges", data, maxtime=None)  # No limit

# Using timeout utilities directly
from core.timing import run_with_timeout, with_timeout

# Functional approach
result = run_with_timeout(my_function, args=(arg1, arg2), timeout_seconds=60)

# Decorator approach
@with_timeout(120)
def my_algorithm(data):
    return process(data)
```

This implementation provides a solid foundation for extending timeout functionality to all algorithm types while maintaining the existing architecture and user interface.
