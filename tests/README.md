# MRLM Test Suite

Comprehensive test suite for MRLM library.

## Test Structure

```
tests/
├── core/                   # Core module tests
│   ├── test_types.py      # Type classes
│   ├── test_base.py       # Base environment classes
│   └── test_llm_environment.py  # LLM environment
├── algorithms/            # Algorithm tests
│   ├── test_ppo.py       # PPO algorithm
│   ├── test_grpo.py      # GRPO algorithm
│   ├── test_dpo.py       # DPO algorithm
│   └── test_sft.py       # SFT algorithm
├── environments/          # Environment tests
│   ├── test_code.py      # Code execution
│   ├── test_math.py      # Math reasoning
│   ├── test_debate.py    # Multi-agent debate
│   └── test_tools.py     # Tool use
├── distributed/           # Distributed training tests
│   └── test_utils.py     # FSDP, DDP, utilities
├── config/                # Configuration tests
│   └── test_training_config.py
├── cli/                   # CLI tests
│   └── test_commands.py  # CLI commands
├── integration/           # Integration tests
│   └── test_end_to_end.py  # End-to-end workflows
├── conftest.py           # Shared fixtures
└── README.md             # This file
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Core tests only
pytest tests/core/

# Algorithm tests
pytest tests/algorithms/

# Environment tests
pytest tests/environments/

# Integration tests
pytest tests/integration/
```

### Run with Coverage

```bash
# Generate coverage report
pytest --cov=mrlm --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Run Specific Test Markers

```bash
# Run only fast tests (exclude slow tests)
pytest -m "not slow"

# Run only integration tests
pytest -m integration

# Run only end-to-end tests
pytest -m e2e

# Run tests that require GPU
pytest -m requires_gpu
```

### Run Specific Test Files

```bash
# Single file
pytest tests/core/test_types.py

# Specific test class
pytest tests/core/test_types.py::TestMessage

# Specific test method
pytest tests/core/test_types.py::TestMessage::test_message_creation
```

### Verbose Output

```bash
# More verbose
pytest -v

# Very verbose
pytest -vv

# Show print statements
pytest -s
```

## Test Markers

Tests are organized using pytest markers:

- **`slow`**: Tests that take longer to run (model loading, training)
- **`integration`**: Integration tests that test multiple components together
- **`e2e`**: End-to-end tests that test complete workflows
- **`requires_gpu`**: Tests that require GPU (will be skipped if no GPU available)

## Writing Tests

### Test File Structure

```python
"""Tests for <module>."""

import pytest
from mrlm.<module> import <class>


class TestClassName:
    """Test <ClassName> class."""

    def test_feature(self):
        """Test specific feature."""
        # Arrange
        obj = ClassName()

        # Act
        result = obj.method()

        # Assert
        assert result == expected
```

### Using Fixtures

Common fixtures are defined in `conftest.py`:

```python
def test_with_model(model, tokenizer):
    """Test using model fixture."""
    # model and tokenizer are automatically loaded
    assert model is not None
```

Available fixtures:
- `model`: Small test model
- `tokenizer`: Tokenizer for test model
- `device`: CPU or CUDA device
- `temp_dir`: Temporary directory (auto-cleaned)
- `sample_messages`: Sample message list
- `sample_observation`: Sample observation
- `sample_reward`: Sample reward

### Adding New Tests

1. Create test file in appropriate directory
2. Follow naming convention: `test_<module>.py`
3. Use descriptive test names: `test_<what_is_being_tested>`
4. Add docstrings explaining what is tested
5. Use markers for slow/integration tests
6. Add fixtures to `conftest.py` if needed by multiple tests

### Test Categories

**Unit Tests**: Test individual functions/methods in isolation
```python
def test_add_numbers():
    """Test add function."""
    assert add(2, 3) == 5
```

**Integration Tests**: Test multiple components working together
```python
@pytest.mark.integration
def test_environment_with_model(model, tokenizer):
    """Test environment integrated with model."""
    env = LLMEnvironment(model, tokenizer)
    obs = env.reset()
    assert obs is not None
```

**End-to-End Tests**: Test complete workflows
```python
@pytest.mark.e2e
@pytest.mark.slow
def test_complete_training_pipeline(model, tokenizer):
    """Test full training workflow."""
    # Setup, train, evaluate, save
    ...
```

## Continuous Integration

Tests are run automatically on:
- Every pull request
- Every push to main branch
- Nightly builds

### CI Test Matrix

- **Python versions**: 3.9, 3.10, 3.11
- **OS**: Ubuntu, macOS, Windows
- **Test suites**:
  - Fast tests (unit tests)
  - Integration tests
  - GPU tests (on GPU runners)

## Coverage Requirements

- **Minimum coverage**: 80%
- **Core modules**: 90%
- **Algorithm implementations**: 85%
- **Environments**: 80%

## Troubleshooting

### Import Errors

Make sure MRLM is installed:
```bash
pip install -e .
```

### Slow Tests

Skip slow tests during development:
```bash
pytest -m "not slow"
```

### GPU Tests

GPU tests are automatically skipped if no GPU available. To force run:
```bash
pytest -m requires_gpu
```

### Debugging Failed Tests

```bash
# Show local variables on failure
pytest -l

# Drop into debugger on failure
pytest --pdb

# Run last failed tests only
pytest --lf
```

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Clear Names**: Use descriptive test and variable names
3. **One Assertion**: Prefer one logical assertion per test
4. **Fast Feedback**: Keep unit tests fast (<1s)
5. **Clean Up**: Use fixtures for setup/teardown
6. **Mock External**: Mock external dependencies when possible
7. **Parameterize**: Use `@pytest.mark.parametrize` for multiple inputs
8. **Document**: Add docstrings explaining what is tested

## Example: Parameterized Test

```python
@pytest.mark.parametrize("input,expected", [
    (2, 4),
    (3, 9),
    (4, 16),
])
def test_square(input, expected):
    """Test squaring numbers."""
    assert square(input) == expected
```

## Example: Using Fixtures

```python
@pytest.fixture
def sample_env():
    """Create sample environment for testing."""
    env = DummyEnvironment()
    yield env
    env.close()  # Cleanup

def test_env_reset(sample_env):
    """Test environment reset."""
    obs = sample_env.reset()
    assert obs is not None
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [MRLM Contributing Guide](../CONTRIBUTING.md)
