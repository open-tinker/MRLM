# Contributing to MRLM

Thank you for your interest in contributing to MRLM! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Code](#contributing-code)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project follows a Code of Conduct. By participating, you agree to:
- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Prioritize community wellbeing

## Getting Started

### Areas for Contribution

We welcome contributions in several areas:

**New Features**:
- Additional RL algorithms (SAC, TD3, A2C, etc.)
- New task environments (summarization, translation, etc.)
- Performance optimizations
- Enhanced monitoring and visualization

**Bug Fixes**:
- Fix reported issues
- Improve error handling
- Address edge cases

**Documentation**:
- Improve existing documentation
- Add tutorials and examples
- Fix typos and clarify explanations
- Add code comments

**Testing**:
- Increase test coverage
- Add integration tests
- Improve test performance

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/MRLM.git
cd MRLM

# Add upstream remote
git remote add upstream https://github.com/youjiaxuan/MRLM.git
```

### 2. Create Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
```

## Contributing Code

### Code Style

MRLM follows these coding standards:

**Formatting**:
- Use Black for code formatting: `black src/ tests/`
- Line length: 100 characters
- Use double quotes for strings

**Linting**:
- Pass Ruff checks: `ruff check src/ tests/`
- Fix issues: `ruff check --fix src/ tests/`

**Type Hints**:
- Use type hints for all functions
- Pass mypy: `mypy src/`
- Use Python 3.9+ type syntax

**Documentation**:
- Google-style docstrings for all public functions
- Include examples in docstrings
- Document parameters and return values

### Example Code Style

```python
from typing import List, Optional

def process_trajectories(
    trajectories: List[Trajectory],
    min_reward: float = 0.0,
    max_length: Optional[int] = None,
) -> List[Trajectory]:
    """Process and filter trajectories based on criteria.
    
    Args:
        trajectories: List of trajectory objects to process
        min_reward: Minimum cumulative reward threshold
        max_length: Maximum trajectory length (None for no limit)
        
    Returns:
        Filtered list of trajectories meeting criteria
        
    Example:
        >>> trajs = [Trajectory(...), Trajectory(...)]
        >>> filtered = process_trajectories(trajs, min_reward=1.0)
        >>> len(filtered) <= len(trajs)
        True
    """
    filtered = [t for t in trajectories if t.total_reward() >= min_reward]
    
    if max_length is not None:
        filtered = [t for t in filtered if len(t) <= max_length]
    
    return filtered
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

**Examples**:
```
feat: add SAC algorithm implementation
fix: resolve CUDA memory leak in PPO trainer
docs: improve README quick start example
test: add integration tests for multi-env training
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/algorithms/test_ppo.py

# Run with coverage
pytest --cov=mrlm --cov-report=html

# Skip slow tests during development
pytest -m "not slow"
```

### Writing Tests

- Add tests for all new features
- Maintain or improve code coverage
- Use descriptive test names
- Include docstrings

**Example**:
```python
def test_trajectory_filtering_by_reward():
    """Test that trajectories are correctly filtered by reward threshold."""
    # Arrange
    high_reward_traj = create_trajectory(reward=5.0)
    low_reward_traj = create_trajectory(reward=0.5)
    
    # Act
    filtered = filter_trajectories([high_reward_traj, low_reward_traj], min_reward=1.0)
    
    # Assert
    assert len(filtered) == 1
    assert filtered[0] == high_reward_traj
```

### Test Requirements

- All tests must pass before merging
- New code must have >80% test coverage
- Integration tests for new features
- No warnings in test output

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def train(
    self,
    num_iterations: int,
    eval_every: int = 10,
    save_every: int = 50,
) -> None:
    """Train the model for specified iterations.
    
    Performs rollout collection, policy updates, and periodic evaluation.
    Saves checkpoints at specified intervals.
    
    Args:
        num_iterations: Number of training iterations to run
        eval_every: Evaluate every N iterations (default: 10)
        save_every: Save checkpoint every N iterations (default: 50)
        
    Raises:
        ValueError: If num_iterations is not positive
        RuntimeError: If training fails due to device errors
        
    Example:
        >>> trainer = PPOTrainer(policy_env, eval_envs, config)
        >>> trainer.train(num_iterations=100, eval_every=10)
    """
```

### Documentation Files

Update documentation when adding features:
- README.md for user-facing changes
- ARCHITECTURE.md for design changes
- examples/README.md for new examples
- API docs (docstrings) for all code

## Pull Request Process

### 1. Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] Added tests for new features
- [ ] Updated documentation
- [ ] Commit messages follow conventions
- [ ] No merge conflicts with main

### 2. Create Pull Request

**Title**: Use conventional commit format
```
feat: add multi-modal environment support
```

**Description**: Include:
- What changes were made
- Why the changes are needed
- How to test the changes
- Related issue numbers

**Template**:
```markdown
## Description
Brief description of changes

## Motivation
Why this change is needed

## Changes
- Change 1
- Change 2

## Testing
How to test these changes

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Code follows style guide

Closes #123
```

### 3. Review Process

- Maintainers will review your PR
- Address review comments
- Keep PR up to date with main
- PR must pass all CI checks

### 4. After Merge

- Delete your feature branch
- Sync your fork with upstream
- Celebrate! 🎉

## Development Workflow

### Typical Workflow

```bash
# 1. Sync with upstream
git checkout main
git pull upstream main

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Make changes and test
# ... edit files ...
pytest
black src/ tests/
ruff check src/ tests/

# 4. Commit changes
git add .
git commit -m "feat: add my feature"

# 5. Push to fork
git push origin feature/my-feature

# 6. Create pull request on GitHub
```

### Keeping Fork Updated

```bash
# Fetch upstream changes
git fetch upstream

# Merge upstream main
git checkout main
git merge upstream/main

# Push to your fork
git push origin main
```

## Release Process

See [RELEASE.md](RELEASE.md) for detailed release instructions.

Maintainers handle releases, but contributors can:
- Suggest version bumps in PRs
- Help update CHANGELOG.md
- Test release candidates

## Getting Help

- **Questions**: Open a Discussion on GitHub
- **Bugs**: Open an Issue
- **Chat**: Join our community (link TBD)
- **Email**: mrlm-dev@example.com

## Recognition

Contributors are recognized in:
- CHANGELOG.md for significant contributions
- README.md contributors section
- GitHub contributors page

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

## Additional Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Pytest Documentation](https://docs.pytest.org/)
- [Black Documentation](https://black.readthedocs.io/)

Thank you for contributing to MRLM! 🚀
