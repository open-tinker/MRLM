# MRLM Documentation

This directory contains the Sphinx documentation for MRLM.

## Building the Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install -e ".[dev]"
```

Or install just the documentation tools:

```bash
pip install sphinx sphinx-rtd-theme
```

### Build HTML Documentation

```bash
cd docs
make html
```

The generated documentation will be in `docs/_build/html/`. Open `docs/_build/html/index.html` in your browser.

### Other Build Formats

```bash
# PDF (requires LaTeX)
make latexpdf

# Plain text
make text

# Man pages
make man

# Check for broken links
make linkcheck
```

### Clean Build

To clean the build directory:

```bash
make clean
```

## Documentation Structure

```
docs/
├── conf.py                 # Sphinx configuration
├── index.rst              # Main documentation page
├── installation.rst       # Installation guide
├── quickstart.rst         # Quick start guide
├── api/                   # API reference
│   ├── core.rst          # Core module docs
│   ├── algorithms.rst    # Algorithms module docs
│   ├── environments.rst  # Environments module docs
│   ├── distributed.rst   # Distributed training docs
│   ├── config.rst        # Configuration docs
│   └── cli.rst           # CLI docs
├── Makefile              # Build automation (Unix)
└── make.bat              # Build automation (Windows)
```

## Contributing to Documentation

### Writing Documentation

1. API documentation is auto-generated from docstrings in the source code
2. User guides are written in reStructuredText (.rst files)
3. Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for docstrings

### Docstring Example

```python
def my_function(param1: str, param2: int) -> bool:
    """Brief description of the function.

    More detailed explanation of what the function does,
    its behavior, and any important notes.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative

    Example:
        >>> my_function("test", 42)
        True
    """
    pass
```

### Building Locally

Before submitting documentation changes, build locally to check for errors:

```bash
cd docs
make clean
make html
```

Check the output for warnings or errors.

## Online Documentation

Once published, the documentation will be available at:
- **Read the Docs**: https://mrlm.readthedocs.io
- **GitHub Pages**: https://youjiaxuan.github.io/MRLM

## Related Documentation Files

- **[Main README](../README.md)**: Project overview and quick start
- **[Architecture Guide](../ARCHITECTURE.md)**: System architecture and design
- **[Installation Guide](../INSTALL.md)**: Detailed installation instructions
- **[Contributing Guide](../CONTRIBUTING.md)**: How to contribute to MRLM
- **[Examples README](../examples/README.md)**: Comprehensive examples guide
