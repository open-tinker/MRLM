# Release Guide for MRLM

This document describes the process for releasing new versions of MRLM to PyPI.

## Pre-Release Checklist

Before releasing a new version, ensure all items are complete:

### 1. Code Quality
- [ ] All tests pass: `pytest`
- [ ] Code coverage meets targets: `pytest --cov=mrlm --cov-report=term`
- [ ] Type checking passes: `mypy src/`
- [ ] Linting passes: `ruff check src/`
- [ ] Code formatting is correct: `black --check src/`

### 2. Documentation
- [ ] README.md is up to date
- [ ] CHANGELOG.md is updated with new version
- [ ] API documentation builds: `cd docs && make html`
- [ ] Examples work with new version
- [ ] ARCHITECTURE.md reflects any changes

### 3. Version Management
- [ ] Version number updated in `pyproject.toml`
- [ ] Version number updated in `src/mrlm/__init__.py`
- [ ] CHANGELOG.md has entry for new version
- [ ] Git tag created for release

### 4. Package Contents
- [ ] All required files in MANIFEST.in
- [ ] Protocol buffer files included
- [ ] Example configs included
- [ ] py.typed marker file present

## Release Process

### Step 1: Update Version Number

Update version in **two places**:

1. `pyproject.toml`:
```toml
[project]
name = "mrlm"
version = "0.2.0"  # Update this
```

2. `src/mrlm/__init__.py`:
```python
__version__ = "0.2.0"  # Update this
```

### Step 2: Update CHANGELOG

Add entry to `CHANGELOG.md`:

```markdown
## [0.2.0] - 2024-XX-XX

### Added
- New feature 1
- New feature 2

### Changed
- Updated dependency X to version Y

### Fixed
- Bug fix 1
- Bug fix 2
```

### Step 3: Commit Changes

```bash
git add pyproject.toml src/mrlm/__init__.py CHANGELOG.md
git commit -m "chore: bump version to 0.2.0"
```

### Step 4: Create Git Tag

```bash
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin main
git push origin v0.2.0
```

### Step 5: Build Package

Clean previous builds:
```bash
rm -rf dist/ build/ *.egg-info
```

Build source and wheel distributions:
```bash
python -m build
```

This creates:
- `dist/mrlm-0.2.0.tar.gz` (source distribution)
- `dist/mrlm-0.2.0-py3-none-any.whl` (wheel)

### Step 6: Test Package Locally

Install in a fresh virtual environment:

```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from local build
pip install dist/mrlm-0.2.0-py3-none-any.whl

# Test imports
python -c "import mrlm; print(mrlm.__version__)"

# Test CLI
mrlm info

# Run a simple example
python examples/quickstart/simple_ppo.py

# Deactivate
deactivate
```

### Step 7: Upload to TestPyPI (Optional but Recommended)

First, upload to TestPyPI to catch any issues:

```bash
# Install twine if needed
pip install twine

# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*
```

Test installation from TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ mrlm
```

### Step 8: Upload to PyPI

Once everything is verified:

```bash
python -m twine upload dist/*
```

You'll be prompted for your PyPI credentials or API token.

### Step 9: Verify PyPI Release

1. Check package page: https://pypi.org/project/mrlm/
2. Verify README renders correctly
3. Test installation: `pip install mrlm`

### Step 10: Create GitHub Release

1. Go to https://github.com/youjiaxuan/MRLM/releases/new
2. Select the tag you created (v0.2.0)
3. Title: "MRLM v0.2.0"
4. Description: Copy relevant section from CHANGELOG.md
5. Attach built distributions (optional)
6. Publish release

## Post-Release

### Announce Release

- [ ] Update README.md to remove "(coming soon)" from PyPI installation
- [ ] Post announcement on GitHub Discussions
- [ ] Tweet/social media announcement (if applicable)
- [ ] Update documentation site

### Monitor

- [ ] Check PyPI download stats
- [ ] Monitor GitHub issues for bug reports
- [ ] Watch for user feedback

## Versioning Strategy

MRLM follows [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Incompatible API changes
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, backward compatible

### Examples:
- `0.1.0` → `0.2.0`: New environments or algorithms added
- `0.2.0` → `0.2.1`: Bug fixes
- `0.9.0` → `1.0.0`: API stabilization, breaking changes

## Hotfix Release Process

For urgent bug fixes:

1. Create hotfix branch: `git checkout -b hotfix/0.2.1`
2. Fix bug and add test
3. Update version to `0.2.1`
4. Update CHANGELOG
5. Merge to main
6. Follow normal release process

## Troubleshooting

### Build Fails

**Issue**: `python -m build` fails

**Solution**: 
- Ensure `build` is installed: `pip install build`
- Check pyproject.toml syntax
- Verify all imports work

### Upload Fails

**Issue**: `twine upload` fails with authentication error

**Solution**:
- Create PyPI API token at https://pypi.org/manage/account/token/
- Use token as password (username: `__token__`)
- Or configure `.pypirc`:
```ini
[pypi]
username = __token__
password = pypi-xxx...
```

### Version Conflict

**Issue**: Version already exists on PyPI

**Solution**:
- Cannot overwrite existing versions
- Increment version number and re-release
- Delete and re-create git tag if needed

### Missing Files in Package

**Issue**: Files missing in installed package

**Solution**:
- Check MANIFEST.in
- Verify files are in source tree
- Rebuild package
- Test with `tar -tzf dist/mrlm-*.tar.gz`

## Security

### API Tokens

- Use API tokens instead of passwords
- Scope tokens to specific projects
- Store tokens securely (use environment variables)
- Never commit tokens to git

### Code Signing

For production releases:
- Consider signing releases with GPG
- Add checksums to GitHub releases

## Automation (Future)

Consider setting up:
- GitHub Actions for automatic PyPI upload on tag push
- Automated changelog generation
- Automated version bumping
- Release notes generation

## Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)

## Support

For questions about releases:
- Open an issue on GitHub
- Contact maintainers at mrlm-dev@example.com
