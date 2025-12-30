# Installation Fix for Import Errors

If you're getting `ModuleNotFoundError: No module named 'mrlm.data'` or similar errors, you need to install the package.

## Solution: Install MRLM in Development Mode

From the MRLM root directory, run:

```bash
cd /path/to/MRLM
pip install -e .
```

The `-e` flag installs in "editable" mode, which means:
- Changes to the code are immediately reflected
- No need to reinstall after editing
- Perfect for development

## Alternative: Add to Python Path (Temporary)

If you don't want to install, you can add to Python path in your script:

```python
import sys
from pathlib import Path

# Add MRLM to path
mrlm_root = Path(__file__).parent
sys.path.insert(0, str(mrlm_root / "src"))

# Now imports will work
from mrlm.core import LLMEnvironment, EnvironmentMode
```

## Verify Installation

After installing, verify it works:

```bash
python -c "import mrlm; print(f'MRLM version: {mrlm.__version__}')"
```

You should see:
```
MRLM version: 0.1.0
```

## If You Get Other Errors

If you get errors about missing dependencies (torch, transformers, etc.), install them:

```bash
# Install all dependencies
pip install -e ".[dev]"

# Or install minimal dependencies
pip install torch transformers grpcio pyyaml
```
