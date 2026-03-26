"""
Pytest configuration.

Some CI environments may have a third-party package named `llm_factory`
installed, which can shadow this repo's local `llm_factory.py` depending on
how the test runner configures import paths.

Force the repository root to the front of sys.path so `import llm_factory`
resolves to the local module.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

