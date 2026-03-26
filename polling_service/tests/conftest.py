"""
Pytest configuration for polling_service tests.

Ensure imports like `import llm_factory` resolve to this repository's modules
instead of any third-party packages that might be installed in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path


# polling_service/tests -> polling_service -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

