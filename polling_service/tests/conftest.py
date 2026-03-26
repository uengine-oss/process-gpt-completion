"""
Pytest configuration for polling_service tests.

`working-directory: polling_service` jobs still need the repo root on sys.path
for third-party `llm_factory` shadowing, but **polling_service/** must come
*before* the repo root so unqualified imports like `database` and
`process_definition` resolve to this service package (not the repo-root modules,
which differ and can break `workitem_processor` unit tests).
"""

from __future__ import annotations

import sys
from pathlib import Path


_POLLING_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
# After both inserts: [POLLING_ROOT, REPO_ROOT, ...]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_POLLING_ROOT))

