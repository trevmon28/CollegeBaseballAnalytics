import sys
from pathlib import Path

import pytest

# Ensure project root is importable so 'api' and 'pipeline' resolve correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "asyncio: mark test coroutine to be run by pytest-asyncio",
    )
