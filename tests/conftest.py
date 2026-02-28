import asyncio

import pytest


def pytest_collection_modifyitems(items):
    """Use signal-based timeout for async tests (thread method can't interrupt event loops)."""
    for item in items:
        if asyncio.iscoroutinefunction(item.obj):
            item.add_marker(pytest.mark.timeout(method="signal"))
