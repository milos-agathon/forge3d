import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as asyncio (requires pytest-asyncio)")


def pytest_collection_modifyitems(config, items):
    try:
        import pytest_asyncio  # noqa: F401
        has_asyncio = True
    except Exception:
        has_asyncio = False

    if has_asyncio:
        return

    skip_asyncio = pytest.mark.skip(reason="pytest-asyncio not installed; skipping asyncio tests")
    for item in items:
        if 'asyncio' in item.keywords:
            item.add_marker(skip_asyncio)

