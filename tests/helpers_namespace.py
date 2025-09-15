# tests/helpers_namespace.py
# No-op pytest plugin shim to satisfy tests that import 'helpers_namespace'.
# This file exists to avoid optional dependency on pytest-helpers-namespace for local runs.
# RELEVANT FILES:tests/test_restir.py

def pytest_configure(config):
    # Provide a minimal hook; no helpers injected.
    pass

