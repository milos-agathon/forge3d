"""P2.2 COPC LAZ Fixture tests -- proxy to test_api_contracts.py Section 18.

The checklist gate ``python -m pytest tests/test_copc_laz_fixture.py -v -x``
expects this file to exist.  Canonical tests live in
``tests/test_api_contracts.py::TestCopcLazDecompression``.
"""

from tests.test_api_contracts import TestCopcLazDecompression  # noqa: F401
