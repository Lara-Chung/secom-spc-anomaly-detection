"""Shared test helpers."""

import os

import pytest


def skip_or_fail(reason: str) -> None:
    """Skip a test whose database or raw files are unavailable.

    With SECOM_TESTS_REQUIRE_DB=1 (set by run_all.sh) the test fails instead,
    so a full reproduction cannot succeed with the database tests skipped.
    """
    if os.environ.get("SECOM_TESTS_REQUIRE_DB") == "1":
        pytest.fail(reason)
    pytest.skip(reason)
