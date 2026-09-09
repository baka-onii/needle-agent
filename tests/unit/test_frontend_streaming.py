"""Presentation timing tested with a deterministic clock, independent of a browser."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_pacing_primitives():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is only needed for the optional presentation unit tests.")
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_name("pacing.test.cjs"))],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
