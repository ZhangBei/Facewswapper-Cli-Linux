"""Shared error types.

Kept in their own module so both the headless pipeline and the CLI can
raise/catch them without pulling in torch or OpenCV.
"""

from __future__ import annotations


class HeadlessError(RuntimeError):
    """Raised for user-facing failures (bad input, no face found, ...)."""
