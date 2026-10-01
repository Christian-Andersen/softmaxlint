"""Error types shared across softmaxlint.

Every one of these signals that the run cannot produce a trustworthy verdict,
as opposed to a verdict that simply came back negative. The CLI maps them to
exit code 2; findings map to exit code 1.
"""

from __future__ import annotations


class SoftmaxLintError(Exception):
    """Base class for every recoverable softmaxlint failure."""


class DiscoveryError(SoftmaxLintError):
    """A requested path is missing, unreadable, or otherwise not lintable."""
