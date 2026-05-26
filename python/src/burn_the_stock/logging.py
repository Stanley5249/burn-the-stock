"""Shared logging configuration for command-line entry points."""

import logging
import sys


def setup() -> None:
    """Configure root logging to stdout with brace-style formatting.

    Uses brace-style formatting so log records print as
    levelname:name:message without timestamps.
    """
    handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, handlers=[handler], style="{")
