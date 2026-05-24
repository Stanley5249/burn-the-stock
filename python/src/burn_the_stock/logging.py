"""Shared logging configuration for command-line entry points."""

import logging
import sys


def setup() -> None:
    """Configure root logging to stdout with a plain message formatter.

    Uses brace-style formatting so log records print as bare messages
    without timestamps or level prefixes, matching script output style.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("{message}", style="{"))
    logging.basicConfig(level=logging.INFO, handlers=[handler])
