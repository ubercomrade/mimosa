"""Progress-bar and logging helpers for CLI-facing long-running workflows."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm import tqdm

_T = TypeVar("_T")


def should_enable_progress(progress: bool | None) -> bool:
    """Resolve tri-state progress settings into one runtime decision."""
    if progress is None:
        return sys.stderr.isatty()
    return bool(progress)


def iter_progress(
    iterable: Iterable[_T],
    *,
    enabled: bool | None,
    desc: str | None = None,
    total: int | None = None,
    leave: bool = True,
) -> Iterator[_T]:
    """Yield from an iterable, optionally rendering a tqdm bar on stderr."""
    if not should_enable_progress(enabled):
        yield from iterable
        return

    yield from tqdm(iterable, desc=desc, total=total, leave=leave)


class TqdmLoggingHandler(logging.StreamHandler):
    """Logging handler that writes via tqdm so log lines do not corrupt bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            tqdm.write(message, file=self.stream)
            self.flush()
        except Exception:  # pragma: no cover - mirrors logging.Handler.emit robustness
            self.handleError(record)
