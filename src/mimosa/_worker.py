"""ProcessPool worker for compare_many: one serial numba thread per process."""

from __future__ import annotations

from numba import set_num_threads

from .profiles.alignment import profile_compare

_QUERY = None
_CONFIG = None


def _init_worker(query, config):
    set_num_threads(1)
    global _QUERY, _CONFIG
    _QUERY = query
    _CONFIG = config


def _compare_target(target):
    score, shift, orientation, n_sites, metric_str = profile_compare(
        _QUERY.bundle,
        _QUERY.anchors,
        target.bundle,
        target.anchors,
        _CONFIG,
    )
    return (target.name, score, shift, orientation, n_sites, metric_str)