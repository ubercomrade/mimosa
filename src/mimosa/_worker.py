"""ProcessPool worker for compare_many: one serial numba thread per process."""

from __future__ import annotations

from numba import set_num_threads

from .cache import Cache
from .profiles.alignment import profile_compare
from .profiles.prepared import PreparedProfile, prepare_profile

_QUERY = None
_CONFIG = None
_SEQUENCES = None
_BACKGROUND = None
_CACHE = None
_NORMALIZATION = None


def _init_worker(query, config, sequences, background, cache_dir, normalization):
    set_num_threads(1)
    global _QUERY, _CONFIG, _SEQUENCES, _BACKGROUND, _CACHE, _NORMALIZATION
    _QUERY = query
    _CONFIG = config
    _SEQUENCES = sequences
    _BACKGROUND = background
    _CACHE = Cache(cache_dir) if cache_dir else None
    _NORMALIZATION = normalization


def _prepare_and_compare(target):
    if isinstance(target, PreparedProfile):
        prepared = target
    else:
        prepared = prepare_profile(
            target,
            _SEQUENCES,
            background=_BACKGROUND,
            min_logerr=_QUERY.min_logerr,
            normalization=_NORMALIZATION,
            cache=_CACHE,
        )
    score, shift, orientation, n_sites, metric_str = profile_compare(
        _QUERY.bundle,
        _QUERY.anchors,
        prepared.bundle,
        prepared.anchors,
        _CONFIG,
    )
    return (prepared.name, score, shift, orientation, n_sites, metric_str)