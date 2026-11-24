"""Ordered-pair R-B Monte Carlo using JAX score kernels."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .backend import _compare_circular, normalize_exact, scan_circular
from .arrays import EncodedSequences, RaggedArray, StrandPair
from .errors import MimosaError, ModelFormatError
from .io.bundles import model_fingerprint
from .models import MotifModel, SiteGA, strict_integer
from .profiles.anchors import parse_min_logerr
from .profiles.normalization import EmpiricalLogTail
from .profiles.sparse import count_exceedances as _count_sparse_exceedances, make_anchors
from .randomization import InputRelabeledModel, _compile_relabel_scorer, relabel_model

_COMPARISON_BLOCK_SIZE = 8
_REBUILD_ANCHORS = object()


def _target_storage_estimate(sequences, targets):
    """Estimate bytes needed to store normalized target profiles."""
    return len(targets) * (2 * sequences.data.size * 4 + 256)


@partial(jax.jit, static_argnames=("search_range", "return_mask"))
def _count_exceedances(queries, target, starts, sizes, threshold, observed, *,
                       search_range, return_mask=False):
    """Compare a bounded block of query profiles against one target on device."""
    scores = jax.vmap(
        lambda query: _compare_circular(
            query, target, starts, sizes, threshold, search_range=search_range,
        )[0]
    )(queries)
    hits = scores >= observed
    return hits if return_mask else jnp.sum(hits, dtype=jnp.int64)


def _permutations(model, seed, replicate, group):
    identity = model_fingerprint(model)
    digest = hashlib.sha256(f"mimosa-direct-mc-v2|{seed}|{replicate}|{identity}".encode()).digest()
    rng = np.random.default_rng(np.random.SeedSequence(np.frombuffer(digest, dtype="<u4")))
    if group == "full":
        return np.array([rng.permutation(4) for _ in range(model.motif_length)], dtype=np.uint8)
    result = np.tile(np.arange(4, dtype=np.uint8), (model.motif_length, 1))
    for row in result:
        row[[0, 3]] = rng.permutation([0, 3])
        row[[1, 2]] = rng.permutation([1, 2])
    return result


def _profile(model, sequences, background, *, relabel_scorer=None):
    if getattr(model, "sequence_scanner", False):
        raw_pair = scan_circular(model, sequences)
        cal_pair = raw_pair if background is None else scan_circular(model, background)
        raw = jnp.concatenate(raw_pair)
        calibration = jnp.maximum(*cal_pair)
        # Reduce on device and synchronize only the scalar; checking both batches
        # also covers every observed and randomized profile before normalization.
        finite = jnp.all(jnp.isfinite(raw))
        if background is not None:
            finite = finite & jnp.all(jnp.isfinite(calibration))
        if not bool(finite):
            raise ModelFormatError(
                "", f"{type(model).__name__} scan produced non-finite Float32 scores."
            )
        return normalize_exact(raw, calibration).reshape((2, sequences.data.size))
    # Generic window scorers run through the shared scan path, which validates
    # finiteness and yields host-side Float32 tracks.
    scan = model.scan if relabel_scorer is None else model._bind_relabel_scorer(relabel_scorer)
    pair = scan(sequences, strands="both", circular=True)
    if background is None:
        calibration = pair
    else:
        calibration = scan(background, strands="both", circular=True)
    raw = np.concatenate((pair.forward.data, pair.reverse.data))
    cal = jnp.maximum(jnp.asarray(calibration.forward.data),
                      jnp.asarray(calibration.reverse.data))
    return normalize_exact(jnp.asarray(raw), jnp.asarray(cal)).reshape((2, sequences.data.size))


def _read_cached_bundle(cache, key, offsets):
    from .cache import _cached_mmap_prepared_profile

    with cache._lock:
        entry = _cached_mmap_prepared_profile(cache, key)
    if entry is None:
        return None
    _, bundle, normalization, offset, circular = entry
    if normalization != EmpiricalLogTail() or not circular or offset != 0:
        return None
    if (not np.array_equal(bundle.forward.offsets, offsets)
            or not np.array_equal(bundle.reverse.offsets, offsets)):
        return None
    return bundle


def _cached_observed_profile(model, sequences, background, cache, context):
    """Return a cache key and mmap-backed bundle, computing it only on a miss."""
    from .cache import _prepared_profile_cache_key, _store_normalized_profile

    normalization = EmpiricalLogTail()
    key = _prepared_profile_cache_key(
        model, sequences, background=background, normalization=normalization, circular=True,
        sequence_fp=context[0], background_fp=context[1],
    )
    bundle = _read_cached_bundle(cache, key, sequences.offsets)
    if bundle is None:
        values = np.asarray(_profile(model, sequences, background), dtype=np.float32)
        bundle = StrandPair(
            RaggedArray(values[0], sequences.offsets),
            RaggedArray(values[1], sequences.offsets),
        )
        _store_normalized_profile(cache, key, model.name, bundle, normalization, 0, True)
        bundle = _read_cached_bundle(cache, key, sequences.offsets)
        if bundle is None:
            raise OSError("could not read the stored prepared profile from the cache.")
    return key, bundle


def _bundle_to_device(bundle):
    return jnp.stack((jnp.asarray(bundle.forward.data), jnp.asarray(bundle.reverse.data)))


@contextmanager
def _target_sources(profiles, *, use_disk, cache, prefetch, offsets, active):
    """Yield profiles in order; optionally prefetch a bounded number on the host."""
    source = enumerate(profiles) if active is None else ((i, profiles[i]) for i in active)
    if not prefetch or (not use_disk and cache is None):
        yield source
        return

    def load(saved):
        if cache is not None:
            bundle = _read_cached_bundle(cache, saved, offsets)
            if bundle is None:
                raise OSError("cached target profile became unavailable during Monte Carlo.")
            return np.stack((bundle.forward.data, bundle.reverse.data))
        return np.array(np.load(saved, mmap_mode="r", allow_pickle=False), copy=True)

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mimosa-prefetch")
    pending = deque()

    def submit_next():
        item = next(source, None)
        if item is not None:
            index, saved = item
            pending.append((index, executor.submit(load, saved)))

    try:
        for _ in range(min(prefetch, len(profiles) if active is None else len(active))):
            submit_next()

        def iterator():
            while pending:
                index, future = pending.popleft()
                value = future.result()
                submit_next()
                yield index, value

        yield iterator()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


class _TargetProfileStore:
    """Keep target handles on the host, backed by memory, temporary files or cache.

    The caller owns the temporary-directory lifetime. Only ``load`` transfers a
    saved target to the device, so no device profiles accumulate in this store.
    """

    def __init__(self, sequences, background, *, root, cache):
        self.sequences = sequences
        self.background = background
        self.root = root
        self.cache = cache
        self.profiles = []
        self.context = None
        if cache is not None:
            from .cache import _make_preparation_context

            self.context = _make_preparation_context(sequences, background)

    def prepare(self, model):
        if self.cache is None:
            return None, _profile(model, self.sequences, self.background)
        key, bundle = _cached_observed_profile(
            model, self.sequences, self.background, self.cache, self.context,
        )
        return key, _bundle_to_device(bundle)

    def append(self, key, profile):
        if self.cache is not None:
            self.profiles.append(key)
        elif self.root is not None:
            path = os.path.join(self.root, f"{len(self.profiles)}.npy")
            np.save(path, np.asarray(profile, dtype=np.float32), allow_pickle=False)
            self.profiles.append(path)
        else:
            self.profiles.append(np.asarray(profile, dtype=np.float32))

    def sources(self, prefetch, active):
        return _target_sources(
            self.profiles, use_disk=self.root is not None, cache=self.cache,
            prefetch=prefetch, offsets=self.sequences.offsets, active=active,
        )

    def load(self, saved, *, prefetch):
        if self.cache is not None and not prefetch:
            bundle = _read_cached_bundle(self.cache, saved, self.sequences.offsets)
            if bundle is None:
                raise OSError("cached target profile became unavailable during Monte Carlo.")
            return _bundle_to_device(bundle)
        if self.root is not None and not prefetch:
            saved = np.load(saved, mmap_mode="r", allow_pickle=False)
        return jnp.asarray(saved)


def _prepare_observed_profiles(query, targets, store, score, *, threshold, algorithm,
                               anchor_memory_limit, return_details):
    """Score observed pairs and save targets with a bounded host anchor budget."""
    observed = np.empty((1, len(targets)), dtype=np.float32)
    details, target_anchors = [], []
    anchor_bytes_used = 0
    _, query_profile = store.prepare(query)
    for j, target in enumerate(targets):
        key, target_profile = store.prepare(target)
        observed_score, shift, orientation, sites = score(query_profile, target_profile)
        observed[0, j] = np.asarray(observed_score)
        if return_details:
            details.append((int(shift), int(orientation), int(sites)))
        anchors = make_anchors(target_profile, threshold) if algorithm == "sparse" else None
        if anchors is not None:
            anchor_bytes = sum(item.nbytes for item in anchors)
            if anchor_bytes_used + anchor_bytes <= anchor_memory_limit:
                anchors = tuple(np.asarray(item) for item in anchors)
                anchor_bytes_used += anchor_bytes
            else:
                anchors = _REBUILD_ANCHORS
        target_anchors.append(anchors)
        store.append(key, target_profile)
        del target_profile
    return observed, details, target_anchors


def _prepare_replicate_blocks(query, sequences, background, *, first, end,
                              comparison_block_size, seed, group, threshold, algorithm,
                              relabel_scorer):
    """Build one bounded batch; replicate identities do not depend on its layout."""
    blocks = []
    for block_start in range(first, end, comparison_block_size):
        profiles = []
        for replicate in range(block_start, min(block_start + comparison_block_size, end)):
            variant = relabel_model(
                query, permutations=_permutations(query, seed, replicate, group),
            )
            if relabel_scorer is not None and isinstance(variant, InputRelabeledModel):
                profiles.append(_profile(variant, sequences, background, relabel_scorer=relabel_scorer))
            else:
                profiles.append(_profile(variant, sequences, background))
        block = jnp.stack(profiles)
        anchors = make_anchors(block, threshold) if algorithm == "sparse" else None
        blocks.append((block, anchors))
        del profiles
    return blocks


def _accumulate_exceedances(blocks, store, target_anchors, observed, counts, replicates, *,
                            starts, sizes, threshold, search_range, prefetch,
                            first, active, stop_after_exceedances):
    """Stream fixed targets through a replicate batch and update host counts."""
    sequential = stop_after_exceedances is not None
    options = {"search_range": search_range}
    if sequential:
        options["return_mask"] = True
    with store.sources(prefetch, active) as sources:
        for j, saved in sources:
            target_profile = store.load(saved, prefetch=prefetch)
            target_anchor = target_anchors[j]
            if target_anchor is _REBUILD_ANCHORS:
                target_anchor = make_anchors(target_profile, threshold)
            counter = jnp.int64(0)
            masks = []
            for block, query_anchor in blocks:
                if not store.sequences.data.size:
                    hit = jnp.float32(0) >= observed[0, j]
                    contribution = (jnp.broadcast_to(hit, (block.shape[0],)) if sequential
                                    else block.shape[0] * hit)
                elif query_anchor is not None and target_anchor is not None:
                    contribution = _count_sparse_exceedances(
                        block, target_profile, *query_anchor, *target_anchor,
                        starts, sizes, threshold, observed[0, j], **options,
                    )
                else:
                    contribution = _count_exceedances(
                        block, target_profile, starts, sizes, threshold, observed[0, j], **options,
                    )
                if sequential:
                    masks.append(contribution)
                else:
                    counter = counter + contribution
            if sequential:
                # Keep the exact stopping position, even inside a vectorized block.
                hits = np.flatnonzero(np.asarray(jnp.concatenate(masks)))
                remaining = stop_after_exceedances - counts[0, j]
                if hits.size >= remaining:
                    replicates[0, j] = first + hits[remaining - 1] + 1
                counts[0, j] += min(hits.size, remaining)
            else:
                counts[0, j] += int(counter)
            del target_profile


def direct_mc(query, targets, sequences, *, k, seed, group="full",
              min_logerr=4.0, search_range=5, background=None,
              batch_size=8, storage="auto", memory_limit=256 * 1024**2,
              temp_dir=None, comparison_block_size=_COMPARISON_BLOCK_SIZE,
              algorithm="sparse", prefetch=0, anchor_memory_limit=256 * 1024**2,
              stop_after_exceedances=None, return_replicates=False):
    """Return observed scores, exceedance counts and p-values for ordered pairs.

    Returns arrays of shape (1, number of targets). The query is randomized
    once per replicate and each target uses a prefix of the same replicate
    stream. Ties count as exceedances. By default p = (count + 1) / (k + 1).
    Set ``stop_after_exceedances=h`` to use the sequential Besag–Clifford
    p-value h / n when the h-th exceedance occurs on replicate n <= k.
    Targets with fewer than h exceedances use all k replicates and the
    fixed-sample formula. Set ``return_replicates=True`` to append the number
    of replicates used per target as a fourth array of the same shape.

    Query replicates are held on device in batches; target profiles are kept
    on the host or in an isolated temporary directory. Auto storage uses disk
    if their estimated size exceeds ``memory_limit`` bytes.
    ``comparison_block_size`` bounds the vectorized comparison block.
    The default ``algorithm="sparse"`` uses threshold anchors with dense
    fallback; ``algorithm="dense"`` always uses the original dense kernel.
    ``prefetch`` bounds host-side read-ahead for disk-backed targets (0 disables).
    Target anchors exceeding ``anchor_memory_limit`` are rebuilt on demand.

    Only PWM, BaMM, Dimont and Slim are supported natively on device; any other
    MotifModel with ``score_sites`` is supported through the shared scan path
    (SiteGA is temporarily disabled). JAX manages device execution; this
    routine does not change the caller's global x64 setting.
    """
    with jax.enable_x64(True):
        return _direct_mc(query, targets, sequences, k=k, seed=seed, group=group,
                          min_logerr=min_logerr, search_range=search_range,
                          background=background, batch_size=batch_size, storage=storage,
                          memory_limit=memory_limit, temp_dir=temp_dir,
                          comparison_block_size=comparison_block_size, algorithm=algorithm,
                          prefetch=prefetch, anchor_memory_limit=anchor_memory_limit,
                          stop_after_exceedances=stop_after_exceedances,
                          return_replicates=return_replicates)


def _direct_mc(query, targets, sequences, *, k, seed, group,
               min_logerr, search_range, background, batch_size, storage,
               memory_limit, temp_dir, comparison_block_size, algorithm, prefetch=0,
               anchor_memory_limit=256 * 1024**2,
               stop_after_exceedances=None, return_replicates=False,
               return_details=False, cache=None):
    k, seed = strict_integer(k, "k"), strict_integer(seed, "seed")
    search_range = strict_integer(search_range, "search_range")
    if k < 1 or seed < 0 or search_range < 0:
        raise ValueError("k must be positive; seed and search_range must be non-negative.")
    if stop_after_exceedances is not None:
        stop_after_exceedances = strict_integer(stop_after_exceedances, "stop_after_exceedances")
        if not 1 <= stop_after_exceedances <= k:
            raise ValueError("stop_after_exceedances must be between 1 and k.")
    threshold = parse_min_logerr(min_logerr)
    if group not in ("full", "gc"):
        raise ValueError("group must be 'full' or 'gc'.")
    batch_size = strict_integer(batch_size, "batch_size")
    comparison_block_size = strict_integer(comparison_block_size, "comparison_block_size")
    memory_limit = strict_integer(memory_limit, "memory_limit")
    prefetch = strict_integer(prefetch, "prefetch")
    anchor_memory_limit = strict_integer(anchor_memory_limit, "anchor_memory_limit")
    if batch_size < 1 or comparison_block_size < 1 or memory_limit < 0 or anchor_memory_limit < 0:
        raise ValueError("batch_size and comparison_block_size must be positive "
                         "and memory limits non-negative.")
    if storage not in ("auto", "memory", "disk"):
        raise ValueError("storage must be 'auto', 'memory' or 'disk'.")
    if algorithm not in ("dense", "sparse"):
        raise ValueError("algorithm must be 'dense' or 'sparse'.")
    if not 0 <= prefetch <= 64:
        raise ValueError("prefetch must be between 0 and 64.")
    if not isinstance(sequences, EncodedSequences):
        raise TypeError("sequences must be an EncodedSequences batch.")
    if background is not None and not isinstance(background, EncodedSequences):
        raise TypeError("background must be an EncodedSequences batch.")
    targets = list(targets)
    for model in [query, *targets]:
        if not isinstance(model, MotifModel):
            raise TypeError("direct Monte Carlo requires one query MotifModel and MotifModel targets.")
        if isinstance(model, SiteGA):
            raise MimosaError("SiteGA is temporarily disabled.")
        if not callable(getattr(model, "score_sites", None)):
            raise TypeError(
                f"{type(model).__name__} must provide a callable score_sites method."
            )
    counts = np.zeros((1, len(targets)), dtype=np.int64)
    if not targets:
        observed = np.empty((1, 0), dtype=np.float32)
        result = (observed, counts, np.empty_like(observed, dtype=np.float64))
        if return_replicates:
            result = (*result, np.empty_like(counts))
        return (*result, []) if return_details else result
    replicates = np.full_like(counts, k)
    if sequences.data.size > np.iinfo(np.int32).max:
        raise ValueError("JAX comparison currently supports at most 2**31-1 positions.")
    lengths = np.diff(sequences.offsets)
    starts = jnp.asarray(np.repeat(sequences.offsets[:-1], lengths).astype(np.int32))
    sizes = jnp.asarray(np.repeat(lengths, lengths).astype(np.int32))

    def score(query_profile, target_profile):
        if sequences.data.size == 0:
            return jnp.float32(0), jnp.int32(0), jnp.int32(0), jnp.int64(0)
        return _compare_circular(
            query_profile, target_profile, starts, sizes, threshold, search_range=search_range,
        )

    estimate = _target_storage_estimate(sequences, targets)
    use_disk = cache is None and (storage == "disk" or (storage == "auto" and estimate > memory_limit))
    storage_context = (
        tempfile.TemporaryDirectory(prefix="mimosa-mc-", dir=temp_dir)
        if use_disk else nullcontext(None)
    )
    with storage_context as root:
        if use_disk and shutil.disk_usage(root).free < estimate:
            raise OSError(f"Monte Carlo target profiles need approximately {estimate} bytes "
                          f"of temporary space in {root}.")
        target_store = _TargetProfileStore(sequences, background, root=root, cache=cache)
        observed, details, target_anchors = _prepare_observed_profiles(
            query, targets, target_store, score, threshold=threshold, algorithm=algorithm,
            anchor_memory_limit=anchor_memory_limit, return_details=return_details,
        )
        relabel_scorer = None if query.sequence_scanner else _compile_relabel_scorer(query)
        active = None
        for first in range(0, k, batch_size):
            if stop_after_exceedances is not None:
                active = np.flatnonzero(counts[0] < stop_after_exceedances)
                if active.size == 0:
                    break
            blocks = _prepare_replicate_blocks(
                query, sequences, background, first=first, end=min(first + batch_size, k),
                comparison_block_size=comparison_block_size, seed=seed, group=group,
                threshold=threshold, algorithm=algorithm, relabel_scorer=relabel_scorer,
            )
            _accumulate_exceedances(
                blocks, target_store, target_anchors, observed, counts, replicates,
                starts=starts, sizes=sizes, threshold=threshold,
                search_range=search_range, prefetch=prefetch, first=first,
                active=active, stop_after_exceedances=stop_after_exceedances,
            )
            del blocks
    pvalues = (counts + 1) / (k + 1)
    if stop_after_exceedances is not None:
        reached = counts == stop_after_exceedances
        pvalues[reached] = stop_after_exceedances / replicates[reached]
    result = (observed, counts, pvalues)
    if return_replicates:
        result = (*result, replicates)
    return (*result, details) if return_details else result


def _direct_mc_with_details(query, targets, sequences, **options):
    """Internal CLI path sharing observed and randomized profile preparation."""
    with jax.enable_x64(True):
        return _direct_mc(query, targets, sequences, return_details=True,
                          return_replicates=True, **options)
