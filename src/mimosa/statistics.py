"""Null distributions, p-values, BH FDR, and result annotation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .compare import ComparisonResult, compare
from .errors import MimosaError
from .io.bundles import (
    NULL_FORMAT_VERSION,
    content_fingerprint_float64,
    model_collection_fingerprint,
    sequence_fingerprint,
)
from .models import PWM, strict_integer
from .profiles.alignment import METRIC_NAME, ProfileConfig
from .profiles.anchors import DEFAULT_MIN_LOGERR, parse_min_logerr
from .profiles.normalization import HybridEmpiricalLogTail, normalization_fingerprint
from .profiles.prepared import _prepare_profile

SAMPLING_VERSION = "original-shuffled-ordered-pairs-v3"
ALIGNMENT_VERSION = "circular-jaccard-v1"


@dataclass(frozen=True)
class NullDistribution:
    strategy: str
    metric: str
    raw_scores: np.ndarray
    pairs: tuple[tuple[str, str, float], ...]
    n_null: int
    n_models: int
    shuffle: bool
    seed: int
    sampling_version: str
    model_collection_fingerprint: str | None
    sequence_fingerprint: str
    background_fingerprint: str
    contract: Mapping

    def __post_init__(self):
        try:
            raw_scores = np.array(self.raw_scores, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("null distribution raw_scores must be numeric.") from exc
        if raw_scores.ndim != 1:
            raise ValueError("null distribution raw_scores must be one-dimensional.")
        if not np.all(np.isfinite(raw_scores)):
            raise ValueError("null distribution raw_scores must be finite.")
        raw_scores.setflags(write=False)
        object.__setattr__(self, "raw_scores", raw_scores)
        try:
            n_null = strict_integer(self.n_null, "n_null")
            n_models = strict_integer(self.n_models, "n_models")
            seed = strict_integer(self.seed, "seed")
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
        object.__setattr__(self, "n_null", n_null)
        object.__setattr__(self, "n_models", n_models)
        object.__setattr__(self, "seed", seed)
        if self.n_null <= 0:
            raise ValueError("null distribution n_null must be positive.")
        if self.n_null != len(self.raw_scores):
            raise ValueError("null distribution n_null does not match raw_scores length.")
        if self.n_models < 2:
            raise ValueError("null distribution requires at least two source models.")
        if len(self.pairs) != self.n_null:
            raise ValueError("null distribution pairs do not match n_null.")
        if self.seed < 0:
            raise ValueError("null distribution seed must be non-negative.")
        if not self.sampling_version:
            raise ValueError("null distribution sampling_version must not be empty.")
        if self.strategy != "profile":
            raise ValueError("only profile null distributions are supported.")
        metric = self.metric
        if metric != METRIC_NAME:
            raise ValueError(
                f"null distribution metric must be '{METRIC_NAME}', got '{metric}'."
            )
        pairs = []
        for index, pair in enumerate(self.pairs):
            if not isinstance(pair, (tuple, list)) or len(pair) != 3:
                raise ValueError(
                    f"null distribution pair {index} must contain variant IDs and a score."
                )
            query_id, target_id, score = pair
            if not isinstance(query_id, str) or not query_id:
                raise ValueError(f"null distribution pair {index} query variant ID is invalid.")
            if not isinstance(target_id, str) or not target_id:
                raise ValueError(f"null distribution pair {index} target variant ID is invalid.")
            if not np.isfinite(score):
                raise ValueError(f"null distribution pair {index} score must be finite.")
            if float(score) != float(raw_scores[index]):
                raise ValueError(
                    f"null distribution pair {index} score does not match raw_scores."
                )
            pairs.append((query_id, target_id, float(score)))
        object.__setattr__(self, "pairs", tuple(pairs))
        if not isinstance(self.contract, Mapping):
            raise ValueError("null distribution contract must be a mapping.")
        contract = dict(self.contract)
        required = (
            "metric",
            "search_range",
            "min_logerr",
            "normalization_version",
            "alignment_version",
            "sequence_fingerprint",
            "background_fingerprint",
            "raw_scores_fingerprint",
        )
        missing = [key for key in required if key not in contract]
        if missing:
            raise ValueError(
                "null distribution contract is missing required fields: "
                + ", ".join(missing)
                + "."
            )
        if contract["metric"] != metric:
            raise ValueError("null distribution contract metric does not match metric.")
        try:
            search_range = strict_integer(contract["search_range"], "search_range")
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
        if search_range < 0:
            raise ValueError("null distribution contract search_range must be non-negative.")
        contract["search_range"] = search_range
        try:
            min_logerr = parse_min_logerr(contract["min_logerr"])
        except ValueError as exc:
            raise ValueError(
                "null distribution contract min_logerr must be a finite, positive Float32 value."
            ) from exc
        contract["min_logerr"] = min_logerr
        for key in (
            "normalization_version",
            "alignment_version",
            "sequence_fingerprint",
            "background_fingerprint",
        ):
            if not isinstance(contract[key], str) or not contract[key]:
                raise ValueError(f"null distribution contract {key} must be a non-empty string.")
        actual_fingerprint = content_fingerprint_float64(raw_scores)
        if contract["raw_scores_fingerprint"] != actual_fingerprint:
            raise ValueError(
                "null distribution raw_scores fingerprint does not match the payload."
            )
        contract["raw_scores_fingerprint"] = actual_fingerprint
        object.__setattr__(self, "contract", MappingProxyType(contract))

@dataclass(frozen=True)
class AnnotatedResult:
    result: ComparisonResult
    p_value: float | None = None
    adj_p_value: float | None = None
    e_value: float | None = None
    null_id: str | None = None
    null_n: int | None = None
    null_estimator: str | None = None

    def __getattr__(self, name):
        # Only reached for names missing on the dataclass itself; guard
        # "result" so copy/pickle on a half-built instance cannot recurse.
        if name == "result" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self.result, name)

    def to_dict(self):
        d = self.result.to_dict()
        d["annotation_schema_version"] = 1
        if self.p_value is not None:
            d["p-value"] = self.p_value
        if self.adj_p_value is not None:
            d["adj.p-value"] = self.adj_p_value
        if self.e_value is not None:
            d["E-value"] = self.e_value
        if self.null_id is not None:
            d["null_id"] = self.null_id
        if self.null_n is not None:
            d["null_n"] = self.null_n
        if self.null_estimator is not None:
            d["null_estimator"] = self.null_estimator
        return d


def adjusted_pvalues(pvalues):
    p = np.asarray(pvalues, dtype=np.float64)
    n = p.size
    if n == 0:
        return np.array([], dtype=np.float64)
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be finite and lie in [0, 1].")
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)
    adj = np.minimum.accumulate((sorted_p * n / ranks)[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    result = np.empty(n, dtype=np.float64)
    result[order] = adj
    return result


def evalue(pvalue, effective_n):
    p = float(pvalue)
    if not (np.isfinite(p) and 0.0 <= p <= 1.0):
        raise ValueError("p-value must be finite and lie in [0, 1].")
    if isinstance(effective_n, bool) or not isinstance(effective_n, (int, float, np.integer, np.floating)):
        raise TypeError("effective_n must be a real number.")
    if not (np.isfinite(effective_n) and effective_n >= 0):
        raise ValueError("effective_n must be finite and non-negative.")
    return p * effective_n


def empirical_upper_tail_pvalue(scores, score):
    scores = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("null scores must be finite.")
    if not np.isfinite(score):
        raise ValueError("score must be finite.")
    sorted_scores = np.sort(scores)
    return _empirical_upper_tail_pvalue_sorted(sorted_scores, score)


def _empirical_upper_tail_pvalue_sorted(sorted_scores, score, *, validated=False):
    sorted_scores = np.asarray(sorted_scores, dtype=np.float64)
    if not validated and not np.all(np.isfinite(sorted_scores)):
        raise ValueError("null scores must be finite.")
    if not np.isfinite(score):
        raise ValueError("score must be finite.")
    n = sorted_scores.size
    if n <= 0:
        raise ValueError("null distribution must contain at least one score.")
    first_ge = np.searchsorted(sorted_scores, float(score), side="left")
    n_ge = n - first_ge
    return (n_ge + 1) / (n + 1)


def _null_id(dist):
    c = dist.contract
    parts = [
        f"format_version={NULL_FORMAT_VERSION}",
        f"strategy={dist.strategy}",
        f"metric={dist.metric}",
        f"n_null={dist.n_null}",
        f"n_models={dist.n_models}",
        f"shuffle={'true' if dist.shuffle else 'false'}",
        f"seed={dist.seed}",
        f"sampling={dist.sampling_version}",
        f"raw={c['raw_scores_fingerprint']}",
        f"seq={c['sequence_fingerprint']}",
        f"bg={c['background_fingerprint']}",
        "contract="
        + ":".join(
            str(c[k])
            for k in ("search_range", "min_logerr", "normalization_version", "alignment_version")
        ),
    ]
    if dist.model_collection_fingerprint is not None:
        parts.append(f"mcf={dist.model_collection_fingerprint}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def check_null_compatibility(
    dist, *, sequences, background, search_range, min_logerr
):
    """Raise MimosaError unless ``dist`` applies to the described comparison.

    ``sequences`` and ``background`` are the comparison inputs (None if
    absent). The null is not tied to a model type: profiles are compared on
    the same normalized logERR scale whatever model produced them.
    """
    expected_sequences = "none" if sequences is None else sequence_fingerprint(sequences)
    expected_background = "none" if background is None else sequence_fingerprint(background)
    checks = [
        (dist.strategy, "profile", "null distribution strategy is incompatible with profile comparison."),
        (dist.contract["search_range"], search_range, "null distribution search range is incompatible with this comparison."),
        (dist.contract["min_logerr"], np.float32(min_logerr), "null distribution minimum log-ERR is incompatible with this comparison."),
        (dist.sequence_fingerprint, expected_sequences, "null distribution sequence fingerprint is incompatible with this comparison."),
        (dist.background_fingerprint, expected_background, "null distribution background fingerprint is incompatible with this comparison."),
    ]
    for actual, expected, msg in checks:
        if actual != expected:
            raise MimosaError(msg)
    if dist.contract["alignment_version"] != ALIGNMENT_VERSION:
        raise MimosaError("null distribution alignment version is incompatible with this comparison.")


def _check_result_parameters(result, dist):
    """Reject results computed with parameters the null was not built for."""
    contract = dist.contract
    if result.search_range is not None and result.search_range != contract["search_range"]:
        raise ValueError(
            "result search_range does not match the null distribution contract."
        )
    if result.min_logerr is not None and np.float32(result.min_logerr) != contract["min_logerr"]:
        raise ValueError(
            "result min_logerr does not match the null distribution contract."
        )


def annotate_results(results, dist, *, effective_number_of_targets=None):
    n_null = dist.n_null
    effective = len(results) if effective_number_of_targets is None else effective_number_of_targets
    if effective < 0:
        raise ValueError("effective_number_of_targets must be non-negative.")
    sorted_scores = np.sort(np.asarray(dist.raw_scores, dtype=np.float64))
    if not np.all(np.isfinite(sorted_scores)):
        raise ValueError("null scores must be finite.")
    pvalues = np.empty(len(results), dtype=np.float64)
    for idx, result in enumerate(results):
        if result.metric != dist.metric:
            raise ValueError(
                f"result metric '{result.metric}' does not match null metric '{dist.metric}'."
            )
        _check_result_parameters(result, dist)
        pvalues[idx] = _empirical_upper_tail_pvalue_sorted(
            sorted_scores, result.score, validated=True
        )
    adj = adjusted_pvalues(pvalues)
    null_id = _null_id(dist)
    annotated = []
    for idx, r in enumerate(results):
        annotated.append(
            AnnotatedResult(
                r,
                p_value=float(pvalues[idx]),
                adj_p_value=float(adj[idx]),
                e_value=evalue(pvalues[idx], effective),
                null_id=null_id,
                null_n=n_null,
                null_estimator="empirical_upper_tail",
            )
        )
    return annotated


def _shuffle_null_model(model, seed):
    rng = np.random.default_rng(seed)
    column_order = rng.permutation(model.motif_length)
    representation = model.weights[:, column_order].copy()
    for column in range(representation.shape[1]):
        base_order = rng.permutation(4)
        representation[:4, column] = representation[base_order, column]
        representation[4, column] = representation[:4, column].min()
    return PWM(model.name, representation, model.background)


def _next_null_work_item(n_models, rng):
    query = int(rng.integers(0, 2 * n_models))
    target = int(rng.integers(0, 2 * n_models))
    while (query < n_models and target < n_models) or (query >= n_models and query == target):
        target = int(rng.integers(0, 2 * n_models))
    return query, target


def build_null(
    models,
    *,
    sequences,
    background=None,
    n_samples=2000,
    seed=127,
    search_range=5,
    min_logerr=DEFAULT_MIN_LOGERR,
    normalization=None,
    cache=None,
):
    config = ProfileConfig(search_range=search_range, min_logerr=min_logerr)
    search_range = config.search_range
    n_samples = strict_integer(n_samples, "n_samples")
    seed = strict_integer(seed, "seed")
    if len(models) < 2:
        raise ValueError("at least two models are required for null construction.")
    if not all(isinstance(m, PWM) for m in models):
        raise ValueError("null construction requires PWM models for mandatory shuffling.")
    names = [m.name for m in models]
    if len(set(names)) != len(names):
        raise ValueError("model names must be unique for null construction.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if normalization is None:
        normalization = HybridEmpiricalLogTail()

    rng = np.random.default_rng(seed)
    shuffled_models = [
        _shuffle_null_model(model, int(rng.integers(0, 2**63 - 1))) for model in models
    ]
    profile_models = list(models) + shuffled_models
    variant_ids = [
        f"original:{index}:{model.name}" for index, model in enumerate(models)
    ] + [
        f"shuffled:{index}:{model.name}" for index, model in enumerate(models)
    ]

    prepared = []
    preparation_context = None
    if cache is not None:
        from .cache import _make_preparation_context

        preparation_context = _make_preparation_context(sequences, background)
    for model in profile_models:
        prepared.append(
            _prepare_profile(
                model,
                sequences,
                background=background,
                min_logerr=min_logerr,
                normalization=normalization,
                cache=cache,
                circular=True,
                _preparation_context=preparation_context,
            )
        )

    work_items = [_next_null_work_item(len(models), rng) for _ in range(n_samples)]
    raw_scores = np.empty(n_samples, dtype=np.float64)
    pairs = []
    comparison_cache = {}
    for i, (query_idx, target_idx) in enumerate(work_items):
        pair_key = (query_idx, target_idx)
        result = comparison_cache.get(pair_key)
        if result is None:
            result = compare(
                prepared[query_idx],
                prepared[target_idx],
                search_range=search_range,
            )
            comparison_cache[pair_key] = result
        raw_scores[i] = float(result.score)
        pairs.append(
            (
                variant_ids[query_idx],
                variant_ids[target_idx],
                float(result.score),
            )
        )

    seq_fp = sequence_fingerprint(sequences)
    bg_fp = "none" if background is None else sequence_fingerprint(background)
    contract = {
        "metric": METRIC_NAME,
        "search_range": search_range,
        "min_logerr": config.min_logerr,
        "normalization_version": normalization_fingerprint(normalization),
        "alignment_version": ALIGNMENT_VERSION,
        "sequence_fingerprint": seq_fp,
        "background_fingerprint": bg_fp,
        "raw_scores_fingerprint": content_fingerprint_float64(raw_scores),
    }
    dist = NullDistribution(
        strategy="profile",
        metric=METRIC_NAME,
        raw_scores=raw_scores,
        pairs=pairs,
        n_null=n_samples,
        n_models=len(models),
        shuffle=True,
        seed=seed,
        sampling_version=SAMPLING_VERSION,
        model_collection_fingerprint=model_collection_fingerprint(models),
        sequence_fingerprint=seq_fp,
        background_fingerprint=bg_fp,
        contract=contract,
    )
    return dist
