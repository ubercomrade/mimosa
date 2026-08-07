"""Null distributions, p-values, BH FDR, and result annotation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .compare import compare
from .io.bundles import (
    content_fingerprint_float64,
    model_collection_fingerprint,
    sequence_fingerprint,
)
from .models import PWM
from .profiles.alignment import parse_profile_metric
from .profiles.normalization import HybridEmpiricalLogTail, normalization_fingerprint
from .profiles.prepared import prepare_profile

SAMPLING_VERSION = "original-shuffled-ordered-pairs-v3"
ALIGNMENT_VERSION = "profile-alignment-v1"


@dataclass(frozen=True)
class NullDistribution:
    strategy: str
    metric: str
    raw_scores: np.ndarray
    pairs: list
    n_null: int
    n_models: int
    model_type: str
    shuffle: bool
    seed: int
    sampling_version: str
    model_collection_fingerprint: str | None
    sequence_fingerprint: str
    background_fingerprint: str
    contract: dict

    def __post_init__(self):
        if self.n_null <= 0:
            raise ValueError("null distribution n_null must be positive.")
        if self.n_null != len(self.raw_scores):
            raise ValueError("null distribution n_null does not match raw_scores length.")
        if self.n_models < 2:
            raise ValueError("null distribution requires at least two source models.")
        if len(self.pairs) != self.n_null:
            raise ValueError("null distribution pairs do not match n_null.")
        if not self.model_type:
            raise ValueError("null distribution model_type must not be empty.")
        if self.seed < 0:
            raise ValueError("null distribution seed must be non-negative.")
        if not self.sampling_version:
            raise ValueError("null distribution sampling_version must not be empty.")

    def to_dict(self):
        return {
            "strategy": self.strategy,
            "metric": self.metric,
            "raw_scores": self.raw_scores,
            "pairs": self.pairs,
            "n_null": self.n_null,
            "n_models": self.n_models,
            "model_type": self.model_type,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "sampling_version": self.sampling_version,
            "model_collection_fingerprint": self.model_collection_fingerprint,
            "sequence_fingerprint": self.sequence_fingerprint,
            "background_fingerprint": self.background_fingerprint,
            "contract": self.contract,
        }


@dataclass(frozen=True)
class AnnotatedResult:
    query: str
    target: str
    score: np.float32
    offset: int
    orientation: str
    metric: str
    n_sites: int
    p_value: float | None = None
    adj_p_value: float | None = None
    e_value: float | None = None
    null_id: str | None = None
    null_n: int | None = None
    null_estimator: str | None = None

    def to_dict(self):
        d = {
            "annotation_schema_version": 1,
            "query": self.query,
            "target": self.target,
            "score": float(self.score),
            "offset": self.offset,
            "orientation": self.orientation,
            "metric": self.metric,
        }
        if self.n_sites > 0:
            d["n_sites"] = int(self.n_sites)
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
    adj = np.empty(n, dtype=np.float64)
    adj[n - 1] = min(sorted_p[n - 1], 1.0)
    for i in range(n - 2, -1, -1):
        val = sorted_p[i] * n / (i + 1)
        adj[i] = min(adj[i + 1], val)
    adj = np.minimum(adj, 1.0)
    result = np.empty(n, dtype=np.float64)
    result[order] = adj
    return result


def evalue(pvalue, effective_n):
    p = float(pvalue)
    if not (np.isfinite(p) and 0.0 <= p <= 1.0):
        raise ValueError("p-value must be finite and lie in [0, 1].")
    if effective_n < 0:
        raise ValueError("effective_n must be non-negative.")
    return p * effective_n


def empirical_upper_tail_pvalue(scores, score):
    scores = np.asarray(scores, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("null scores must be finite.")
    if not np.isfinite(score):
        raise ValueError("score must be finite.")
    sorted_scores = np.sort(scores)
    n = sorted_scores.size
    if n <= 0:
        raise ValueError("null distribution must contain at least one score.")
    first_ge = np.searchsorted(sorted_scores, float(score), side="left")
    n_ge = n - first_ge
    return (n_ge + 1) / (n + 1)


def _null_id(dist):
    c = dist.contract
    parts = [
        "format_version=7",
        f"strategy={dist.strategy}",
        f"metric={dist.metric}",
        f"n_null={dist.n_null}",
        f"n_models={dist.n_models}",
        f"model_type={dist.model_type}",
        f"shuffle={'true' if dist.shuffle else 'false'}",
        f"seed={dist.seed}",
        f"sampling={dist.sampling_version}",
        f"raw={c['raw_scores_fingerprint']}",
        f"seq={c['sequence_fingerprint']}",
        f"bg={c['background_fingerprint']}",
        "contract="
        + ":".join(
            str(c[k])
            for k in ("search_range", "window_radius", "realign_window", "min_logerr", "normalization_version", "alignment_version")
        ),
    ]
    if dist.model_collection_fingerprint is not None:
        parts.append(f"mcf={dist.model_collection_fingerprint}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def annotate_results(results, dist, *, effective_number_of_targets=None):
    n_null = dist.n_null
    effective = len(results) if effective_number_of_targets is None else effective_number_of_targets
    if effective < 0:
        raise ValueError("effective_number_of_targets must be non-negative.")
    sorted_scores = np.sort(np.asarray(dist.raw_scores, dtype=np.float64))
    pvalues = np.empty(len(results), dtype=np.float64)
    for idx, result in enumerate(results):
        if result.metric != dist.metric:
            raise ValueError(
                f"result metric '{result.metric}' does not match null metric '{dist.metric}'."
            )
        pvalues[idx] = empirical_upper_tail_pvalue(sorted_scores, result.score)
    adj = adjusted_pvalues(pvalues)
    null_id = _null_id(dist)
    annotated = []
    for idx, r in enumerate(results):
        annotated.append(
            AnnotatedResult(
                r.query,
                r.target,
                r.score,
                r.offset,
                r.orientation,
                r.metric,
                r.n_sites,
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
    metric="co",
    n_samples=2000,
    seed=127,
    search_range=10,
    window_radius=10,
    realign_window=3,
    min_logerr=0.0,
    normalization=None,
    cache=None,
    on_progress=None,
):
    if len(models) < 2:
        raise ValueError("at least two models are required for null construction.")
    if not all(isinstance(m, PWM) for m in models):
        raise ValueError("null construction requires PWM models for mandatory shuffling.")
    names = [m.name for m in models]
    if len(set(names)) != len(names):
        raise ValueError("model names must be unique for null construction.")
    if search_range < 0 or window_radius < 0 or realign_window < 0:
        raise ValueError("search/window/realign ranges must be non-negative.")
    if not np.isfinite(min_logerr):
        raise ValueError("min_logerr must be finite.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    if normalization is None:
        normalization = HybridEmpiricalLogTail()
    metric_str = parse_profile_metric(metric)

    rng = np.random.default_rng(seed)
    shuffled_models = [
        _shuffle_null_model(model, int(rng.integers(0, 2**63 - 1))) for model in models
    ]
    profile_models = list(models) + shuffled_models

    if on_progress is not None:
        on_progress(("prepare", 0, len(profile_models), ""))
    prepared = []
    for i, model in enumerate(profile_models):
        prepared.append(
            prepare_profile(
                model,
                sequences,
                background=background,
                min_logerr=min_logerr,
                normalization=normalization,
                cache=cache,
            )
        )
        if on_progress is not None:
            on_progress(("prepare", i + 1, len(profile_models), model.name))

    work_items = [_next_null_work_item(len(models), rng) for _ in range(n_samples)]
    raw_scores = np.empty(n_samples, dtype=np.float64)
    pairs = []
    if on_progress is not None:
        on_progress(("null", 0, n_samples, ""))
    for i, (query_idx, target_idx) in enumerate(work_items):
        result = compare(
            prepared[query_idx],
            prepared[target_idx],
            metric=metric_str,
            search_range=search_range,
            window_radius=window_radius,
            realign_window=realign_window,
        )
        raw_scores[i] = float(result.score)
        pairs.append((profile_models[query_idx].name, profile_models[target_idx].name, float(result.score)))
        if on_progress is not None:
            on_progress(("null", i + 1, n_samples, profile_models[target_idx].name))

    seq_fp = sequence_fingerprint(sequences)
    bg_fp = "none" if background is None else sequence_fingerprint(background)
    contract = {
        "metric": metric_str,
        "search_range": search_range,
        "window_radius": window_radius,
        "realign_window": realign_window,
        "min_logerr": np.float32(min_logerr),
        "normalization_version": normalization_fingerprint(normalization),
        "alignment_version": ALIGNMENT_VERSION,
        "sequence_fingerprint": seq_fp,
        "background_fingerprint": bg_fp,
        "raw_scores_fingerprint": content_fingerprint_float64(raw_scores),
    }
    dist = NullDistribution(
        strategy="profile",
        metric=metric_str,
        raw_scores=raw_scores,
        pairs=pairs,
        n_null=n_samples,
        n_models=len(models),
        model_type="pwm",
        shuffle=True,
        seed=seed,
        sampling_version=SAMPLING_VERSION,
        model_collection_fingerprint=model_collection_fingerprint(models),
        sequence_fingerprint=seq_fp,
        background_fingerprint=bg_fp,
        contract=contract,
    )
    return dist
