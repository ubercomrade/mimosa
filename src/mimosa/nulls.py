"""Distribution-backed significance helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any, Iterable, TypedDict, cast

import joblib
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from mimosa.cache import fingerprint_batch, fingerprint_model
from mimosa.comparison import ComparatorConfig
from mimosa.models import GenericModel

logger = logging.getLogger(__name__)

NULL_FORMAT_VERSION = 1
NULL_CACHE_DIR = Path.home() / ".cache" / "mimosa" / "nulls"
MIN_KDE_SAMPLES = 3
_SCORE_CONFIG_KEYS = (
    "metric",
    "search_range",
    "window_radius",
    "realign_window",
    "min_logfpr",
    "pfm_mode",
    "pfm_top_fraction",
    "profile_normalization",
)


class NullArtifactMetadata(TypedDict):
    format_version: int
    created_at: str
    strategy: str
    metric: str
    config_signature: dict[str, Any]
    config_signature_hash: str
    sequence_fingerprint: str
    background_fingerprint: str
    model_collection_fingerprint: str | None
    relation_fingerprint: str | None
    package_version: str


class NullArtifactEntry(TypedDict, total=False):
    estimator_type: str
    sorted_scores: np.ndarray
    parameters: dict[str, Any]
    query_name: str
    query_fingerprint: str
    included_target_names: list[str]
    included_target_fingerprints: list[str]
    effective_number_of_targets: int
    raw_null_scores: np.ndarray
    n_null: int


class NullArtifact(TypedDict):
    metadata: NullArtifactMetadata
    entries: dict[str, NullArtifactEntry]


@dataclass
class NullBuildResult:
    """Summary returned after building one null-distribution artifact."""

    artifact: NullArtifact
    skipped: list[dict[str, Any]]
    total_comparisons: int


def stable_json_dumps(value: Any) -> str:
    """Serialize JSON-compatible data with stable key order."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible data."""
    return hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()


def file_fingerprint(path: str | Path) -> str:
    """Return a content hash for one input file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def comparator_signature(config: ComparatorConfig) -> dict[str, Any]:
    """Extract score-affecting comparator options."""
    return {key: config.get(key) for key in _SCORE_CONFIG_KEYS if key in config}


def environment_metadata(  # noqa: PLR0913
    *,
    strategy: str,
    config: ComparatorConfig,
    sequences=None,
    background=None,
    model_collection_fingerprint: str | None = None,
    relation_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Build the compatibility metadata stored in each null artifact."""
    try:
        version = package_metadata.version("mimosa-tool")
    except package_metadata.PackageNotFoundError:
        version = "0+unknown"

    config_signature = comparator_signature(config)
    return {
        "format_version": NULL_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "metric": config["metric"],
        "config_signature": config_signature,
        "config_signature_hash": stable_hash(config_signature),
        "sequence_fingerprint": fingerprint_batch(sequences) or "none",
        "background_fingerprint": fingerprint_batch(background) or "none",
        "model_collection_fingerprint": model_collection_fingerprint,
        "relation_fingerprint": relation_fingerprint,
        "package_version": version,
    }


class EmpiricalSurvivalEstimator:
    """Finite-sample-corrected empirical upper-tail survival estimator."""

    estimator_type = "ecdf"

    def __init__(self, scores: Iterable[float]):
        values = np.sort(np.asarray(list(scores), dtype=np.float64))
        if values.size == 0:
            raise ValueError("Null estimator requires at least one score.")
        self.scores = values

    @property
    def n(self) -> int:
        return int(self.scores.size)

    def pdf(self, score: float) -> float:
        return float(np.mean(np.isclose(self.scores, score)))

    def sf(self, score: float) -> float:
        count = self.n - int(np.searchsorted(self.scores, score, side="left"))
        return _clamp_pvalue((count + 1.0) / (self.n + 1.0), self.n)

    def to_entry(self) -> dict[str, Any]:
        return {
            "estimator_type": self.estimator_type,
            "sorted_scores": self.scores.astype(np.float64),
            "parameters": {},
        }


class KdeSurvivalEstimator(EmpiricalSurvivalEstimator):
    """Gaussian KDE upper-tail survival estimator with empirical p-value clamping."""

    estimator_type = "kde"

    def __init__(self, scores: Iterable[float]):
        super().__init__(scores)
        if self.n < MIN_KDE_SAMPLES or float(np.var(self.scores)) <= 0.0:
            raise ValueError("KDE requires at least three variable null scores.")
        self._kde = gaussian_kde(self.scores)

    def pdf(self, score: float) -> float:
        return float(self._kde.evaluate([score])[0])

    def sf(self, score: float) -> float:
        return _clamp_pvalue(float(self._kde.integrate_box_1d(score, np.inf)), self.n)

    def to_entry(self) -> dict[str, Any]:
        entry = super().to_entry()
        entry["estimator_type"] = self.estimator_type
        entry["parameters"] = {"bw_method": self._kde.factor}
        return entry


def fit_survival_estimator(scores: Iterable[float]) -> EmpiricalSurvivalEstimator:
    """Fit KDE when stable and otherwise fall back to an empirical estimator."""
    values = np.asarray(list(scores), dtype=np.float64)
    try:
        return KdeSurvivalEstimator(values)
    except Exception:
        return EmpiricalSurvivalEstimator(values)


def estimator_from_entry(entry: NullArtifactEntry) -> EmpiricalSurvivalEstimator:
    """Rehydrate an estimator from one artifact entry."""
    scores = np.asarray(entry["sorted_scores"], dtype=np.float64)
    if entry.get("estimator_type") == "kde":
        try:
            return KdeSurvivalEstimator(scores)
        except Exception:
            logger.warning("KDE entry for %s could not be rehydrated; using ECDF.", entry.get("query_name"))
    return EmpiricalSurvivalEstimator(scores)


def _clamp_pvalue(value: float, n_null: int) -> float:
    lower = 1.0 / (int(n_null) + 1.0)
    return float(min(1.0, max(lower, value)))


def parse_group_relations(
    path: str | Path,
    *,
    name_column: str = "motif",
    group_column: str = "group",
    ignore_missing: bool = False,
    known_names: set[str] | None = None,
) -> dict[str, set[str]]:
    """Parse a motif-to-group table and include pairs whose groups differ."""
    frame = _read_relation_table(path)
    if name_column not in frame or group_column not in frame:
        raise ValueError(f"Group table must contain {name_column!r} and {group_column!r} columns.")

    groups = {str(row[name_column]): str(row[group_column]) for _, row in frame.iterrows()}
    _validate_relation_names(set(groups), known_names, ignore_missing)
    names = sorted(groups)
    return {
        query: {target for target in names if target != query and groups[target] != groups[query]}
        for query in names
        if known_names is None or query in known_names
    }


def parse_pair_relations(  # noqa: PLR0913
    path: str | Path,
    *,
    query_column: str = "query",
    target_column: str = "target",
    include_column: str = "include",
    ignore_missing: bool = False,
    known_names: set[str] | None = None,
) -> dict[str, set[str]]:
    """Parse an explicit pair table where truthy cells include null pairs."""
    frame = _read_relation_table(path)
    required = {query_column, target_column, include_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Pair table is missing required columns: {', '.join(sorted(missing))}")

    relations: dict[str, set[str]] = {}
    seen_names: set[str] = set()
    for _, row in frame.iterrows():
        query = str(row[query_column])
        target = str(row[target_column])
        seen_names.update({query, target})
        if query == target or not _is_truthy(row[include_column]):
            continue
        relations.setdefault(query, set()).add(target)

    _validate_relation_names(seen_names, known_names, ignore_missing)
    return _filter_known_relations(relations, known_names)


def parse_pair_matrix_relations(
    path: str | Path,
    *,
    ignore_missing: bool = False,
    known_names: set[str] | None = None,
) -> dict[str, set[str]]:
    """Parse a square relation matrix where truthy cells include null pairs."""
    frame = _read_relation_table(path, index_col=0)
    seen_names = set(map(str, frame.index)).union(map(str, frame.columns))
    _validate_relation_names(seen_names, known_names, ignore_missing)

    relations: dict[str, set[str]] = {}
    for query, row in frame.iterrows():
        query_name = str(query)
        for target, value in row.items():
            target_name = str(target)
            if query_name != target_name and _is_truthy(value):
                relations.setdefault(query_name, set()).add(target_name)
    return _filter_known_relations(relations, known_names)


def _read_relation_table(path: str | Path, **kwargs) -> pd.DataFrame:
    separator = _sniff_delimiter(path)
    return pd.read_csv(path, sep=separator, **kwargs)


def _sniff_delimiter(path: str | Path) -> str:
    with open(path, newline="") as handle:
        sample = handle.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        return dialect.delimiter
    except csv.Error:
        return "\t" if "\t" in sample else ","


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y", "include", "included"}


def _validate_relation_names(names: set[str], known_names: set[str] | None, ignore_missing: bool) -> None:
    if known_names is None:
        return
    missing = names.difference(known_names)
    if missing and not ignore_missing:
        raise ValueError(f"Relation input references unknown motifs: {', '.join(sorted(missing))}")


def _filter_known_relations(relations: dict[str, set[str]], known_names: set[str] | None) -> dict[str, set[str]]:
    if known_names is None:
        return relations
    return {
        query: {target for target in targets if target in known_names and target != query}
        for query, targets in relations.items()
        if query in known_names
    }


def build_null_distributions(  # noqa: PLR0913
    models: list[GenericModel],
    relations: dict[str, set[str]],
    *,
    strategy: str,
    config: ComparatorConfig,
    sequences=None,
    background=None,
    min_null_targets: int = 1,
    strict: bool = False,
    relation_fingerprint: str | None = None,
) -> NullBuildResult:
    """Build one query-specific null distribution for each eligible model."""
    from mimosa.comparison import compare_one_to_many

    by_name = {model.name: model for model in models}
    collection_fp = stable_hash([fingerprint_model(model) for model in models])
    metadata_block = environment_metadata(
        strategy=strategy,
        config=config,
        sequences=sequences,
        background=background,
        model_collection_fingerprint=collection_fp,
        relation_fingerprint=relation_fingerprint,
    )
    entries: dict[str, NullArtifactEntry] = {}
    skipped: list[dict[str, Any]] = []
    total_comparisons = 0
    score_only_config = cast(ComparatorConfig, dict(config))
    score_only_config["pvalue"] = False

    for query in models:
        target_names = sorted(
            name for name in relations.get(query.name, set()) if name in by_name and name != query.name
        )
        if len(target_names) < min_null_targets:
            reason = f"only {len(target_names)} null target(s); required {min_null_targets}"
            skipped.append({"query": query.name, "reason": reason})
            message = f"Skipping null distribution for {query.name}: {reason}."
            if strict:
                raise ValueError(message)
            logger.warning(message)
            continue

        targets = [by_name[name] for name in target_names]
        results = compare_one_to_many(
            query,
            targets,
            strategy,
            score_only_config,
            sequences=sequences,
            background=background,
        )
        scores = np.asarray([float(result["score"]) for result in results], dtype=np.float64)
        estimator = fit_survival_estimator(scores)
        entry = cast(NullArtifactEntry, estimator.to_entry())
        entry.update(
            {
                "query_name": query.name,
                "query_fingerprint": fingerprint_model(query),
                "included_target_names": target_names,
                "included_target_fingerprints": [fingerprint_model(by_name[name]) for name in target_names],
                "effective_number_of_targets": len(target_names),
                "raw_null_scores": scores,
                "n_null": int(scores.size),
            }
        )
        entries[entry["query_fingerprint"]] = entry
        total_comparisons += len(results)

    artifact: NullArtifact = {"metadata": cast(NullArtifactMetadata, metadata_block), "entries": entries}
    return NullBuildResult(artifact=artifact, skipped=skipped, total_comparisons=total_comparisons)


def save_null_artifact(artifact: NullArtifact, path: str | Path) -> Path:
    """Persist one null-distribution artifact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    return output


def install_null_artifact(path: str | Path, cache_dir: str | Path = NULL_CACHE_DIR) -> Path:
    """Copy one artifact into the user null-distribution cache."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / Path(path).name
    shutil.copy2(path, destination)
    return destination


def load_null_artifact(source: str | Path | dict[str, Any]) -> NullArtifact:
    """Load a trusted null-distribution artifact from a path or return an in-memory artifact."""
    if isinstance(source, dict):
        return cast(NullArtifact, source)
    return cast(NullArtifact, joblib.load(source))


def load_compatible_null_artifact(
    *,
    strategy: str,
    config: ComparatorConfig,
    query_model: GenericModel,
    sequences=None,
    background=None,
) -> NullArtifact | None:
    """Load the explicit or first searched compatible null artifact for one query/config."""
    explicit = config.get("null_distribution")
    if explicit is not None:
        artifact = load_null_artifact(explicit)
        validate_artifact_compatible(
            artifact,
            strategy=strategy,
            config=config,
            query_model=query_model,
            sequences=sequences,
            background=background,
        )
        return artifact

    search_dirs = [Path(p) for p in (config.get("null_search_dirs") or [])]
    search_dirs.append(NULL_CACHE_DIR)
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.joblib")):
            artifact = load_null_artifact(path)
            if is_artifact_compatible(
                artifact,
                strategy=strategy,
                config=config,
                query_model=query_model,
                sequences=sequences,
                background=background,
            ):
                return artifact
    return None


def validate_artifact_compatible(  # noqa: PLR0913
    artifact: NullArtifact,
    *,
    strategy: str,
    config: ComparatorConfig,
    query_model: GenericModel,
    sequences=None,
    background=None,
) -> None:
    """Raise ValueError if an explicit artifact does not match the comparison context."""
    problems = _compatibility_problems(
        artifact,
        strategy=strategy,
        config=config,
        query_model=query_model,
        sequences=sequences,
        background=background,
    )
    if problems:
        raise ValueError("Null distribution is incompatible: " + "; ".join(problems))


def is_artifact_compatible(  # noqa: PLR0913
    artifact: NullArtifact,
    *,
    strategy: str,
    config: ComparatorConfig,
    query_model: GenericModel,
    sequences=None,
    background=None,
) -> bool:
    """Return True when one artifact matches the comparison context."""
    return not _compatibility_problems(
        artifact,
        strategy=strategy,
        config=config,
        query_model=query_model,
        sequences=sequences,
        background=background,
    )


def _compatibility_problems(  # noqa: PLR0913
    artifact: NullArtifact,
    *,
    strategy: str,
    config: ComparatorConfig,
    query_model: GenericModel,
    sequences=None,
    background=None,
) -> list[str]:
    metadata_block = artifact.get("metadata", {})
    expected = environment_metadata(strategy=strategy, config=config, sequences=sequences, background=background)
    problems: list[str] = []
    compatibility_keys = (
        "format_version",
        "strategy",
        "metric",
        "config_signature_hash",
        "sequence_fingerprint",
        "background_fingerprint",
    )
    for key in compatibility_keys:
        if metadata_block.get(key) != expected.get(key):
            problems.append(f"{key} differs")

    query_fp = fingerprint_model(query_model)
    if query_fp not in artifact.get("entries", {}):
        problems.append(f"query fingerprint is missing for {query_model.name!r}")
    return problems


def annotate_results_with_nulls(
    results: list[dict[str, Any]],
    *,
    artifact: NullArtifact,
    query_model: GenericModel,
    effective_number_of_targets: int | None = None,
) -> None:
    """Attach p-value, E-value, and BH-FDR q-value to comparison results in place."""
    entry = artifact["entries"][fingerprint_model(query_model)]
    estimator = estimator_from_entry(entry)
    n_null = int(entry.get("n_null", estimator.n))
    null_id = stable_hash(
        {
            "format_version": artifact.get("metadata", {}).get("format_version"),
            "query_fingerprint": entry.get("query_fingerprint"),
            "config_signature_hash": artifact.get("metadata", {}).get("config_signature_hash"),
        }
    )
    effective = effective_number_of_targets or len(results) or int(entry.get("effective_number_of_targets", 1))

    pvalues: list[float] = []
    valid_indices: list[int] = []
    for idx, result in enumerate(results):
        if "score" not in result:
            continue
        pvalue = estimator.sf(float(result["score"]))
        result.update(
            {
                "p-value": pvalue,
                "E-value": float(pvalue * effective),
                "null_id": null_id,
                "null_n": n_null,
                "null_estimator": entry.get("estimator_type", estimator.estimator_type),
            }
        )
        pvalues.append(pvalue)
        valid_indices.append(idx)

    for idx, qvalue in zip(valid_indices, bh_qvalues(pvalues), strict=False):
        results[idx]["q-value"] = qvalue


def bh_qvalues(pvalues: Iterable[float]) -> list[float]:
    """Compute monotone Benjamini-Hochberg q-values preserving input order."""
    values = np.asarray(list(pvalues), dtype=np.float64)
    if values.size == 0:
        return []
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.empty_like(ranked)
    running = 1.0
    m = ranked.size
    for reverse_idx in range(m - 1, -1, -1):
        rank = reverse_idx + 1
        running = min(running, float(ranked[reverse_idx] * m / rank))
        adjusted[reverse_idx] = running
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return [float(value) for value in result]
