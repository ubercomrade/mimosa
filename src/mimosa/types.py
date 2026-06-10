"""Public immutable contracts shared across the package."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import numpy as np

from mimosa.batches import SequenceBatch


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return an immutable shallow copy of a mapping payload."""
    return MappingProxyType(dict(value or {}))


def _freeze_path_list(value: list[str | Path] | tuple[str | Path, ...] | None) -> tuple[str | Path, ...] | None:
    """Return path-like collections as immutable tuples."""
    if value is None:
        return None
    return tuple(value)


class _FrozenRecord(Mapping[str, Any]):
    """Mapping-like frozen dataclass adapter for existing public payloads."""

    _PUBLIC_KEY_OVERRIDES: dict[str, str] = {}
    _OMIT_NONE_FIELDS: frozenset[str] = frozenset()

    def _field_name_for_key(self, key: str) -> str:
        field_name = self._PUBLIC_KEY_OVERRIDES.get(key, key.replace("-", "_"))
        if not hasattr(self, field_name):
            raise KeyError(key)
        return field_name

    def __getitem__(self, key: str) -> Any:
        field_name = self._field_name_for_key(key)
        value = getattr(self, field_name)
        if field_name in self._OMIT_NONE_FIELDS and value is None:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            if field_info.name in self._OMIT_NONE_FIELDS and value is None:
                continue
            yield next((key for key, name in self._PUBLIC_KEY_OVERRIDES.items() if name == field_info.name), field_info.name)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        """Return the public dictionary payload for JSON and CLI output."""
        return {key: self[key] for key in self}


@dataclass(frozen=True, slots=True)
class ComparatorConfig(_FrozenRecord):
    """Immutable comparison options shared across scoring entry points."""

    metric: str = "pcc"
    seed: int | None = None
    n_jobs: int | None = None
    pfm_mode: bool = False
    pfm_top_fraction: float = 0.05
    search_range: int = 10
    min_logfpr: float | None = None
    window_radius: int = 10
    realign_window: int = 3
    profile_normalization: str = "empirical_log_tail"
    cache_mode: str = "off"
    cache_dir: str = ".mimosa-cache"
    background: SequenceBatch | None = None
    pvalue: bool = False
    null_distribution: str | Path | dict[str, Any] | None = None
    null_search_dirs: tuple[str | Path, ...] | None = None
    effective_number_of_targets: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "null_search_dirs", _freeze_path_list(self.null_search_dirs))


@dataclass(frozen=True, slots=True)
class ComparisonResult(_FrozenRecord):
    """Immutable comparison result with optional significance annotation."""

    _PUBLIC_KEY_OVERRIDES = {
        "p-value": "p_value",
        "E-value": "e_value",
        "q-value": "q_value",
    }
    _OMIT_NONE_FIELDS = frozenset({"n_sites", "p_value", "e_value", "q_value", "null_id", "null_n", "null_estimator"})

    query: str
    target: str
    score: float
    offset: int
    orientation: str
    metric: str
    n_sites: int | None = None
    p_value: float | None = None
    e_value: float | None = None
    q_value: float | None = None
    null_id: str | None = None
    null_n: int | None = None
    null_estimator: str | None = None


@dataclass(frozen=True, slots=True)
class OneToOneConfig(_FrozenRecord):
    """Immutable one-vs-one API config."""

    query: Any
    target: Any
    query_type: str | None
    target_type: str | None
    strategy: str
    sequences: Any
    background: Any
    num_sequences: int
    seq_length: int
    seed: int
    comparator: ComparatorConfig
    query_kwargs: Mapping[str, Any]
    target_kwargs: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_kwargs", _freeze_mapping(self.query_kwargs))
        object.__setattr__(self, "target_kwargs", _freeze_mapping(self.target_kwargs))


@dataclass(frozen=True, slots=True)
class OneToManyConfig(_FrozenRecord):
    """Immutable one-vs-many API config."""

    query: Any
    targets: tuple[Any, ...]
    query_type: str | None
    target_type: str | None
    strategy: str
    sequences: Any
    background: Any
    num_sequences: int
    seq_length: int
    seed: int
    comparator: ComparatorConfig
    query_kwargs: Mapping[str, Any]
    target_kwargs: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "query_kwargs", _freeze_mapping(self.query_kwargs))
        object.__setattr__(self, "target_kwargs", _freeze_mapping(self.target_kwargs))


def result_from_payload(payload: Mapping[str, Any]) -> ComparisonResult:
    """Build a typed result object from a mapping-like payload."""
    return ComparisonResult(
        query=str(payload["query"]),
        target=str(payload["target"]),
        score=float(payload["score"]),
        offset=int(payload["offset"]),
        orientation=str(payload["orientation"]),
        metric=str(payload["metric"]),
        n_sites=int(payload["n_sites"]) if payload.get("n_sites") is not None else None,
        p_value=float(payload["p-value"]) if payload.get("p-value") is not None else None,
        e_value=float(payload["E-value"]) if payload.get("E-value") is not None else None,
        q_value=float(payload["q-value"]) if payload.get("q-value") is not None else None,
        null_id=str(payload["null_id"]) if payload.get("null_id") is not None else None,
        null_n=int(payload["null_n"]) if payload.get("null_n") is not None else None,
        null_estimator=str(payload["null_estimator"]) if payload.get("null_estimator") is not None else None,
    )


def batch_to_rows(batch: SequenceBatch) -> list[np.ndarray]:
    """Materialize sequence rows from a padded batch payload."""
    values = np.asarray(batch["values"])
    lengths = np.asarray(batch["lengths"], dtype=np.int64)
    return [values[index, : int(lengths[index])].copy() for index in range(values.shape[0])]
