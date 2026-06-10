"""Relation table parsers for null-distribution builds."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd


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
