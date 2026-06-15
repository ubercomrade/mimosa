"""Relation table parsers for null-distribution builds."""

from __future__ import annotations

import csv
from pathlib import Path

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
    names = sorted(name for name in groups if known_names is None or name in known_names)
    return {
        query: {target for target in names if target != query and groups[target] != groups[query]}
        for query in names
        if known_names is None or query in known_names
    }


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


def _validate_relation_names(names: set[str], known_names: set[str] | None, ignore_missing: bool) -> None:
    if known_names is None:
        return
    missing = names.difference(known_names)
    if missing and not ignore_missing:
        raise ValueError(f"Relation input references unknown motifs: {', '.join(sorted(missing))}")
