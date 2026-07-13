# Documentation Map

The active package is `Mimosa.jl`. User and API documentation is built with
Documenter from [`Mimosa.jl/docs/src/`](../Mimosa.jl/docs/src/).

## Current Documentation

- [Quick start](../Mimosa.jl/docs/src/quickstart.md)
- [Public API](../Mimosa.jl/docs/src/api.md)
- [CLI](../Mimosa.jl/docs/src/cli.md)
- [Supported models](../Mimosa.jl/docs/src/models.md)
- [Architecture](../Mimosa.jl/docs/src/architecture.md)
- [Data layout](../Mimosa.jl/docs/src/data_layout.md)
- [Numerical compatibility](../Mimosa.jl/docs/src/numerical_compatibility.md)
- [Reproducibility](../Mimosa.jl/docs/src/reproducibility.md)
- [Storage](../Mimosa.jl/docs/src/storage.md)
- [Security](../Mimosa.jl/docs/src/security.md)

The root [feature matrix](feature_matrix.md) summarizes the current public
contract. Root [numerical compatibility](numerical_compatibility.md) records the
scientific invariants that apply across code, fixtures, and storage.

## Decision Records

Files under [`adr/`](adr/) record architectural decisions. Their status sections
identify whether a decision is active, amended, or superseded. Current code and
tests take precedence over historical examples in an ADR.

## Historical Records

- [Python reference architecture](python_reference_architecture.md)
- [Python format inventory](formats/python_formats.md)
- [Stage 1 report](stage1_report.md)

These files describe the retired Python implementation or completed migration
stages. They are retained for provenance and must not be used to infer current
commands, metrics, null strategies, or package layout.
