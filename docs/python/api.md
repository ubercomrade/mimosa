# Python API

The stable workflow API is exported from `mimosa`:

```python
from mimosa import (
    annotate_results,
    build_null,
    compare,
    compare_many,
    prepare_profile,
    read_fasta,
    read_model,
    read_scores,
    reconstruct_pfm,
    scan,
    select_sites,
    write_model,
)
```

## Input and model I/O

```python
read_model(path, *, format="auto", index=0, background=0.25, order=None)
read_fasta(path) -> tuple[EncodedSequences, tuple[str, ...]]
read_scores(path) -> ScoreProfile
write_model(path, model) -> None
```

`read_model` reads MEME, PFM, BaMM, SiteGA, Dimont, Slim, and model bundles
using the built-in format dispatch.

`write_model` serializes built-in model families only. Custom models can scan
and compare, but their arbitrary internal parameters are not a portable model
bundle.

## Scanning

```python
scan(model, sequences, *, strands="forward") -> RaggedArray | StrandPair
```

The public scan boundary validates the model contract and sequence encoding.
Built-in scan values are Float32. For `strands="both"`, the forward and
reverse tracks have the same ragged row layout.

## Profiles and comparison

```python
prepare_profile(
    model_or_scores,
    sequences=None,
    *,
    background=None,
    min_logerr=0.0,
    normalization=None,
    cache=None,
) -> PreparedProfile

compare(
    query,
    target,
    sequences=None,
    *,
    background=None,
    metric="co",
    search_range=10,
    window_radius=10,
    realign_window=3,
    min_logerr=None,
    normalization=None,
    cache=None,
) -> ComparisonResult
```

`query` and `target` may be motif models, `ScoreProfile` values, or prepared
profiles. Prepared profiles used together must have compatible thresholds and
normalization strategies.

`compare_many(query, targets, sequences=None, total_threads=1,
inner_threads=1, **options)` keeps target order stable and reuses a prepared
query. It derives `joblib_workers = min(total_threads / inner_threads, number of targets)`; the
complete target preparation and alignment pipeline runs in each joblib worker
when that value is greater than one. `inner_threads` is limited to 1 through 4
and `total_threads` must be divisible by it. Raw custom models require serial
comparison or preparation before calling the parallel path.

`ComparisonResult` is immutable and exposes `query`, `target`, `score`,
`offset`, `orientation`, `metric`, and `n_sites`. `to_dict()` always includes
`n_sites`, including when no finite site contributes. `offset` is the physical
target-site displacement relative to the query; model context is excluded.
Call `to_dict()` at an API or CLI boundary when a JSON-compatible mapping is
needed.

## Sites and statistics

```python
select_sites(model, sequences, selector, *, strands="best") -> SiteCollection
reconstruct_pfm(model, sequences, selector, *, pseudocount=0.25, strands="best")
annotate_results(results, distribution, *, effective_number_of_targets=None)
```

Available selectors are `BestPerSequence`, `ThresholdHits`, and
`TopFractionHits`. `SiteCollection.to_dict()` returns JSON-compatible lists.

`NullDistribution` uses empirical upper-tail p-values. It does not contain a
parametric fit object.

## Errors

Public input and model failures use `MimosaError` subclasses:

- `ModelFormatError` for malformed source files and bundles.
- `ModelDimensionError` for invalid model shapes or geometry.
- `ModelInterfaceError` for invalid custom model contracts.
- `InvariantError` for impossible or unsupported internal operations.
