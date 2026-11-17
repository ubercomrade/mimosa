# Python API

The stable workflow API is exported from `mimosa`:

```python
from mimosa import (
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

`read_model` reads MEME, PFM, BaMM, Dimont, Slim, and model bundles using the
built-in format dispatch. SiteGA readers and construction are temporarily
disabled; existing files are not modified.

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
    min_logerr=4.0,
    normalization=None,
    cache=None,
    circular=False,
) -> PreparedProfile

compare(
    query,
    target,
    sequences=None,
    *,
    background=None,
    search_range=5,
    min_logerr=None,
    normalization=None,
    cache=None,
) -> ComparisonResult
```

`query` and `target` may be motif models, `ScoreProfile` values, or prepared
profiles. Prepared profiles used together must have compatible thresholds and
normalization strategies.

Comparison is the weighted Jaccard similarity of circular profiles.
`min_logerr=None` uses the threshold of a prepared input, or `4.0` otherwise;
the threshold must be positive. Both profiles must have identical row lengths.
A `UserWarning` is issued when fewer than 50 positions per strand are expected
above the threshold.

`prepare_profile(..., circular=False)` keeps the linear scan layout: rows hold
`n_positions` values and `site_start_offset` is the model's left context. With
`circular=True` every sequence is scanned as a ring, each row holds one value
per base indexed by site start, and `site_start_offset` is `0`. `compare`,
and `compare_many` prepare motif models and score profiles
circularly and reject prepared profiles with `circular=False`. Score-profile
rows are used as given and are only marked circular. `PreparedProfile.circular`
reports the mode. `PreparedProfile` is immutable: to change `min_logerr`,
prepare the profile again.

`compare_many(query, targets, sequences=None, total_threads=1,
inner_threads=1, **options)` keeps target order stable and reuses a prepared
query. Targets run sequentially in one process using JAX. The legacy thread
arguments are validated but do not control JAX or target parallelism:
`inner_threads` is limited to 1 through 4 and `total_threads` must be divisible
by it. Raw custom models are scanned serially.

`ComparisonResult` is immutable and exposes `query`, `target`, `score`,
`offset`, `orientation`, `metric`, and `n_sites`, plus the `search_range` and
`min_logerr` it was computed with (not part of `to_dict()`). `metric` is always
`"jaccard"`. `to_dict()` always includes `n_sites`, including when no position
contributes. `offset` is the circular target-site displacement relative to the
query; model context is excluded.
Call `to_dict()` at an API or CLI boundary when a JSON-compatible mapping is
needed.

## Sites and statistics

```python
select_sites(model, sequences, selector, *, strands="best") -> SiteCollection
reconstruct_pfm(model, sequences, selector, *, pseudocount=0.25, strands="best")
direct_mc(queries, targets, sequences, *, k, seed, group="full",
          min_logerr=4.0, search_range=5, background=None)
```

Available selectors are `BestPerSequence`, `ThresholdHits`, and
`TopFractionHits`. `SiteCollection.to_dict()` returns JSON-compatible lists.

`direct_mc` returns matrices of observed scores, exceedance counts and
p-values, with rows for queries and columns for targets. It supports built-in
PWM, BaMM, Dimont and Slim models, including mixed types and orders. `group`
is `full` or `gc`; ties are included. Custom models and score profiles are not
supported by this Monte Carlo path.

## Fixed-window functions and input relabeling

```python
from mimosa import WindowModel, InputRelabeledModel, alphabet_permutations, relabel_model

WindowModel(name, motif_length, score_windows, *, left_context=0,
            right_context=0, fingerprint=None, calibration_group=None, batch_size=8192)
alphabet_permutations(motif_length, seed=127) -> numpy.ndarray
InputRelabeledModel(model, permutations, *, batch_size=8192)
relabel_model(model, *, seed=127, permutations=None, batch_size=8192) -> MotifModel
```

The callback maps oriented uint8 windows (A/C/G/T/N = 0/1/2/3/4) to one finite
real score per window. It must be deterministic and batch-independent.
`fingerprint` identifies the scorer and learned parameters for caching.
`InputRelabeledModel` always uses the generic input wrapper. `relabel_model`
uses an exactly equivalent native table permutation for built-in types and
the generic wrapper otherwise. Explicit permutations permit paired projections
and replay; context flanks and N are unchanged. See [custom models](extending.md).

## Errors

Public input and model failures use `MimosaError` subclasses:

- `ModelFormatError` for malformed source files and bundles.
- `ModelDimensionError` for invalid model shapes or geometry.
- `ModelInterfaceError` for invalid custom model contracts.
- `InvariantError` for impossible or unsupported internal operations.
