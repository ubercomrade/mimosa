# Python API

The stable workflow API is exported from `mimosa`:

```python
from mimosa import (
    compare,
    compare_many,
    direct_mc,
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
read_fasta(path, max_sequences=1_000_000) -> tuple[EncodedSequences, tuple[str, ...]]
read_scores(path) -> ScoreProfile
write_model(path, model) -> None
```

`read_model` reads MEME, PFM, BaMM, Dimont, Slim, and model bundles using the
built-in format dispatch. SiteGA readers and construction are temporarily
disabled; existing files are not modified.

`write_model` serializes built-in model families only. Custom models can scan
and compare, but their arbitrary internal parameters are not a portable model
bundle.

`index` selects a zero-based motif from a MEME file. The scalar
`read_model(..., background=0.25)` is used for PFM-to-PWM conversion, not for
sequence calibration; only a uniform scalar background is supported by that
conversion. A model bundle retains its stored parameters.

## Scanning

```python
scan(model, sequences, *, strands="forward") -> RaggedArray | StrandPair
model.scan(sequences, *, strands, circular=False) -> RaggedArray | StrandPair
```

The public scan boundary validates the model contract and sequence encoding.
Built-in scan values are Float32. For `strands="both"`, the forward and
reverse tracks have the same ragged row layout. The module function is a thin
wrapper; the model's own `scan` method also supports the circular layout, used
by comparison, and the same four strand policies: `forward`, `reverse`,
`best`, and `both`. The module-level `scan` uses the linear layout.

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
Raw motif models require an `EncodedSequences` batch. Two raw `ScoreProfile`
inputs consume no sequences. Mixing a raw motif model with a raw `ScoreProfile`
is rejected; prepare both sides first with compatible circular row layouts.
The caller must ensure that corresponding positions represent the same
comparison data; equal row lengths alone cannot establish this.

Comparison is the weighted Jaccard similarity of circular profiles.
`min_logerr=None` uses the threshold of a prepared input, or `4.0` otherwise;
the threshold must be positive. Both profiles must have identical row lengths.
A `UserWarning` is issued when fewer than 50 positions are expected
above the threshold.

`prepare_profile(..., circular=False)` keeps the linear scan layout: rows hold
`n_positions` values and `site_start_offset` is the model's left context. With
`circular=True` every sequence is scanned as a ring, each row holds one value
per base indexed by site start, and `site_start_offset` is `0`. `compare`
and `compare_many` prepare motif models and score profiles
circularly and reject prepared profiles with `circular=False`. Score-profile
rows are used as given and are only marked circular. `PreparedProfile.circular`
reports the mode. `PreparedProfile` is immutable: to change `min_logerr`,
prepare the profile again.

```python
compare_many(
    query, targets, sequences=None, *,
    background=None, search_range=5, min_logerr=None,
    normalization=None, cache=None, total_threads=1, inner_threads=1,
) -> list[ComparisonResult]
```

`compare_many` accepts one query, keeps target order stable and reuses the
prepared query. Targets run sequentially in one process using JAX. The legacy thread
arguments are validated but do not control JAX or target parallelism:
`inner_threads` is limited to 1 through 4 and `total_threads` must be divisible
by it. Raw custom models are scanned serially.

`normalization=None` selects `EmpiricalLogTail`, the only supported strategy.
The high-level preparation, comparison and Monte Carlo paths enable the
Float64 operations they require locally, without changing the caller's global
JAX x64 setting. Public score tracks and result arrays are NumPy-backed;
custom scorers receive JAX arrays.

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
direct_mc(query, targets, sequences, *, k, seed, group="full",
          min_logerr=4.0, search_range=5, background=None,
          batch_size=8, storage="auto", memory_limit=268435456, temp_dir=None,
          comparison_block_size=8, algorithm="sparse", prefetch=0,
          anchor_memory_limit=268435456)
```

Available selectors are `BestPerSequence`, `ThresholdHits`, and
`TopFractionHits`. `SiteCollection.to_dict()` returns JSON-compatible lists.
Site selection uses raw scores from linear scans, not normalized comparison
profiles. `ThresholdHits` accepts any finite Float32 threshold, including
negative values; it does not use the positive `min_logerr` threshold.
`TopFractionHits(fraction, base=None)` keeps the highest-scoring fraction of
the base selector's hits, defaulting to `BestPerSequence`; the fraction must be
in `(0, 1]`.

`direct_mc` returns matrices of observed scores, exceedance counts and
p-values, each with shape `(1, number_of_targets)`. It randomizes the single
query and uses the same replicates for all fixed targets. It supports built-in
PWM, BaMM, Dimont and Slim models on the device-level pipeline and any other
`MotifModel` with `score_sites` through the shared scan path (SiteGA is
temporarily disabled). `group` is `full` or `gc`; ties are included. Score
profiles are not supported by this Monte Carlo path.
`k` must be a positive integer and `seed` a non-negative integer. Counts
include `random_score == observed_score`, with `p = (count + 1) / (k + 1)`.
Custom models also need a stable `fingerprint()` to define the random stream.
The public function has no `cache` argument; persistent caching of observed
Monte Carlo profiles is exposed by the CLI's `--cache-dir`.

Random query profiles are held on the device in batches. Target profiles are
stored on the host or in isolated temporary files: `storage` is `memory`,
`disk`, or `auto`, which switches to disk when estimated profile bytes exceed
`memory_limit`. `temp_dir` chooses the temporary directory's parent. Disk
capacity is checked approximately; quotas and concurrent use can still cause
write failures. Temporary files are cleaned on normal completion and handled
exceptions, but may remain after abrupt termination.
`comparison_block_size` controls vectorized comparisons within each batch;
larger values can increase GPU throughput and peak device memory.
The default `algorithm="sparse"` uses threshold-anchor comparisons, falls back
to dense comparison for dense anchors, and checks near-ties with the dense
kernel. Set `algorithm="dense"` to always use the dense kernel. Different
reduction orders may still cause numerical differences; this is not a
cross-device bitwise identity guarantee. Anchors use additional host and
device memory.
Retained target-anchor arrays are capped by `anchor_memory_limit` (default
256 MiB); anchors that do not fit are rebuilt for each query batch. Zero
disables retention. This is not a limit on total host or device memory.
`prefetch` enables bounded host read-ahead of disk-backed targets (0 disables
it; values up to 64). It can improve throughput on large disk-backed workloads
at the cost of approximately one full profile per queued target, plus the
profile currently being processed. It has no effect on in-memory storage.

The root API also exports:

```python
adjusted_pvalues(pvalues) -> numpy.ndarray
evalue(pvalue, effective_n) -> float
empirical_upper_tail_pvalue(scores, score) -> float
```

Use a one-dimensional vector with `adjusted_pvalues`, for example
`adjusted_pvalues(pvalues[0])`, to apply Benjamini–Hochberg adjustment across
the target family. `evalue` multiplies a p-value by a finite non-negative
effective number of targets. `direct_mc` does not apply these annotations
automatically; the CLI does. `AnnotatedResult` wraps a `ComparisonResult` for
serialization of optional statistical fields.

## Fixed-window functions and input relabeling

```python
from mimosa import WindowModel, InputRelabeledModel, alphabet_permutations, relabel_model

WindowModel(name, motif_length, score_sites, *, left_context=0,
            right_context=0, fingerprint=None, calibration_group=None, batch_size=8192)
alphabet_permutations(motif_length, seed=127) -> numpy.ndarray
InputRelabeledModel(model, permutations, *, batch_size=8192)
relabel_model(model, *, seed=127, permutations=None, batch_size=8192) -> MotifModel
```

The JAX-traceable callback maps oriented uint8 windows (A/C/G/T/N = 0/1/2/3/4)
to one finite real score per window. It must be deterministic and batch-independent.
`fingerprint` identifies the scorer and learned parameters for caching.
`InputRelabeledModel` always uses the generic input wrapper. `relabel_model`
uses an exactly equivalent native table permutation for built-in types and
the generic wrapper otherwise. Explicit permutations permit paired projections
and replay; context flanks and N are unchanged. See [custom models](extending.md).
`calibration_group` is optional metadata on `WindowModel`, included in its
fingerprint. Direct Monte Carlo does not require it or pool models into groups.

## Errors

Model-specific failures use `MimosaError` subclasses:

- `ModelFormatError` for malformed source files and bundles.
- `ModelDimensionError` for invalid model shapes or geometry.
- `ModelInterfaceError` for invalid custom model contracts.
- `InvariantError` for impossible or unsupported internal operations.

Invalid arguments may also raise ordinary `TypeError` or `ValueError`; file
operations can raise `OSError`. See [migration notes](migration.md) for removed
APIs.
