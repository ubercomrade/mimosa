# Python Quickstart

## Install

Mimosa requires Python 3.12 or newer.

```bash
uv sync
```

The package can also be installed into an existing environment:

```bash
python -m pip install .
```

## Read and Scan

Read a built-in model and a FASTA batch through the root API:

```python
from mimosa import read_fasta, read_model, scan

model = read_model("examples/foxa2.meme")
sequences, names = read_fasta("examples/foreground.fa")

scores = scan(model, sequences, strands="best")
```

`strands` can be `"forward"`, `"reverse"`, `"best"`, or `"both"`.
The first three policies return a `RaggedArray`; `"both"` returns a
`StrandPair` containing forward and reverse tracks.

## Compare Models

Comparison uses the same sequence batch for both models:

```python
from mimosa import compare, read_fasta, read_model

query = read_model("examples/foxa2.meme")
target = read_model("examples/gata2.meme")
sequences, _ = read_fasta("examples/foreground.fa")

result = compare(
    query,
    target,
    sequences,
    metric="co",
    search_range=10,
    window_radius=10,
    realign_window=3,
)
print(result.to_dict())
```

Available metrics are `co`, `co_rowwise`, `dice`, `dice_rowwise`, and
`cosine`. A separate FASTA batch can be passed as `background=` when score
normalization needs a separate calibration set.

## Reuse Prepared Profiles

Prepare one query once when comparing it with many targets:

```python
from mimosa import compare_many, prepare_profile

prepared_query = prepare_profile(query, sequences)
targets = [target, read_model("examples/gata4.meme")]
results = compare_many(prepared_query, targets, sequences, metric="cosine")
```

For precomputed score tracks, use `read_scores`:

```python
from mimosa import compare, prepare_profile, read_scores

query_scores = read_scores("examples/scores_1.fasta")
target_scores = read_scores("examples/scores_2.fasta")
prepared_query = prepare_profile(query_scores)
result = compare(prepared_query, target_scores)
```

## Sites and PFM Reconstruction

Selectors operate on scanned model scores:

```python
from mimosa import BestPerSequence, reconstruct_pfm, select_sites

sites = select_sites(query, sequences, BestPerSequence(), strands="best")
pfm = reconstruct_pfm(query, sequences, BestPerSequence(), strands="best")
print(len(sites), pfm.shape)
```

`SiteCollection` stores sequence indices, scan starts, strand flags, and
Float32 scores. `reconstruct_pfm` excludes ambiguous `N` bases and
reverse-complements reverse-strand sites before counting.

## Null Distributions

`build_null` currently shuffles PWM models and samples ordered profile pairs:

```python
from mimosa import build_null

models = [query, target, read_model("examples/gata4.meme")]
distribution = build_null(
    models,
    sequences=sequences,
    metric="co",
    n_samples=2000,
    seed=127,
)
```

Annotate comparison results with empirical p-values, Benjamini-Hochberg
adjustment, and E-values:

```python
from mimosa import annotate_results

annotated = annotate_results([result], distribution)
print(annotated[0].to_dict())
```

Use `mimosa.io.write_null_bundle` and `mimosa.io.read_null_bundle` for the
portable null-bundle format. See [Storage](storage.md) for details.
