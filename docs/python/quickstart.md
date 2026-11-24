# Python Quickstart

## Install

Mimosa supports Python 3.12 and 3.13. Run these commands from the repository root:

```bash
uv sync --locked --group dev
```

Run Python examples with `uv run python`, or activate `.venv` first. The
`examples/` paths in this guide refer to the repository's fixtures.
The current checkout can also be installed into an existing environment:

```bash
python -m pip install .
```

JAX is required, including for CPU execution; the project pins `jax==0.10.0`.
The default installation does not install GPU plugins. For NVIDIA CUDA, choose
one of the `cuda12` or `cuda13` extras. For example, use
`uv sync --locked --group dev --extra cuda13`, then include `--extra cuda13`
in each `uv run` command. For pip, use `python -m pip install 'mimosa-tool[cuda13]'`.
The GPU and NVIDIA driver must support the chosen CUDA version. For AMD, use
`python -m pip install 'mimosa-tool[rocm]'`, or run:

```bash
uv sync --locked --group dev --extra rocm
uv run --extra rocm mimosa --version
```

The ROCm extra installs
the plugin and PJRT; it requires a separately installed compatible ROCm 7 and
driver. See the
[installation section of the README](../../README.md#installation) for both
GPU options and links to compatibility instructions. Check device availability
in the environment used for computation:

```bash
uv run --extra rocm python -c "import jax; print(jax.default_backend(), jax.devices())"
```

Use `--extra cuda13` or `--extra cuda12` for the corresponding NVIDIA
environment, or omit `--extra` for the default CPU environment.

See [Migrating to JAX](migration.md) if updating an older installation.

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
`scan` uses the linear layout, omitting windows that extend beyond sequence
ends. Comparison instead uses circular profiles.

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
    search_range=5,
    min_logerr=2.0,
)
print(result.to_dict())
```

The score is the weighted Jaccard similarity of the two circular `-log10(ERR)`
profiles, reported with `metric == "jaccard"`. `min_logerr` defaults to `4.0`,
which suits about `10^6` positions; the bundled FASTA is smaller, so
these examples lower it. A separate FASTA batch can be passed as `background=`
when score normalization needs a separate calibration set.

## Reuse Prepared Profiles

Prepare one query once when comparing it with many targets:

```python
from mimosa import compare_many, prepare_profile

prepared_query = prepare_profile(query, sequences, min_logerr=2.0, circular=True)
targets = [target, read_model("examples/gata4.meme")]
results = compare_many(prepared_query, targets, sequences)
```

Comparison requires `circular=True`; the default linear layout is rejected.
To reuse both sides, prepare each target with `circular=True` as well and call
`compare_many(prepared_query, prepared_targets)` without sequences.

Targets are processed in order in a single process. JAX manages device
execution; the legacy thread-budget arguments do not control JAX.

For precomputed score tracks, use `read_scores`:

```python
from mimosa import compare, prepare_profile, read_scores

query_scores = read_scores("examples/scores_1.fasta")
target_scores = read_scores("examples/scores_2.fasta")
prepared_query = prepare_profile(query_scores, min_logerr=2.0, circular=True)
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

## Pair-specific Monte Carlo

Compare each target against independently randomized replicates of the single query:

```python
from mimosa import direct_mc

scores, counts, pvalues = direct_mc(
    query, [target], sequences, min_logerr=2.0,
    k=99, seed=127, group="full",
)
print(pvalues[0, 0])
```

All three arrays have shape `(1, number_of_targets)`. Targets remain fixed;
the same query replicates are reused for every target, without depending on
target order or batch size. Ties count as exceedances:
`p = (count + 1) / (k + 1)`. Here `k=99` is a small demonstration, with a
minimum possible p-value of `0.01`; use more replicates for finer resolution.
The function returns raw p-values, not BH-adjusted values or E-values.
