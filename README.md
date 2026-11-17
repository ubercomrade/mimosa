# Mimosa

Mimosa is a Python 3.12–3.13 package and command-line tool for DNA motif comparison.
It supports PWM/PFM, BaMM, Dimont, Slim, and precomputed score profiles.
SiteGA is temporarily disabled pending further development.
Mimosa compares heterogeneous motif models through their behavior on the same
DNA sequences rather than by forcing their internal representations into a
common matrix.

## Background

Transcription factors (TFs) are key regulators of gene expression. They bind
specific DNA sequences, termed transcription factor binding sites (TFBSs),
located in gene regulatory regions. Because TFBSs are highly variable, they are
typically described using motifs, which capture the nucleotide preferences of a
TF across a range of binding sites. The position weight matrix (PWM) is the de
facto standard motif model, assuming independence between positions and
additive nucleotide contributions. However, numerous studies have demonstrated
the presence of dependencies between positions within TFBSs. To account for
such dependencies, alternative models have been proposed, including
Markov-based approaches (BaMM, InMoDe, Dimont, Slim, MODER2), discriminative
methods (SiteGA), and deep learning models (DeepBind, DeepGRN, BERT-TFBS).
However, a mature ecosystem of databases and tools has been developed primarily
for PWMs, including motif comparison methods that are critical for result
interpretation. Comparison of alternative motif models usually requires their
conversion into PWMs, which inevitably leads to loss of information about
positional dependencies. At present, no universal tool exists for their direct
comparison. Mimosa (Model-Independent Motif Similarity Assessment) addresses
this gap by evaluating motif similarity based on functional behavior, namely
the scores models assign to DNA sequences rather than their internal parameters.

Mimosa uses the following profile comparison pipeline:

1. Scan sequences by motifs to obtain score profiles.
2. Convert raw scores to exact empirical `-log10(ERR)` values, where ERR is the
   expectation recognition rate, optionally using a separate background sequence
   set for calibration. These values do not depend on `min_logerr`.
3. Keep the positions where either model reaches `-log10(ERR) >= min_logerr`
   (default `4.0`).
4. Score the weighted Jaccard similarity, `sum(min) / sum(max)` over those
   positions, for circular shifts and the four strand orientations.
5. Return the highest-scoring alignment, its offset, orientation, and number of
   contributing sites.

Sequences are treated as circular, so every model yields one value per base
and strand in a shared coordinate system. The default threshold is tuned for
about `10^6` positions per strand, which is what the CLI generates when no
FASTA is given (10 000 uniform random sequences of 100 bp). Mimosa warns when
a batch is too small for the threshold; lower `min_logerr` in that case, as the
examples below do for the small bundled FASTA.

See [Method and Statistics](docs/python/method.md) for the metric definition,
statistical significance, and references.

## Supported Models

| Model | Source formats | CLI type |
|---|---|---|
| PWM/PFM | MEME, plain PFM, model bundle | `pwm` |
| BaMM | `.ihbcp`, model bundle | `bamm` |
| Dimont | Jstacs XML, model bundle | `dimont` |
| Slim | Jstacs XML, model bundle | `slim` |
| Score profile | FASTA-like numeric rows | `scores` |

The SiteGA implementation and formats are retained for future work, but model
construction, reading, and CLI use are currently unavailable.

## Installation

Mimosa supports Python 3.12 and 3.13. It is published on
[PyPI](https://pypi.org/project/mimosa-tool/) as `mimosa-tool`.

```bash
python -m pip install mimosa-tool
```

For development from a source checkout:

```bash
uv sync --locked --group dev
```

Built-in scanning and prepared-profile normalization use JAX. The default
installation supports CPU execution; GPU execution requires a compatible JAX
CUDA or ROCm plugin and driver as described in the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
The migration is ongoing: some lower-level operations still use
Numba. The pooled-null interface has been removed; p-values use direct
pair-specific Monte Carlo.

## CLI

Compare two PWM models on generated sequences with the default settings:

```bash
mimosa compare examples/foxa2.meme examples/gata2.meme \
  --query-type pwm --target-type pwm
```

Compare them on a FASTA batch:

```bash
mimosa compare examples/foxa2.meme examples/gata2.meme \
  --query-type pwm --target-type pwm \
  --fasta examples/foreground.fa --min-logerr 2
```

For a direct pair-specific Monte Carlo p-value, add for example
`--pvalue --mc-replicates 1000 --mc-seed 127 --mc-group full`. The query is
fixed and the target is independently relabeled in each replicate; `gc`
restricts permutations to preserve A/T and C/G groups. The result includes
the replicate count, exceedance count, seed, and permutation group. This
requires motif models, not precomputed score profiles.

Compare precomputed score profiles:

```bash
mimosa compare examples/scores_1.fasta examples/scores_2.fasta \
  --query-type scores --target-type scores --min-logerr 2

mimosa compare-many examples/foxa2.meme examples/gata2.meme examples/gata4.meme \
  --query-type pwm --target-type pwm --fasta examples/foreground.fa \
  --min-logerr 2 --total-threads 4 --numba-threads 1
```

Compute pair-specific Monte Carlo significance:

```bash
mimosa compare examples/foxa2.meme examples/gata2.meme \
  --query-type pwm --target-type pwm \
  --fasta examples/foreground.fa --background examples/background.fa \
  --min-logerr 2 \
  --pvalue --mc-replicates 2000 --mc-seed 127
```

Each replicate independently relabels the target, using either full alphabet
permutations or the GC-preserving group. See
[Method and statistics](docs/python/method.md) for assumptions and
[custom window functions](docs/python/extending.md) for CNN integration.

Manage the optional prepared-profile cache:

```bash
mimosa cache clear --cache-dir .mimosa-cache
```

Successful `compare` results are one JSON object; `compare-many` results are an
ordered JSON array on `stdout`. Diagnostics and errors are
written to `stderr`. Every result includes `n_sites`, including zero-site
comparisons. Run `mimosa --help` for the complete CLI reference.

## Python API

```python
from mimosa import compare, read_fasta, read_model

query = read_model("examples/foxa2.meme")
target = read_model("examples/gata2.meme")
sequences, names = read_fasta("examples/foreground.fa")

result = compare(query, target, sequences, min_logerr=2.0)
print(result.to_dict())
```

For repeated comparisons, prepare the query once:

```python
from mimosa import compare_many, prepare_profile

prepared = prepare_profile(query, sequences, min_logerr=2.0, circular=True)
results = compare_many(prepared, [target], sequences)
```

Comparison only accepts profiles prepared with `circular=True`; the default
`circular=False` keeps the linear scan layout for other uses of
`prepare_profile`.

`compare_many` processes targets in order in a single process, using JAX's
device execution. The legacy `total_threads` and `inner_threads` arguments
are validated but do not control JAX threads or parallelize targets.
`inner_threads` must be from 1 through 4, and `total_threads` divisible by it.

The public API also includes `scan`, `read_scores`, `select_sites`,
`reconstruct_pfm`, `direct_mc`, and `write_model`.
`WindowModel`, `alphabet_permutations`, `InputRelabeledModel`, and
`relabel_model` expose the fixed-window randomization interface.

## Documentation

- [Python Documentation](docs/python/index.md)
- [Python Quickstart](docs/python/quickstart.md)
- [Python API](docs/python/api.md)
- [CLI Reference](docs/python/cli.md)
- [Supported Models and Formats](docs/python/models.md)
- [Comparison Method and Statistics](docs/python/method.md)
- [Custom Models and Readers](docs/python/extending.md)
- [Storage and Cache](docs/python/storage.md)
- [Data Layout and Coordinates](docs/python/data_layout.md)

## Development

Run the test suite with:

```bash
uv run pytest -q
```

Run the production-path performance benchmark with:

```bash
uv run python benchmarks/benchmark_performance.py
```

The report measures full-pipeline `cache_miss_hot_jit` and
`cache_hit_hot_filesystem` modes. Use comma-separated `--total-threads` and
`--numba-threads` values; only divisible pairs are run. These legacy budget
values do not control JAX or target parallelism. Use `--phase-timings`
to add isolated scan, normalization, anchor, fingerprint, cache, alignment,
and JSON serialization timings. On Linux the report includes aggregate peak RSS
for the process tree; other platforms retain the parent-process
lifetime maximum as a fallback.
`--cold-jit-cli` adds a separately labelled clean-process command-line startup
sample; it is emitted only for one-target workloads and is not used for speedup
comparisons.
When projected temporary cache storage is unavailable, the report records that
configuration as `resource_limited` and continues; pass
`--allow-resource-overcommit` only when the host has enough reserved capacity.
Use `--coverage --skewed-rows --thresholds 1,2,4` for the optional
heterogeneous suite (PWM, BaMM, a larger background, ragged rows, and
anchor-density regimes).

For a separate warmed-JIT, single-process input-relabel benchmark:

```bash
NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/mimosa-relabel-numba \
  uv run python benchmarks/benchmark_input_relabel.py examples/myog.ihbcp \
  --orders 0,4 --repeats 3 --output /tmp/mimosa-relabel-benchmark.json
```

It records construction, two-strand scanning, preparation without a disk cache,
and warm profile loading. This measures one scoring function at a time, not
the complete pairwise null construction, and does not predict CNN/GPU speed.

## License

MIT. See [LICENSE](LICENSE).
