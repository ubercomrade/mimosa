# Mimosa

Mimosa is a Python 3.12+ package and command-line tool for DNA motif comparison.
It supports PWM/PFM, BaMM, SiteGA, Dimont, Slim, and precomputed score profiles.
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
2. Convert raw scores to empirical `-log10(ERR)` values, where ERR is the
   expectation recognition rate, optionally using a separate background sequence
   set for calibration.
3. Select one strict best anchor per non-empty sequence when `min_logerr <= 0`,
   or all anchors with `-log10(ERR) >= min_logerr` for a positive threshold.
4. Compare site-centered windows over shifts and the four strand orientations.
5. Return the highest-scoring alignment, its offset, orientation, and number of
   contributing sites.

See [Method and Statistics](docs/python/method.md) for metric definitions,
statistical significance, and references.

## Supported Models

| Model | Source formats | CLI type |
|---|---|---|
| PWM/PFM | MEME, plain PFM, model bundle | `pwm` |
| BaMM | `.ihbcp`, model bundle | `bamm` |
| SiteGA | `.mat`, model bundle | `sitega` |
| Dimont | Jstacs XML, model bundle | `dimont` |
| Slim | Jstacs XML, model bundle | `slim` |
| Score profile | FASTA-like numeric rows | `scores` |

## Installation

Mimosa requires Python 3.12 or newer. It is published on
[PyPI](https://pypi.org/project/mimosa-tool/) as `mimosa-tool`.

```bash
python -m pip install mimosa-tool
```

For development from a source checkout:

```bash
uv sync
```

## CLI

Compare two PWM models on a FASTA batch:

```bash
mimosa profile examples/foxa2.meme examples/gata2.meme \
  --model1-type pwm --model2-type pwm \
  --fasta examples/foreground.fa --metric co
```

Compare precomputed score profiles:

```bash
mimosa profile examples/scores_1.fasta examples/scores_2.fasta \
  --model1-type scores --model2-type scores --metric cosine
```

Build and use an empirical null distribution:

```bash
mimosa build-null examples/ \
  --output output/null_bundle \
  --fasta examples/foreground.fa --num-samples 2000 --seed 127

mimosa profile examples/foxa2.meme examples/gata2.meme \
  --model1-type pwm --model2-type pwm \
  --fasta examples/foreground.fa \
  --pvalue --null-distribution output/null_bundle
```

Manage the optional prepared-profile cache:

```bash
mimosa cache clear --cache-dir .mimosa-cache
```

Successful `profile` results are JSON on `stdout`. Diagnostics and errors are
written to `stderr`. Run `mimosa --help` for the complete CLI reference.

## Python API

```python
from mimosa import compare, read_fasta, read_model

query = read_model("examples/foxa2.meme")
target = read_model("examples/gata2.meme")
sequences, names = read_fasta("examples/foreground.fa")

result = compare(query, target, sequences, metric="co")
print(result.to_dict())
```

For repeated comparisons, prepare the query once:

```python
from mimosa import compare_many, prepare_profile

prepared = prepare_profile(query, sequences)
results = compare_many(prepared, [target], sequences, metric="cosine")
```

The public API also includes `scan`, `read_scores`, `select_sites`,
`reconstruct_pfm`, `build_null`, `annotate_results`, and `write_model`.

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

## License

MIT. See [LICENSE](LICENSE).
