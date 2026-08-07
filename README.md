# Mimosa

Mimosa is a Python package and command-line tool for comparing DNA motif models
through the score profiles they produce on the same sequences. It supports PWM,
BaMM, SiteGA, Dimont, Slim, and precomputed score profiles without converting
all models to a common matrix representation.

The library is built on NumPy and Numba and provides model I/O, scanning,
profile normalization and alignment, site extraction, PFM reconstruction,
empirical null distributions, and a prepared-profile cache.

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

Mimosa requires Python 3.12 or newer.

```bash
uv sync
```

For a regular local installation:

```bash
python -m pip install .
```

## CLI

Compare two PWM models on a FASTA batch:

```bash
uv run mimosa profile examples/foxa2.meme examples/gata2.meme \
  --model1-type pwm --model2-type pwm \
  --fasta examples/foreground.fa --metric co
```

Compare precomputed score profiles:

```bash
uv run mimosa profile examples/scores_1.fasta examples/scores_2.fasta \
  --model1-type scores --model2-type scores --metric cosine
```

Build and use an empirical null distribution:

```bash
uv run mimosa build-null examples/ \
  --model-type pwm --output output/null_bundle \
  --fasta examples/foreground.fa --num-samples 2000 --seed 127

uv run mimosa profile examples/foxa2.meme examples/gata2.meme \
  --model1-type pwm --model2-type pwm \
  --fasta examples/foreground.fa \
  --pvalue --null-distribution output/null_bundle
```

Manage the optional prepared-profile cache:

```bash
uv run mimosa cache clear --cache-dir .mimosa-cache
```

Successful `profile` results are JSON on `stdout`. Diagnostics and errors are
written to `stderr`. Run `uv run mimosa --help` for the complete CLI reference.

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

Run the benchmark with:

```bash
uv run python benchmark/bench.py --fasta examples/foreground.fa
```

## License

MIT. See [LICENSE](LICENSE).
