# Mimosa

Mimosa is a Python package and command-line tool for DNA motif comparison.
It supports PWM/PFM, BaMM, Dimont, Slim, and precomputed score profiles.
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
Markov-based approaches (BaMM, InMoDe, Dimont, Slim, MODER2) and deep learning
models (DeepBind, DeepGRN, BERT-TFBS).
However, a mature ecosystem of databases and tools has been developed primarily
for PWMs, including motif comparison methods that are critical for result
interpretation. Comparison of alternative motif models usually requires their
conversion into PWMs, which inevitably leads to loss of information about
positional dependencies. At present, no universal tool exists for their direct
comparison. Mimosa (Model-Independent Motif Similarity Assessment) addresses
this gap by evaluating motif similarity based on functional behavior, namely
the scores models assign to DNA sequences rather than their internal parameters.

Mimosa uses the following profile comparison pipeline:

1. Scan the same DNA sequences with each model to obtain score profiles.
2. Normalize the scores to make different models comparable.
3. Identify positions with high scores in either profile.
4. Calculate weighted Jaccard similarity across profile alignments and strand
   orientations.
5. Return the best alignment and its similarity score.

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

## Installation

Mimosa supports Python 3.12 and 3.13. The package is published on
[PyPI](https://pypi.org/project/mimosa-tool/) as `mimosa-tool`.

```bash
python -m pip install mimosa-tool
```

To install from source with uv, run from the repository root:

```bash
uv sync --locked --group dev
uv run mimosa --version
```

Alternatively, install this checkout with `python -m pip install .`.
Paths under `examples/` below refer to files in the repository, not files
installed beside the CLI. With the uv environment, prefix CLI commands with
`uv run`.

Mimosa uses JAX for computation. The default installation supports CPU. GPU
plugins are optional: an installer cannot reliably select one from the target
hardware, driver and system libraries, which may differ from the installation
machine.

For an NVIDIA GPU on supported Linux systems, choose **one** CUDA extra:

```bash
python -m pip install 'mimosa-tool[cuda13]'
# Or, for CUDA 12:
python -m pip install 'mimosa-tool[cuda12]'
```

These extras install the matching JAX plugin and CUDA libraries; a compatible
NVIDIA driver must already be installed. Check GPU, driver and platform support
in the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).
For a source checkout use `uv sync --locked --group dev --extra cuda13` (or
`cuda12`) and include the same `--extra cuda13` in subsequent `uv run`
commands, for example `uv run --extra cuda13 mimosa --version`. Do not select
both CUDA extras in one environment.

For an AMD GPU on supported Linux systems, install the ROCm extra:

```bash
python -m pip install 'mimosa-tool[rocm]'
```

For a source checkout, use `uv sync --locked --group dev --extra rocm`, then
include `--extra rocm` in subsequent `uv run` commands. This installs the JAX
ROCm 7 plugin and PJRT, but **not** the ROCm system libraries or GPU driver.
Install a compatible ROCm version and driver first, and check your GPU and
platform against [AMD's JAX installation guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/jax/install.html)
and [compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).
The project pins JAX 0.10.0; the extra uses its `rocm7-local` variant, not
newer JAX extras that install ROCm libraries. Select only one GPU extra.

JAX selects the compute device automatically. Inspect the selected backend and
its available devices in the same environment used to run Mimosa:

```bash
python -c "import jax; print(jax.default_backend(), jax.devices())"
```

For the uv checkout, use `uv run --extra rocm python -c 'import jax; print(jax.default_backend(), jax.devices())'`
(or the selected CUDA extra). To require a GPU instead of silently using CPU, run
with `JAX_PLATFORMS=gpu`; JAX will report an error if it cannot initialize one.

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

Compare precomputed score profiles:

```bash
mimosa compare examples/scores_1.fasta examples/scores_2.fasta \
  --query-type scores --target-type scores --min-logerr 2
```

Compare one query against multiple targets:

```bash
mimosa compare-many examples/foxa2.meme examples/gata2.meme examples/gata4.meme \
  --query-type pwm --target-type pwm --fasta examples/foreground.fa \
  --min-logerr 2
```

Compute pair-specific Monte Carlo significance:

```bash
mimosa compare examples/foxa2.meme examples/gata2.meme \
  --query-type pwm --target-type pwm \
  --fasta examples/foreground.fa --background examples/background.fa \
  --min-logerr 2 \
  --pvalue --mc-replicates 2000 --mc-seed 127
```

Monte Carlo randomizes the query while keeping targets fixed. This option
requires motif models, not precomputed score profiles. See
[Method and statistics](docs/python/method.md) for assumptions and settings.

Manage the optional prepared-profile cache:

```bash
mimosa cache clear --cache-dir .mimosa-cache
```

Successful `compare` results are one JSON object; `compare-many` results are an
ordered JSON array on `stdout`. Diagnostics and errors are
written to `stderr`. Run `mimosa --help` to list commands.

Command-specific options are listed by `mimosa compare --help`,
`mimosa compare-many --help`, and `mimosa cache --help`. In particular:

- `--fasta` supplies the shared comparison sequences.
- `--background` supplies separate sequences for score calibration; without it,
  calibration uses the comparison sequences.
- `--background-freq` is the per-base frequency used for PFM-to-PWM conversion
  (uniform `0.25`), not a sequence-generation or calibration-file option.
- `--cache-dir` enables persistent normalized-profile caching across runs;
  caching is disabled if omitted.

## Python API

```python
from mimosa import compare, read_fasta, read_model

query = read_model("examples/foxa2.meme")
target = read_model("examples/gata2.meme")
sequences, names = read_fasta("examples/foreground.fa")

result = compare(query, target, sequences, min_logerr=2.0)
print(result.to_dict())
```

Compare one query against multiple targets:

```python
from mimosa import compare_many

targets = [target, read_model("examples/gata4.meme")]
results = compare_many(query, targets, sequences, min_logerr=2.0)
```

The [quickstart](docs/python/quickstart.md) and [API reference](docs/python/api.md)
cover profile reuse, site selection, PFM reconstruction, and Monte Carlo.
See [custom models](docs/python/extending.md) to integrate another scorer.

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
- [Migrating to JAX](docs/python/migration.md)

## Development

Run the code checks, tests, and example commands:

```bash
uv run ruff check .
uv run pytest -q tests
uv run bash examples/run.sh
```

Run the production-path performance benchmark with:

```bash
uv run python benchmarks/benchmark_performance.py --output /tmp/mimosa-performance.json
```

The default run uses 10 000 sequences, 1/64/128/256 targets and three repeats;
it can require substantial memory and temporary disk space. For a smaller check:

```bash
uv run python benchmarks/benchmark_performance.py \
  --sequence-count 100 --target-counts 1 --repeats 1 --min-logerr 2 \
  --output /tmp/mimosa-performance-small.json
```

Run the benchmark with `--help` for detailed options. Report the selected JAX
device and distinguish compilation time from warmed execution when sharing
performance measurements.

## License

MIT. See [LICENSE](LICENSE).
