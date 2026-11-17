# CLI Reference

The executable is `mimosa` and is also available as `python -m mimosa.cli`.
With `uv`, use `uv run mimosa` from the repository checkout.

Successful machine-readable output is written to `stdout`. Diagnostics,
errors, and progress reporting are written to `stderr`.

## `compare`

Compare two model-derived or precomputed profiles:

```bash
uv run mimosa compare QUERY TARGET \
  --query-type QUERY_TYPE --target-type TARGET_TYPE \
  [--fasta FASTA] [--background FASTA] [options]
```

Model types are `scores`, `pwm`, `bamm`, `sitega`, `dimont`, and `slim`.
Every comparison reports the weighted Jaccard similarity of circular profiles
as `"metric": "jaccard"`; see [Method and statistics](method.md).

Important options:

- `--fasta PATH` supplies the comparison sequences.
- Without `--fasta`, `--num-sequences` (default `10000`), `--seq-length` (default `100`), and `--seed` control generated uniform DNA sequences.
- `--background PATH` supplies a separate sequence batch for normalization.
- `--search-range` (default `5`) sets the largest circular shift tried in each direction.
- `--min-logerr` (default `4.0`, must be positive) is the threshold a position must reach in either profile to contribute.
- `--cache-dir PATH` enables prepared-profile disk/mmap caching.
- `--pvalue --null-distribution PATH` adds empirical significance annotations.
- `--effective-number-of-targets N` changes E-values without changing BH adjustment.
- `--query-order N` and `--target-order N` select an available BaMM order when reading input; other types reject these options.

For a scores-only comparison, `--fasta` and `--background` are rejected because
the profiles already contain the comparison data. `--null-distribution` and
`--effective-number-of-targets` require `--pvalue`.

The default threshold expects about `10^6` positions per strand. With fewer
than 50 expected positions above the threshold, a warning is written to
`stderr`; lower `--min-logerr` for small FASTA batches.

Example:

```bash
uv run mimosa compare examples/foxa2.meme examples/gata2.meme \
  --query-type pwm --target-type pwm \
  --fasta examples/foreground.fa --min-logerr 2
```

## `compare-many`

Compare one query against an ordered list of targets. All targets use the same
`--target-type`:

```bash
uv run mimosa compare-many QUERY TARGET [TARGET ...] \
  --query-type QUERY_TYPE --target-type TARGET_TYPE \
  [--total-threads TOTAL] [--numba-threads NUMBA] [options]
```

`--numba-threads` is limited to 1 through 4. `TOTAL` must be positive and
divisible by `NUMBA`; the derived joblib worker count is `TOTAL / NUMBA`.
Results are emitted as an ordered JSON array. `--pvalue` annotates every result
after validating the comparison parameters against the null bundle.
`--cache-dir` is optional; without it no cache directory or mmap reuse is
created.

## `build-null`

Build a profile null distribution from a motif directory:

```bash
uv run mimosa build-null MOTIFS \
  --output OUTPUT \
  [--fasta FASTA] [--background FASTA] [options]
```

The default `--randomization pwm-shuffle` uses the first motif from each MEME
file, preserving the historical PWM behavior. To use another model type, pass
`--randomization input-relabel --model-type TYPE`. Types are `pwm` (MEME/PFM),
`bamm` (ihbcp), `sitega` (mat), `dimont` and `slim` (xml). At least two uniquely
named models of one type/order are required. `--order N` selects a BaMM order;
it is rejected for other types. `--background` supplies an optional separate
normalization batch.
`--num-samples`, `--seed`, `--search-range`, `--min-logerr`, and `--cache-dir`
control construction. Application requires the same sequences, background,
`--search-range` and `--min-logerr`. Input-relabel bundles also require both
compared models to match the recorded kind/order. The summary is JSON on
`stdout`, including the randomization and calibration group when present.

```bash
uv run mimosa build-null examples/ --model-type bamm --order 0 \
  --randomization input-relabel --min-logerr 3 --num-samples 2000 \
  --seed 127 --output output/bamm0_null
```

This generates uniform DNA by default. The scalar-score invariance of input
relabeling assumes uniform iid DNA; arbitrary FASTA/background inputs require
separate justification. See [Method and statistics](method.md). Custom neural
window functions are available through the Python API, not a built-in neural
checkpoint reader.

## `cache clear`

Remove prepared-profile cache entries:

```bash
uv run mimosa cache clear --cache-dir .mimosa-cache
```

## Process behavior

- `--version` prints the package version.
- `--help` prints command and option help.
- `compare` returns one JSON object; `compare-many` returns an ordered JSON array.
- Runtime, input, and model failures return exit code `2` and are emitted as
  `error: <message>` on stderr.
- Standard argparse usage errors also use argparse's exit code `2`.
