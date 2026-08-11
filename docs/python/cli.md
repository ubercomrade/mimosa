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
Metrics are `co`, `dice`, and `cosine`.

Important options:

- `--fasta PATH` supplies the comparison sequences.
- Without `--fasta`, `--num-sequences`, `--seq-length`, and `--seed` control generated DNA sequences.
- `--background PATH` supplies a separate sequence batch for normalization.
- `--search-range`, `--window-radius`, and `--realign-window` configure alignment.
- `--min-logerr` controls anchor selection and exact-tail calibration.
- `--cache-dir PATH` enables prepared-profile disk/mmap caching.
- `--pvalue --null-distribution PATH` adds empirical significance annotations.
- `--effective-number-of-targets N` changes E-values without changing BH adjustment.

Example:

```bash
uv run mimosa compare examples/foxa2.meme examples/gata2.meme \
  --query-type pwm --target-type pwm \
  --fasta examples/foreground.fa --metric cosine
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
after validating the query and shared target types against the null bundle.
`--cache-dir` is optional; without it no cache directory or mmap reuse is
created.

## `build-null`

Build a PWM-only profile null distribution from a motif directory:

```bash
uv run mimosa build-null MOTIFS \
  --output OUTPUT \
  [--fasta FASTA] [options]
```

`MOTIFS` must be a directory containing at least two readable PWM files.
`--num-samples`, `--seed`, `--metric`, the alignment options, and `--cache-dir`
control construction. The summary is JSON on `stdout`.

## `cache clear`

Remove prepared-profile cache entries:

```bash
uv run mimosa cache clear --cache-dir .mimosa-cache
```

## Process behavior

- `--version` prints the package version.
- `--help` prints command and option help.
- `compare` returns one JSON object; `compare-many` returns an ordered JSON array.
- `CLIError` failures return exit code `1`.
- Runtime, input, and model failures return exit code `2`.
- Standard argparse usage errors also use argparse's exit code `2`.
