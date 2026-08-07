# CLI Reference

The executable is `mimosa` and is also available as `python -m mimosa.cli`.
With `uv`, use `uv run mimosa` from the repository checkout.

Successful machine-readable output is written to `stdout`. Diagnostics,
errors, and progress reporting are written to `stderr`.

## `profile`

Compare two model-derived or precomputed profiles:

```bash
uv run mimosa profile MODEL1 MODEL2 \
  --model1-type TYPE1 --model2-type TYPE2 \
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
- `--cache-dir PATH` enables prepared-profile caching.
- `--pvalue --null-distribution PATH` adds empirical significance annotations.
- `--effective-number-of-targets N` changes E-values without changing BH adjustment.
- `--quiet` disables terminal progress output.

Example:

```bash
uv run mimosa profile examples/foxa2.meme examples/gata2.meme \
  --model1-type pwm --model2-type pwm \
  --fasta examples/foreground.fa --metric cosine
```

## `build-null`

Build a PWM-only profile null distribution from a motif directory:

```bash
uv run mimosa build-null MOTIFS \
  --model-type pwm --output OUTPUT \
  [--fasta FASTA] [options]
```

`MOTIFS` must be a directory containing at least two readable PWM files.
`--num-samples`, `--seed`, `--metric`, the alignment options, and `--cache-dir`
control construction. The summary is JSON on `stdout` unless `--quiet` is
used.

## `cache clear`

Remove prepared-profile cache entries:

```bash
uv run mimosa cache clear --cache-dir .mimosa-cache
```

## Process behavior

- `--version` prints the package version.
- `--help` prints command and option help.
- `profile` returns JSON for successful comparisons.
- `CLIError` failures return exit code `1`.
- Runtime, input, and model failures return exit code `2`.
- Standard argparse usage errors also use argparse's exit code `2`.
