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

Model types are `scores`, `pwm`, `bamm`, `dimont`, and `slim`. SiteGA is
temporarily disabled.
Every comparison reports the weighted Jaccard similarity of circular profiles
as `"metric": "jaccard"`; see [Method and statistics](method.md).

Important options:

- `--fasta PATH` supplies the comparison sequences.
- Without `--fasta`, `--num-sequences` (default `10000`), `--seq-length` (default `100`), and `--seed` control generated uniform DNA sequences.
- `--background PATH` supplies a separate sequence batch for normalization.
- `--search-range` (default `5`) sets the largest circular shift tried in each direction.
- `--min-logerr` (default `4.0`, must be positive) is the threshold a position must reach in either profile to contribute.
- `--cache-dir PATH` enables prepared-profile disk/mmap caching.
- `--pvalue` computes pair-specific Monte Carlo significance. `--mc-replicates`
  sets K, `--mc-seed` the seed, and `--mc-group` either `full` or `gc`.
- `--mc-batch-size` bounds random query profiles held on the device (default 8).
  `--mc-comparison-block-size` bounds the vectorized comparison block (default 8);
  larger values may use substantially more device memory.
  `--mc-algorithm` selects `sparse` (default, with dense fallback) or `dense`.
  Sparse and dense reductions can differ numerically, so validate results for
  your data and device when this distinction matters.
  `--mc-prefetch N` enables bounded host read-ahead for disk-backed targets
  (0–64; default 0). Queued profiles consume additional host memory.
  `--mc-anchor-memory-mib` limits retained target-anchor arrays (default 256;
  0 rebuilds them for each query batch). It is not a total RAM or VRAM limit.
  `--mc-storage` selects `auto`, `memory` or `disk` for target profiles;
  `--mc-memory-limit-mib` sets the auto threshold (default 256 MiB), and
  `--mc-temp-dir` selects the parent directory for temporary disk storage.
- `--effective-number-of-targets N` changes E-values without changing BH adjustment.
- `--query-order N` and `--target-order N` select an available BaMM order when reading input; other types reject these options.

For a scores-only comparison, `--fasta` and `--background` are rejected because
the profiles already contain the comparison data. `--pvalue` requires motif
models. `--effective-number-of-targets` requires `--pvalue`.

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
  [options]
```

Targets are processed in one process; JAX manages device execution.
Results are emitted as an ordered JSON array. `--pvalue` computes a separate
Monte Carlo p-value for every target.
`--cache-dir` is optional and controls the persistent prepared-profile cache.
Independently, `--pvalue` may use temporary disk storage for target profiles.
Abrupt process termination can leave temporary files behind.
When `--pvalue` and `--cache-dir` are both set, observed profiles are reused
directly from the persistent cache, without a second temporary profile copy.


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
