# CLI

Mimosa.jl provides a thin command-line interface over the public API. The CLI
is a simple adapter: it parses arguments, calls the API, and serializes results
as JSON to stdout. Diagnostics go to stderr only.

## Usage

```bash
julia --project=Mimosa.jl app/mimosa.jl <command> [options]
```

Global options:
- `--help`, `-h` — Show help message
- `--version`, `-V` — Show version
- `--quiet` — Suppress informational stderr output
- `--verbose` — Enable verbose stderr diagnostics

## Commands

- `--pfm-mode` — Force PFM reconstruction before comparison
- `--pfm-top-fraction <f>` — Fraction of top sites for PFM (default: 0.05)
- `--fasta <path>` — FASTA sequences for PFM reconstruction
- `--num-sequences <n>` — Random sequences if no FASTA (default: 20000)
- `--seq-length <n>` — Random sequence length (default: 100)
- `--seed <n>` — Random seed (default: 127)
- `--background <f>` — Background frequency for PWM (default: 0.25)
- `--query-index <n>` — MEME motif index for model1 (default: 0)
- `--target-index <n>` — MEME motif index for model2 (default: 0)
- `--threads <n>` — Number of threads (default: 1)
- `--pvalue` — Annotate result using an explicit `--null-distribution` bundle
- `--null-distribution <path>` — Portable null-distribution bundle for `--pvalue`
- `--effective-number-of-targets <n>` — E-value target-count override
- `--quiet` — Suppress informational output
- `--verbose` — Verbose diagnostics to stderr

### `profile` — Profile-based comparison

```bash
julia --project=Mimosa.jl app/mimosa.jl profile examples/pif4.meme examples/gata2.meme \
  --model1-type pwm --model2-type pwm --metric co --num-sequences 50 --seq-length 100
```

Required arguments:
- `model1` — Path to first model or score-profile file (positional)
- `model2` — Path to second model or score-profile file (positional)
- `--model1-type <type>` — Type: `scores`, `pwm`, `bamm`, `sitega`, `dimont`, `slim`
- `--model2-type <type>` — Type: `scores`, `pwm`, `bamm`, `sitega`, `dimont`, `slim`

Profile comparison options:
- `--metric <name>` — Metric: `co`, `co_rowwise`, `dice`, `dice_rowwise`, `cosine` (default: `co`)
- `--search-range <n>` — Max site-center shift (default: 10)
- `--window-radius <n>` — Window radius in profile positions (default: 10)
- `--realign-window <n>` — Local realignment half-width (default: 3)
- `--min-logfpr <f>` — Threshold logFPR (0 = best site per sequence)

Sequence options:
- `--fasta <path>` — FASTA for motif scanning
- `--background <path>` — FASTA for normalization calibration
- `--num-sequences <n>` — Random sequences if no FASTA (default: 1000)
- `--seq-length <n>` — Random sequence length (default: 200)
- `--seed <n>` — Random seed (default: 127)
- `--background-freq <f>` — Background frequency for PWM (default: 0.25)

Annotation options:
- `--pvalue` — Annotate result using an explicit `--null-distribution` bundle
- `--null-distribution <path>` — Portable null-distribution bundle for `--pvalue`
- `--effective-number-of-targets <n>` — E-value target-count override

Technical options:
- `--threads <n>` — Number of threads (default: 1)
- `--quiet` — Suppress informational output
- `--verbose` — Verbose diagnostics to stderr

### `build-null` — Build null distribution

```bash
julia --project=Mimosa.jl app/mimosa.jl build-null examples/ \
  --model-type pwm --groups groups.tsv --output null_dist
```

Required arguments:
- `motifs` — Motif collection: directory or multi-motif MEME file (positional)
- `--model-type <type>` — Motif format: `pwm`, `bamm`, `sitega`, `dimont`, `slim`
- `--groups <path>` — TSV/CSV with motif and group columns
- `--output <path>` — Output path for null distribution

Relation options:
- `--name-column <s>` — Motif-name column (default: `motif`)
- `--group-column <s>` — Group column (default: `group`)
- `--ignore-missing` — Ignore relation names not loaded

Comparison options:
- `--metric <name>` — Profile metric (default: `co`)
- `--fasta <path>` — FASTA for profile scanning
- `--num-sequences <n>` — Random sequences (default: 1000)
- `--seq-length <n>` — Random sequence length (default: 200)
- `--seed <n>` — Random seed (default: 127)
- `--search-range <n>` — Max shift (default: 10)
- `--window-radius <n>` — Window radius (default: 10)
- `--realign-window <n>` — Realignment window (default: 3)
- `--min-logfpr <f>` — Threshold logFPR

Output options:
- `--strict` — Fail when a query lacks enough null targets
- `--min-null-targets <n>` — Minimum null targets (default: 1)

Technical options:
- `--threads <n>` — Number of threads (default: 1)
- `--jobs <n>` — Alias for `--threads` (deprecated)
- `--quiet` — Suppress informational output
- `--verbose` — Verbose diagnostics to stderr

### `cache clear` — Clear disk cache

```bash
julia --project=Mimosa.jl app/mimosa.jl cache clear --cache-dir /tmp/mimosa_cache
```

Options:
- `--cache-dir <dir>` — Cache directory (default: `.mimosa-cache`)

### `inspect-model` — Display model metadata

```bash
julia --project=Mimosa.jl app/mimosa.jl inspect-model examples/foxa2.ihbcp --type bamm
```

Required arguments:
- `path` — Path to model file (positional)

Options:
- `--type <type>` — Model type (default: auto-detect)
- `--index <n>` — MEME motif index (default: 0)
- `--background <f>` — Background frequency for PWM (default: 0.25)

### `convert-model` — Convert legacy model to portable format

```bash
julia --project=Mimosa.jl app/mimosa.jl convert-model examples/pif4.meme output/pif4_bundle
```

Required arguments:
- `input` — Path to input model file (positional)
- `output` — Path to output bundle directory (positional)

Options:
- `--type <type>` — Model type (default: auto-detect)
- `--index <n>` — MEME motif index (default: 0)
- `--background <f>` — Background frequency for PWM (default: 0.25)

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Usage error (missing/invalid arguments) |
| 2 | Runtime error (file not found, parse error, etc.) |

## Output conventions

- JSON results are written **only** to stdout
- Diagnostics, progress, and errors are written **only** to stderr
- No stack traces by default (use `--verbose` for debug output)
- No interactive prompts in batch mode
