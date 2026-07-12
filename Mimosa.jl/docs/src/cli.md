# CLI

Mimosa.jl provides a thin command-line interface over the public API. The CLI
is a simple adapter: it parses arguments, calls the API, and serializes results.

## Usage

```bash
julia --project=Mimosa.jl app/mimosa.jl <command> [options]
```

## Commands

### `motif` — Direct motif comparison

```bash
julia app/mimosa.jl motif --query model1.meme --target model2.meme --metric pcc
```

Options:
- `--query <path>`: Query model file (required)
- `--target <path>`: Target model file (required)
- `--metric <name>`: Metric: `pcc`, `ed`, `cosine` (default: `pcc`)
- `--output <path>`: Write JSON to file instead of stdout
- `--threads <n>`: Number of threads (default: 1)
- `--quiet`: Suppress informational output
- `--verbose`: Verbose diagnostics to stderr

### `profile` — Profile-based comparison

```bash
julia app/mimosa.jl profile --query model.meme --fasta sequences.fa --metric co
```

Options:
- `--query <path>`: Query model file (required)
- `--target <path>`: Target model file (for motif-vs-motif profile comparison)
- `--fasta <path>`: FASTA sequences for scanning
- `--metric <name>`: Profile metric: `co`, `co_rowwise`, `dice`, `dice_rowwise`, `cosine`
- `--num-sequences <n>`: Random sequences if no FASTA (default: 1000)
- `--seq-length <n>`: Random sequence length (default: 200)
- `--seed <n>`: Random seed (default: 127)

### `build-null` — Build null distribution

```bash
julia app/mimosa.jl build-null --models models_dir --relations groups.tsv --output null
```

Options:
- `--models <path>`: Directory containing model files (required)
- `--relations <path>`: TSV with motif group assignments (required)
- `--strategy <name>`: `motif` or `profile` (required)
- `--metric <name>`: Metric (default: `pcc` for motif, `co` for profile)
- `--output <path>`: Output directory for null distribution
- `--threads <n>` / `--jobs <n>`: Number of threads (alias)

### `cache clear` — Clear disk cache

```bash
julia app/mimosa.jl cache clear --dir /tmp/mimosa_cache
```

### `inspect-model` — Display model metadata

```bash
julia app/mimosa.jl inspect-model --model model.meme
```

### `convert-model` — Convert legacy model to portable format

```bash
julia app/mimosa.jl convert-model --input model.ihbcp --output model_bundle
```

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