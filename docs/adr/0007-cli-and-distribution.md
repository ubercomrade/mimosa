# ADR 0007: CLI and distribution

## Context

Stage 8 requires a CLI that is a thin adapter over the public API, with stable
exit codes, JSON output in stdout, diagnostics in stderr, and legacy migration
tools. The PLAN specifies commands: `motif`, `profile`, `build-null`,
`cache clear`, `inspect-model`, `convert-model`, `convert-null`.

## Decision

### CLI parser

Use a custom minimal parser (stdlib only, 0 external dependencies) instead of
ArgParse.jl or Comonicon.jl. Rationale:
- Current project has 0 external runtime dependencies (besides stdlib)
- Adding ArgParse/Comonicon would increase latency, install size, and maintenance
- The custom parser covers all needed cases: subcommands, flags, key-value args,
  help text, and error messages

### CLI structure

```
app/mimosa.jl — standalone entry point
src/cli.jl    — parser + command dispatch
```

Six commands: `motif`, `profile`, `build-null`, `cache clear`, `inspect-model`,
`convert-model`.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Usage error (missing/invalid arguments) |
| 2 | Runtime error (file not found, parse error, etc.) |

### Output conventions

- JSON results → stdout only
- Errors, progress, diagnostics → stderr only
- No stack traces by default
- No interactive prompts in batch mode

### Legacy conversion

- `scripts/convert_legacy_model.py` — trusted legacy model converter
- `scripts/convert_legacy_null.py` — trusted legacy null converter
- Both require explicit `--trusted-input` flag (security guard)
- Julia package never reads pickle/joblib directly

## Alternatives considered

1. **ArgParse.jl**: Mature, well-tested, but adds dependency weight and latency
2. **Comonicon.jl**: Feature-rich with app generation, but heavy for this use case
3. **Package app with PackageCompiler**: Considered for distribution, but adds
   complexity for initial release. Evaluated separately in Stage 10.

## Consequences

- CLI is simple, maintainable, and has 0 external dependencies
- JSON output is stable and machine-readable
- Exit codes are documented and tested
- Legacy conversion is a separate concern, not part of core package
- `--progress` interactive mode deferred (batch-safe behavior sufficient for CI)
- `--debug` stacktrace mode deferred (use `--verbose` for now)

## Migration impact

- Python CLI commands have compatible Julia equivalents
- JSON output schema matches Python where possible
- `--threads` replaces `--jobs` (alias kept for compatibility)