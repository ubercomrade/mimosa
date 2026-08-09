# Repository Instructions

## Shape

- This is one Python package/CLI, not a monorepo: source is `src/mimosa`, tests are `tests`, and runnable fixtures are `examples`.
- Entrypoints are `mimosa.cli:main` and `python -m mimosa.cli`.
- `compare_many(query, targets, ...)` is one-query, one-metric-per-call, and preserves target order.

## Environment

- Python `>=3.12` is required. Use the checkout's `uv` environment, never a system `pytest` or unrelated installed `mimosa`.
- Set up with `uv sync --locked --group dev`; keep `uv.lock` synchronized with `pyproject.toml` dependency changes.
- Run commands from the repository root because tests and CLI examples use paths such as `examples/foreground.fa`.

## Verification

- Full tests: `uv run pytest -q`.
- Focused test: `uv run pytest -q tests/test_profiles.py::TestCompare::test_one_to_many`.
- Lint: `uv run ruff check .`; Ruff is the only configured static check. There is no configured formatter, typecheck, or codegen step.
- CLI tests spawn `python -m mimosa.cli` and require JSON-only stdout; diagnostics and progress belong on stderr.
- CI-equivalent example smoke test: `uv run bash examples/run.sh`.
- Numba JIT needs a separate warmup before timing. For performance work set `NUMBA_NUM_THREADS` explicitly and put `NUMBA_CACHE_DIR` outside the repository.

## Runtime Constraints

- Numba row-parallel work requires more than one thread, at least `50_000` items, and at least `64` rows; normalization uses the item/thread thresholds.
- Built-in scan uses row-parallel kernels when eligible; custom models remain serial.
- `compare_many` prepares and compares targets sequentially, preserving target order.
- Do not add nested `prange`, a thread pool, or a process pool without measurements; parallelism is intentionally kept inside Numba kernels.

## Cache

- Prepared-profile cache format v4 stores raw score, offset, and anchor sections in read-only mmap-backed arrays; legacy pickle entries are still accepted.
- Never load an untrusted cache directory: the legacy fallback executes pickle, and checksums detect corruption but do not make pickle safe.
- The in-process prepared-profile LRU defaults to `1 GiB` of backing-array bytes; configure it with `Cache(..., memory_budget_bytes=...)`.
- Cache keys include model, sequences, background, Float32 `min_logerr`, normalization, and algorithm versions. Clear stale entries with `uv run mimosa cache clear --cache-dir .mimosa-cache`.

## Benchmarks

- Production-path benchmark: `uv run python benchmarks/benchmark_performance.py`; it includes one-target and larger target workloads.
- The production benchmark defaults to `10_000` sequences, real example FASTA inputs, `1/64/128/256` targets, three repeats, and `1/2/4/6/8` Numba threads; it is memory-heavy. Write results outside the repository with `--output /tmp/mimosa-performance.json`.
- Keep `.mimosa-cache`, build/test caches, benchmark output, and Numba cache files out of version control.

## CI And Release

- GitHub CI runs locked `uv` setup, Ruff, pytest on Python `3.12` and `3.13`, and the example smoke script.
- Release tags must match `uv version --short` after removing the `v` prefix; the workflow builds with `uv build`, checks with `uvx twine check dist/*`, tests the sdist, and publishes with `uv publish`.
