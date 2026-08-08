# Repository Instructions

## Project Shape

- This is a single Python package/CLI, not a monorepo. Source is under `src/mimosa`; tests are under `tests`; runnable fixtures are under `examples`.
- The package entry point is `mimosa.cli:main`; the CLI can also be run as `python -m mimosa.cli`.
- `compare_many(query, targets, ...)` is intentionally one-query and preserves target order. Large target batches use the Numba target-parallel kernel; there is no runtime process pool.

## Environment

- Requires Python `>=3.12`; use `uv` for the checkout environment, not a system `pytest` or an unrelated installed `mimosa` package.
- Set up with `uv sync`. `uv.lock` is the lockfile and must stay in sync with dependency changes in `pyproject.toml`.
- Run commands from the repository root because tests and CLI fixtures use paths such as `examples/foreground.fa`.

## Verification

- Full tests: `uv run pytest -q`.
- Focused test: `uv run pytest -q tests/test_profiles.py::TestCompare::test_one_to_many`.
- CLI tests are subprocess-based and assert JSON-only stdout; diagnostics/progress belong on stderr.
- Lint: `uv run ruff check .`. Ruff is the only configured static check; no typecheck, codegen, CI workflow, or formatter configuration is present.
- Numba JIT makes the first call a compilation warmup. For performance work set `NUMBA_NUM_THREADS` explicitly, warm up once, and use `NUMBA_CACHE_DIR` outside the repository to avoid benchmark artifacts.

## Runtime Constraints

- Numba scan dispatch requires more than one thread, at least `50_000` items, and at least `64` rows; normalization requires more than one thread and `50_000` items; `compare_many` uses the target-parallel path at `64` or more prepared targets.
- Do not add nested `prange`, a thread pool, or a process pool without measurements; the existing design keeps parallelism inside Numba kernels to avoid oversubscription and profile IPC/copying.
- Prepared-profile cache entries are trusted Python pickle payloads. Never load a cache directory from an untrusted source; checksums detect corruption but do not make pickle safe.
- Cache keys include model, sequences, background, Float32 `min_logerr`, normalization, and algorithm versions. Clear stale entries with `uv run mimosa cache clear --cache-dir .mimosa-cache`.
- `.mimosa-cache`, build outputs, test caches, and virtual environments are ignored by `.gitignore`; do not add generated benchmark or Numba cache files.
