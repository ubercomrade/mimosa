# Repository Instructions

## Shape

- This is one Python package/CLI, not a monorepo: source is `src/mimosa`, tests are `tests`, and runnable fixtures are `examples`.
- Entrypoints are `mimosa.cli:main` and `python -m mimosa.cli`.
- `compare_many(query, targets, ...)` is one-query and preserves target order.
- The only similarity measure is the weighted Jaccard index (`metric == "jaccard"`) over circular logERR profiles; `compare`, `compare_many`, and `build_null` prepare profiles with `circular=True` and reject linear `PreparedProfile` inputs. `prepare_profile` defaults to the linear layout because other tools use it. `PreparedProfile` is immutable; change `min_logerr` by preparing again.
- Check hypotheses, build mini-prototypes, and run other exploratory experiments in `experiments/`; it is ignored by Git.

## Environment

- Python `>=3.12` is required. Use the checkout's `uv` environment, never a system `pytest` or unrelated installed `mimosa`.
- Set up with `uv sync --locked --group dev`; keep `uv.lock` synchronized with `pyproject.toml` dependency changes.
- Run commands from the repository root because tests and CLI examples use paths such as `examples/foreground.fa`.

## Verification

- Full tests: `uv run pytest -q`.
- Focused test: `uv run pytest -q tests/test_profiles.py::TestCompare::test_one_to_many`.
- The example FASTA has about `1.5e5` positions per strand, too few for the default `min_logerr=4.0`; tests and examples that use it pass `min_logerr=2.0` to avoid the few-positions warning.
- Lint: `uv run ruff check .`; Ruff is the only configured static check. There is no configured formatter, typecheck, or codegen step.
- CLI tests spawn `python -m mimosa.cli` and require JSON-only stdout; diagnostics and progress belong on stderr.
- CI-equivalent example smoke test: `uv run bash examples/run.sh`.
- Numba JIT needs a separate warmup before timing. For performance work set `NUMBA_NUM_THREADS` explicitly and put `NUMBA_CACHE_DIR` outside the repository.

## Runtime Constraints

- Numba row-parallel work requires more than one thread, at least `50_000` items, and at least `64` rows; normalization uses the item/thread thresholds.
- Built-in scan uses row-parallel kernels when eligible; custom models remain serial.
- Row-parallel alignment additionally requires `(query anchors + target anchors) * shifts >= 25_000`; serial and parallel kernels reduce rows in order and must stay bitwise identical.
- `compare_many` derives joblib target workers from `total_threads / inner_threads`, runs the complete target pipeline in parallel workers when eligible, and preserves target order.
- Do not add nested `prange` or another process pool; joblib outer parallelism and Numba inner parallelism are the measured execution layers.

## Cache

- Prepared-profile cache format v7 stores normalized score and offset sections plus the `circular` flag in read-only mmap-backed arrays; anchors are rebuilt per `min_logerr`, and entries from other formats are misses.
- Cache and bundle readers use `allow_pickle=False`; there is no pickle fallback. Checksums detect corruption, not tampering, so still avoid loading untrusted cache directories.
- The disk cache does not retain prepared profiles in a process-local store; verified-entry state is per cache instance and prepared arrays are read-only mmap-backed sections.
- Cache keys include model, sequences, background, normalization, the `circular` layout, and algorithm versions; `min_logerr` is not part of the key. Clear stale entries with `uv run mimosa cache clear --cache-dir .mimosa-cache`.

## Benchmarks

- Production-path benchmark: `uv run python benchmarks/benchmark_performance.py`; it includes one-target and larger target workloads.
- The production benchmark defaults to `10_000` sequences, real example FASTA inputs, `1/64/128/256` targets, three repeats, and divisible comma-separated total/Numba budgets; it is memory-heavy. Write results outside the repository with `--output /tmp/mimosa-performance.json`.
- Keep `.mimosa-cache`, build/test caches, benchmark output, and Numba cache files out of version control.

## CI And Release

- GitHub CI runs locked `uv` setup, Ruff, pytest on Python `3.12` and `3.13`, and the example smoke script.
- Release tags must match `uv version --short` after removing the `v` prefix; the workflow builds with `uv build`, checks with `uvx twine check dist/*`, tests the sdist, and publishes with `uv publish`.
