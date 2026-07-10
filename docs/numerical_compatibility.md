# Numerical Compatibility

> **Stage 0 audit artifact.** Defines tolerance classes, comparison methodology, and known
> numerical risk areas for the Python → Julia port. All tolerances must be justified; none may be
> widened merely to make a test pass.

## 1. Tolerance classes

| Class | Applicable values | Default tolerance | Rationale |
|---|---|---|---|
| `exact` | Encoded sequence bytes, offsets, lengths, shapes, orientations, indices, counts, schema fields, pair ordering, tie-breaking | Exact equality (`==`) | These are deterministic control flow and integer quantities. Any difference indicates a bug. |
| `float32_kernel` | Raw scan scores, motif alignment scores, per-position profile values | `atol=1e-5, rtol=1e-4` (subject to accumulation study) | Python uses `np.float32` in Numba kernels with `fastmath=True`. Julia will use `Float32` without fastmath initially. Differences from FMA/reduction order are expected at the ULP level. |
| `float64_reduction` | Aggregated scores (CO/Dice/cosine over windows), shifted alignment reductions | `atol=1e-10, rtol=1e-8` | Python reduces `np.float64` partials from `np.float32` inputs. Julia will do the same. Sums are deterministic for the same iteration order. |
| `statistical_fit` | GEV shape/location/scale parameters, survival function values | `atol=1e-6, rtol=1e-4` (corpus-specific) | SciPy and native Julia use different optimizers and initialization. Parameters may differ at the optimization tolerance level. Survival function values should agree more closely than parameters. See ADR 0005. |
| `tail_probability` | P-values, E-values, adjusted p-values (BH) | `atol=1e-8, rtol=1e-5` (relative to `max(p, 1)`) | P-values derive from GEV SF. BH adjustment is a sorting + cumulative operation; should be near-exact given identical input p-values. |
| `documented_divergence` | Any value where Julia intentionally differs | Per-case, with ADR + migration note | Requires explicit ADR, migration note, and regression test asserting the Julia behavior. |

## 2. Accumulation study plan

Before finalizing `float32_kernel` tolerances, the following experiments must be run:

1. **PWM scan**: Compare `np.float32` Numba kernel output vs Julia `Float32` loop for widths 8, 15, 30 and sequence lengths 100, 1000, 20000. Measure max absolute and relative deviation per position.
2. **BaMM scan**: Same as above for orders 1-5. The 5-ary encoding and context padding may introduce additional accumulation differences.
3. **Profile alignment**: Compare `score_shift` output (float64 reduction of float32 inputs) for CO, Dice, cosine, rowwise variants. The fused kernel sums per-window overlap/cosine; reduction order matters.
4. **Motif alignment**: Compare PCC/ED/cosine column-wise scores. These use `np.float32` with `np.mean`, `np.sum`, `np.sqrt`. Julia will use ordinary loops.

If the accumulation study reveals drift exceeding the default `float32_kernel` tolerance, the options are:
- Widen the tolerance with a documented numerical justification.
- Switch the accumulation type to `Float64` for that kernel (with a documented performance impact).
- Adjust the computation order (with ADR).

## 3. Known numerical risk areas

### 3.1 GEV fitting (`scipy.stats.genextreme`)

**Risk: high.** This is the most dangerous numerical component.

Python code:
```python
raw_params = stats.genextreme.fit(self.scores)  # MLE, no fixed params
sf = stats.genextreme.sf(score, shape, loc, scale)
```

Issues to audit in Julia:
- **Shape sign convention**: SciPy's `genextreme` uses `c` (shape) where positive `c` corresponds to Type III (left-skewed, bounded below). Many Julia libraries use the opposite sign or the standard `k` convention. A sign error produces completely wrong tail probabilities.
- **Initialization**: SciPy uses method-of-moments or percentile-based starting points. Julia must either replicate or use a different, documented initialization.
- **Optimizer**: SciPy uses `scipy.optimize` (typically `-fmin_l_bfgs_b` or similar). Julia may use `Optim.jl` or a custom Newton-type solver. Different optimizers converge to slightly different points.
- **Constraints**: Scale must be positive. SciPy handles this internally; Julia must enforce it.
- **Degenerate samples**: constant arrays, very small samples, heavy-tailed empirical distributions. SciPy may produce warnings or non-convergence. Julia must detect and fail with a typed error.
- **Survival function**: `genextreme.sf(score, *params)` is the upper-tail survival `P(X > score)`. The sign of the shape parameter determines whether the distribution has a finite upper bound. If the fitted shape is positive (in SciPy's convention), the upper tail is finite; `sf(score)` returns 0 for scores above the bound. Julia must handle this correctly.

**Mitigation**: Create a GEV compatibility corpus with:
- 20+ null score samples of varying sizes (10²–10⁵).
- SciPy fitted parameters for each.
- SciPy SF values at 10–20 score points per sample.
- Expected failure cases (constant, tiny, extreme-tail).
- Separate tolerance for parameters vs SF values.

### 3.2 Float32 vs Float64 in scanning

Python uses `np.float32` for all scan scores and `fastmath=True` in Numba. The `fastmath` flag allows:
- Reassociation of floating-point operations.
- Treatment of NaNs/Infs as exceptions (not propagated).
- Contraction of multiply-add into FMA.

Julia's default `Float32` arithmetic does not reassociate. This means:
- Per-position scores may differ by a few ULPs.
- Summation over positions (for kmer > 1) may differ more due to different summation order.
- Reverse complement scoring reads in a different order than forward, so roundoff differs.

**Mitigation**: The accumulation study (Section 2) will quantify this. If drift is acceptable, `float32_kernel` tolerance stands. If not, consider:
- Using `Float64` accumulation in the inner loop (convert input to Float64, sum, convert back).
- Documenting the accumulation type per kernel in an ADR.

### 3.3 Empirical log-tail normalization

Python:
```python
scores_sorted = np.sort(flat)[::-1]  # descending
unique_scores, counts = np.unique(scores_sorted, return_counts=True)
unique_scores = unique_scores[::-1]  # back to ascending
counts = counts[::-1]
cum_counts = np.cumsum(counts)
tail_probabilities = cum_counts / flat.size
log_tail = -np.log10(tail_probabilities)
```

This produces a lookup table of `(score, -log10(tail_probability))` in ascending score order. The lookup uses a binary search (`_lower_bound_desc`) that finds the first index where `table_score <= query_score` — effectively a lower-bound search on a descending array, implemented as a binary search on negated values.

Risk areas:
- **Empty sample**: returns `[[0.0, 0.0]]`. Julia must match this edge case or document a divergence.
- **Single unique score**: `cum_counts = [n]`, `tail_prob = [1.0]`, `log_tail = [0.0]`. Every score maps to log_tail = 0.
- **Repeated scores**: `np.unique` with `return_counts` deduplicates; cumulative counts skip non-unique entries. Julia must sort and deduplicate identically.
- **Lookup at exact table boundary**: The binary search uses `side="left"` on negated values. Off-by-one in the Julia implementation would map to the wrong table entry.

**Mitigation**: Fixture with scores containing ties, a single unique value, and boundary cases. Compare table arrays exactly.

### 3.4 Tie-breaking in alignment

**Motif alignment**: Python iterates offsets from `-(target_length - 1)` to `query_length - 1` and keeps the first best (`score > best_score`, strictly greater). Equal scores do not replace the incumbent, so the first offset in iteration order wins. Orientation ties are broken by `max(score, -rank)`.

**Profile alignment**: Python iterates shifts from `-search_range` to `+search_range`. The tie-breaking is:
```python
if score > best_score or (
    score == best_score and (
        n_sites > best_n_sites or (
            n_sites == best_n_sites and abs(shift) < abs(best_shift)
        )
    )
):
    best = candidate
```
This means: higher score → more sites → smaller absolute shift. The `==` comparison on float scores is exact, not tolerance-based. If Julia computes scores in a different order, ties may break differently.

**Mitigation**: Fixtures must include cases with exact score ties (symmetric motifs, identical inputs). Tie-breaking behavior must be exact-matched or documented as divergence.

### 3.5 Reverse complement scoring

The reverse scan kernel reads from the same window position `pos` but reverses the read order:
```python
src = pos + (window_size - 1 - (term + offset))
encoded = 3 - base  # complement
```

This means position `pos` in the forward track and position `pos` in the reverse track correspond to the same window starting position, but the reverse track scores the reverse complement of the window. The coordinate correspondence is: reverse score at position `pos` corresponds to the forward strand site at `length - (pos + motif_len)` (zero-based) or `length - pos - motif_len + 1` (one-based, Julia).

**Mitigation**: Fixture with a known sequence and PWM, compare forward and reverse score tracks position-by-position. The reverse track is NOT the reverse of the forward track — it is the forward scan of the reverse complement, indexed by the same window start positions.

### 3.6 N state handling in tensors

For BaMM, Dimont, and Slim, the 5th state (N/ambiguous) is filled with the minimum over concrete nucleotides:
```python
position_tensor[..., 4] = np.min(position_tensor[..., :4], axis=-1)
```
and on context axes:
```python
arr[..., 4, ...] = np.min(arr[..., :4, ...], axis=axis)
```

This means an N in the context contributes the worst-case score for that position. Julia must replicate this exactly.

### 3.7 Random sequence generation

Python:
```python
rng = np.random.default_rng(seed)
rows = [rng.integers(0, 4, size=seq_length, dtype=np.int8) for _ in range(num_sequences)]
```

Julia's `Random` uses a different RNG algorithm. The generated sequences will NOT match byte-for-byte. This is expected and acceptable — random sequences are used for background calibration, not for deterministic comparison. However, the oracle fixtures must be generated with a pinned Python environment and stored, not regenerated in Julia.

**Mitigation**: Oracle fixtures store the actual encoded byte arrays. Julia tests load these from disk. The Python generation script is pinned and reproducible.

## 4. Comparison methodology

For each compatibility fixture:

1. **Generate** intermediate and final values using the pinned Python environment.
2. **Store** values in a versioned fixture directory with a manifest (Python commit, environment, checksums).
3. **Load** in Julia tests and compare against the stored values.
4. **Classify** each comparison into a tolerance class.
5. **Report** any `documented_divergence` with an ADR reference.

Intermediate values to freeze (per feature area):

| Area | Intermediate values |
|---|---|
| Parsers | Parsed array shape, dtype, values (exact); metadata (name, length, order/span) |
| Sequence I/O | Encoded byte array, offsets/lengths, padding value |
| Reverse complement | Complemented bytes for a sample sequence |
| PWM scanning | Forward track, reverse track, best track, both-strand bundle |
| Normalization | Sorted unique scores, counts, cumulative counts, log-tail table, normalized profile |
| Motif alignment | Per-offset scores, per-orientation scores, chosen offset/orientation |
| Profile alignment | Anchor positions, candidate windows, per-shift scores, chosen shift/orientation |
| Sites | Hit arrays (seq_index, start, strand, score), site strings, PCM, PFM |
| Nulls | Eligible pair order, raw scores, GEV parameters, SF values, p/E/adj-p values |
| CLI | stdout JSON, stderr class, exit code, output file checksums |